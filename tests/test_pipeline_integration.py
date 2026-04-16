"""
Integration tests for the full query pipeline and the FastAPI endpoints.

All Mistral API calls are mocked — no real network calls are made.
Tests cover:
  - POST /api/ingest  (valid PDF, invalid file, re-ingestion)
  - GET  /api/ingest/status
  - DELETE /api/ingest/{filename}
  - POST /api/query  (factual, conversational, refused, insufficient evidence,
                      empty KB)
"""

import io
import json
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from backend.main import app

# ---------------------------------------------------------------------------
# Shared mocks
# ---------------------------------------------------------------------------

DIM = 1024  # mistral-embed dimension

def _embed_response(texts):
    """Return a fake Mistral embedding API response."""
    n = len(texts)
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    mock.json.return_value = {
        "data": [
            {"index": i, "embedding": np.random.rand(DIM).tolist()}
            for i in range(n)
        ]
    }
    return mock


def _chat_response(content: str):
    """Return a fake Mistral chat completion response."""
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    mock.json.return_value = {
        "choices": [{"message": {"content": content}}]
    }
    return mock


@pytest.fixture(autouse=True)
def clear_stores():
    """Reset the global vector store and BM25 index before each test."""
    from backend.services.vector_store import store
    from backend.services.bm25 import bm25_index
    store.clear()
    bm25_index.clear()
    yield
    store.clear()
    bm25_index.clear()


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# Ingestion endpoint tests
# ---------------------------------------------------------------------------

class TestIngestEndpoint:

    def _pdf_upload(self, filename: str, pdf_bytes: bytes):
        return ("files", (filename, io.BytesIO(pdf_bytes), "application/pdf"))

    @patch("backend.routers.ingestion.embed_texts")
    def test_ingest_valid_pdf(self, mock_embed, client, test_pdf_bytes):
        mock_embed.return_value = np.random.rand(10, DIM).astype(np.float32)

        resp = client.post(
            "/api/ingest",
            files=[self._pdf_upload("test.pdf", test_pdf_bytes)],
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["files_processed"] >= 1
        assert data["total_chunks"] > 0

    def test_ingest_no_files_returns_400(self, client):
        # Send a request with no files at all — FastAPI should reject it
        resp = client.post("/api/ingest")
        assert resp.status_code == 422  # Unprocessable entity (missing required field)

    def test_ingest_non_pdf_returns_415(self, client):
        fake_txt = b"This is plain text, not a PDF."
        resp = client.post(
            "/api/ingest",
            files=[("files", ("doc.txt", io.BytesIO(fake_txt), "text/plain"))],
        )
        assert resp.status_code == 415

    def test_ingest_invalid_pdf_magic_returns_415(self, client):
        fake_pdf = b"NOTPDF content here"
        resp = client.post(
            "/api/ingest",
            files=[("files", ("fake.pdf", io.BytesIO(fake_pdf), "application/pdf"))],
        )
        assert resp.status_code == 415

    @patch("backend.routers.ingestion.embed_texts")
    def test_ingest_status_reflects_ingested_file(self, mock_embed, client, test_pdf_bytes):
        mock_embed.return_value = np.random.rand(10, DIM).astype(np.float32)
        client.post("/api/ingest", files=[self._pdf_upload("status_test.pdf", test_pdf_bytes)])

        status_resp = client.get("/api/ingest/status")
        assert status_resp.status_code == 200
        status = status_resp.json()
        assert "status_test.pdf" in status["indexed_files"]
        assert status["total_chunks"] > 0

    @patch("backend.routers.ingestion.embed_texts")
    def test_re_ingest_same_file_replaces_old(self, mock_embed, client, test_pdf_bytes):
        mock_embed.return_value = np.random.rand(10, DIM).astype(np.float32)
        client.post("/api/ingest", files=[self._pdf_upload("dup.pdf", test_pdf_bytes)])
        chunks_after_first = client.get("/api/ingest/status").json()["total_chunks"]

        mock_embed.return_value = np.random.rand(10, DIM).astype(np.float32)
        client.post("/api/ingest", files=[self._pdf_upload("dup.pdf", test_pdf_bytes)])
        chunks_after_second = client.get("/api/ingest/status").json()["total_chunks"]

        # Re-ingestion should replace, not accumulate
        assert chunks_after_second == chunks_after_first

    @patch("backend.routers.ingestion.embed_texts")
    def test_delete_existing_file(self, mock_embed, client, test_pdf_bytes):
        mock_embed.return_value = np.random.rand(10, DIM).astype(np.float32)
        client.post("/api/ingest", files=[self._pdf_upload("to_delete.pdf", test_pdf_bytes)])

        del_resp = client.delete("/api/ingest/to_delete.pdf")
        assert del_resp.status_code == 200
        assert "to_delete.pdf" not in client.get("/api/ingest/status").json()["indexed_files"]

    def test_delete_nonexistent_file_returns_404(self, client):
        resp = client.delete("/api/ingest/ghost.pdf")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Query endpoint tests
# ---------------------------------------------------------------------------

class TestQueryEndpoint:

    def test_query_empty_kb_no_documents(self, client):
        resp = client.post(
            "/api/query",
            json={"question": "What is transformers?", "top_k": 3}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["sufficient_evidence"] is False

    @patch("backend.routers.query.detect_intent", return_value="conversational")
    @patch("backend.routers.query.conversational_reply", return_value="Hello! How can I help?")
    def test_conversational_query_no_retrieval(self, mock_reply, mock_intent, client):
        resp = client.post("/api/query", json={"question": "hello"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["intent"] == "conversational"
        assert data["citations"] == []
        assert data["sufficient_evidence"] is True

    @patch("backend.routers.query.detect_intent", return_value="refused")
    @patch("backend.routers.query.refused_reply", return_value="I cannot answer that.")
    def test_refused_query(self, mock_refused, mock_intent, client):
        resp = client.post("/api/query", json={"question": "What is my SSN?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["intent"] == "refused"
        assert data["sufficient_evidence"] is False

    @patch("backend.routers.query.detect_intent", return_value="factual")
    @patch("backend.routers.query.transform_query", return_value="transformed query text")
    @patch("backend.routers.query.hybrid_search")
    @patch("backend.routers.query.generate_answer")
    def test_factual_query_with_results(
        self, mock_gen, mock_search, mock_transform, mock_intent, client
    ):
        mock_search.return_value = [
            {
                "text": "Transformers use self-attention mechanisms.",
                "source_file": "paper.pdf",
                "page": 1,
                "chunk_index": 0,
                "score": 0.88,
            }
        ]
        mock_gen.return_value = ("Transformers use self-attention. [Source: paper.pdf, p.1]", True)

        # Add a fake chunk to the store so the "empty KB" guard doesn't fire
        from backend.services.vector_store import store
        from backend.services.bm25 import bm25_index
        store.add_chunks(
            [{"text": "dummy", "source_file": "paper.pdf", "page": 1, "chunk_index": 0}],
            np.random.rand(1, 128).astype(np.float32),
        )

        resp = client.post(
            "/api/query",
            json={"question": "How do transformers work?", "top_k": 3}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["intent"] == "factual"
        assert data["sufficient_evidence"] is True
        assert len(data["citations"]) == 1
        assert data["citations"][0]["source_file"] == "paper.pdf"
        assert data["citations"][0]["score"] == 0.88

    @patch("backend.routers.query.detect_intent", return_value="factual")
    @patch("backend.routers.query.transform_query", return_value="transformed query")
    @patch("backend.routers.query.hybrid_search")
    @patch("backend.routers.query.generate_answer")
    def test_insufficient_evidence_response(
        self, mock_gen, mock_search, mock_transform, mock_intent, client
    ):
        mock_search.return_value = [
            {
                "text": "Unrelated content.",
                "source_file": "doc.pdf",
                "page": 1,
                "chunk_index": 0,
                "score": 0.10,  # below threshold
            }
        ]
        mock_gen.return_value = ("insufficient evidence message", False)

        from backend.services.vector_store import store
        store.add_chunks(
            [{"text": "dummy", "source_file": "doc.pdf", "page": 1, "chunk_index": 0}],
            np.random.rand(1, 128).astype(np.float32),
        )

        resp = client.post("/api/query", json={"question": "Tell me about quantum physics?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["sufficient_evidence"] is False

    def test_query_too_long_returns_422(self, client):
        resp = client.post("/api/query", json={"question": "x" * 2001})
        assert resp.status_code == 422

    def test_query_empty_string_returns_422(self, client):
        resp = client.post("/api/query", json={"question": ""})
        assert resp.status_code == 422

    def test_query_invalid_top_k_returns_422(self, client):
        resp = client.post("/api/query", json={"question": "hello", "top_k": 0})
        assert resp.status_code == 422

    def test_query_response_schema(self, client):
        """Ensure the response always contains all required fields."""
        with patch("backend.routers.query.detect_intent", return_value="conversational"), \
             patch("backend.routers.query.conversational_reply", return_value="Hi!"):
            resp = client.post("/api/query", json={"question": "hello"})
        data = resp.json()
        for field in ["answer", "intent", "citations", "query_used", "sufficient_evidence"]:
            assert field in data, f"Missing field: {field}"


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
