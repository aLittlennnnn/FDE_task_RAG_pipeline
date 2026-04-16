"""
Unit tests for hybrid retrieval + RRF (services/retriever.py).

All Mistral API calls are mocked — tests verify:
  - RRF fusion of two ranked lists
  - Deduplication of chunks that appear in both lists
  - Score ordering of merged results
  - top_k enforcement
  - Handling of empty semantic or BM25 results
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from backend.services.retriever import _reciprocal_rank_fusion, hybrid_search
from backend.services.vector_store import VectorStore
from backend.services.bm25 import BM25Index


# ---------------------------------------------------------------------------
# RRF fusion logic
# ---------------------------------------------------------------------------

def _make_chunk(source: str, idx: int, score: float = 0.5) -> dict:
    return {
        "text": f"Chunk {idx}",
        "source_file": source,
        "page": 1,
        "chunk_index": idx,
        "score": score,
    }


class TestRRF:

    def test_empty_both_lists_returns_empty(self):
        result = _reciprocal_rank_fusion([], [])
        assert result == []

    def test_only_semantic_results(self):
        sem = [_make_chunk("a.pdf", i) for i in range(5)]
        result = _reciprocal_rank_fusion(sem, [])
        assert len(result) == 5

    def test_only_bm25_results(self):
        bm = [_make_chunk("a.pdf", i) for i in range(5)]
        result = _reciprocal_rank_fusion([], bm)
        assert len(result) == 5

    def test_deduplication(self):
        """A chunk in both lists should appear only once in the merged list."""
        shared = _make_chunk("a.pdf", 0)
        sem = [shared, _make_chunk("a.pdf", 1), _make_chunk("a.pdf", 2)]
        bm  = [shared, _make_chunk("a.pdf", 3), _make_chunk("a.pdf", 4)]
        result = _reciprocal_rank_fusion(sem, bm)
        # chunk 0 should appear exactly once
        keys = [(r["source_file"], r["chunk_index"]) for r in result]
        assert keys.count(("a.pdf", 0)) == 1

    def test_shared_chunk_ranks_higher(self):
        """A chunk appearing in both lists should rank above a chunk in only one."""
        shared = _make_chunk("a.pdf", 0)
        unique_only_sem  = _make_chunk("a.pdf", 1)
        unique_only_bm   = _make_chunk("a.pdf", 2)

        sem = [shared, unique_only_sem]
        bm  = [shared, unique_only_bm]
        result = _reciprocal_rank_fusion(sem, bm)

        assert result[0]["chunk_index"] == 0  # shared should be first

    def test_rrf_scores_present(self):
        sem = [_make_chunk("a.pdf", i) for i in range(3)]
        bm  = [_make_chunk("a.pdf", i) for i in range(3)]
        result = _reciprocal_rank_fusion(sem, bm)
        for r in result:
            assert "rrf_score" in r
            assert r["rrf_score"] > 0

    def test_results_sorted_descending_by_rrf(self):
        sem = [_make_chunk("a.pdf", i) for i in range(5)]
        bm  = [_make_chunk("a.pdf", i) for i in range(5)]
        result = _reciprocal_rank_fusion(sem, bm)
        scores = [r["rrf_score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_cross_file_chunks_deduplicated_by_file_and_index(self):
        """Two chunks from different files with the same chunk_index are distinct."""
        a = _make_chunk("file_a.pdf", 0)
        b = _make_chunk("file_b.pdf", 0)  # same chunk_index, different file
        result = _reciprocal_rank_fusion([a], [b])
        assert len(result) == 2  # should not be deduplicated


# ---------------------------------------------------------------------------
# hybrid_search (mocked embedder + patched store + bm25)
# ---------------------------------------------------------------------------

class TestHybridSearch:

    @patch("backend.services.retriever.embed_query")
    @patch("backend.services.retriever.store")
    @patch("backend.services.retriever.bm25_index")
    def test_returns_merged_results(self, mock_bm25, mock_store, mock_embed):
        mock_embed.return_value = np.random.rand(128).astype(np.float32)
        mock_store.search.return_value = [
            _make_chunk("sem.pdf", i, score=0.9 - i * 0.1) for i in range(4)
        ]
        mock_bm25.search.return_value = [
            _make_chunk("bm.pdf", i) for i in range(3)
        ]

        results = hybrid_search("What is attention?", top_k=5)
        assert len(results) <= 5
        assert all("rrf_score" in r for r in results)

    @patch("backend.services.retriever.embed_query")
    @patch("backend.services.retriever.store")
    @patch("backend.services.retriever.bm25_index")
    def test_top_k_enforced(self, mock_bm25, mock_store, mock_embed):
        mock_embed.return_value = np.random.rand(128).astype(np.float32)
        mock_store.search.return_value = [_make_chunk("a.pdf", i) for i in range(10)]
        mock_bm25.search.return_value   = [_make_chunk("b.pdf", i) for i in range(10)]

        results = hybrid_search("query", top_k=3)
        assert len(results) == 3

    @patch("backend.services.retriever.embed_query")
    @patch("backend.services.retriever.store")
    @patch("backend.services.retriever.bm25_index")
    def test_empty_both_returns_empty(self, mock_bm25, mock_store, mock_embed):
        mock_embed.return_value = np.random.rand(128).astype(np.float32)
        mock_store.search.return_value = []
        mock_bm25.search.return_value  = []

        results = hybrid_search("query", top_k=5)
        assert results == []

    @patch("backend.services.retriever.embed_query")
    @patch("backend.services.retriever.store")
    @patch("backend.services.retriever.bm25_index")
    def test_embed_called_once(self, mock_bm25, mock_store, mock_embed):
        mock_embed.return_value = np.random.rand(128).astype(np.float32)
        mock_store.search.return_value = []
        mock_bm25.search.return_value  = []

        hybrid_search("some question", top_k=5)
        mock_embed.assert_called_once_with("some question")
