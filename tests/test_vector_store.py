"""
Unit tests for the in-memory vector store (services/vector_store.py).

Tests cover:
  - Adding chunks and embeddings
  - Cosine similarity correctness
  - Top-k retrieval
  - Score ordering
  - Removing a file
  - Thread safety (concurrent add + search)
  - Edge cases: empty store, unit vectors, zero vectors, single chunk
"""

import threading
import pytest
import numpy as np

from backend.services.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_chunks(n: int, source_file: str = "test.pdf") -> list[dict]:
    return [
        {"text": f"Chunk {i}", "source_file": source_file, "page": 1, "chunk_index": i}
        for i in range(n)
    ]


def unit_vec(dim: int, idx: int) -> np.ndarray:
    """Return a unit vector with 1 at position idx."""
    v = np.zeros(dim, dtype=np.float32)
    v[idx] = 1.0
    return v


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestVectorStore:

    @pytest.fixture(autouse=True)
    def fresh_store(self):
        self.store = VectorStore()

    # ------------------------------------------------------------------
    # Empty store
    # ------------------------------------------------------------------

    def test_empty_search_returns_empty(self):
        q = np.random.rand(64).astype(np.float32)
        assert self.store.search(q, top_k=5) == []

    def test_total_chunks_zero_initially(self):
        assert self.store.total_chunks == 0

    def test_indexed_files_empty_initially(self):
        assert self.store.indexed_files == []

    # ------------------------------------------------------------------
    # Adding chunks
    # ------------------------------------------------------------------

    def test_add_increments_total_chunks(self, sample_chunks, sample_embeddings):
        self.store.add_chunks(sample_chunks, sample_embeddings)
        assert self.store.total_chunks == len(sample_chunks)

    def test_indexed_files_populated(self, sample_chunks, sample_embeddings):
        self.store.add_chunks(sample_chunks, sample_embeddings)
        files = self.store.indexed_files
        assert "attention_paper.pdf" in files
        assert "rag_paper.pdf" in files

    def test_mismatched_chunks_embeddings_raises(self, sample_chunks, sample_embeddings):
        with pytest.raises(AssertionError):
            self.store.add_chunks(sample_chunks, sample_embeddings[:3])

    # ------------------------------------------------------------------
    # Cosine similarity correctness
    # ------------------------------------------------------------------

    def test_identical_vector_scores_one(self):
        """Searching with the exact same vector as stored should give score ≈ 1."""
        dim = 64
        chunks = make_chunks(1)
        v = np.random.rand(dim).astype(np.float32)
        self.store.add_chunks(chunks, v.reshape(1, -1))

        results = self.store.search(v, top_k=1)
        assert len(results) == 1
        assert abs(results[0]["score"] - 1.0) < 1e-5

    def test_orthogonal_vectors_score_zero(self):
        """Two orthogonal unit vectors have cosine similarity = 0."""
        dim = 4
        e0 = unit_vec(dim, 0).reshape(1, -1)
        e1 = unit_vec(dim, 1)

        chunks = make_chunks(1)
        self.store.add_chunks(chunks, e0)
        results = self.store.search(e1, top_k=1)
        assert abs(results[0]["score"]) < 1e-5

    def test_opposite_vectors_score_negative(self):
        """Opposite unit vectors → cosine similarity ≈ -1."""
        dim = 4
        pos = unit_vec(dim, 0).reshape(1, -1)
        neg = -unit_vec(dim, 0)

        chunks = make_chunks(1)
        self.store.add_chunks(chunks, pos)
        results = self.store.search(neg, top_k=1)
        assert results[0]["score"] < 0

    def test_results_sorted_by_score_descending(self):
        """Results must be sorted from highest to lowest cosine score."""
        dim = 8
        rng = np.random.default_rng(99)
        n = 10
        chunks = make_chunks(n)
        embeddings = rng.random((n, dim)).astype(np.float32)
        self.store.add_chunks(chunks, embeddings)

        query = rng.random(dim).astype(np.float32)
        results = self.store.search(query, top_k=n)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_nearest_neighbour_is_most_similar(self):
        """
        Insert 5 random vectors and one query-like vector.
        The query-like vector should be the nearest neighbour to itself.
        """
        dim = 32
        rng = np.random.default_rng(7)
        n = 5
        chunks = make_chunks(n)
        embeddings = rng.random((n, dim)).astype(np.float32)
        self.store.add_chunks(chunks, embeddings)

        # Query is very close to chunk 2
        query = embeddings[2].copy() + rng.random(dim).astype(np.float32) * 0.01
        results = self.store.search(query, top_k=1)
        assert results[0]["chunk_index"] == 2

    # ------------------------------------------------------------------
    # Top-k behaviour
    # ------------------------------------------------------------------

    def test_top_k_limits_results(self, sample_chunks, sample_embeddings):
        self.store.add_chunks(sample_chunks, sample_embeddings)
        results = self.store.search(sample_embeddings[0], top_k=2)
        assert len(results) == 2

    def test_top_k_larger_than_store_returns_all(self, sample_chunks, sample_embeddings):
        self.store.add_chunks(sample_chunks, sample_embeddings)
        results = self.store.search(sample_embeddings[0], top_k=100)
        assert len(results) == len(sample_chunks)

    # ------------------------------------------------------------------
    # Remove file
    # ------------------------------------------------------------------

    def test_remove_file_reduces_chunk_count(self, sample_chunks, sample_embeddings):
        self.store.add_chunks(sample_chunks, sample_embeddings)
        original = self.store.total_chunks
        removed = self.store.remove_file("rag_paper.pdf")
        assert removed > 0
        assert self.store.total_chunks == original - removed

    def test_removed_file_not_in_results(self, sample_chunks, sample_embeddings):
        self.store.add_chunks(sample_chunks, sample_embeddings)
        self.store.remove_file("rag_paper.pdf")
        query = sample_embeddings[2]  # was from rag_paper.pdf
        results = self.store.search(query, top_k=10)
        for r in results:
            assert r["source_file"] != "rag_paper.pdf"

    def test_remove_nonexistent_file_returns_zero(self, sample_chunks, sample_embeddings):
        self.store.add_chunks(sample_chunks, sample_embeddings)
        removed = self.store.remove_file("ghost.pdf")
        assert removed == 0
        assert self.store.total_chunks == len(sample_chunks)

    def test_remove_all_files_empties_store(self, sample_chunks, sample_embeddings):
        self.store.add_chunks(sample_chunks, sample_embeddings)
        for fname in list(self.store.indexed_files):
            self.store.remove_file(fname)
        assert self.store.total_chunks == 0
        q = sample_embeddings[0]
        assert self.store.search(q, top_k=5) == []

    # ------------------------------------------------------------------
    # Clear
    # ------------------------------------------------------------------

    def test_clear_empties_store(self, sample_chunks, sample_embeddings):
        self.store.add_chunks(sample_chunks, sample_embeddings)
        self.store.clear()
        assert self.store.total_chunks == 0
        assert self.store.search(sample_embeddings[0], top_k=5) == []

    # ------------------------------------------------------------------
    # Incremental adds
    # ------------------------------------------------------------------

    def test_multiple_adds_accumulate(self):
        dim = 16
        rng = np.random.default_rng(1)
        for i in range(3):
            chunks = [{"text": f"t{i}", "source_file": f"f{i}.pdf", "page": 1, "chunk_index": 0}]
            emb = rng.random((1, dim)).astype(np.float32)
            self.store.add_chunks(chunks, emb)
        assert self.store.total_chunks == 3

    # ------------------------------------------------------------------
    # Thread safety
    # ------------------------------------------------------------------

    def test_concurrent_add_and_search_no_crash(self):
        dim = 32
        errors = []

        def add_and_search(seed):
            try:
                rng = np.random.default_rng(seed)
                n = 5
                chunks = [
                    {"text": f"text {seed} {i}", "source_file": f"f{seed}.pdf",
                     "page": 1, "chunk_index": i}
                    for i in range(n)
                ]
                emb = rng.random((n, dim)).astype(np.float32)
                self.store.add_chunks(chunks, emb)
                q = rng.random(dim).astype(np.float32)
                self.store.search(q, top_k=3)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_and_search, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"

    # ------------------------------------------------------------------
    # Zero vector edge case
    # ------------------------------------------------------------------

    def test_zero_query_vector_does_not_crash(self, sample_chunks, sample_embeddings):
        self.store.add_chunks(sample_chunks, sample_embeddings)
        zero = np.zeros(sample_embeddings.shape[1], dtype=np.float32)
        results = self.store.search(zero, top_k=3)
        # Should return results without crashing (all scores ≈ 0)
        assert isinstance(results, list)
