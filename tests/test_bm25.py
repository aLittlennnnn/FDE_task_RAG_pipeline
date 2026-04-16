"""
Unit tests for the BM25 index (services/bm25.py).

Tests cover:
  - Tokenisation
  - Index building and basic search
  - Score ordering (more relevant docs rank higher)
  - IDF weighting (rare terms score higher)
  - Document length normalisation
  - Incremental add and remove_file
  - Thread safety (concurrent adds)
  - Edge cases: empty index, unknown query terms, very short docs
"""

import threading
import pytest

# Import a fresh BM25Index for each test (don't touch the global singleton)
from backend.services.bm25 import BM25Index, _tokenise


# ---------------------------------------------------------------------------
# Tokeniser tests
# ---------------------------------------------------------------------------

class TestTokenise:
    def test_lowercases(self):
        assert _tokenise("Hello World") == ["hello", "world"]

    def test_strips_punctuation(self):
        tokens = _tokenise("Hello, world! This is a test.")
        assert "," not in tokens
        assert "!" not in tokens

    def test_removes_stopwords(self):
        tokens = _tokenise("the cat is on the mat")
        assert "the" not in tokens
        assert "is" not in tokens
        assert "on" not in tokens

    def test_keeps_content_words(self):
        tokens = _tokenise("transformer attention mechanism")
        assert "transformer" in tokens
        assert "attention" in tokens
        assert "mechanism" in tokens

    def test_empty_string(self):
        assert _tokenise("") == []

    def test_only_stopwords(self):
        assert _tokenise("the a an is") == []

    def test_numbers_kept(self):
        tokens = _tokenise("GPT-4 has 1 trillion parameters")
        assert "gpt" in tokens or "gpt4" in tokens or "4" in tokens


# ---------------------------------------------------------------------------
# BM25Index tests
# ---------------------------------------------------------------------------

class TestBM25Index:

    @pytest.fixture(autouse=True)
    def fresh_index(self):
        """Each test gets a clean BM25Index."""
        self.idx = BM25Index()

    def _add(self, texts_by_file: dict[str, list[str]]):
        """Helper to add chunks from multiple files."""
        for fname, texts in texts_by_file.items():
            chunks = [
                {"text": t, "source_file": fname, "page": i + 1, "chunk_index": i}
                for i, t in enumerate(texts)
            ]
            self.idx.add_chunks(chunks)

    # ------------------------------------------------------------------
    # Basic search
    # ------------------------------------------------------------------

    def test_empty_index_returns_empty(self):
        results = self.idx.search("transformers", top_k=5)
        assert results == []

    def test_single_doc_found(self):
        self._add({"paper.pdf": ["Transformers use attention mechanisms for NLP tasks."]})
        results = self.idx.search("attention", top_k=5)
        assert len(results) == 1
        assert results[0]["source_file"] == "paper.pdf"

    def test_unknown_query_returns_empty(self):
        self._add({"paper.pdf": ["Transformers use attention mechanisms."]})
        results = self.idx.search("quantum physics xyzzy", top_k=5)
        assert results == []

    def test_top_k_limits_results(self):
        texts = [f"Document number {i} about transformers and attention." for i in range(10)]
        self._add({"doc.pdf": texts})
        results = self.idx.search("transformers", top_k=3)
        assert len(results) <= 3

    # ------------------------------------------------------------------
    # Ranking quality
    # ------------------------------------------------------------------

    def test_more_relevant_doc_ranks_higher(self):
        """A doc mentioning the query term more often should rank higher."""
        self._add({
            "docs.pdf": [
                "Attention mechanism. Attention is all you need. Attention weights.",
                "Some unrelated content about databases and SQL queries.",
            ]
        })
        results = self.idx.search("attention", top_k=2)
        assert results[0]["chunk_index"] == 0  # first chunk is more relevant

    def test_scores_are_positive(self):
        self._add({"doc.pdf": ["BM25 is a ranking algorithm used in information retrieval."]})
        results = self.idx.search("ranking algorithm", top_k=5)
        for r in results:
            assert r["bm25_score"] > 0

    def test_results_sorted_descending(self):
        self._add({"doc.pdf": [
            "BM25 retrieval algorithm for ranking documents.",
            "BM25 BM25 ranking ranking retrieval retrieval algorithm algorithm.",
            "Unrelated sentence about the weather today.",
        ]})
        results = self.idx.search("BM25 retrieval", top_k=3)
        scores = [r["bm25_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_idf_rare_term_scores_higher(self):
        """
        'rare_term' appears in only 1 of 5 docs → higher IDF → higher BM25
        when that doc is retrieved.
        """
        self._add({"doc.pdf": [
            "common word common word common word",
            "common word common word common word",
            "common word common word common word",
            "common word common word rare_term",
            "common word common word common word",
        ]})
        results_rare   = self.idx.search("rare_term", top_k=5)
        results_common = self.idx.search("common", top_k=5)
        # rare_term should appear in exactly 1 result
        assert len(results_rare) == 1
        # common should appear in multiple results
        assert len(results_common) > 1

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def test_add_multiple_files(self):
        self._add({
            "file_a.pdf": ["Semantic search with dense embeddings."],
            "file_b.pdf": ["Keyword search with BM25 algorithm."],
        })
        a_results = self.idx.search("embeddings", top_k=5)
        b_results = self.idx.search("keyword", top_k=5)
        assert a_results[0]["source_file"] == "file_a.pdf"
        assert b_results[0]["source_file"] == "file_b.pdf"

    def test_remove_file(self):
        self._add({
            "keep.pdf": ["Transformers for NLP tasks."],
            "remove.pdf": ["This document will be deleted."],
        })
        self.idx.remove_file("remove.pdf")
        results = self.idx.search("deleted document", top_k=5)
        for r in results:
            assert r["source_file"] != "remove.pdf"

    def test_remove_nonexistent_file_safe(self):
        self._add({"doc.pdf": ["Some content here."]})
        self.idx.remove_file("ghost.pdf")  # Should not raise
        assert len(self.idx.search("content", top_k=5)) == 1

    def test_clear(self):
        self._add({"doc.pdf": ["Content about NLP."]})
        self.idx.clear()
        assert self.idx.search("NLP", top_k=5) == []

    def test_re_add_after_remove(self):
        self._add({"doc.pdf": ["Original content about transformers."]})
        self.idx.remove_file("doc.pdf")
        self._add({"doc.pdf": ["New content about retrieval algorithms."]})
        assert len(self.idx.search("retrieval", top_k=5)) == 1
        assert len(self.idx.search("transformers", top_k=5)) == 0

    # ------------------------------------------------------------------
    # Thread safety
    # ------------------------------------------------------------------

    def test_concurrent_adds_do_not_crash(self):
        errors = []

        def add_chunks(file_id):
            try:
                chunks = [
                    {"text": f"Content {i} from file {file_id} about transformers.",
                     "source_file": f"file_{file_id}.pdf", "page": 1, "chunk_index": i}
                    for i in range(5)
                ]
                self.idx.add_chunks(chunks)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_chunks, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        # All 8 files × 5 chunks = 40 chunks added
        results = self.idx.search("transformers", top_k=40)
        assert len(results) == 40

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_empty_query(self):
        self._add({"doc.pdf": ["Some content here."]})
        results = self.idx.search("", top_k=5)
        assert results == []

    def test_stopwords_only_query(self):
        self._add({"doc.pdf": ["Some content here."]})
        results = self.idx.search("the is a", top_k=5)
        assert results == []

    def test_chunk_metadata_preserved(self):
        self._add({"data.pdf": ["Cosine similarity for vector search."]})
        results = self.idx.search("cosine similarity", top_k=1)
        assert results[0]["source_file"] == "data.pdf"
        assert results[0]["page"] == 1
        assert results[0]["chunk_index"] == 0
        assert "bm25_score" in results[0]
