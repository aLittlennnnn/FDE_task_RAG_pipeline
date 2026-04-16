"""
Unit tests for PDF text extraction and chunking (services/pdf_parser.py).

Tests cover:
  - Sentence splitting
  - Chunk size / overlap properties
  - Short chunk filtering
  - Metadata fields on output chunks
  - Multi-page document produces page-tagged chunks
  - Empty / unreadable PDF handling
  - Chunk text is non-empty and within reasonable length bounds
"""

import pytest

from backend.services.pdf_parser import (
    _clean_text,
    _split_sentences,
    _chunk_sentences,
    extract_chunks,
    MIN_CHUNK_LENGTH,
)


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

class TestCleanText:
    def test_collapses_whitespace(self):
        assert _clean_text("hello   world") == "hello world"

    def test_removes_newlines(self):
        assert _clean_text("hello\nworld") == "hello world"

    def test_strips_leading_trailing(self):
        assert _clean_text("  hello  ") == "hello"

    def test_empty_string(self):
        assert _clean_text("") == ""

    def test_only_whitespace(self):
        assert _clean_text("   \n\t  ") == ""


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

class TestSplitSentences:
    def test_splits_on_period(self):
        sents = _split_sentences("Hello world. This is a test. End here.")
        assert len(sents) >= 2

    def test_does_not_split_abbreviations_aggressively(self):
        # "Dr. Smith" should not be split into two sentences
        text = "The transformer model was introduced by Dr. Vaswani et al."
        sents = _split_sentences(text)
        # Should be 1 or 2 sentences but not 3
        assert len(sents) <= 2

    def test_empty_string_returns_empty(self):
        assert _split_sentences("") == []

    def test_single_sentence(self):
        result = _split_sentences("Just one sentence here")
        assert len(result) == 1

    def test_strips_each_sentence(self):
        sents = _split_sentences("  Hello world.  This is clean.  ")
        for s in sents:
            assert s == s.strip()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

class TestChunkSentences:
    def _sentences(self, n=20, length=50):
        return [f"Sentence number {i} with some padding text here." for i in range(n)]

    def test_produces_at_least_one_chunk(self):
        sents = self._sentences(5)
        chunks = list(_chunk_sentences(sents, chunk_size=300, overlap=50))
        assert len(chunks) >= 1

    def test_chunk_size_respected(self):
        sents = self._sentences(30)
        chunks = list(_chunk_sentences(sents, chunk_size=200, overlap=50))
        for c in chunks:
            # Chunks may slightly exceed chunk_size due to final sentence boundary
            assert len(c) <= 200 * 2  # generous upper bound

    def test_overlap_creates_repeated_content(self):
        """Words from the end of chunk N should appear at the start of chunk N+1."""
        # Use long sentences to force multiple chunks
        long_sents = [f"Word{i} " * 20 + "sentence end." for i in range(20)]
        chunks = list(_chunk_sentences(long_sents, chunk_size=100, overlap=50))
        if len(chunks) < 2:
            pytest.skip("Not enough chunks produced to test overlap")
        # Last few words of chunk 0 should appear somewhere in chunk 1
        chunk0_words = set(chunks[0].split()[-5:])
        chunk1_words = set(chunks[1].split())
        assert len(chunk0_words & chunk1_words) > 0

    def test_empty_sentences_returns_empty(self):
        chunks = list(_chunk_sentences([], chunk_size=300, overlap=50))
        assert chunks == []

    def test_single_long_sentence(self):
        sents = ["word " * 100 + "end."]
        chunks = list(_chunk_sentences(sents, chunk_size=200, overlap=50))
        assert len(chunks) >= 1

    def test_all_content_preserved(self):
        """Every word from the input should appear in at least one chunk."""
        sents = [f"unique_term_{i} content here." for i in range(10)]
        chunks = list(_chunk_sentences(sents, chunk_size=100, overlap=30))
        full_text = " ".join(chunks)
        for i in range(10):
            assert f"unique_term_{i}" in full_text


# ---------------------------------------------------------------------------
# Full extract_chunks (requires a real PDF)
# ---------------------------------------------------------------------------

class TestExtractChunks:

    def test_returns_list_of_dicts(self, test_pdf_bytes):
        chunks = extract_chunks(test_pdf_bytes, "test.pdf")
        assert isinstance(chunks, list)
        for c in chunks:
            assert isinstance(c, dict)

    def test_required_keys_present(self, test_pdf_bytes):
        chunks = extract_chunks(test_pdf_bytes, "test.pdf")
        for c in chunks:
            assert "text" in c
            assert "source_file" in c
            assert "page" in c
            assert "chunk_index" in c

    def test_source_file_set_correctly(self, test_pdf_bytes):
        chunks = extract_chunks(test_pdf_bytes, "myfile.pdf")
        for c in chunks:
            assert c["source_file"] == "myfile.pdf"

    def test_no_short_chunks(self, test_pdf_bytes):
        chunks = extract_chunks(test_pdf_bytes, "test.pdf")
        for c in chunks:
            assert len(c["text"]) >= MIN_CHUNK_LENGTH

    def test_chunk_indices_are_sequential(self, test_pdf_bytes):
        chunks = extract_chunks(test_pdf_bytes, "test.pdf")
        if chunks:
            indices = [c["chunk_index"] for c in chunks]
            assert indices == list(range(len(indices)))

    def test_pages_are_positive_integers(self, test_pdf_bytes):
        chunks = extract_chunks(test_pdf_bytes, "test.pdf")
        for c in chunks:
            assert isinstance(c["page"], int)
            assert c["page"] >= 1

    def test_content_extracted_from_pdf(self, test_pdf_bytes):
        """Chunks should contain some of the text we put into the PDF."""
        chunks = extract_chunks(test_pdf_bytes, "test.pdf")
        all_text = " ".join(c["text"] for c in chunks).lower()
        # These words were in the fixture PDF text
        assert any(word in all_text for word in ["transform", "attention", "retrieval", "chunk"])

    def test_custom_chunk_size(self, test_pdf_bytes):
        small_chunks = extract_chunks(test_pdf_bytes, "test.pdf", chunk_size=200, overlap=50)
        large_chunks = extract_chunks(test_pdf_bytes, "test.pdf", chunk_size=1000, overlap=100)
        # Smaller chunk size → more chunks (or equal)
        assert len(small_chunks) >= len(large_chunks)

    def test_invalid_pdf_raises(self):
        with pytest.raises(Exception):
            extract_chunks(b"not a pdf", "fake.pdf")

    def test_empty_bytes_raises(self):
        with pytest.raises(Exception):
            extract_chunks(b"", "empty.pdf")

    def test_text_chunks_not_empty(self, test_pdf_bytes):
        chunks = extract_chunks(test_pdf_bytes, "test.pdf")
        for c in chunks:
            assert c["text"].strip() != ""
