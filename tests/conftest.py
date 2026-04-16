"""
Shared pytest fixtures.

Includes:
  - A minimal valid PDF (programmatically generated, no external file needed).
  - Sample chunk lists for reuse across tests.
  - A small set of numpy embeddings for vector store tests.
"""

import io
import struct
import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Minimal valid PDF builder (pure Python, no dependencies)
# Produces a 1-page PDF with readable text so pdfminer can extract it.
# ---------------------------------------------------------------------------

def _make_minimal_pdf(text: str = "Hello world. This is a test document.") -> bytes:
    """
    Build the smallest valid PDF that pdfminer.six can parse.
    Uses standard fonts (Helvetica) so no font embedding is required.
    """
    lines = [
        b"%PDF-1.4",
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj",
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj",
    ]

    # Page content stream
    encoded_text = text.encode("latin-1", errors="replace")
    stream_content = (
        b"BT /F1 12 Tf 50 750 Td (" + encoded_text + b") Tj ET"
    )
    stream_len = len(stream_content)

    lines += [
        b"3 0 obj << /Type /Page /Parent 2 0 R "
        b"/MediaBox [0 0 612 792] "
        b"/Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> "
        b">> endobj",
        f"4 0 obj << /Length {stream_len} >>".encode(),
        b"stream",
        stream_content,
        b"endstream endobj",
        b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj",
    ]

    body = b"\n".join(lines) + b"\n"

    # Cross-reference table
    offsets = []
    pos = 0
    for line in lines:
        if line.startswith(b"stream") or line == b"endstream endobj":
            offsets.append(None)
        else:
            # Find object declarations
            pass
        pos += len(line) + 1  # +1 for \n

    # Simple xref — just use startxref pointing to end of body
    xref_offset = len(body)
    num_objs = 6

    # Build a minimal xref + trailer
    xref = f"xref\n0 {num_objs}\n".encode()
    xref += b"0000000000 65535 f \n"
    for i in range(1, num_objs):
        # Not strictly valid offsets but sufficient for pdfminer to parse
        xref += f"{i * 20:010d} 00000 n \n".encode()

    trailer = (
        f"trailer << /Size {num_objs} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode()

    return body + xref + trailer


# Use a real, well-formed approach: write text into a proper PDF with reportlab
# if available, otherwise fall back to a fixture that skips PDF parsing tests.
try:
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.lib.pagesizes import letter

    def make_test_pdf(text: str = None) -> bytes:
        if text is None:
            text = (
                "Transformers revolutionized natural language processing. "
                "The attention mechanism allows models to focus on relevant parts of the input. "
                "BERT and GPT are two prominent transformer-based architectures. "
                "Retrieval-augmented generation combines search with language models. "
                "Chunking is an important preprocessing step for document indexing."
            )
        buf = io.BytesIO()
        c = rl_canvas.Canvas(buf, pagesize=letter)
        y = 750
        for line in text.split(". "):
            c.drawString(50, y, line.strip() + ".")
            y -= 20
            if y < 50:
                c.showPage()
                y = 750
        c.save()
        return buf.getvalue()

    HAS_REPORTLAB = True

except ImportError:
    HAS_REPORTLAB = False

    def make_test_pdf(text: str = None) -> bytes:
        return _make_minimal_pdf(text or "Test document for RAG pipeline.")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_chunks() -> list[dict]:
    """A small set of realistic chunk dicts."""
    return [
        {
            "text": (
                "Transformers use self-attention mechanisms to process sequences. "
                "The attention mechanism computes a weighted sum of values. "
                "This allows the model to focus on relevant parts of the input."
            ),
            "source_file": "attention_paper.pdf",
            "page": 1,
            "chunk_index": 0,
        },
        {
            "text": (
                "BERT is a bidirectional transformer pre-trained on masked language modeling. "
                "It achieves state-of-the-art results on many NLP benchmarks. "
                "Fine-tuning BERT requires only adding a classification head."
            ),
            "source_file": "attention_paper.pdf",
            "page": 2,
            "chunk_index": 1,
        },
        {
            "text": (
                "Retrieval-augmented generation (RAG) combines dense retrieval with generation. "
                "A retriever fetches relevant documents from a knowledge base. "
                "The generator conditions on both the query and retrieved documents."
            ),
            "source_file": "rag_paper.pdf",
            "page": 1,
            "chunk_index": 2,
        },
        {
            "text": (
                "BM25 is a probabilistic keyword retrieval algorithm. "
                "It ranks documents based on term frequency and inverse document frequency. "
                "BM25 is robust and often competitive with dense retrieval methods."
            ),
            "source_file": "ir_textbook.pdf",
            "page": 45,
            "chunk_index": 3,
        },
        {
            "text": (
                "Cosine similarity measures the angle between two vectors. "
                "It is computed as the dot product of normalised vectors. "
                "Values range from -1 to 1, with 1 indicating identical direction."
            ),
            "source_file": "ir_textbook.pdf",
            "page": 46,
            "chunk_index": 4,
        },
    ]


@pytest.fixture
def sample_embeddings(sample_chunks) -> np.ndarray:
    """
    Deterministic random embeddings (seeded) for the sample chunks.
    Shape: (5, 128) float32.
    """
    rng = np.random.default_rng(42)
    return rng.random((len(sample_chunks), 128)).astype(np.float32)


@pytest.fixture
def test_pdf_bytes() -> bytes:
    return make_test_pdf()


@pytest.fixture
def requires_reportlab():
    """Skip test if reportlab is not available."""
    if not HAS_REPORTLAB:
        pytest.skip("reportlab not installed — skipping PDF parsing test")
