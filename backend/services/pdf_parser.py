import re
import io
from pathlib import Path
from typing import Generator

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def _extract_text_by_page(pdf_bytes: bytes) -> list[tuple[int, str]]:
    """Return a list of (page_number, page_text) tuples (1-indexed pages)."""
    pages: list[tuple[int, str]] = []
    pdf_file = io.BytesIO(pdf_bytes)

    for page_num, page_layout in enumerate(extract_pages(pdf_file), start=1):
        page_text_parts = []
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                page_text_parts.append(element.get_text())
        raw = " ".join(page_text_parts)
        cleaned = _clean_text(raw)
        if cleaned:
            pages.append((page_num, cleaned))

    return pages


def _clean_text(text: str) -> str:
    """Normalise whitespace and remove junk characters."""
    # Replace unusual whitespace
    text = re.sub(r"[\r\n\t\f\v]+", " ", text)
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    # Remove null bytes / control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Sentence splitting (no NLTK)
# ---------------------------------------------------------------------------

_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _split_sentences(text: str) -> list[str]:
    """Simple regex sentence splitter."""
    sentences = _SENT_BOUNDARY.split(text)
    return [s.strip() for s in sentences if s.strip()]


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _chunk_sentences(
    sentences: list[str],
    chunk_size: int = 800,
    overlap: int = 150,
) -> Generator[str, None, None]:
    """
    Sliding-window chunking over a sentence list.

    We greedily add sentences until the chunk reaches `chunk_size` chars,
    then backtrack by `overlap` chars worth of sentences for the next window.
    This keeps chunk boundaries at sentence edges (better for retrieval).
    """
    current_chunk: list[str] = []
    current_len = 0

    for sent in sentences:
        current_chunk.append(sent)
        current_len += len(sent) + 1  # +1 for space separator

        if current_len >= chunk_size:
            chunk_text = " ".join(current_chunk)
            yield chunk_text

            # Rewind: drop sentences from front until we're within `overlap` chars
            while current_len > overlap and current_chunk:
                removed = current_chunk.pop(0)
                current_len -= len(removed) + 1

    # Yield any remaining text
    if current_chunk:
        yield " ".join(current_chunk)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

MIN_CHUNK_LENGTH = 50  # characters — skip near-empty chunks


def extract_chunks(
    pdf_bytes: bytes,
    filename: str,
    chunk_size: int = 800,
    overlap: int = 150,
) -> list[dict]:
    """
    Extract text from a PDF and return a flat list of chunk dicts:

        {
            "text":        str,
            "source_file": str,
            "page":        int,
            "chunk_index": int,   # global index across the whole document
        }
    """
    pages = _extract_text_by_page(pdf_bytes)
    chunks: list[dict] = []
    global_idx = 0

    for page_num, page_text in pages:
        sentences = _split_sentences(page_text)
        for chunk_text in _chunk_sentences(sentences, chunk_size, overlap):
            if len(chunk_text) < MIN_CHUNK_LENGTH:
                continue
            chunks.append({
                "text": chunk_text,
                "source_file": filename,
                "page": page_num,
                "chunk_index": global_idx,
            })
            global_idx += 1

    return chunks
