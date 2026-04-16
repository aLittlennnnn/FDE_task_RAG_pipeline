"""
In-memory vector store — no third-party vector database.

Storage:
    A flat numpy matrix (N × D) of all chunk embeddings.
    A parallel list of chunk metadata dicts.

Search:
    Cosine similarity computed via matrix multiplication after L2 normalisation.
    O(N) scan — sufficient for moderate corpus sizes; can be replaced with
    a HNSW-style index if N grows large.
"""

from __future__ import annotations

import threading
import numpy as np
from typing import Optional


class VectorStore:
    """Thread-safe, in-memory vector store."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # (N, D) float32 — L2-normalised for fast cosine via dot product
        self._matrix: Optional[np.ndarray] = None
        # Parallel list of chunk metadata
        self._chunks: list[dict] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: list[dict], embeddings: np.ndarray) -> None:
        """
        Add chunks and their embeddings to the store.

        Args:
            chunks:     list of metadata dicts (must have 'text', 'source_file',
                        'page', 'chunk_index' keys)
            embeddings: (len(chunks), D) float32 array
        """
        assert len(chunks) == len(embeddings), "chunks and embeddings must be same length"

        # L2-normalise so dot-product == cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normalised = (embeddings / norms).astype(np.float32)

        with self._lock:
            if self._matrix is None:
                self._matrix = normalised
            else:
                self._matrix = np.vstack([self._matrix, normalised])
            self._chunks.extend(chunks)

    def remove_file(self, filename: str) -> int:
        """Remove all chunks from a given file. Returns number removed."""
        with self._lock:
            keep_indices = [
                i for i, c in enumerate(self._chunks) if c["source_file"] != filename
            ]
            if len(keep_indices) == len(self._chunks):
                return 0  # nothing removed

            removed = len(self._chunks) - len(keep_indices)
            self._chunks = [self._chunks[i] for i in keep_indices]
            if keep_indices:
                self._matrix = self._matrix[keep_indices]
            else:
                self._matrix = None
            return removed

    def clear(self) -> None:
        with self._lock:
            self._matrix = None
            self._chunks = []

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> list[dict]:
        """
        Return top_k chunks ranked by cosine similarity.

        Returns list of dicts with extra 'score' field, sorted descending.
        """
        with self._lock:
            if self._matrix is None or len(self._chunks) == 0:
                return []

            # L2-normalise query
            q = query_embedding.astype(np.float32)
            norm = np.linalg.norm(q)
            if norm > 0:
                q = q / norm

            # Cosine similarity via dot product (matrix is pre-normalised)
            scores: np.ndarray = self._matrix @ q  # shape (N,)

            k = min(top_k, len(scores))
            top_indices = np.argpartition(scores, -k)[-k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

            results = []
            for idx in top_indices:
                chunk = dict(self._chunks[idx])
                chunk["score"] = float(scores[idx])
                results.append(chunk)

            return results

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    @property
    def total_chunks(self) -> int:
        return len(self._chunks)

    @property
    def indexed_files(self) -> list[str]:
        return list({c["source_file"] for c in self._chunks})


# Singleton instance shared across the app
store = VectorStore()
