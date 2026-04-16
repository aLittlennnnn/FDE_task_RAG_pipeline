"""
BM25 keyword search — implemented from scratch, no external libraries.

Algorithm:
    BM25 Okapi (Robertson et al., 1994) with standard hyperparameters:
        k1 = 1.5  (term-frequency saturation)
        b  = 0.75 (document-length normalisation)

Index structure:
    - inverted_index: term -> {chunk_id: term_freq}
    - doc_lengths:    chunk_id -> token count
    - idf_cache:      term -> idf score (lazily computed, cleared on update)

Tokenisation:
    Lowercase, strip punctuation, simple whitespace split.
    No stemming — keeps implementation dependency-free.
"""

from __future__ import annotations

import math
import re
import threading
from collections import defaultdict
from typing import Optional


# ------------------------------------------------------------------
# Tokeniser
# ------------------------------------------------------------------

_PUNCT = re.compile(r"[^a-z0-9\s]")
_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would could should may might shall can of in on at to for "
    "with by from and or but not this that these those it its i you "
    "he she we they what which who whom when where why how".split()
)


def _tokenise(text: str) -> list[str]:
    text = text.lower()
    text = _PUNCT.sub(" ", text)
    tokens = text.split()
    return [t for t in tokens if t and t not in _STOPWORDS and len(t) > 1]


# ------------------------------------------------------------------
# BM25 index
# ------------------------------------------------------------------

class BM25Index:
    """Thread-safe BM25 index over chunk texts."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._lock = threading.Lock()

        # chunk_id (int position in self._chunks) -> token count
        self._doc_lengths: list[int] = []
        # term -> {chunk_id: term_freq}
        self._inverted: dict[str, dict[int, int]] = defaultdict(dict)
        # parallel metadata list (same indexing as _doc_lengths)
        self._chunks: list[dict] = []
        # cached idf values; cleared when index changes
        self._idf_cache: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: list[dict]) -> None:
        with self._lock:
            start = len(self._chunks)
            for i, chunk in enumerate(chunks):
                cid = start + i
                tokens = _tokenise(chunk["text"])
                self._doc_lengths.append(len(tokens))
                for token in tokens:
                    self._inverted[token][cid] = self._inverted[token].get(cid, 0) + 1
            self._chunks.extend(chunks)
            self._idf_cache.clear()

    def remove_file(self, filename: str) -> None:
        """Rebuild index without chunks from the given file."""
        with self._lock:
            keep = [c for c in self._chunks if c["source_file"] != filename]
            if len(keep) == len(self._chunks):
                return
            # Full rebuild (infrequent operation)
            self._chunks = []
            self._doc_lengths = []
            self._inverted = defaultdict(dict)
            self._idf_cache = {}
            # Re-add without locking (we already hold it)
            for chunk in keep:
                cid = len(self._chunks)
                tokens = _tokenise(chunk["text"])
                self._doc_lengths.append(len(tokens))
                for token in tokens:
                    self._inverted[token][cid] = self._inverted[token].get(cid, 0) + 1
                self._chunks.append(chunk)

    def clear(self) -> None:
        with self._lock:
            self._chunks = []
            self._doc_lengths = []
            self._inverted = defaultdict(dict)
            self._idf_cache = {}

    # ------------------------------------------------------------------
    # IDF
    # ------------------------------------------------------------------

    def _idf(self, term: str) -> float:
        if term in self._idf_cache:
            return self._idf_cache[term]
        N = len(self._chunks)
        df = len(self._inverted.get(term, {}))
        if df == 0:
            score = 0.0
        else:
            score = math.log((N - df + 0.5) / (df + 0.5) + 1)
        self._idf_cache[term] = score
        return score

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Return top_k chunks ranked by BM25 score.
        Each result dict has the original chunk fields plus a 'bm25_score'.
        """
        with self._lock:
            if not self._chunks:
                return []

            query_tokens = _tokenise(query)
            if not query_tokens:
                return []

            N = len(self._chunks)
            avg_dl = sum(self._doc_lengths) / N if N else 1

            scores: dict[int, float] = {}

            for term in query_tokens:
                idf = self._idf(term)
                if idf == 0:
                    continue
                for cid, tf in self._inverted.get(term, {}).items():
                    dl = self._doc_lengths[cid]
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * dl / avg_dl)
                    scores[cid] = scores.get(cid, 0.0) + idf * numerator / denominator

            if not scores:
                return []

            sorted_cids = sorted(scores, key=lambda c: scores[c], reverse=True)[:top_k]
            results = []
            for cid in sorted_cids:
                chunk = dict(self._chunks[cid])
                chunk["bm25_score"] = scores[cid]
                results.append(chunk)

            return results


# Singleton
bm25_index = BM25Index()
