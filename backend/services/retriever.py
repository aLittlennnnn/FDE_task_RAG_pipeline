"""
Hybrid retrieval: semantic search + BM25, merged via Reciprocal Rank Fusion.

Reciprocal Rank Fusion (RRF):
    score(d) = Σ  1 / (k + rank_i(d))
    where k=60 is a smoothing constant and rank_i is the rank in list i.

This simple fusion strategy has been shown to outperform individual rankings
and requires no training. Results are then optionally re-ranked by Mistral.
"""

from __future__ import annotations

import numpy as np
from .vector_store import store
from .bm25 import bm25_index
from .embedder import embed_query


RRF_K = 60  # smoothing constant


def _reciprocal_rank_fusion(
    semantic_results: list[dict],
    keyword_results: list[dict],
) -> list[dict]:
    """
    Merge two ranked lists via RRF.
    Chunks are keyed by (source_file, chunk_index) to deduplicate.
    """
    rrf_scores: dict[tuple, float] = {}
    chunk_map: dict[tuple, dict] = {}

    for rank, chunk in enumerate(semantic_results, start=1):
        key = (chunk["source_file"], chunk["chunk_index"])
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
        chunk_map[key] = chunk

    for rank, chunk in enumerate(keyword_results, start=1):
        key = (chunk["source_file"], chunk["chunk_index"])
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
        if key not in chunk_map:
            chunk_map[key] = chunk

    sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)
    merged = []
    for key in sorted_keys:
        chunk = dict(chunk_map[key])
        chunk["rrf_score"] = rrf_scores[key]
        # Preserve the semantic similarity score if present
        if "score" not in chunk:
            chunk["score"] = 0.0
        merged.append(chunk)

    return merged


def hybrid_search(
    query: str,
    top_k: int = 5,
    semantic_k: int = 10,
    keyword_k: int = 10,
) -> list[dict]:
    """
    Run hybrid retrieval and return top_k fused results.

    Steps:
        1. Embed the query with Mistral.
        2. Semantic search in VectorStore (cosine similarity).
        3. Keyword search with BM25.
        4. Merge lists via RRF.
        5. Return top_k.
    """
    query_vec = embed_query(query)

    semantic_results = store.search(query_vec, top_k=semantic_k)
    keyword_results = bm25_index.search(query, top_k=keyword_k)

    merged = _reciprocal_rank_fusion(semantic_results, keyword_results)
    return merged[:top_k]
