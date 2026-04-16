"""
Embedding client for Mistral's mistral-embed model.

Batches requests to stay under the API token limit and returns
numpy float32 arrays for efficient cosine similarity computation.
"""

import os
import httpx
import numpy as np
from typing import Sequence

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "CF2DvjloshzasO0mtBkPj44fo2nXDwPk")
EMBED_MODEL = "mistral-embed"
EMBED_URL = "https://api.mistral.ai/v1/embeddings"
BATCH_SIZE = 16          # Mistral allows up to 16k tokens per request; keep batches small
TIMEOUT_SECONDS = 30.0


def _embed_batch(texts: list[str]) -> list[list[float]]:
    """Call Mistral Embeddings API for a single batch."""
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": EMBED_MODEL,
        "input": texts,
        "encoding_format": "float",
    }
    response = httpx.post(
        EMBED_URL,
        json=payload,
        headers=headers,
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    data = response.json()
    # data["data"] is sorted by index
    return [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]


def embed_texts(texts: Sequence[str]) -> np.ndarray:
    """
    Embed a list of texts.
    Returns shape (N, D) float32 array where D is the embedding dimension.
    """
    all_embeddings: list[list[float]] = []
    texts = list(texts)

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        all_embeddings.extend(_embed_batch(batch))

    return np.array(all_embeddings, dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    """Embed a single query string. Returns shape (D,) float32 array."""
    return embed_texts([text])[0]
