"""
Query transformation to improve retrieval quality.

Two strategies applied in sequence:

1. Query rewriting
   Ask the LLM to rephrase the question in a way that is more explicit and
   self-contained (removes pronouns, expands acronyms, adds context).

2. HyDE (Hypothetical Document Embeddings — Gao et al., 2022)
   Generate a short hypothetical answer to the question.  We embed THAT text
   instead of the raw question, because its vocabulary will better match the
   actual document chunks (documents answer questions; queries ask them).

The final search string is: rewritten_query + " " + hypothetical_answer.
This hybrid gives both lexical precision (BM25 benefits from keywords) and
semantic alignment (cosine search benefits from the topic vocabulary).
"""

from __future__ import annotations

import os
import httpx

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "CF2DvjloshzasO0mtBkPj44fo2nXDwPk")
CHAT_URL = "https://api.mistral.ai/v1/chat/completions"
TRANSFORM_MODEL = "mistral-small-latest"

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _call_mistral(system: str, user: str, max_tokens: int = 200) -> str:
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": TRANSFORM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }
    try:
        resp = httpx.post(CHAT_URL, json=payload, headers=headers, timeout=15.0)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return user  # fall back to original on error


_REWRITE_SYSTEM = """You are a query rewriting assistant.
Rewrite the user's question to be more explicit, self-contained, and specific.
- Expand abbreviations.
- Replace pronouns with their referents if clear.
- Add relevant technical keywords if appropriate.
- Keep it as a question.
- Output ONLY the rewritten question, nothing else."""

_HYDE_SYSTEM = """You are a helpful assistant generating a hypothetical document excerpt.
Given a question, write 2-3 sentences that would appear in a document that answers it.
Write it as factual prose (not as an answer to the user), as if it were a paragraph from a reference document.
Output ONLY the hypothetical excerpt, nothing else."""


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------

def transform_query(query: str) -> str:
    """
    Rewrite the query and append a HyDE excerpt.

    Returns a combined string for embedding and BM25 search.
    """
    rewritten = _call_mistral(_REWRITE_SYSTEM, query, max_tokens=150)
    hyde = _call_mistral(_HYDE_SYSTEM, rewritten, max_tokens=200)
    return f"{rewritten} {hyde}"
