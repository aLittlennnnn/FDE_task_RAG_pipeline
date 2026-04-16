"""
Answer generation with Mistral, plus post-processing:

  1. Template selection by intent (factual / list / comparison).
  2. Evidence threshold — refuse to answer if best chunk score is too low.
  3. Hallucination filter — post-hoc "evidence check" per sentence.
  4. Citation extraction — map answer sentences back to source chunks.
"""

from __future__ import annotations

import os
import re
import httpx
from typing import Literal

from .intent_detector import Intent

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise EnvironmentError("MISTRAL_API_KEY is not set. Run: export MISTRAL_API_KEY=your_key_here")
CHAT_URL = "https://api.mistral.ai/v1/chat/completions"
GENERATE_MODEL = "mistral-medium-latest"

INSUFFICIENT_EVIDENCE_MSG = (
    "I couldn't find sufficient evidence in the uploaded documents to answer "
    "your question reliably. Please upload relevant documents or rephrase your query."
)

CONVERSATIONAL_RESPONSES = {
    "hi": "Hello! How can I help you with your documents today?",
    "hello": "Hi there! Feel free to ask me anything about your uploaded documents.",
    "hey": "Hey! What would you like to know from your documents?",
}

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_BASE_CONTEXT = """You are a precise assistant that answers questions strictly from the provided document excerpts.

Rules:
- Use ONLY the information in the CONTEXT section below. Do not add outside knowledge.
- If the context lacks enough information, say so clearly and briefly.
- Cite sources inline like this: [Source: filename.pdf, p.3]
- Use clear formatting: blank lines between paragraphs, numbered steps, or bullet points where appropriate.
- Never run multiple points together on a single line — each distinct point must be on its own line.

CONTEXT:
{context}
"""

_FACTUAL_TEMPLATE = _BASE_CONTEXT + """
Answer the question directly and concisely.
Use paragraphs with blank lines between them if the answer has multiple parts.

Question: {question}

Answer:"""

_LIST_TEMPLATE = _BASE_CONTEXT + """
Provide a numbered list. Format strictly as:

1. First item — explanation here.

2. Second item — explanation here.

3. Third item — explanation here.

Each item must be on its own line with a blank line between items.

Question: {question}

Answer:"""

_COMPARISON_TEMPLATE = _BASE_CONTEXT + """
Structure your answer as a comparison with clearly labelled sections.
Use this format:

**[First subject]**
- Key point 1
- Key point 2

**[Second subject]**
- Key point 1
- Key point 2

**Key differences**
- Difference 1
- Difference 2

Question: {question}

Answer:"""

_TEMPLATES = {
    "factual": _FACTUAL_TEMPLATE,
    "list": _LIST_TEMPLATE,
    "comparison": _COMPARISON_TEMPLATE,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a context block."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[{i}] Source: {chunk['source_file']}, page {chunk['page']}\n"
            f"{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


def _call_mistral_chat(messages: list[dict], max_tokens: int = 800) -> str:
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GENERATE_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }
    resp = httpx.post(CHAT_URL, json=payload, headers=headers, timeout=30.0)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Hallucination filter
# ---------------------------------------------------------------------------

_EVIDENCE_CHECK_SYSTEM = """You are a fact-checking assistant.
Given a CLAIM and a set of DOCUMENT EXCERPTS, determine if the claim is directly supported by the excerpts.
Answer with only "supported" or "unsupported"."""


def _is_supported(sentence: str, context: str) -> bool:
    """Check if a single sentence is supported by the context."""
    if len(sentence.split()) < 5:
        return True  # Too short to check (likely a header or transition)
    prompt = f"CLAIM: {sentence}\n\nDOCUMENT EXCERPTS:\n{context[:2000]}"
    try:
        result = _call_mistral_chat(
            [
                {"role": "system", "content": _EVIDENCE_CHECK_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=5,
        )
        return result.lower().startswith("supported")
    except Exception:
        return True  # On API error, don't filter


def hallucination_filter(answer: str, context: str) -> str:
    """
    Remove sentences from the answer that are not supported by the context.
    Adds a note if sentences were removed.
    """
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    kept = []
    removed_count = 0

    for sent in sentences:
        if _is_supported(sent, context):
            kept.append(sent)
        else:
            removed_count += 1

    filtered = " ".join(kept)
    if removed_count > 0:
        filtered += (
            f"\n\n[Note: {removed_count} sentence(s) were removed — "
            "could not be verified against the source documents.]"
        )
    return filtered


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------

def generate_answer(
    question: str,
    chunks: list[dict],
    intent: Intent,
    similarity_threshold: float = 0.35,
    apply_hallucination_filter: bool = True,
) -> tuple[str, bool]:
    """
    Generate an answer from retrieved chunks.

    Returns:
        (answer_text, sufficient_evidence_bool)
    """
    if not chunks:
        return INSUFFICIENT_EVIDENCE_MSG, False

    # Evidence threshold check — use the best semantic score available
    best_score = max(c.get("score", 0.0) for c in chunks)
    if best_score < similarity_threshold:
        return INSUFFICIENT_EVIDENCE_MSG, False

    context = _build_context(chunks)
    template = _TEMPLATES.get(intent, _FACTUAL_TEMPLATE)
    prompt = template.format(context=context, question=question)

    messages = [{"role": "user", "content": prompt}]
    answer = _call_mistral_chat(messages, max_tokens=900)

    if apply_hallucination_filter:
        answer = hallucination_filter(answer, context)

    return answer, True


def conversational_reply(question: str) -> str:
    """Handle conversational messages without KB search."""
    q_lower = question.lower().strip().rstrip("?!.,")
    for key, response in CONVERSATIONAL_RESPONSES.items():
        if key in q_lower:
            return response
    # Generic conversational reply
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful document assistant. The user is making small talk. "
                "Respond briefly and warmly, and remind them you can answer questions "
                "about their uploaded documents."
            ),
        },
        {"role": "user", "content": question},
    ]
    try:
        return _call_mistral_chat(messages, max_tokens=100)
    except Exception:
        return "Hello! I'm here to help you find information in your documents."


def refused_reply(question: str) -> str:
    """Return a polite refusal message."""
    return (
        "I'm unable to answer this type of question. I can't provide personal data lookups, "
        "specific medical advice, or specific legal advice. For those topics, please consult "
        "a qualified professional. I'm happy to help you find information within your uploaded documents!"
    )
