"""
Two-stage detection:
    1. Fast heuristic: if the message is a short greeting / small-talk phrase
       we classify immediately without an LLM call.
    2. LLM classification: ask Mistral to label the intent in one token.

Intent labels:
    - conversational   : greetings, small-talk; no KB search needed
    - factual          : direct factual question → retrieve + answer
    - list             : asks for a list/enumeration → structured output
    - comparison       : asks to compare multiple items
    - refused          : PII request, legal/medical advice, harmful content

Refusal policies:
    - PII requests  (e.g. "what is John's SSN")
    - Medical advice (e.g. "should I take this drug")
    - Legal advice   (e.g. "am I liable for …")
"""

from __future__ import annotations

import re
import os
import httpx
from typing import Literal

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise EnvironmentError("MISTRAL_API_KEY is not set. Run: export MISTRAL_API_KEY=your_key_here")
CHAT_URL = "https://api.mistral.ai/v1/chat/completions"
INTENT_MODEL = "mistral-small-latest"

Intent = Literal["conversational", "factual", "list", "comparison", "refused"]

# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------

_GREETING_PATTERNS = re.compile(
    r"^(hi+|hello+|hey+|howdy|greetings|good\s*(morning|afternoon|evening|day)|"
    r"what'?s up|yo|sup|hiya|how are you|how's it going|nice to meet you)[?!.,\s]*$",
    re.IGNORECASE,
)

_PII_PATTERNS = re.compile(
    r"\b(ssn|social security|passport number|credit card|cvv|date of birth|dob"
    r"|driver.?s license|medical record|bank account)\b",
    re.IGNORECASE,
)

_MEDICAL_PATTERNS = re.compile(
    r"\b(should i (take|use|stop|start)|can i take|is it safe to|dosage for me|"
    r"do i have|am i sick|diagnos)\b",
    re.IGNORECASE,
)

_LEGAL_PATTERNS = re.compile(
    r"\b(am i liable|can i sue|is it legal|legal advice|should i sign|"
    r"am i at fault|my rights in)\b",
    re.IGNORECASE,
)


def _heuristic_check(query: str) -> Intent | None:
    """Return an intent if a fast pattern matches, else None."""
    q = query.strip()
    if _GREETING_PATTERNS.match(q):
        return "conversational"
    if _PII_PATTERNS.search(q):
        return "refused"
    if _MEDICAL_PATTERNS.search(q):
        return "refused"
    if _LEGAL_PATTERNS.search(q):
        return "refused"
    return None


# ---------------------------------------------------------------------------
# LLM classification
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are an intent classifier for a document Q&A system.
Classify the user's message into exactly ONE of these categories and respond with only that word:
  conversational - greetings, small talk, not a real question
  factual        - asks for a specific fact or explanation
  list           - asks for a list, steps, or enumeration
  comparison     - asks to compare or contrast items
  refused        - requests personal/sensitive information, specific medical or legal advice
"""


def _llm_classify(query: str) -> Intent:
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": INTENT_MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        "max_tokens": 5,
        "temperature": 0.0,
    }
    try:
        resp = httpx.post(CHAT_URL, json=payload, headers=headers, timeout=10.0)
        resp.raise_for_status()
        label = resp.json()["choices"][0]["message"]["content"].strip().lower()
        valid: set[Intent] = {"conversational", "factual", "list", "comparison", "refused"}
        return label if label in valid else "factual"  # type: ignore[return-value]
    except Exception:
        return "factual"


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------

def detect_intent(query: str) -> Intent:
    """
    Detect the intent of a query.
    Returns one of: 'conversational', 'factual', 'list', 'comparison', 'refused'.
    """
    fast = _heuristic_check(query)
    if fast is not None:
        return fast
    return _llm_classify(query)


def needs_retrieval(intent: Intent) -> bool:
    """Return True if the intent requires a knowledge base search."""
    return intent not in ("conversational", "refused")
