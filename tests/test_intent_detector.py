"""
Unit tests for intent detection (services/intent_detector.py).

Split into two classes:
  1. TestHeuristics  — tests the fast regex path, no Mistral API calls needed.
  2. TestLLMClassify — mocks httpx to test the LLM classification path.
"""

import pytest
from unittest.mock import patch, MagicMock

from backend.services.intent_detector import (
    _heuristic_check,
    detect_intent,
    needs_retrieval,
    _GREETING_PATTERNS,
    _PII_PATTERNS,
    _MEDICAL_PATTERNS,
    _LEGAL_PATTERNS,
)


# ---------------------------------------------------------------------------
# Heuristic tests (no network calls)
# ---------------------------------------------------------------------------

class TestHeuristics:

    # -- Greetings --

    @pytest.mark.parametrize("query", [
        "hi",
        "Hi",
        "HI",
        "hello",
        "Hello!",
        "hey",
        "Hey there",
        "good morning",
        "good afternoon",
        "what's up",
        "howdy",
        "greetings",
        "how are you",
        "nice to meet you",
    ])
    def test_greeting_detected_as_conversational(self, query):
        assert _heuristic_check(query) == "conversational"

    @pytest.mark.parametrize("query", [
        "What is attention mechanism?",
        "Explain transformers.",
        "How does BM25 work?",
        "List the key findings.",
        "Compare BERT and GPT.",
    ])
    def test_non_greeting_not_conversational(self, query):
        result = _heuristic_check(query)
        assert result != "conversational"

    # -- PII --

    @pytest.mark.parametrize("query", [
        "What is John's SSN?",
        "Give me the social security number",
        "What is the passport number for this person?",
        "Show me the credit card number",
        "What is the CVV?",
        "Tell me the bank account number",
        "Medical record for patient ID 123",
    ])
    def test_pii_detected_as_refused(self, query):
        assert _heuristic_check(query) == "refused"

    # -- Medical advice --

    @pytest.mark.parametrize("query", [
        "Should I take ibuprofen?",
        "Can I take aspirin with alcohol?",
        "Is it safe to take this medication?",
        "Do I have diabetes?",
        "Am I sick based on these symptoms?",
    ])
    def test_medical_advice_refused(self, query):
        assert _heuristic_check(query) == "refused"

    # -- Legal advice --

    @pytest.mark.parametrize("query", [
        "Am I liable for this accident?",
        "Can I sue my employer?",
        "Is it legal to do this in California?",
        "Should I sign this contract?",
        "Am I at fault for the crash?",
    ])
    def test_legal_advice_refused(self, query):
        assert _heuristic_check(query) == "refused"

    # -- Normal factual queries should not be caught by heuristics --

    @pytest.mark.parametrize("query", [
        "What does the paper say about transformers?",
        "How many parameters does GPT-3 have?",
        "What are the evaluation metrics?",
        "List the steps in the algorithm.",
        "Compare performance of BERT vs RoBERTa.",
    ])
    def test_factual_queries_pass_heuristics(self, query):
        # Heuristics should return None → falls through to LLM
        assert _heuristic_check(query) is None


# ---------------------------------------------------------------------------
# LLM classification tests (mocked)
# ---------------------------------------------------------------------------

def _mock_mistral_response(label: str):
    """Build a fake httpx response returning the given label."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": label}}]
    }
    return mock_resp


class TestDetectIntentWithMockedLLM:

    @patch("backend.services.intent_detector.httpx.post")
    def test_factual_intent(self, mock_post):
        mock_post.return_value = _mock_mistral_response("factual")
        result = detect_intent("What are the main contributions of this paper?")
        assert result == "factual"

    @patch("backend.services.intent_detector.httpx.post")
    def test_list_intent(self, mock_post):
        mock_post.return_value = _mock_mistral_response("list")
        result = detect_intent("Give me a list of all evaluation metrics.")
        assert result == "list"

    @patch("backend.services.intent_detector.httpx.post")
    def test_comparison_intent(self, mock_post):
        mock_post.return_value = _mock_mistral_response("comparison")
        result = detect_intent("Compare BERT and GPT architectures.")
        assert result == "comparison"

    @patch("backend.services.intent_detector.httpx.post")
    def test_unknown_label_falls_back_to_factual(self, mock_post):
        mock_post.return_value = _mock_mistral_response("unknown_label")
        result = detect_intent("Tell me something interesting.")
        assert result == "factual"

    @patch("backend.services.intent_detector.httpx.post")
    def test_api_error_falls_back_to_factual(self, mock_post):
        mock_post.side_effect = Exception("Network timeout")
        result = detect_intent("What is the accuracy of this model?")
        assert result == "factual"  # Graceful fallback

    # -- Greetings bypass the LLM entirely --

    @patch("backend.services.intent_detector.httpx.post")
    def test_greeting_does_not_call_api(self, mock_post):
        result = detect_intent("hello")
        mock_post.assert_not_called()
        assert result == "conversational"

    @patch("backend.services.intent_detector.httpx.post")
    def test_pii_does_not_call_api(self, mock_post):
        result = detect_intent("What is the social security number?")
        mock_post.assert_not_called()
        assert result == "refused"


# ---------------------------------------------------------------------------
# needs_retrieval
# ---------------------------------------------------------------------------

class TestNeedsRetrieval:

    def test_factual_needs_retrieval(self):
        assert needs_retrieval("factual") is True

    def test_list_needs_retrieval(self):
        assert needs_retrieval("list") is True

    def test_comparison_needs_retrieval(self):
        assert needs_retrieval("comparison") is True

    def test_conversational_no_retrieval(self):
        assert needs_retrieval("conversational") is False

    def test_refused_no_retrieval(self):
        assert needs_retrieval("refused") is False
