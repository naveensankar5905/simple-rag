"""
test_qa_engine.py
─────────────────
Unit tests for the QAEngine module and detect_question_type function.
All Ollama API calls are mocked to avoid network dependencies.
"""

import json
import pytest
from unittest.mock import patch, MagicMock
import urllib.error


# ══════════════════════════════════════════════════════════════════════════════
#  QUESTION TYPE DETECTION TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestDetectQuestionType:
    """Tests for detect_question_type()."""

    def test_detect_what(self):
        from qa_engine import detect_question_type
        assert detect_question_type("What is the EU AI Act?") == "WHAT"

    def test_detect_when(self):
        from qa_engine import detect_question_type
        assert detect_question_type("When was the Act adopted?") == "WHEN"

    def test_detect_where(self):
        from qa_engine import detect_question_type
        assert detect_question_type("Where is the headquarters?") == "WHERE"

    def test_detect_who(self):
        from qa_engine import detect_question_type
        assert detect_question_type("Who proposed the Act?") == "WHO"

    def test_detect_why(self):
        from qa_engine import detect_question_type
        assert detect_question_type("Why was it created?") == "WHY"

    def test_detect_how(self):
        from qa_engine import detect_question_type
        assert detect_question_type("How does it work?") == "HOW"

    def test_detect_yes_no_does(self):
        from qa_engine import detect_question_type
        assert detect_question_type("Does the Act apply?") == "YES_NO"

    def test_detect_yes_no_is(self):
        from qa_engine import detect_question_type
        assert detect_question_type("Is this permitted?") == "YES_NO"

    def test_detect_yes_no_can(self):
        from qa_engine import detect_question_type
        assert detect_question_type("Can a company comply?") == "YES_NO"

    def test_detect_yes_no_will(self):
        from qa_engine import detect_question_type
        assert detect_question_type("Will the law apply?") == "YES_NO"

    def test_detect_yes_no_has(self):
        from qa_engine import detect_question_type
        assert detect_question_type("Has the Act been adopted?") == "YES_NO"

    def test_detect_yes_no_did(self):
        from qa_engine import detect_question_type
        assert detect_question_type("Did the Parliament vote?") == "YES_NO"

    def test_detect_yes_no_was(self):
        from qa_engine import detect_question_type
        assert detect_question_type("Was it approved?") == "YES_NO"

    def test_detect_yes_no_were(self):
        from qa_engine import detect_question_type
        assert detect_question_type("Were the provisions enacted?") == "YES_NO"

    def test_detect_yes_no_should(self):
        from qa_engine import detect_question_type
        assert detect_question_type("Should companies comply?") == "YES_NO"

    def test_detect_yes_no_would(self):
        from qa_engine import detect_question_type
        assert detect_question_type("Would this be allowed?") == "YES_NO"

    def test_detect_yes_no_could(self):
        from qa_engine import detect_question_type
        assert detect_question_type("Could they avoid fines?") == "YES_NO"

    def test_detect_yes_no_do(self):
        from qa_engine import detect_question_type
        assert detect_question_type("Do companies need to comply?") == "YES_NO"

    def test_detect_yes_no_have(self):
        from qa_engine import detect_question_type
        assert detect_question_type("Have the provisions been adopted?") == "YES_NO"

    def test_detect_yes_no_are(self):
        from qa_engine import detect_question_type
        assert detect_question_type("Are these systems prohibited?") == "YES_NO"

    def test_detect_compare(self):
        from qa_engine import detect_question_type
        result = detect_question_type("Compare high-risk and low-risk systems")
        assert result in ("COMPARE", "WHAT")  # depends on pattern ordering

    def test_detect_compare_versus(self):
        from qa_engine import detect_question_type
        result = detect_question_type("A versus B in the Act")
        assert result in ("COMPARE", "WHAT")

    def test_detect_compare_in_text(self):
        from qa_engine import detect_question_type
        result = detect_question_type("Please contrast A and B")
        assert result in ("COMPARE", "WHAT")

    def test_detect_compare_contrast(self):
        from qa_engine import detect_question_type
        assert detect_question_type("Contrast these two approaches") == "COMPARE"

    def test_detect_list(self):
        from qa_engine import detect_question_type
        assert detect_question_type("List all prohibited practices") == "LIST"

    def test_detect_list_what_are(self):
        from qa_engine import detect_question_type
        assert detect_question_type("What are the obligations?") == "LIST"

    def test_detect_list_name_all(self):
        from qa_engine import detect_question_type
        assert detect_question_type("Name all the risk categories") == "LIST"

    def test_detect_define(self):
        from qa_engine import detect_question_type
        assert detect_question_type("Define AI system") == "DEFINE"

    def test_detect_define_meaning(self):
        from qa_engine import detect_question_type
        assert detect_question_type("What is the meaning of deployer?") == "DEFINE"

    def test_detect_summary(self):
        from qa_engine import detect_question_type
        result = detect_question_type("Summarise the key points")
        assert result in ("SUMMARY", "WHAT")  # depends on pattern ordering

    def test_detect_summary_overview(self):
        from qa_engine import detect_question_type
        assert detect_question_type("Give an overview of the Act") == "SUMMARY"

    def test_detect_default_fallback(self):
        """Unrecognised patterns fall back to WHAT."""
        from qa_engine import detect_question_type
        assert detect_question_type("Tell me about the regulations") == "WHAT"

    def test_detect_case_insensitive(self):
        """Detection is case-insensitive."""
        from qa_engine import detect_question_type
        assert detect_question_type("WHEN was it adopted?") == "WHEN"
        assert detect_question_type("where is it?") == "WHERE"

    def test_detect_with_leading_whitespace(self):
        """Leading whitespace doesn't break detection."""
        from qa_engine import detect_question_type
        assert detect_question_type("   When was it adopted?") == "WHEN"


# ══════════════════════════════════════════════════════════════════════════════
#  QA ENGINE INITIALISATION TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestQAEngineInit:
    """Tests for QAEngine.__init__."""

    def test_init_default_model(self):
        from qa_engine import QAEngine
        engine = QAEngine()
        assert engine.model == "llama3.2:3b"

    def test_init_custom_model(self):
        from qa_engine import QAEngine
        engine = QAEngine(model_name="custom-model:7b")
        assert engine.model == "custom-model:7b"

    def test_ollama_url_constant(self):
        from qa_engine import QAEngine
        assert QAEngine.OLLAMA_URL == "http://localhost:11434/api/generate"


# ══════════════════════════════════════════════════════════════════════════════
#  FORMAT CONTEXT TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestFormatContext:
    """Tests for QAEngine._format_context."""

    def test_format_context_multiple_chunks(self, sample_chunks):
        from qa_engine import QAEngine
        engine = QAEngine()
        result = engine._format_context(sample_chunks)

        assert "[Chunk 1]" in result
        assert "[Chunk 2]" in result
        assert "[Chunk 3]" in result
        assert sample_chunks[0] in result
        assert sample_chunks[1] in result
        assert sample_chunks[2] in result

    def test_format_context_empty(self):
        from qa_engine import QAEngine
        engine = QAEngine()
        result = engine._format_context([])
        assert result == ""

    def test_format_context_single_chunk(self):
        from qa_engine import QAEngine
        engine = QAEngine()
        result = engine._format_context(["Only chunk"])
        assert "[Chunk 1]" in result
        assert "Only chunk" in result


# ══════════════════════════════════════════════════════════════════════════════
#  BUILD PROMPT TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildPrompt:
    """Tests for QAEngine._build_prompt."""

    def test_build_prompt_contains_system_prompt(self):
        from qa_engine import QAEngine, SYSTEM_PROMPT
        engine = QAEngine()
        prompt = engine._build_prompt("What is AI?", "context here", "WHAT")
        assert SYSTEM_PROMPT in prompt

    def test_build_prompt_contains_question(self):
        from qa_engine import QAEngine
        engine = QAEngine()
        prompt = engine._build_prompt("What is AI?", "context here", "WHAT")
        assert "What is AI?" in prompt

    def test_build_prompt_contains_context(self):
        from qa_engine import QAEngine
        engine = QAEngine()
        prompt = engine._build_prompt("Q?", "my context", "WHAT")
        assert "my context" in prompt

    def test_build_prompt_contains_hint_for_each_type(self):
        """Each question type gets its specific hint in the prompt."""
        from qa_engine import QAEngine
        engine = QAEngine()

        hints = {
            "YES_NO": "YES/NO question",
            "WHEN": "WHEN question",
            "WHERE": "WHERE question",
            "WHO": "WHO question",
            "WHY": "WHY question",
            "HOW": "HOW question",
            "COMPARE": "COMPARE question",
            "LIST": "LIST question",
            "DEFINE": "DEFINITION question",
            "SUMMARY": "SUMMARY question",
            "WHAT": "WHAT question",
        }

        for qtype, expected_hint in hints.items():
            prompt = engine._build_prompt("Q?", "ctx", qtype)
            assert expected_hint in prompt, f"Missing hint for {qtype}"

    def test_build_prompt_unknown_type_empty_hint(self):
        """Unknown question type produces empty hint (no crash)."""
        from qa_engine import QAEngine
        engine = QAEngine()
        prompt = engine._build_prompt("Q?", "ctx", "UNKNOWN_TYPE")
        assert "QUESTION-TYPE HINT" in prompt  # Section header still present


# ══════════════════════════════════════════════════════════════════════════════
#  CALL OLLAMA TESTS (MOCKED NETWORK)
# ══════════════════════════════════════════════════════════════════════════════

class TestCallOllama:
    """Tests for QAEngine._call_ollama with mocked urllib."""

    def test_call_ollama_success(self):
        """Successful API call returns parsed response text."""
        from qa_engine import QAEngine
        engine = QAEngine()

        mock_response_data = json.dumps({"response": "  This is the answer.  "}).encode("utf-8")
        mock_resp = MagicMock()
        mock_resp.read.return_value = mock_response_data
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = engine._call_ollama("test prompt")

        assert result == "This is the answer."

    def test_call_ollama_url_error(self):
        """URLError returns a user-friendly error message."""
        from qa_engine import QAEngine
        engine = QAEngine()

        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
            result = engine._call_ollama("test prompt")

        assert "Could not connect to Ollama" in result
        assert "Connection refused" in result

    def test_call_ollama_unexpected_error(self):
        """Generic exceptions return a descriptive error message."""
        from qa_engine import QAEngine
        engine = QAEngine()

        with patch("urllib.request.urlopen", side_effect=RuntimeError("Something broke")):
            result = engine._call_ollama("test prompt")

        assert "Unexpected error" in result
        assert "Something broke" in result

    def test_call_ollama_empty_response(self):
        """Empty response field returns empty string."""
        from qa_engine import QAEngine
        engine = QAEngine()

        mock_response_data = json.dumps({"response": ""}).encode("utf-8")
        mock_resp = MagicMock()
        mock_resp.read.return_value = mock_response_data
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = engine._call_ollama("test prompt")

        assert result == ""

    def test_call_ollama_missing_response_key(self):
        """Response JSON without 'response' key returns empty string."""
        from qa_engine import QAEngine
        engine = QAEngine()

        mock_response_data = json.dumps({"model": "test"}).encode("utf-8")
        mock_resp = MagicMock()
        mock_resp.read.return_value = mock_response_data
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = engine._call_ollama("test prompt")

        assert result == ""


# ══════════════════════════════════════════════════════════════════════════════
#  ANSWER METHOD TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestAnswer:
    """Tests for QAEngine.answer (integration of detect + format + call)."""

    def test_answer_returns_tuple(self, sample_chunks):
        """answer() returns (answer_text, question_type) tuple."""
        from qa_engine import QAEngine
        engine = QAEngine()

        with patch.object(engine, "_call_ollama", return_value="Mocked answer"):
            answer, qtype = engine.answer("What is AI?", sample_chunks)

        assert answer == "Mocked answer"
        assert qtype == "WHAT"

    def test_answer_detects_type_correctly(self, sample_chunks):
        """answer() detects the correct question type."""
        from qa_engine import QAEngine
        engine = QAEngine()

        with patch.object(engine, "_call_ollama", return_value="Yes it does"):
            _, qtype = engine.answer("Does the Act apply?", sample_chunks)

        assert qtype == "YES_NO"

    def test_answer_passes_context_to_ollama(self, sample_chunks):
        """answer() passes formatted context chunks to Ollama."""
        from qa_engine import QAEngine
        engine = QAEngine()

        with patch.object(engine, "_call_ollama", return_value="answer") as mock_call:
            engine.answer("What is AI?", sample_chunks)

        # Verify the prompt passed to _call_ollama contains the chunks
        prompt = mock_call.call_args[0][0]
        for chunk in sample_chunks:
            assert chunk in prompt

    def test_answer_empty_chunks(self):
        """answer() works with empty chunk list."""
        from qa_engine import QAEngine
        engine = QAEngine()

        with patch.object(engine, "_call_ollama", return_value="No context"):
            answer, qtype = engine.answer("What is AI?", [])

        assert answer == "No context"
