"""
test_evaluator.py
─────────────────
Unit and functional tests for the Evaluator module.
Tests cover: normalisation, tokenisation, exact match, token F1,
ROUGE scores (library + manual fallback), semantic similarity
(model + Jaccard fallback), cosine similarity, LCS, verdicts,
and full evaluate() integration.
"""

import math
import pytest
from unittest.mock import patch, MagicMock
from collections import Counter


# ══════════════════════════════════════════════════════════════════════════════
#  NORMALISATION & TOKENISATION TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestNormalisation:
    """Tests for Evaluator._normalise and _tokenise."""

    def test_normalise_lowercase(self):
        from evaluator import Evaluator
        assert "hello world" == Evaluator._normalise("Hello World")

    def test_normalise_removes_punctuation(self):
        from evaluator import Evaluator
        result = Evaluator._normalise("Hello, World! It's great.")
        assert "," not in result
        assert "!" not in result
        assert "'" not in result

    def test_normalise_collapses_whitespace(self):
        from evaluator import Evaluator
        result = Evaluator._normalise("hello   world   foo")
        assert result == "hello world foo"

    def test_normalise_strips(self):
        from evaluator import Evaluator
        result = Evaluator._normalise("  hello  ")
        assert result == "hello"

    def test_normalise_preserves_numbers(self):
        from evaluator import Evaluator
        result = Evaluator._normalise("Article 99 section 5")
        assert "99" in result
        assert "5" in result

    def test_normalise_empty_string(self):
        from evaluator import Evaluator
        assert Evaluator._normalise("") == ""

    def test_normalise_only_punctuation(self):
        from evaluator import Evaluator
        result = Evaluator._normalise("!@#$%^&*()")
        assert result == ""

    def test_tokenise(self):
        from evaluator import Evaluator
        tokens = Evaluator._tokenise("hello world foo")
        assert tokens == ["hello", "world", "foo"]

    def test_tokenise_empty(self):
        from evaluator import Evaluator
        assert Evaluator._tokenise("") == []  # split on empty returns []


# ══════════════════════════════════════════════════════════════════════════════
#  EXACT MATCH TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestExactMatch:
    """Tests for exact match logic in evaluate()."""

    def test_exact_match_true(self, evaluator_instance):
        """Identical normalised strings → exact match = True."""
        with patch.object(evaluator_instance, "_semantic_sim", return_value=1.0):
            result = evaluator_instance.evaluate("Hello World!", "hello world")
        assert result.exact_match is True

    def test_exact_match_false(self, evaluator_instance):
        """Different strings → exact match = False."""
        with patch.object(evaluator_instance, "_semantic_sim", return_value=0.5):
            result = evaluator_instance.evaluate("Hello World", "Goodbye Moon")
        assert result.exact_match is False

    def test_exact_match_ignores_punctuation(self, evaluator_instance):
        """Punctuation is stripped before comparison."""
        with patch.object(evaluator_instance, "_semantic_sim", return_value=1.0):
            result = evaluator_instance.evaluate("Hello, World!", "Hello World")
        assert result.exact_match is True


# ══════════════════════════════════════════════════════════════════════════════
#  TOKEN F1 TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestTokenF1:
    """Tests for Evaluator._token_f1."""

    def test_token_f1_perfect_overlap(self, evaluator_instance):
        """Identical tokens → precision = recall = F1 = 1.0."""
        p, r, f1 = evaluator_instance._token_f1("hello world", "hello world")
        assert p == 1.0
        assert r == 1.0
        assert f1 == 1.0

    def test_token_f1_no_overlap(self, evaluator_instance):
        """Zero overlap → all metrics 0.0."""
        p, r, f1 = evaluator_instance._token_f1("hello world", "foo bar")
        assert p == 0.0
        assert r == 0.0
        assert f1 == 0.0

    def test_token_f1_partial_overlap(self, evaluator_instance):
        """Partial overlap gives correct precision, recall, F1."""
        # pred: "hello world foo" → 3 tokens
        # gt:   "hello world bar" → 3 tokens
        # common: "hello", "world" → 2
        p, r, f1 = evaluator_instance._token_f1("hello world foo", "hello world bar")
        assert p == pytest.approx(2 / 3, rel=1e-3)
        assert r == pytest.approx(2 / 3, rel=1e-3)
        assert f1 == pytest.approx(2 / 3, rel=1e-3)

    def test_token_f1_different_lengths(self, evaluator_instance):
        """Different token counts give different precision and recall."""
        # pred: "a b c d" → 4 tokens, gt: "a b" → 2 tokens
        # common: 2
        p, r, f1 = evaluator_instance._token_f1("a b c d", "a b")
        assert p == pytest.approx(2 / 4, rel=1e-3)  # 0.5
        assert r == pytest.approx(2 / 2, rel=1e-3)  # 1.0

    def test_token_f1_empty_prediction(self, evaluator_instance):
        """Empty prediction → all 0.0."""
        p, r, f1 = evaluator_instance._token_f1("", "hello world")
        # Note: empty string splits to [''], which has no overlap with real words
        assert f1 == 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  ROUGE TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestRouge:
    """Tests for Evaluator._rouge, _manual_rouge_n, _manual_rouge_l."""

    def test_rouge_with_library(self, evaluator_instance):
        """ROUGE via rouge-score library returns valid scores."""
        mock_score = MagicMock()
        mock_score.fmeasure = 0.85

        mock_scorer_instance = MagicMock()
        mock_scorer_instance.score.return_value = {
            "rouge1": mock_score,
            "rouge2": mock_score,
            "rougeL": mock_score,
        }

        # Create a mock module structure that mimics rouge_score.rouge_scorer
        mock_rouge_scorer_module = MagicMock()
        mock_rouge_scorer_module.RougeScorer.return_value = mock_scorer_instance

        mock_rouge_score_module = MagicMock()
        mock_rouge_score_module.rouge_scorer = mock_rouge_scorer_module

        with patch.dict("sys.modules", {
            "rouge_score": mock_rouge_score_module,
            "rouge_score.rouge_scorer": mock_rouge_scorer_module,
        }):
            r1, r2, rl = evaluator_instance._rouge("pred text", "gt text")

        assert r1 == 0.85
        assert r2 == 0.85
        assert rl == 0.85

    def test_rouge_fallback_manual(self, evaluator_instance):
        """When rouge_score import fails, uses manual fallback."""
        with patch.object(evaluator_instance, "_manual_rouge_n", return_value=0.7) as mock_rn, \
             patch.object(evaluator_instance, "_manual_rouge_l", return_value=0.6) as mock_rl:
            # Simulate ImportError by patching the import inside _rouge
            with patch.dict("sys.modules", {"rouge_score": None, "rouge_score.rouge_scorer": None}):
                # Force ImportError
                original_rouge = evaluator_instance._rouge.__func__

                def rouge_with_import_error(self, pred, gt):
                    try:
                        raise ImportError("no rouge_score")
                    except ImportError:
                        r1 = self._manual_rouge_n(pred, gt, n=1)
                        r2 = self._manual_rouge_n(pred, gt, n=2)
                        rl = self._manual_rouge_l(pred, gt)
                        return r1, r2, rl

                with patch.object(type(evaluator_instance), "_rouge", rouge_with_import_error):
                    r1, r2, rl = evaluator_instance._rouge("pred", "gt")

                assert r1 == 0.7
                assert r2 == 0.7
                assert rl == 0.6

    def test_manual_rouge_n_unigram(self, evaluator_instance):
        """Manual ROUGE-1 with known overlap."""
        # pred: "the cat sat", gt: "the cat ran"
        # unigrams: pred={the, cat, sat}, gt={the, cat, ran}
        # common: {the, cat} → 2
        r1 = evaluator_instance._manual_rouge_n("the cat sat", "the cat ran", n=1)
        assert r1 > 0
        assert r1 <= 1.0

    def test_manual_rouge_n_bigram(self, evaluator_instance):
        """Manual ROUGE-2 with known overlap."""
        r2 = evaluator_instance._manual_rouge_n("the cat sat on mat", "the cat sat on rug", n=2)
        assert r2 > 0
        assert r2 <= 1.0

    def test_manual_rouge_n_no_overlap(self, evaluator_instance):
        """No overlap → 0.0."""
        r = evaluator_instance._manual_rouge_n("hello world", "foo bar", n=1)
        assert r == 0.0

    def test_manual_rouge_l(self, evaluator_instance):
        """Manual ROUGE-L with known LCS."""
        rl = evaluator_instance._manual_rouge_l("the cat sat on the mat", "the cat on the mat")
        assert rl > 0
        assert rl <= 1.0

    def test_manual_rouge_l_identical(self, evaluator_instance):
        """Identical strings → ROUGE-L = 1.0."""
        rl = evaluator_instance._manual_rouge_l("hello world", "hello world")
        assert rl == 1.0

    def test_manual_rouge_l_no_overlap(self, evaluator_instance):
        """No overlap → ROUGE-L = 0.0."""
        rl = evaluator_instance._manual_rouge_l("hello world", "foo bar")
        assert rl == 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  NGRAMS TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestNgrams:
    """Tests for Evaluator._ngrams."""

    def test_ngrams_unigrams(self, evaluator_instance):
        tokens = ["a", "b", "c"]
        result = evaluator_instance._ngrams(tokens, 1)
        assert result == Counter({("a",): 1, ("b",): 1, ("c",): 1})

    def test_ngrams_bigrams(self, evaluator_instance):
        tokens = ["a", "b", "c"]
        result = evaluator_instance._ngrams(tokens, 2)
        assert result == Counter({("a", "b"): 1, ("b", "c"): 1})

    def test_ngrams_empty(self, evaluator_instance):
        result = evaluator_instance._ngrams([], 1)
        assert result == Counter()

    def test_ngrams_n_greater_than_length(self, evaluator_instance):
        result = evaluator_instance._ngrams(["a"], 2)
        assert result == Counter()


# ══════════════════════════════════════════════════════════════════════════════
#  LCS TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestLCS:
    """Tests for Evaluator._lcs_length."""

    def test_lcs_identical(self):
        from evaluator import Evaluator
        assert Evaluator._lcs_length(["a", "b", "c"], ["a", "b", "c"]) == 3

    def test_lcs_partial(self):
        from evaluator import Evaluator
        assert Evaluator._lcs_length(["a", "b", "c"], ["a", "c"]) == 2

    def test_lcs_no_common(self):
        from evaluator import Evaluator
        assert Evaluator._lcs_length(["a", "b"], ["c", "d"]) == 0

    def test_lcs_empty_first(self):
        from evaluator import Evaluator
        assert Evaluator._lcs_length([], ["a", "b"]) == 0

    def test_lcs_empty_second(self):
        from evaluator import Evaluator
        assert Evaluator._lcs_length(["a", "b"], []) == 0

    def test_lcs_both_empty(self):
        from evaluator import Evaluator
        assert Evaluator._lcs_length([], []) == 0

    def test_lcs_single_element(self):
        from evaluator import Evaluator
        assert Evaluator._lcs_length(["a"], ["a"]) == 1
        assert Evaluator._lcs_length(["a"], ["b"]) == 0


# ══════════════════════════════════════════════════════════════════════════════
#  COSINE SIMILARITY TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestCosine:
    """Tests for Evaluator._cosine static method."""

    def test_cosine_identical_vectors(self):
        from evaluator import Evaluator
        result = Evaluator._cosine([1, 2, 3], [1, 2, 3])
        assert result == pytest.approx(1.0, rel=1e-6)

    def test_cosine_orthogonal_vectors(self):
        from evaluator import Evaluator
        result = Evaluator._cosine([1, 0], [0, 1])
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_cosine_opposite_vectors(self):
        from evaluator import Evaluator
        result = Evaluator._cosine([1, 0], [-1, 0])
        assert result == pytest.approx(-1.0, rel=1e-6)

    def test_cosine_zero_vector_a(self):
        from evaluator import Evaluator
        assert Evaluator._cosine([0, 0, 0], [1, 2, 3]) == 0.0

    def test_cosine_zero_vector_b(self):
        from evaluator import Evaluator
        assert Evaluator._cosine([1, 2, 3], [0, 0, 0]) == 0.0

    def test_cosine_both_zero(self):
        from evaluator import Evaluator
        assert Evaluator._cosine([0, 0], [0, 0]) == 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  SEMANTIC SIMILARITY TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestSemanticSimilarity:
    """Tests for Evaluator._semantic_sim."""

    def test_semantic_sim_with_model(self, evaluator_instance):
        """When sentence-transformer is available, uses cosine of embeddings."""
        mock_model = MagicMock()
        mock_model.encode.return_value = [[1, 0, 0], [1, 0, 0]]  # identical

        with patch.object(type(evaluator_instance), "_get_st_model", return_value=mock_model):
            result = evaluator_instance._semantic_sim("text a", "text b")

        assert result == pytest.approx(1.0, rel=1e-3)

    def test_semantic_sim_fallback_jaccard(self, evaluator_instance):
        """When model fails, falls back to Jaccard similarity."""
        with patch.object(type(evaluator_instance), "_get_st_model", side_effect=Exception("no model")):
            result = evaluator_instance._semantic_sim("hello world foo", "hello world bar")

        # Jaccard: {hello, world, foo} ∩ {hello, world, bar} = {hello, world} = 2
        # Union = {hello, world, foo, bar} = 4
        # Jaccard = 2/4 = 0.5
        assert result == pytest.approx(0.5, rel=1e-3)

    def test_semantic_sim_fallback_both_empty(self, evaluator_instance):
        """Both empty strings → Jaccard returns 1.0."""
        with patch.object(type(evaluator_instance), "_get_st_model", side_effect=Exception("no model")):
            result = evaluator_instance._semantic_sim("", "")
        assert result == 1.0

    def test_semantic_sim_fallback_no_overlap(self, evaluator_instance):
        """No overlap → Jaccard = 0.0."""
        with patch.object(type(evaluator_instance), "_get_st_model", side_effect=Exception("no model")):
            result = evaluator_instance._semantic_sim("hello world", "foo bar")
        assert result == 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  GET ST MODEL TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestGetSTModel:
    """Tests for Evaluator._get_st_model class method."""

    def test_get_st_model_caches(self):
        """Model is loaded once and cached at class level."""
        from evaluator import Evaluator

        # Reset cache
        original = Evaluator._st_model
        Evaluator._st_model = None

        mock_model = MagicMock()
        with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
            model1 = Evaluator._get_st_model()
            model2 = Evaluator._get_st_model()

        assert model1 is model2
        assert model1 is mock_model

        # Restore
        Evaluator._st_model = original


# ══════════════════════════════════════════════════════════════════════════════
#  VERDICT TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestVerdict:
    """Tests for Evaluator._verdict."""

    def test_verdict_excellent(self):
        from evaluator import Evaluator
        assert "Excellent" in Evaluator._verdict(0.90)
        assert "Excellent" in Evaluator._verdict(0.85)

    def test_verdict_good(self):
        from evaluator import Evaluator
        assert "Good" in Evaluator._verdict(0.75)
        assert "Good" in Evaluator._verdict(0.70)

    def test_verdict_partial(self):
        from evaluator import Evaluator
        assert "Partial" in Evaluator._verdict(0.60)
        assert "Partial" in Evaluator._verdict(0.50)

    def test_verdict_weak(self):
        from evaluator import Evaluator
        assert "Weak" in Evaluator._verdict(0.40)
        assert "Weak" in Evaluator._verdict(0.30)

    def test_verdict_poor(self):
        from evaluator import Evaluator
        assert "Poor" in Evaluator._verdict(0.20)
        assert "Poor" in Evaluator._verdict(0.0)

    def test_verdict_boundary_85(self):
        from evaluator import Evaluator
        assert "Excellent" in Evaluator._verdict(0.85)
        assert "Good" in Evaluator._verdict(0.84)

    def test_verdict_boundary_70(self):
        from evaluator import Evaluator
        assert "Good" in Evaluator._verdict(0.70)
        assert "Partial" in Evaluator._verdict(0.69)

    def test_verdict_boundary_50(self):
        from evaluator import Evaluator
        assert "Partial" in Evaluator._verdict(0.50)
        assert "Weak" in Evaluator._verdict(0.49)

    def test_verdict_boundary_30(self):
        from evaluator import Evaluator
        assert "Weak" in Evaluator._verdict(0.30)
        assert "Poor" in Evaluator._verdict(0.29)


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATE INTEGRATION TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestEvaluateIntegration:
    """Integration tests for the full evaluate() method."""

    def test_evaluate_returns_eval_result(self, evaluator_instance):
        """evaluate() returns an EvalResult dataclass."""
        from evaluator import EvalResult
        with patch.object(evaluator_instance, "_semantic_sim", return_value=0.8):
            result = evaluator_instance.evaluate("test answer", "test ground truth")
        assert isinstance(result, EvalResult)

    def test_evaluate_all_fields_populated(self, evaluator_instance):
        """All EvalResult fields are populated after evaluate()."""
        with patch.object(evaluator_instance, "_semantic_sim", return_value=0.75):
            result = evaluator_instance.evaluate("The AI Act is a law", "The AI Act is a regulation")

        assert isinstance(result.exact_match, bool)
        assert 0.0 <= result.token_precision <= 1.0
        assert 0.0 <= result.token_recall <= 1.0
        assert 0.0 <= result.token_f1 <= 1.0
        assert 0.0 <= result.rouge1 <= 1.0
        assert 0.0 <= result.rouge2 <= 1.0
        assert 0.0 <= result.rougeL <= 1.0
        assert 0.0 <= result.semantic_similarity <= 1.0
        assert 0.0 <= result.overall_score <= 1.0
        assert isinstance(result.verdict, str)
        assert isinstance(result.details, dict)

    def test_evaluate_overall_score_formula(self, evaluator_instance):
        """Verify the weighted formula: 0.35*sem + 0.25*f1 + 0.20*rl + 0.15*r1 + 0.05*em."""
        with patch.object(evaluator_instance, "_semantic_sim", return_value=0.9):
            result = evaluator_instance.evaluate("hello world", "hello world")

        # Since pred == gt, exact_match should be True (em_bonus=1.0)
        # token_f1 = 1.0, rouge1 should be ~1.0, rougeL should be ~1.0
        expected = round(
            0.35 * result.semantic_similarity
            + 0.25 * result.token_f1
            + 0.20 * result.rougeL
            + 0.15 * result.rouge1
            + 0.05 * (1.0 if result.exact_match else 0.0),
            4,
        )
        assert result.overall_score == pytest.approx(expected, rel=1e-3)

    def test_evaluate_identical_texts(self, evaluator_instance):
        """Identical texts → high overall score."""
        with patch.object(evaluator_instance, "_semantic_sim", return_value=1.0):
            result = evaluator_instance.evaluate(
                "The EU AI Act regulates artificial intelligence",
                "The EU AI Act regulates artificial intelligence",
            )
        assert result.exact_match is True
        assert result.token_f1 == 1.0
        assert result.overall_score >= 0.90

    def test_evaluate_completely_different_texts(self, evaluator_instance):
        """Completely different texts → low overall score."""
        with patch.object(evaluator_instance, "_semantic_sim", return_value=0.1):
            result = evaluator_instance.evaluate(
                "The weather is sunny today",
                "Quantum mechanics describes particle behaviour",
            )
        assert result.exact_match is False
        assert result.overall_score < 0.30

    def test_evaluate_details_contain_normalised_texts(self, evaluator_instance):
        """Details dict includes truncated normalised texts."""
        with patch.object(evaluator_instance, "_semantic_sim", return_value=0.5):
            result = evaluator_instance.evaluate("Hello, World!", "Goodbye, Moon!")
        assert "normalised_prediction" in result.details
        assert "normalised_ground_truth" in result.details
        assert result.details["normalised_prediction"] == "hello world"
        assert result.details["normalised_ground_truth"] == "goodbye moon"


# ══════════════════════════════════════════════════════════════════════════════
#  EVAL RESULT DATACLASS TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestEvalResult:
    """Tests for EvalResult dataclass defaults."""

    def test_eval_result_defaults(self):
        from evaluator import EvalResult
        result = EvalResult()
        assert result.exact_match is False
        assert result.token_precision == 0.0
        assert result.token_recall == 0.0
        assert result.token_f1 == 0.0
        assert result.rouge1 == 0.0
        assert result.rouge2 == 0.0
        assert result.rougeL == 0.0
        assert result.semantic_similarity == 0.0
        assert result.overall_score == 0.0
        assert result.verdict == ""
        assert result.details == {}
