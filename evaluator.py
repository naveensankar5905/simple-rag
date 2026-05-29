"""
evaluator.py
────────────
Computes a suite of NLP evaluation metrics comparing a model-generated answer
against a human-provided ground-truth answer.

Metrics implemented:
  • Exact Match (EM)
  • Token-level F1 (precision / recall / F1)
  • ROUGE-1, ROUGE-2, ROUGE-L   (via rouge-score)
  • Semantic Similarity          (cosine of sentence-transformer embeddings)
  • Overall Score                (weighted composite of the above)
"""

from __future__ import annotations
import re
import math
from collections import Counter
from dataclasses import dataclass, field


# ══════════════════════════════════════════════════════════════════════════════
#  DATA CLASS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EvalResult:
    exact_match: bool = False
    token_precision: float = 0.0
    token_recall: float = 0.0
    token_f1: float = 0.0
    rouge1: float = 0.0
    rouge2: float = 0.0
    rougeL: float = 0.0
    semantic_similarity: float = 0.0
    overall_score: float = 0.0
    verdict: str = ""
    details: dict = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATOR
# ══════════════════════════════════════════════════════════════════════════════

class Evaluator:
    """
    Compare a predicted answer against a ground-truth answer using multiple metrics.
    Sentence-transformer model is loaded lazily and cached.
    """

    _st_model = None   # class-level cache

    # ── Public API ─────────────────────────────────────────────────────────────

    def evaluate(self, prediction: str, ground_truth: str) -> EvalResult:
        pred_norm = self._normalise(prediction)
        gt_norm   = self._normalise(ground_truth)

        result = EvalResult()

        # 1. Exact Match
        result.exact_match = (pred_norm == gt_norm)

        # 2. Token F1
        result.token_precision, result.token_recall, result.token_f1 = \
            self._token_f1(pred_norm, gt_norm)

        # 3. ROUGE
        r1, r2, rl = self._rouge(pred_norm, gt_norm)
        result.rouge1 = r1
        result.rouge2 = r2
        result.rougeL = rl

        # 4. Semantic similarity
        result.semantic_similarity = self._semantic_sim(prediction, ground_truth)

        # 5. Overall weighted score
        # Weights: semantic (35%) + token-F1 (25%) + ROUGE-L (20%) + ROUGE-1 (15%) + EM bonus (5%)
        em_bonus = 1.0 if result.exact_match else 0.0
        result.overall_score = round(
            0.35 * result.semantic_similarity
            + 0.25 * result.token_f1
            + 0.20 * result.rougeL
            + 0.15 * result.rouge1
            + 0.05 * em_bonus,
            4,
        )

        # 6. Human-readable verdict
        result.verdict = self._verdict(result.overall_score)

        result.details = {
            "normalised_prediction": pred_norm[:200],
            "normalised_ground_truth": gt_norm[:200],
        }
        return result

    # ── Text helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _normalise(text: str) -> str:
        """Lower-case, remove punctuation, collapse whitespace."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        return text.split()

    # ── Token F1 ───────────────────────────────────────────────────────────────

    def _token_f1(self, pred: str, gt: str) -> tuple[float, float, float]:
        pred_tokens = Counter(self._tokenise(pred))
        gt_tokens   = Counter(self._tokenise(gt))

        common = sum((pred_tokens & gt_tokens).values())
        if common == 0:
            return 0.0, 0.0, 0.0

        precision = common / max(sum(pred_tokens.values()), 1)
        recall    = common / max(sum(gt_tokens.values()), 1)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return round(precision, 4), round(recall, 4), round(f1, 4)

    # ── ROUGE ──────────────────────────────────────────────────────────────────

    def _rouge(self, pred: str, gt: str) -> tuple[float, float, float]:
        """ROUGE-1, ROUGE-2, ROUGE-L (recall-oriented, then averaged with precision)."""
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
            scores = scorer.score(gt, pred)
            return (
                round(scores["rouge1"].fmeasure, 4),
                round(scores["rouge2"].fmeasure, 4),
                round(scores["rougeL"].fmeasure, 4),
            )
        except ImportError:
            # Manual ROUGE-1 / ROUGE-L fallback (no rouge_score installed)
            r1 = self._manual_rouge_n(pred, gt, n=1)
            r2 = self._manual_rouge_n(pred, gt, n=2)
            rl = self._manual_rouge_l(pred, gt)
            return r1, r2, rl

    def _ngrams(self, tokens: list[str], n: int) -> Counter:
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))

    def _manual_rouge_n(self, pred: str, gt: str, n: int) -> float:
        pred_ng = self._ngrams(self._tokenise(pred), n)
        gt_ng   = self._ngrams(self._tokenise(gt), n)
        common  = sum((pred_ng & gt_ng).values())
        denom_p = max(sum(pred_ng.values()), 1)
        denom_r = max(sum(gt_ng.values()), 1)
        p = common / denom_p
        r = common / denom_r
        return round(2*p*r/(p+r), 4) if (p+r) > 0 else 0.0

    def _manual_rouge_l(self, pred: str, gt: str) -> float:
        """LCS-based ROUGE-L."""
        a = self._tokenise(pred)
        b = self._tokenise(gt)
        lcs = self._lcs_length(a, b)
        p = lcs / max(len(a), 1)
        r = lcs / max(len(b), 1)
        return round(2*p*r/(p+r), 4) if (p+r) > 0 else 0.0

    @staticmethod
    def _lcs_length(a: list, b: list) -> int:
        """O(mn) dynamic-programming LCS length."""
        m, n = len(a), len(b)
        if m == 0 or n == 0:
            return 0
        # Space-optimised (two rows)
        prev = [0] * (n + 1)
        for i in range(m):
            curr = [0] * (n + 1)
            for j in range(n):
                curr[j+1] = prev[j] + 1 if a[i] == b[j] else max(curr[j], prev[j+1])
            prev = curr
        return prev[n]

    # ── Semantic similarity ────────────────────────────────────────────────────

    def _semantic_sim(self, pred: str, gt: str) -> float:
        """Cosine similarity of sentence-transformer embeddings."""
        try:
            model = self._get_st_model()
            embs = model.encode([pred, gt], convert_to_tensor=False)
            return round(float(self._cosine(embs[0], embs[1])), 4)
        except Exception:
            # Fallback: Jaccard on tokens
            a = set(self._tokenise(self._normalise(pred)))
            b = set(self._tokenise(self._normalise(gt)))
            if not a and not b:
                return 1.0
            return round(len(a & b) / max(len(a | b), 1), 4)

    @classmethod
    def _get_st_model(cls):
        if cls._st_model is None:
            from sentence_transformers import SentenceTransformer
            cls._st_model = SentenceTransformer("all-MiniLM-L6-v2")
        return cls._st_model

    @staticmethod
    def _cosine(a, b) -> float:
        dot   = sum(x*y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x*x for x in a))
        mag_b = math.sqrt(sum(x*x for x in b))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    # ── Verdict ────────────────────────────────────────────────────────────────

    @staticmethod
    def _verdict(score: float) -> str:
        if score >= 0.85:
            return "🟢 Excellent"
        if score >= 0.70:
            return "🟡 Good"
        if score >= 0.50:
            return "🟠 Partial"
        if score >= 0.30:
            return "🔴 Weak"
        return "⛔ Poor"
