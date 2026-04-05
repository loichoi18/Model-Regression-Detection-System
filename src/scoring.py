"""
Multi-dimensional scoring for eval results.

Scores each test case on multiple axes:
  1. Category match (binary)
  2. Summary relevance (1–5 scale, via LLM-as-judge or text similarity fallback)
  3. Latency (pass-through from provider)
  4. Token usage (pass-through from provider)

Design decision: the summary scorer uses a Protocol so you can swap between
a fast text-similarity mock (for CI) and a real LLM-as-judge (for deep evals).
"""

from __future__ import annotations

import re
from typing import Protocol

from src.models import CaseResult, EmailCategory, EvalRun


# ---------------------------------------------------------------------------
# Summary relevance scoring
# ---------------------------------------------------------------------------

class SummaryScorer(Protocol):
    """Interface for summary relevance scoring."""

    def score(self, expected: str, predicted: str) -> float:
        """Return a relevance score from 1.0 (irrelevant) to 5.0 (perfect)."""
        ...


class TextSimilarityScorer:
    """
    Fast, deterministic summary scorer using token-level Jaccard + key-phrase overlap.

    Used in CI pipelines where speed matters and LLM calls are expensive.
    Produces scores on a 1–5 scale that correlate reasonably well with
    human judgments for short classification summaries.
    """

    def score(self, expected: str, predicted: str) -> float:
        expected_tokens = self._normalize(expected)
        predicted_tokens = self._normalize(predicted)

        if not expected_tokens or not predicted_tokens:
            return 1.0

        # Jaccard similarity on word tokens
        intersection = expected_tokens & predicted_tokens
        union = expected_tokens | predicted_tokens
        jaccard = len(intersection) / len(union) if union else 0.0

        # Bonus for preserving key noun phrases (crude but effective)
        expected_bigrams = self._bigrams(expected)
        predicted_bigrams = self._bigrams(predicted)
        bigram_overlap = 0.0
        if expected_bigrams:
            bigram_overlap = len(expected_bigrams & predicted_bigrams) / len(
                expected_bigrams
            )

        # Weighted combination → scale to 1–5
        raw = 0.6 * jaccard + 0.4 * bigram_overlap
        return round(1.0 + raw * 4.0, 2)  # Maps [0,1] → [1,5]

    @staticmethod
    def _normalize(text: str) -> set[str]:
        """Lowercase, strip punctuation, split into token set."""
        text = re.sub(r"[^\w\s]", "", text.lower())
        # Remove common stop words that add noise
        stop = {
            "a", "an", "the", "is", "are", "was", "were", "and", "or", "but",
            "in", "on", "at", "to", "for", "of", "with", "by", "from", "that",
            "this", "it", "as", "be", "has", "had", "have", "do", "does",
            "their", "them", "they", "its", "about",
        }
        return {w for w in text.split() if w and w not in stop}

    @staticmethod
    def _bigrams(text: str) -> set[tuple[str, str]]:
        """Extract consecutive word pairs after normalization."""
        words = re.sub(r"[^\w\s]", "", text.lower()).split()
        return {(words[i], words[i + 1]) for i in range(len(words) - 1)}


# ---------------------------------------------------------------------------
# Aggregate scoring
# ---------------------------------------------------------------------------

def score_case(
    case_result: CaseResult,
    scorer: SummaryScorer | None = None,
) -> CaseResult:
    """Apply scoring to a single case result. Mutates and returns the result."""
    scorer = scorer or TextSimilarityScorer()

    # 1. Category match
    case_result.category_match = (
        case_result.predicted_category == case_result.expected_category
    )

    # 2. Summary relevance
    if case_result.predicted_summary and case_result.expected_summary:
        case_result.summary_relevance_score = scorer.score(
            case_result.expected_summary,
            case_result.predicted_summary,
        )
    else:
        case_result.summary_relevance_score = 1.0

    return case_result


def compute_run_metrics(run: EvalRun) -> EvalRun:
    """Compute aggregate metrics for a completed eval run. Mutates and returns."""
    if not run.results:
        return run

    successful = [r for r in run.results if r.error is None]
    if not successful:
        return run

    run.overall_accuracy = sum(
        1 for r in successful if r.category_match
    ) / len(successful)

    run.avg_summary_relevance = sum(
        r.summary_relevance_score for r in successful
    ) / len(successful)

    run.avg_latency_ms = sum(r.latency_ms for r in successful) / len(successful)

    run.total_tokens = sum(r.tokens_used for r in successful)

    return run


def accuracy_by_category(run: EvalRun) -> dict[EmailCategory, dict[str, float]]:
    """Break down accuracy by category. Returns {category: {correct, total, rate}}."""
    buckets: dict[EmailCategory, dict[str, float]] = {}

    for result in run.results:
        if result.error:
            continue

        cat = result.expected_category
        if cat not in buckets:
            buckets[cat] = {"correct": 0, "total": 0, "rate": 0.0}

        buckets[cat]["total"] += 1
        if result.category_match:
            buckets[cat]["correct"] += 1

    for cat, data in buckets.items():
        data["rate"] = data["correct"] / data["total"] if data["total"] > 0 else 0.0

    return buckets
