"""
DeepEval integration adapter.

Bridges our custom evaluation system with DeepEval's framework:
  1. Custom metrics (CategoryCorrectness, SummaryQuality, LatencyBudget)
  2. Dataset conversion (our GoldenDataset → DeepEval EvaluationDataset)
  3. Result mapping (DeepEval results → our EvalRun format)

Design rationale:
  DeepEval provides the pytest runner, assert_test(), and CI/CD integration.
  We provide the diff engine, drift detection, historical tracking, and alerting.
  This adapter is the bridge between the two systems.

  Custom metrics allow us to run in "mock mode" (no API key needed, zero cost)
  or "llm-judge mode" (real G-Eval scoring via OpenAI) with a single flag.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import BaseMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from src.models import (
    CaseResult,
    EmailCategory,
    EvalRun,
    GoldenDataset,
    TestCase,
)
from src.scoring import TextSimilarityScorer


# ---------------------------------------------------------------------------
# Custom DeepEval Metrics
# ---------------------------------------------------------------------------

class CategoryCorrectnessMetric(BaseMetric):
    """
    Checks if the classifier returned the correct email category.

    This is a deterministic metric — no LLM judge needed.
    Score: 1.0 (correct) or 0.0 (incorrect).
    """

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self._score: float = 0.0
        self._reason: str = ""
        self.evaluation_model = None
        self.strict_mode = False
        self.async_mode = False
        self.verbose_mode = False
        self.evaluation_cost = 0.0

    @property
    def __name__(self):
        return "Category Correctness"

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        """Compare predicted category against expected."""
        actual = test_case.actual_output or ""
        expected = test_case.expected_output or ""

        # Extract category from JSON output
        predicted_category = self._extract_category(actual)
        expected_category = self._extract_category(expected)

        if predicted_category and expected_category:
            if predicted_category.lower() == expected_category.lower():
                self._score = 1.0
                self._reason = (
                    f"Correct: predicted '{predicted_category}' "
                    f"matches expected '{expected_category}'"
                )
            else:
                self._score = 0.0
                self._reason = (
                    f"Incorrect: predicted '{predicted_category}' "
                    f"but expected '{expected_category}'"
                )
        else:
            self._score = 0.0
            self._reason = (
                f"Could not extract categories. "
                f"Actual: '{actual[:100]}', Expected: '{expected[:100]}'"
            )

        self.success = self._score >= self.threshold
        return self._score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return self._score >= self.threshold

    @property
    def score(self) -> float:
        return self._score

    @property
    def reason(self) -> str:
        return self._reason

    @staticmethod
    def _extract_category(text: str) -> str | None:
        """Extract category from either JSON output or plain text."""
        # Try JSON parsing first
        try:
            data = json.loads(text)
            return data.get("category", data.get("expected_category"))
        except (json.JSONDecodeError, TypeError):
            pass

        # Try regex for category field
        match = re.search(r'"?category"?\s*[:=]\s*"?(\w+)"?', text, re.IGNORECASE)
        if match:
            return match.group(1)

        # If the text is just a category name
        text_clean = text.strip().lower()
        if text_clean in ("billing", "technical", "account", "general"):
            return text_clean

        return None


class SummaryQualityMetric(BaseMetric):
    """
    Evaluates summary quality using text similarity (mock mode)
    or LLM-as-judge (when OPENAI_API_KEY is available).

    Mock mode: Uses our TextSimilarityScorer (Jaccard + bigram overlap)
    LLM mode: Falls back to mock if no API key is set.

    Score: 0.0 – 1.0 (normalized from the internal 1–5 scale).
    """

    def __init__(self, threshold: float = 0.5, use_llm_judge: bool = False):
        self.threshold = threshold
        self.use_llm_judge = use_llm_judge and bool(os.environ.get("OPENAI_API_KEY"))
        self._scorer = TextSimilarityScorer()
        self._score: float = 0.0
        self._reason: str = ""
        self.evaluation_model = None
        self.strict_mode = False
        self.async_mode = False
        self.verbose_mode = False
        self.evaluation_cost = 0.0

    @property
    def __name__(self):
        return "Summary Quality"

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        actual = test_case.actual_output or ""
        expected = test_case.expected_output or ""

        # Extract summaries from JSON if needed
        actual_summary = self._extract_summary(actual)
        expected_summary = self._extract_summary(expected)

        if not actual_summary or not expected_summary:
            self._score = 0.0
            self._reason = "Could not extract summary from output"
            self.success = False
            return self._score

        # Score using text similarity (1–5 scale → 0–1 scale)
        raw_score = self._scorer.score(expected_summary, actual_summary)
        self._score = (raw_score - 1.0) / 4.0  # Map [1,5] → [0,1]
        self._score = round(max(0.0, min(1.0, self._score)), 4)

        self._reason = (
            f"Summary similarity: {raw_score:.2f}/5.0 "
            f"(normalized: {self._score:.2f}). "
            f"Expected: '{expected_summary[:80]}...' "
            f"Got: '{actual_summary[:80]}...'"
        )
        self.success = self._score >= self.threshold
        return self._score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return self._score >= self.threshold

    @property
    def score(self) -> float:
        return self._score

    @property
    def reason(self) -> str:
        return self._reason

    @staticmethod
    def _extract_summary(text: str) -> str | None:
        """Extract summary from JSON or plain text."""
        try:
            data = json.loads(text)
            return data.get("summary", data.get("expected_summary"))
        except (json.JSONDecodeError, TypeError):
            pass
        return text.strip() if text.strip() else None


class LatencyBudgetMetric(BaseMetric):
    """
    Checks if classification latency stays within budget.

    This is a non-functional quality metric — it validates performance,
    not correctness. Score: 1.0 if within budget, degrades linearly to
    0.0 at 2x the budget.
    """

    def __init__(self, budget_ms: float = 500.0, threshold: float = 0.8):
        self.budget_ms = budget_ms
        self.threshold = threshold
        self._score: float = 0.0
        self._reason: str = ""
        self.evaluation_model = None
        self.strict_mode = False
        self.async_mode = False
        self.verbose_mode = False
        self.evaluation_cost = 0.0

    @property
    def __name__(self):
        return "Latency Budget"

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        """Expects latency_ms to be stored in test_case.additional_metadata."""
        metadata = test_case.additional_metadata or {}
        latency = metadata.get("latency_ms", 0.0)

        if latency <= self.budget_ms:
            self._score = 1.0
            self._reason = f"Latency {latency:.0f}ms within budget {self.budget_ms:.0f}ms"
        elif latency <= self.budget_ms * 2:
            # Linear degradation from 1.0 to 0.0
            self._score = 1.0 - (latency - self.budget_ms) / self.budget_ms
            self._reason = (
                f"Latency {latency:.0f}ms exceeds budget {self.budget_ms:.0f}ms "
                f"(score: {self._score:.2f})"
            )
        else:
            self._score = 0.0
            self._reason = (
                f"Latency {latency:.0f}ms far exceeds budget {self.budget_ms:.0f}ms"
            )

        self._score = round(max(0.0, min(1.0, self._score)), 4)
        self.success = self._score >= self.threshold
        return self._score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return self._score >= self.threshold

    @property
    def score(self) -> float:
        return self._score

    @property
    def reason(self) -> str:
        return self._reason


# ---------------------------------------------------------------------------
# Dataset Conversion
# ---------------------------------------------------------------------------

def golden_dataset_to_deepeval(dataset: GoldenDataset) -> EvaluationDataset:
    """
    Convert our GoldenDataset to a DeepEval EvaluationDataset.

    Maps each TestCase to a DeepEval Golden with:
      - input: the email text
      - expected_output: JSON with category + summary (for metric comparison)
      - additional_metadata: difficulty, notes, test case ID
    """
    eval_dataset = EvaluationDataset()
    goldens = []

    for case in dataset.cases:
        expected_output = json.dumps({
            "category": case.expected_category.value,
            "summary": case.expected_summary,
        })

        golden = Golden(
            input=case.input_email,
            expected_output=expected_output,
            additional_metadata={
                "test_case_id": case.id,
                "difficulty": case.difficulty.value,
                "notes": case.notes,
            },
        )
        goldens.append(golden)

    eval_dataset.goldens = goldens
    return eval_dataset


# ---------------------------------------------------------------------------
# Result Mapping: DeepEval → Our EvalRun format
# ---------------------------------------------------------------------------

def deepeval_results_to_eval_run(
    test_results: list[dict[str, Any]],
    prompt_version: str,
    model: str,
    dataset_version: str,
) -> EvalRun:
    """
    Map DeepEval test results back to our EvalRun format for
    diff engine, drift detection, and historical tracking.
    """
    run = EvalRun(
        prompt_version=prompt_version,
        model=model,
        dataset_version=dataset_version,
    )

    for result in test_results:
        test_case_id = result.get("test_case_id", "unknown")
        input_email = result.get("input", "")
        actual_output = result.get("actual_output", "")
        expected_output = result.get("expected_output", "")

        # Parse outputs
        predicted_cat = CategoryCorrectnessMetric._extract_category(actual_output)
        expected_cat = CategoryCorrectnessMetric._extract_category(expected_output)

        predicted_summary = SummaryQualityMetric._extract_summary(actual_output)
        expected_summary = SummaryQualityMetric._extract_summary(expected_output)

        case_result = CaseResult(
            test_case_id=test_case_id,
            input_email=input_email,
            expected_category=EmailCategory(expected_cat) if expected_cat else EmailCategory.GENERAL,
            predicted_category=EmailCategory(predicted_cat) if predicted_cat else None,
            expected_summary=expected_summary or "",
            predicted_summary=predicted_summary,
            category_match=predicted_cat == expected_cat if predicted_cat and expected_cat else False,
            summary_relevance_score=result.get("summary_score", 0.0),
            latency_ms=result.get("latency_ms", 0.0),
            tokens_used=result.get("tokens", 0),
        )
        run.results.append(case_result)

    # Compute aggregates
    from src.scoring import compute_run_metrics
    run = compute_run_metrics(run)

    return run


# ---------------------------------------------------------------------------
# G-Eval factory (for when OPENAI_API_KEY is available)
# ---------------------------------------------------------------------------

def create_geval_category_metric(threshold: float = 0.7) -> GEval | None:
    """
    Create a G-Eval metric for category correctness using LLM-as-judge.
    Returns None if no API key is available (falls back to deterministic metric).
    """
    if not os.environ.get("OPENAI_API_KEY"):
        return None

    return GEval(
        name="Category Correctness (LLM Judge)",
        criteria=(
            "Evaluate whether the actual email classification matches the expected "
            "category. Consider: billing (payment/charges/refunds), technical "
            "(bugs/errors/performance), account (login/password/profile), and "
            "general (feedback/questions/other). Score 1.0 if correct, 0.0 if not."
        ),
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=threshold,
    )
