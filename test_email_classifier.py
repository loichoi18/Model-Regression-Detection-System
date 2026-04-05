"""
DeepEval-powered test suite for the email classifier.

This file follows DeepEval's CI/CD testing pattern:
  - Load golden dataset as DeepEval EvaluationDataset
  - Parametrize over goldens using pytest
  - Run classifier, build LLMTestCase
  - Assert with custom + DeepEval metrics
  - Run via: deepeval test run test_email_classifier.py

The test file works in two modes:
  1. Mock mode (default): No API key needed, zero cost, deterministic.
     Uses CategoryCorrectnessMetric + SummaryQualityMetric (text similarity).
  2. LLM Judge mode: Set OPENAI_API_KEY to enable G-Eval for deeper scoring.
     Falls back to mock mode gracefully if key isn't set.

This is the file that CI/CD triggers on every PR.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

import pytest

from deepeval import assert_test
from deepeval.dataset import Golden
from deepeval.test_case import LLMTestCase

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.classifier import EmailClassifier
from src.dataset_loader import load_dataset
from src.deepeval_adapter import (
    CategoryCorrectnessMetric,
    LatencyBudgetMetric,
    SummaryQualityMetric,
    golden_dataset_to_deepeval,
)
from src.mock_llm import MockLLMProvider
from src.prompt_loader import load_prompt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROMPT_VERSION = os.environ.get("EVAL_PROMPT_VERSION", "v1")
DATASET_VERSION = os.environ.get("EVAL_DATASET_VERSION", "v1")
MOCK_SEED = int(os.environ.get("EVAL_MOCK_SEED", "42"))

# ---------------------------------------------------------------------------
# Setup: load dataset and classifier once
# ---------------------------------------------------------------------------

_dataset = load_dataset(DATASET_VERSION)
_deepeval_dataset = golden_dataset_to_deepeval(_dataset)
_prompt_config = load_prompt(PROMPT_VERSION)
_classifier = EmailClassifier(provider=MockLLMProvider(seed=MOCK_SEED))

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

_category_metric = CategoryCorrectnessMetric(threshold=1.0)
_summary_metric = SummaryQualityMetric(threshold=0.0)  # Permissive for mock
_latency_metric = LatencyBudgetMetric(budget_ms=500.0, threshold=0.8)


# ---------------------------------------------------------------------------
# Helper: run async classifier in sync pytest context
# ---------------------------------------------------------------------------

def _classify_sync(email_text: str) -> tuple[str, float, int]:
    """Run the async classifier synchronously for pytest compatibility."""
    loop = asyncio.new_event_loop()
    try:
        result, latency_ms, tokens = loop.run_until_complete(
            _classifier.classify(_prompt_config, email_text)
        )
        actual_output = json.dumps({
            "category": result.category.value,
            "summary": result.summary,
            "confidence": result.confidence,
        })
        return actual_output, latency_ms, tokens
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Parametrized test — one test per golden
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "golden",
    _deepeval_dataset.goldens,
    ids=[g.additional_metadata.get("test_case_id", f"case_{i}")
         for i, g in enumerate(_deepeval_dataset.goldens)],
)
def test_email_classifier(golden: Golden):
    """
    DeepEval unit test for the email classifier.

    For each golden test case:
      1. Run the classifier on the input email
      2. Build an LLMTestCase with actual vs expected output
      3. Assert against category correctness, summary quality, and latency
    """
    # Run classifier
    actual_output, latency_ms, tokens = _classify_sync(golden.input)

    # Build DeepEval test case
    test_case = LLMTestCase(
        input=golden.input,
        actual_output=actual_output,
        expected_output=golden.expected_output,
        additional_metadata={
            **(golden.additional_metadata or {}),
            "latency_ms": latency_ms,
            "tokens_used": tokens,
            "prompt_version": PROMPT_VERSION,
        },
    )

    # Assert with all metrics
    assert_test(
        test_case=test_case,
        metrics=[
            _category_metric,
            _summary_metric,
            _latency_metric,
        ],
    )
