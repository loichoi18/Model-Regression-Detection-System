"""
Async evaluation runner.

Executes every test case in the golden dataset against the classifier,
collects raw outputs, scores them, and assembles an EvalRun.

Design decisions:
  - asyncio.Semaphore controls concurrency (avoids API rate limits)
  - Each case is isolated: one failure doesn't stop the run
  - Progress callback supports both CLI and Streamlit progress bars
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable

from src.classifier import EmailClassifier
from src.mock_llm import MockLLMProvider
from src.models import (
    CaseResult,
    EvalRun,
    GoldenDataset,
    PromptConfig,
    TestCase,
)
from src.scoring import TextSimilarityScorer, compute_run_metrics, score_case

logger = logging.getLogger(__name__)


class EvalRunner:
    """
    Runs a full evaluation of a prompt config against a golden dataset.

    Usage:
        runner = EvalRunner(classifier=EmailClassifier(MockLLMProvider()))
        run = await runner.execute(prompt_config, dataset)
    """

    def __init__(
        self,
        classifier: EmailClassifier | None = None,
        max_concurrency: int = 10,
        scorer: TextSimilarityScorer | None = None,
    ):
        self.classifier = classifier or EmailClassifier(MockLLMProvider())
        self.max_concurrency = max_concurrency
        self.scorer = scorer or TextSimilarityScorer()
        self._semaphore: asyncio.Semaphore | None = None

    async def execute(
        self,
        prompt_config: PromptConfig,
        dataset: GoldenDataset,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> EvalRun:
        """
        Run evaluation on all test cases.

        Args:
            prompt_config: The prompt version to evaluate
            dataset: The golden dataset to test against
            on_progress: Optional callback(completed, total) for progress updates

        Returns:
            A fully scored EvalRun with aggregate metrics
        """
        self._semaphore = asyncio.Semaphore(self.max_concurrency)

        run = EvalRun(
            prompt_version=prompt_config.version,
            model=prompt_config.model,
            dataset_version=dataset.version,
        )

        total = len(dataset.cases)
        completed = 0

        # Launch all cases concurrently (bounded by semaphore)
        tasks = [
            self._eval_single_case(prompt_config, case)
            for case in dataset.cases
        ]

        # Gather with return_exceptions so one failure doesn't abort all
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Case failed entirely — create error result
                case = dataset.cases[i]
                error_result = CaseResult(
                    test_case_id=case.id,
                    input_email=case.input_email,
                    expected_category=case.expected_category,
                    expected_summary=case.expected_summary,
                    error=str(result),
                )
                run.results.append(error_result)
            else:
                run.results.append(result)

            completed += 1
            if on_progress:
                on_progress(completed, total)

        # Compute aggregate metrics
        run = compute_run_metrics(run)

        logger.info(
            "Eval complete: run=%s prompt=%s accuracy=%.1f%% cases=%d",
            run.run_id,
            run.prompt_version,
            run.overall_accuracy * 100,
            len(run.results),
        )

        return run

    async def _eval_single_case(
        self,
        prompt_config: PromptConfig,
        case: TestCase,
    ) -> CaseResult:
        """Evaluate a single test case with semaphore-bounded concurrency."""
        async with self._semaphore:
            start = time.perf_counter()
            try:
                classification, latency_ms, tokens = await self.classifier.classify(
                    prompt_config, case.input_email
                )
                elapsed_ms = (time.perf_counter() - start) * 1000

                result = CaseResult(
                    test_case_id=case.id,
                    input_email=case.input_email,
                    expected_category=case.expected_category,
                    predicted_category=classification.category,
                    expected_summary=case.expected_summary,
                    predicted_summary=classification.summary,
                    latency_ms=round(latency_ms, 2),
                    tokens_used=tokens,
                )

                # Score the result
                result = score_case(result, self.scorer)
                return result

            except Exception as e:
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.warning(
                    "Case %s failed after %.0fms: %s", case.id, elapsed_ms, e
                )
                return CaseResult(
                    test_case_id=case.id,
                    input_email=case.input_email,
                    expected_category=case.expected_category,
                    expected_summary=case.expected_summary,
                    latency_ms=round(elapsed_ms, 2),
                    error=str(e),
                )
