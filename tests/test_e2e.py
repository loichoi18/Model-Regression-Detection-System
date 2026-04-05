"""
End-to-end integration test for the Model Regression Detection System.

Validates the full pipeline:
  1. Dataset loading and validation
  2. Prompt loading and versioning
  3. Eval execution with mock LLM
  4. Scoring (category match + summary relevance)
  5. Run persistence (SQLite store)
  6. Run comparison and regression detection
  7. Drift detection
  8. HTML report generation
  9. Slack alert payload construction

Run with: python -m pytest tests/test_e2e.py -v
    or:   python tests/test_e2e.py
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classifier import EmailClassifier
from src.dataset_loader import load_dataset, dataset_stats, filter_cases
from src.diff_engine import DiffConfig, compare_runs, format_comparison_summary
from src.drift import DriftConfig, detect_drift
from src.eval_runner import EvalRunner
from src.mock_llm import MockLLMProvider
from src.models import (
    DifficultyTag,
    EmailCategory,
    EvalRun,
    RegressionSeverity,
)
from src.prompt_loader import list_prompt_versions, load_prompt
from src.reporter import generate_html_report
from src.run_store import RunStore
from src.scoring import (
    TextSimilarityScorer,
    accuracy_by_category,
    compute_run_metrics,
    score_case,
)
from src.slack_alert import _build_payload


class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BOLD = "\033[1m"
    END = "\033[0m"


passed = 0
failed = 0
total = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed, total
    total += 1
    if condition:
        passed += 1
        print(f"  {Colors.GREEN}✓{Colors.END} {name}")
    else:
        failed += 1
        msg = f" — {detail}" if detail else ""
        print(f"  {Colors.RED}✗ {name}{msg}{Colors.END}")


async def run_all_tests():
    global passed, failed, total

    print(f"\n{Colors.BOLD}{'=' * 70}")
    print("  MODEL REGRESSION DETECTOR — END-TO-END TEST SUITE")
    print(f"{'=' * 70}{Colors.END}\n")

    # ──────────────────────────────────────────────────────────────────
    # 1. DATASET
    # ──────────────────────────────────────────────────────────────────
    print(f"{Colors.BOLD}[1/10] Dataset Loading & Validation{Colors.END}")

    dataset = load_dataset("v1")
    stats = dataset_stats(dataset)

    check("Dataset loads without error", dataset is not None)
    check(
        f"Has {stats['total_cases']} cases (≥50 required)",
        stats["total_cases"] >= 50,
    )
    check(
        "All 4 categories represented",
        set(stats["by_category"].keys()) == {"billing", "technical", "account", "general"},
    )
    check(
        "Has edge cases",
        stats["by_difficulty"].get("edge_case", 0) >= 5,
    )

    # Verify no duplicate IDs
    ids = [c.id for c in dataset.cases]
    check("No duplicate case IDs", len(ids) == len(set(ids)))

    # Filter works
    billing_only = filter_cases(dataset, category=EmailCategory.BILLING)
    check(
        f"Category filter works (billing: {len(billing_only.cases)} cases)",
        len(billing_only.cases) > 0
        and all(c.expected_category == EmailCategory.BILLING for c in billing_only.cases),
    )

    edge_only = filter_cases(dataset, difficulty=DifficultyTag.EDGE_CASE)
    check(
        f"Difficulty filter works (edge_case: {len(edge_only.cases)} cases)",
        len(edge_only.cases) > 0,
    )

    # ──────────────────────────────────────────────────────────────────
    # 2. PROMPT LOADING
    # ──────────────────────────────────────────────────────────────────
    print(f"\n{Colors.BOLD}[2/10] Prompt Versioning{Colors.END}")

    versions = list_prompt_versions()
    check(f"Found {len(versions)} prompt versions", len(versions) >= 2)

    config_v1 = load_prompt("v1")
    check(
        "v1 loads correctly",
        config_v1.version == "v1" and len(config_v1.system_prompt) > 50,
    )
    check(
        f"v1 has {len(config_v1.few_shot_examples)} few-shot examples",
        len(config_v1.few_shot_examples) >= 2,
    )

    config_v2 = load_prompt("v2")
    check("v2 loads correctly", config_v2.version == "v2")
    check("v2 has fewer few-shots than v1",
          len(config_v2.few_shot_examples) < len(config_v1.few_shot_examples))

    messages = config_v1.build_messages("test email")
    check(
        f"Message builder produces {len(messages)} messages",
        len(messages) == 1 + len(config_v1.few_shot_examples) * 2 + 1,
    )

    # ──────────────────────────────────────────────────────────────────
    # 3. CLASSIFICATION
    # ──────────────────────────────────────────────────────────────────
    print(f"\n{Colors.BOLD}[3/10] Mock LLM Classification{Colors.END}")

    provider = MockLLMProvider(seed=42)
    classifier = EmailClassifier(provider=provider)

    result, latency, tokens = await classifier.classify(
        config_v1, "I was charged $50 twice on my card."
    )
    check("Classification returns a result", result is not None)
    check(
        f"Billing email → {result.category.value}",
        result.category == EmailCategory.BILLING,
    )
    check(f"Latency is positive ({latency:.0f}ms)", latency > 0)
    check(f"Token count is positive ({tokens})", tokens > 0)
    check(
        f"Confidence is valid ({result.confidence})",
        0.0 <= result.confidence <= 1.0,
    )

    # v2 degradation check
    provider_v2 = MockLLMProvider(seed=42)
    classifier_v2 = EmailClassifier(provider=provider_v2)
    result_v2, _, _ = await classifier_v2.classify(
        config_v2, "I want to downgrade my plan."
    )
    check(
        f"v2 misclassifies 'downgrade' as {result_v2.category.value} (expected regression)",
        result_v2.category == EmailCategory.ACCOUNT,
    )

    # ──────────────────────────────────────────────────────────────────
    # 4. SCORING
    # ──────────────────────────────────────────────────────────────────
    print(f"\n{Colors.BOLD}[4/10] Multi-Dimensional Scoring{Colors.END}")

    scorer = TextSimilarityScorer()

    # Identical summaries → high score
    score_same = scorer.score(
        "Customer reports duplicate charge and requests refund.",
        "Customer reports duplicate charge and requests refund.",
    )
    check(f"Identical summaries score high ({score_same:.2f})", score_same >= 4.5)

    # Completely different → low score
    score_diff = scorer.score(
        "Customer reports duplicate charge and requests refund.",
        "The weather is sunny today with a high of 75 degrees.",
    )
    check(f"Unrelated summaries score low ({score_diff:.2f})", score_diff < 2.5)

    # Similar summaries → score above unrelated (word-overlap scorer is strict)
    score_similar = scorer.score(
        "Customer reports duplicate charge and requests refund.",
        "Customer was charged twice and wants their money back.",
    )
    check(f"Similar summaries score above unrelated ({score_similar:.2f})", score_similar >= score_diff)

    # Score ordering is correct
    check(
        "Score ordering: identical > similar > unrelated",
        score_same > score_similar > score_diff,
    )

    # ──────────────────────────────────────────────────────────────────
    # 5. EVAL RUNNER
    # ──────────────────────────────────────────────────────────────────
    print(f"\n{Colors.BOLD}[5/10] Async Eval Runner{Colors.END}")

    runner = EvalRunner(
        classifier=EmailClassifier(MockLLMProvider(seed=42)),
        max_concurrency=5,
    )

    progress_calls = []
    def on_progress(done, total):
        progress_calls.append((done, total))

    run_v1 = await runner.execute(config_v1, dataset, on_progress=on_progress)

    check(
        f"Run completed with {len(run_v1.results)} results",
        len(run_v1.results) == len(dataset.cases),
    )
    check(
        f"Overall accuracy computed ({run_v1.overall_accuracy:.1%})",
        0 < run_v1.overall_accuracy <= 1.0,
    )
    check(
        f"Average relevance computed ({run_v1.avg_summary_relevance:.2f})",
        run_v1.avg_summary_relevance > 0,
    )
    check(
        f"Average latency computed ({run_v1.avg_latency_ms:.0f}ms)",
        run_v1.avg_latency_ms > 0,
    )
    check(
        f"Total tokens computed ({run_v1.total_tokens})",
        run_v1.total_tokens > 0,
    )
    check(
        f"Progress callback fired {len(progress_calls)} times",
        len(progress_calls) == len(dataset.cases),
    )
    check("No errors in results", all(r.error is None for r in run_v1.results))

    # Per-category breakdown
    by_cat = accuracy_by_category(run_v1)
    check(
        f"Per-category breakdown has {len(by_cat)} categories",
        len(by_cat) == 4,
    )

    # Run v2
    runner_v2 = EvalRunner(
        classifier=EmailClassifier(MockLLMProvider(seed=42)),
    )
    run_v2 = await runner_v2.execute(config_v2, dataset)
    check(
        f"v2 accuracy ({run_v2.overall_accuracy:.1%}) < v1 ({run_v1.overall_accuracy:.1%})",
        run_v2.overall_accuracy < run_v1.overall_accuracy,
    )

    # ──────────────────────────────────────────────────────────────────
    # 6. RUN STORE (SQLite)
    # ──────────────────────────────────────────────────────────────────
    print(f"\n{Colors.BOLD}[6/10] SQLite Run Store{Colors.END}")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_eval.db"
        store = RunStore(db_path)

        store.save_run(run_v1)
        check("v1 run saved", True)

        store.save_run(run_v2)
        check("v2 run saved", True)

        loaded = store.get_run(run_v1.run_id)
        check("Run loads from DB", loaded is not None)
        check(
            "Loaded run matches original accuracy",
            abs(loaded.overall_accuracy - run_v1.overall_accuracy) < 0.001,
        )
        check(
            f"Loaded run has {len(loaded.results)} case results",
            len(loaded.results) == len(run_v1.results),
        )

        latest = store.get_latest_run()
        check(
            f"Latest run is v2 ({latest.run_id})",
            latest.run_id == run_v2.run_id,
        )

        all_runs = store.list_runs(limit=10)
        check(f"List runs returns {len(all_runs)} runs", len(all_runs) == 2)

        history = store.get_accuracy_history(limit=10)
        check(
            f"Accuracy history has {len(history)} entries",
            len(history) == 2,
        )

        # Idempotent save
        store.save_run(run_v1)
        all_runs2 = store.list_runs()
        check("Idempotent save doesn't duplicate runs", len(all_runs2) == 2)

        # ──────────────────────────────────────────────────────────────
        # 7. DIFF ENGINE
        # ──────────────────────────────────────────────────────────────
        print(f"\n{Colors.BOLD}[7/10] Diff Engine & Regression Detection{Colors.END}")

        comparison = compare_runs(
            run_v1, run_v2, DiffConfig(warning_threshold=0.02, critical_threshold=0.04)
        )

        check(
            f"Accuracy delta is negative ({comparison.accuracy_delta:+.1%})",
            comparison.accuracy_delta < 0,
        )
        check(
            f"Found {len(comparison.regressions)} regressions",
            len(comparison.regressions) > 0,
        )
        check(
            f"Found {len(comparison.improvements)} improvements",
            len(comparison.improvements) >= 0,
        )
        check(
            f"Severity is CRITICAL (delta={comparison.accuracy_delta:.1%})",
            comparison.severity == RegressionSeverity.CRITICAL,
        )

        # All regressions are valid (were correct in v1, wrong in v2)
        for diff in comparison.regressions:
            check(
                f"  Regression {diff.test_case_id}: was correct → now wrong",
                diff.old_category_match is True and diff.new_category_match is False,
            )

        # Format summary
        summary = format_comparison_summary(comparison)
        check("Summary contains regression count", "Regressions:" in summary)
        check("Summary contains severity", "CRITICAL" in summary)

        # Test WARNING threshold (critical at 10% so 5% drop = warning only)
        comparison_mild = compare_runs(
            run_v1,
            run_v2,
            DiffConfig(warning_threshold=0.02, critical_threshold=0.10),
        )
        check(
            "With high critical threshold → WARNING instead of CRITICAL",
            comparison_mild.severity == RegressionSeverity.WARNING,
        )

        # ──────────────────────────────────────────────────────────────
        # 8. DRIFT DETECTION
        # ──────────────────────────────────────────────────────────────
        print(f"\n{Colors.BOLD}[8/10] Drift Detection{Colors.END}")

        trend_data = history  # From store above

        # With 98% floor, our 97.5% rolling accuracy (v1=100%, v2=95%) triggers drift
        drift_result = detect_drift(trend_data, DriftConfig(accuracy_floor=0.98))
        check("Drift detected (accuracy below 98% floor)", drift_result.has_drift)
        check(
            f"Rolling accuracy computed ({drift_result.rolling_accuracy:.1%})",
            drift_result.rolling_accuracy > 0,
        )
        check("Drift messages generated", len(drift_result.messages) > 0)

        # With low floors across all dimensions, no drift
        drift_ok = detect_drift(
            trend_data,
            DriftConfig(accuracy_floor=0.50, relevance_floor=1.0, latency_ceiling_ms=2000.0),
        )
        check("No drift when floors are low", not drift_ok.has_drift)

        # Empty history
        drift_empty = detect_drift([], DriftConfig())
        check("Empty history → no drift", not drift_empty.has_drift)

        # ──────────────────────────────────────────────────────────────
        # 9. HTML REPORT & SLACK PAYLOAD
        # ──────────────────────────────────────────────────────────────
        print(f"\n{Colors.BOLD}[9/10] Report Generation & Slack Alerts{Colors.END}")

        report_path = Path(tmpdir) / "test_report.html"
        html = generate_html_report(
            new_run=run_v2,
            comparison=comparison,
            baseline_run=run_v1,
            trend_data=trend_data,
            output_path=report_path,
        )

        check("HTML report generated", len(html) > 1000)
        check("Report file written to disk", report_path.exists())
        check("Report contains run ID", run_v2.run_id in html)
        check("Report contains prompt version", "v2" in html)
        check("Report contains CRITICAL status", "CRITICAL" in html)
        check(
            "Report contains regression rows",
            any(d.test_case_id in html for d in comparison.regressions),
        )
        check("Report contains SVG trend chart", "<svg" in html)

        # Slack payload
        payload = _build_payload(run_v2, comparison, "https://example.com/report.html")
        check("Slack payload has blocks", len(payload["blocks"]) >= 3)
        check("Slack payload has status", payload["_status"] == "CRITICAL")
        check("Slack headline mentions regressions", "regression" in payload["_headline"])

        # Baseline report (no comparison)
        html_baseline = generate_html_report(new_run=run_v1)
        check("Baseline report generates (no comparison)", len(html_baseline) > 500)
        check("Baseline report shows BASELINE status", "BASELINE" in html_baseline)

        store.close()

    # ──────────────────────────────────────────────────────────────────
    # 10. DEEPEVAL ADAPTER
    # ──────────────────────────────────────────────────────────────────
    print(f"\n{Colors.BOLD}[10/10] DeepEval Integration{Colors.END}")

    from src.deepeval_adapter import (
        CategoryCorrectnessMetric,
        LatencyBudgetMetric,
        SummaryQualityMetric,
        golden_dataset_to_deepeval,
    )
    from deepeval.test_case import LLMTestCase

    # Dataset conversion
    de_dataset = golden_dataset_to_deepeval(dataset)
    check(
        f"Dataset converts to DeepEval ({len(de_dataset.goldens)} goldens)",
        len(de_dataset.goldens) == len(dataset.cases),
    )

    first_golden = de_dataset.goldens[0]
    check(
        "Golden has input",
        first_golden.input is not None and len(first_golden.input) > 0,
    )
    check(
        "Golden has expected_output as JSON",
        '"category"' in first_golden.expected_output,
    )
    check(
        "Golden carries metadata (test_case_id)",
        first_golden.additional_metadata.get("test_case_id") is not None,
    )

    # Category metric — correct prediction
    cat_metric = CategoryCorrectnessMetric(threshold=1.0)
    correct_case = LLMTestCase(
        input="test email",
        actual_output='{"category": "billing", "summary": "test"}',
        expected_output='{"category": "billing", "summary": "expected"}',
    )
    cat_metric.measure(correct_case)
    check(
        f"Category metric: correct → score {cat_metric.score}",
        cat_metric.score == 1.0 and cat_metric.is_successful(),
    )

    # Category metric — wrong prediction
    wrong_case = LLMTestCase(
        input="test email",
        actual_output='{"category": "technical", "summary": "test"}',
        expected_output='{"category": "billing", "summary": "expected"}',
    )
    cat_metric.measure(wrong_case)
    check(
        f"Category metric: incorrect → score {cat_metric.score}",
        cat_metric.score == 0.0 and not cat_metric.is_successful(),
    )

    # Summary metric
    sum_metric = SummaryQualityMetric(threshold=0.0)
    sum_metric.measure(correct_case)
    check(
        f"Summary metric produces score ({sum_metric.score:.2f})",
        0.0 <= sum_metric.score <= 1.0,
    )

    # Latency metric — within budget
    lat_metric = LatencyBudgetMetric(budget_ms=500.0)
    fast_case = LLMTestCase(
        input="test",
        actual_output="test",
        additional_metadata={"latency_ms": 150.0},
    )
    lat_metric.measure(fast_case)
    check(
        f"Latency metric: 150ms < 500ms budget → score {lat_metric.score}",
        lat_metric.score == 1.0,
    )

    # Latency metric — over budget
    slow_case = LLMTestCase(
        input="test",
        actual_output="test",
        additional_metadata={"latency_ms": 750.0},
    )
    lat_metric.measure(slow_case)
    check(
        f"Latency metric: 750ms > 500ms budget → degraded score {lat_metric.score:.2f}",
        0.0 < lat_metric.score < 1.0,
    )

    # ──────────────────────────────────────────────────────────────────
    # RESULTS
    # ──────────────────────────────────────────────────────────────────
    print(f"\n{Colors.BOLD}{'=' * 70}")
    print(f"  RESULTS: {passed}/{total} passed, {failed} failed")
    if failed == 0:
        print(f"  {Colors.GREEN}ALL TESTS PASSED ✓{Colors.END}")
    else:
        print(f"  {Colors.RED}{failed} TESTS FAILED ✗{Colors.END}")
    print(f"{'=' * 70}{Colors.END}\n")

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
