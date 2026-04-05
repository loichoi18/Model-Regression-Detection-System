"""
Diff engine — the core value of the regression detection system.

Compares two EvalRun objects and produces a RunComparison that identifies:
  - Per-case regressions (pass → fail)
  - Per-case improvements (fail → pass)
  - Aggregate metric deltas
  - Severity classification based on configurable thresholds

Design decision: thresholds are configurable via DiffConfig rather than
hardcoded, because different teams have different tolerance for regression.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.models import (
    CaseDiff,
    CaseResult,
    EvalRun,
    RegressionSeverity,
    RunComparison,
)


@dataclass
class DiffConfig:
    """
    Configurable thresholds for regression severity.

    Defaults are sensible starting points:
    - warning_threshold: flag if accuracy drops > 3%
    - critical_threshold: block merge if accuracy drops > 8%
    - relevance_warning: flag if avg relevance drops > 0.3 points
    - relevance_critical: block if avg relevance drops > 0.8 points
    """

    warning_threshold: float = 0.03   # 3% accuracy drop
    critical_threshold: float = 0.08  # 8% accuracy drop
    relevance_warning: float = 0.3    # 0.3 point relevance drop (1–5 scale)
    relevance_critical: float = 0.8   # 0.8 point relevance drop
    latency_warning_ms: float = 200.0  # 200ms latency increase


def compare_runs(
    baseline: EvalRun,
    new_run: EvalRun,
    config: DiffConfig | None = None,
) -> RunComparison:
    """
    Compare a new eval run against a baseline.

    Produces a RunComparison with regressions, improvements, deltas,
    and an overall severity classification.
    """
    config = config or DiffConfig()

    # Index baseline results by test case ID for O(1) lookup
    baseline_map: dict[str, CaseResult] = {
        r.test_case_id: r for r in baseline.results
    }

    regressions: list[CaseDiff] = []
    improvements: list[CaseDiff] = []

    for new_result in new_run.results:
        old_result = baseline_map.get(new_result.test_case_id)
        if old_result is None:
            # New test case not in baseline — skip comparison
            continue

        # Skip cases where either run had an error
        if old_result.error or new_result.error:
            continue

        old_match = old_result.category_match
        new_match = new_result.category_match

        if old_match and not new_match:
            # REGRESSION: was correct, now wrong
            regressions.append(
                CaseDiff(
                    test_case_id=new_result.test_case_id,
                    direction="regression",
                    old_category=old_result.predicted_category,
                    new_category=new_result.predicted_category,
                    old_summary=old_result.predicted_summary,
                    new_summary=new_result.predicted_summary,
                    old_category_match=True,
                    new_category_match=False,
                )
            )
        elif not old_match and new_match:
            # IMPROVEMENT: was wrong, now correct
            improvements.append(
                CaseDiff(
                    test_case_id=new_result.test_case_id,
                    direction="improvement",
                    old_category=old_result.predicted_category,
                    new_category=new_result.predicted_category,
                    old_summary=old_result.predicted_summary,
                    new_summary=new_result.predicted_summary,
                    old_category_match=False,
                    new_category_match=True,
                )
            )

    # Compute deltas
    accuracy_delta = new_run.overall_accuracy - baseline.overall_accuracy
    relevance_delta = new_run.avg_summary_relevance - baseline.avg_summary_relevance
    latency_delta = new_run.avg_latency_ms - baseline.avg_latency_ms

    # Classify severity
    severity = _classify_severity(
        accuracy_delta=accuracy_delta,
        relevance_delta=relevance_delta,
        latency_delta=latency_delta,
        num_regressions=len(regressions),
        config=config,
    )

    return RunComparison(
        baseline_run_id=baseline.run_id,
        new_run_id=new_run.run_id,
        baseline_prompt_version=baseline.prompt_version,
        new_prompt_version=new_run.prompt_version,
        accuracy_delta=round(accuracy_delta, 4),
        relevance_delta=round(relevance_delta, 4),
        latency_delta_ms=round(latency_delta, 2),
        regressions=regressions,
        improvements=improvements,
        severity=severity,
    )


def _classify_severity(
    accuracy_delta: float,
    relevance_delta: float,
    latency_delta: float,
    num_regressions: int,
    config: DiffConfig,
) -> RegressionSeverity:
    """
    Determine severity based on multiple signals.

    Critical if ANY critical threshold is breached.
    Warning if ANY warning threshold is breached.
    OK otherwise.
    """
    # Check critical thresholds
    if (
        accuracy_delta < -config.critical_threshold
        or relevance_delta < -config.relevance_critical
    ):
        return RegressionSeverity.CRITICAL

    # Check warning thresholds
    if (
        accuracy_delta < -config.warning_threshold
        or relevance_delta < -config.relevance_warning
        or latency_delta > config.latency_warning_ms
        or num_regressions >= 3
    ):
        return RegressionSeverity.WARNING

    return RegressionSeverity.OK


def format_comparison_summary(comparison: RunComparison) -> str:
    """Produce a human-readable summary of a run comparison."""
    lines = [
        f"Comparison: {comparison.baseline_prompt_version} → {comparison.new_prompt_version}",
        f"Severity: {comparison.severity.value.upper()}",
        f"Accuracy:  {comparison.accuracy_delta:+.1%}",
        f"Relevance: {comparison.relevance_delta:+.2f}",
        f"Latency:   {comparison.latency_delta_ms:+.0f}ms",
        f"Regressions: {len(comparison.regressions)}",
        f"Improvements: {len(comparison.improvements)}",
    ]

    if comparison.regressions:
        lines.append("\nRegressed cases:")
        for diff in comparison.regressions:
            lines.append(
                f"  {diff.test_case_id}: "
                f"{diff.old_category.value if diff.old_category else '?'} → "
                f"{diff.new_category.value if diff.new_category else '?'}"
            )

    if comparison.improvements:
        lines.append("\nImproved cases:")
        for diff in comparison.improvements:
            lines.append(
                f"  {diff.test_case_id}: "
                f"{diff.old_category.value if diff.old_category else '?'} → "
                f"{diff.new_category.value if diff.new_category else '?'}"
            )

    return "\n".join(lines)
