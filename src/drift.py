"""
Drift detection — catches gradual quality degradation.

Per-run diff checks are great for catching big regressions in a single prompt
change. But what about slow drift? If accuracy drops 0.5% across 10 runs,
no single run triggers an alert — but the cumulative effect is real.

This module tracks rolling averages and fires "slow drift" warnings when
the trend line crosses a threshold, even if no individual run looks bad.

This is the insight that separates this project from a basic eval script.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.models import RegressionSeverity


@dataclass
class DriftConfig:
    """
    Configuration for drift detection.

    window_size: Number of recent runs to include in the rolling average
    accuracy_floor: Minimum acceptable rolling accuracy
    relevance_floor: Minimum acceptable rolling relevance (1–5 scale)
    latency_ceiling_ms: Maximum acceptable rolling latency
    """

    window_size: int = 7
    accuracy_floor: float = 0.85   # 85%
    relevance_floor: float = 3.0   # 3.0 / 5.0
    latency_ceiling_ms: float = 500.0


@dataclass
class DriftResult:
    """Result of a drift analysis."""

    has_drift: bool
    severity: RegressionSeverity
    rolling_accuracy: float
    rolling_relevance: float
    rolling_latency_ms: float
    window_size: int
    actual_window: int  # May be smaller if not enough history
    messages: list[str]


def detect_drift(
    trend_data: list[dict],
    config: DriftConfig | None = None,
) -> DriftResult:
    """
    Analyze recent run history for slow drift.

    Args:
        trend_data: List of dicts with keys: accuracy, relevance, latency_ms
                    (from RunStore.get_accuracy_history())
        config: Drift detection thresholds

    Returns:
        DriftResult with findings
    """
    config = config or DriftConfig()

    if not trend_data:
        return DriftResult(
            has_drift=False,
            severity=RegressionSeverity.OK,
            rolling_accuracy=0.0,
            rolling_relevance=0.0,
            rolling_latency_ms=0.0,
            window_size=config.window_size,
            actual_window=0,
            messages=["No run history available for drift analysis."],
        )

    # Take the most recent N runs
    window = trend_data[-config.window_size :]
    n = len(window)

    rolling_accuracy = sum(d["accuracy"] for d in window) / n
    rolling_relevance = sum(d["relevance"] for d in window) / n
    rolling_latency = sum(d["latency_ms"] for d in window) / n

    messages: list[str] = []
    has_drift = False
    worst_severity = RegressionSeverity.OK

    # Check accuracy floor
    if rolling_accuracy < config.accuracy_floor:
        has_drift = True
        messages.append(
            f"Rolling accuracy ({rolling_accuracy:.1%}) is below floor "
            f"({config.accuracy_floor:.1%}) over {n} runs."
        )
        # How far below determines severity
        gap = config.accuracy_floor - rolling_accuracy
        if gap > 0.10:
            worst_severity = RegressionSeverity.CRITICAL
        else:
            worst_severity = max_severity(worst_severity, RegressionSeverity.WARNING)

    # Check relevance floor
    if rolling_relevance < config.relevance_floor:
        has_drift = True
        messages.append(
            f"Rolling relevance ({rolling_relevance:.2f}) is below floor "
            f"({config.relevance_floor:.1f}) over {n} runs."
        )
        worst_severity = max_severity(worst_severity, RegressionSeverity.WARNING)

    # Check latency ceiling
    if rolling_latency > config.latency_ceiling_ms:
        has_drift = True
        messages.append(
            f"Rolling latency ({rolling_latency:.0f}ms) exceeds ceiling "
            f"({config.latency_ceiling_ms:.0f}ms) over {n} runs."
        )
        worst_severity = max_severity(worst_severity, RegressionSeverity.WARNING)

    # Check for downward trend (compare first half vs second half of window)
    if n >= 4:
        mid = n // 2
        first_half_acc = sum(d["accuracy"] for d in window[:mid]) / mid
        second_half_acc = sum(d["accuracy"] for d in window[mid:]) / (n - mid)
        trend_delta = second_half_acc - first_half_acc

        if trend_delta < -0.03:  # Declining by more than 3%
            has_drift = True
            messages.append(
                f"Accuracy trending downward: {first_half_acc:.1%} → "
                f"{second_half_acc:.1%} ({trend_delta:+.1%}) across window."
            )
            worst_severity = max_severity(worst_severity, RegressionSeverity.WARNING)

    if not has_drift:
        messages.append(
            f"No drift detected. Rolling accuracy: {rolling_accuracy:.1%}, "
            f"relevance: {rolling_relevance:.2f}, latency: {rolling_latency:.0f}ms "
            f"(window: {n} runs)."
        )

    return DriftResult(
        has_drift=has_drift,
        severity=worst_severity,
        rolling_accuracy=rolling_accuracy,
        rolling_relevance=rolling_relevance,
        rolling_latency_ms=rolling_latency,
        window_size=config.window_size,
        actual_window=n,
        messages=messages,
    )


def max_severity(
    a: RegressionSeverity, b: RegressionSeverity
) -> RegressionSeverity:
    """Return the more severe of two severities."""
    order = {
        RegressionSeverity.OK: 0,
        RegressionSeverity.WARNING: 1,
        RegressionSeverity.CRITICAL: 2,
    }
    return a if order[a] >= order[b] else b
