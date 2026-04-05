"""
CLI entry point for the Model Regression Detection System.

Orchestrates the full pipeline:
  1. Load prompt config and golden dataset
  2. Run evaluation
  3. Compare against baseline (if exists)
  4. Generate HTML report
  5. Send Slack alert
  6. Check for drift
  7. Exit with appropriate code for CI

Usage:
  python run_eval.py --prompt v1                    # Run eval, set as baseline
  python run_eval.py --prompt v2                    # Run eval, compare to latest
  python run_eval.py --prompt v2 --baseline <id>    # Compare to specific run
  python run_eval.py --prompt v1 --report-only      # Just generate report for latest

Environment:
  SLACK_WEBHOOK_URL    Slack webhook for alerts (optional)
  OPENAI_API_KEY       OpenAI API key (optional, uses mock if not set)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.classifier import EmailClassifier
from src.dataset_loader import load_dataset, dataset_stats
from src.diff_engine import DiffConfig, compare_runs, format_comparison_summary
from src.drift import DriftConfig, detect_drift
from src.eval_runner import EvalRunner
from src.mock_llm import MockLLMProvider
from src.models import RegressionSeverity
from src.prompt_loader import load_prompt, list_prompt_versions
from src.reporter import generate_html_report
from src.run_store import RunStore
from src.slack_alert import send_slack_alert

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_eval")

REPORTS_DIR = Path(__file__).parent / "reports"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Model Regression Detection — Eval Pipeline",
    )
    p.add_argument(
        "--prompt",
        required=True,
        help="Prompt version to evaluate (e.g., v1, v2)",
    )
    p.add_argument(
        "--dataset",
        default="v1",
        help="Golden dataset version (default: v1)",
    )
    p.add_argument(
        "--baseline",
        default=None,
        help="Baseline run ID to compare against (default: latest run)",
    )
    p.add_argument(
        "--warn-threshold",
        type=float,
        default=0.02,
        help="Accuracy drop %% to trigger warning (default: 2%%)",
    )
    p.add_argument(
        "--critical-threshold",
        type=float,
        default=0.04,
        help="Accuracy drop %% to trigger critical (default: 4%%)",
    )
    p.add_argument(
        "--report-url",
        default=None,
        help="Base URL for the published report (for Slack links)",
    )
    p.add_argument(
        "--no-slack",
        action="store_true",
        help="Disable Slack alerts",
    )
    p.add_argument(
        "--ci",
        action="store_true",
        help="CI mode: exit with non-zero code on critical regressions",
    )
    p.add_argument(
        "--db",
        default=None,
        help="Path to SQLite database (default: runs/eval_history.db)",
    )
    return p.parse_args()


async def run_pipeline(args: argparse.Namespace) -> int:
    """Execute the full eval pipeline. Returns exit code (0=pass, 1=fail)."""

    # ── 1. Setup ──────────────────────────────────────────────────────
    logger.info("Loading prompt version: %s", args.prompt)
    prompt_config = load_prompt(args.prompt)

    logger.info("Loading dataset version: %s", args.dataset)
    dataset = load_dataset(args.dataset)
    stats = dataset_stats(dataset)
    logger.info(
        "Dataset: %d cases (%s)",
        stats["total_cases"],
        ", ".join(f"{k}:{v}" for k, v in stats["by_category"].items()),
    )

    db_path = Path(args.db) if args.db else None
    store = RunStore(db_path) if db_path else RunStore()

    diff_config = DiffConfig(
        warning_threshold=args.warn_threshold,
        critical_threshold=args.critical_threshold,
    )

    # ── 2. Run evaluation ─────────────────────────────────────────────
    provider = MockLLMProvider(seed=42)
    classifier = EmailClassifier(provider=provider)
    runner = EvalRunner(classifier=classifier)

    logger.info("Running evaluation...")

    def on_progress(done: int, total: int) -> None:
        if done % 10 == 0 or done == total:
            logger.info("  Progress: %d/%d (%.0f%%)", done, total, done / total * 100)

    run = await runner.execute(prompt_config, dataset, on_progress=on_progress)

    logger.info(
        "Eval complete: accuracy=%.1f%%, relevance=%.2f, latency=%.0fms",
        run.overall_accuracy * 100,
        run.avg_summary_relevance,
        run.avg_latency_ms,
    )

    # ── 3. Save run ───────────────────────────────────────────────────
    store.save_run(run)
    logger.info("Run saved: %s", run.run_id)

    # ── 4. Compare to baseline ────────────────────────────────────────
    comparison = None
    baseline_run = None

    if args.baseline:
        baseline_run = store.get_run(args.baseline)
        if not baseline_run:
            logger.warning("Baseline run '%s' not found, skipping comparison", args.baseline)
    else:
        # Get latest run that isn't the current one
        all_runs = store.list_runs(limit=5)
        for r in all_runs:
            if r.run_id != run.run_id:
                baseline_run = store.get_run(r.run_id)
                break

    if baseline_run:
        comparison = compare_runs(baseline_run, run, diff_config)
        logger.info("\n%s", format_comparison_summary(comparison))
    else:
        logger.info("No baseline found — this run becomes the baseline.")

    # ── 5. Check for drift ────────────────────────────────────────────
    trend_data = store.get_accuracy_history(limit=30)
    drift_result = detect_drift(trend_data, DriftConfig())

    if drift_result.has_drift:
        logger.warning("DRIFT DETECTED:")
        for msg in drift_result.messages:
            logger.warning("  %s", msg)
    else:
        for msg in drift_result.messages:
            logger.info("  %s", msg)

    # ── 6. Generate report ────────────────────────────────────────────
    timestamp_str = run.timestamp.strftime("%Y%m%d_%H%M%S")
    report_filename = f"report_{run.prompt_version}_{timestamp_str}.html"
    report_path = REPORTS_DIR / report_filename

    generate_html_report(
        new_run=run,
        comparison=comparison,
        baseline_run=baseline_run,
        trend_data=trend_data,
        output_path=report_path,
    )
    logger.info("Report generated: %s", report_path)

    # ── 7. Send Slack alert ───────────────────────────────────────────
    if not args.no_slack:
        report_url = f"{args.report_url}/{report_filename}" if args.report_url else None
        send_slack_alert(run, comparison, report_url)

    # ── 8. Determine exit code ────────────────────────────────────────
    final_severity = RegressionSeverity.OK

    if comparison:
        final_severity = comparison.severity

    if drift_result.has_drift and drift_result.severity == RegressionSeverity.CRITICAL:
        final_severity = RegressionSeverity.CRITICAL

    print("\n" + "=" * 60)
    print(f"  Run ID:     {run.run_id}")
    print(f"  Prompt:     {run.prompt_version}")
    print(f"  Accuracy:   {run.overall_accuracy:.1%}")
    print(f"  Relevance:  {run.avg_summary_relevance:.2f}/5.0")
    print(f"  Status:     {final_severity.value.upper()}")
    print(f"  Report:     {report_path}")
    print("=" * 60)

    if args.ci and final_severity == RegressionSeverity.CRITICAL:
        logger.error("CI FAILURE: Critical regression detected. Blocking merge.")
        return 1

    return 0


def main() -> None:
    args = parse_args()
    exit_code = asyncio.run(run_pipeline(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
