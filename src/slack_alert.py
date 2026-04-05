"""
Slack webhook integration for eval alerts.

Sends structured Block Kit messages to a Slack channel with:
  - Pass / Warning / Critical status
  - Headline metrics (accuracy, regressions, improvements)
  - Link to the full HTML diff report

Configuration:
  Set SLACK_WEBHOOK_URL environment variable, or pass it directly.
  If not configured, alerts are logged to stdout instead (useful for dev).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from src.models import EvalRun, RegressionSeverity, RunComparison

logger = logging.getLogger(__name__)


def send_slack_alert(
    run: EvalRun,
    comparison: RunComparison | None = None,
    report_url: str | None = None,
    webhook_url: str | None = None,
) -> bool:
    """
    Send an eval alert to Slack.

    Returns True if the message was sent (or logged) successfully.
    """
    webhook_url = webhook_url or os.environ.get("SLACK_WEBHOOK_URL")
    payload = _build_payload(run, comparison, report_url)

    if not webhook_url:
        # No webhook configured — log the payload for development
        logger.info(
            "Slack alert (no webhook configured):\n%s",
            json.dumps(payload, indent=2),
        )
        print("\n📢 SLACK ALERT (stdout fallback):")
        print(f"   Status: {payload['_status']}")
        print(f"   Message: {payload['_headline']}")
        if report_url:
            print(f"   Report: {report_url}")
        return True

    try:
        import requests

        resp = requests.post(
            webhook_url,
            json={"blocks": payload["blocks"]},
            timeout=10,
        )
        resp.raise_for_status()
        logger.info("Slack alert sent successfully")
        return True
    except Exception as e:
        logger.error("Failed to send Slack alert: %s", e)
        return False


def _build_payload(
    run: EvalRun,
    comparison: RunComparison | None,
    report_url: str | None,
) -> dict[str, Any]:
    """Build a Slack Block Kit payload."""
    severity = comparison.severity if comparison else RegressionSeverity.OK

    emoji = {
        RegressionSeverity.OK: "✅",
        RegressionSeverity.WARNING: "⚠️",
        RegressionSeverity.CRITICAL: "🚨",
    }[severity]

    status = severity.value.upper()

    # Headline
    if comparison and comparison.regressions:
        headline = (
            f"{emoji} *{status}*: {len(comparison.regressions)} regression(s) detected — "
            f"accuracy {run.overall_accuracy:.1%} "
            f"({comparison.accuracy_delta:+.1%} from baseline)"
        )
    elif comparison:
        headline = (
            f"{emoji} *{status}*: Prompt {run.prompt_version} eval passed — "
            f"accuracy {run.overall_accuracy:.1%} "
            f"({comparison.accuracy_delta:+.1%})"
        )
    else:
        headline = (
            f"{emoji} *BASELINE*: Prompt {run.prompt_version} — "
            f"accuracy {run.overall_accuracy:.1%}"
        )

    blocks: list[dict[str, Any]] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"Model Eval: {status}",
                "emoji": True,
            },
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": headline},
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Prompt:*\n{run.prompt_version}"},
                {"type": "mrkdwn", "text": f"*Model:*\n{run.model}"},
                {"type": "mrkdwn", "text": f"*Accuracy:*\n{run.overall_accuracy:.1%}"},
                {
                    "type": "mrkdwn",
                    "text": f"*Relevance:*\n{run.avg_summary_relevance:.2f}/5.0",
                },
                {"type": "mrkdwn", "text": f"*Latency:*\n{run.avg_latency_ms:.0f}ms"},
                {"type": "mrkdwn", "text": f"*Tokens:*\n{run.total_tokens:,}"},
            ],
        },
    ]

    # Add regression details if any
    if comparison and comparison.regressions:
        cases_text = "\n".join(
            f"• `{d.test_case_id}`: {d.old_category.value if d.old_category else '?'} → {d.new_category.value if d.new_category else '?'}"
            for d in comparison.regressions[:5]  # Limit to 5 in Slack
        )
        remaining = len(comparison.regressions) - 5
        if remaining > 0:
            cases_text += f"\n_...and {remaining} more_"

        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Regressed cases:*\n{cases_text}",
            },
        })

    # Add report link
    if report_url:
        blocks.append({
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "📊 View Full Report"},
                    "url": report_url,
                    "style": "primary",
                }
            ],
        })

    blocks.append({"type": "divider"})

    return {
        "blocks": blocks,
        "_status": status,
        "_headline": headline,
    }
