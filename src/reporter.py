"""
HTML report generator.

Produces a self-contained HTML file (no external dependencies) that shows:
  1. Run metadata and scorecard
  2. Regression/improvement table with side-by-side diffs
  3. Accuracy trend chart (inline SVG)
  4. Per-category breakdown

The report is designed to be:
  - Viewable in any browser (zero JS dependencies)
  - Attachable to GitHub PR comments
  - Hostable as a static file
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.models import (
    CaseDiff,
    EvalRun,
    RegressionSeverity,
    RunComparison,
)
from src.scoring import accuracy_by_category


def generate_html_report(
    new_run: EvalRun,
    comparison: RunComparison | None = None,
    baseline_run: EvalRun | None = None,
    trend_data: list[dict] | None = None,
    output_path: Path | None = None,
) -> str:
    """
    Generate a self-contained HTML diff report.

    Args:
        new_run: The current eval run
        comparison: Optional comparison against a baseline
        baseline_run: The baseline run (for per-category breakdown)
        trend_data: Historical accuracy data for trend chart
        output_path: If provided, write the HTML to this file

    Returns:
        The HTML string
    """
    severity_color = {
        RegressionSeverity.OK: "#22c55e",
        RegressionSeverity.WARNING: "#f59e0b",
        RegressionSeverity.CRITICAL: "#ef4444",
    }

    severity = comparison.severity if comparison else RegressionSeverity.OK
    status_color = severity_color[severity]
    status_label = severity.value.upper() if comparison else "BASELINE"

    # Build per-category stats
    new_by_cat = accuracy_by_category(new_run)
    base_by_cat = accuracy_by_category(baseline_run) if baseline_run else {}

    html = _build_html(
        new_run=new_run,
        comparison=comparison,
        status_label=status_label,
        status_color=status_color,
        new_by_cat=new_by_cat,
        base_by_cat=base_by_cat,
        trend_data=trend_data or [],
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")

    return html


def _build_html(
    new_run: EvalRun,
    comparison: RunComparison | None,
    status_label: str,
    status_color: str,
    new_by_cat: dict,
    base_by_cat: dict,
    trend_data: list[dict],
) -> str:
    """Assemble the full HTML document."""

    # Scorecard values
    accuracy_pct = f"{new_run.overall_accuracy * 100:.1f}%"
    relevance = f"{new_run.avg_summary_relevance:.2f}"
    latency = f"{new_run.avg_latency_ms:.0f}ms"
    tokens = f"{new_run.total_tokens:,}"

    delta_acc = ""
    delta_rel = ""
    delta_lat = ""
    if comparison:
        da = comparison.accuracy_delta * 100
        delta_acc = f'<span class="delta {"neg" if da < 0 else "pos"}">{da:+.1f}%</span>'
        dr = comparison.relevance_delta
        delta_rel = f'<span class="delta {"neg" if dr < 0 else "pos"}">{dr:+.2f}</span>'
        dl = comparison.latency_delta_ms
        delta_lat = f'<span class="delta {"neg" if dl > 0 else "pos"}">{dl:+.0f}ms</span>'

    # Regressions table
    regression_rows = ""
    if comparison and comparison.regressions:
        for diff in comparison.regressions:
            regression_rows += _diff_row(diff, "regression")

    improvement_rows = ""
    if comparison and comparison.improvements:
        for diff in comparison.improvements:
            improvement_rows += _diff_row(diff, "improvement")

    # Category breakdown
    category_rows = ""
    for cat_enum in ["billing", "technical", "account", "general"]:
        new_data = new_by_cat.get(cat_enum, {}) if isinstance(new_by_cat, dict) else {}
        # Handle EmailCategory enum keys
        for k, v in new_by_cat.items():
            if hasattr(k, 'value') and k.value == cat_enum:
                new_data = v
                break
            elif k == cat_enum:
                new_data = v
                break

        base_data = {}
        for k, v in base_by_cat.items():
            if hasattr(k, 'value') and k.value == cat_enum:
                base_data = v
                break
            elif k == cat_enum:
                base_data = v
                break

        new_rate = new_data.get("rate", 0) * 100 if new_data else 0
        base_rate = base_data.get("rate", 0) * 100 if base_data else 0
        new_total = int(new_data.get("total", 0)) if new_data else 0
        delta = new_rate - base_rate if base_data else 0

        delta_html = ""
        if base_data:
            css = "neg" if delta < 0 else "pos"
            delta_html = f'<span class="delta {css}">{delta:+.1f}%</span>'

        category_rows += f"""
        <tr>
            <td><strong>{cat_enum.title()}</strong></td>
            <td>{new_rate:.1f}% ({new_total} cases)</td>
            <td>{delta_html}</td>
        </tr>"""

    # Trend chart SVG
    trend_svg = _build_trend_svg(trend_data) if trend_data else ""

    # Count totals for summary
    total_cases = len(new_run.results)
    errors = sum(1 for r in new_run.results if r.error)
    passed = sum(1 for r in new_run.results if r.category_match)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Eval Report — {new_run.prompt_version} | {new_run.run_id}</title>
<style>
  :root {{
    --bg: #0f172a;
    --surface: #1e293b;
    --surface2: #334155;
    --text: #f1f5f9;
    --text-muted: #94a3b8;
    --border: #475569;
    --green: #22c55e;
    --red: #ef4444;
    --amber: #f59e0b;
    --blue: #3b82f6;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    background: var(--bg);
    color: var(--text);
    padding: 2rem;
    line-height: 1.6;
  }}
  .container {{ max-width: 1100px; margin: 0 auto; }}
  h1 {{ font-size: 1.5rem; margin-bottom: 0.5rem; }}
  h2 {{
    font-size: 1.1rem;
    color: var(--text-muted);
    margin: 2rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
  }}
  .header {{
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 2rem;
    flex-wrap: wrap;
    gap: 1rem;
  }}
  .status-badge {{
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 4px;
    font-weight: bold;
    font-size: 0.85rem;
    color: #000;
  }}
  .meta {{ color: var(--text-muted); font-size: 0.85rem; }}
  .meta span {{ margin-right: 1.5rem; }}
  .scorecard {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
  }}
  .score-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
  }}
  .score-card .label {{ font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; }}
  .score-card .value {{ font-size: 1.8rem; font-weight: bold; margin: 0.25rem 0; }}
  .delta {{ font-size: 0.8rem; font-weight: 600; }}
  .delta.pos {{ color: var(--green); }}
  .delta.neg {{ color: var(--red); }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 0.5rem 0 1.5rem;
    font-size: 0.85rem;
  }}
  th, td {{
    text-align: left;
    padding: 0.6rem 0.8rem;
    border-bottom: 1px solid var(--border);
  }}
  th {{
    color: var(--text-muted);
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}
  tr:hover {{ background: var(--surface); }}
  .tag {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 3px;
    font-size: 0.75rem;
    font-weight: 600;
  }}
  .tag-regression {{ background: rgba(239,68,68,0.2); color: var(--red); }}
  .tag-improvement {{ background: rgba(34,197,94,0.2); color: var(--green); }}
  .trend-chart {{ margin: 1rem 0; }}
  .summary-bar {{
    display: flex;
    gap: 0.5rem;
    margin: 1rem 0;
    font-size: 0.85rem;
  }}
  .summary-bar .item {{
    padding: 0.3rem 0.8rem;
    border-radius: 4px;
    background: var(--surface);
    border: 1px solid var(--border);
  }}
  .empty-state {{
    color: var(--text-muted);
    font-style: italic;
    padding: 1rem;
    text-align: center;
  }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <div>
      <h1>Eval Report</h1>
      <div class="meta">
        <span>Run: {new_run.run_id}</span>
        <span>Prompt: {new_run.prompt_version}</span>
        <span>Model: {new_run.model}</span>
        <span>{new_run.timestamp.strftime("%Y-%m-%d %H:%M UTC")}</span>
      </div>
    </div>
    <div>
      <span class="status-badge" style="background:{status_color}">{status_label}</span>
    </div>
  </div>

  <div class="summary-bar">
    <div class="item">{total_cases} cases</div>
    <div class="item" style="color:var(--green)">{passed} passed</div>
    <div class="item" style="color:var(--red)">{total_cases - passed - errors} failed</div>
    {"<div class='item' style='color:var(--amber)'>" + str(errors) + " errors</div>" if errors else ""}
  </div>

  <div class="scorecard">
    <div class="score-card">
      <div class="label">Accuracy</div>
      <div class="value">{accuracy_pct}</div>
      {delta_acc}
    </div>
    <div class="score-card">
      <div class="label">Avg Relevance (1–5)</div>
      <div class="value">{relevance}</div>
      {delta_rel}
    </div>
    <div class="score-card">
      <div class="label">Avg Latency</div>
      <div class="value">{latency}</div>
      {delta_lat}
    </div>
    <div class="score-card">
      <div class="label">Total Tokens</div>
      <div class="value">{tokens}</div>
    </div>
  </div>

  <h2>Category Breakdown</h2>
  <table>
    <thead><tr><th>Category</th><th>Accuracy</th><th>Delta</th></tr></thead>
    <tbody>{category_rows}</tbody>
  </table>

  <h2>Regressions ({len(comparison.regressions) if comparison else 0})</h2>
  {f'''<table>
    <thead><tr><th>Case ID</th><th>Old Prediction</th><th>New Prediction</th><th>Old Summary</th><th>New Summary</th></tr></thead>
    <tbody>{regression_rows}</tbody>
  </table>''' if regression_rows else '<div class="empty-state">No regressions detected</div>'}

  <h2>Improvements ({len(comparison.improvements) if comparison else 0})</h2>
  {f'''<table>
    <thead><tr><th>Case ID</th><th>Old Prediction</th><th>New Prediction</th><th>Old Summary</th><th>New Summary</th></tr></thead>
    <tbody>{improvement_rows}</tbody>
  </table>''' if improvement_rows else '<div class="empty-state">No improvements detected</div>'}

  {f'<h2>Accuracy Trend</h2><div class="trend-chart">{trend_svg}</div>' if trend_svg else ''}

</div>
</body>
</html>"""


def _diff_row(diff: CaseDiff, kind: str) -> str:
    """Generate a table row for a regression or improvement."""
    tag_class = f"tag-{kind}"
    old_cat = diff.old_category.value if diff.old_category else "—"
    new_cat = diff.new_category.value if diff.new_category else "—"
    old_sum = (diff.old_summary or "—")[:80]
    new_sum = (diff.new_summary or "—")[:80]

    return f"""
    <tr>
      <td><code>{diff.test_case_id}</code></td>
      <td>{old_cat}</td>
      <td><span class="tag {tag_class}">{new_cat}</span></td>
      <td style="font-size:0.8rem;color:var(--text-muted)">{old_sum}</td>
      <td style="font-size:0.8rem">{new_sum}</td>
    </tr>"""


def _build_trend_svg(data: list[dict]) -> str:
    """Build an inline SVG line chart of accuracy over time."""
    if len(data) < 2:
        return ""

    w, h = 700, 200
    pad_left, pad_right, pad_top, pad_bottom = 50, 20, 20, 40

    chart_w = w - pad_left - pad_right
    chart_h = h - pad_top - pad_bottom

    accuracies = [d["accuracy"] for d in data]
    min_a = max(0, min(accuracies) - 0.05)
    max_a = min(1, max(accuracies) + 0.05)
    range_a = max_a - min_a if max_a != min_a else 0.1

    n = len(data)
    points = []
    for i, d in enumerate(data):
        x = pad_left + (i / (n - 1)) * chart_w
        y = pad_top + chart_h - ((d["accuracy"] - min_a) / range_a) * chart_h
        points.append((x, y))

    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    dots = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="#3b82f6"/>'
        for x, y in points
    )

    # Y-axis labels
    y_labels = ""
    for frac in [0, 0.25, 0.5, 0.75, 1.0]:
        val = min_a + frac * range_a
        y_pos = pad_top + chart_h - frac * chart_h
        y_labels += f'<text x="{pad_left - 8}" y="{y_pos + 4}" text-anchor="end" fill="#94a3b8" font-size="11">{val:.0%}</text>'
        y_labels += f'<line x1="{pad_left}" y1="{y_pos}" x2="{w - pad_right}" y2="{y_pos}" stroke="#334155" stroke-dasharray="4"/>'

    # X-axis labels (show prompt version)
    x_labels = ""
    step = max(1, n // 6)
    for i in range(0, n, step):
        x = pad_left + (i / (n - 1)) * chart_w
        label = data[i].get("prompt_version", f"#{i+1}")
        x_labels += f'<text x="{x:.1f}" y="{h - 8}" text-anchor="middle" fill="#94a3b8" font-size="10">{label}</text>'

    return f"""<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:{w}px;">
  {y_labels}
  {x_labels}
  <polyline points="{polyline}" fill="none" stroke="#3b82f6" stroke-width="2"/>
  {dots}
</svg>"""
