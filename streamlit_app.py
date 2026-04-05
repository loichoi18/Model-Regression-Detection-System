"""
Streamlit dashboard for the Model Regression Detection System.

Run with: streamlit run streamlit_app.py

Tabs (top navigation):
  1. Overview   — latest run scorecard + accuracy trend
  2. Compare    — side-by-side diff of any two runs
  3. Run Detail — per-case results for a single run
  4. Dataset    — browse and filter the golden dataset
"""

import sys
sys.path.insert(0, ".")

import streamlit as st
import pandas as pd
from pathlib import Path

from src.run_store import RunStore
from src.dataset_loader import load_dataset, dataset_stats
from src.prompt_loader import list_prompt_versions
from src.diff_engine import compare_runs, DiffConfig
from src.drift import detect_drift, DriftConfig
from src.scoring import accuracy_by_category
from src.models import RegressionSeverity

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Model Regression Detector",
    page_icon="🔍",
    layout="wide",
)

DB_PATH = Path("runs/eval_history.db")


@st.cache_resource
def get_store() -> RunStore:
    return RunStore(DB_PATH)


def _cat_accuracy(run, cat_name):
    """Helper to get accuracy for a single category."""
    by_cat = accuracy_by_category(run)
    for k, v in by_cat.items():
        if (hasattr(k, "value") and k.value == cat_name) or k == cat_name:
            return v
    return {}


# ---------------------------------------------------------------------------
# Top navigation tabs
# ---------------------------------------------------------------------------

st.title("🔍 Model Regression Detector")

store = get_store()

tab_overview, tab_compare, tab_detail, tab_dataset = st.tabs([
    "📊 Overview",
    "🔀 Compare Runs",
    "🔎 Run Detail",
    "📋 Golden Dataset",
])


# ---------------------------------------------------------------------------
# Tab: Overview
# ---------------------------------------------------------------------------

with tab_overview:

    runs = store.list_runs(limit=20)

    if not runs:
        st.warning(
            "No eval runs found. Run the pipeline first:\n\n"
            "```bash\npython run_eval.py --prompt v1\n```"
        )
    else:
        latest = store.get_run(runs[0].run_id)

        # Scorecard
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{latest.overall_accuracy:.1%}")
        col2.metric("Relevance", f"{latest.avg_summary_relevance:.2f}/5.0")
        col3.metric("Avg Latency", f"{latest.avg_latency_ms:.0f}ms")
        col4.metric("Total Tokens", f"{latest.total_tokens:,}")

        st.caption(
            f"Run `{latest.run_id}` · Prompt `{latest.prompt_version}` · "
            f"Model `{latest.model}` · {latest.timestamp.strftime('%Y-%m-%d %H:%M UTC')}"
        )

        # Per-category accuracy
        st.subheader("Per-Category Accuracy")
        cat_cols = st.columns(4)
        for i, cat_name in enumerate(["billing", "technical", "account", "general"]):
            data = _cat_accuracy(latest, cat_name)
            rate = data.get("rate", 0) * 100
            total = int(data.get("total", 0))
            cat_cols[i].metric(
                cat_name.title(),
                f"{rate:.0f}%",
                f"{total} cases",
            )

        # Trend chart
        st.subheader("Accuracy Trend")
        trend_data = store.get_accuracy_history(limit=30)

        if len(trend_data) >= 2:
            df_trend = pd.DataFrame(trend_data)
            st.line_chart(df_trend, x="prompt_version", y="accuracy", height=300)
        else:
            st.info("Need at least 2 runs to show trend.")

        # Drift check
        st.subheader("Drift Status")
        drift = detect_drift(trend_data, DriftConfig())
        if drift.has_drift:
            st.error(f"⚠️ Drift detected ({drift.severity.value})")
            for msg in drift.messages:
                st.write(f"- {msg}")
        else:
            st.success("✅ No drift detected")
            for msg in drift.messages:
                st.write(msg)

        # Run history table
        st.subheader("Recent Runs")
        run_rows = [
            {
                "Run ID": r.run_id,
                "Prompt": r.prompt_version,
                "Accuracy": f"{r.overall_accuracy:.1%}",
                "Relevance": f"{r.avg_summary_relevance:.2f}",
                "Latency": f"{r.avg_latency_ms:.0f}ms",
                "Timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M"),
            }
            for r in runs
        ]
        st.dataframe(
            pd.DataFrame(run_rows), use_container_width=True, hide_index=True
        )


# ---------------------------------------------------------------------------
# Tab: Compare Runs
# ---------------------------------------------------------------------------

with tab_compare:

    runs = store.list_runs(limit=20)
    if len(runs) < 2:
        st.warning("Need at least 2 runs to compare. Run the pipeline with different prompts first.")
    else:
        run_options = {f"{r.run_id} ({r.prompt_version})": r.run_id for r in runs}
        ids = list(run_options.keys())

        col1, col2 = st.columns(2)
        with col1:
            baseline_label = st.selectbox("Baseline", ids, index=1, key="cmp_baseline")
        with col2:
            new_label = st.selectbox("New Run", ids, index=0, key="cmp_new")

        baseline_id = run_options[baseline_label]
        new_id = run_options[new_label]

        if baseline_id == new_id:
            st.warning("Select two different runs to compare.")
        else:
            baseline_run = store.get_run(baseline_id)
            new_run = store.get_run(new_id)
            comparison = compare_runs(baseline_run, new_run, DiffConfig())

            # Severity badge
            severity_icons = {
                RegressionSeverity.OK: "🟢",
                RegressionSeverity.WARNING: "🟡",
                RegressionSeverity.CRITICAL: "🔴",
            }
            st.subheader(
                f"{severity_icons[comparison.severity]} Status: "
                f"{comparison.severity.value.upper()}"
            )

            # Delta metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric(
                "Accuracy",
                f"{new_run.overall_accuracy:.1%}",
                f"{comparison.accuracy_delta:+.1%}",
            )
            col2.metric(
                "Relevance",
                f"{new_run.avg_summary_relevance:.2f}",
                f"{comparison.relevance_delta:+.2f}",
            )
            col3.metric(
                "Latency",
                f"{new_run.avg_latency_ms:.0f}ms",
                f"{comparison.latency_delta_ms:+.0f}ms",
                delta_color="inverse",
            )
            col4.metric("Regressions", len(comparison.regressions))
            col5.metric("Improvements", len(comparison.improvements))

            # Regressions
            if comparison.regressions:
                st.subheader(f"🔴 Regressions ({len(comparison.regressions)})")
                reg_rows = [
                    {
                        "Case ID": d.test_case_id,
                        "Old Prediction": d.old_category.value if d.old_category else "—",
                        "New Prediction": d.new_category.value if d.new_category else "—",
                        "Old Summary": (d.old_summary or "—")[:80],
                        "New Summary": (d.new_summary or "—")[:80],
                    }
                    for d in comparison.regressions
                ]
                st.dataframe(
                    pd.DataFrame(reg_rows),
                    use_container_width=True,
                    hide_index=True,
                )

            # Improvements
            if comparison.improvements:
                st.subheader(f"🟢 Improvements ({len(comparison.improvements)})")
                imp_rows = [
                    {
                        "Case ID": d.test_case_id,
                        "Old Prediction": d.old_category.value if d.old_category else "—",
                        "New Prediction": d.new_category.value if d.new_category else "—",
                    }
                    for d in comparison.improvements
                ]
                st.dataframe(
                    pd.DataFrame(imp_rows),
                    use_container_width=True,
                    hide_index=True,
                )

            # Per-category comparison
            st.subheader("Category Breakdown")
            cat_rows = []
            for cat_name in ["billing", "technical", "account", "general"]:
                base_data = _cat_accuracy(baseline_run, cat_name)
                new_data = _cat_accuracy(new_run, cat_name)

                base_rate = base_data.get("rate", 0)
                new_rate = new_data.get("rate", 0)

                cat_rows.append({
                    "Category": cat_name.title(),
                    "Baseline": f"{base_rate:.0%}" if base_data else "—",
                    "New": f"{new_rate:.0%}" if new_data else "—",
                    "Delta": f"{(new_rate - base_rate):+.0%}" if base_data and new_data else "—",
                })
            st.dataframe(
                pd.DataFrame(cat_rows),
                use_container_width=True,
                hide_index=True,
            )


# ---------------------------------------------------------------------------
# Tab: Run Detail
# ---------------------------------------------------------------------------

with tab_detail:

    runs = store.list_runs(limit=20)
    if not runs:
        st.warning("No runs found.")
    else:
        run_options = {f"{r.run_id} ({r.prompt_version})": r.run_id for r in runs}
        selected_label = st.selectbox(
            "Select Run", list(run_options.keys()), key="detail_run"
        )
        selected_id = run_options[selected_label]
        run = store.get_run(selected_id)

        # Scorecard
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{run.overall_accuracy:.1%}")
        col2.metric("Relevance", f"{run.avg_summary_relevance:.2f}/5.0")
        col3.metric("Avg Latency", f"{run.avg_latency_ms:.0f}ms")
        col4.metric("Total Tokens", f"{run.total_tokens:,}")

        # Filters
        st.subheader("Case Results")
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            show_only = st.selectbox(
                "Filter",
                ["All", "Correct Only", "Incorrect Only", "Errors Only"],
                key="detail_filter",
            )
        with filter_col2:
            cat_filter = st.selectbox(
                "Category",
                ["All", "billing", "technical", "account", "general"],
                key="detail_cat",
            )

        results = run.results
        if show_only == "Correct Only":
            results = [r for r in results if r.category_match]
        elif show_only == "Incorrect Only":
            results = [r for r in results if not r.category_match and not r.error]
        elif show_only == "Errors Only":
            results = [r for r in results if r.error]

        if cat_filter != "All":
            results = [r for r in results if r.expected_category.value == cat_filter]

        # Results table
        rows = [
            {
                "Case ID": r.test_case_id,
                "Match": "✅" if r.category_match else ("❌" if not r.error else "⚠️"),
                "Expected": r.expected_category.value,
                "Predicted": r.predicted_category.value if r.predicted_category else "—",
                "Relevance": f"{r.summary_relevance_score:.1f}",
                "Latency": f"{r.latency_ms:.0f}ms",
                "Email": (
                    r.input_email[:60] + "..."
                    if len(r.input_email) > 60
                    else r.input_email
                ),
            }
            for r in results
        ]
        st.dataframe(
            pd.DataFrame(rows), use_container_width=True, hide_index=True
        )

        # Expandable detail for each case
        st.subheader("Case Details")
        for r in results:
            icon = "✅" if r.category_match else "❌"
            with st.expander(f"{icon} {r.test_case_id} — {r.expected_category.value}"):
                st.markdown(f"**Input email:**\n> {r.input_email}")
                st.markdown(
                    f"**Expected:** `{r.expected_category.value}` · "
                    f"**Predicted:** `{r.predicted_category.value if r.predicted_category else 'N/A'}`"
                )
                st.markdown(f"**Expected summary:** {r.expected_summary}")
                st.markdown(f"**Predicted summary:** {r.predicted_summary or 'N/A'}")
                st.markdown(
                    f"**Relevance:** {r.summary_relevance_score:.1f}/5.0 · "
                    f"**Latency:** {r.latency_ms:.0f}ms · "
                    f"**Tokens:** {r.tokens_used}"
                )
                if r.error:
                    st.error(f"Error: {r.error}")


# ---------------------------------------------------------------------------
# Tab: Golden Dataset
# ---------------------------------------------------------------------------

with tab_dataset:

    dataset = load_dataset("v1")
    stats = dataset_stats(dataset)

    st.metric("Total Cases", stats["total_cases"])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("By Category")
        st.bar_chart(pd.Series(stats["by_category"]))
    with col2:
        st.subheader("By Difficulty")
        st.bar_chart(pd.Series(stats["by_difficulty"]))

    # Filters
    st.subheader("Browse Cases")
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        cat_filter = st.selectbox(
            "Category",
            ["All", "billing", "technical", "account", "general"],
            key="ds_cat",
        )
    with filter_col2:
        diff_filter = st.selectbox(
            "Difficulty",
            ["All", "easy", "medium", "hard", "edge_case"],
            key="ds_diff",
        )

    cases = dataset.cases
    if cat_filter != "All":
        cases = [c for c in cases if c.expected_category.value == cat_filter]
    if diff_filter != "All":
        cases = [c for c in cases if c.difficulty.value == diff_filter]

    for case in cases:
        with st.expander(
            f"[{case.difficulty.value.upper()}] {case.id} — {case.expected_category.value}"
        ):
            st.markdown(f"**Email:**\n> {case.input_email}")
            st.markdown(f"**Expected category:** `{case.expected_category.value}`")
            st.markdown(f"**Expected summary:** {case.expected_summary}")
            if case.notes:
                st.info(f"📝 {case.notes}")
