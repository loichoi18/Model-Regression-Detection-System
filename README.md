# Model Regression Detection System

CI/CD pipeline that continuously evaluates LLM-powered features against a golden dataset, detects quality regressions on prompt or model changes, and alerts the team before degraded outputs reach users.

## Architecture

```
prompts/*.yaml          Versioned prompt configurations (the "code" under test)
golden_dataset/         Human-verified test cases with expected outputs
src/
  models.py             Pydantic type contracts for the entire pipeline
  prompt_loader.py      Loads and validates versioned YAML prompts
  classifier.py         The LLM feature under test (email classifier)
  mock_llm.py           Deterministic mock provider (swappable with real API)
  dataset_loader.py     Golden dataset loading and filtering
  eval_runner.py        Async test runner with concurrency control
  scoring.py            Multi-dimensional scoring (accuracy, relevance, latency)
  diff_engine.py        Run comparison and regression detection
  run_store.py          SQLite persistence for eval history
  reporter.py           Self-contained HTML diff reports
  slack_alert.py        Slack webhook integration
  drift.py              Rolling-average drift detection
  deepeval_adapter.py   DeepEval integration (custom metrics, dataset conversion)
test_email_classifier.py  DeepEval pytest tests — assert_test() on every golden
conftest.py             Pytest config with DeepEval hyperparameter logging
run_eval.py             CLI entry point — orchestrates the full pipeline
streamlit_app.py        Dashboard for browsing runs and comparisons
```

The system has two evaluation layers that work together:

1. **DeepEval layer** (`test_email_classifier.py`): Pytest-style unit tests using `assert_test()` with custom metrics. Runs via `deepeval test run` in CI. Fails individual test cases — great for per-case visibility.

2. **Custom pipeline** (`run_eval.py`): Run-to-run comparison, drift detection, HTML reports, and Slack alerts. DeepEval doesn't do historical diff analysis — this is what makes our system production-grade.

## Quick Start

```bash
# Install
pip install -r requirements.txt

# ── DeepEval tests (pytest-style, per-case pass/fail) ──
deepeval test run test_email_classifier.py                          # v1: 100/100
EVAL_PROMPT_VERSION=v3 deepeval test run test_email_classifier.py   # v3:  99/100 (1 subtle regression)
EVAL_PROMPT_VERSION=v2 deepeval test run test_email_classifier.py   # v2:  95/100 (5 regressions)

# ── Custom pipeline (diff analysis, drift, reports) ──
python run_eval.py --prompt v1                    # Baseline: 100%
python run_eval.py --prompt v3                    # Subtle: 99%, 1 regression detected
python run_eval.py --prompt v2                    # Aggressive: 95%, CRITICAL

# ── Dashboard ──
streamlit run streamlit_app.py

# ── Full test suite ──
python tests/test_e2e.py                          # 79 assertions
```

## Adding Test Cases

Edit `golden_dataset/dataset_v1.json`. Each case needs:

- `id`: Stable identifier (e.g., `bil_013`). Never reuse deleted IDs.
- `input_email`: The raw email text. Write realistic content — don't generate with an LLM.
- `expected_category`: One of `billing`, `technical`, `account`, `general`.
- `expected_summary`: The ideal one-sentence summary. This is ground truth.
- `difficulty`: One of `easy`, `medium`, `hard`, `edge_case`.
- `notes`: Why this case matters. What does it test?

**When to add cases:** After every production incident where the classifier got something wrong, add the real email (anonymized) to the dataset. The eval bar should rise over time.

## Adjusting Thresholds

Pass as CLI flags or set in CI:

| Flag | Default | What it does |
|------|---------|-------------|
| `--warn-threshold` | `0.03` | Accuracy drop to trigger WARNING |
| `--critical-threshold` | `0.08` | Accuracy drop to trigger CRITICAL (blocks merge) |

Drift detection thresholds are in `src/drift.py` (`DriftConfig`). Adjust `accuracy_floor` if your baseline accuracy shifts after dataset changes.

## Swapping to a Real LLM Provider

The system uses a `LLMProvider` protocol. To use OpenAI:

1. Create a class implementing `async def classify(self, prompt_config, email_text) -> tuple[ClassificationResult, float, int]`
2. Pass it to `EmailClassifier(provider=YourProvider())`
3. The mock and real provider are interchangeable — no other code changes needed.

## Slack Setup

1. Create an [Incoming Webhook](https://api.slack.com/messaging/webhooks) in your Slack workspace.
2. Set `SLACK_WEBHOOK_URL` as an environment variable (or GitHub secret for CI).
3. Alerts fire automatically on WARNING and CRITICAL runs.

## Design Decisions

**Why DeepEval + custom pipeline instead of one or the other?**
DeepEval gives us pytest-style test assertions, per-case pass/fail visibility, and a battle-tested CI/CD runner. But it doesn't do run-to-run comparison, historical drift detection, or custom HTML diff reports. Our custom pipeline fills those gaps. The two layers complement each other: DeepEval catches regressions at the test-case level, our pipeline catches them at the aggregate level across runs.

**Why custom DeepEval metrics instead of built-in G-Eval?**
G-Eval requires an API key and costs money per eval. Our custom `CategoryCorrectnessMetric` is deterministic and free — ideal for CI that runs on every PR. When `OPENAI_API_KEY` is set, the system can use G-Eval for deeper scoring in nightly runs. The adapter (`src/deepeval_adapter.py`) makes both modes seamless.

**Why hand-curated test cases, not LLM-generated?**
Evaluation quality is bounded by data quality. LLM-generated test cases inherit the biases and failure modes of the generator — they can't test for blind spots they share. The dataset grows organically from real failures.

**Why track drift separately from per-run regressions?**
A prompt can pass every individual eval but still degrade slowly. If accuracy drops 0.5% across 10 runs, no single run triggers an alert. The drift detector watches the rolling average and catches cumulative degradation that per-run checks miss.

**Why SQLite instead of Postgres?**
This is an append-only audit log with single-writer access (the CI runner). SQLite is zero-config, portable, and fast enough for thousands of runs. The database file ships with the repo for local development. If you need multi-writer access, swap `RunStore` for a Postgres-backed implementation.

**Why mock the LLM instead of always calling the real API?**
CI runs on every PR. At 58 test cases per run, that's 58 API calls per PR. With multiple PRs per day, costs and latency add up fast. The mock produces deterministic, version-aware outputs that exercise the full pipeline. Real API runs happen on a schedule (nightly) or on-demand.

## Docker

```bash
# Build
docker build -t model-eval .

# Run
docker run --rm \
  -e SLACK_WEBHOOK_URL="https://hooks.slack.com/..." \
  -v $(pwd)/runs:/app/runs \
  model-eval --prompt v2 --ci
```

Mount `/app/runs` to persist eval history across container runs.
