"""
Core data models for the Model Regression Detection System.

Every component in the pipeline consumes and produces these types.
This is the single source of truth for data shapes.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Email Classification Domain
# ---------------------------------------------------------------------------

class EmailCategory(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    GENERAL = "general"


class ClassificationResult(BaseModel):
    """Output of the email classifier LLM feature."""
    category: EmailCategory
    summary: str = Field(..., max_length=300)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Prompt Versioning
# ---------------------------------------------------------------------------

class FewShotExample(BaseModel):
    """A single few-shot example embedded in a prompt config."""
    email: str
    category: EmailCategory
    summary: str


class PromptConfig(BaseModel):
    """
    A versioned prompt configuration.
    Loaded from YAML files in /prompts.
    """
    version: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model: str = "gpt-4o-mini"
    system_prompt: str
    few_shot_examples: list[FewShotExample] = Field(default_factory=list)
    temperature: float = 0.0
    max_tokens: int = 256

    def build_messages(self, email_text: str) -> list[dict[str, str]]:
        """Assemble the full message list for the LLM API call."""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]

        for ex in self.few_shot_examples:
            messages.append({"role": "user", "content": ex.email})
            messages.append({
                "role": "assistant",
                "content": ClassificationResult(
                    category=ex.category,
                    summary=ex.summary,
                ).model_dump_json(),
            })

        messages.append({"role": "user", "content": email_text})
        return messages


# ---------------------------------------------------------------------------
# Golden Dataset
# ---------------------------------------------------------------------------

class DifficultyTag(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EDGE_CASE = "edge_case"


class TestCase(BaseModel):
    """A single entry in the golden evaluation dataset."""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    input_email: str
    expected_category: EmailCategory
    expected_summary: str
    difficulty: DifficultyTag = DifficultyTag.MEDIUM
    notes: str = ""


class GoldenDataset(BaseModel):
    """Versioned collection of test cases."""
    version: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    description: str = ""
    cases: list[TestCase]


# ---------------------------------------------------------------------------
# Eval Run Results
# ---------------------------------------------------------------------------

class CaseResult(BaseModel):
    """Result of running a single test case through the eval pipeline."""
    test_case_id: str
    input_email: str
    expected_category: EmailCategory
    predicted_category: EmailCategory | None = None
    expected_summary: str
    predicted_summary: str | None = None
    category_match: bool = False
    summary_relevance_score: float = 0.0   # 1-5 scale
    latency_ms: float = 0.0
    tokens_used: int = 0
    error: str | None = None


class EvalRun(BaseModel):
    """A complete evaluation run — the unit of comparison."""
    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    prompt_version: str
    model: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    dataset_version: str
    results: list[CaseResult] = Field(default_factory=list)

    # Aggregate metrics (computed after run completes)
    overall_accuracy: float = 0.0
    avg_summary_relevance: float = 0.0
    avg_latency_ms: float = 0.0
    total_tokens: int = 0


# ---------------------------------------------------------------------------
# Diff / Comparison
# ---------------------------------------------------------------------------

class RegressionSeverity(str, Enum):
    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"


class CaseDiff(BaseModel):
    """A single test case that changed between two runs."""
    test_case_id: str
    direction: str  # "regression" or "improvement"
    old_category: EmailCategory | None = None
    new_category: EmailCategory | None = None
    old_summary: str | None = None
    new_summary: str | None = None
    old_category_match: bool = False
    new_category_match: bool = False


class RunComparison(BaseModel):
    """Diff between a baseline run and a new run."""
    baseline_run_id: str
    new_run_id: str
    baseline_prompt_version: str
    new_prompt_version: str
    accuracy_delta: float = 0.0
    relevance_delta: float = 0.0
    latency_delta_ms: float = 0.0
    regressions: list[CaseDiff] = Field(default_factory=list)
    improvements: list[CaseDiff] = Field(default_factory=list)
    severity: RegressionSeverity = RegressionSeverity.OK
