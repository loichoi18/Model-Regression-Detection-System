"""
Pytest configuration for DeepEval integration.

Provides:
  - Hyperparameter logging (prompt version, model, dataset)
  - Shared fixtures for classifier and dataset
  - Custom markers for test filtering by difficulty
"""

import os
import sys

import deepeval
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.prompt_loader import load_prompt


# ---------------------------------------------------------------------------
# DeepEval hyperparameter logging
# ---------------------------------------------------------------------------

PROMPT_VERSION = os.environ.get("EVAL_PROMPT_VERSION", "v1")

_config = load_prompt(PROMPT_VERSION)


@deepeval.log_hyperparameters
def hyperparameters():
    """
    Log eval hyperparameters for tracking in Confident AI (if configured).
    This function is auto-discovered by deepeval.
    """
    return {
        "prompt_version": _config.version,
        "model": _config.model,
        "temperature": _config.temperature,
        "max_tokens": _config.max_tokens,
        "few_shot_count": len(_config.few_shot_examples),
        "system_prompt_length": len(_config.system_prompt),
        "prompt_template": _config.system_prompt[:200],
    }


# ---------------------------------------------------------------------------
# Custom markers
# ---------------------------------------------------------------------------

def pytest_configure(config):
    """Register custom markers for test filtering."""
    config.addinivalue_line(
        "markers", "edge_case: marks tests as edge cases (deselect with '-m \"not edge_case\"')"
    )
    config.addinivalue_line(
        "markers", "billing: marks tests for billing category"
    )
    config.addinivalue_line(
        "markers", "technical: marks tests for technical category"
    )
    config.addinivalue_line(
        "markers", "account: marks tests for account category"
    )
    config.addinivalue_line(
        "markers", "general: marks tests for general category"
    )
