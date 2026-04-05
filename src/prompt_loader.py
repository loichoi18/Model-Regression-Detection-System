"""
Prompt versioning system.

Loads PromptConfig from versioned YAML files in the /prompts directory.
Supports listing available versions and loading specific ones.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from src.models import FewShotExample, PromptConfig

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def list_prompt_versions(prompts_dir: Path = PROMPTS_DIR) -> list[str]:
    """Return sorted list of available prompt version IDs."""
    versions = []
    for f in prompts_dir.glob("*.yaml"):
        with open(f) as fh:
            data = yaml.safe_load(fh)
            versions.append(data["version"])
    return sorted(versions)


def load_prompt(version: str, prompts_dir: Path = PROMPTS_DIR) -> PromptConfig:
    """Load a specific prompt version from YAML."""
    # Try direct filename match first (e.g., v1.yaml)
    candidates = list(prompts_dir.glob("*.yaml"))
    for filepath in candidates:
        with open(filepath) as fh:
            data = yaml.safe_load(fh)
            if data.get("version") == version:
                return _parse_prompt(data)

    available = list_prompt_versions(prompts_dir)
    raise FileNotFoundError(
        f"Prompt version '{version}' not found. Available: {available}"
    )


def load_latest_prompt(prompts_dir: Path = PROMPTS_DIR) -> PromptConfig:
    """Load the most recent prompt version (by version string sort)."""
    versions = list_prompt_versions(prompts_dir)
    if not versions:
        raise FileNotFoundError(f"No prompt files found in {prompts_dir}")
    return load_prompt(versions[-1], prompts_dir)


def _parse_prompt(data: dict) -> PromptConfig:
    """Parse raw YAML dict into a PromptConfig."""
    few_shots = [
        FewShotExample(**ex) for ex in data.get("few_shot_examples", [])
    ]
    return PromptConfig(
        version=data["version"],
        created_at=data.get("created_at"),
        model=data.get("model", "gpt-4o-mini"),
        system_prompt=data["system_prompt"],
        few_shot_examples=few_shots,
        temperature=data.get("temperature", 0.0),
        max_tokens=data.get("max_tokens", 256),
    )
