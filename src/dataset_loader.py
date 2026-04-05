"""
Golden dataset loader.

Reads versioned JSON test case files, validates them against Pydantic models,
and provides utilities for filtering and inspecting the dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.models import GoldenDataset, DifficultyTag, EmailCategory

DATASET_DIR = Path(__file__).parent.parent / "golden_dataset"


def load_dataset(
    version: str = "v1",
    dataset_dir: Path = DATASET_DIR,
) -> GoldenDataset:
    """Load and validate a golden dataset by version."""
    filepath = dataset_dir / f"dataset_{version}.json"
    if not filepath.exists():
        available = [f.stem for f in dataset_dir.glob("dataset_*.json")]
        raise FileNotFoundError(
            f"Dataset '{version}' not found at {filepath}. Available: {available}"
        )

    with open(filepath) as f:
        raw = json.load(f)

    dataset = GoldenDataset(**raw)

    # Validate no duplicate IDs
    ids = [c.id for c in dataset.cases]
    dupes = [x for x in ids if ids.count(x) > 1]
    if dupes:
        raise ValueError(f"Duplicate test case IDs: {set(dupes)}")

    return dataset


def dataset_stats(dataset: GoldenDataset) -> dict:
    """Return summary statistics about the dataset."""
    by_category: dict[str, int] = {}
    by_difficulty: dict[str, int] = {}

    for case in dataset.cases:
        cat = case.expected_category.value
        diff = case.difficulty.value
        by_category[cat] = by_category.get(cat, 0) + 1
        by_difficulty[diff] = by_difficulty.get(diff, 0) + 1

    return {
        "version": dataset.version,
        "total_cases": len(dataset.cases),
        "by_category": by_category,
        "by_difficulty": by_difficulty,
    }


def filter_cases(
    dataset: GoldenDataset,
    category: EmailCategory | None = None,
    difficulty: DifficultyTag | None = None,
) -> GoldenDataset:
    """Return a filtered copy of the dataset."""
    filtered = [
        c for c in dataset.cases
        if (category is None or c.expected_category == category)
        and (difficulty is None or c.difficulty == difficulty)
    ]
    return GoldenDataset(
        version=dataset.version,
        created_at=dataset.created_at,
        description=dataset.description,
        cases=filtered,
    )
