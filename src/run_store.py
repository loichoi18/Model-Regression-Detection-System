"""
SQLite-backed storage for eval runs.

Persists EvalRun objects so the pipeline can:
  - Compare any two historical runs
  - Track trends over time
  - Detect slow drift via rolling averages

Schema is intentionally simple: runs + case_results, both with JSON fallbacks
for fields that don't need indexing. This keeps queries fast without over-
normalizing for what is fundamentally an append-only audit log.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.models import CaseResult, EmailCategory, EvalRun

DEFAULT_DB_PATH = Path(__file__).parent.parent / "runs" / "eval_history.db"


class RunStore:
    """Persistent store for evaluation runs."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path), check_same_thread=False
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                prompt_version TEXT NOT NULL,
                model TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                dataset_version TEXT NOT NULL,
                overall_accuracy REAL DEFAULT 0.0,
                avg_summary_relevance REAL DEFAULT 0.0,
                avg_latency_ms REAL DEFAULT 0.0,
                total_tokens INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS case_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL REFERENCES runs(run_id),
                test_case_id TEXT NOT NULL,
                input_email TEXT NOT NULL,
                expected_category TEXT NOT NULL,
                predicted_category TEXT,
                expected_summary TEXT NOT NULL,
                predicted_summary TEXT,
                category_match INTEGER DEFAULT 0,
                summary_relevance_score REAL DEFAULT 0.0,
                latency_ms REAL DEFAULT 0.0,
                tokens_used INTEGER DEFAULT 0,
                error TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_case_results_run
                ON case_results(run_id);

            CREATE INDEX IF NOT EXISTS idx_runs_timestamp
                ON runs(timestamp);

            CREATE INDEX IF NOT EXISTS idx_runs_prompt_version
                ON runs(prompt_version);
            """
        )
        conn.commit()

    # ----- Write operations -----

    def save_run(self, run: EvalRun) -> None:
        """Persist a complete eval run."""
        conn = self._get_conn()

        conn.execute(
            """
            INSERT OR REPLACE INTO runs
                (run_id, prompt_version, model, timestamp, dataset_version,
                 overall_accuracy, avg_summary_relevance, avg_latency_ms, total_tokens)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.run_id,
                run.prompt_version,
                run.model,
                run.timestamp.isoformat(),
                run.dataset_version,
                run.overall_accuracy,
                run.avg_summary_relevance,
                run.avg_latency_ms,
                run.total_tokens,
            ),
        )

        # Delete existing case results for this run (for idempotent saves)
        conn.execute("DELETE FROM case_results WHERE run_id = ?", (run.run_id,))

        for r in run.results:
            conn.execute(
                """
                INSERT INTO case_results
                    (run_id, test_case_id, input_email, expected_category,
                     predicted_category, expected_summary, predicted_summary,
                     category_match, summary_relevance_score, latency_ms,
                     tokens_used, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    r.test_case_id,
                    r.input_email,
                    r.expected_category.value,
                    r.predicted_category.value if r.predicted_category else None,
                    r.expected_summary,
                    r.predicted_summary,
                    int(r.category_match),
                    r.summary_relevance_score,
                    r.latency_ms,
                    r.tokens_used,
                    r.error,
                ),
            )

        conn.commit()

    # ----- Read operations -----

    def get_run(self, run_id: str) -> EvalRun | None:
        """Load a single run by ID."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_run(row)

    def get_latest_run(self, prompt_version: str | None = None) -> EvalRun | None:
        """Get the most recent run, optionally filtered by prompt version."""
        conn = self._get_conn()
        if prompt_version:
            row = conn.execute(
                "SELECT * FROM runs WHERE prompt_version = ? ORDER BY timestamp DESC LIMIT 1",
                (prompt_version,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM runs ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()

        if row is None:
            return None
        return self._row_to_run(row)

    def list_runs(
        self, limit: int = 50, prompt_version: str | None = None
    ) -> list[EvalRun]:
        """List runs in reverse chronological order (metadata only, no case results)."""
        conn = self._get_conn()
        if prompt_version:
            rows = conn.execute(
                "SELECT * FROM runs WHERE prompt_version = ? ORDER BY timestamp DESC LIMIT ?",
                (prompt_version, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()

        return [self._row_to_run(row, include_results=False) for row in rows]

    def get_accuracy_history(self, limit: int = 30) -> list[dict]:
        """Get accuracy trend data for the last N runs."""
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT run_id, prompt_version, timestamp,
                   overall_accuracy, avg_summary_relevance, avg_latency_ms
            FROM runs
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        return [
            {
                "run_id": r["run_id"],
                "prompt_version": r["prompt_version"],
                "timestamp": r["timestamp"],
                "accuracy": r["overall_accuracy"],
                "relevance": r["avg_summary_relevance"],
                "latency_ms": r["avg_latency_ms"],
            }
            for r in reversed(rows)  # Chronological order
        ]

    # ----- Internal helpers -----

    def _row_to_run(self, row: sqlite3.Row, include_results: bool = True) -> EvalRun:
        """Convert a DB row to an EvalRun, optionally loading case results."""
        run = EvalRun(
            run_id=row["run_id"],
            prompt_version=row["prompt_version"],
            model=row["model"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            dataset_version=row["dataset_version"],
            overall_accuracy=row["overall_accuracy"],
            avg_summary_relevance=row["avg_summary_relevance"],
            avg_latency_ms=row["avg_latency_ms"],
            total_tokens=row["total_tokens"],
        )

        if include_results:
            conn = self._get_conn()
            case_rows = conn.execute(
                "SELECT * FROM case_results WHERE run_id = ? ORDER BY test_case_id",
                (run.run_id,),
            ).fetchall()

            run.results = [
                CaseResult(
                    test_case_id=cr["test_case_id"],
                    input_email=cr["input_email"],
                    expected_category=EmailCategory(cr["expected_category"]),
                    predicted_category=(
                        EmailCategory(cr["predicted_category"])
                        if cr["predicted_category"]
                        else None
                    ),
                    expected_summary=cr["expected_summary"],
                    predicted_summary=cr["predicted_summary"],
                    category_match=bool(cr["category_match"]),
                    summary_relevance_score=cr["summary_relevance_score"],
                    latency_ms=cr["latency_ms"],
                    tokens_used=cr["tokens_used"],
                    error=cr["error"],
                )
                for cr in case_rows
            ]

        return run

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
