from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import EvaluationReport, SampleResult


def utcnow() -> str:
    return datetime.now(UTC).isoformat()


class LocalStore:
    """Local SQLite state for runs, caches, candidates, and promotion evidence."""

    def __init__(self, directory: str | Path, database: str = "loopeval.db") -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.runs_dir = self.directory / "runs"
        self.runs_dir.mkdir(exist_ok=True)
        self.db_path = self.directory / database
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._migrate()

    def close(self) -> None:
        self._connection.close()

    def _migrate(self) -> None:
        with self._connection:
            self._connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA foreign_keys=ON;

                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    config_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    sample_count INTEGER NOT NULL DEFAULT 0,
                    escalated_count INTEGER NOT NULL DEFAULT 0,
                    cost_usd REAL,
                    report_json TEXT
                );

                CREATE TABLE IF NOT EXISTS results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                    sample_id TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    escalated INTEGER NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(run_id, sample_id)
                );

                CREATE TABLE IF NOT EXISTS provider_cache (
                    key TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS candidates (
                    id TEXT PRIMARY KEY,
                    check_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    check_json TEXT NOT NULL,
                    evidence_count INTEGER NOT NULL DEFAULT 1,
                    sample_ids_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    review_notes TEXT,
                    validation_json TEXT
                );

                CREATE TABLE IF NOT EXISTS reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_id TEXT NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
                    decision TEXT NOT NULL,
                    notes TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS candidate_evidence (
                    candidate_id TEXT NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
                    sample_id TEXT NOT NULL,
                    evidence TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(candidate_id, sample_id)
                );
                """
            )

    def start_run(self, config_hash: str) -> tuple[str, datetime]:
        run_id = f"run_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
        started = datetime.now(UTC)
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT INTO runs(id, started_at, config_hash, status) VALUES (?, ?, ?, ?)",
                (run_id, started.isoformat(), config_hash, "running"),
            )
        return run_id, started

    def save_result(self, run_id: str, result: SampleResult) -> None:
        payload = result.model_dump_json()
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO results(run_id, sample_id, verdict, escalated, result_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, sample_id) DO UPDATE SET
                  verdict=excluded.verdict,
                  escalated=excluded.escalated,
                  result_json=excluded.result_json,
                  created_at=excluded.created_at
                """,
                (
                    run_id,
                    result.sample_id,
                    result.verdict.value,
                    int(result.escalated),
                    payload,
                    result.created_at.isoformat(),
                ),
            )

    def finish_run(self, report: EvaluationReport) -> Path:
        cost = report.total_cost_usd
        payload = report.model_dump_json(indent=2)
        with self._lock, self._connection:
            self._connection.execute(
                """
                UPDATE runs
                   SET completed_at=?, status='completed', sample_count=?, escalated_count=?,
                       cost_usd=?, report_json=?
                 WHERE id=?
                """,
                (
                    report.completed_at.isoformat(),
                    report.sample_count,
                    sum(r.escalated for r in report.results),
                    cost,
                    payload,
                    report.run_id,
                ),
            )
        artifact = self.runs_dir / f"{report.run_id}.json"
        artifact.write_text(payload + "\n")
        jsonl = self.runs_dir / f"{report.run_id}.jsonl"
        jsonl.write_text("".join(result.model_dump_json() + "\n" for result in report.results))
        return artifact

    def fail_run(self, run_id: str, error: str) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE runs SET completed_at=?, status=?, report_json=? WHERE id=?",
                (utcnow(), "failed", json.dumps({"error": error}), run_id),
            )

    def get_report(self, run_id: str) -> EvaluationReport | None:
        row = self._connection.execute(
            "SELECT report_json FROM runs WHERE id=?", (run_id,)
        ).fetchone()
        if not row or not row["report_json"]:
            return None
        return EvaluationReport.model_validate_json(row["report_json"])

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            """
            SELECT id, started_at, completed_at, status, sample_count, escalated_count, cost_usd
              FROM runs ORDER BY started_at DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def cache_get(self, key: str) -> dict[str, Any] | None:
        row = self._connection.execute(
            "SELECT response_json FROM provider_cache WHERE key=?", (key,)
        ).fetchone()
        return json.loads(row["response_json"]) if row else None

    def cache_put(self, key: str, kind: str, response: dict[str, Any]) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT OR REPLACE INTO provider_cache(key, kind, response_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (key, kind, json.dumps(response, default=str), utcnow()),
            )

    def upsert_candidate(
        self,
        *,
        candidate_id: str,
        check_id: str,
        title: str,
        description: str | None,
        check_json: dict[str, Any],
        sample_id: str,
        evidence: str = "",
        confidence: float = 0.0,
    ) -> None:
        now = utcnow()
        existing = self._connection.execute(
            "SELECT evidence_count, sample_ids_json FROM candidates WHERE id=?", (candidate_id,)
        ).fetchone()
        if existing:
            samples = set(json.loads(existing["sample_ids_json"]))
            samples.add(sample_id)
            with self._lock, self._connection:
                self._connection.execute(
                    """
                    UPDATE candidates
                       SET evidence_count=?, sample_ids_json=?, updated_at=?
                     WHERE id=?
                    """,
                    (
                        len(samples),
                        json.dumps(sorted(samples)),
                        now,
                        candidate_id,
                    ),
                )
                self._save_candidate_evidence(
                    candidate_id, sample_id, evidence, confidence, created_at=now
                )
            return
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO candidates(
                    id, check_id, status, title, description, check_json,
                    evidence_count, sample_ids_json, created_at, updated_at
                ) VALUES (?, ?, 'proposed', ?, ?, ?, 1, ?, ?, ?)
                """,
                (
                    candidate_id,
                    check_id,
                    title,
                    description,
                    json.dumps(check_json),
                    json.dumps([sample_id]),
                    now,
                    now,
                ),
            )
            self._save_candidate_evidence(
                candidate_id, sample_id, evidence, confidence, created_at=now
            )

    def _save_candidate_evidence(
        self,
        candidate_id: str,
        sample_id: str,
        evidence: str,
        confidence: float,
        *,
        created_at: str,
    ) -> None:
        self._connection.execute(
            """
            INSERT OR IGNORE INTO candidate_evidence(
                candidate_id, sample_id, evidence, confidence, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (candidate_id, sample_id, evidence, confidence, created_at),
        )

    def list_candidate_evidence(self, candidate_id: str) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            """
            SELECT sample_id, evidence, confidence, created_at
              FROM candidate_evidence
             WHERE candidate_id=?
             ORDER BY created_at ASC
            """,
            (candidate_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def list_candidates(self, status: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM candidates"
        args: tuple[Any, ...] = ()
        if status:
            sql += " WHERE status=?"
            args = (status,)
        sql += " ORDER BY evidence_count DESC, created_at ASC"
        rows = self._connection.execute(sql, args).fetchall()
        return [self._decode_candidate(dict(row)) for row in rows]

    def get_candidate(self, candidate_id: str) -> dict[str, Any] | None:
        row = self._connection.execute(
            "SELECT * FROM candidates WHERE id=?", (candidate_id,)
        ).fetchone()
        return self._decode_candidate(dict(row)) if row else None

    @staticmethod
    def _decode_candidate(row: dict[str, Any]) -> dict[str, Any]:
        row["check"] = json.loads(row.pop("check_json"))
        row["sample_ids"] = json.loads(row.pop("sample_ids_json"))
        row["validation"] = (
            json.loads(row.pop("validation_json")) if row["validation_json"] else None
        )
        return row

    def review_candidate(self, candidate_id: str, decision: str, notes: str | None) -> None:
        if decision not in {"approve", "reject"}:
            raise ValueError("decision must be approve or reject")
        status = "approved" if decision == "approve" else "rejected"
        with self._lock, self._connection:
            cursor = self._connection.execute(
                "UPDATE candidates SET status=?, review_notes=?, updated_at=? WHERE id=?",
                (status, notes, utcnow(), candidate_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"candidate not found: {candidate_id}")
            self._connection.execute(
                "INSERT INTO reviews(candidate_id, decision, notes, created_at) VALUES (?, ?, ?, ?)",
                (candidate_id, decision, notes, utcnow()),
            )

    def save_validation(self, candidate_id: str, metrics: dict[str, Any], passed: bool) -> None:
        status = "shadow" if passed else "approved"
        with self._lock, self._connection:
            cursor = self._connection.execute(
                "UPDATE candidates SET validation_json=?, status=?, updated_at=? WHERE id=?",
                (json.dumps(metrics), status, utcnow(), candidate_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"candidate not found: {candidate_id}")

    def mark_promoted(self, candidate_id: str) -> None:
        with self._lock, self._connection:
            cursor = self._connection.execute(
                "UPDATE candidates SET status='active', updated_at=? WHERE id=?",
                (utcnow(), candidate_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"candidate not found: {candidate_id}")
