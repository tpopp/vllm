from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


@dataclass
class ChangeRecord:
    commit_sha: str
    commit_date: str | None
    author: str | None
    commit_subject: str | None
    pr_number: int | None
    pr_url: str | None
    pr_title: str | None
    summary: str | None
    is_breaking_api: bool
    categories: list[str]
    rocm_relevance: str
    changed_files: list[str]
    heuristic_score: float
    action_hint: str | None
    model_used: str | None
    analysis_backend: str | None
    analysis_cached_at: str | None
    model_impacts: list[dict[str, str]]


class Database:
    def __init__(self, db_path: Path, schema_path: Path) -> None:
        self.db_path = db_path
        self.schema_path = schema_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        schema = self.schema_path.read_text(encoding="utf-8")
        self._conn.executescript(schema)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def start_sync_run(
        self,
        upstream_sha_before: str | None,
        upstream_sha_after: str | None,
    ) -> int:
        cur = self._conn.execute(
            """
            INSERT INTO sync_runs (
                run_at, upstream_sha_before, upstream_sha_after, status
            ) VALUES (?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                upstream_sha_before,
                upstream_sha_after,
                "running",
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def finish_sync_run(
        self,
        run_id: int,
        *,
        commits_count: int,
        fork_push_success: bool,
        status: str,
        message: str | None = None,
    ) -> None:
        self._conn.execute(
            """
            UPDATE sync_runs
            SET commits_count = ?, fork_push_success = ?, status = ?, message = ?
            WHERE id = ?
            """,
            (
                commits_count,
                int(fork_push_success),
                status,
                message,
                run_id,
            ),
        )
        self._conn.commit()

    def has_commit(self, commit_sha: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM upstream_changes WHERE commit_sha = ?",
            (commit_sha,),
        ).fetchone()
        return row is not None

    def upsert_change(self, run_id: int | None, record: ChangeRecord) -> int:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT INTO upstream_changes (
                sync_run_id, commit_sha, commit_date, author, commit_subject,
                pr_number, pr_url, pr_title, summary, is_breaking_api,
                categories, rocm_relevance, changed_files, heuristic_score,
                action_hint, model_used, analysis_backend, analysis_cached_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(commit_sha) DO UPDATE SET
                sync_run_id = excluded.sync_run_id,
                commit_date = excluded.commit_date,
                author = excluded.author,
                commit_subject = excluded.commit_subject,
                pr_number = excluded.pr_number,
                pr_url = excluded.pr_url,
                pr_title = excluded.pr_title,
                summary = excluded.summary,
                is_breaking_api = excluded.is_breaking_api,
                categories = excluded.categories,
                rocm_relevance = excluded.rocm_relevance,
                changed_files = excluded.changed_files,
                heuristic_score = excluded.heuristic_score,
                action_hint = excluded.action_hint,
                model_used = excluded.model_used,
                analysis_backend = excluded.analysis_backend,
                analysis_cached_at = excluded.analysis_cached_at
            """,
            (
                run_id,
                record.commit_sha,
                record.commit_date,
                record.author,
                record.commit_subject,
                record.pr_number,
                record.pr_url,
                record.pr_title,
                record.summary,
                int(record.is_breaking_api),
                json.dumps(record.categories),
                record.rocm_relevance,
                json.dumps(record.changed_files),
                record.heuristic_score,
                record.action_hint,
                record.model_used,
                record.analysis_backend,
                record.analysis_cached_at or now,
            ),
        )
        row = self._conn.execute(
            "SELECT id FROM upstream_changes WHERE commit_sha = ?",
            (record.commit_sha,),
        ).fetchone()
        change_id = int(row["id"])
        self._conn.execute(
            "DELETE FROM model_impacts WHERE change_id = ?",
            (change_id,),
        )
        for impact in record.model_impacts:
            self._conn.execute(
                """
                INSERT INTO model_impacts (
                    change_id, architecture, module_name, impact_level, rationale
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    change_id,
                    impact.get("architecture", ""),
                    impact.get("module_name"),
                    impact.get("impact", impact.get("impact_level", "none")),
                    impact.get("rationale"),
                ),
            )
        self._conn.commit()
        return change_id

    def query_changes(
        self,
        *,
        model: str | None = None,
        category: str | None = None,
        breaking: bool = False,
        since_days: int | None = None,
    ) -> list[dict[str, Any]]:
        clauses = ["1=1"]
        params: list[Any] = []
        if breaking:
            clauses.append("c.is_breaking_api = 1")
        if category:
            clauses.append("c.categories LIKE ?")
            params.append(f"%{category}%")
        if since_days is not None:
            cutoff = (
                datetime.now(timezone.utc) - timedelta(days=since_days)
            ).isoformat()
            clauses.append("c.commit_date >= ?")
            params.append(cutoff)
        join = ""
        if model:
            join = "JOIN model_impacts m ON m.change_id = c.id"
            clauses.append("m.architecture = ?")
            clauses.append("m.impact_level IN ('definite', 'possible')")
            params.append(model)
        sql = f"""
            SELECT DISTINCT c.*
            FROM upstream_changes c
            {join}
            WHERE {' AND '.join(clauses)}
            ORDER BY c.commit_date DESC
        """
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def export_jsonl(
        self,
        *,
        model: str | None = None,
        since_days: int | None = None,
    ) -> list[dict[str, Any]]:
        changes = self.query_changes(model=model, since_days=since_days)
        results: list[dict[str, Any]] = []
        for change in changes:
            impacts = self._conn.execute(
                """
                SELECT architecture, module_name, impact_level, rationale
                FROM model_impacts WHERE change_id = (
                    SELECT id FROM upstream_changes WHERE commit_sha = ?
                )
                """,
                (change["commit_sha"],),
            ).fetchall()
            item = dict(change)
            item["categories"] = json.loads(change["categories"] or "[]")
            item["changed_files"] = json.loads(change["changed_files"] or "[]")
            item["model_impacts"] = [dict(r) for r in impacts]
            results.append(item)
        return results
