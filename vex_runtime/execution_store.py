"""One project-local execution ledger for Studio runs and CLI jobs.

Records are committed to the same SQLite catalog as project revisions. The
ledger tracks status and ownership; media effects remain the responsibility of
the tool's project mutation contract.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Any

from vex_runtime.project_catalog import CATALOG_SCHEMA_VERSION, ProjectCatalogError, catalog_path


ACTIVE_STATUSES = frozenset({"queued", "running"})


class ExecutionConflict(ProjectCatalogError):
    """Another execution owns an exclusive project run slot."""


class ExecutionStore:
    def put(
        self,
        working_dir: str | Path,
        *,
        execution_id: str,
        project_id: str,
        kind: str,
        status: str,
        payload: dict[str, Any],
        owner_pid: int = 0,
        owner_instance: str = "",
        stage: str = "",
        progress: float = 0.0,
    ) -> None:
        if not execution_id or not project_id or not kind:
            raise ProjectCatalogError("Execution id, project id, and kind are required.")
        if status not in {"queued", "running", "succeeded", "failed", "cancelled"}:
            raise ProjectCatalogError(f"Invalid execution status: {status}")
        if payload.get("status") != status or payload.get("project_id") != project_id:
            raise ProjectCatalogError("Execution row and payload disagree.")
        if (payload.get("task_id") or payload.get("job_id")) != execution_id:
            raise ProjectCatalogError("Execution id and payload disagree.")
        try:
            normalized_progress = float(progress)
        except (TypeError, ValueError) as exc:
            raise ProjectCatalogError("Execution progress must be between 0 and 1.") from exc
        if not math.isfinite(normalized_progress) or not 0.0 <= normalized_progress <= 1.0:
            raise ProjectCatalogError("Execution progress must be a finite value between 0 and 1.")
        path = catalog_path(working_dir)
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        try:
            with closing(_connect(path)) as connection, connection:
                _ensure_schema(connection)
                existing = connection.execute(
                    "SELECT project_id, kind FROM executions WHERE execution_id=?",
                    (execution_id,),
                ).fetchone()
                if existing is not None and (existing["project_id"], existing["kind"]) != (project_id, kind):
                    raise ProjectCatalogError("Execution id belongs to another project or kind.")
                connection.execute(
                    "INSERT INTO executions "
                    "(execution_id, project_id, kind, status, stage, progress, updated_epoch, "
                    "owner_pid, owner_instance, payload_json, version) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1) "
                    "ON CONFLICT(execution_id) DO UPDATE SET "
                    "status=excluded.status, stage=excluded.stage, progress=excluded.progress, "
                    "updated_epoch=excluded.updated_epoch, owner_pid=excluded.owner_pid, "
                    "owner_instance=excluded.owner_instance, payload_json=excluded.payload_json, "
                    "version=executions.version+1",
                    (
                        execution_id,
                        project_id,
                        kind,
                        status,
                        str(stage or status),
                        normalized_progress,
                        time.time(),
                        int(owner_pid),
                        owner_instance,
                        encoded,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise ExecutionConflict("Another Studio task is already active for this project.") from exc
        except sqlite3.Error as exc:
            raise ProjectCatalogError(f"Unable to persist execution: {path}") from exc

    def get(self, working_dir: str | Path, execution_id: str, *, kind: str) -> dict[str, Any] | None:
        rows = self._query(
            working_dir,
            "SELECT * FROM executions WHERE execution_id=? AND kind=?",
            (execution_id, kind),
        )
        return rows[0] if rows else None

    def latest(self, working_dir: str | Path, *, kind: str) -> dict[str, Any] | None:
        rows = self.list(working_dir, kind=kind, limit=1)
        return rows[0] if rows else None

    def active(self, working_dir: str | Path, *, kind: str) -> dict[str, Any] | None:
        rows = self._query(
            working_dir,
            "SELECT * FROM executions WHERE kind=? AND status IN ('queued', 'running') "
            "ORDER BY updated_epoch DESC LIMIT 1",
            (kind,),
        )
        return rows[0] if rows else None

    def list(self, working_dir: str | Path, *, kind: str, limit: int = 25) -> list[dict[str, Any]]:
        return self._query(
            working_dir,
            "SELECT * FROM executions WHERE kind=? ORDER BY updated_epoch DESC LIMIT ?",
            (kind, max(0, int(limit))),
        )

    def prune(
        self,
        working_dir: str | Path,
        *,
        kind: str,
        retention_seconds: int,
        max_records: int,
    ) -> None:
        path = catalog_path(working_dir)
        try:
            with closing(_connect(path)) as connection, connection:
                _ensure_schema(connection)
                connection.execute(
                    "DELETE FROM executions WHERE kind=? AND status NOT IN ('queued', 'running') "
                    "AND updated_epoch < ?",
                    (kind, time.time() - max(0, retention_seconds)),
                )
                connection.execute(
                    "DELETE FROM executions WHERE execution_id IN ("
                    "SELECT execution_id FROM executions WHERE kind=? "
                    "AND status NOT IN ('queued', 'running') "
                    "ORDER BY updated_epoch DESC LIMIT -1 OFFSET ?)",
                    (kind, max(0, max_records)),
                )
        except sqlite3.Error as exc:
            raise ProjectCatalogError(f"Unable to prune executions: {path}") from exc

    def _query(
        self,
        working_dir: str | Path,
        sql: str,
        params: tuple[Any, ...],
    ) -> list[dict[str, Any]]:
        path = catalog_path(working_dir)
        try:
            with closing(_connect(path)) as connection, connection:
                _ensure_schema(connection)
                rows = connection.execute(sql, params).fetchall()
        except sqlite3.Error as exc:
            raise ProjectCatalogError(f"Unable to read executions: {path}") from exc
        return [_decode(row) for row in rows]


def _connect(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise ProjectCatalogError(f"Project catalog is missing: {path}")
    if path.resolve(strict=True).parent != path.parent.resolve(strict=True):
        raise ProjectCatalogError(f"Project catalog escapes its working directory: {path}")
    connection = sqlite3.connect(path, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA synchronous=FULL")
    return connection


def _ensure_schema(connection: sqlite3.Connection) -> None:
    row = connection.execute("SELECT schema_version FROM catalog_meta LIMIT 1").fetchone()
    if row is None or int(row["schema_version"]) != CATALOG_SCHEMA_VERSION:
        raise ProjectCatalogError("Unsupported project catalog schema for executions.")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS executions ("
        "execution_id TEXT PRIMARY KEY, project_id TEXT NOT NULL, kind TEXT NOT NULL, "
        "status TEXT NOT NULL, stage TEXT NOT NULL, progress REAL NOT NULL, "
        "updated_epoch REAL NOT NULL, owner_pid INTEGER NOT NULL, "
        "owner_instance TEXT NOT NULL, payload_json TEXT NOT NULL, version INTEGER NOT NULL)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS executions_kind_status_idx "
        "ON executions(kind, status, updated_epoch DESC)"
    )
    # Import records written by the first catalog-backed Studio release. Keep
    # its old table for rollback to that release; the new ledger wins thereafter.
    legacy = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='studio_tasks'"
    ).fetchone()
    if legacy is not None:
        connection.execute(
            "INSERT OR IGNORE INTO executions "
            "(execution_id, project_id, kind, status, stage, progress, updated_epoch, "
            "owner_pid, owner_instance, payload_json, version) "
            "SELECT task_id, project_id, 'studio', status, status, 0.0, updated_epoch, "
            "owner_pid, owner_instance, payload_json, 1 FROM studio_tasks"
        )
    connection.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS executions_one_active_studio_idx "
        "ON executions(project_id) "
        "WHERE kind='studio' AND status IN ('queued', 'running')"
    )


def _decode(row: sqlite3.Row) -> dict[str, Any]:
    try:
        payload = json.loads(row["payload_json"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise ProjectCatalogError("Execution payload is invalid.") from exc
    if not isinstance(payload, dict):
        raise ProjectCatalogError("Execution payload is invalid.")
    if payload.get("status") != row["status"] or payload.get("project_id") != row["project_id"]:
        raise ProjectCatalogError("Execution row and payload disagree.")
    payload_id = payload.get("task_id") or payload.get("job_id")
    if payload_id != row["execution_id"]:
        raise ProjectCatalogError("Execution id and payload disagree.")
    return {
        "execution_id": str(row["execution_id"]),
        "project_id": str(row["project_id"]),
        "kind": str(row["kind"]),
        "status": str(row["status"]),
        "stage": str(row["stage"]),
        "progress": float(row["progress"]),
        "updated_epoch": float(row["updated_epoch"]),
        "owner_pid": int(row["owner_pid"]),
        "owner_instance": str(row["owner_instance"]),
        "version": int(row["version"]),
        "payload": payload,
    }
