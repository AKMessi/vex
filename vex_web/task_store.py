"""Project-local persistence for Studio task status and bounded progress."""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Any

from vex_runtime.project_catalog import ProjectCatalogError, catalog_path


class StudioTaskStore:
    def put(
        self,
        working_dir: str | Path,
        payload: dict[str, Any],
        *,
        owner_pid: int,
        owner_instance: str,
    ) -> None:
        path = catalog_path(working_dir)
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        try:
            with closing(_connect(path)) as connection, connection:
                _ensure_table(connection, create=True)
                connection.execute(
                    "INSERT INTO studio_tasks "
                    "(task_id, project_id, status, updated_epoch, owner_pid, owner_instance, payload_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(task_id) DO UPDATE SET "
                    "status=excluded.status, updated_epoch=excluded.updated_epoch, "
                    "owner_pid=excluded.owner_pid, owner_instance=excluded.owner_instance, "
                    "payload_json=excluded.payload_json",
                    (
                        payload["task_id"],
                        payload["project_id"],
                        payload["status"],
                        time.time(),
                        int(owner_pid),
                        owner_instance,
                        encoded,
                    ),
                )
        except sqlite3.Error as exc:
            raise ProjectCatalogError(f"Unable to persist Studio task: {path}") from exc

    def get(self, working_dir: str | Path, task_id: str) -> dict[str, Any] | None:
        path = catalog_path(working_dir)
        try:
            with closing(_connect(path)) as connection:
                if not _ensure_table(connection):
                    return None
                row = connection.execute(
                    "SELECT payload_json, owner_pid, owner_instance FROM studio_tasks WHERE task_id=?",
                    (task_id,),
                ).fetchone()
        except sqlite3.Error as exc:
            raise ProjectCatalogError(f"Unable to read Studio task: {path}") from exc
        return _decode(row) if row is not None else None

    def active(self, working_dir: str | Path) -> dict[str, Any] | None:
        path = catalog_path(working_dir)
        try:
            with closing(_connect(path)) as connection:
                if not _ensure_table(connection):
                    return None
                row = connection.execute(
                    "SELECT payload_json, owner_pid, owner_instance FROM studio_tasks "
                    "WHERE status IN ('queued', 'running') "
                    "ORDER BY updated_epoch DESC LIMIT 1"
                ).fetchone()
        except sqlite3.Error as exc:
            raise ProjectCatalogError(f"Unable to read Studio tasks: {path}") from exc
        return _decode(row) if row is not None else None

    def latest(self, working_dir: str | Path) -> dict[str, Any] | None:
        path = catalog_path(working_dir)
        try:
            with closing(_connect(path)) as connection:
                if not _ensure_table(connection):
                    return None
                row = connection.execute(
                    "SELECT payload_json, owner_pid, owner_instance FROM studio_tasks "
                    "ORDER BY updated_epoch DESC LIMIT 1"
                ).fetchone()
        except sqlite3.Error as exc:
            raise ProjectCatalogError(f"Unable to read Studio tasks: {path}") from exc
        return _decode(row) if row is not None else None

    def prune(self, working_dir: str | Path, *, retention_seconds: int, max_tasks: int) -> None:
        path = catalog_path(working_dir)
        try:
            with closing(_connect(path)) as connection, connection:
                if not _ensure_table(connection):
                    return
                connection.execute(
                    "DELETE FROM studio_tasks WHERE status NOT IN ('queued', 'running') "
                    "AND updated_epoch < ?",
                    (time.time() - retention_seconds,),
                )
                connection.execute(
                    "DELETE FROM studio_tasks WHERE task_id IN ("
                    "SELECT task_id FROM studio_tasks WHERE status NOT IN ('queued', 'running') "
                    "ORDER BY updated_epoch DESC LIMIT -1 OFFSET ?)",
                    (max_tasks,),
                )
        except sqlite3.Error as exc:
            raise ProjectCatalogError(f"Unable to prune Studio tasks: {path}") from exc


def _connect(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise ProjectCatalogError(f"Project catalog is missing: {path}")
    if path.resolve(strict=True).parent != path.parent.resolve(strict=True):
        raise ProjectCatalogError(f"Project catalog escapes its working directory: {path}")
    connection = sqlite3.connect(path, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA synchronous=FULL")
    return connection


def _ensure_table(connection: sqlite3.Connection, *, create: bool = False) -> bool:
    row = connection.execute("SELECT schema_version FROM catalog_meta LIMIT 1").fetchone()
    if row is None or int(row["schema_version"]) != 1:
        raise ProjectCatalogError("Unsupported project catalog schema for Studio tasks.")
    if not create:
        return connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='studio_tasks'"
        ).fetchone() is not None
    connection.execute(
        "CREATE TABLE IF NOT EXISTS studio_tasks ("
        "task_id TEXT PRIMARY KEY, project_id TEXT NOT NULL, status TEXT NOT NULL, "
        "updated_epoch REAL NOT NULL, owner_pid INTEGER NOT NULL, "
        "owner_instance TEXT NOT NULL, payload_json TEXT NOT NULL)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS studio_tasks_status_idx "
        "ON studio_tasks(status, updated_epoch DESC)"
    )
    return True


def _decode(row: sqlite3.Row) -> dict[str, Any]:
    try:
        payload = json.loads(row["payload_json"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise ProjectCatalogError("Studio task payload is invalid.") from exc
    if not isinstance(payload, dict):
        raise ProjectCatalogError("Studio task payload is invalid.")
    return {
        "payload": payload,
        "owner_pid": int(row["owner_pid"]),
        "owner_instance": str(row["owner_instance"]),
    }
