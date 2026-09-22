"""Durable, revisioned project snapshots.

The SQLite catalog is authoritative once created. The adjacent project JSON file
remains a compatibility export, never a fallback for a damaged catalog.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any


CATALOG_FILENAME = "project.sqlite3"
CATALOG_SCHEMA_VERSION = 1


class ProjectCatalogError(ValueError):
    """A project catalog cannot be read or updated safely."""


class ProjectRevisionConflict(ProjectCatalogError):
    """The caller is editing an older project revision."""


def catalog_path(working_dir: str | Path) -> Path:
    return Path(working_dir) / CATALOG_FILENAME


def read_snapshot(working_dir: str | Path) -> dict[str, Any] | None:
    path = catalog_path(working_dir)
    if not path.exists():
        return None
    _assert_local_path(path)
    try:
        with closing(_connect(path)) as connection:
            _check_schema(connection)
            row = connection.execute(
                "SELECT revision, payload_json, payload_sha256 FROM project_revisions "
                "ORDER BY revision DESC LIMIT 1"
            ).fetchone()
    except sqlite3.Error as exc:
        raise ProjectCatalogError(f"Unable to read project catalog: {path}") from exc
    if row is None:
        raise ProjectCatalogError(f"Project catalog has no revisions: {path}")
    return _decode_revision(row, path)


def list_revisions(working_dir: str | Path) -> list[dict[str, Any]]:
    path = catalog_path(working_dir)
    if not path.exists():
        return []
    _assert_local_path(path)
    try:
        with closing(_connect(path)) as connection:
            _check_schema(connection)
            rows = connection.execute(
                "SELECT revision, created_at, payload_sha256 FROM project_revisions "
                "ORDER BY revision DESC"
            ).fetchall()
    except sqlite3.Error as exc:
        raise ProjectCatalogError(f"Unable to read project catalog: {path}") from exc
    return [dict(row) for row in rows]


def write_snapshot(
    working_dir: str | Path,
    payload: dict[str, Any],
    *,
    expected_revision: int,
    legacy_path: Path,
) -> int:
    """Commit one snapshot with compare-and-swap concurrency protection.

    A legacy JSON snapshot is imported as revision 1 inside the same transaction
    before the caller's new revision is written. Failed writes roll back both.
    """
    path = catalog_path(working_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    _assert_local_path(path)
    try:
        with closing(_connect(path)) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            _create_schema(connection)
            row = connection.execute(
                "SELECT revision, payload_json, payload_sha256 FROM project_revisions "
                "ORDER BY revision DESC LIMIT 1"
            ).fetchone()
            current = int(row["revision"]) if row else 0
            if row is not None:
                latest = _decode_revision(row, path)
                if latest.get("project_id") != payload.get("project_id"):
                    raise ProjectCatalogError("Project catalog belongs to a different project.")
            imported = False
            if current == 0 and legacy_path.is_file():
                _assert_local_path(legacy_path)
                try:
                    legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise ProjectCatalogError(
                        f"Cannot migrate unreadable project state: {legacy_path}"
                    ) from exc
                if not isinstance(legacy, dict) or legacy.get("project_id") != payload.get("project_id"):
                    raise ProjectCatalogError("Legacy project state does not match this project.")
                _insert_revision(connection, 1, legacy)
                current = 1
                imported = True
            if expected_revision != current and not (imported and expected_revision == 0):
                raise ProjectRevisionConflict(
                    f"Project changed since it was loaded (expected revision "
                    f"{expected_revision}, current revision {current}). Reload before editing."
                )
            revision = current + 1
            committed = dict(payload, revision=revision)
            _insert_revision(connection, revision, committed)
            connection.commit()
            return revision
    except sqlite3.Error as exc:
        raise ProjectCatalogError(f"Unable to write project catalog: {path}") from exc


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA synchronous=FULL")
    return connection


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        "CREATE TABLE IF NOT EXISTS catalog_meta (schema_version INTEGER NOT NULL)"
    )
    row = connection.execute("SELECT schema_version FROM catalog_meta LIMIT 1").fetchone()
    if row is None:
        connection.execute(
            "INSERT INTO catalog_meta (schema_version) VALUES (?)", (CATALOG_SCHEMA_VERSION,)
        )
    elif int(row["schema_version"]) != CATALOG_SCHEMA_VERSION:
        raise ProjectCatalogError(
            f"Unsupported project catalog version {row['schema_version']}; "
            "update Vex before opening this project."
        )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS project_revisions ("
        "revision INTEGER PRIMARY KEY, "
        "created_at TEXT NOT NULL, "
        "payload_json TEXT NOT NULL, "
        "payload_sha256 TEXT NOT NULL)"
    )


def _check_schema(connection: sqlite3.Connection) -> None:
    row = connection.execute("SELECT schema_version FROM catalog_meta LIMIT 1").fetchone()
    if row is None or int(row["schema_version"]) != CATALOG_SCHEMA_VERSION:
        raise ProjectCatalogError("Unsupported or incomplete project catalog schema.")


def _insert_revision(connection: sqlite3.Connection, revision: int, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    connection.execute(
        "INSERT INTO project_revisions "
        "(revision, created_at, payload_json, payload_sha256) VALUES (?, ?, ?, ?)",
        (revision, str(payload.get("updated_at") or ""), encoded, digest),
    )


def _decode_revision(row: sqlite3.Row, path: Path) -> dict[str, Any]:
    encoded = str(row["payload_json"])
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    if digest != row["payload_sha256"]:
        raise ProjectCatalogError(f"Project revision checksum mismatch: {path}")
    try:
        payload = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise ProjectCatalogError(f"Invalid project revision JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ProjectCatalogError(f"Invalid project revision payload: {path}")
    payload["revision"] = int(row["revision"])
    return payload


def _assert_local_path(path: Path) -> None:
    if path.exists() and path.resolve(strict=True).parent != path.parent.resolve(strict=True):
        raise ProjectCatalogError(f"Project file escapes its working directory: {path}")
