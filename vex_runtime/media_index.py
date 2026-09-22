"""Catalog-backed asset and cache indexes with legacy JSON migration."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from vex_runtime.project_catalog import CATALOG_SCHEMA_VERSION, ProjectCatalogError, catalog_path


_SPECS = {
    "asset": ("project_assets", "asset_id", "assets.json", "assets"),
    "cache": ("project_cache", "cache_key", "cache/cache_index.json", "entries"),
}


def read_media_records(working_dir: str | Path, kind: str) -> list[dict[str, Any]] | None:
    """Return None when no catalog index exists; an empty list is authoritative."""
    table, key, _, _ = _spec(kind)
    path = catalog_path(working_dir)
    if not path.is_file():
        return None
    try:
        with closing(_connect(path)) as connection:
            _check_catalog(connection)
            exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()
            if exists is None:
                return None
            rows = connection.execute(
                f"SELECT record_id, payload_json, payload_sha256 FROM {table} "
                "ORDER BY created_at, record_id"
            ).fetchall()
    except sqlite3.Error as exc:
        raise ProjectCatalogError(f"Unable to read project {kind} index: {path}") from exc
    records: list[dict[str, Any]] = []
    for row in rows:
        encoded = str(row["payload_json"])
        if hashlib.sha256(encoded.encode("utf-8")).hexdigest() != row["payload_sha256"]:
            raise ProjectCatalogError(f"Project {kind} index checksum mismatch: {path}")
        try:
            payload = json.loads(encoded)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ProjectCatalogError(f"Invalid {kind} index record: {path}") from exc
        if not isinstance(payload, dict) or payload.get(key) != row["record_id"]:
            raise ProjectCatalogError(f"Invalid {kind} index identity: {path}")
        records.append(payload)
    return records


def upsert_media_record(working_dir: str | Path, kind: str, record: dict[str, Any]) -> None:
    path = catalog_path(working_dir)
    try:
        with closing(_connect(path)) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            ensure_media_tables(connection, path.parent)
            insert_media_record(connection, kind, record)
    except sqlite3.Error as exc:
        raise ProjectCatalogError(f"Unable to write project {kind} index: {path}") from exc


def ensure_media_tables(connection: sqlite3.Connection, working_dir: Path) -> None:
    """Create both indexes and import legacy JSON inside the caller's transaction."""
    _check_catalog(connection)
    for kind, (table, _key, legacy_name, collection) in _SPECS.items():
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        if exists is not None:
            continue
        connection.execute(
            f"CREATE TABLE {table} (record_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, "
            "payload_json TEXT NOT NULL, payload_sha256 TEXT NOT NULL)"
        )
        legacy_path = working_dir / legacy_name
        if not legacy_path.is_file():
            continue
        if not legacy_path.resolve(strict=True).is_relative_to(working_dir.resolve(strict=True)):
            raise ProjectCatalogError(f"Legacy {kind} index escapes the project: {legacy_path}")
        try:
            legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProjectCatalogError(f"Unable to migrate legacy {kind} index: {legacy_path}") from exc
        if not isinstance(legacy, dict) or not isinstance(legacy.get(collection), list):
            raise ProjectCatalogError(f"Invalid legacy {kind} index: {legacy_path}")
        version = legacy.get("schema_version", 1)
        if version != 1:
            raise ProjectCatalogError(f"Unsupported legacy {kind} index version: {version}")
        for item in legacy[collection]:
            insert_media_record(connection, kind, item)


def insert_media_record(connection: sqlite3.Connection, kind: str, record: dict[str, Any]) -> None:
    table, key, _, _ = _spec(kind)
    if not isinstance(record, dict) or not isinstance(record.get(key), str) or not record[key]:
        raise ProjectCatalogError(f"Invalid {kind} record identity.")
    encoded = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    checksum = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    connection.execute(
        f"INSERT INTO {table} (record_id, created_at, payload_json, payload_sha256) "
        "VALUES (?, ?, ?, ?) "
        "ON CONFLICT(record_id) DO UPDATE SET "
        "created_at=excluded.created_at, payload_json=excluded.payload_json, "
        "payload_sha256=excluded.payload_sha256",
        (record[key], str(record.get("created_at") or ""), encoded, checksum),
    )


def _connect(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise ProjectCatalogError(f"Project catalog is missing: {path}")
    if path.resolve(strict=True).parent != path.parent.resolve(strict=True):
        raise ProjectCatalogError(f"Project catalog escapes its working directory: {path}")
    connection = sqlite3.connect(path, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA synchronous=FULL")
    return connection


def _check_catalog(connection: sqlite3.Connection) -> None:
    row = connection.execute("SELECT schema_version FROM catalog_meta LIMIT 1").fetchone()
    if row is None or int(row["schema_version"]) != CATALOG_SCHEMA_VERSION:
        raise ProjectCatalogError("Unsupported project catalog schema for media indexes.")


def _spec(kind: str) -> tuple[str, str, str, str]:
    try:
        return _SPECS[kind]
    except KeyError as exc:
        raise ProjectCatalogError(f"Unknown media index kind: {kind}") from exc
