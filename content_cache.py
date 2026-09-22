from __future__ import annotations

import hashlib
import json
import os
import tempfile
import warnings
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vex_runtime.media_index import read_media_records, upsert_media_record
from vex_runtime.project_catalog import ProjectCatalogError, catalog_path


CACHE_SCHEMA_VERSION = 1
CACHE_INDEX_NAME = "cache_index.json"


@dataclass(frozen=True)
class CacheEntry:
    cache_key: str
    checksum_sha256: str
    original_path: str
    cached_path: str
    kind: str
    size_bytes: int
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = CACHE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ContentCacheError(ValueError):
    pass


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def cache_root(working_dir: str | Path) -> Path:
    return Path(working_dir) / "cache"


def cache_index_path(working_dir: str | Path) -> Path:
    return cache_root(working_dir) / CACHE_INDEX_NAME


def load_cache_index(working_dir: str | Path) -> dict[str, Any]:
    catalog_records = read_media_records(working_dir, "cache")
    if catalog_records is not None:
        return {
            "schema_version": CACHE_SCHEMA_VERSION,
            "updated_at": max((str(item.get("created_at") or "") for item in catalog_records), default=""),
            "entries": catalog_records,
        }
    path = cache_index_path(working_dir)
    if not path.exists():
        return {"schema_version": CACHE_SCHEMA_VERSION, "updated_at": "", "entries": []}
    if not path.resolve(strict=True).is_relative_to(Path(working_dir).resolve(strict=True)):
        raise ContentCacheError(f"Cache index escapes the project: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContentCacheError(f"Unable to read content cache index: {path}") from exc
    return _normalize_index(payload)


def cache_project_file(
    state: object,
    path: str | Path,
    *,
    kind: str,
    metadata: Mapping[str, Any] | None = None,
) -> CacheEntry:
    return cache_file(
        getattr(state, "working_dir"),
        path,
        kind=kind,
        metadata=metadata,
    )


def cache_file(
    working_dir: str | Path,
    path: str | Path,
    *,
    kind: str,
    metadata: Mapping[str, Any] | None = None,
) -> CacheEntry:
    entry = prepare_cache_file(working_dir, path, kind=kind, metadata=metadata)
    if catalog_path(working_dir).is_file():
        upsert_media_record(working_dir, "cache", entry.to_dict())
        sync_cache_index_json(working_dir)
        return entry
    index = load_cache_index(working_dir)
    entries_by_key = {
        str(item.get("cache_key")): item
        for item in index.get("entries", [])
        if isinstance(item, Mapping) and item.get("cache_key")
    }
    entries_by_key[entry.cache_key] = entry.to_dict()
    index["schema_version"] = CACHE_SCHEMA_VERSION
    index["updated_at"] = entry.created_at
    index["entries"] = sorted(
        entries_by_key.values(),
        key=lambda item: (str(item.get("created_at") or ""), str(item.get("cache_key") or "")),
    )
    _atomic_write_json(cache_index_path(working_dir), index)
    return entry


def prepare_cache_file(
    working_dir: str | Path,
    path: str | Path,
    *,
    kind: str,
    metadata: Mapping[str, Any] | None = None,
) -> CacheEntry:
    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file():
        raise FileNotFoundError(f"Cache source is not a file: {source}")
    normalized_kind = str(kind or "").strip()
    if not normalized_kind:
        raise ContentCacheError("Cache kind is required.")
    checksum = _sha256_file(source)
    stat = source.stat()
    destination = _cache_object_path(working_dir, checksum, source.suffix)
    working_root = Path(working_dir).expanduser().resolve(strict=True)
    if not destination.parent.resolve(strict=False).is_relative_to(working_root):
        raise ContentCacheError("Cache object path escapes the project working directory.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.parent.resolve(strict=True).is_relative_to(
        working_root
    ):
        raise ContentCacheError("Cache object path escapes the project working directory.")
    _materialize_cache_object(source, destination, checksum)
    now = utc_now_iso()
    entry = CacheEntry(
        cache_key=f"sha256:{checksum}",
        checksum_sha256=checksum,
        original_path=str(source),
        cached_path=str(destination),
        kind=normalized_kind,
        size_bytes=stat.st_size,
        created_at=now,
        metadata=dict(metadata or {}),
    )
    return entry


def sync_cache_index_json(working_dir: str | Path) -> None:
    if not catalog_path(working_dir).is_file():
        return
    try:
        _atomic_write_json(cache_index_path(working_dir), load_cache_index(working_dir))
    except (OSError, ProjectCatalogError) as exc:
        warnings.warn(
            f"Cache index was saved in the catalog but its JSON export failed: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )


def find_cached_file(working_dir: str | Path, cache_key: str) -> CacheEntry | None:
    normalized_key = str(cache_key or "").strip()
    for item in load_cache_index(working_dir).get("entries") or []:
        if not isinstance(item, Mapping):
            continue
        if item.get("cache_key") == normalized_key:
            return _entry_from_mapping(item)
    return None


def _cache_object_path(working_dir: str | Path, checksum: str, suffix: str) -> Path:
    safe_suffix = suffix if suffix.startswith(".") and len(suffix) <= 16 else ".bin"
    return cache_root(working_dir) / "objects" / checksum[:2] / f"{checksum}{safe_suffix}"


def _normalize_index(raw: object) -> dict[str, Any]:
    payload = dict(raw) if isinstance(raw, Mapping) else {}
    entries = [
        _entry_from_mapping(item).to_dict()
        for item in payload.get("entries") or []
        if isinstance(item, Mapping) and item.get("cache_key")
    ]
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "updated_at": str(payload.get("updated_at") or ""),
        "entries": entries,
    }


def _entry_from_mapping(item: Mapping[str, Any]) -> CacheEntry:
    return CacheEntry(
        cache_key=str(item.get("cache_key") or ""),
        checksum_sha256=str(item.get("checksum_sha256") or ""),
        original_path=str(item.get("original_path") or ""),
        cached_path=str(item.get("cached_path") or ""),
        kind=str(item.get("kind") or ""),
        size_bytes=_coerce_int(item.get("size_bytes")),
        created_at=str(item.get("created_at") or ""),
        metadata=dict(item.get("metadata") or {}) if isinstance(item.get("metadata"), Mapping) else {},
    )


def _materialize_cache_object(source: Path, destination: Path, checksum: str) -> None:
    if destination.is_symlink():
        raise ContentCacheError(f"Cached object path is a symbolic link: {destination}")
    if destination.exists():
        if not destination.is_file() or _sha256_file(destination) != checksum:
            raise ContentCacheError(f"Cached object checksum mismatch: {destination}")
        return
    temp_path: Path | None = None
    try:
        digest = hashlib.sha256()
        with source.open("rb") as input_file, tempfile.NamedTemporaryFile(
            "wb",
            dir=destination.parent,
            prefix=f".{checksum[:12]}.",
            suffix=".tmp",
            delete=False,
        ) as output_file:
            temp_path = Path(output_file.name)
            for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                digest.update(chunk)
                output_file.write(chunk)
            output_file.flush()
            os.fsync(output_file.fileno())
        if digest.hexdigest() != checksum:
            raise ContentCacheError("Cache source changed while it was being copied.")
        if destination.exists():
            if not destination.is_file() or _sha256_file(destination) != checksum:
                raise ContentCacheError(f"Cached object checksum mismatch: {destination}")
            return
        os.replace(temp_path, destination)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _coerce_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            json.dump(payload, temp_file, indent=2)
            temp_file.write("\n")
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)
