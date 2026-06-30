from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ASSET_REGISTRY_FILENAME = "assets.json"
ASSET_REGISTRY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class AssetRecord:
    asset_id: str
    kind: str
    path: str
    created_at: str
    role: str = ""
    source: str = ""
    checksum_sha256: str = ""
    size_bytes: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    parents: list[str] = field(default_factory=list)
    retention: str = "project"
    schema_version: int = ASSET_REGISTRY_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AssetRegistryError(ValueError):
    pass


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def asset_registry_path(working_dir: str | Path) -> Path:
    return Path(working_dir) / ASSET_REGISTRY_FILENAME


def load_asset_registry(working_dir: str | Path) -> dict[str, Any]:
    path = asset_registry_path(working_dir)
    if not path.exists():
        return {
            "schema_version": ASSET_REGISTRY_SCHEMA_VERSION,
            "updated_at": "",
            "assets": [],
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AssetRegistryError(f"Unable to read asset registry: {path}") from exc
    return _normalize_registry(payload)


def record_project_asset(
    state: object,
    asset_path: str | Path,
    *,
    kind: str,
    role: str = "",
    source: str = "",
    metadata: Mapping[str, Any] | None = None,
    parents: Iterable[str] | None = None,
    retention: str = "project",
) -> AssetRecord:
    working_dir = Path(str(getattr(state, "working_dir", "")))
    roots = _project_roots(state)
    return record_asset(
        working_dir,
        asset_path,
        kind=kind,
        role=role,
        source=source,
        metadata=metadata,
        parents=parents,
        retention=retention,
        allowed_roots=roots,
    )


def record_asset(
    working_dir: str | Path,
    asset_path: str | Path,
    *,
    kind: str,
    role: str = "",
    source: str = "",
    metadata: Mapping[str, Any] | None = None,
    parents: Iterable[str] | None = None,
    retention: str = "project",
    allowed_roots: Iterable[str | Path] | None = None,
) -> AssetRecord:
    resolved_asset = Path(asset_path).expanduser().resolve(strict=True)
    if not resolved_asset.is_file():
        raise FileNotFoundError(f"Asset path is not a file: {resolved_asset}")

    resolved_working_dir = Path(working_dir).expanduser().resolve(strict=False)
    roots = [
        Path(root).expanduser().resolve(strict=False)
        for root in (allowed_roots or [resolved_working_dir])
    ]
    if not roots or not any(_is_within(resolved_asset, root) for root in roots):
        allowed_text = ", ".join(str(root) for root in roots)
        raise AssetRegistryError(f"Asset path must stay inside project roots: {allowed_text}")

    checksum = _sha256_file(resolved_asset)
    stat = resolved_asset.stat()
    now = utc_now_iso()
    normalized_kind = str(kind or "").strip()
    if not normalized_kind:
        raise AssetRegistryError("Asset kind is required.")

    parent_list = [str(parent) for parent in (parents or []) if str(parent or "").strip()]
    record = AssetRecord(
        asset_id=_asset_id(
            kind=normalized_kind,
            path=str(resolved_asset),
            checksum=checksum,
            role=role,
            source=source,
        ),
        kind=normalized_kind,
        path=str(resolved_asset),
        created_at=now,
        role=str(role or ""),
        source=str(source or ""),
        checksum_sha256=checksum,
        size_bytes=stat.st_size,
        metadata=dict(metadata or {}),
        parents=parent_list,
        retention=str(retention or "project"),
    )

    registry = load_asset_registry(resolved_working_dir)
    assets_by_id = {
        str(item.get("asset_id")): item
        for item in registry.get("assets", [])
        if isinstance(item, Mapping) and item.get("asset_id")
    }
    assets_by_id[record.asset_id] = record.to_dict()
    registry["schema_version"] = ASSET_REGISTRY_SCHEMA_VERSION
    registry["updated_at"] = now
    registry["assets"] = sorted(
        assets_by_id.values(),
        key=lambda item: (str(item.get("created_at") or ""), str(item.get("asset_id") or "")),
    )
    _atomic_write_json(asset_registry_path(resolved_working_dir), registry)
    return record


def latest_assets(
    working_dir: str | Path,
    *,
    kind: str | None = None,
    role: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    registry = load_asset_registry(working_dir)
    assets = [
        item
        for item in registry.get("assets", [])
        if isinstance(item, Mapping)
        and (kind is None or item.get("kind") == kind)
        and (role is None or item.get("role") == role)
    ]
    assets.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return [dict(item) for item in assets[: max(int(limit), 0)]]


def _normalize_registry(raw: object) -> dict[str, Any]:
    payload = dict(raw) if isinstance(raw, Mapping) else {}
    assets: list[dict[str, Any]] = []
    for item in payload.get("assets") or []:
        if not isinstance(item, Mapping):
            continue
        asset_id = str(item.get("asset_id") or "").strip()
        path = str(item.get("path") or "").strip()
        kind = str(item.get("kind") or "").strip()
        if not asset_id or not path or not kind:
            continue
        normalized = dict(item)
        normalized["schema_version"] = ASSET_REGISTRY_SCHEMA_VERSION
        normalized["asset_id"] = asset_id
        normalized["path"] = path
        normalized["kind"] = kind
        normalized["metadata"] = dict(item.get("metadata") or {}) if isinstance(item.get("metadata"), Mapping) else {}
        normalized["parents"] = [str(parent) for parent in item.get("parents") or []]
        assets.append(normalized)
    return {
        "schema_version": ASSET_REGISTRY_SCHEMA_VERSION,
        "updated_at": str(payload.get("updated_at") or ""),
        "assets": assets,
    }


def _project_roots(state: object) -> list[Path]:
    roots: list[Path] = []
    for value in (
        getattr(state, "working_dir", None),
        getattr(state, "output_dir", None),
    ):
        if value:
            roots.append(Path(str(value)))
    for source_file in getattr(state, "source_files", None) or []:
        if source_file:
            roots.append(Path(str(source_file)).expanduser().resolve(strict=False).parent)
    return _dedupe_roots(roots)


def _dedupe_roots(roots: Iterable[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        resolved = root.expanduser().resolve(strict=False)
        key = os.path.normcase(os.path.abspath(str(resolved)))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(resolved)
    return deduped


def _asset_id(*, kind: str, path: str, checksum: str, role: str, source: str) -> str:
    payload = {
        "kind": kind,
        "path": path,
        "checksum": checksum,
        "role": str(role or ""),
        "source": str(source or ""),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"asset_{hashlib.sha256(encoded).hexdigest()[:20]}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath([
            os.path.normcase(os.path.abspath(str(path))),
            os.path.normcase(os.path.abspath(str(root))),
        ]) == os.path.normcase(os.path.abspath(str(root)))
    except ValueError:
        return False


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
