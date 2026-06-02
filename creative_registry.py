from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
from typing import Any

from state import utc_now_iso


REGISTRY_VERSION = "creative-run-registry-v1"
REGISTRY_FILENAME = "creative_runs.json"
MAX_REGISTRY_RECORDS = 80


@dataclass(frozen=True)
class CreativeRunRecord:
    run_id: str
    feature: str
    created_at: str
    manifest_path: str | None = None
    output_path: str | None = None
    graph_version: str | None = None
    quality_score: float | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.quality_score is not None:
            payload["quality_score"] = round(float(self.quality_score), 4)
        return payload


def creative_registry_path(working_dir: str | Path) -> Path:
    return Path(working_dir) / REGISTRY_FILENAME


def load_creative_registry(working_dir: str | Path) -> dict[str, Any]:
    path = creative_registry_path(working_dir)
    if not path.is_file():
        return _empty_registry()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _empty_registry()
    if not isinstance(payload, dict):
        return _empty_registry()
    records = payload.get("runs")
    if not isinstance(records, list):
        records = []
    return {
        "version": str(payload.get("version") or REGISTRY_VERSION),
        "updated_at": str(payload.get("updated_at") or ""),
        "runs": [dict(item) for item in records if isinstance(item, dict)],
    }


def record_creative_run(
    *,
    working_dir: str | Path,
    feature: str,
    manifest_path: str | None = None,
    output_path: str | None = None,
    graph_version: str | None = None,
    quality_score: float | None = None,
    summary: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    created_at = utc_now_iso()
    run_id = _run_id(feature, created_at)
    record = CreativeRunRecord(
        run_id=run_id,
        feature=_safe_feature(feature),
        created_at=created_at,
        manifest_path=str(manifest_path) if manifest_path else None,
        output_path=str(output_path) if output_path else None,
        graph_version=str(graph_version) if graph_version else None,
        quality_score=float(quality_score) if quality_score is not None else None,
        summary=dict(summary or {}),
        artifacts=dict(artifacts or {}),
    )
    registry = load_creative_registry(working_dir)
    records = [dict(item) for item in registry.get("runs", []) if isinstance(item, dict)]
    records.append(record.to_dict())
    records = records[-MAX_REGISTRY_RECORDS:]
    payload = {
        "version": REGISTRY_VERSION,
        "updated_at": created_at,
        "runs": records,
    }
    path = creative_registry_path(working_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(path, payload)
    except OSError as exc:
        return {
            "registered": False,
            "registry_path": str(path),
            "error": str(exc),
            "record": record.to_dict(),
        }
    return {
        "registered": True,
        "registry_path": str(path),
        "record": record.to_dict(),
    }


def latest_creative_runs(
    working_dir: str | Path,
    *,
    feature: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    registry = load_creative_registry(working_dir)
    normalized_feature = _safe_feature(feature) if feature else None
    records = [
        dict(item)
        for item in registry.get("runs", [])
        if isinstance(item, dict)
        and (normalized_feature is None or _safe_feature(item.get("feature")) == normalized_feature)
    ]
    return list(reversed(records[-max(1, int(limit)):]))


def _empty_registry() -> dict[str, Any]:
    return {
        "version": REGISTRY_VERSION,
        "updated_at": "",
        "runs": [],
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(path)


def _run_id(feature: str, created_at: str) -> str:
    stamp = created_at.replace(":", "-").replace("+00:00", "Z")
    return f"{_safe_feature(feature)}:{stamp}"


def _safe_feature(value: Any) -> str:
    text = str(value or "creative").strip().lower().replace("-", "_")
    cleaned = "".join(ch for ch in text if ch.isalnum() or ch == "_").strip("_")
    return cleaned or "creative"


__all__ = [
    "CreativeRunRecord",
    "REGISTRY_FILENAME",
    "REGISTRY_VERSION",
    "creative_registry_path",
    "latest_creative_runs",
    "load_creative_registry",
    "record_creative_run",
]
