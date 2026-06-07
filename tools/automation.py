from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from state import utc_now_iso


COVERAGE_POLICIES = {"quality_only", "target_count", "exact_count"}
DENSITY_LEVELS = {"sparse", "balanced", "dense", "chapter_coverage"}


def normalize_coverage_policy(value: object, *, explicit_count: bool = False) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    if normalized in COVERAGE_POLICIES:
        return normalized
    return "target_count" if explicit_count else "quality_only"


def normalize_density(value: object, *, clip_duration: float = 0.0) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    if normalized in DENSITY_LEVELS:
        return normalized
    if clip_duration >= 600.0:
        return "chapter_coverage"
    if clip_duration >= 180.0:
        return "balanced"
    return "sparse"


def clamp_int(value: object, *, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(number, maximum))


def coverage_counts(
    *,
    requested_count: int | None,
    selected_count: int,
    rejected_count: int,
    rejection_reasons: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "requested_count": requested_count,
        "selected_count": max(int(selected_count), 0),
        "rejected_count": max(int(rejected_count), 0),
        "rejection_reasons": list(rejection_reasons or []),
    }


def status_path(bundle_dir: Path) -> Path:
    return Path(bundle_dir) / "run_status.json"


def write_run_status(
    bundle_dir: Path,
    *,
    feature: str,
    phase: str,
    status: str = "running",
    payload: dict[str, Any] | None = None,
) -> Path:
    path = status_path(bundle_dir)
    current: dict[str, Any] = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                current = loaded
        except (OSError, json.JSONDecodeError):
            current = {}
    history = list(current.get("history") or [])
    event = {
        "at": utc_now_iso(),
        "phase": phase,
        "status": status,
        **(payload or {}),
    }
    history.append(event)
    current.update(
        {
            "feature": feature,
            "status": status,
            "phase": phase,
            "updated_at": event["at"],
            "history": history[-80:],
        }
    )
    for key, value in (payload or {}).items():
        current[key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(current, indent=2), encoding="utf-8")
    return path
