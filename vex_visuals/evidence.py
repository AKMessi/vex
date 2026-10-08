"""Content-bound, chronological evidence shared by every visual workflow."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def select_evidence_frames(paths: Iterable[Path], *, limit: int = 8) -> list[Path]:
    """Retain the beginning and resolved ending, never a chronological prefix."""
    values = list(dict.fromkeys(Path(path) for path in paths if Path(path).is_file()))
    budget = max(2, min(int(limit), 24))
    if len(values) <= budget:
        return values
    if budget == 2:
        return [values[0], values[-1]]
    indices = {0, len(values) - 2, len(values) - 1}
    for index in range(budget):
        indices.add(round(index * (len(values) - 1) / (budget - 1)))
    while len(indices) > budget:
        removable = indices - {0, len(values) - 2, len(values) - 1}
        indices.remove(min(removable, key=lambda n: min(abs(n - other) for other in indices if other != n)))
    return [values[index] for index in sorted(indices)]


def evidence_capture_plan(program: dict[str, Any], *, limit: int = 8) -> list[dict[str, Any]]:
    points = {0.03, 0.42, 0.68, 0.9, 0.97}
    hold = float((program.get("quality_contract") or {}).get("final_hold_start") or 0.8)
    points.add(min(0.97, hold + 0.05))
    for track in program.get("tracks") or []:
        if not isinstance(track, dict):
            continue
        for key in track.get("keyframes") or []:
            if isinstance(key, dict):
                time = float(key.get("t") or 0.0)
                if 0.08 < time < hold:
                    points.add(round(time, 4))
    values = sorted(points)
    budget = max(5, min(int(limit), 24))
    if len(values) > budget:
        indices = sorted({round(i * (len(values) - 1) / (budget - 1)) for i in range(budget)})
        values = [values[i] for i in indices]
    return [{"capture_id": f"evidence_{index:02d}", "fraction": value} for index, value in enumerate(values, 1)]


def build_verification_receipt(asset: Any, spec: dict[str, Any], frames: Iterable[Path], report: dict[str, Any]) -> dict[str, Any]:
    path = Path(asset.asset_path)
    frame_values = list(frames)
    unsigned = {
        "version": "vex-verification-receipt-v1",
        "asset_path": str(path.resolve()),
        "asset_sha256": file_digest(path),
        "program_sha256": payload_digest(spec.get("open_visual_program") or spec.get("scene_program_v2") or spec),
        "contract_sha256": payload_digest(spec.get("visual_communication_contract") or {}),
        "renderer": str(asset.renderer),
        "fps": float((asset.metadata or {}).get("fps") or 30),
        "width":int((asset.metadata or {}).get("width") or getattr(asset,"width",0)),
        "height":int((asset.metadata or {}).get("height") or getattr(asset,"height",0)),
        "frames": [{"path": str(Path(frame).resolve()), "sha256": file_digest(frame)} for frame in frame_values],
        "quality_state": report.get("selected_quality_state"),
        "passed": bool(report.get("passed")) and bool(frame_values),
        "verification_sha256": payload_digest(report),
    }
    return {**unsigned, "signature": payload_digest(unsigned)}


def validate_verification_receipt(receipt: dict[str, Any], asset_path: str | Path) -> bool:
    unsigned = {key: value for key, value in receipt.items() if key != "signature"}
    try:
        return bool(receipt.get("passed") and receipt.get("frames") and receipt.get("signature") == payload_digest(unsigned) and receipt.get("asset_sha256") == file_digest(asset_path) and all(frame.get("sha256") == file_digest(frame["path"]) for frame in receipt["frames"]))
    except (OSError, TypeError, ValueError):
        return False
