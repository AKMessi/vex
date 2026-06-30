from __future__ import annotations

import json
import os
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from state import ProjectState, utc_now_iso
from timeline import normalize_timeline


NLE_EXPORT_SCHEMA_VERSION = 1
SUPPORTED_NLE_FORMATS = {"json", "fcpxml", "edl"}


@dataclass(frozen=True)
class NLEExportResult:
    output_dir: str
    files: dict[str, str]


def export_nle_bundle(
    state: ProjectState,
    output_dir: str | Path | None = None,
    *,
    formats: set[str] | None = None,
) -> NLEExportResult:
    requested = set(formats or SUPPORTED_NLE_FORMATS)
    unknown = requested - SUPPORTED_NLE_FORMATS
    if unknown:
        raise ValueError(f"Unsupported NLE export format: {', '.join(sorted(unknown))}")

    target_dir = Path(output_dir) if output_dir is not None else Path(state.working_dir) / "nle_exports"
    target_dir.mkdir(parents=True, exist_ok=True)
    base_name = _safe_stem(state.project_name or state.project_id or "vex_project")
    payload = build_nle_timeline_payload(state)
    files: dict[str, str] = {}
    if "json" in requested:
        path = target_dir / f"{base_name}.timeline.json"
        _atomic_write_text(path, json.dumps(payload, indent=2) + "\n")
        files["json"] = str(path)
    if "fcpxml" in requested:
        path = target_dir / f"{base_name}.fcpxml"
        _atomic_write_text(path, build_fcpxml(state, payload))
        files["fcpxml"] = str(path)
    if "edl" in requested:
        path = target_dir / f"{base_name}.edl"
        _atomic_write_text(path, build_edl(state, payload))
        files["edl"] = str(path)
    return NLEExportResult(output_dir=str(target_dir), files=files)


def build_nle_timeline_payload(state: ProjectState) -> dict[str, Any]:
    metadata = dict(state.metadata or {})
    duration = _as_float(metadata.get("duration_sec"), 0.0)
    fps = _as_float(metadata.get("fps"), 30.0) or 30.0
    operations = normalize_timeline(state.timeline)
    return {
        "schema_version": NLE_EXPORT_SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "project": {
            "project_id": state.project_id,
            "project_name": state.project_name,
            "created_at": state.created_at,
            "updated_at": state.updated_at,
        },
        "media": {
            "source_files": list(state.source_files or []),
            "working_file": state.working_file,
            "duration_sec": duration,
            "fps": fps,
            "width": int(_as_float(metadata.get("width"), 1920)),
            "height": int(_as_float(metadata.get("height"), 1080)),
        },
        "operations": operations,
        "markers": _operation_markers(operations, duration_sec=duration),
    }


def build_fcpxml(state: ProjectState, payload: dict[str, Any] | None = None) -> str:
    payload = payload or build_nle_timeline_payload(state)
    media = dict(payload.get("media") or {})
    duration = max(_as_float(media.get("duration_sec"), 1.0), 1.0)
    fps = max(_as_float(media.get("fps"), 30.0), 1.0)
    width = int(_as_float(media.get("width"), 1920))
    height = int(_as_float(media.get("height"), 1080))
    working_file = str(media.get("working_file") or state.working_file)
    clip_name = Path(working_file).name or "Vex working cut"

    fcpxml = ET.Element("fcpxml", {"version": "1.10"})
    resources = ET.SubElement(fcpxml, "resources")
    ET.SubElement(
        resources,
        "format",
        {
            "id": "r1",
            "name": f"Vex {width}x{height} {fps:g}fps",
            "frameDuration": _frame_duration(fps),
            "width": str(width),
            "height": str(height),
        },
    )
    ET.SubElement(
        resources,
        "asset",
        {
            "id": "r2",
            "name": clip_name,
            "src": _file_uri(working_file),
            "start": "0s",
            "duration": _fcpx_duration(duration),
            "hasVideo": "1",
            "format": "r1",
        },
    )
    library = ET.SubElement(fcpxml, "library")
    event = ET.SubElement(library, "event", {"name": state.project_name or "Vex Project"})
    project = ET.SubElement(event, "project", {"name": state.project_name or "Vex Project"})
    sequence = ET.SubElement(
        project,
        "sequence",
        {
            "duration": _fcpx_duration(duration),
            "format": "r1",
            "tcStart": "0s",
            "tcFormat": "NDF",
        },
    )
    spine = ET.SubElement(sequence, "spine")
    asset_clip = ET.SubElement(
        spine,
        "asset-clip",
        {
            "name": clip_name,
            "ref": "r2",
            "offset": "0s",
            "start": "0s",
            "duration": _fcpx_duration(duration),
        },
    )
    for marker in payload.get("markers") or []:
        if not isinstance(marker, dict):
            continue
        ET.SubElement(
            asset_clip,
            "marker",
            {
                "start": _fcpx_duration(_as_float(marker.get("start_sec"), 0.0)),
                "value": str(marker.get("label") or "Vex operation")[:255],
            },
        )
    ET.indent(fcpxml, space="  ")
    return ET.tostring(fcpxml, encoding="unicode", xml_declaration=True) + "\n"


def build_edl(state: ProjectState, payload: dict[str, Any] | None = None) -> str:
    payload = payload or build_nle_timeline_payload(state)
    media = dict(payload.get("media") or {})
    duration = max(_as_float(media.get("duration_sec"), 1.0), 1.0)
    fps = max(int(round(_as_float(media.get("fps"), 30.0))), 1)
    working_file = str(media.get("working_file") or state.working_file)
    out_tc = _timecode(duration, fps=fps)
    lines = [
        f"TITLE: {state.project_name or state.project_id or 'Vex Project'}",
        "FCM: NON-DROP FRAME",
        "",
        f"001  AX       V     C        00:00:00:00 {out_tc} 00:00:00:00 {out_tc}",
        f"* FROM CLIP NAME: {Path(working_file).name}",
        f"* SOURCE FILE: {working_file}",
    ]
    for index, op in enumerate(payload.get("operations") or [], start=1):
        if not isinstance(op, dict):
            continue
        description = str(op.get("description") or op.get("op") or "operation")
        lines.append(f"* VEX_OP {index:03d}: {op.get('op', 'unknown')} - {description}")
    return "\n".join(lines).rstrip() + "\n"


def _operation_markers(operations: list[dict[str, Any]], *, duration_sec: float) -> list[dict[str, Any]]:
    markers: list[dict[str, Any]] = []
    for index, op in enumerate(operations, start=1):
        params = dict(op.get("params") or {})
        start = _as_float(params.get("start"), 0.0)
        if start <= 0 and params.get("overlays"):
            overlays = params.get("overlays")
            if isinstance(overlays, list) and overlays and isinstance(overlays[0], dict):
                start = _as_float(overlays[0].get("start"), 0.0)
        start = max(0.0, min(start, max(duration_sec, 0.0)))
        label = str(op.get("description") or op.get("op") or "Vex operation")
        markers.append(
            {
                "index": index,
                "start_sec": round(start, 3),
                "operation": str(op.get("op") or "unknown"),
                "label": f"{index}. {label}",
                "op_id": str(op.get("op_id") or ""),
            }
        )
    return markers


def _safe_stem(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value).strip("_")
    return cleaned[:80] or "vex_project"


def _as_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _frame_duration(fps: float) -> str:
    rounded = max(int(round(fps)), 1)
    return f"1/{rounded}s"


def _fcpx_duration(seconds: float) -> str:
    millis = max(int(round(float(seconds) * 1000)), 0)
    return f"{millis}/1000s"


def _timecode(seconds: float, *, fps: int) -> str:
    total_frames = max(int(round(seconds * fps)), 0)
    frames = total_frames % fps
    total_seconds = total_frames // fps
    secs = total_seconds % 60
    minutes = (total_seconds // 60) % 60
    hours = total_seconds // 3600
    return f"{hours:02d}:{minutes:02d}:{secs:02d}:{frames:02d}"


def _file_uri(path: str) -> str:
    return Path(path).expanduser().resolve(strict=False).as_uri()


def _atomic_write_text(path: Path, text: str) -> None:
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
            temp_file.write(text)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)
