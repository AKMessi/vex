from __future__ import annotations

import json
import os
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from state import ProjectState, utc_now_iso
from timeline import normalize_timeline
from vex_runtime.edit_graph import EditGraph, rational, rational_text


NLE_EXPORT_SCHEMA_VERSION = 2
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
    graph = _validated_graph_for_state(state)
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
        "edit_graph": graph.to_dict() if graph is not None else None,
        "handoff_mode": graph.provenance if graph is not None else "flattened",
    }


def build_fcpxml(state: ProjectState, payload: dict[str, Any] | None = None) -> str:
    payload = payload or build_nle_timeline_payload(state)
    graph_payload = payload.get("edit_graph")
    if isinstance(graph_payload, dict):
        return _build_graph_fcpxml(state, payload, EditGraph.from_mapping(graph_payload))
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
    graph_payload = payload.get("edit_graph")
    if isinstance(graph_payload, dict):
        return _build_graph_edl(state, payload, EditGraph.from_mapping(graph_payload))
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


def _validated_graph_for_state(state: ProjectState) -> EditGraph | None:
    if not state.edit_graph:
        return None
    graph = EditGraph.from_mapping(state.edit_graph)
    working_dir = Path(state.working_dir).expanduser().resolve(strict=False)
    allowed_sources = {
        Path(item).expanduser().resolve(strict=False)
        for item in state.source_files
        if item
    }
    for source in graph.sources.values():
        path = Path(source.media_path).expanduser().resolve(strict=False)
        if not path.is_relative_to(working_dir) and path not in allowed_sources:
            raise ValueError(f"Edit graph source is outside project media roots: {path}")
    return graph


def _build_graph_fcpxml(state: ProjectState, payload: dict[str, Any], graph: EditGraph) -> str:
    if any(span.duration != span.source_end - span.source_start for span in graph.spans):
        raise ValueError("NLE export cannot represent retimed graph spans yet.")
    media = dict(payload.get("media") or {})
    width = int(_as_float(media.get("width"), 1920))
    height = int(_as_float(media.get("height"), 1080))
    fcpxml = ET.Element("fcpxml", {"version": "1.10"})
    resources = ET.SubElement(fcpxml, "resources")
    ET.SubElement(
        resources,
        "format",
        {
            "id": "r1",
            "name": f"Vex {width}x{height} {float(graph.fps):g}fps",
            "frameDuration": _fcpx_time(1 / graph.fps),
            "width": str(width),
            "height": str(height),
        },
    )
    source_refs: dict[str, str] = {}
    for index, source in enumerate(graph.sources.values(), start=2):
        ref = f"r{index}"
        source_refs[source.source_id] = ref
        ET.SubElement(
            resources,
            "asset",
            {
                "id": ref,
                "name": Path(source.media_path).name,
                "src": _file_uri(source.media_path),
                "start": "0s",
                "duration": _fcpx_time(source.duration),
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
            "duration": _fcpx_time(graph.duration),
            "format": "r1",
            "tcStart": "0s",
            "tcFormat": "NDF",
        },
    )
    spine = ET.SubElement(sequence, "spine")
    clips: list[ET.Element] = []
    for span in graph.spans:
        source = graph.sources[span.source_id]
        clips.append(
            ET.SubElement(
                spine,
                "asset-clip",
                {
                    "name": Path(source.media_path).name,
                    "ref": source_refs[span.source_id],
                    "offset": _fcpx_time(span.output_start),
                    "start": _fcpx_time(span.source_start),
                    "duration": _fcpx_time(span.duration),
                },
            )
        )
    for marker in payload.get("markers") or []:
        if not isinstance(marker, dict):
            continue
        marker_time = rational(marker.get("start_sec") or 0)
        if not 0 <= marker_time <= graph.duration:
            continue
        index = next(
            (i for i, span in enumerate(graph.spans) if span.output_start <= marker_time < span.output_end),
            len(graph.spans) - 1,
        )
        span = graph.spans[index]
        ET.SubElement(
            clips[index],
            "marker",
            {
                "start": _fcpx_time(span.source_at(marker_time)),
                "value": str(marker.get("label") or "Vex operation")[:255],
            },
        )
    ET.indent(fcpxml, space="  ")
    return ET.tostring(fcpxml, encoding="unicode", xml_declaration=True) + "\n"


def _build_graph_edl(state: ProjectState, payload: dict[str, Any], graph: EditGraph) -> str:
    if any(span.duration != span.source_end - span.source_start for span in graph.spans):
        raise ValueError("EDL export cannot represent retimed graph spans yet.")
    nominal_fps = max(int(round(float(graph.fps))), 1)
    source_numbers = {source_id: index for index, source_id in enumerate(graph.sources, start=1)}
    lines = [
        f"TITLE: {state.project_name or state.project_id or 'Vex Project'}",
        "FCM: NON-DROP FRAME",
        "",
    ]
    for index, span in enumerate(graph.spans, start=1):
        source = graph.sources[span.source_id]
        reel = f"A{source_numbers[span.source_id]:03d}"
        lines.append(
            f"{index:03d}  {reel:<8} V     C        "
            f"{_graph_timecode(span.source_start, graph.fps, nominal_fps)} "
            f"{_graph_timecode(span.source_end, graph.fps, nominal_fps)} "
            f"{_graph_timecode(span.output_start, graph.fps, nominal_fps)} "
            f"{_graph_timecode(span.output_end, graph.fps, nominal_fps)}"
        )
        lines.append(f"* FROM CLIP NAME: {Path(source.media_path).name}")
        lines.append(f"* SOURCE FILE: {source.media_path}")
    for index, op in enumerate(payload.get("operations") or [], start=1):
        if not isinstance(op, dict):
            continue
        description = str(op.get("description") or op.get("op") or "operation")
        lines.append(f"* VEX_OP {index:03d}: {op.get('op', 'unknown')} - {description}")
    return "\n".join(lines).rstrip() + "\n"


def _fcpx_time(value: Fraction) -> str:
    return f"{rational_text(value)}s"


def _graph_timecode(value: Fraction, fps: Fraction, nominal_fps: int) -> str:
    frames = value * fps
    rounded_frames = (2 * frames.numerator + frames.denominator) // (2 * frames.denominator)
    frame = rounded_frames % nominal_fps
    total_seconds = rounded_frames // nominal_fps
    seconds = total_seconds % 60
    minutes = (total_seconds // 60) % 60
    hours = total_seconds // 3600
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frame:02d}"


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
