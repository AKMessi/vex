from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from engine import probe_video
from renderers.remotion_renderer import (
    _entry_template_path,
    _find_remotion_node_root,
    _remotion_bundle_cache_dir,
    _remotion_concurrency,
    _remotion_timeout_ms,
    _remotion_timeout_sec,
    _runner_path,
)
from vex_remotion import compile_remotion_scene_program, evaluate_remotion_structure
from vex_runtime.hyperframes import RuntimeInstallError, install_hyperframes_runtime, resolve_node_executable
from vex_visuals.fonts import font_runtime_source
from video_generation.cinematographer import CinematicPlan
from video_generation.models import BeatGraph, VideoGenerationRequest
from video_generation.renderer import CommandRecord, VideoGenerationRuntimeError, _run_logged


TIMELINE_VERSION = "vex-remotion-video-v1"


def write_remotion_project(
    project_dir: Path,
    request: VideoGenerationRequest,
    beat_graph: BeatGraph,
    cinematic_plan: CinematicPlan,
    *,
    narration: Path | None,
    music: Path | None,
) -> dict[str, str]:
    job_dir = project_dir / "remotion"
    job_dir.mkdir()
    source_dir = _entry_template_path().parent
    shutil.copyfile(_entry_template_path(), job_dir / "entry.jsx")
    for filename in ("remotion_scene_graph.jsx", "visual_motion.mjs", "visual_telemetry.mjs"):
        shutil.copyfile(source_dir / filename, job_dir / filename)
    (job_dir / "visual_fonts.mjs").write_text(font_runtime_source(), encoding="utf-8")

    specs = {item.beat_id: item.spec for item in cinematic_plan.beat_compositions}
    fps = request.fps
    frame_count = max(len(beat_graph.beats), round(beat_graph.duration_sec * fps), 1)
    starts = [0]
    for index, beat in enumerate(beat_graph.beats[1:], 1):
        latest_start = frame_count - (len(beat_graph.beats) - index)
        starts.append(min(latest_start, max(starts[-1] + 1, round(beat.start * fps))))
    scenes = []
    for beat, start, end in zip(beat_graph.beats, starts, [*starts[1:], frame_count]):
        spec = specs.get(beat.beat_id) or {
            "visual_id": beat.beat_id,
            "headline": beat.title,
            "sentence_text": beat.narration,
            "context_text": beat.narration,
        }
        frames = end - start
        compilation = compile_remotion_scene_program(
            {**spec, "start": 0, "end": frames / fps, "duration": frames / fps},
            width=request.width, height=request.height, fps=fps,
        )
        program = compilation.program.to_dict() if compilation.passed and compilation.program else None
        issues = list(compilation.errors)
        if program:
            structure = evaluate_remotion_structure(program)
            if not structure.passed:
                issues.extend(structure.issues)
                program = None
        scenes.append({
            "id": beat.beat_id,
            "start_frame": start,
            "duration_frames": frames,
            "program": program,
            "standfirst": beat.caption or beat.title,
            "compiler_issues": issues,
        })

    timeline = {
        "version": TIMELINE_VERSION,
        "width": request.width,
        "height": request.height,
        "fps": fps,
        "duration_frames": frame_count,
        "timing_source": beat_graph.source,
        "scenes": scenes,
        "captions": _caption_cues(beat_graph, fps, frame_count) if narration else [],
        "audio": {
            "narration": _stage_media(job_dir, narration, "narration"),
            "music": _stage_media(job_dir, music, "music") if request.music_volume > 0 else "",
            "music_volume": request.music_volume,
        },
    }
    timeline_path = job_dir / "input_props.json"
    _write_json(timeline_path, {"timeline": timeline})
    request_path = job_dir / "render_request.json"
    _write_json(request_path, {
        "render_mode": "video",
        "video_format": request.output_format,
        "video_quality": request.quality,
        "workers": request.workers,
    })
    return {
        "index_path": str(job_dir / "entry.jsx"),
        "timeline_path": str(timeline_path),
        "render_request_path": str(request_path),
    }


def render_remotion_project(
    job_dir: Path, request: VideoGenerationRequest,
) -> tuple[Path, dict[str, Any], CommandRecord]:
    node_path = resolve_node_executable()
    node_root, reason = _find_remotion_node_root(node_path)
    if node_root is None and request.auto_install_runtime:
        try:
            install_hyperframes_runtime(force=False)
        except RuntimeInstallError as exc:
            raise VideoGenerationRuntimeError(f"Remotion runtime installation failed: {exc}") from exc
        node_path = resolve_node_executable()
        node_root, reason = _find_remotion_node_root(node_path)
    if node_path is None or node_root is None:
        raise VideoGenerationRuntimeError(reason)
    env = {
        **os.environ,
        "VEX_REMOTION_NODE_ROOT": str(node_root),
        "VEX_REMOTION_TIMEOUT_MS": str(_remotion_timeout_ms()),
        "VEX_REMOTION_BUNDLE_CACHE_DIR": str(_remotion_bundle_cache_dir()),
    }
    concurrency = request.workers or _remotion_concurrency()
    if concurrency and concurrency != "auto":
        env["VEX_REMOTION_CONCURRENCY"] = concurrency
    elif concurrency == "auto":
        env.pop("VEX_REMOTION_CONCURRENCY", None)
    timeout = _remotion_timeout_sec()
    output = job_dir / f"final.{request.output_format}"
    output.unlink(missing_ok=True)
    record = _run_logged(
        [node_path, str(_runner_path()), str(job_dir)],
        cwd=job_dir,
        name="remotion_render",
        log_path=job_dir.parent / "logs" / "remotion_render.log",
        timeout=timeout + 45 if timeout else None,
        env=env,
    )
    if record.returncode or not output.is_file():
        raise VideoGenerationRuntimeError(f"Remotion render failed. See {record.log_path}")
    return output, probe_video(str(output)), record


def check_remotion_delivery(timeline_path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))["timeline"]
    expected_duration = timeline["duration_frames"] / timeline["fps"]
    issues = [
        f"rendered_{key}_mismatch"
        for key in ("width", "height", "fps")
        if timeline[key] != metadata.get(key)
    ]
    # Audio muxing can add one packet to the container duration.
    if abs(expected_duration - metadata["duration_sec"]) > max(1 / timeline["fps"], 0.05):
        issues.append("rendered_frame_clock_duration_mismatch")
    if (timeline["audio"]["narration"] or timeline["audio"]["music"]) and not metadata["has_audio"]:
        issues.append("rendered_audio_track_missing")
    return {
        "passed": not issues,
        "issues": issues,
        "expected_duration_sec": expected_duration,
        "scene_count": len(timeline["scenes"]),
        "typography_fallbacks": [scene["id"] for scene in timeline["scenes"] if not scene["program"]],
    }


def _caption_cues(beat_graph: BeatGraph, fps: int, frame_count: int) -> list[dict[str, Any]]:
    groups = []
    for word in beat_graph.words:
        if not word.text.strip() or word.end <= word.start:
            continue
        if groups:
            group = groups[-1]
            if len(group) < 6 and word.start - group[-1].end < 0.35 and word.end - group[0].start <= 2:
                group.append(word)
                continue
        groups.append([word])

    cues = []
    for index, group in enumerate(groups):
        start = max(0, round(group[0].start * fps))
        end = min(frame_count, max(start + 1, round(group[-1].end * fps)))
        if index + 1 < len(groups):
            end = min(end, round(groups[index + 1][0].start * fps))
        if end > start:
            cues.append({
                "start_frame": start,
                "duration_frames": end - start,
                "text": " ".join(word.text for word in group),
            })
    return cues


def _stage_media(job_dir: Path, source: Path | None, role: str) -> str:
    if source is None:
        return ""
    public_dir = job_dir / "public"
    public_dir.mkdir(exist_ok=True)
    filename = f"{role}{source.suffix.lower()}"
    shutil.copyfile(source, public_dir / filename)
    return filename


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
