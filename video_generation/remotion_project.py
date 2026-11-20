from __future__ import annotations

import json
import math
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
from video_generation.cinematographer import build_beat_visual_specs
from video_generation.director import DirectorPackage
from video_generation.models import BeatGraph, ScriptPlan, VideoGenerationRequest
from video_generation.renderer import CommandRecord, VideoGenerationRuntimeError, _run_logged
from video_generation.skill_graph import VideoSkillGraph


TIMELINE_VERSION = "vex-remotion-video-v1"


def write_remotion_project(
    project_dir: Path,
    request: VideoGenerationRequest,
    beat_graph: BeatGraph,
    *,
    plan: ScriptPlan,
    director_package: DirectorPackage,
    video_skill_graph: VideoSkillGraph,
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

    fps = request.fps
    frame_count = max(len(beat_graph.beats), math.ceil(beat_graph.duration_sec * fps), 1)
    starts = [0]
    for index, beat in enumerate(beat_graph.beats[1:], 1):
        latest_start = frame_count - (len(beat_graph.beats) - index)
        starts.append(min(latest_start, max(starts[-1] + 1, round(beat.start * fps))))
    scenes = []
    shot_reports = []
    previous_medium = ""
    for beat, start, end in zip(beat_graph.beats, starts, [*starts[1:], frame_count]):
        frames = end - start
        specs = build_beat_visual_specs(
            request=request, plan=plan, beat=beat, visual_world_history=[],
            director_package=director_package, video_skill_graph=video_skill_graph,
        )
        program, report = _direct_shot(specs, request, frames, previous_medium)
        previous_medium = report["medium"]
        shot_reports.append({"id": beat.beat_id, **report})
        scenes.append({
            "id": beat.beat_id,
            "start_frame": start,
            "duration_frames": frames,
            "program": program,
            "standfirst": beat.caption or beat.title,
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
    shot_report_path = job_dir / "shot_direction.json"
    _write_json(shot_report_path, {
        "version": "vex-remotion-shot-direction-v1",
        "scene_graph_count": sum(scene["program"] is not None for scene in scenes),
        "shots": shot_reports,
    })
    return {
        "index_path": str(job_dir / "entry.jsx"),
        "timeline_path": str(timeline_path),
        "render_request_path": str(request_path),
        "shot_direction_path": str(shot_report_path),
    }


def _direct_shot(
    specs: list[dict[str, Any]], request: VideoGenerationRequest, frame_count: int,
    previous_medium: str,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    records = []
    selected = None
    chosen_medium = "typography"
    best_score = -1.0
    duration = frame_count / request.fps
    # Canvas in the React runtime lays out on an orientation-specific design
    # grid and scales that grid to the delivery resolution.
    aspect = request.width / request.height
    design_width, design_height = (
        (720, 1280) if aspect < 0.82 else (1280, 720) if aspect > 1.22 else (900, 900)
    )
    for index, spec in enumerate(specs):
        compilation = compile_remotion_scene_program(
            {**spec, "start": 0, "end": duration, "duration": duration},
            width=design_width, height=design_height, fps=min(120, max(15, request.fps)),
        )
        if compilation.program is None:
            records.append({"index": index, "issues": compilation.errors})
            continue
        program = compilation.program.to_dict()
        structure = evaluate_remotion_structure(program)
        medium = program["creative_direction"].get("medium_family") or program["scene_family"]
        score = 0.65 * program["semantic_score"] + 0.35 * structure.score
        if medium == previous_medium:
            score -= 0.05
        records.append({
            "index": index, "program_id": program["program_id"], "medium": medium,
            "score": round(score, 4), "issues": structure.issues,
        })
        if structure.passed and score > best_score:
            selected, chosen_medium, best_score = program, medium, score
    return selected, {
        "selected_program_id": selected["program_id"] if selected else "",
        "medium": chosen_medium,
        "candidates": records,
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
