from __future__ import annotations

import json
import math
import os
import shutil
from bisect import bisect_right
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
    stage_remotion_runtime(job_dir)

    fps = request.fps
    frame_count = max(len(beat_graph.beats), math.ceil(beat_graph.duration_sec * fps), 1)
    starts = [0]
    for index, beat in enumerate(beat_graph.beats[1:], 1):
        latest_start = frame_count - (len(beat_graph.beats) - index)
        starts.append(min(latest_start, max(starts[-1] + 1, round(beat.start * fps))))
    scenes = []
    shot_reports = []
    candidates = []
    previous_medium = ""
    last_shot = len(beat_graph.beats) - 1
    gemini_shots = (
        {round(i * last_shot / 3) for i in range(4)} if request.gemini_production else set()
    )
    for index, (beat, start, end) in enumerate(zip(beat_graph.beats, starts, [*starts[1:], frame_count])):
        frames = end - start
        specs = build_beat_visual_specs(
            request=request, plan=plan, beat=beat, visual_world_history=[],
            director_package=director_package, video_skill_graph=video_skill_graph,
        )
        focus = plan.shot_focus[index] if index < len(plan.shot_focus) else ""
        if focus:
            specs = [{**spec, "display_title": focus} for spec in specs]
        shortlist, report = _direct_shot(
            specs, request, frames, previous_medium, use_gemini=index in gemini_shots,
        )
        program = shortlist[0]["program"] if shortlist else None
        previous_medium = report["medium"]
        shot_reports.append({"id": beat.beat_id, "brief_focus": focus, **report})
        if len(shortlist) > 1:
            candidates.append({"id": beat.beat_id, "candidates": shortlist})
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
    if request.render and (request.quality == "high" or request.gemini_production) and candidates:
        # The runner can render eight candidates in one browser session.
        indices = (
            range(len(candidates)) if len(candidates) <= 4
            else [round(i * (len(candidates) - 1) / 3) for i in range(4)]
        )
        _write_json(job_dir / "shot_candidates.json", {"shots": [candidates[i] for i in indices]})
    return {
        "index_path": str(job_dir / "entry.jsx"),
        "timeline_path": str(timeline_path),
        "render_request_path": str(request_path),
        "shot_direction_path": str(shot_report_path),
    }


def _direct_shot(
    specs: list[dict[str, Any]], request: VideoGenerationRequest, frame_count: int,
    previous_medium: str, *, use_gemini: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = []
    accepted: dict[str, dict[str, Any]] = {}
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
        empty_panels = [
            f"empty_bound_panel:{node['node_id']}"
            for node in program["scene_graph"]["nodes"]
            if node["primitive"] in {"graph_node", "semantic_token"}
            and node["binding"].get("kind") == "object"
            and not node["decorative"]
            and not node["content"].get("text", "").strip()
            and node["role"] != "resolved_outcome"
        ]
        medium = program["creative_direction"].get("medium_family") or program["scene_family"]
        score = 0.65 * program["semantic_score"] + 0.35 * structure.score
        if medium == previous_medium:
            score -= 0.05
        records.append({
            "index": index, "program_id": program["program_id"], "medium": medium,
            "score": round(score, 4), "issues": [*structure.issues, *empty_panels],
        })
        if structure.passed and not empty_panels and program["program_id"] not in accepted:
            accepted[program["program_id"]] = {
                "program": program, "score": score, "medium": medium, "index": index, "origin": "local",
            }
    ranked = sorted(accepted.values(), key=lambda item: item["score"], reverse=True)[:2]
    gemini = {}
    if use_gemini and ranked:
        from video_generation.remotion_gemini import propose_shot

        base = {**specs[ranked[0]["index"]], "start": 0, "end": duration, "duration": duration}
        proposal, gemini = propose_shot(
            base, width=design_width, height=design_height, fps=min(120, max(15, request.fps)),
        )
        if proposal:
            variants, proposal_report = _direct_shot([proposal], request, frame_count, previous_medium)
            gemini["compilation"] = proposal_report["candidates"]
            if variants and variants[0]["program"]["program_id"] not in accepted:
                ranked = [ranked[0], {**variants[0], "origin": "gemini"}]
    return ranked, {
        "selected_program_id": ranked[0]["program"]["program_id"] if ranked else "",
        "medium": ranked[0]["medium"] if ranked else "typography",
        "candidates": records,
        **({"gemini_authoring": gemini} if gemini else {}),
    }


def render_remotion_project(
    job_dir: Path, *, output_format: str, workers: str = "", auto_install_runtime: bool = True,
) -> tuple[Path, dict[str, Any], CommandRecord]:
    from video_generation.film_artwork import validate_artwork

    props_path = job_dir / "input_props.json"
    if props_path.is_file():
        timeline = json.loads(props_path.read_text(encoding="utf-8"))["timeline"]
        for scene in timeline["scenes"]:
            if "artwork" in scene:
                validate_artwork(scene["artwork"])
    output = job_dir / f"final.{output_format}"
    output.unlink(missing_ok=True)
    record = run_remotion_job(job_dir, workers=workers, auto_install_runtime=auto_install_runtime)
    if record.returncode or not output.is_file():
        raise VideoGenerationRuntimeError(f"Remotion render failed. See {record.log_path}")
    return output, probe_video(str(output)), record


def run_remotion_job(
    job_dir: Path, *, workers: str, auto_install_runtime: bool, name: str = "remotion_render",
) -> CommandRecord:
    job_dir = job_dir.resolve(strict=True)
    node_path = resolve_node_executable()
    node_root, reason = _find_remotion_node_root(node_path)
    if node_root is None and auto_install_runtime:
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
    concurrency = workers or _remotion_concurrency()
    if concurrency and concurrency != "auto":
        env["VEX_REMOTION_CONCURRENCY"] = concurrency
    elif concurrency == "auto":
        env.pop("VEX_REMOTION_CONCURRENCY", None)
    timeout = _remotion_timeout_sec()
    return _run_logged(
        [node_path, str(_runner_path()), str(job_dir)],
        # Remotion chooses its browser cache relative to cwd. Use the shared
        # Node runtime so external, deeply nested job folders reuse the install.
        cwd=Path(node_root).resolve(strict=True),
        name=name,
        log_path=job_dir / f"{name}.log",
        timeout=timeout + 45 if timeout else None,
        env=env,
    )


def stage_remotion_runtime(job_dir: Path) -> None:
    entry_path = _entry_template_path()
    shutil.copyfile(entry_path, job_dir / "entry.jsx")
    for filename in ("remotion_scene_graph.jsx", "visual_motion.mjs", "visual_telemetry.mjs", "film_motion.mjs"):
        shutil.copyfile(entry_path.with_name(filename), job_dir / filename)
    (job_dir / "visual_fonts.mjs").write_text(font_runtime_source(), encoding="utf-8")


def _caption_cues(
    beat_graph: BeatGraph, fps: int, frame_count: int, *, cut_frames: tuple[int, ...] = (),
) -> list[dict[str, Any]]:
    groups = []
    for word in beat_graph.words:
        if not word.text.strip() or word.end <= word.start:
            continue
        if groups:
            group = groups[-1]
            same_shot = bisect_right(cut_frames, round(word.start * fps)) == bisect_right(
                cut_frames, round(group[0].start * fps),
            )
            if same_shot and len(group) < 6 and word.start - group[-1].end < 0.35 and word.end - group[0].start <= 2:
                group.append(word)
                continue
        groups.append([word])

    cues = []
    for index, group in enumerate(groups):
        start = max(0, round(group[0].start * fps))
        end = min(frame_count, max(start + 1, round(group[-1].end * fps)))
        cut = bisect_right(cut_frames, start)
        if cut < len(cut_frames):
            end = min(end, cut_frames[cut])
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
