from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vex_visuals.aesthetic_critic import evaluate_frame_aesthetics
from vex_visuals.evidence import file_digest
from vex_visuals.telemetry import evaluate_browser_telemetry
from video_generation.models import VideoGenerationRequest
from video_generation.remotion_project import run_remotion_job, stage_remotion_runtime
from video_generation.renderer import CommandRecord, VideoGenerationRuntimeError


SAMPLE_FRACTIONS = [0.0, 0.42, 0.68, 1.0]


def select_rendered_shots(
    job_dir: Path, request: VideoGenerationRequest,
) -> tuple[CommandRecord | None, list[str]]:
    candidate_path = job_dir / "shot_candidates.json"
    if not candidate_path.is_file():
        return None, []
    shortlist = _read(candidate_path)["shots"]
    timeline_path = job_dir / "input_props.json"
    shot_report_path = job_dir / "shot_direction.json"
    props = _read(timeline_path)
    report = _read(shot_report_path)
    timeline = props["timeline"]
    stage = job_dir / "shot_preflight"
    stage.mkdir()
    stage_remotion_runtime(stage)

    scale = min(1.0, 960 / max(timeline["width"], timeline["height"]))
    batch = []
    for shot in shortlist:
        for index, candidate in enumerate(shot["candidates"]):
            candidate["id"] = f"{shot['id']}_{index + 1:02d}"
            preview_program = {
                **candidate["program"],
                "width": max(320, round(timeline["width"] * scale)),
                "height": max(240, round(timeline["height"] * scale)),
            }
            batch.append({"candidate_id": candidate["id"], "input_props": {"program": preview_program}})
    _write(stage / "candidates.json", batch)
    _write(stage / "input_props.json", batch[0]["input_props"])
    _write(stage / "render_request.json", {
        "render_mode": "stills",
        "sample_fractions": SAMPLE_FRACTIONS,
        "candidate_input_props_file": "candidates.json",
    })
    command = None
    preflight = {
        "version": "vex-remotion-rendered-shot-selection-v1",
        "sample_fractions": SAMPLE_FRACTIONS,
        "shot_count": len(shortlist),
        "candidate_count": len(batch),
    }
    try:
        command = run_remotion_job(
            stage, workers=request.workers, auto_install_runtime=request.auto_install_runtime,
            name="remotion_shot_preflight",
        )
        if command.returncode:
            raise VideoGenerationRuntimeError(f"See {command.log_path}")
        captured = {item["candidate_id"]: item for item in _read(stage / "remotion_result.json")["candidate_stills"]}
    except (OSError, ValueError, KeyError, VideoGenerationRuntimeError) as exc:
        report["preflight"] = {**preflight, "status": "capture_failed", "error": str(exc)}
        _write(shot_report_path, report)
        return command, ["remotion_shot_preflight_capture_failed"]

    scenes = {item["id"]: item for item in timeline["scenes"]}
    directed = {item["id"]: item for item in report["shots"]}
    unresolved = []
    for shot in shortlist:
        shot_id = shot["id"]
        evaluations = [
            _evaluate_candidate(candidate, captured.get(candidate["id"], {}), job_dir)
            for candidate in shot["candidates"]
        ]
        accepted = [row for row in evaluations if row["accepted"]]
        selected = max(accepted, key=lambda row: row["score"], default=None)
        record = directed[shot_id]
        record["rendered_preflight"] = evaluations
        if selected is None:
            unresolved.append(shot_id)
            continue
        chosen = next(item for item in shot["candidates"] if item["id"] == selected["candidate_id"])
        record["selected_program_id"] = chosen["program"]["program_id"]
        record["medium"] = chosen["medium"]
        record["selection_basis"] = "rendered_preflight"
        scenes[shot_id]["program"] = chosen["program"]
    report["preflight"] = {**preflight, "status": "complete", "shots_without_passing_candidate": unresolved}
    _write(timeline_path, props)
    _write(shot_report_path, report)
    return command, ["remotion_shot_preflight_kept_structural_selection"] if unresolved else []


def _evaluate_candidate(candidate: dict[str, Any], captured: dict[str, Any], job_dir: Path) -> dict[str, Any]:
    frames = [Path(path) for path in captured.get("frame_paths", []) if Path(path).is_file()]
    program = candidate["program"]
    aesthetics = evaluate_frame_aesthetics(frames, program["creative_direction"])
    measurements = []
    for log in captured.get("browser_logs", []):
        text = str(log.get("text") or "")
        if text.startswith("VEX_TELEMETRY:"):
            try:
                measurements.append(json.loads(text.removeprefix("VEX_TELEMETRY:")))
            except ValueError:
                continue
    telemetry = evaluate_browser_telemetry(measurements)
    blocking = [
        issue for issue in telemetry["issues"]
        if issue == "browser_fonts_not_ready" or issue.startswith("browser_clipped_text:")
    ]
    score = 0.55 * candidate["score"] + 0.45 * aesthetics.score
    return {
        "candidate_id": candidate["id"],
        "program_id": program["program_id"],
        "preview_dimensions": [captured.get("width"), captured.get("height")],
        "accepted": len(frames) == len(SAMPLE_FRACTIONS) and telemetry["available"] and not blocking,
        "score": round(score, 4),
        "aesthetics": aesthetics.to_dict(),
        "browser_telemetry": telemetry,
        "frames": [{"path": path.relative_to(job_dir).as_posix(), "sha256": file_digest(path)} for path in frames],
    }


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
