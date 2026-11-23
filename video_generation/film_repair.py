from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from video_generation.film_comparison import compare_film_repair, reviewed_frames
from video_generation.film_direction import FilmDirection, revise_film_direction
from video_generation.film_project import build_film_project
from video_generation.film_review import review_film
from video_generation.models import BeatGraph, VideoGenerationRequest
from video_generation.remotion_incremental import render_revision_delivery
from video_generation.remotion_project import _write_json
from video_generation.renderer import CommandRecord, HyperframesVideoRuntime


@dataclass(frozen=True)
class FilmRepair:
    output: Path
    metadata: dict[str, Any]
    delivery: dict[str, Any]
    review: dict[str, Any]
    artifacts: dict[str, str]


def attempt_film_repair(
    project_dir: Path, request: VideoGenerationRequest, graph: BeatGraph,
    direction: FilmDirection, original_review: dict[str, Any], runtime: HyperframesVideoRuntime,
    commands: list[CommandRecord], *, original_delivery: dict[str, Any],
    narration: Path | None, music: Path | None,
) -> tuple[FilmRepair | None, Path | None]:
    flagged = [row for row in original_review["shots"] if row["verdict"] == "revise"]
    # Provider outages and incomplete observations cannot ground repairs.
    if not flagged or not 1 <= len(flagged) <= 3 or len(original_review["shots"]) != len(graph.beats):
        return None, None
    findings = {row["id"]: row["finding"] for row in flagged}
    report_path = project_dir / "film_repair.json"
    report = {"version": "vex-hybrid-film-repair-v2", "original_review": original_review, "findings": findings}
    candidate = project_dir / "film_repair"
    try:
        original_root = project_dir / "remotion"
        if original_review["delivery_receipt_sha256"] != original_delivery["receipt_sha256"]:
            raise ValueError("The original review does not match this delivery.")
        original_frames = reviewed_frames(original_root, original_delivery, findings.keys())
        beat_ids = [beat.beat_id for beat in graph.beats]
        updated = revise_film_direction(direction, findings, beat_ids, evidence=original_frames)
        changed = {
            beat_id for beat_id, before, after in zip(beat_ids, direction.shots, updated.shots, strict=True)
            if before != after
        }
        candidate.mkdir()
        direction_path = candidate / "film_direction.json"
        _write_json(direction_path, asdict(updated))
        artifacts, shot_commands = build_film_project(
            candidate, request, graph, updated, runtime,
            narration=narration, music=music, reuse_project=project_dir,
        )
        commands.extend(shot_commands)
        timeline = Path(artifacts["timeline_path"])
        comparison_dir = candidate / "comparison"
        comparison = compare_film_repair(
            original_root / "input_props.json", timeline, comparison_dir, changed, commands,
            workers=request.workers, auto_install_runtime=request.auto_install_runtime,
        )
        report["comparison"] = {
            **comparison, "path": (comparison_dir / "comparison.json").relative_to(project_dir).as_posix(),
        }
        if not comparison["passed"]:
            _write_json(report_path, {**report, "accepted": False})
            return None, report_path
        output, metadata, delivery_commands, local, rendering = render_revision_delivery(
            original_root, timeline.parent, output_format=request.output_format, workers=request.workers,
            auto_install_runtime=request.auto_install_runtime,
        )
        commands.extend(delivery_commands)
        report["rendering"] = rendering
        report["candidate_delivery"] = local
        if local["passed"]:
            # The delivery judge assesses the candidate independently of the repair diagnosis.
            review = review_film(
                output, local, updated, {beat.beat_id: beat.narration for beat in graph.beats},
            )
            report["candidate_review"] = review
            if review["passed"]:
                review_path = output.parent / "film_review.json"
                _write_json(review_path, review)
                _write_json(report_path, {**report, "accepted": True})
                return FilmRepair(
                    output, metadata, local, review,
                    {**artifacts, "film_review_path": str(review_path), "film_direction_path": str(direction_path)},
                ), report_path
    except (ValueError, OSError, RuntimeError) as exc:
        report["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
    _write_json(report_path, {**report, "accepted": False})
    return None, report_path
