from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from video_generation.film_direction import FilmDirection, revise_film_direction
from video_generation.film_project import build_film_project
from video_generation.film_review import review_film
from video_generation.models import BeatGraph, VideoGenerationRequest
from video_generation.remotion_delivery import check_remotion_delivery
from video_generation.remotion_project import _write_json, render_remotion_project
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
    commands: list[CommandRecord], *, narration: Path | None, music: Path | None,
) -> tuple[FilmRepair | None, Path | None]:
    flagged = [row for row in original_review["shots"] if row["verdict"] == "revise"]
    # Provider outages and incomplete observations cannot ground repairs.
    if not flagged or not 1 <= len(flagged) <= 3 or len(original_review["shots"]) != len(graph.beats):
        return None, None
    findings = {row["id"]: row["finding"] for row in flagged}
    report_path = project_dir / "film_repair.json"
    report = {"version": "vex-hybrid-film-repair-v1", "original_review": original_review, "findings": findings}
    candidate = project_dir / "film_repair"
    try:
        updated = revise_film_direction(direction, findings, [beat.beat_id for beat in graph.beats])
        candidate.mkdir()
        artifacts, shot_commands = build_film_project(
            candidate, request, graph, updated, runtime,
            narration=narration, music=music, reuse_project=project_dir,
        )
        commands.extend(shot_commands)
        timeline = Path(artifacts["timeline_path"])
        output, metadata, command = render_remotion_project(
            timeline.parent, output_format=request.output_format, workers=request.workers,
            auto_install_runtime=request.auto_install_runtime,
        )
        commands.append(command)
        local = check_remotion_delivery(timeline, output, metadata)
        report["candidate_delivery"] = local
        if local["passed"]:
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
                    {**artifacts, "film_review_path": str(review_path)},
                ), report_path
    except (ValueError, OSError, RuntimeError) as exc:
        report["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
    _write_json(report_path, {**report, "accepted": False})
    return None, report_path
