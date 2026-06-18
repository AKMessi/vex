from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from video_generation.models import BeatGraph, VideoGenerationRequest


QA_VERSION = "generated-video-qa-v1"


@dataclass(frozen=True)
class GeneratedVideoQA:
    version: str
    passed: bool
    score: float
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_generated_video(
    *,
    request: VideoGenerationRequest,
    beat_graph: BeatGraph,
    output_metadata: dict[str, Any] | None,
    audio_path: Path | None,
    transcript_path: Path | None,
    render_requested: bool,
) -> GeneratedVideoQA:
    issues: list[str] = []
    warnings: list[str] = list(beat_graph.warnings)
    evidence: dict[str, Any] = {
        "beat_count": len(beat_graph.beats),
        "timing_source": beat_graph.source,
        "render_requested": render_requested,
        "audio_requested": request.generate_audio,
        "audio_path": str(audio_path or ""),
        "transcript_path": str(transcript_path or ""),
    }
    if len(beat_graph.beats) < 2:
        issues.append("beat_graph_has_too_few_beats")
    if request.generate_audio and audio_path is None:
        issues.append("audio_generation_requested_but_missing")
    if request.transcribe_audio and transcript_path is None:
        warning = "transcription_requested_but_missing"
        if request.strict_audio_timing:
            issues.append(warning)
        else:
            warnings.append(warning)
    if output_metadata:
        output_duration = float(output_metadata.get("duration_sec") or 0.0)
        evidence["output_metadata"] = output_metadata
        if output_duration <= 0:
            issues.append("rendered_output_has_no_duration")
        elif abs(output_duration - beat_graph.duration_sec) > max(0.75, beat_graph.duration_sec * 0.08):
            warnings.append("rendered_duration_differs_from_beat_graph")
        if request.generate_audio and not bool(output_metadata.get("has_audio")):
            issues.append("rendered_output_has_no_audio_stream")
    elif render_requested:
        issues.append("render_requested_but_no_output_metadata")
    score = _score(issues=issues, warnings=warnings, beat_graph=beat_graph)
    return GeneratedVideoQA(
        version=QA_VERSION,
        passed=not issues,
        score=score,
        issues=issues,
        warnings=warnings,
        evidence=evidence,
    )


def write_manifest(
    *,
    project_dir: Path,
    request: VideoGenerationRequest,
    plan: dict[str, Any],
    beat_graph: BeatGraph,
    artifacts: dict[str, Any],
    commands: list[dict[str, Any]],
    qa: GeneratedVideoQA,
) -> Path:
    manifest_path = project_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": "generated-video-manifest-v1",
                "request": request.to_dict(),
                "plan": plan,
                "beat_graph": beat_graph.to_dict(),
                "artifacts": artifacts,
                "commands": commands,
                "qa": qa.to_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path


def _score(
    *,
    issues: list[str],
    warnings: list[str],
    beat_graph: BeatGraph,
) -> float:
    score = 1.0
    score -= min(len(issues) * 0.22, 0.7)
    score -= min(len(warnings) * 0.045, 0.2)
    if beat_graph.source == "hyperframes_transcript":
        score += 0.04
    if len(beat_graph.beats) >= 3:
        score += 0.03
    return round(max(0.0, min(score, 1.0)), 4)
