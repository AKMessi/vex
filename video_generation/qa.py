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
    director_package: dict[str, Any] | None = None,
    cinematic_plan: dict[str, Any] | None = None,
    motion_plan: dict[str, Any] | None = None,
    portfolio_judge: dict[str, Any] | None = None,
    visual_quality: dict[str, Any] | None = None,
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
    if director_package:
        evidence["director_package"] = director_package
        contracts = director_package.get("beat_contracts") or []
        if len(contracts) < len(beat_graph.beats):
            issues.append("director_contract_incomplete")
        if not director_package.get("brief"):
            issues.append("director_brief_missing")
    elif len(beat_graph.beats) >= 2:
        issues.append("director_package_missing")
    if cinematic_plan:
        evidence["cinematic_plan"] = cinematic_plan
        accepted = int(cinematic_plan.get("accepted_count") or 0)
        beat_count = int(cinematic_plan.get("beat_count") or len(beat_graph.beats))
        if beat_count and accepted / beat_count < 0.67:
            issues.append("semantic_cinematographer_coverage_below_67_percent")
        tournament_missing = [
            item
            for item in cinematic_plan.get("beat_compositions") or []
            if item.get("compiler_passed") and not item.get("tournament")
        ]
        if tournament_missing:
            issues.append("cinematic_beat_tournament_missing")
    if motion_plan:
        evidence["motion_plan"] = motion_plan
        native_count = int(motion_plan.get("native_composition_count") or 0)
        cue_count = int(motion_plan.get("audio_cue_count") or 0)
        capabilities = {
            str(item)
            for item in motion_plan.get("advanced_capabilities") or []
            if str(item).strip()
        }
        if len(beat_graph.beats) >= 2 and native_count / max(len(beat_graph.beats), 1) < 0.67:
            issues.append("native_motion_coverage_below_67_percent")
        if request.generate_audio and cue_count <= 0:
            warnings.append("audio_motion_cues_missing")
        required_capabilities = {
            "nested_compositions",
            "registered_seekable_timelines",
            "deterministic_audio_cues",
            "transition_overlays",
        }
        if not required_capabilities.issubset(capabilities):
            warnings.append("native_motion_capability_contract_incomplete")
    elif len(beat_graph.beats) >= 2:
        warnings.append("motion_plan_missing")
    if portfolio_judge:
        evidence["portfolio_judge"] = portfolio_judge
        if not bool(portfolio_judge.get("passed")):
            issues.append("portfolio_judge_failed")
            for item in portfolio_judge.get("issues") or []:
                warnings.append(f"portfolio:{item}")
    elif len(beat_graph.beats) >= 2:
        issues.append("portfolio_judge_missing")
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
    if visual_quality:
        evidence["visual_quality"] = visual_quality
        if not bool(visual_quality.get("passed")):
            issues.append("rendered_visual_quality_gate_failed")
            for item in visual_quality.get("issues") or []:
                warnings.append(f"visual_quality:{item}")
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
