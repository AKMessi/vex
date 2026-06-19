from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from engine import probe_video
from tools.creative_registry import record_creative_run
from video_generation.beat_graph import (
    build_initial_beat_graph,
    load_transcript_words,
    retime_beat_graph,
)
from video_generation.cinematographer import (
    build_cinematic_plan,
    evaluate_rendered_cinematography,
)
from video_generation.hyperframes_project import (
    build_index_html,
    copy_background_music,
    write_generation_project,
)
from video_generation.models import (
    GeneratedVideoResult,
    VideoGenerationRequest,
    make_project_dir,
    normalize_generation_request,
)
from video_generation.motion import build_motion_plan
from video_generation.qa import evaluate_generated_video, write_manifest
from video_generation.renderer import (
    CommandRecord,
    HyperframesVideoRuntime,
    VideoGenerationRuntimeError,
)
from video_generation.script_planner import build_script_plan


def generate_video(params: dict[str, Any]) -> GeneratedVideoResult:
    request = normalize_generation_request(params)
    project_dir = make_project_dir(request)
    plan = build_script_plan(request)
    runtime = HyperframesVideoRuntime(auto_install=request.auto_install_runtime)
    commands: list[CommandRecord] = []
    warnings: list[str] = []

    narration_path = project_dir / "narration.txt"
    narration_path.write_text(plan.narration + "\n", encoding="utf-8")
    beat_graph = build_initial_beat_graph(
        plan,
        target_duration_sec=request.duration_sec,
        voice_speed=request.voice_speed,
    )

    audio_path: Path | None = None
    transcript_path: Path | None = None
    output_metadata: dict[str, Any] | None = None
    if request.generate_audio:
        audio_path = project_dir / "audio" / "narration.wav"
        commands.append(
            runtime.tts(
                text_path=narration_path,
                output_path=audio_path,
                voice=request.voice,
                speed=request.voice_speed,
                language=request.language,
            )
        )
        audio_duration = _probe_duration(audio_path)
        if audio_duration > 0:
            beat_graph = build_initial_beat_graph(
                plan,
                target_duration_sec=audio_duration,
                voice_speed=request.voice_speed,
            )
        if request.transcribe_audio:
            try:
                transcript_path, command = runtime.transcribe(
                    media_path=audio_path,
                    project_dir=project_dir,
                )
                commands.append(command)
            except VideoGenerationRuntimeError as exc:
                if request.strict_audio_timing:
                    raise
                warnings.append(
                    "HyperFrames transcription failed; using estimated word timings"
                    f" ({_short_error(exc)})."
                )
            if transcript_path is not None:
                transcript_words = load_transcript_words(transcript_path)
                beat_graph = retime_beat_graph(
                    plan,
                    transcript_words=transcript_words,
                    duration_sec=audio_duration or beat_graph.duration_sec,
                    fallback=beat_graph,
                )
    else:
        transcript_path = _write_estimated_transcript(project_dir, beat_graph)

    background_music_path = copy_background_music(request.background_music_path, project_dir)
    cinematic_plan = build_cinematic_plan(
        request=request,
        plan=plan,
        beat_graph=beat_graph,
    )
    motion_plan = build_motion_plan(
        request=request,
        plan=plan,
        beat_graph=beat_graph,
        cinematic_plan=cinematic_plan,
    )
    artifact_paths = write_generation_project(
        project_dir=project_dir,
        request=request,
        plan=plan,
        beat_graph=beat_graph,
        audio_path=audio_path,
        transcript_path=transcript_path,
        background_music_path=background_music_path,
        cinematic_plan=cinematic_plan,
        motion_plan=motion_plan,
    )

    output_path = project_dir / "renders" / f"final.{request.output_format}"
    visual_quality: dict[str, Any] | None = None
    if request.render:
        commands.append(runtime.lint(project_dir=project_dir))
        output_metadata, render_record = runtime.render(
            project_dir=project_dir,
            output_path=output_path,
            fps=request.fps,
            quality=request.quality,
            output_format=request.output_format,
            resolution=request.render_resolution,
            workers=request.workers,
        )
        commands.append(render_record)
        root_html = build_index_html(
            request=request,
            plan=plan,
            beat_graph=beat_graph,
            audio_path=audio_path,
            background_music_path=background_music_path,
            cinematic_plan=cinematic_plan,
            motion_plan=motion_plan,
        )
        visual_quality = evaluate_rendered_cinematography(
            output_path=output_path,
            project_dir=project_dir,
            request=request,
            beat_graph=beat_graph,
            root_html=root_html,
            cinematic_plan=cinematic_plan,
            output_metadata=output_metadata,
        )

    qa = evaluate_generated_video(
        request=request,
        beat_graph=beat_graph,
        output_metadata=output_metadata,
        audio_path=audio_path,
        transcript_path=transcript_path,
        render_requested=request.render,
        cinematic_plan=cinematic_plan.to_dict(),
        motion_plan=motion_plan.to_dict(),
        visual_quality=visual_quality,
    )
    if warnings:
        combined_warnings = [*qa.warnings, *warnings]
        if any("HyperFrames transcription failed" in item for item in combined_warnings):
            combined_warnings = [
                item
                for item in combined_warnings
                if item != "transcription_requested_but_missing"
            ]
        qa = type(qa)(
            version=qa.version,
            passed=qa.passed,
            score=max(0.0, round(qa.score - 0.03 * len(warnings), 4)),
            issues=qa.issues,
            warnings=combined_warnings,
            evidence=qa.evidence,
        )

    qa_path = project_dir / "generated_video_qa.json"
    qa_path.write_text(json.dumps(qa.to_dict(), indent=2), encoding="utf-8")
    artifacts = {
        **artifact_paths,
        "project_dir": str(project_dir),
        "narration_text_path": str(narration_path),
        "audio_path": str(audio_path or ""),
        "background_music_path": str(background_music_path or ""),
        "output_path": str(output_path if request.render else ""),
        "qa_path": str(qa_path),
    }
    manifest_path = write_manifest(
        project_dir=project_dir,
        request=request,
        plan=plan.to_dict(),
        beat_graph=beat_graph,
        artifacts=artifacts,
        commands=[record.to_dict() for record in commands],
        qa=qa,
    )
    record_creative_run(
        working_dir=project_dir,
        feature="generate_video",
        manifest_path=str(manifest_path),
        output_path=str(output_path if request.render else ""),
        graph_version=beat_graph.version,
        quality_score=qa.score,
        summary={
            "title": plan.title,
            "duration_sec": beat_graph.duration_sec,
            "beat_count": len(beat_graph.beats),
            "cinematic_beat_count": cinematic_plan.accepted_count,
            "native_motion_beat_count": motion_plan.native_composition_count,
            "audio_motion_cue_count": motion_plan.audio_cue_count,
            "rendered": request.render,
            "has_audio": bool(audio_path),
            "timing_source": beat_graph.source,
        },
        artifacts=artifacts,
    )
    return GeneratedVideoResult(
        project_dir=str(project_dir),
        manifest_path=str(manifest_path),
        output_path=str(output_path if request.render else ""),
        index_path=str(artifact_paths["index_path"]),
        script_path=str(artifact_paths["script_path"]),
        storyboard_path=str(artifact_paths["storyboard_path"]),
        design_path=str(artifact_paths["design_path"]),
        beat_graph_path=str(artifact_paths["beat_graph_path"]),
        audio_path=str(audio_path or ""),
        transcript_path=str(transcript_path or ""),
        qa_path=str(qa_path),
        rendered=request.render,
        has_audio=bool(audio_path),
        duration_sec=beat_graph.duration_sec,
        warnings=qa.warnings,
    )


def _probe_duration(path: Path) -> float:
    try:
        return float(probe_video(str(path)).get("duration_sec") or 0.0)
    except Exception:
        return 0.0


def _short_error(exc: Exception) -> str:
    text = " ".join(str(exc).split())
    if len(text) > 240:
        return text[:237].rstrip(" ,.;:") + "..."
    return text


def _write_estimated_transcript(project_dir: Path, beat_graph) -> Path:
    transcript_path = project_dir / "transcript.estimated.json"
    transcript_path.write_text(
        json.dumps(
            {
                "source": "estimated_words",
                "duration_sec": beat_graph.duration_sec,
                "words": [word.to_dict() for word in beat_graph.words],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return transcript_path
