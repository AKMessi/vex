from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict
from pathlib import Path
from typing import Any

import config
from engine import probe_video
from tools.creative_registry import record_creative_run
from video_generation.beat_graph import (
    build_beat_graph_from_words,
    build_initial_beat_graph,
    estimate_timed_words,
    load_transcript_words,
    retime_beat_graph,
)
from video_generation.cinematographer import (
    build_cinematic_plan,
    evaluate_rendered_cinematography,
)
from video_generation.director import build_director_package, direct_script_plan
from video_generation.hyperframes_project import (
    build_index_html,
    copy_background_music,
    write_generation_project,
)
from video_generation.models import (
    GeneratedVideoResult,
    ScriptPlan,
    VideoGenerationRequest,
    make_project_dir,
    normalize_generation_request,
)
from video_generation.motion import build_motion_plan
from video_generation.portfolio_judge import judge_generation_portfolio
from video_generation.qa import evaluate_generated_video, write_manifest
from video_generation.remotion_brief import author_remotion_script
from video_generation.remotion_delivery import check_remotion_delivery
from video_generation.remotion_preflight import select_rendered_shots
from video_generation.remotion_project import (
    _write_json,
    remotion_runtime_scope,
    render_remotion_project,
    write_remotion_project,
)
from video_generation.renderer import (
    CommandRecord,
    HyperframesVideoRuntime,
    VideoGenerationRuntimeError,
)
from video_generation.script_planner import build_script_plan
from video_generation.skill_graph import build_video_skill_graph
from vex_runtime.transcription import TranscriptionInstallError, transcribe_with_whisper


_generation_progress: ContextVar[Callable[[str, float], None] | None] = ContextVar(
    "generation_progress", default=None,
)


@contextmanager
def observe_generation(callback: Callable[[str, float], None]):
    """Observe generation through the normal tool transaction."""
    token = _generation_progress.set(callback)
    try:
        yield
    finally:
        _generation_progress.reset(token)


def generate_video(
    params: dict[str, Any], *, on_progress: Callable[[str, float], None] | None = None,
    checkpoint_dir: Path | None = None,
) -> GeneratedVideoResult:
    request = normalize_generation_request(params)
    if request.gemini_production and not config.GEMINI_API_KEY:
        raise ValueError("Gemini production requires a Gemini Developer API project key in GEMINI_API_KEY.")
    project_dir = make_project_dir(request)
    from vex_runtime.visual_run import visual_run
    with visual_run(checkpoint_dir or project_dir) as run, remotion_runtime_scope():
        return _generate_video(request, project_dir, on_progress or _generation_progress.get(), run if checkpoint_dir else None)


def _generate_video(
    request: VideoGenerationRequest, project_dir: Path, on_progress: Callable[[str, float], None] | None,
    checkpoint=None,
) -> GeneratedVideoResult:
    def stage(label: str, completed: float) -> None:
        if on_progress is not None:
            on_progress(label, completed)

    stage("Directing the film", 0.08)
    initial_plan = build_script_plan(request)
    film_direction = None
    direction_path = None
    if checkpoint and not request.film_direction_path:
        from providers import resolve_reasoning_model
        from video_generation.film_direction import parse_film_direction

        def direct() -> dict:
            script, direction, advice = _direct_generation(request, initial_plan)
            return {"plan": asdict(script), "direction": asdict(direction) if direction else None, "warnings": advice}

        inputs = asdict(request)
        inputs.pop("output_dir", None)
        inputs.pop("created_at", None)
        cached, reused = checkpoint.stage("production-direction-v1", {
            "request": inputs, "model": resolve_reasoning_model(), "pipeline": config.VERSION,
        }, direct)
        plan = ScriptPlan(**cached["plan"])
        warnings = cached["warnings"]
        if saved := cached["direction"]:
            film_direction = parse_film_direction(saved, saved["provider"], saved["model"])
        if reused:
            stage("Restored the saved film direction", 0.16)
    else:
        plan, film_direction, warnings = _direct_generation(request, initial_plan)
    if film_direction:
        direction_path = project_dir / "film_direction.json"
        _write_json(direction_path, asdict(film_direction))
    script_rewrite_applied = plan.narration != initial_plan.narration
    runtime = HyperframesVideoRuntime(auto_install=request.auto_install_runtime)
    commands: list[CommandRecord] = []

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
        stage("Recording narration", 0.24)
        audio_path = project_dir / "audio" / "narration.wav"
        def record() -> dict:
            command = runtime.tts(
                text_path=narration_path,
                output_path=audio_path,
                voice=request.voice,
                speed=request.voice_speed,
                language=request.language,
            )
            commands.append(command)
            return {"audio": str(audio_path)}

        if checkpoint:
            take, reused = checkpoint.stage("production-narration-v1", {
                "text": plan.narration, "voice": request.voice, "speed": request.voice_speed,
                "language": request.language, "pipeline": config.VERSION,
            }, record, output_paths=lambda result: [result["audio"]])
            if reused:
                audio_path.parent.mkdir(exist_ok=True)
                shutil.copyfile(take["audio"], audio_path)
                stage("Restored the recorded narration", 0.30)
        else:
            record()
        audio_duration = _probe_duration(audio_path)
        if audio_duration > 0 and request.renderer in {"remotion", "hybrid"}:
            beat_graph = build_beat_graph_from_words(
                plan, timed_words=estimate_timed_words(plan.narration, duration_sec=audio_duration),
                duration_sec=audio_duration, source="estimated_words",
            )
        elif audio_duration > 0:
            beat_graph = build_initial_beat_graph(
                plan,
                target_duration_sec=audio_duration,
                voice_speed=request.voice_speed,
            )
        if request.transcribe_audio:
            stage("Timing captions", 0.32)
            try:
                transcript_path, command = runtime.transcribe(
                    media_path=audio_path,
                    project_dir=project_dir,
                )
                commands.append(command)
            except VideoGenerationRuntimeError as exc:
                try:
                    transcript_path = _write_vex_whisper_transcript(
                        audio_path,
                        project_dir,
                    )
                except TranscriptionInstallError as fallback_exc:
                    if request.strict_audio_timing:
                        raise VideoGenerationRuntimeError(
                            "HyperFrames transcription failed and Vex Whisper "
                            "fallback also failed: "
                            f"{_short_error(exc)}; {_short_error(fallback_exc)}"
                        ) from fallback_exc
                    warnings.append(
                        "HyperFrames transcription failed; Vex Whisper fallback "
                        "also failed; using estimated word timings "
                        f"({_short_error(exc)}; {_short_error(fallback_exc)})."
                    )
                else:
                    warnings.append(
                        "HyperFrames transcription failed; used Vex Whisper "
                        f"fallback ({_short_error(exc)})."
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

    stage("Preparing the shots", 0.40)
    background_music_path = copy_background_music(request.background_music_path, project_dir)
    director_package = video_skill_graph = None
    if film_direction is None:
        director_package = build_director_package(
            request=request,
            plan=plan,
            beat_graph=beat_graph,
            script_rewrite_applied=script_rewrite_applied,
        )
        video_skill_graph = build_video_skill_graph(
            request=request,
            plan=plan,
            beat_graph=beat_graph,
            director_package=director_package,
        )
    cinematic_plan = motion_plan = portfolio_judge = None
    if request.renderer == "hyperframes":
        cinematic_plan = build_cinematic_plan(
            request=request,
            plan=plan,
            beat_graph=beat_graph,
            director_package=director_package,
            video_skill_graph=video_skill_graph,
        )
        motion_plan = build_motion_plan(
            request=request,
            plan=plan,
            beat_graph=beat_graph,
            cinematic_plan=cinematic_plan,
            video_skill_graph=video_skill_graph,
        )
        portfolio_judge = judge_generation_portfolio(
            request=request,
            plan=plan,
            beat_graph=beat_graph,
            cinematic_plan=cinematic_plan,
            motion_plan=motion_plan,
            director_package=director_package,
            video_skill_graph=video_skill_graph,
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
        director_package=director_package,
        video_skill_graph=video_skill_graph,
        portfolio_judge=portfolio_judge,
    )

    output_path = project_dir / "renders" / f"final.{request.output_format}"
    visual_quality: dict[str, Any] | None = None
    model_review: dict[str, Any] | None = None
    if request.renderer in {"remotion", "hybrid"}:
        if film_direction is not None:
            from video_generation.film_project import build_film_project

            remotion_artifacts, shot_commands = build_film_project(
                project_dir, request, beat_graph, film_direction, runtime,
                narration=audio_path, music=background_music_path,
            )
            commands.extend(shot_commands)
        else:
            remotion_artifacts = write_remotion_project(
                project_dir, request, beat_graph, plan=plan,
                director_package=director_package, video_skill_graph=video_skill_graph,
                narration=audio_path, music=background_music_path,
            )
        artifact_paths.update(remotion_artifacts)
        if direction_path:
            artifact_paths["film_direction_path"] = str(direction_path)
        if request.render:
            stage("Rendering the film", 0.62)
            timeline_path = Path(remotion_artifacts["timeline_path"])
            preflight_record, preflight_warnings = select_rendered_shots(timeline_path.parent, request)
            if preflight_record:
                commands.append(preflight_record)
            warnings.extend(preflight_warnings)
            output_path, output_metadata, render_record = render_remotion_project(
                timeline_path.parent,
                output_format=request.output_format,
                workers=request.workers,
                auto_install_runtime=request.auto_install_runtime,
            )
            commands.append(render_record)
            stage("Checking the exported frames", 0.78)
            visual_quality = check_remotion_delivery(timeline_path, output_path, output_metadata)
            if film_direction is not None and not request.film_direction_path and visual_quality["passed"]:
                from video_generation.film_review import review_film, supports_film_vision

                if supports_film_vision(film_direction):
                    model_review = review_film(
                        output_path, visual_quality, film_direction,
                        {beat.beat_id: beat.narration for beat in beat_graph.beats},
                    )
                    review_path = output_path.parent / "film_review.json"
                    review_path.write_text(json.dumps(model_review, indent=2), encoding="utf-8")
                    artifact_paths["film_review_path"] = str(review_path)
                    if not model_review["passed"]:
                        from video_generation.film_repair import attempt_film_repair

                        stage("Reviewing a targeted repair", 0.84)
                        repair, repair_path = attempt_film_repair(
                            project_dir, request, beat_graph, film_direction, model_review, runtime,
                            commands, original_delivery=visual_quality, narration=audio_path,
                            music=background_music_path,
                        )
                        if repair_path:
                            artifact_paths["film_repair_path"] = str(repair_path)
                        if repair:
                            output_path, output_metadata = repair.output, repair.metadata
                            visual_quality, model_review = repair.delivery, repair.review
                            artifact_paths.update(repair.artifacts)
            if request.gemini_production and visual_quality["passed"]:
                from video_generation.remotion_gemini import review_delivery

                model_review = review_delivery(
                    output_path, visual_quality, {beat.beat_id: beat.narration for beat in beat_graph.beats},
                )
                review_path = output_path.parent / "gemini_review.json"
                review_path.write_text(json.dumps(model_review, indent=2), encoding="utf-8")
                artifact_paths["gemini_review_path"] = str(review_path)
    elif request.render:
        stage("Rendering the film", 0.62)
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
            video_skill_graph=video_skill_graph,
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
        from vex_visuals.harness import verify_generated_portfolio
        shared_qa=verify_generated_portfolio(output_path,project_dir=project_dir,request=request,beat_graph=beat_graph,cinematic_plan=cinematic_plan,output_metadata=output_metadata,local_quality=visual_quality)
        visual_quality={**visual_quality,"shared_visual_harness":shared_qa,"passed":bool(shared_qa["passed"])}
        output_path=Path(shared_qa["output_path"])
        if shared_qa.get("repaired_overlays"):
            output_metadata=probe_video(str(output_path))

    stage("Saving the production and quality report", 0.94)
    qa = evaluate_generated_video(
        request=request,
        beat_graph=beat_graph,
        output_metadata=output_metadata,
        audio_path=audio_path,
        transcript_path=transcript_path,
        render_requested=request.render,
        director_package=director_package.to_dict() if director_package else None,
        cinematic_plan=cinematic_plan.to_dict() if cinematic_plan else None,
        motion_plan=motion_plan.to_dict() if motion_plan else None,
        portfolio_judge=portfolio_judge,
        skill_graph=video_skill_graph.to_dict() if video_skill_graph else None,
        visual_quality=visual_quality,
        model_review=model_review,
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
    has_audio = (
        bool(output_metadata.get("has_audio")) if output_metadata is not None
        else bool(audio_path or (background_music_path and request.music_volume > 0))
    )
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
        skill_graph=video_skill_graph.to_dict() if video_skill_graph else None,
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
            "renderer": request.renderer,
            "duration_sec": beat_graph.duration_sec,
            "beat_count": len(beat_graph.beats),
            "cinematic_beat_count": cinematic_plan.accepted_count if cinematic_plan else 0,
            "native_motion_beat_count": motion_plan.native_composition_count if motion_plan else 0,
            "audio_motion_cue_count": motion_plan.audio_cue_count if motion_plan else 0,
            "video_skill_graph": {
                "version": video_skill_graph.version,
                "production_skill_id": video_skill_graph.production_skill_id,
                "coverage": video_skill_graph.coverage,
                "passed": video_skill_graph.passed,
            } if video_skill_graph else {},
            "script_rewrite_applied": script_rewrite_applied,
            "portfolio_score": (portfolio_judge or {}).get("score"),
            "portfolio_passed": (portfolio_judge or {}).get("passed"),
            "rendered": request.render,
            "has_audio": has_audio,
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
        has_audio=has_audio,
        duration_sec=beat_graph.duration_sec,
        qa_passed=qa.passed,
        qa_score=qa.score,
        qa_issues=list(qa.issues),
        warnings=qa.warnings,
        renderer=request.renderer,
        timeline_path=str(artifact_paths.get("timeline_path", "")),
        title=plan.title,
    )


def _direct_generation(request: VideoGenerationRequest, initial_plan: ScriptPlan):
    if request.renderer == "hybrid":
        from video_generation.film_direction import apply_film_direction, direct_film, load_film_direction

        if request.film_direction_path:
            direction = load_film_direction(Path(request.film_direction_path))
            plan = apply_film_direction(initial_plan, direction)
            if request.generate_audio and not plan.narration.strip():
                raise VideoGenerationRuntimeError("The saved film is silent. Replay it with --no-audio.")
        else:
            plan, direction = direct_film(request, initial_plan)
        return plan, direction, []
    plan = direct_script_plan(request, initial_plan)
    if request.renderer == "remotion":
        plan, warnings = author_remotion_script(request, plan)
        if request.gemini_production and warnings:
            raise VideoGenerationRuntimeError(
                "Gemini could not author a valid script. Check API model access or supply --script."
            )
        return plan, None, warnings
    return plan, None, []


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


def _write_vex_whisper_transcript(media_path: Path, project_dir: Path) -> Path:
    payload = transcribe_with_whisper(
        media_path,
        model_name=config.WHISPER_MODEL,
        configured_python=config.WHISPER_PYTHON_PATH,
        timeout_sec=config.WHISPER_TRANSCRIBE_TIMEOUT_SEC,
    )
    if not isinstance(payload, dict):
        raise TranscriptionInstallError("Vex Whisper returned a non-object transcript.")
    transcript_path = project_dir / "transcript.vex-whisper.json"
    transcript_path.write_text(
        json.dumps(_with_estimated_segment_words(payload), indent=2),
        encoding="utf-8",
    )
    return transcript_path


def _with_estimated_segment_words(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("words"):
        return payload
    if any(
        isinstance(segment, dict) and segment.get("words")
        for segment in payload.get("segments") or []
    ):
        return payload
    words: list[dict[str, Any]] = []
    for segment in payload.get("segments") or []:
        if not isinstance(segment, dict):
            continue
        text = str(segment.get("text") or "").strip()
        tokens = [item for item in text.split() if item.strip()]
        if not tokens:
            continue
        try:
            start = float(segment.get("start") or 0.0)
            end = float(segment.get("end") or start)
        except (TypeError, ValueError):
            continue
        duration = max(end - start, 0.05)
        step = duration / max(len(tokens), 1)
        for index, token in enumerate(tokens):
            word_start = start + index * step
            words.append(
                {
                    "word": token,
                    "start": round(word_start, 3),
                    "end": round(min(end, word_start + step), 3),
                    "confidence": None,
                }
            )
    if not words:
        return payload
    return {**payload, "words": words}
