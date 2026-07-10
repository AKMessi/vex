from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from engine import VideoEngineError, add_song_to_video, probe_video
from state import ProjectState, utc_now_iso
from tools.automation import create_unique_bundle_dir
from tools.creative_registry import record_creative_run
from tools.path_security import UnsafeInputPathError, resolve_existing_project_file
from tools.promotion import promote_working_file
from tools.song_director import (
    SONG_MIX_DIRECTOR_VERSION,
    build_song_mix_plan,
    evaluate_song_mix_output,
    manifest_json,
    write_song_mix_notes,
)


AUDIO_INPUT_SUFFIXES = {".aac", ".aiff", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".wma"}
MAX_SONG_BYTES = 500 * 1024 * 1024


def execute(params: dict[str, Any], state: ProjectState) -> dict[str, Any]:
    song_value = str(params.get("song_path") or params.get("audio_path") or params.get("music_path") or "").strip()
    if not song_value:
        return _failure("Missing song_path.", state)

    try:
        song_path = resolve_existing_project_file(
            song_value,
            state,
            allowed_suffixes=AUDIO_INPUT_SUFFIXES,
            max_size_bytes=MAX_SONG_BYTES,
        )
        input_working_file = state.working_file
        source_metadata = dict(state.metadata or probe_video(input_working_file))
        song_metadata = probe_video(str(song_path))
        plan = build_song_mix_plan(
            params=dict(params or {}),
            source_metadata=source_metadata,
            song_metadata=song_metadata,
        )

        bundle_dir = _bundle_dir(state, song_path)
        filtergraph_path = bundle_dir / "filtergraph.txt"
        plan_path = bundle_dir / "mix_plan.json"
        qa_path = bundle_dir / "audio_qa.json"
        manifest_path = bundle_dir / "manifest.json"
        notes_path = bundle_dir / "notes.md"
        plan_path.write_text(manifest_json(plan.to_dict()), encoding="utf-8")

        output_path = add_song_to_video(
            input_working_file,
            str(song_path),
            state.working_dir,
            plan.to_dict(),
            filtergraph_path=str(filtergraph_path),
        )
        output_metadata = probe_video(output_path)
        qa = evaluate_song_mix_output(
            source_path=input_working_file,
            output_path=output_path,
            source_metadata=source_metadata,
            output_metadata=output_metadata,
            plan=plan,
        )
        qa_path.write_text(manifest_json(qa), encoding="utf-8")

        manifest = {
            "created_at": utc_now_iso(),
            "version": SONG_MIX_DIRECTOR_VERSION,
            "project_id": state.project_id,
            "project_name": state.project_name,
            "input_working_file": input_working_file,
            "source_video": state.source_files[0] if state.source_files else input_working_file,
            "song_path": str(song_path),
            "output_path": output_path,
            "source_metadata": source_metadata,
            "song_metadata": song_metadata,
            "output_metadata": output_metadata,
            "mix_plan": plan.to_dict(),
            "qa": qa,
            "filtergraph_path": str(filtergraph_path),
            "plan_path": str(plan_path),
            "qa_path": str(qa_path),
        }
        manifest_path.write_text(manifest_json(manifest), encoding="utf-8")
        write_song_mix_notes(notes_path, plan=plan, qa=qa)

        if not qa.get("passed"):
            return {
                "success": False,
                "message": (
                    "Song mix was rendered but rejected by audio QA. "
                    f"Manifest: {manifest_path}. Issues: "
                    + "; ".join(str(issue) for issue in (qa.get("issues") or ["song mix QA failed"])[:4])
                ),
                "suggestion": "Try a lower volume, enable ducking, or choose a shorter timing window.",
                "updated_state": state,
                "tool_name": "add_song",
                "manifest_path": str(manifest_path),
                "output_path": output_path,
                "qa": qa,
                "plan": plan.to_dict(),
            }

        description = _description(plan.selected_skill_id, song_path.name)
        operation = {
            "op": "add_song",
            "params": {
                "song_path": str(song_path),
                "mode": plan.mode,
                "selected_skill_id": plan.selected_skill_id,
                "preserve_original_audio": plan.preserve_original_audio,
                "ducking_enabled": plan.ducking_enabled,
                "loop_song": plan.loop_song,
                "music_volume": plan.music_volume,
                "placements": [placement.to_dict() for placement in plan.placements],
                "manifest_path": str(manifest_path),
                "filtergraph_path": str(filtergraph_path),
                "mix_plan": plan.to_dict(),
                "qa": qa,
            },
            "timestamp": utc_now_iso(),
            "result_file": output_path,
            "description": description,
            "metadata": {
                "song_mix_manifest": str(manifest_path),
                "song_mix_score": qa.get("score"),
            },
        }
        promotion = promote_working_file(
            state,
            output_path,
            operation=operation,
            metadata=output_metadata,
            asset_source="add_song",
            asset_metadata={
                "song_path": str(song_path),
                "selected_skill_id": plan.selected_skill_id,
                "qa_score": qa.get("score"),
            },
        )
        registry_result = record_creative_run(
            working_dir=state.working_dir,
            feature="add_song",
            manifest_path=str(manifest_path),
            output_path=promotion.output_path,
            graph_version=plan.version,
            quality_score=float(qa.get("score") or 0.0),
            summary={
                "selected_skill_id": plan.selected_skill_id,
                "mode": plan.mode,
                "placement_count": len(plan.placements),
                "ducking_enabled": plan.ducking_enabled,
                "loop_song": plan.loop_song,
                "music_volume": plan.music_volume,
                "qa_passed": qa.get("passed"),
            },
            artifacts={
                "song_path": str(song_path),
                "filtergraph_path": str(filtergraph_path),
                "plan_path": str(plan_path),
                "qa_path": str(qa_path),
            },
        )
        latest = {
            "created_at": manifest["created_at"],
            "manifest_path": str(manifest_path),
            "bundle_dir": str(bundle_dir),
            "song_path": str(song_path),
            "output_path": promotion.output_path,
            "selected_skill_id": plan.selected_skill_id,
            "mode": plan.mode,
            "placement_count": len(plan.placements),
            "ducking_enabled": plan.ducking_enabled,
            "loop_song": plan.loop_song,
            "qa": qa,
            "creative_registry": registry_result,
            "asset_id": promotion.asset.asset_id,
        }
        state.artifacts["latest_added_song"] = latest
        history = list(state.artifacts.get("added_song_history") or [])
        history.append(latest)
        state.artifacts["added_song_history"] = history[-10:]
        state.save()

        warnings = [str(item) for item in qa.get("warnings") or [] if str(item).strip()]
        warning_text = ""
        if warnings:
            warning_text = " Warnings: " + "; ".join(warnings[:3]) + "."
        return {
            "success": True,
            "message": (
                f"{description}. Skill: {plan.selected_skill_id}. "
                f"QA {float(qa.get('score') or 0.0):.2f}. Manifest: {manifest_path}."
                f"{warning_text}"
            ),
            "suggestion": None,
            "updated_state": state,
            "tool_name": "add_song",
            "manifest_path": str(manifest_path),
            "output_path": promotion.output_path,
            "asset_id": promotion.asset.asset_id,
            "operation_id": promotion.operation["op_id"],
            "qa": qa,
            "plan": plan.to_dict(),
        }
    except (UnsafeInputPathError, ValueError, VideoEngineError, OSError) as exc:
        return _failure(str(exc), state)


def _bundle_dir(state: ProjectState, song_path: Path) -> Path:
    return create_unique_bundle_dir(
        Path(state.working_dir) / "song_mix_bundles",
        _safe_stem(song_path.stem),
    )


def _safe_stem(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value or "").strip())
    return cleaned.strip("_")[:80] or "song"


def _description(skill_id: str, song_name: str) -> str:
    label = {
        "voiceover_bed": "Added background song under source audio",
        "silent_video_soundtrack": "Added soundtrack to silent video",
        "replace_soundtrack": "Replaced soundtrack with song",
        "intro_sting": "Added intro song cue",
        "outro_sting": "Added outro song cue",
        "intro_outro_sting": "Added intro and outro song cues",
        "highlight_montage": "Added highlight song bed",
        "segment_music": "Added song to selected segment",
    }.get(skill_id, "Added song")
    return f"{label} using {song_name}"


def _failure(message: str, state: ProjectState) -> dict[str, Any]:
    return {
        "success": False,
        "message": message,
        "suggestion": None,
        "updated_state": state,
        "tool_name": "add_song",
    }
