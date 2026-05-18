from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from broll_intelligence import ensure_writable_dir, safe_stem, writable_dir_candidates
from effects import build_subtitle_cards, plan_subtitle_effects
from effects.qa import validate_effect_output
from engine import VideoEngineError, apply_timed_effects, probe_video
from state import ProjectState, restrict_timed_items_to_available_ranges, utc_now_iso
from tools.transcript import execute as transcribe
from tools.transcript_utils import load_transcript_bundle
from tools.undo import refresh_generated_overlay_ops
from visual_intelligence import detect_scene_cuts


def _emit_progress(message: str) -> None:
    print(f"[auto_effects] {message}", flush=True)


def _ensure_transcript_bundle(state: ProjectState) -> dict[str, object]:
    transcript_bundle = load_transcript_bundle(state.working_dir)
    if transcript_bundle.get("segments") or transcript_bundle.get("sentences"):
        return transcript_bundle
    result = transcribe({}, state)
    if not result["success"]:
        raise RuntimeError(result["message"])
    transcript_bundle = load_transcript_bundle(state.working_dir)
    if not transcript_bundle.get("segments") and not transcript_bundle.get("sentences"):
        raise RuntimeError("Transcript generation completed, but no usable subtitle segments were found.")
    return transcript_bundle


def _refresh_existing_auto_effects(state: ProjectState) -> dict[str, int]:
    return refresh_generated_overlay_ops(state, remove_ops={"add_auto_effects"})


def _subtitle_context_from_timeline(state: ProjectState, fallback: str) -> tuple[str, bool]:
    for op in reversed(state.timeline):
        if str(op.get("op") or "") != "burn_subtitles":
            continue
        position = str((op.get("params") or {}).get("position") or "").strip().lower()
        if position in {"bottom", "center", "top"}:
            return position, True
    return fallback if fallback in {"bottom", "center", "top"} else "bottom", False


def execute(params: dict[str, Any], state: ProjectState) -> dict[str, Any]:
    density = str(params.get("density") or "medium").strip().lower()
    if density not in {"low", "medium", "high"}:
        density = "medium"
    intensity = params.get("intensity", "medium")
    max_effects = max(1, min(int(params.get("max_effects", 12) or 12), 32))
    include_style_effects = _as_bool(params.get("include_style_effects"), True)
    subtitle_position, subtitle_highlight_enabled = _subtitle_context_from_timeline(
        state,
        str(params.get("subtitle_position") or "bottom").strip().lower(),
    )
    if "subtitle_highlight" in params:
        subtitle_highlight_enabled = _as_bool(params.get("subtitle_highlight"), subtitle_highlight_enabled)
    refresh_existing = _as_bool(params.get("refresh_existing"), True)

    try:
        if refresh_existing:
            refreshed = _refresh_existing_auto_effects(state)
            if refreshed.get("add_auto_effects"):
                count = refreshed["add_auto_effects"]
                _emit_progress(f"Cleared {count} prior auto-effects pass{'es' if count != 1 else ''} before replanning.")

        metadata = state.metadata or probe_video(state.working_file)
        clip_duration = float(metadata.get("duration_sec") or 0.0)
        if clip_duration <= 0.0:
            raise RuntimeError("The current working video does not have valid duration metadata.")

        _emit_progress("Loading subtitle and transcript timing...")
        transcript_bundle = _ensure_transcript_bundle(state)
        segments = list(transcript_bundle.get("segments") or [])
        sentences = list(transcript_bundle.get("sentences") or [])
        words = list(transcript_bundle.get("words") or [])

        _emit_progress("Detecting scene cuts for timing-safe emphasis windows...")
        scene_cuts = detect_scene_cuts(state.working_file)
        blocked_ranges = state.replace_overlay_ranges(exclude_ops={"add_auto_effects"})
        cards = build_subtitle_cards(
            segments,
            sentences,
            clip_duration,
            words=words,
            scene_cuts=scene_cuts,
        )
        cards = restrict_timed_items_to_available_ranges(
            cards,
            blocked_ranges,
            min_duration_sec=0.35,
        )
        if not cards:
            raise RuntimeError("No subtitle-aligned timing windows were available for auto effects.")

        _emit_progress("Planning subtitle-aware emphasis effects...")
        plan = plan_subtitle_effects(
            cards,
            clip_duration,
            max_effects=max_effects,
            density=density,
            intensity=intensity,
            include_style_effects=include_style_effects,
            subtitle_position=subtitle_position,
            subtitle_highlight_enabled=subtitle_highlight_enabled,
            blocked_ranges=blocked_ranges,
        )
        if not plan.effects:
            return {
                "success": False,
                "message": "No subtitle beats were strong enough for auto effects at the selected density.",
                "suggestion": None,
                "updated_state": state,
                "tool_name": "add_auto_effects",
            }

        timestamp_label = utc_now_iso().replace(":", "-").replace("+00:00", "Z")
        bundle_root = ensure_writable_dir(
            writable_dir_candidates(state.working_dir, state.output_dir, state.project_id, "auto_effect_bundles")
        )
        bundle_dir = bundle_root / f"{safe_stem(state.project_name)}_auto_effects_{timestamp_label}"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        filtergraph_path = bundle_dir / "filtergraph.txt"

        _emit_progress(f"Rendering {len(plan.effects)} subtitle-aware effect{'s' if len(plan.effects) != 1 else ''}...")
        source_metadata = dict(metadata)
        input_working_file = state.working_file
        output_path = apply_timed_effects(
            input_working_file,
            state.working_dir,
            plan.to_dict(),
            filtergraph_path=str(filtergraph_path),
        )
        output_metadata = probe_video(output_path)
        validation = validate_effect_output(source_metadata, output_metadata, plan)
        state.working_file = output_path
        state.metadata = output_metadata

        manifest = {
            "created_at": utc_now_iso(),
            "project_id": state.project_id,
            "project_name": state.project_name,
            "source_video": state.source_files[0] if state.source_files else state.working_file,
            "input_working_file": input_working_file,
            "source_metadata": source_metadata,
            "output_path": output_path,
            "density": density,
            "intensity": intensity,
            "include_style_effects": include_style_effects,
            "subtitle_position": subtitle_position,
            "subtitle_highlight_enabled": subtitle_highlight_enabled,
            "transcript_paths": transcript_bundle.get("paths", {}),
            "scene_cuts": scene_cuts,
            "blocked_ranges": blocked_ranges,
            "subtitle_cards": cards,
            "effect_plan": plan.to_dict(),
            "validation": validation,
            "filtergraph_path": str(filtergraph_path),
        }
        manifest_path = bundle_dir / "manifest.json"
        plan_path = bundle_dir / "effect_plan.json"
        validation_path = bundle_dir / "validation.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        plan_path.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
        validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")

        notes_lines = [
            "# Auto Effects Notes",
            "",
            f"Density: {density}",
            f"Subtitle position: {subtitle_position}",
            f"Subtitle highlight: {'enabled' if subtitle_highlight_enabled else 'disabled'}",
            f"Effects: {len(plan.effects)}",
            "",
        ]
        card_by_id = {str(card.get("card_id") or ""): card for card in cards}
        for effect in plan.effects:
            card = card_by_id.get(effect.source_card_id, {})
            notes_lines.extend(
                [
                    f"## {effect.start:.2f}s-{effect.end:.2f}s",
                    f"Type: {effect.effect_type}",
                    f"Subtitle: {card.get('text', '')}",
                    f"Reason: {effect.reason}",
                    "",
                ]
            )
        (bundle_dir / "notes.md").write_text("\n".join(notes_lines), encoding="utf-8")

        state.artifacts["latest_auto_effects"] = {
            "created_at": manifest["created_at"],
            "manifest_path": str(manifest_path),
            "bundle_dir": str(bundle_dir),
            "count": len(plan.effects),
            "density": density,
            "validation": validation,
        }
        history = list(state.artifacts.get("auto_effects_history") or [])
        history.append(state.artifacts["latest_auto_effects"])
        state.artifacts["auto_effects_history"] = history[-10:]
        state.apply_operation(
            {
                "op": "add_auto_effects",
                "params": {
                    "density": density,
                    "intensity": intensity,
                    "max_effects": max_effects,
                    "include_style_effects": include_style_effects,
                    "subtitle_position": subtitle_position,
                    "subtitle_highlight_enabled": subtitle_highlight_enabled,
                    "manifest_path": str(manifest_path),
                    "effect_plan": plan.to_dict(),
                },
                "timestamp": utc_now_iso(),
                "result_file": output_path,
                "description": f"Added {len(plan.effects)} subtitle-aware auto emphasis effects",
            }
        )
        warning_text = ""
        if validation.get("warnings"):
            warning_text = "\nWarnings:\n" + "\n".join(f"- {warning}" for warning in validation["warnings"])
        return {
            "success": True,
            "message": (
                f"Added {len(plan.effects)} subtitle-aware auto emphasis effects. "
                f"Manifest: {manifest_path}{warning_text}"
            ),
            "suggestion": None,
            "updated_state": state,
            "tool_name": "add_auto_effects",
            "plan": plan.to_dict(),
        }
    except (RuntimeError, VideoEngineError, OSError, ValueError) as exc:
        return {
            "success": False,
            "message": str(exc),
            "suggestion": None,
            "updated_state": state,
            "tool_name": "add_auto_effects",
        }


def _as_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default
