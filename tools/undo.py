from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path

from engine import (
    apply_visual_overlays,
    apply_color_grade,
    VideoEngineError,
    add_text,
    adjust_speed,
    burn_subtitles,
    extract_segments,
    fade_in,
    fade_out,
    merge,
    mute_segment,
    probe_video,
    replace_audio,
    trim,
    trim_silence,
)
from state import ProjectState, utc_now_iso


def _load_visual_overlays(params: dict) -> list[dict]:
    overlays = list(params.get("overlays") or [])
    if overlays:
        return overlays
    manifest_path = str(params.get("manifest_path") or "").strip()
    if not manifest_path:
        return []
    manifest_file = Path(manifest_path)
    if not manifest_file.is_file():
        raise VideoEngineError(f"Cannot rebuild project because manifest is missing: {manifest_path}")
    try:
        payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise VideoEngineError(f"Cannot rebuild project because manifest is invalid JSON: {manifest_path}") from exc
    loaded = payload.get("overlays") or []
    return list(loaded) if isinstance(loaded, list) else []


def _reset_to_source_copy(state: ProjectState) -> None:
    source = state.source_files[0]
    fresh = Path(state.working_dir) / f"{uuid.uuid4().hex}{Path(source).suffix}"
    shutil.copy2(source, fresh)
    state.working_file = str(fresh)
    state.metadata = probe_video(str(fresh))
    state.save()


def _restore_from_retained_timeline_result(state: ProjectState) -> bool:
    if not state.timeline:
        return False
    last_result = str(state.timeline[-1].get("result_file") or "").strip()
    if not last_result:
        return False
    result_path = Path(last_result)
    if not result_path.is_file():
        return False
    try:
        state.working_file = str(result_path)
        state.metadata = probe_video(str(result_path))
        state.save()
        return True
    except VideoEngineError:
        return False


def refresh_generated_overlay_ops(
    state: ProjectState,
    *,
    remove_ops: set[str],
) -> dict[str, int]:
    original_timeline = list(state.timeline)
    retained_timeline = [op for op in original_timeline if str(op.get("op") or "").strip() not in remove_ops]
    removed_count = len(original_timeline) - len(retained_timeline)
    if removed_count <= 0:
        return {}

    removed_by_op: dict[str, int] = {}
    for op in original_timeline:
        op_name = str(op.get("op") or "").strip()
        if op_name in remove_ops:
            removed_by_op[op_name] = removed_by_op.get(op_name, 0) + 1

    state.timeline = retained_timeline
    state.redo_stack.clear()
    state.updated_at = utc_now_iso()
    if isinstance(state.artifacts, dict):
        if "add_auto_visuals" in remove_ops:
            state.artifacts.pop("latest_auto_visuals", None)
        if "add_auto_broll" in remove_ops:
            state.artifacts.pop("latest_auto_broll", None)

    first_removed_index = next(
        (index for index, op in enumerate(original_timeline) if str(op.get("op") or "").strip() in remove_ops),
        None,
    )
    trailing_only = first_removed_index is not None and all(
        str(op.get("op") or "").strip() in remove_ops for op in original_timeline[first_removed_index:]
    )
    if trailing_only:
        if _restore_from_retained_timeline_result(state):
            return removed_by_op
        if state.source_files and Path(state.source_files[0]).is_file():
            _reset_to_source_copy(state)
            return removed_by_op

    rebuild_timeline(state)
    return removed_by_op


def rebuild_timeline(
    state: ProjectState,
    *,
    timeline_override: list[dict] | None = None,
) -> None:
    source = state.source_files[0]
    if not os.path.isfile(source):
        raise VideoEngineError(
            f"Source file no longer exists at {source}. Cannot rebuild timeline."
        )
    timeline = list(state.timeline if timeline_override is None else timeline_override)
    if not timeline:
        fresh = Path(state.working_dir) / f"{uuid.uuid4().hex}{Path(source).suffix}"
        shutil.copy2(source, fresh)
        state.working_file = str(fresh)
        state.metadata = probe_video(str(fresh))
        state.save()
        return
    current_path = source
    for op in timeline:
        params = op.get("params", {})
        name = op["op"]
        if name == "trim_clip":
            current_path = trim(current_path, state.working_dir, params["start"], params.get("end"))
        elif name == "merge_clips":
            paths = []
            for item in params.get("file_paths", []):
                paths.append(current_path if item == "__CURRENT__" else item)
            current_path = merge(paths, state.working_dir)
        elif name == "adjust_speed":
            # Stored values are already parsed seconds, so positional mapping is intentional.
            current_path = adjust_speed(
                current_path,
                state.working_dir,
                params["factor"],
                params.get("start"),
                params.get("end"),
            )
        elif name == "add_transition":
            transition_type = params["type"]
            position = params["position"]
            duration = params["duration"]
            if transition_type == "fade_in":
                current_path = fade_in(current_path, state.working_dir, duration)
            elif transition_type == "fade_out":
                current_path = fade_out(current_path, state.working_dir, duration)
            else:
                temp = fade_out(current_path, state.working_dir, duration)
                current_path = fade_in(temp, state.working_dir, duration) if position == "between" else temp
        elif name == "add_text_overlay":
            current_path = add_text(
                current_path,
                state.working_dir,
                text=params["text"],
                position=params["position"],
                font_size=params["font_size"],
                color=params["color"],
                start_sec=params["start"],
                end_sec=params["end"],
                bg_opacity=params["background_opacity"],
            )
        elif name == "replace_audio":
            audio_path = params["audio_path"]
            if not os.path.isfile(audio_path):
                raise VideoEngineError(f"Cannot rebuild project because audio file is missing: {audio_path}")
            current_path = replace_audio(
                current_path,
                audio_path,
                state.working_dir,
                params["mix_with_original"],
                params["mix_ratio"],
            )
        elif name == "mute_segment":
            current_path = mute_segment(current_path, state.working_dir, params["start"], params["end"])
        elif name == "trim_silence":
            current_path = trim_silence(
                current_path,
                state.working_dir,
                params.get("min_silence_duration", 0.5),
                params.get("silence_threshold_db", -35.0),
                max(float(params.get("speech_padding_ms", 120.0)), 0.0) / 1000.0,
                max(float(params.get("merge_gap_ms", 180.0)), 0.0) / 1000.0,
                max(float(params.get("min_keep_duration_ms", 280.0)), 0.0) / 1000.0,
                bool(params.get("trim_edges", False)),
            )
        elif name == "burn_subtitles":
            current_path = burn_subtitles(
                current_path,
                state.working_dir,
                srt_path=params["srt_path"],
                font_size=params.get("font_size", 24),
                font_color=params.get("font_color", "white"),
                outline_color=params.get("outline_color", "black"),
                position=params.get("position", "bottom"),
            )
        elif name == "auto_color_grade":
            filter_graph = str(params.get("filter_graph") or "").strip()
            if not filter_graph:
                raise VideoEngineError("Cannot rebuild project because the stored color grade filter is missing.")
            current_path = apply_color_grade(
                current_path,
                state.working_dir,
                filter_graph,
                render_mode=str(params.get("render_mode") or "vf"),
                output_label=str(params.get("output_label") or "[vout]"),
            )
        elif name == "summarize_clip":
            segments = [(segment["start"], segment["end"]) for segment in params.get("segments", [])]
            current_path = extract_segments(current_path, state.working_dir, segments)
        elif name in {"add_auto_broll", "add_auto_visuals"}:
            overlays = _load_visual_overlays(params)
            if not overlays:
                raise VideoEngineError(
                    f"Cannot rebuild project because stored overlays are missing for {name}."
                )
            current_path = apply_visual_overlays(current_path, state.working_dir, overlays)
    state.working_file = current_path
    state.metadata = probe_video(current_path)
    state.save()


def execute_undo(params: dict, state: ProjectState) -> dict:
    undone = state.undo()
    if undone is None:
        return {
            "success": True,
            "message": "Nothing to undo.",
            "suggestion": None,
            "updated_state": state,
            "tool_name": "undo",
        }
    try:
        rebuild_timeline(state)
        return {
            "success": True,
            "message": f"Undid {undone['op']}.",
            "suggestion": None,
            "updated_state": state,
            "tool_name": "undo",
        }
    except VideoEngineError as exc:
        return {
            "success": False,
            "message": str(exc),
            "suggestion": None,
            "updated_state": state,
            "tool_name": "undo",
        }


def execute_redo(params: dict, state: ProjectState) -> dict:
    redone = state.redo()
    if redone is None:
        return {
            "success": True,
            "message": "Nothing to redo.",
            "suggestion": None,
            "updated_state": state,
            "tool_name": "redo",
        }
    try:
        rebuild_timeline(state)
        return {
            "success": True,
            "message": f"Redid {redone['op']}.",
            "suggestion": None,
            "updated_state": state,
            "tool_name": "redo",
        }
    except VideoEngineError as exc:
        return {
            "success": False,
            "message": str(exc),
            "suggestion": None,
            "updated_state": state,
            "tool_name": "redo",
        }
