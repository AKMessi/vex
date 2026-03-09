from __future__ import annotations

import os

from engine import (
    VideoEngineError,
    add_text,
    adjust_speed,
    fade_in,
    fade_out,
    merge,
    mute_segment,
    probe_video,
    replace_audio,
    trim,
)
from state import ProjectState


def _reapply_operations(state: ProjectState) -> None:
    source = state.source_files[0]
    if not os.path.isfile(source):
        raise VideoEngineError(
            f"Source file no longer exists at {source}. Cannot rebuild timeline."
        )
    current_path = source
    for op in state.timeline:
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
        _reapply_operations(state)
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
        _reapply_operations(state)
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
