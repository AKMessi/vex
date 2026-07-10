from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from engine import (
    VideoEngineError,
    extract_audio,
    mute_segment,
    parse_timestamp,
    probe_video,
    replace_audio,
)
from tools.path_security import (
    UnsafeInputPathError,
    UnsafeOutputPathError,
    resolve_existing_project_file,
    resolve_output_path,
)
from state import ProjectState

AUDIO_INPUT_SUFFIXES = {".aac", ".aiff", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".wma"}


def execute_extract(params: dict, state: ProjectState) -> dict:
    fmt = params.get("format", "mp3")
    try:
        if fmt not in {"mp3", "wav", "aac"}:
            raise ValueError(f"Unsupported audio format: {fmt}")
        requested_output = params.get("output_path")
        output_path = None
        if requested_output:
            suffix = ".m4a" if fmt == "aac" else f".{fmt}"
            output_path = resolve_output_path(
                str(requested_output),
                default_root=state.output_dir,
                allowed_roots=[state.output_dir, Path(state.working_dir) / "exports"],
                allowed_suffixes={suffix},
            )
        temp_output = extract_audio(state.working_file, state.working_dir, fmt)
        saved_path = str(output_path) if output_path is not None else temp_output
        if saved_path != temp_output:
            os.replace(temp_output, saved_path)
        return {
            "success": True,
            "message": f"Extracted audio to {saved_path}.",
            "suggestion": None,
            "updated_state": state,
            "tool_name": "extract_audio",
        }
    except (TypeError, ValueError, UnsafeOutputPathError, VideoEngineError, OSError) as exc:
        return {
            "success": False,
            "message": str(exc),
            "suggestion": None,
            "updated_state": state,
            "tool_name": "extract_audio",
        }


def execute_replace(params: dict, state: ProjectState) -> dict:
    try:
        audio_path = resolve_existing_project_file(
            str(params["audio_path"]),
            state,
            allowed_suffixes=AUDIO_INPUT_SUFFIXES,
        )
    except (KeyError, UnsafeInputPathError) as exc:
        return {
            "success": False,
            "message": str(exc),
            "suggestion": None,
            "updated_state": state,
            "tool_name": "replace_audio",
        }
    try:
        mix = params.get("mix_with_original", False)
        if not isinstance(mix, bool):
            raise TypeError("mix_with_original must be a boolean.")
        mix_ratio = float(params.get("mix_ratio", 0.5))
        if not 0.0 <= mix_ratio <= 1.0:
            raise ValueError("mix_ratio must be between 0.0 and 1.0.")
    except (TypeError, ValueError) as exc:
        return {
            "success": False,
            "message": str(exc),
            "suggestion": None,
            "updated_state": state,
            "tool_name": "replace_audio",
        }
    audio_path_text = os.path.abspath(str(audio_path))
    snapshot = state.capture_snapshot()
    try:
        output_path = replace_audio(
            state.working_file,
            audio_path_text,
            state.working_dir,
            mix=mix,
            mix_ratio=mix_ratio,
        )
        state.working_file = output_path
        state.metadata = probe_video(output_path)
        description = f"Replaced audio using {os.path.basename(audio_path_text)}"
        if mix:
            description = f"Mixed audio using {os.path.basename(audio_path_text)} at ratio {mix_ratio:.2f}"
        op = {
            "op": "replace_audio",
            "params": {
                "audio_path": audio_path_text,
                "mix_with_original": mix,
                "mix_ratio": mix_ratio,
            },
            "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "result_file": output_path,
            "description": description,
        }
        state.apply_operation(op)
        return {
            "success": True,
            "message": description + ".",
            "suggestion": None,
            "updated_state": state,
            "tool_name": "replace_audio",
        }
    except (KeyError, TypeError, ValueError, VideoEngineError, OSError) as exc:
        state.restore_snapshot(snapshot)
        return {
            "success": False,
            "message": str(exc),
            "suggestion": None,
            "updated_state": state,
            "tool_name": "replace_audio",
        }
    except BaseException:
        state.restore_snapshot(snapshot)
        raise


def execute_mute(params: dict, state: ProjectState) -> dict:
    try:
        start_sec = parse_timestamp(params["start"])
        end_sec = parse_timestamp(params["end"])
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "success": False,
            "message": str(exc),
            "suggestion": None,
            "updated_state": state,
            "tool_name": "mute_segment",
        }
    snapshot = state.capture_snapshot()
    try:
        output_path = mute_segment(state.working_file, state.working_dir, start_sec, end_sec)
        state.working_file = output_path
        state.metadata = probe_video(output_path)
        description = f"Muted segment from {params['start']} to {params['end']}"
        op = {
            "op": "mute_segment",
            "params": {
                "start": start_sec,
                "end": end_sec,
                "start_label": params["start"],
                "end_label": params["end"],
            },
            "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "result_file": output_path,
            "description": description,
        }
        state.apply_operation(op)
        return {
            "success": True,
            "message": description + ".",
            "suggestion": None,
            "updated_state": state,
            "tool_name": "mute_segment",
        }
    except (KeyError, TypeError, ValueError, VideoEngineError, OSError) as exc:
        state.restore_snapshot(snapshot)
        return {
            "success": False,
            "message": str(exc),
            "suggestion": None,
            "updated_state": state,
            "tool_name": "mute_segment",
        }
    except BaseException:
        state.restore_snapshot(snapshot)
        raise
