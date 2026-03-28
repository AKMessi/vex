from __future__ import annotations

from datetime import datetime, timezone

from engine import VideoEngineError, probe_video, trim_silence
from state import ProjectState

AGGRESSIVENESS_PRESETS = {
    "low": {"min_silence_duration": 0.7, "silence_threshold_db": -38.0},
    "medium": {"min_silence_duration": 0.5, "silence_threshold_db": -35.0},
    "high": {"min_silence_duration": 0.35, "silence_threshold_db": -32.0},
}


def execute(params: dict, state: ProjectState) -> dict:
    aggressiveness = str(params.get("aggressiveness") or "medium").strip().lower()
    preset = AGGRESSIVENESS_PRESETS.get(aggressiveness, AGGRESSIVENESS_PRESETS["medium"])
    min_silence_duration = float(params.get("min_silence_duration", preset["min_silence_duration"]))
    silence_threshold_db = float(params.get("silence_threshold_db", preset["silence_threshold_db"]))
    speech_padding_ms = float(params.get("speech_padding_ms", 120.0))
    merge_gap_ms = float(params.get("merge_gap_ms", 180.0))
    min_keep_duration_ms = float(params.get("min_keep_duration_ms", 280.0))
    trim_edges = bool(params.get("trim_edges", False))
    try:
        output_path = trim_silence(
            state.working_file,
            state.working_dir,
            min_silence_duration=min_silence_duration,
            silence_threshold_db=silence_threshold_db,
            speech_padding_sec=max(speech_padding_ms, 0.0) / 1000.0,
            merge_gap_sec=max(merge_gap_ms, 0.0) / 1000.0,
            min_keep_duration_sec=max(min_keep_duration_ms, 0.0) / 1000.0,
            trim_edges=trim_edges,
        )
        if output_path == state.working_file:
            return {
                "success": True,
                "message": "No meaningful silent gaps were removed.",
                "suggestion": None,
                "updated_state": state,
                "tool_name": "trim_silence",
            }
        state.working_file = output_path
        state.metadata = probe_video(output_path)
        description = (
            f"Trimmed silent gaps longer than {min_silence_duration:.2f}s below {silence_threshold_db:.1f} dB"
        )
        if trim_edges:
            description += ", including edge pauses"
        op = {
            "op": "trim_silence",
            "params": {
                "min_silence_duration": min_silence_duration,
                "silence_threshold_db": silence_threshold_db,
                "aggressiveness": aggressiveness,
                "speech_padding_ms": speech_padding_ms,
                "merge_gap_ms": merge_gap_ms,
                "min_keep_duration_ms": min_keep_duration_ms,
                "trim_edges": trim_edges,
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
            "tool_name": "trim_silence",
        }
    except (ValueError, VideoEngineError) as exc:
        return {
            "success": False,
            "message": str(exc),
            "suggestion": None,
            "updated_state": state,
            "tool_name": "trim_silence",
        }
