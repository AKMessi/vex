from __future__ import annotations

import math
from datetime import datetime, timezone

from engine import VideoEngineError, apply_transition, probe_video
from state import ProjectState


def execute(params: dict, state: ProjectState) -> dict:
    try:
        transition_type = params["type"]
        duration = float(params["duration"])
        position = params["position"]
        if transition_type not in {"fade_in", "fade_out", "crossfade"}:
            raise ValueError(f"Unsupported transition type: {transition_type}")
        if position not in {"start", "end", "between"}:
            raise ValueError(f"Unsupported transition position: {position}")
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Transition duration must be a positive finite number.")
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "success": False,
            "message": str(exc),
            "suggestion": None,
            "updated_state": state,
            "tool_name": "add_transition",
        }
    snapshot = state.capture_snapshot()
    try:
        output_path = apply_transition(state.working_file, state.working_dir, transition_type, position, duration)
        state.working_file = output_path
        state.metadata = probe_video(output_path)
        if transition_type == "crossfade" and position == "between":
            description = f"Applied fade-through-black transition at {position} for {duration}s"
        else:
            description = f"Applied {transition_type} transition at {position} for {duration}s"
        op = {
            "op": "add_transition",
            "params": {"type": transition_type, "duration": duration, "position": position},
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
            "tool_name": "add_transition",
        }
    except (KeyError, TypeError, ValueError, VideoEngineError, OSError) as exc:
        state.restore_snapshot(snapshot)
        return {
            "success": False,
            "message": str(exc),
            "suggestion": None,
            "updated_state": state,
            "tool_name": "add_transition",
        }
    except BaseException:
        state.restore_snapshot(snapshot)
        raise
