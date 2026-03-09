from __future__ import annotations

from datetime import datetime, timezone

from engine import VideoEngineError, fade_in, fade_out, probe_video
from state import ProjectState


def execute(params: dict, state: ProjectState) -> dict:
    transition_type = params["type"]
    duration = float(params["duration"])
    position = params["position"]
    try:
        if transition_type == "fade_in":
            output_path = fade_in(state.working_file, state.working_dir, duration)
        elif transition_type == "fade_out":
            output_path = fade_out(state.working_file, state.working_dir, duration)
        elif transition_type == "crossfade":
            if position == "between":
                temp = fade_out(state.working_file, state.working_dir, duration)
                output_path = fade_in(temp, state.working_dir, duration)
            elif position == "start":
                output_path = fade_in(state.working_file, state.working_dir, duration)
            else:
                output_path = fade_out(state.working_file, state.working_dir, duration)
        else:
            return {
                "success": False,
                "message": f"Unsupported transition type: {transition_type}",
                "suggestion": None,
                "updated_state": state,
                "tool_name": "add_transition",
            }
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
    except VideoEngineError as exc:
        return {
            "success": False,
            "message": str(exc),
            "suggestion": None,
            "updated_state": state,
            "tool_name": "add_transition",
        }
