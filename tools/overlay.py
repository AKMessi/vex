from __future__ import annotations

from datetime import datetime, timezone

from engine import VideoEngineError, add_text, parse_timestamp, probe_video
from state import ProjectState

VALID_POSITIONS = {
    "top",
    "center",
    "bottom",
    "top_left",
    "top_right",
    "bottom_left",
    "bottom_right",
}


def execute(params: dict, state: ProjectState) -> dict:
    try:
        if params["position"] not in VALID_POSITIONS:
            return {
                "success": False,
                "message": f"Invalid position: {params['position']}",
                "suggestion": None,
                "updated_state": state,
                "tool_name": "add_text_overlay",
            }
        start_sec = parse_timestamp(params["start"])
        end_sec = parse_timestamp(params["end"])
        output_path = add_text(
            state.working_file,
            state.working_dir,
            text=params["text"],
            position=params["position"],
            font_size=int(params.get("font_size", 48)),
            color=params.get("color", "white"),
            start_sec=start_sec,
            end_sec=end_sec,
            bg_opacity=float(params.get("background_opacity", 0.0)),
        )
        state.working_file = output_path
        state.metadata = probe_video(output_path)
        description = (
            f'Text overlay added: "{params["text"]}", {params["position"]}, '
            f'{params["start"]}-{params["end"]}'
        )
        op = {
            "op": "add_text_overlay",
            "params": {
                "text": params["text"],
                "position": params["position"],
                "start": start_sec,
                "end": end_sec,
                "start_label": params["start"],
                "end_label": params["end"],
                "font_size": int(params.get("font_size", 48)),
                "color": params.get("color", "white"),
                "background_opacity": float(params.get("background_opacity", 0.0)),
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
            "tool_name": "add_text_overlay",
        }
    except (ValueError, VideoEngineError, OSError, KeyError) as exc:
        return {
            "success": False,
            "message": str(exc),
            "suggestion": None,
            "updated_state": state,
            "tool_name": "add_text_overlay",
        }
