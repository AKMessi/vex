from __future__ import annotations

from datetime import datetime, timezone

from engine import VideoEngineError, parse_timestamp, probe_video, trim
from state import ProjectState
from tools.promotion import promote_working_file


def execute(params: dict, state: ProjectState) -> dict:
    try:
        start_sec = parse_timestamp(params["start"])
        end_sec = parse_timestamp(params["end"]) if params.get("end") else None
        output_path = trim(state.working_file, state.working_dir, start_sec, end_sec)
        output_metadata = probe_video(output_path)
        description = (
            f"Trimmed from {params['start']} to {params.get('end', 'end')}"
            if params.get("end")
            else f"Trimmed from {params['start']} to end"
        )
        op = {
            "op": "trim_clip",
            "params": {"start": start_sec, "end": end_sec, "start_label": params["start"], "end_label": params.get("end")},
            "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "result_file": output_path,
            "description": description,
        }
        promotion = promote_working_file(
            state,
            output_path,
            operation=op,
            metadata=output_metadata,
            asset_kind="video",
            asset_role="working_file",
            asset_source="trim_clip",
            asset_metadata={"start_sec": start_sec, "end_sec": end_sec},
        )
        suggestion = None
        if state.metadata.get("duration_sec", 0.0) < 2.0:
            suggestion = "[SUGGESTION]: The resulting clip is under 2 seconds and may be too short for transitions - reply 'yes' to apply or continue."
        return {
            "success": True,
            "message": description + ".",
            "suggestion": suggestion,
            "updated_state": state,
            "tool_name": "trim_clip",
            "asset_id": promotion.asset.asset_id,
            "operation_id": promotion.operation["op_id"],
        }
    except (OSError, ValueError, VideoEngineError) as exc:
        return {
            "success": False,
            "message": str(exc),
            "suggestion": None,
            "updated_state": state,
            "tool_name": "trim_clip",
        }
