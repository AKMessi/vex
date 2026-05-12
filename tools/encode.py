from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from encode_planner import (
    EncodePlanningError,
    build_encode_plan,
    pending_plan_is_current,
    run_encode_plan,
    source_fingerprint,
)
from engine import VideoEngineError, check_disk_space
from state import ProjectState


def execute_plan(params: dict[str, Any], state: ProjectState) -> dict[str, Any]:
    try:
        plan = build_encode_plan(
            state.working_file,
            state.output_dir,
            state.project_name,
            params,
            metadata=state.metadata or None,
        )
    except (EncodePlanningError, VideoEngineError, OSError) as exc:
        return {
            "success": False,
            "message": str(exc),
            "suggestion": None,
            "updated_state": state,
            "tool_name": "plan_encode",
        }

    plan_payload = plan.to_dict()
    plan_payload["source_fingerprint"] = source_fingerprint(
        state.working_file,
        timeline_count=len(state.timeline),
    )
    state.artifacts["pending_encode"] = plan_payload
    state.save()
    warnings = plan_payload.get("warnings") or []
    warning_text = ""
    if warnings:
        warning_text = "\nWarnings:\n" + "\n".join(f"- {warning}" for warning in warnings)
    estimated = _format_bytes(plan_payload.get("estimated_size_bytes"))
    estimated_text = f"\nEstimated target size: {estimated}" if estimated else ""
    message = (
        f"Encode plan ready: {plan.summary}\n"
        f"Output: {plan.output_path}{estimated_text}\n"
        f"Command:\n{plan.display_command}"
        f"{warning_text}"
    )
    return {
        "success": True,
        "message": message,
        "suggestion": "[SUGGESTION]: Reply 'yes' to run this encode, or describe a change to re-plan it.",
        "updated_state": state,
        "tool_name": "plan_encode",
        "plan_id": plan.plan_id,
        "command": plan.display_command,
        "output_path": plan.output_path,
    }


def execute_run_pending(params: dict[str, Any], state: ProjectState) -> dict[str, Any]:
    pending = state.artifacts.get("pending_encode")
    if not isinstance(pending, dict):
        return {
            "success": False,
            "message": "No pending encode plan. Ask me to encode, convert, or compress the video first.",
            "suggestion": None,
            "updated_state": state,
            "tool_name": "run_pending_encode",
        }
    requested_plan_id = str(params.get("plan_id") or "").strip()
    actual_plan_id = str(pending.get("plan_id") or "").strip()
    if requested_plan_id and requested_plan_id != actual_plan_id:
        return {
            "success": False,
            "message": "That encode confirmation does not match the latest pending plan. Ask me to plan it again.",
            "suggestion": None,
            "updated_state": state,
            "tool_name": "run_pending_encode",
        }
    if not pending_plan_is_current(pending, state.working_file, timeline_count=len(state.timeline)):
        state.artifacts.pop("pending_encode", None)
        state.save()
        return {
            "success": False,
            "message": "The pending encode plan is stale because the working file changed. Ask me to plan the encode again.",
            "suggestion": None,
            "updated_state": state,
            "tool_name": "run_pending_encode",
        }

    output_path = str(pending.get("output_path") or "")
    if not output_path:
        return {
            "success": False,
            "message": "The pending encode plan is missing an output path. Ask me to plan the encode again.",
            "suggestion": None,
            "updated_state": state,
            "tool_name": "run_pending_encode",
        }
    required = int(pending.get("estimated_size_bytes") or _fallback_required_space(state.working_file))
    if not check_disk_space(output_path, required):
        return {
            "success": False,
            "message": f"Not enough disk space for encode. Estimated requirement: {_format_bytes(required)}.",
            "suggestion": None,
            "updated_state": state,
            "tool_name": "run_pending_encode",
        }

    try:
        result = run_encode_plan(pending)
    except VideoEngineError as exc:
        return {
            "success": False,
            "message": str(exc),
            "suggestion": None,
            "updated_state": state,
            "tool_name": "run_pending_encode",
        }

    input_size = int((pending.get("source_metadata") or {}).get("size_bytes") or 0)
    output_size = int(result.get("output_size_bytes") or 0)
    state.artifacts["latest_encode"] = {
        "plan": pending,
        "output_path": result["output_path"],
        "output_metadata": result["output_metadata"],
        "output_size_bytes": output_size,
        "validation": result["validation"],
        "completed_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    state.artifacts.pop("pending_encode", None)
    state.save()
    validation = result.get("validation") or {}
    validation_warnings = validation.get("warnings") or []
    validation_text = ""
    if validation_warnings:
        validation_text = "\nValidation warnings:\n" + "\n".join(f"- {warning}" for warning in validation_warnings)
    size_text = _size_delta_text(input_size, output_size)
    return {
        "success": True,
        "message": f"Encoded video to {result['output_path']}. {size_text}{validation_text}".strip(),
        "suggestion": None,
        "updated_state": state,
        "tool_name": "run_pending_encode",
        "output_path": result["output_path"],
    }


def _fallback_required_space(input_path: str) -> int:
    try:
        return max(Path(input_path).stat().st_size * 2, 256 * 1024 * 1024)
    except OSError:
        return 512 * 1024 * 1024


def _format_bytes(value: Any) -> str:
    try:
        size = float(value)
    except (TypeError, ValueError):
        return ""
    if size <= 0:
        return ""
    units = ["B", "KB", "MB", "GB", "TB"]
    index = 0
    while size >= 1024 and index < len(units) - 1:
        size /= 1024
        index += 1
    return f"{size:.1f} {units[index]}"


def _size_delta_text(input_size: int, output_size: int) -> str:
    if input_size <= 0 or output_size <= 0:
        return f"Output size: {_format_bytes(output_size)}."
    change = (1.0 - (output_size / input_size)) * 100.0
    if change >= 0:
        return f"Output size: {_format_bytes(output_size)} ({change:.1f}% smaller)."
    return f"Output size: {_format_bytes(output_size)} ({abs(change):.1f}% larger)."
