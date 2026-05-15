from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from color_grading import ColorGradePlanningError, SUPPORTED_COLOR_GRADE_LOOKS, normalize_color_grade_look
from engine import VideoEngineError, auto_color_grade, probe_video
from state import ProjectState


def execute(params: dict[str, Any], state: ProjectState) -> dict[str, Any]:
    try:
        look = normalize_color_grade_look(str(params.get("look") or params.get("style") or "auto"))
        intensity = float(params.get("intensity", 1.0))
        if intensity < 0.0 or intensity > 1.5:
            raise ColorGradePlanningError("Color grade intensity must be between 0.0 and 1.5.")
        sample_count = max(1, min(int(params.get("sample_count", 9) or 9), 15))
        mode = str(params.get("mode") or params.get("grading_mode") or "auto")
        max_shots = max(1, min(int(params.get("max_shots", 18) or 18), 64))
        candidate_count = max(2, min(int(params.get("candidate_count", 4) or 4), 5))
        output_path, plan = auto_color_grade(
            state.working_file,
            state.working_dir,
            look=look,
            intensity=intensity,
            sample_count=sample_count,
            mode=mode,
            max_shots=max_shots,
            candidate_count=candidate_count,
        )
        state.working_file = output_path
        state.metadata = probe_video(output_path)
        resolved_look = str(plan.get("resolved_look") or look)
        description = f"Applied {resolved_look} auto color grade"
        op = {
            "op": "auto_color_grade",
            "params": {
                "look": look,
                "resolved_look": resolved_look,
                "intensity": intensity,
                "sample_count": sample_count,
                "mode": mode,
                "max_shots": max_shots,
                "candidate_count": candidate_count,
                "filter_graph": plan["filter_graph"],
                "render_mode": plan.get("render_mode", "vf"),
                "output_label": plan.get("output_label", ""),
                "adjustments": plan.get("adjustments", {}),
                "analysis": plan.get("analysis", {}),
                "manifest": plan.get("manifest"),
                "validation": plan.get("validation", {}),
                "warnings": plan.get("warnings", []),
            },
            "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "result_file": output_path,
            "description": description,
        }
        state.artifacts["latest_auto_color_grade"] = {
            "look": look,
            "resolved_look": resolved_look,
            "intensity": intensity,
            "sample_count": sample_count,
            "mode": mode,
            "max_shots": max_shots,
            "candidate_count": candidate_count,
            "output_path": output_path,
            "filter_graph": plan["filter_graph"],
            "render_mode": plan.get("render_mode", "vf"),
            "output_label": plan.get("output_label", ""),
            "adjustments": plan.get("adjustments", {}),
            "analysis": plan.get("analysis", {}),
            "manifest": plan.get("manifest"),
            "validation": plan.get("validation", {}),
            "warnings": plan.get("warnings", []),
            "completed_at": op["timestamp"],
        }
        state.apply_operation(op)
        return {
            "success": True,
            "message": _format_success_message(description, plan),
            "suggestion": None,
            "updated_state": state,
            "tool_name": "auto_color_grade",
            "output_path": output_path,
            "plan": plan,
        }
    except (ColorGradePlanningError, VideoEngineError, OSError, ValueError) as exc:
        return {
            "success": False,
            "message": str(exc),
            "suggestion": None,
            "updated_state": state,
            "tool_name": "auto_color_grade",
        }


def _format_success_message(description: str, plan: dict[str, Any]) -> str:
    adjustments = dict(plan.get("adjustments") or {})
    analysis = dict(plan.get("analysis") or {})
    parts = [
        f"brightness {float(adjustments.get('brightness', 0.0)):+.3f}",
        f"contrast {float(adjustments.get('contrast', 1.0)):.2f}x",
        f"saturation {float(adjustments.get('saturation', 1.0)):.2f}x",
        (
            "white balance "
            f"R {float(adjustments.get('red_gain', 1.0)):.2f} / "
            f"G {float(adjustments.get('green_gain', 1.0)):.2f} / "
            f"B {float(adjustments.get('blue_gain', 1.0)):.2f}"
        ),
    ]
    sampled = int(analysis.get("sample_count") or 0)
    message = f"{description}. Adjustments: {', '.join(parts)}."
    if sampled:
        message += f" Sampled {sampled} frame{'s' if sampled != 1 else ''}."
    manifest = dict(plan.get("manifest") or {})
    shot_count = int(manifest.get("shot_count") or adjustments.get("shot_count") or 0)
    if shot_count:
        candidate_count = int(manifest.get("candidate_count") or adjustments.get("candidate_count") or 0)
        message += f" Shot-aware plan: {shot_count} shot{'s' if shot_count != 1 else ''}"
        if candidate_count:
            message += f", {candidate_count} candidates each"
        message += "."
    selected_score = adjustments.get("average_selected_score")
    if selected_score is not None:
        message += f" Average candidate score: {float(selected_score):.2f}."
    overall_need = adjustments.get("overall_need")
    correction_strength = adjustments.get("correction_strength")
    if overall_need is not None and correction_strength is not None:
        message += (
            f" Correction need: {float(overall_need):.2f};"
            f" strength: {float(correction_strength):.2f}x."
        )
    confidence = analysis.get("white_balance_confidence")
    if confidence is not None:
        message += f" White-balance confidence: {float(confidence):.2f}."
    validation = dict(plan.get("validation") or {})
    if validation.get("score") is not None:
        status = "passed" if validation.get("passed") else "needs review"
        message += f" Output validation {status} ({float(validation.get('score') or 0.0):.2f})."
    warnings = [str(item) for item in plan.get("warnings") or [] if str(item).strip()]
    warnings.extend(str(item) for item in validation.get("warnings") or [] if str(item).strip())
    if warnings:
        message += "\nWarnings:\n" + "\n".join(f"- {warning}" for warning in warnings)
    return message


__all__ = ["SUPPORTED_COLOR_GRADE_LOOKS", "execute"]
