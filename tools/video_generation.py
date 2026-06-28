from __future__ import annotations

from typing import Any

from state import ProjectState, utc_now_iso
from video_generation import generate_video


def execute(params: dict[str, Any], state: ProjectState | None = None) -> dict[str, Any]:
    try:
        result = generate_video(dict(params or {}))
    except Exception as exc:  # noqa: BLE001
        return {
            "success": False,
            "message": f"generate_video failed: {exc}",
            "suggestion": None,
            "updated_state": state,
            "tool_name": "generate_video",
        }

    if state is not None:
        state.artifacts["latest_generated_video"] = result.to_dict()
        state.artifacts.setdefault("generated_videos", []).append(result.to_dict())
        state.artifacts["generated_videos"] = state.artifacts["generated_videos"][-20:]
        state.session_log.append(
            {
                "role": "tool",
                "name": "generate_video",
                "content": result.manifest_path,
                "timestamp": utc_now_iso(),
            }
        )
        state.save()

    message = (
        f"Generated HyperFrames video project at {result.project_dir}. "
        f"Manifest: {result.manifest_path}."
    )
    if result.rendered:
        message += f" Render: {result.output_path}."
    if result.audio_path:
        message += f" Audio: {result.audio_path}."
    if result.warnings:
        message += " Warnings: " + "; ".join(result.warnings[:3])
    if not result.qa_passed:
        issues = "; ".join(result.qa_issues[:4]) or "generated video QA failed"
        message = (
            f"Generated video project was rejected by QA at {result.project_dir}. "
            f"Manifest: {result.manifest_path}. Issues: {issues}."
        )
    return {
        "success": bool(result.qa_passed),
        "message": message,
        "suggestion": None if result.qa_passed else "Review the manifest QA evidence and rerun with a narrower prompt or script.",
        "updated_state": state,
        "tool_name": "generate_video",
        "result": result.to_dict(),
        "manifest_path": result.manifest_path,
        "output_path": result.output_path,
        "project_dir": result.project_dir,
    }
