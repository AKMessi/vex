from __future__ import annotations

import os
from pathlib import Path

from engine import VideoEngineError, check_disk_space, estimate_output_size, export
from state import ProjectState, utc_now_iso
from tools.path_security import UnsafeOutputPathError, resolve_output_path


SCALE_MODES = {"fit", "fill", "stretch"}


def _safe_stem(project_name: str) -> str:
    return "".join(
        ch
        for ch in project_name.replace(" ", "_")
        if ch.isalnum() or ch in {"_", "-"}
    ) or "video"


def _default_output_path(state: ProjectState, resolution: str) -> str:
    base = Path(state.output_dir) / f"{_safe_stem(state.project_name)}_scaled_{resolution}.mp4"
    if not base.exists():
        return str(base)
    for index in range(2, 1000):
        candidate = base.with_name(f"{base.stem}_{index:03d}{base.suffix}")
        if not candidate.exists():
            return str(candidate)
    raise UnsafeOutputPathError("Could not find an unused upscale output filename.")


def execute(params: dict, state: ProjectState) -> dict:
    resolution = str(params.get("resolution") or "").strip().lower()
    if not resolution:
        return {
            "success": False,
            "message": "Missing resolution. Use WIDTHxHEIGHT, for example 1920x1080.",
            "suggestion": None,
            "updated_state": state,
            "tool_name": "upscale_video",
        }
    scale_mode = str(params.get("scale_mode") or "fit").strip().lower()
    if scale_mode not in SCALE_MODES:
        return {
            "success": False,
            "message": "scale_mode must be one of: fit, fill, stretch.",
            "suggestion": None,
            "updated_state": state,
            "tool_name": "upscale_video",
        }
    preset = {
        "format": "mp4",
        "resolution": resolution,
        "scale_mode": scale_mode,
        "video_codec": str(params.get("video_codec") or "libx264"),
        "audio_codec": str(params.get("audio_codec") or "aac"),
        "video_bitrate": params.get("video_bitrate") or "8000k",
        "audio_bitrate": params.get("audio_bitrate") or "192k",
        "x264_preset": params.get("x264_preset") or "medium",
    }
    output_path = params.get("output_path")
    try:
        if output_path:
            output_path = str(
                resolve_output_path(
                    str(output_path),
                    default_root=state.output_dir,
                    allowed_roots=[state.output_dir, Path(state.working_dir) / "exports"],
                    allowed_suffixes={".mp4"},
                )
            )
        else:
            output_path = _default_output_path(state, resolution)
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    except UnsafeOutputPathError as exc:
        return {
            "success": False,
            "message": str(exc),
            "suggestion": None,
            "updated_state": state,
            "tool_name": "upscale_video",
        }
    try:
        estimate = estimate_output_size(state.working_file, preset)
    except (OSError, ValueError, VideoEngineError) as exc:
        return {
            "success": False,
            "message": f"Could not estimate scaled export size: {exc}",
            "suggestion": None,
            "updated_state": state,
            "tool_name": "upscale_video",
        }
    if not check_disk_space(output_path, estimate):
        return {
            "success": False,
            "message": f"Not enough disk space for scaled export. Estimated size: {estimate / (1024 * 1024):.1f} MB.",
            "suggestion": None,
            "updated_state": state,
            "tool_name": "upscale_video",
        }
    try:
        saved = export(state.working_file, output_path, preset)
    except VideoEngineError as exc:
        return {
            "success": False,
            "message": str(exc),
            "suggestion": None,
            "updated_state": state,
            "tool_name": "upscale_video",
        }
    state.artifacts["latest_upscale"] = {
        "created_at": utc_now_iso(),
        "output_path": saved,
        "resolution": resolution,
        "scale_mode": scale_mode,
        "method": "ffmpeg_lanczos_scale",
    }
    history = list(state.artifacts.get("upscale_history") or [])
    history.append(state.artifacts["latest_upscale"])
    state.artifacts["upscale_history"] = history[-10:]
    state.save()
    return {
        "success": True,
        "message": (
            f"Scaled video to {resolution} with FFmpeg Lanczos ({scale_mode}) and saved {saved}. "
            "This is resize/export scaling, not AI super-resolution."
        ),
        "suggestion": None,
        "updated_state": state,
        "tool_name": "upscale_video",
    }
