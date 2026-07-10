from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import config
from engine import VideoEngineError, apply_visual_overlays, probe_video
from renderers.hyperframes_renderer import _hyperframes_command, _write_command_log
from state import ProjectState, utc_now_iso
from tools.path_security import UnsafeInputPathError, resolve_existing_project_file


VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
GIF_SUFFIXES = {".gif"}
HTML_SUFFIXES = {".html", ".htm"}
SUPPORTED_SUFFIXES = VIDEO_SUFFIXES | IMAGE_SUFFIXES | GIF_SUFFIXES | HTML_SUFFIXES


def _safe_stem(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value or "").strip())
    return cleaned.strip("_")[:96] or "manual_visual"


def _normalize_composition_mode(value: object) -> str:
    normalized = str(value or "replace").strip().lower().replace("-", "_")
    if normalized in {"replace", "fullscreen", "full_screen"}:
        return "replace"
    if normalized in {"overlay", "alpha_overlay"}:
        return "overlay"
    if normalized in {"pip", "picture_in_picture", "pictureinpicture"}:
        return "picture_in_picture"
    raise ValueError("composition_mode must be one of: replace, overlay, picture_in_picture.")


def _parse_time(value: object, *, default: float | None = None) -> float:
    if value is None:
        if default is None:
            raise ValueError("Missing time value.")
        return default
    text = str(value).strip().replace(",", ".")
    if not text:
        if default is None:
            raise ValueError("Missing time value.")
        return default
    if ":" not in text:
        return float(text)
    parts = [float(part) for part in text.split(":")]
    if len(parts) == 2:
        return parts[0] * 60.0 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600.0 + parts[1] * 60.0 + parts[2]
    raise ValueError(f"Invalid timestamp: {value!r}")


def _validate_html_asset(path: Path) -> None:
    html = path.read_text(encoding="utf-8", errors="ignore")
    if re.search(r"\bhttps?://|//[A-Za-z0-9.-]+", html, flags=re.IGNORECASE):
        raise UnsafeInputPathError("HTML visual assets must be local-only and cannot reference remote URLs.")
    if re.search(r"\b(?:child_process|spawn|exec|require\s*\()", html, flags=re.IGNORECASE):
        raise UnsafeInputPathError("HTML visual assets cannot include shell or Node execution hooks.")


def _render_html_asset(
    asset_path: Path,
    *,
    job_dir: Path,
    duration_sec: float,
    width: int,
    height: int,
    fps: float,
) -> tuple[Path, dict[str, Any]]:
    _validate_html_asset(asset_path)
    job_dir.mkdir(parents=True, exist_ok=True)
    index_path = job_dir / "index.html"
    shutil.copyfile(asset_path, index_path)
    output_path = job_dir / "manual_html_visual.mp4"
    render_log_path = job_dir / "hyperframes_render.log"
    command = _hyperframes_command(
        "render",
        "--output",
        str(output_path),
        "--fps",
        str(max(15, int(round(fps or 30.0)))),
        ".",
    )
    try:
        result = subprocess.run(
            command,
            cwd=str(job_dir),
            capture_output=True,
            text=True,
            timeout=max(30, int(duration_sec * 8)),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise VideoEngineError(f"HyperFrames HTML render failed to start: {exc}") from exc
    _write_command_log(render_log_path, command, result)
    if result.returncode != 0 or not output_path.is_file():
        detail = (result.stderr or result.stdout or "").strip()
        raise VideoEngineError(
            f"HyperFrames HTML render failed. Log: {render_log_path}. {detail}"
        )
    return output_path, {
        "renderer": "hyperframes",
        "html_render_mode": "local_hyperframes_cli",
        "renderer_log_path": str(render_log_path),
        "renderer_job_dir": str(job_dir),
        "rendered_asset_path": str(output_path),
    }


def _render_image_or_gif_asset(
    asset_path: Path,
    *,
    job_dir: Path,
    duration_sec: float,
    width: int,
    height: int,
    fps: float,
) -> tuple[Path, dict[str, Any]]:
    job_dir.mkdir(parents=True, exist_ok=True)
    output_path = job_dir / "manual_asset.mp4"
    log_path = job_dir / "ffmpeg_asset_render.log"
    command = [config.FFMPEG_PATH, "-hide_banner", "-nostdin"]
    if asset_path.suffix.lower() in IMAGE_SUFFIXES:
        command.extend(["-loop", "1"])
    else:
        command.extend(["-stream_loop", "-1"])
    command.extend(
        [
            "-i",
            str(asset_path),
            "-t",
            f"{duration_sec:.3f}",
            "-vf",
            (
                f"fps={max(15, int(round(fps or 30.0)))},"
                f"scale={width}:{height}:force_original_aspect_ratio=decrease:flags=lanczos,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1"
            ),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-y",
            str(output_path),
        ]
    )
    timeout = _ffmpeg_asset_timeout_sec(duration_sec)
    try:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise VideoEngineError(f"FFmpeg asset render timed out after {timeout}s.") from exc
    except OSError as exc:
        raise VideoEngineError(f"FFmpeg asset render could not start: {exc}") from exc
    log_path.write_text(
        "\n".join(["$ " + " ".join(command), "", result.stdout or "", result.stderr or ""]),
        encoding="utf-8",
    )
    if result.returncode != 0 or not output_path.is_file():
        detail = (result.stderr or result.stdout or "").strip()
        raise VideoEngineError(f"FFmpeg asset render failed. Log: {log_path}. {detail}")
    return output_path, {
        "renderer": "ffmpeg",
        "renderer_log_path": str(log_path),
        "renderer_job_dir": str(job_dir),
        "rendered_asset_path": str(output_path),
    }


def _ffmpeg_asset_timeout_sec(duration_sec: float) -> int | None:
    try:
        configured = int(getattr(config, "FFMPEG_RENDER_TIMEOUT_SEC", 7200))
    except (TypeError, ValueError):
        configured = 7200
    if configured <= 0:
        return None
    return max(30, configured, int(max(duration_sec, 1.0) * 12))


def _prepare_asset(
    asset_path: Path,
    *,
    bundle_dir: Path,
    duration_sec: float,
    width: int,
    height: int,
    fps: float,
) -> tuple[Path, dict[str, Any]]:
    suffix = asset_path.suffix.lower()
    if suffix in HTML_SUFFIXES:
        return _render_html_asset(
            asset_path,
            job_dir=bundle_dir / "html_render",
            duration_sec=duration_sec,
            width=width,
            height=height,
            fps=fps,
        )
    if suffix in IMAGE_SUFFIXES | GIF_SUFFIXES:
        return _render_image_or_gif_asset(
            asset_path,
            job_dir=bundle_dir / "asset_render",
            duration_sec=duration_sec,
            width=width,
            height=height,
            fps=fps,
        )
    return asset_path, {
        "renderer": "ffmpeg",
        "html_render_mode": None,
        "rendered_asset_path": str(asset_path),
    }


def execute(params: dict, state: ProjectState) -> dict:
    asset_value = str(params.get("asset_path") or params.get("asset") or "").strip()
    if not asset_value:
        return {
            "success": False,
            "message": "Missing asset_path.",
            "suggestion": None,
            "updated_state": state,
            "tool_name": "add_visual_asset",
        }
    try:
        asset_path = resolve_existing_project_file(
            asset_value,
            state,
            allowed_suffixes=SUPPORTED_SUFFIXES,
            max_size_bytes=250 * 1024 * 1024,
        )
        metadata = state.metadata or probe_video(state.working_file)
        clip_duration = float(metadata.get("duration_sec") or 0.0)
        width = int(metadata.get("width") or 0)
        height = int(metadata.get("height") or 0)
        fps = float(metadata.get("fps") or 30.0) or 30.0
        if clip_duration <= 0 or width <= 0 or height <= 0:
            raise ValueError("The working video does not have valid timing/resolution metadata.")
        start_sec = max(0.0, _parse_time(params.get("start")))
        end_sec = _parse_time(params.get("end"), default=start_sec + _parse_time(params.get("duration"), default=4.0))
        end_sec = min(clip_duration, end_sec)
        if end_sec <= start_sec:
            raise ValueError("Manual visual end time must be after start time.")
        composition_mode = _normalize_composition_mode(
            params.get("composition_mode") or params.get("mode")
        )
        timestamp_label = utc_now_iso().replace(":", "-").replace("+00:00", "Z")
        bundle_dir = (
            Path(state.working_dir)
            / "manual_visual_bundles"
            / f"{_safe_stem(asset_path.stem)}_{timestamp_label}"
        )
        bundle_dir.mkdir(parents=True, exist_ok=True)
        prepared_asset, render_info = _prepare_asset(
            asset_path,
            bundle_dir=bundle_dir,
            duration_sec=end_sec - start_sec,
            width=width,
            height=height,
            fps=fps,
        )
        overlay = {
            "start": round(start_sec, 3),
            "end": round(end_sec, 3),
            "asset_path": str(prepared_asset),
            "source_asset_path": str(asset_path),
            "compose_mode": composition_mode,
            "position": str(params.get("position") or "bottom_right"),
            "scale": float(params.get("scale") or 0.42),
            "visual_id": "manual_asset_001",
            "card_id": "manual_asset_001",
            "renderer": render_info.get("renderer"),
            "html_render_mode": render_info.get("html_render_mode"),
            "manual_visual_asset": True,
        }
        output_path = apply_visual_overlays(state.working_file, state.working_dir, [overlay])
        state.working_file = output_path
        state.metadata = probe_video(output_path)
        manifest = {
            "created_at": utc_now_iso(),
            "project_id": state.project_id,
            "project_name": state.project_name,
            "source_video": state.source_files[0] if state.source_files else state.working_file,
            "working_file": state.working_file,
            "asset_path": str(asset_path),
            "prepared_asset_path": str(prepared_asset),
            "composition_mode": composition_mode,
            "renderer": render_info.get("renderer"),
            "html_render_mode": render_info.get("html_render_mode"),
            "manual_visual_specs": True,
            "transcript_scoring_bypassed": True,
            "overlays": [overlay],
            "render_info": render_info,
        }
        manifest_path = bundle_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        state.artifacts["latest_manual_visual_asset"] = {
            "created_at": manifest["created_at"],
            "manifest_path": str(manifest_path),
            "bundle_dir": str(bundle_dir),
            "asset_path": str(asset_path),
            "start": round(start_sec, 3),
            "end": round(end_sec, 3),
            "composition_mode": composition_mode,
        }
        history = list(state.artifacts.get("manual_visual_asset_history") or [])
        history.append(state.artifacts["latest_manual_visual_asset"])
        state.artifacts["manual_visual_asset_history"] = history[-10:]
        state.apply_operation(
            {
                "op": "add_visual_asset",
                "params": {
                    "asset_path": str(asset_path),
                    "start": round(start_sec, 3),
                    "end": round(end_sec, 3),
                    "composition_mode": composition_mode,
                    "manifest_path": str(manifest_path),
                    "overlays": [overlay],
                },
                "timestamp": utc_now_iso(),
                "result_file": output_path,
                "description": f"Inserted manual visual asset from {start_sec:.2f}s to {end_sec:.2f}s",
            }
        )
        return {
            "success": True,
            "message": (
                f"Inserted manual visual asset from {start_sec:.2f}s to {end_sec:.2f}s "
                f"as {composition_mode}. Transcript scoring was bypassed; render/timing validation passed. "
                f"Manifest: {manifest_path}"
            ),
            "suggestion": None,
            "updated_state": state,
            "tool_name": "add_visual_asset",
        }
    except (UnsafeInputPathError, ValueError, VideoEngineError) as exc:
        return {
            "success": False,
            "message": str(exc),
            "suggestion": None,
            "updated_state": state,
            "tool_name": "add_visual_asset",
        }
