from __future__ import annotations

import shutil
import subprocess
from typing import Any

import config
from renderers import renderer_capabilities
from renderers.hyperframes_renderer import _hyperframes_cli_path, _node_major_version


def _version(command: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "reason": str(exc), "version": None}
    output = (result.stdout or result.stderr or "").strip().splitlines()
    return {
        "available": result.returncode == 0,
        "reason": "" if result.returncode == 0 else (result.stderr or result.stdout or "").strip(),
        "version": output[0] if output else None,
    }


def renderer_doctor_report() -> dict[str, Any]:
    hyperframes_cli = _hyperframes_cli_path()
    node_major = _node_major_version()
    ffmpeg_path = shutil.which(config.FFMPEG_PATH)
    manim_path = shutil.which("manim")
    blender_path = shutil.which(config.BLENDER_PATH)
    report = {
        "hyperframes": {
            "available": bool(hyperframes_cli and node_major and node_major >= 22 and ffmpeg_path),
            "cli_path": hyperframes_cli,
            "node_major": node_major,
            "reason": (
                ""
                if hyperframes_cli and node_major and node_major >= 22 and ffmpeg_path
                else "Requires local HyperFrames CLI, Node.js 22+, and FFmpeg."
            ),
        },
        "node": {
            "available": node_major is not None,
            "major": node_major,
            "version": _version(["node", "--version"]).get("version") if shutil.which("node") else None,
        },
        "ffmpeg": {
            "available": ffmpeg_path is not None,
            "path": ffmpeg_path,
            "version": _version([config.FFMPEG_PATH, "-version"]).get("version") if ffmpeg_path else None,
        },
        "manim": {
            "available": manim_path is not None,
            "path": manim_path,
            "version": _version(["manim", "--version"]).get("version") if manim_path else None,
        },
        "blender": {
            "available": blender_path is not None,
            "path": blender_path,
            "version": _version([config.BLENDER_PATH, "--version"]).get("version") if blender_path else None,
        },
        "renderer_capabilities": renderer_capabilities(),
    }
    return report


def execute(_params: dict, state: object | None = None) -> dict:
    report = renderer_doctor_report()
    unavailable = [
        name
        for name in ("hyperframes", "ffmpeg", "manim", "blender")
        if not bool((report.get(name) or {}).get("available"))
    ]
    return {
        "success": True,
        "message": (
            "Renderer doctor completed. "
            + (f"Unavailable: {', '.join(unavailable)}." if unavailable else "All primary renderers are available.")
        ),
        "suggestion": None,
        "updated_state": state,
        "tool_name": "renderers_doctor",
        "report": report,
    }
