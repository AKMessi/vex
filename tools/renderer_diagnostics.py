from __future__ import annotations

import shutil
import subprocess
from typing import Any

import config
from renderers import renderer_capabilities
from renderers.hyperframes_renderer import _hyperframes_cli_path, _node_major_version
from renderers.remotion_renderer import (
    _node_platform_arch,
    _remotion_platform_blocker,
    _run_node_package_probe,
)
from vex_runtime.hyperframes import installed_runtime_status
from vex_runtime.imaging import imaging_runtime_status


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
    managed_runtime = installed_runtime_status()
    imaging = imaging_runtime_status()
    node_major = _node_major_version()
    ffmpeg_path = shutil.which(config.FFMPEG_PATH)
    manim_path = shutil.which("manim")
    blender_path = shutil.which(config.BLENDER_PATH)
    remotion_platform, remotion_arch, remotion_platform_reason = _node_platform_arch()
    remotion_platform_blocker = _remotion_platform_blocker(
        remotion_platform,
        remotion_arch,
    )
    remotion_packages_available, remotion_reason = _run_node_package_probe()
    remotion_node_reason = (
        f"Node.js {node_major} is below required version 22"
        if node_major is not None and node_major < 22
        else ""
    )
    remotion_blocker_reason = (
        remotion_platform_blocker
        or remotion_platform_reason
        or remotion_node_reason
        or remotion_reason
    )
    hyperframes_blockers = []
    if not hyperframes_cli:
        hyperframes_blockers.append(
            "HyperFrames CLI missing; run `vex renderers install hyperframes`"
        )
    if node_major is None:
        hyperframes_blockers.append("Node.js missing")
    elif node_major < 22:
        hyperframes_blockers.append(f"Node.js {node_major} is below required version 22")
    if not ffmpeg_path:
        hyperframes_blockers.append("FFmpeg missing")
    if not imaging["available"]:
        hyperframes_blockers.append(str(imaging["reason"]))
    managed_cli_path = str(managed_runtime.get("cli_path") or "")
    if hyperframes_cli and managed_cli_path and hyperframes_cli == managed_cli_path:
        hyperframes_source = "managed"
    elif hyperframes_cli:
        hyperframes_source = "configured_or_repository"
    else:
        hyperframes_source = "missing"
    report = {
        "hyperframes": {
            "available": bool(
                hyperframes_cli
                and node_major
                and node_major >= 22
                and ffmpeg_path
                and imaging["available"]
            ),
            "cli_path": hyperframes_cli,
            "node_major": node_major,
            "source": hyperframes_source,
            "reason": "; ".join(hyperframes_blockers),
            "managed_runtime": managed_runtime,
        },
        "imaging": imaging,
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
        "remotion": {
            "available": bool(
                node_major
                and node_major >= 22
                and remotion_packages_available
                and not remotion_platform_blocker
                and not remotion_platform_reason
            ),
            "node_major": node_major,
            "platform": remotion_platform,
            "arch": remotion_arch,
            "package_version": "4.0.487" if remotion_packages_available else None,
            "reason": remotion_blocker_reason,
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
        for name in ("hyperframes", "imaging", "ffmpeg", "manim", "remotion", "blender")
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
