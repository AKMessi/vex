"""Trusted entry points for a native scene container."""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path


FINAL = Path("/out/clip.mp4")
DIAGNOSTIC = Path("/out/render.log")
MAX_BYTES = 128 * 1024 * 1024


def extract() -> None:
    descriptor = os.open(FINAL, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as source:
        metadata = os.fstat(source.fileno())
        if not stat.S_ISREG(metadata.st_mode) or not 0 < metadata.st_size <= MAX_BYTES:
            raise ValueError("Missing or oversized sandbox output")
        remaining = metadata.st_size
        while remaining:
            block = source.read(min(remaining, 1024 * 1024))
            if not block:
                raise ValueError("The sandbox output changed during extraction")
            sys.stdout.buffer.write(block)
            remaining -= len(block)


def extract_log() -> None:
    descriptor = os.open(DIAGNOSTIC, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as log:
        metadata = os.fstat(log.fileno())
        if stat.S_ISREG(metadata.st_mode):
            log.seek(max(0, metadata.st_size - 12000))
            sys.stdout.buffer.write(log.read(12000))


def invoke(command: list[str], timeout: int, cwd: Path | None = None) -> None:
    descriptor = os.open(DIAGNOSTIC, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    with os.fdopen(descriptor, "ab") as log:
        if not stat.S_ISREG(os.fstat(log.fileno()).st_mode):
            raise ValueError("Invalid sandbox diagnostics file")
        subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=True)


def run() -> None:
    if sys.argv[1:] == ["extract"]:
        extract()
        return
    if sys.argv[1:] == ["extract-log"]:
        extract_log()
        return
    engine, width, height, fps, frames = sys.argv[1:]
    # Limit the number of cores reported to browser and codec subprocesses.
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    with tempfile.TemporaryDirectory(dir="/tmp") as name:
        workspace = Path(name)
        project = workspace / "project"
        shutil.copytree("/input", project)
        for item in project.iterdir():
            item.chmod(0o600)
        staged = workspace / "scene.mp4"
        if engine == "remotion":
            command = ["node", "/opt/vex/sandbox/remotion.mjs", str(project), str(staged), width, height, fps, frames]
        elif engine == "manim":
            command = [sys.executable, "-m", "manim", str(project / "scene.py"), "VexScene", "-o", "scene.mp4",
                       "--fps", fps, "-r", f"{width},{height}", "--disable_caching", "--media_dir", str(workspace / "media")]
        elif engine == "hyperframes":
            command = ["/opt/vex/node_modules/.bin/hyperframes", "render", "--output", str(staged),
                       "--fps", fps, "--quality", "standard", "--format", "mp4", "--workers", "1", "."]
        else:
            raise ValueError("Unknown engine.")
        invoke(command, 235, project)
        if engine == "manim":
            candidates = list((workspace / "media" / "videos").glob("**/scene.mp4"))
            if len(candidates) != 1:
                raise ValueError("The Manim scene did not produce a unique video.")
            staged = candidates[0]
        invoke([
            "ffmpeg", "-nostdin", "-v", "error", "-filter_threads", "1", "-y", "-i", str(staged),
            "-vf", f"fps={fps},scale={width}:{height}:force_original_aspect_ratio=decrease,"
                   f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1",
            "-frames:v", frames, "-an", "-c:v", "libx264", "-threads", "2",
            "-preset", "fast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(FINAL),
        ], 50)


if __name__ == "__main__":
    run()
