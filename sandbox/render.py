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


def run() -> None:
    if sys.argv[1:] == ["extract"]:
        extract()
        return
    engine, width, height, fps, frames = sys.argv[1:]
    with tempfile.TemporaryDirectory(dir="/tmp") as name:
        workspace = Path(name)
        project = workspace / "project"
        shutil.copytree("/input", project)
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
        subprocess.run(command, cwd=project, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=235, check=True)
        if engine == "manim":
            candidates = list((workspace / "media" / "videos").glob("**/scene.mp4"))
            if len(candidates) != 1:
                raise ValueError("The Manim scene did not produce a unique video.")
            staged = candidates[0]
        subprocess.run([
            "ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(staged),
            "-vf", f"fps={fps},scale={width}:{height}:force_original_aspect_ratio=decrease,"
                   f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1",
            "-frames:v", frames, "-an", "-c:v", "libx264", "-threads", "2",
            "-preset", "fast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(FINAL),
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=50, check=True)


if __name__ == "__main__":
    run()
