"""On-demand browser playback for footage the browser cannot decode."""
from __future__ import annotations

import hashlib
import os
import subprocess
import uuid
from pathlib import Path

import config
from vex_runtime.locking import exclusive_file_lock


def media_fingerprint(path: Path) -> str:
    stat = path.stat()
    return hashlib.sha256(f"{path}|{stat.st_size}|{stat.st_mtime_ns}".encode()).hexdigest()[:16]


def browser_preview(working_dir: Path, source: Path, *, prepare: bool = True) -> Path:
    directory = working_dir.resolve() / ".studio-previews"
    if directory.resolve() != directory:
        raise ValueError("Browser previews must stay in the project workspace.")
    fingerprint = media_fingerprint(source)
    file = directory / f"{fingerprint}.mp4"
    if file.resolve() != file:
        raise ValueError("The browser preview must stay in the project workspace.")
    if not prepare:
        if not file.is_file():
            raise FileNotFoundError("The browser preview is not ready.")
        return file

    with exclusive_file_lock(file.with_suffix(".lock"), timeout_sec=180):
        if file.is_file():
            return file
        temporary = directory / f".{uuid.uuid4().hex}.mp4"
        try:
            process = subprocess.run(
                [
                    config.FFMPEG_PATH, "-nostdin", "-v", "error", "-y", "-i", str(source),
                    "-map", "0:v:0", "-map", "0:a:0?", "-vf",
                    r"scale=w=min(960\,iw):h=min(960\,ih):force_original_aspect_ratio=decrease:force_divisible_by=2",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                    "-threads", "2", "-c:a", "aac", "-ac", "2", "-movflags", "+faststart", str(temporary),
                ],
                capture_output=True, timeout=180, check=False,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            if process.returncode or not temporary.is_file():
                raise ValueError("FFmpeg could not create a browser preview for this file.")
            if media_fingerprint(source) != fingerprint:
                raise ValueError("The video changed while the preview was being prepared.")
            temporary.replace(file)
        except subprocess.TimeoutExpired as exc:
            raise ValueError("The browser preview took too long to prepare.") from exc
        finally:
            temporary.unlink(missing_ok=True)
    return file
