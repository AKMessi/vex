from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".flv"}
YOUTUBE_URL_PATTERN = re.compile(
    r"https?://(?:www\.)?(?:youtube\.com/(?:watch\?[^\s\"'<>]+|shorts/[^\s\"'<>]+|live/[^\s\"'<>]+)|youtu\.be/[^\s\"'<>]+)",
    re.IGNORECASE,
)


@dataclass
class DownloadedVideo:
    source_url: str
    title: str
    video_id: str
    uploader: str
    downloaded_path: str


def _strip_url_punctuation(url: str) -> str:
    return url.rstrip(").,!?:;]}")


def normalize_source_url(url: str) -> str:
    return _strip_url_punctuation(url.strip().strip('"').strip("'"))


def extract_youtube_url(text: str) -> str | None:
    match = YOUTUBE_URL_PATTERN.search(text)
    if not match:
        return None
    return normalize_source_url(match.group(0))


def _resolve_downloaded_path(info: dict, destination_dir: Path, ydl) -> str:
    candidates: list[str] = []
    for item in info.get("requested_downloads") or []:
        filepath = item.get("filepath")
        if filepath:
            candidates.append(filepath)
    for key in ("filepath", "_filename"):
        value = info.get(key)
        if value:
            candidates.append(value)
    try:
        candidates.append(ydl.prepare_filename(info))
    except Exception:
        pass

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate:
            continue
        absolute_candidate = os.path.abspath(candidate)
        if absolute_candidate in seen:
            continue
        seen.add(absolute_candidate)
        if os.path.isfile(absolute_candidate):
            return absolute_candidate
        base = Path(absolute_candidate)
        for extension in VIDEO_EXTENSIONS:
            alternative = str(base.with_suffix(extension))
            if alternative not in seen and os.path.isfile(alternative):
                return alternative

    downloaded_files = [
        path for path in destination_dir.glob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    if downloaded_files:
        downloaded_files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        return str(downloaded_files[0].resolve())
    raise RuntimeError("yt-dlp reported success, but no downloaded video file was found.")


def download_youtube_video(url: str, working_dir: str) -> DownloadedVideo:
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError(
            "yt-dlp is not installed. Install it with `pip install yt-dlp` to enable YouTube source downloads."
        ) from exc

    normalized_url = normalize_source_url(url)
    destination_dir = Path(working_dir) / "downloads"
    destination_dir.mkdir(parents=True, exist_ok=True)
    options = {
        "format": "bv*+ba/b",
        "merge_output_format": "mp4",
        "outtmpl": str(destination_dir / "%(title).80s [%(id)s].%(ext)s"),
        "restrictfilenames": False,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(normalized_url, download=True)
        if info is None:
            raise RuntimeError("Failed to fetch video metadata from YouTube.")
        if "entries" in info:
            info = next((entry for entry in info.get("entries") or [] if entry), None) or info
        downloaded_path = _resolve_downloaded_path(info, destination_dir, ydl)
        return DownloadedVideo(
            source_url=normalized_url,
            title=str(info.get("title") or Path(downloaded_path).stem),
            video_id=str(info.get("id") or "youtube-video"),
            uploader=str(info.get("uploader") or info.get("channel") or ""),
            downloaded_path=downloaded_path,
        )
