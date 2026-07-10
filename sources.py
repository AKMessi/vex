from __future__ import annotations

import os
import re
from collections.abc import Mapping
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


def _is_within_directory(path: Path, directory: Path) -> bool:
    try:
        return os.path.commonpath(
            [
                os.path.normcase(os.path.abspath(str(path))),
                os.path.normcase(os.path.abspath(str(directory))),
            ]
        ) == os.path.normcase(os.path.abspath(str(directory)))
    except ValueError:
        return False


def normalize_source_url(url: str) -> str:
    return _strip_url_punctuation(url.strip().strip('"').strip("'"))


def extract_youtube_url(text: str) -> str | None:
    match = YOUTUBE_URL_PATTERN.search(text)
    if not match:
        return None
    return normalize_source_url(match.group(0))


def _resolve_downloaded_path(
    info: dict,
    destination_dir: Path,
    ydl,
    *,
    preexisting_paths: set[str] | None = None,
) -> str:
    destination_root = destination_dir.expanduser().resolve(strict=True)
    excluded = {
        os.path.normcase(os.path.abspath(path))
        for path in (preexisting_paths or set())
    }
    candidates: list[str] = []
    for item in info.get("requested_downloads") or []:
        if not isinstance(item, Mapping):
            continue
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
        candidate_path = Path(candidate).expanduser()
        try:
            resolved_candidate = candidate_path.resolve(strict=True)
        except OSError:
            resolved_candidate = None
        if resolved_candidate is not None:
            candidate_key = os.path.normcase(os.path.abspath(str(resolved_candidate)))
            if candidate_key not in seen:
                seen.add(candidate_key)
                if (
                    candidate_key not in excluded
                    and _is_within_directory(resolved_candidate, destination_root)
                    and resolved_candidate.is_file()
                ):
                    return str(resolved_candidate)
        base = candidate_path
        for extension in sorted(VIDEO_EXTENSIONS):
            try:
                alternative = base.with_suffix(extension).resolve(strict=True)
            except OSError:
                continue
            alternative_key = os.path.normcase(os.path.abspath(str(alternative)))
            if alternative_key in seen or alternative_key in excluded:
                continue
            seen.add(alternative_key)
            if _is_within_directory(alternative, destination_root) and alternative.is_file():
                return str(alternative)

    downloaded_files: list[Path] = []
    for path in destination_root.glob("*"):
        try:
            resolved_path = path.resolve(strict=True)
        except OSError:
            continue
        resolved_key = os.path.normcase(os.path.abspath(str(resolved_path)))
        if (
            resolved_key not in excluded
            and _is_within_directory(resolved_path, destination_root)
            and resolved_path.is_file()
            and resolved_path.suffix.lower() in VIDEO_EXTENSIONS
        ):
            downloaded_files.append(resolved_path)
    if downloaded_files:
        try:
            downloaded_files.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
        except OSError as exc:
            raise RuntimeError("Could not inspect downloaded YouTube files.") from exc
        return str(downloaded_files[0].resolve())
    raise RuntimeError("yt-dlp reported success, but no downloaded video file was found.")


def download_youtube_video(url: str, working_dir: str) -> DownloadedVideo:
    normalized_url = normalize_source_url(url)
    if not YOUTUBE_URL_PATTERN.fullmatch(normalized_url):
        raise RuntimeError("Only standard YouTube video, Shorts, or live URLs are supported.")
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError(
            "yt-dlp is not installed. Install it with `pip install yt-dlp` to enable YouTube source downloads."
        ) from exc

    destination_dir = Path(working_dir) / "downloads"
    destination_dir.mkdir(parents=True, exist_ok=True)
    preexisting_paths = {
        os.path.normcase(os.path.abspath(str(path.resolve(strict=False))))
        for path in destination_dir.glob("*")
    }
    options = {
        "format": "bv*+ba/b",
        "merge_output_format": "mp4",
        "outtmpl": str(destination_dir / "source_%(id).80s.%(ext)s"),
        "restrictfilenames": True,
        "windowsfilenames": True,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 30,
        "retries": 3,
        "fragment_retries": 3,
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(normalized_url, download=True)
        if info is None:
            raise RuntimeError("Failed to fetch video metadata from YouTube.")
        if not isinstance(info, Mapping):
            raise RuntimeError("YouTube returned an invalid metadata payload.")
        if "entries" in info:
            info = next((entry for entry in info.get("entries") or [] if entry), None) or info
        if not isinstance(info, Mapping):
            raise RuntimeError("YouTube returned an invalid metadata payload.")
        downloaded_path = _resolve_downloaded_path(
            dict(info),
            destination_dir,
            ydl,
            preexisting_paths=preexisting_paths,
        )
        return DownloadedVideo(
            source_url=normalized_url,
            title=str(info.get("title") or Path(downloaded_path).stem),
            video_id=str(info.get("id") or "youtube-video"),
            uploader=str(info.get("uploader") or info.get("channel") or ""),
            downloaded_path=downloaded_path,
        )
