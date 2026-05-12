from __future__ import annotations

import os
import re
import shlex
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import config
from engine import VideoEngineError, parse_timestamp, probe_video
from tools.path_security import UnsafeOutputPathError, is_trusted_output_path_request, resolve_output_path


DEFAULT_AVAILABLE_ENCODERS = {"aac", "libx264", "libx265", "libaom-av1", "libvpx-vp9", "libopus", "libmp3lame"}
VIDEO_CODEC_ALIASES = {
    "h264": "h264",
    "h.264": "h264",
    "avc": "h264",
    "x264": "h264",
    "hevc": "hevc",
    "h265": "hevc",
    "h.265": "hevc",
    "x265": "hevc",
    "av1": "av1",
    "vp9": "vp9",
    "prores": "prores",
}
VIDEO_ENCODERS = {
    "h264": "libx264",
    "hevc": "libx265",
    "av1": "libaom-av1",
    "vp9": "libvpx-vp9",
    "prores": "prores_ks",
}
VIDEO_DECODABLE_IN_MP4 = {"h264", "hevc"}
MP4_AUDIO_COPY_CODECS = {"aac", "mp3", "ac3", "eac3", "alac"}
VIDEO_CONTAINERS = {"mp4", "mov", "mkv", "webm", "m4v"}
NULL_OUTPUT = os.devnull


class EncodePlanningError(ValueError):
    pass


@dataclass(frozen=True)
class EncodeIntent:
    raw_request: str = ""
    target_format: str = "mp4"
    video_codec: str | None = None
    audio_codec: str | None = None
    quality: str = "balanced"
    optimize_for: str = "compatibility_quality"
    target_size_mb: float | None = None
    max_width: int | None = None
    max_height: int | None = None
    fps: float | None = None
    strip_audio: bool = False
    copy_streams: bool | None = None
    output_path: str | None = None
    allow_overwrite: bool = False


@dataclass(frozen=True)
class EncodePlan:
    plan_id: str
    created_at: str
    input_path: str
    output_path: str
    intent: dict[str, Any]
    source_metadata: dict[str, Any]
    commands: list[list[str]]
    display_command: str
    mode: str
    summary: str
    warnings: list[str] = field(default_factory=list)
    estimated_size_bytes: int | None = None
    passlog_file: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def available_ffmpeg_encoders(ffmpeg_path: str | None = None) -> set[str]:
    command = [ffmpeg_path or config.FFMPEG_PATH, "-hide_banner", "-encoders"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=12)
    except (OSError, subprocess.TimeoutExpired):
        return set(DEFAULT_AVAILABLE_ENCODERS)
    if result.returncode != 0:
        return set(DEFAULT_AVAILABLE_ENCODERS)
    encoders: set[str] = set()
    for line in result.stdout.splitlines():
        match = re.match(r"^\s*[A-Z.]{6}\s+(\S+)\s", line)
        if match:
            encoders.add(match.group(1))
    return encoders or set(DEFAULT_AVAILABLE_ENCODERS)


def source_fingerprint(input_path: str, *, timeline_count: int = 0) -> dict[str, Any]:
    path = Path(input_path)
    try:
        stat = path.stat()
        size = stat.st_size
        mtime_ns = stat.st_mtime_ns
    except OSError:
        size = None
        mtime_ns = None
    return {
        "path": str(path.resolve()),
        "size_bytes": size,
        "mtime_ns": mtime_ns,
        "timeline_count": int(timeline_count),
    }


def pending_plan_is_current(plan: dict[str, Any], input_path: str, *, timeline_count: int = 0) -> bool:
    expected = plan.get("source_fingerprint")
    if not isinstance(expected, dict):
        return False
    return expected == source_fingerprint(input_path, timeline_count=timeline_count)


def intent_from_params(params: dict[str, Any]) -> EncodeIntent:
    raw_request = str(params.get("raw_request") or params.get("request") or "").strip()
    output_path = _clean_output_path(params.get("output_path"))
    output_format = _normalize_format(Path(output_path).suffix) if output_path else None
    target_format = _normalize_format(params.get("target_format") or params.get("format") or "")
    parsed_from_text = _parse_raw_request(raw_request)
    values: dict[str, Any] = {
        "raw_request": raw_request,
        "target_format": target_format or parsed_from_text.get("target_format") or output_format or "mp4",
        "video_codec": _normalize_video_codec(params.get("video_codec")) or parsed_from_text.get("video_codec"),
        "audio_codec": _normalize_audio_codec(params.get("audio_codec")) or parsed_from_text.get("audio_codec"),
        "quality": _normalize_choice(params.get("quality"), {"max", "high", "balanced", "small"}, None)
        or parsed_from_text.get("quality")
        or "balanced",
        "optimize_for": _normalize_optimize_for(params.get("optimize_for"))
        or parsed_from_text.get("optimize_for")
        or "compatibility_quality",
        "target_size_mb": _positive_float(params.get("target_size_mb")) or parsed_from_text.get("target_size_mb"),
        "max_width": _positive_int(params.get("max_width")) or parsed_from_text.get("max_width"),
        "max_height": _positive_int(params.get("max_height")) or parsed_from_text.get("max_height"),
        "fps": _positive_float(params.get("fps")) or parsed_from_text.get("fps"),
        "strip_audio": _truthy(params.get("strip_audio")) or bool(parsed_from_text.get("strip_audio")),
        "copy_streams": _optional_bool(params.get("copy_streams")),
        "output_path": output_path,
        "allow_overwrite": _truthy(params.get("allow_overwrite")),
    }
    if values["copy_streams"] is None and "copy_streams" in parsed_from_text:
        values["copy_streams"] = bool(parsed_from_text["copy_streams"])
    return EncodeIntent(**values)


def build_encode_plan(
    input_path: str,
    output_dir: str,
    project_name: str,
    params: dict[str, Any],
    *,
    metadata: dict[str, Any] | None = None,
    available_encoders: set[str] | None = None,
) -> EncodePlan:
    intent = intent_from_params(params)
    source_metadata = dict(metadata or probe_video(input_path))
    encoders = available_encoders if available_encoders is not None else available_ffmpeg_encoders()
    warnings: list[str] = []
    target_format = _container_for_intent(intent)
    output_path = _resolve_output_path(
        output_dir,
        project_name,
        target_format,
        intent.output_path,
        allow_overwrite=intent.allow_overwrite,
        trusted=is_trusted_output_path_request(params),
    )
    requested_video_codec = intent.video_codec or _default_video_codec(intent, target_format)
    selected_video_encoder = _select_video_encoder(
        requested_video_codec,
        encoders,
        warnings,
        target_format=target_format,
    )
    video_codec = str(source_metadata.get("codec") or "").lower()
    audio_codec = str(source_metadata.get("audio_codec") or "").lower()
    has_audio = bool(source_metadata.get("has_audio"))
    if intent.copy_streams is True:
        _validate_explicit_stream_copy(
            target_format=target_format,
            source_video_codec=video_codec,
            source_audio_codec=audio_codec,
            has_audio=has_audio,
        )
    stream_copy = _can_stream_copy(
        intent,
        target_format=target_format,
        source_video_codec=video_codec,
        source_audio_codec=audio_codec,
        has_audio=has_audio,
    )
    filters = _video_filters(intent, source_metadata, warnings)
    fps_args = _fps_args(intent, source_metadata)

    if stream_copy:
        commands = [
            _stream_copy_command(
                input_path,
                output_path,
                target_format=target_format,
                has_audio=has_audio,
                source_audio_codec=audio_codec,
            )
        ]
        mode = "stream_copy"
        summary = "Remux compatible streams without re-encoding."
        estimated_size = int(source_metadata.get("size_bytes") or 0) or None
        passlog_file = None
    elif intent.target_size_mb:
        commands, estimated_size, passlog_file = _two_pass_commands(
            input_path,
            output_path,
            intent,
            source_metadata,
            selected_video_encoder,
            filters,
            fps_args,
            has_audio=has_audio,
            warnings=warnings,
        )
        mode = "two_pass_target_size"
        summary = (
            f"Two-pass {_codec_label_for_encoder(selected_video_encoder)} encode "
            f"targeting about {intent.target_size_mb:.1f} MB."
        )
    else:
        commands = [
            _crf_command(
                input_path,
                output_path,
                intent,
                source_metadata,
                selected_video_encoder,
                filters,
                fps_args,
                has_audio=has_audio,
            )
        ]
        mode = "quality_crf"
        summary = f"{_codec_label_for_encoder(selected_video_encoder)} quality encode using CRF {_crf_for_intent(intent)}."
        estimated_size = None
        passlog_file = None

    return EncodePlan(
        plan_id=uuid.uuid4().hex,
        created_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        input_path=str(Path(input_path).resolve()),
        output_path=output_path,
        intent=asdict(intent),
        source_metadata=source_metadata,
        commands=commands,
        display_command=_display_command(commands),
        mode=mode,
        summary=summary,
        warnings=warnings,
        estimated_size_bytes=estimated_size,
        passlog_file=passlog_file,
    )


def run_encode_plan(
    plan: dict[str, Any],
    *,
    progress_callback: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    commands = plan.get("commands")
    if not isinstance(commands, list) or not commands:
        raise VideoEngineError("Encode plan has no ffmpeg commands.")
    duration = float((plan.get("source_metadata") or {}).get("duration_sec") or 0.0)
    total_steps = len(commands)
    try:
        for index, command in enumerate(commands, start=1):
            if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
                raise VideoEngineError("Encode plan contains an invalid command.")
            step_offset = (index - 1) / total_steps
            step_fraction = 1.0 / total_steps

            def on_step_progress(value: float) -> None:
                if progress_callback is not None:
                    progress_callback(min(step_offset + value * step_fraction, 1.0))

            _run_ffmpeg_command(command, duration=duration, progress_callback=on_step_progress)
        if progress_callback is not None:
            progress_callback(1.0)
    finally:
        _cleanup_pass_logs(plan.get("passlog_file"))
    output_path = str(plan.get("output_path") or "")
    if not output_path or not Path(output_path).is_file():
        raise VideoEngineError("Encode completed but the expected output file was not created.")
    output_metadata = probe_video(output_path)
    return {
        "output_path": output_path,
        "output_metadata": output_metadata,
        "output_size_bytes": int(Path(output_path).stat().st_size),
        "validation": validate_output(plan, output_metadata),
    }


def validate_output(plan: dict[str, Any], output_metadata: dict[str, Any]) -> dict[str, Any]:
    intent = plan.get("intent") or {}
    expected_format = str(intent.get("target_format") or "").lower()
    format_name = str(output_metadata.get("format") or "").lower()
    warnings: list[str] = []
    if expected_format == "mp4" and "mp4" not in format_name and "mov" not in format_name:
        warnings.append(f"Output format probe reported {format_name!r}, not mp4-compatible.")
    if intent.get("strip_audio") and output_metadata.get("has_audio"):
        warnings.append("Audio was expected to be stripped, but the output still has audio.")
    return {
        "ok": not warnings,
        "warnings": warnings,
        "format": output_metadata.get("format"),
        "codec": output_metadata.get("codec"),
        "has_audio": output_metadata.get("has_audio"),
        "width": output_metadata.get("width"),
        "height": output_metadata.get("height"),
        "fps": output_metadata.get("fps"),
    }


def _parse_raw_request(raw_request: str) -> dict[str, Any]:
    text = raw_request.lower()
    parsed: dict[str, Any] = {}
    fmt_match = re.search(r"\b(?:to|as|into|format)\s+\.?(mp4|mov|mkv|webm|m4v)\b", text)
    if not fmt_match:
        fmt_match = re.search(r"\.(mp4|mov|mkv|webm|m4v)\b", text)
    if fmt_match:
        parsed["target_format"] = fmt_match.group(1)
    for alias, codec in VIDEO_CODEC_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", text):
            parsed["video_codec"] = codec
            break
    if re.search(r"\b(?:without|no|remove|strip)\s+audio\b|\bmute(?:d)?\s+video\b", text):
        parsed["strip_audio"] = True
    size_match = re.search(
        r"\b(?:under|below|less than|smaller than|to|around|about|target(?:ing)?|maximum|max)\s+"
        r"(\d+(?:\.\d+)?)\s*(mb|mib|gb|gib)\b",
        text,
    )
    if size_match:
        value = float(size_match.group(1))
        unit = size_match.group(2)
        parsed["target_size_mb"] = value * 1024 if unit.startswith("g") else value
    resolution = _parse_resolution(text)
    if resolution:
        parsed["max_width"], parsed["max_height"] = resolution
    fps_match = re.search(r"\b(\d+(?:\.\d+)?)\s*fps\b", text)
    if fps_match:
        parsed["fps"] = float(fps_match.group(1))
    if re.search(r"\b(?:compress|smaller|reduce(?:\s+the)?\s+size|file\s+size|shrink)\b", text):
        parsed["quality"] = "balanced"
        parsed["optimize_for"] = "compatibility_quality"
    if re.search(r"\b(?:smallest|tiny|minimum\s+size|max(?:imum)?\s+compression)\b", text):
        parsed["quality"] = "small"
        parsed["optimize_for"] = "smallest_size"
    if re.search(r"\b(?:without losing much quality|little quality loss|near original|high quality)\b", text):
        parsed["quality"] = "balanced"
        parsed["optimize_for"] = "compatibility_quality"
    if re.search(r"\b(?:fast|quick|quickly|draft)\b", text):
        parsed["optimize_for"] = "fastest"
    if re.search(r"\b(?:remux|no re-encode|without re-encoding|stream copy|copy streams)\b", text):
        parsed["copy_streams"] = True
    return parsed


def _parse_resolution(text: str) -> tuple[int, int] | None:
    exact = re.search(r"\b(\d{3,5})\s*x\s*(\d{3,5})\b", text)
    if exact:
        return int(exact.group(1)), int(exact.group(2))
    if re.search(r"\b4k\b|\buhd\b", text):
        return 3840, 2160
    height_match = re.search(r"\b(2160|1440|1080|720|480|360)p\b", text)
    if height_match:
        height = int(height_match.group(1))
        return _even_int(height * 16 / 9), height
    return None


def _normalize_format(value: Any) -> str | None:
    cleaned = str(value or "").strip().lower().lstrip(".")
    return cleaned if cleaned in VIDEO_CONTAINERS else None


def _normalize_video_codec(value: Any) -> str | None:
    cleaned = str(value or "").strip().lower()
    return VIDEO_CODEC_ALIASES.get(cleaned)


def _normalize_audio_codec(value: Any) -> str | None:
    cleaned = str(value or "").strip().lower()
    if cleaned in {"aac", "mp3", "opus", "copy"}:
        return cleaned
    if cleaned in {"none", "no_audio", "strip"}:
        return "none"
    return None


def _normalize_choice(value: Any, allowed: set[str], default: str | None) -> str | None:
    cleaned = str(value or "").strip().lower()
    if cleaned in allowed:
        return cleaned
    return default


def _normalize_optimize_for(value: Any) -> str | None:
    cleaned = str(value or "").strip().lower()
    aliases = {
        "compatibility": "compatibility_quality",
        "quality": "compatibility_quality",
        "compatibility_quality": "compatibility_quality",
        "small": "smallest_size",
        "size": "smallest_size",
        "smallest_size": "smallest_size",
        "fast": "fastest",
        "speed": "fastest",
        "fastest": "fastest",
    }
    return aliases.get(cleaned)


def _positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    cleaned = str(value).strip().lower()
    if cleaned in {"1", "true", "yes", "y", "on"}:
        return True
    if cleaned in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _clean_output_path(value: Any) -> str | None:
    cleaned = str(value or "").strip()
    return cleaned or None


def _container_for_intent(intent: EncodeIntent) -> str:
    if intent.target_format == "m4v":
        return "mp4"
    return intent.target_format or "mp4"


def _default_video_codec(intent: EncodeIntent, target_format: str) -> str:
    if target_format == "webm":
        return "vp9"
    if intent.optimize_for == "smallest_size" and intent.video_codec:
        return intent.video_codec
    return "h264"


def _select_video_encoder(
    codec: str,
    available_encoders: set[str],
    warnings: list[str],
    *,
    target_format: str,
) -> str:
    encoder = VIDEO_ENCODERS.get(codec)
    if not encoder:
        raise EncodePlanningError(f"Unsupported video codec: {codec}")
    if encoder in available_encoders:
        return encoder
    if target_format == "webm":
        raise EncodePlanningError(f"{encoder} is required for WebM output but is not available in this FFmpeg build.")
    if codec != "h264" and "libx264" in available_encoders:
        warnings.append(f"{encoder} is not available in this FFmpeg build; using libx264 for compatibility.")
        return "libx264"
    if encoder not in available_encoders:
        warnings.append(f"Could not confirm {encoder} availability; the encode may fail if FFmpeg lacks it.")
    return encoder


def _resolve_output_path(
    output_dir: str,
    project_name: str,
    target_format: str,
    output_path: str | None,
    *,
    allow_overwrite: bool,
    trusted: bool = False,
) -> str:
    if output_path:
        try:
            candidate = resolve_output_path(
                output_path,
                default_root=output_dir,
                allowed_roots=[output_dir],
                trusted=trusted,
                allow_overwrite=allow_overwrite,
                allowed_suffixes={".mp4", ".m4v"} if target_format == "mp4" else {f".{target_format}"},
            )
        except UnsafeOutputPathError as exc:
            raise EncodePlanningError(str(exc)) from exc
        return str(candidate)
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    stem = "".join(ch for ch in project_name.replace(" ", "_") if ch.isalnum() or ch in {"_", "-"}) or "encode"
    base = directory / f"{stem}_encode.{target_format}"
    if not base.exists():
        return str(base)
    for index in range(2, 1000):
        candidate = directory / f"{stem}_encode_{index:03d}.{target_format}"
        if not candidate.exists():
            return str(candidate)
    raise EncodePlanningError("Could not find an unused encode output filename.")


def _validate_explicit_stream_copy(
    *,
    target_format: str,
    source_video_codec: str,
    source_audio_codec: str,
    has_audio: bool,
) -> None:
    if target_format != "mp4":
        return
    if source_video_codec not in VIDEO_DECODABLE_IN_MP4:
        raise EncodePlanningError(
            f"Cannot stream-copy {source_video_codec or 'unknown'} video into MP4 safely. "
            "Remove the copy-streams request so Vex can re-encode it."
        )
    if has_audio and source_audio_codec not in MP4_AUDIO_COPY_CODECS:
        raise EncodePlanningError(
            f"Cannot stream-copy {source_audio_codec or 'unknown'} audio into MP4 safely. "
            "Remove the copy-streams request so Vex can encode AAC audio."
        )


def _can_stream_copy(
    intent: EncodeIntent,
    *,
    target_format: str,
    source_video_codec: str,
    source_audio_codec: str,
    has_audio: bool,
) -> bool:
    if intent.copy_streams is False or intent.target_size_mb or intent.strip_audio:
        return False
    if intent.video_codec or intent.audio_codec or intent.max_width or intent.max_height or intent.fps:
        return False
    if intent.copy_streams is True:
        return True
    if re.search(r"\b(?:compress|smaller|reduce|shrink|target|under|below)\b", intent.raw_request.lower()):
        return False
    if target_format != "mp4":
        return False
    if source_video_codec not in VIDEO_DECODABLE_IN_MP4:
        return False
    return not has_audio or source_audio_codec in MP4_AUDIO_COPY_CODECS


def _video_filters(intent: EncodeIntent, metadata: dict[str, Any], warnings: list[str]) -> list[str]:
    source_width = int(metadata.get("width") or 0)
    source_height = int(metadata.get("height") or 0)
    if source_width <= 0 or source_height <= 0:
        return []
    target = _fit_within(
        source_width,
        source_height,
        intent.max_width,
        intent.max_height,
        allow_upscale=False,
    )
    if target is None:
        return []
    target_width, target_height = target
    if target_width == source_width and target_height == source_height:
        return []
    warnings.append(f"Scaling output to {target_width}x{target_height}.")
    return [f"scale={target_width}:{target_height}"]


def _fit_within(
    width: int,
    height: int,
    max_width: int | None,
    max_height: int | None,
    *,
    allow_upscale: bool,
) -> tuple[int, int] | None:
    if not max_width and not max_height:
        return None
    target_width = max_width or width
    target_height = max_height or height
    ratio = min(target_width / width, target_height / height)
    if ratio >= 1.0 and not allow_upscale:
        return None
    return _even_int(width * ratio), _even_int(height * ratio)


def _even_int(value: float) -> int:
    parsed = max(2, int(round(value)))
    return parsed if parsed % 2 == 0 else parsed - 1


def _fps_args(intent: EncodeIntent, metadata: dict[str, Any]) -> list[str]:
    if not intent.fps:
        return []
    source_fps = float(metadata.get("fps") or 0.0)
    if source_fps > 0 and intent.fps > source_fps:
        return []
    return ["-r", _format_number(intent.fps)]


def _stream_copy_command(
    input_path: str,
    output_path: str,
    *,
    target_format: str,
    has_audio: bool,
    source_audio_codec: str,
) -> list[str]:
    command = _base_command(input_path)
    command.extend(["-map", "0:v:0"])
    if has_audio:
        command.extend(["-map", "0:a:0?"])
    command.extend(["-c:v", "copy"])
    if has_audio:
        command.extend(["-c:a", "copy" if target_format != "mp4" or source_audio_codec in MP4_AUDIO_COPY_CODECS else "aac"])
    else:
        command.append("-an")
    if target_format == "mp4":
        command.extend(["-movflags", "+faststart"])
    command.extend(["-y", output_path])
    return command


def _crf_command(
    input_path: str,
    output_path: str,
    intent: EncodeIntent,
    metadata: dict[str, Any],
    video_encoder: str,
    filters: list[str],
    fps_args: list[str],
    *,
    has_audio: bool,
) -> list[str]:
    command = _base_command(input_path)
    command.extend(["-map", "0:v:0"])
    if has_audio and not intent.strip_audio:
        command.extend(["-map", "0:a:0?"])
    command.extend(fps_args)
    if filters:
        command.extend(["-vf", ",".join(filters)])
    command.extend(["-c:v", video_encoder])
    command.extend(_quality_args(intent, video_encoder))
    if _needs_pix_fmt(video_encoder):
        command.extend(["-pix_fmt", "yuv420p"])
    command.extend(_audio_args(intent, metadata, has_audio=has_audio))
    if _container_for_intent(intent) == "mp4":
        command.extend(["-movflags", "+faststart"])
    command.extend(["-y", output_path])
    return command


def _two_pass_commands(
    input_path: str,
    output_path: str,
    intent: EncodeIntent,
    metadata: dict[str, Any],
    video_encoder: str,
    filters: list[str],
    fps_args: list[str],
    *,
    has_audio: bool,
    warnings: list[str],
) -> tuple[list[list[str]], int, str]:
    duration = max(float(metadata.get("duration_sec") or 0.0), 0.001)
    target_bytes = int(float(intent.target_size_mb or 0.0) * 1024 * 1024)
    audio_bitrate = 0 if intent.strip_audio or not has_audio else _audio_bitrate_bits(intent)
    total_bitrate = int(target_bytes * 8 * 0.94 / duration)
    video_bitrate = max(total_bitrate - audio_bitrate, 150_000)
    if video_bitrate <= 350_000:
        warnings.append("The requested target size is very aggressive and may visibly reduce quality.")
    bitrate_arg = f"{max(1, round(video_bitrate / 1000))}k"
    passlog_file = str(Path(output_path).with_suffix("")) + "_ffmpeg2pass"

    first_pass = _base_command(input_path)
    first_pass.extend(["-map", "0:v:0"])
    first_pass.extend(fps_args)
    if filters:
        first_pass.extend(["-vf", ",".join(filters)])
    first_pass.extend(["-c:v", video_encoder, "-b:v", bitrate_arg, "-pass", "1", "-passlogfile", passlog_file])
    if video_encoder == "libx264":
        first_pass.extend(["-preset", _preset_for_intent(intent)])
    first_pass.extend(["-an", "-f", "null", "-y", NULL_OUTPUT])

    second_pass = _base_command(input_path)
    second_pass.extend(["-map", "0:v:0"])
    if has_audio and not intent.strip_audio:
        second_pass.extend(["-map", "0:a:0?"])
    second_pass.extend(fps_args)
    if filters:
        second_pass.extend(["-vf", ",".join(filters)])
    second_pass.extend(["-c:v", video_encoder, "-b:v", bitrate_arg, "-pass", "2", "-passlogfile", passlog_file])
    if video_encoder == "libx264":
        second_pass.extend(["-preset", _preset_for_intent(intent)])
        second_pass.extend(["-pix_fmt", "yuv420p"])
    second_pass.extend(_audio_args(intent, metadata, has_audio=has_audio))
    if _container_for_intent(intent) == "mp4":
        second_pass.extend(["-movflags", "+faststart"])
    second_pass.extend(["-y", output_path])
    return [first_pass, second_pass], target_bytes, passlog_file


def _base_command(input_path: str) -> list[str]:
    return [config.FFMPEG_PATH, "-hide_banner", "-i", str(input_path)]


def _quality_args(intent: EncodeIntent, video_encoder: str) -> list[str]:
    if video_encoder == "libx264":
        return ["-preset", _preset_for_intent(intent), "-crf", str(_crf_for_intent(intent))]
    if video_encoder == "libx265":
        return ["-preset", _preset_for_intent(intent), "-crf", str(_crf_for_intent(intent) + 2)]
    if video_encoder == "libaom-av1":
        return ["-crf", str(_crf_for_intent(intent) + 8), "-b:v", "0", "-cpu-used", "4"]
    if video_encoder == "libvpx-vp9":
        return ["-crf", str(_crf_for_intent(intent) + 8), "-b:v", "0"]
    if video_encoder == "prores_ks":
        return ["-profile:v", "3"]
    return []


def _codec_label_for_encoder(video_encoder: str) -> str:
    labels = {
        "libx264": "H.264",
        "libx265": "HEVC",
        "libaom-av1": "AV1",
        "libvpx-vp9": "VP9",
        "prores_ks": "ProRes",
    }
    return labels.get(video_encoder, video_encoder)


def _preset_for_intent(intent: EncodeIntent) -> str:
    if intent.optimize_for == "fastest":
        return "veryfast"
    if intent.quality == "max":
        return "veryslow"
    return "slow"


def _crf_for_intent(intent: EncodeIntent) -> int:
    if intent.quality == "max" or re.search(r"\b(?:best|maximum quality|visually lossless)\b", intent.raw_request.lower()):
        return 18
    if intent.quality == "high":
        return 20
    if intent.quality == "small" or intent.optimize_for == "smallest_size":
        return 26
    return 23


def _audio_args(intent: EncodeIntent, metadata: dict[str, Any], *, has_audio: bool) -> list[str]:
    if intent.strip_audio or intent.audio_codec == "none" or not has_audio:
        return ["-an"]
    if intent.audio_codec == "copy":
        return ["-c:a", "copy"]
    codec = intent.audio_codec or ("opus" if intent.target_format == "webm" else "aac")
    if codec == "mp3":
        return ["-c:a", "libmp3lame", "-b:a", _audio_bitrate_label(intent)]
    if codec == "opus":
        return ["-c:a", "libopus", "-b:a", "128k"]
    source_audio = str(metadata.get("audio_codec") or "").lower()
    if codec == "aac" and source_audio == "aac" and intent.copy_streams is True:
        return ["-c:a", "copy"]
    return ["-c:a", "aac", "-b:a", _audio_bitrate_label(intent)]


def _audio_bitrate_bits(intent: EncodeIntent) -> int:
    label = _audio_bitrate_label(intent)
    return int(label.removesuffix("k")) * 1000


def _audio_bitrate_label(intent: EncodeIntent) -> str:
    if intent.quality in {"max", "high"}:
        return "192k"
    if intent.quality == "small" or intent.optimize_for == "smallest_size":
        return "96k"
    return "128k"


def _needs_pix_fmt(video_encoder: str) -> bool:
    return video_encoder in {"libx264", "libx265"}


def _format_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.3f}".rstrip("0").rstrip(".")


def _display_command(commands: list[list[str]]) -> str:
    return " && ".join(shlex.join(command) for command in commands)


def _run_ffmpeg_command(
    command: list[str],
    *,
    duration: float,
    progress_callback: Callable[[float], None] | None,
) -> None:
    command_text = _display_command([command])
    try:
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    except OSError as exc:
        raise VideoEngineError(f"Failed to launch encode command: {exc}", command=command_text) from exc
    if process.stderr is None:
        raise VideoEngineError("Failed to capture encode progress.", command=command_text)
    stderr_tail: list[str] = []
    for line in process.stderr:
        stderr_tail.append(line)
        if len(stderr_tail) > 80:
            stderr_tail.pop(0)
        if duration > 0 and "time=" in line and progress_callback is not None:
            marker = line.split("time=", 1)[1].split()[0]
            try:
                seconds = parse_timestamp(marker)
            except ValueError:
                continue
            progress_callback(min(seconds / duration, 1.0))
    if process.wait() != 0:
        stderr_text = "".join(stderr_tail).strip()
        message = f"Encode failed: {stderr_text}" if stderr_text else "Encode failed."
        raise VideoEngineError(message, command=command_text)


def _cleanup_pass_logs(passlog_file: Any) -> None:
    if not passlog_file:
        return
    base = Path(str(passlog_file))
    candidates = [base, base.with_suffix(".log"), base.with_suffix(".log.mbtree")]
    parent = base.parent
    prefix = base.name
    if parent.is_dir():
        candidates.extend(parent.glob(prefix + "*"))
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        try:
            if path.is_file():
                path.unlink()
        except OSError:
            pass
