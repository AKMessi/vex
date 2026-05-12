from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import config
from engine import probe_video


CONTAINER_FORMATS = {
    "mp4": {"mov", "mp4", "m4a", "3gp", "3g2", "mj2"},
    "m4v": {"mov", "mp4", "m4a", "3gp", "3g2", "mj2"},
    "mov": {"mov", "mp4", "m4a", "3gp", "3g2", "mj2"},
    "mkv": {"matroska", "webm"},
    "webm": {"matroska", "webm"},
}
VIDEO_ENCODER_CODECS = {
    "libx264": "h264",
    "libx265": "hevc",
    "libaom-av1": "av1",
    "libvpx-vp9": "vp9",
    "prores_ks": "prores",
}
VIDEO_CODEC_ALIASES = {
    "h264": {"h264"},
    "hevc": {"hevc", "h265"},
    "av1": {"av1"},
    "vp9": {"vp9"},
    "prores": {"prores"},
}
AUDIO_CODEC_ALIASES = {
    "aac": {"aac"},
    "mp3": {"mp3"},
    "opus": {"opus"},
    "copy": set(),
}


@dataclass(frozen=True)
class EncodeValidationIssue:
    severity: str
    code: str
    message: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EncodeValidationReport:
    ok: bool
    issues: list[EncodeValidationIssue]
    input_metadata: dict[str, Any]
    output_metadata: dict[str, Any]
    output_size_bytes: int
    decode_checked: bool
    decode_command: list[str]

    @property
    def fatal_errors(self) -> list[str]:
        return [issue.message for issue in self.issues if issue.severity == "fatal"]

    @property
    def warnings(self) -> list[str]:
        return [issue.message for issue in self.issues if issue.severity == "warning"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "issues": [issue.to_dict() for issue in self.issues],
            "fatal_errors": self.fatal_errors,
            "warnings": self.warnings,
            "input_metadata": self.input_metadata,
            "output_metadata": self.output_metadata,
            "output_size_bytes": self.output_size_bytes,
            "decode_checked": self.decode_checked,
            "decode_command": self.decode_command,
            "format": self.output_metadata.get("format"),
            "codec": self.output_metadata.get("codec"),
            "profile": self.output_metadata.get("profile"),
            "pix_fmt": self.output_metadata.get("pix_fmt"),
            "audio_codec": self.output_metadata.get("audio_codec"),
            "has_audio": self.output_metadata.get("has_audio"),
            "width": self.output_metadata.get("width"),
            "height": self.output_metadata.get("height"),
            "fps": self.output_metadata.get("fps"),
        }


def validate_encode_output(plan: dict[str, Any]) -> EncodeValidationReport:
    issues: list[EncodeValidationIssue] = []
    input_metadata = dict(plan.get("source_metadata") or {})
    output_metadata: dict[str, Any] = {}
    raw_output_path = str(plan.get("output_path") or "").strip()
    output_size = 0

    if not raw_output_path:
        issues.append(_fatal("missing_output_path", "Encode plan did not include an output path."))
        return _report(issues, input_metadata, output_metadata, output_size, decode_checked=False, decode_command=[])
    output_path = Path(raw_output_path)
    if not output_path.is_file():
        issues.append(_fatal("missing_output", f"Encoded output was not created: {output_path}"))
        return _report(issues, input_metadata, output_metadata, output_size, decode_checked=False, decode_command=[])

    output_size = output_path.stat().st_size
    if output_size <= 0:
        issues.append(_fatal("empty_output", f"Encoded output is empty: {output_path}"))
        return _report(issues, input_metadata, output_metadata, output_size, decode_checked=False, decode_command=[])

    try:
        output_metadata = probe_video(str(output_path))
    except Exception as exc:  # noqa: BLE001
        issues.append(_fatal("probe_failed", f"ffprobe could not read encoded output: {exc}"))
        return _report(issues, input_metadata, output_metadata, output_size, decode_checked=False, decode_command=[])

    _validate_streams(plan, input_metadata, output_metadata, issues)
    _validate_container(plan, output_metadata, issues)
    _validate_video_codec(plan, input_metadata, output_metadata, issues)
    _validate_audio(plan, input_metadata, output_metadata, issues)
    _validate_dimensions(plan, input_metadata, output_metadata, issues)
    _validate_fps(plan, input_metadata, output_metadata, issues)
    _validate_duration(input_metadata, output_metadata, issues)
    _validate_size(plan, input_metadata, output_size, issues)
    decode_command = _decode_command(str(output_path))
    decode_checked = _validate_decode(decode_command, issues)

    return _report(
        issues,
        input_metadata,
        output_metadata,
        output_size,
        decode_checked=decode_checked,
        decode_command=decode_command,
    )


def format_validation_failure(report: EncodeValidationReport) -> str:
    if report.ok:
        return ""
    errors = report.fatal_errors
    if not errors:
        return "Encoded output failed validation."
    rendered = "; ".join(errors[:3])
    if len(errors) > 3:
        rendered += f"; plus {len(errors) - 3} more issue(s)"
    return f"Encoded output failed validation: {rendered}"


def _report(
    issues: list[EncodeValidationIssue],
    input_metadata: dict[str, Any],
    output_metadata: dict[str, Any],
    output_size: int,
    *,
    decode_checked: bool,
    decode_command: list[str],
) -> EncodeValidationReport:
    return EncodeValidationReport(
        ok=not any(issue.severity == "fatal" for issue in issues),
        issues=issues,
        input_metadata=input_metadata,
        output_metadata=output_metadata,
        output_size_bytes=output_size,
        decode_checked=decode_checked,
        decode_command=decode_command,
    )


def _fatal(code: str, message: str, **detail: Any) -> EncodeValidationIssue:
    return EncodeValidationIssue("fatal", code, message, detail)


def _warning(code: str, message: str, **detail: Any) -> EncodeValidationIssue:
    return EncodeValidationIssue("warning", code, message, detail)


def _info(code: str, message: str, **detail: Any) -> EncodeValidationIssue:
    return EncodeValidationIssue("info", code, message, detail)


def _intent(plan: dict[str, Any]) -> dict[str, Any]:
    intent = plan.get("intent")
    return intent if isinstance(intent, dict) else {}


def _format_tokens(format_name: Any) -> set[str]:
    return {token.strip().lower() for token in str(format_name or "").split(",") if token.strip()}


def _validate_streams(
    _plan: dict[str, Any],
    _input_metadata: dict[str, Any],
    output_metadata: dict[str, Any],
    issues: list[EncodeValidationIssue],
) -> None:
    if int(output_metadata.get("width") or 0) <= 0 or int(output_metadata.get("height") or 0) <= 0:
        issues.append(_fatal("missing_video_stream", "Encoded output has no readable video stream."))
    if float(output_metadata.get("duration_sec") or 0.0) <= 0.0:
        issues.append(_fatal("invalid_duration", "Encoded output has no positive duration."))


def _validate_container(plan: dict[str, Any], output_metadata: dict[str, Any], issues: list[EncodeValidationIssue]) -> None:
    expected = str(_intent(plan).get("target_format") or "mp4").lower()
    accepted = CONTAINER_FORMATS.get(expected)
    if not accepted:
        return
    observed = _format_tokens(output_metadata.get("format"))
    if not observed.intersection(accepted):
        issues.append(
            _fatal(
                "container_mismatch",
                f"Encoded output format {output_metadata.get('format')!r} does not match requested {expected}.",
                expected=expected,
                observed=output_metadata.get("format"),
            )
        )


def _validate_video_codec(
    plan: dict[str, Any],
    input_metadata: dict[str, Any],
    output_metadata: dict[str, Any],
    issues: list[EncodeValidationIssue],
) -> None:
    expected = _expected_video_codec(plan, input_metadata)
    observed = str(output_metadata.get("codec") or "").lower()
    if not expected or not observed:
        return
    accepted = VIDEO_CODEC_ALIASES.get(expected, {expected})
    if observed not in accepted:
        issues.append(
            _fatal(
                "video_codec_mismatch",
                f"Encoded output video codec {observed!r} does not match expected {expected!r}.",
                expected=expected,
                observed=observed,
            )
        )
    pix_fmt = str(output_metadata.get("pix_fmt") or "").lower()
    expected_format = str(_intent(plan).get("target_format") or "mp4").lower()
    if expected_format in {"mp4", "m4v"} and expected == "h264" and pix_fmt and pix_fmt != "yuv420p":
        issues.append(
            _warning(
                "mp4_pixel_format",
                f"MP4/H.264 output uses pixel format {pix_fmt!r}; yuv420p is safest for compatibility.",
                observed=pix_fmt,
            )
        )


def _expected_video_codec(plan: dict[str, Any], input_metadata: dict[str, Any]) -> str:
    commands = plan.get("commands") or []
    if not commands or not isinstance(commands[-1], list):
        return ""
    final_command = commands[-1]
    encoder = (
        _argument_after(final_command, "-c:v")
        or _argument_after(final_command, "-codec:v")
        or _argument_after(final_command, "-vcodec")
    )
    if not encoder:
        return ""
    if encoder == "copy":
        return str(input_metadata.get("codec") or "").lower()
    return VIDEO_ENCODER_CODECS.get(encoder, str(_intent(plan).get("video_codec") or "").lower())


def _validate_audio(
    plan: dict[str, Any],
    input_metadata: dict[str, Any],
    output_metadata: dict[str, Any],
    issues: list[EncodeValidationIssue],
) -> None:
    intent = _intent(plan)
    input_has_audio = bool(input_metadata.get("has_audio"))
    output_has_audio = bool(output_metadata.get("has_audio"))
    strip_audio = bool(intent.get("strip_audio")) or str(intent.get("audio_codec") or "").lower() == "none"
    if strip_audio:
        if output_has_audio:
            issues.append(_fatal("audio_not_stripped", "Audio was expected to be stripped, but output still has audio."))
        return
    if input_has_audio and not output_has_audio:
        issues.append(_fatal("audio_missing", "Source has audio, but encoded output has no audio stream."))
        return
    if not input_has_audio and output_has_audio:
        issues.append(_warning("unexpected_audio", "Source has no audio, but encoded output has an audio stream."))
        return
    if not input_has_audio:
        return
    expected = _expected_audio_codec(plan, input_metadata)
    observed = str(output_metadata.get("audio_codec") or "").lower()
    if expected and observed and observed not in AUDIO_CODEC_ALIASES.get(expected, {expected}):
        issues.append(
            _fatal(
                "audio_codec_mismatch",
                f"Encoded output audio codec {observed!r} does not match expected {expected!r}.",
                expected=expected,
                observed=observed,
            )
        )


def _expected_audio_codec(plan: dict[str, Any], input_metadata: dict[str, Any]) -> str:
    commands = plan.get("commands") or []
    if not commands or not isinstance(commands[-1], list):
        return ""
    final_command = commands[-1]
    if "-an" in final_command:
        return "none"
    codec = _argument_after(final_command, "-c:a") or _argument_after(final_command, "-acodec")
    if codec == "copy":
        return str(input_metadata.get("audio_codec") or "").lower()
    if codec == "libmp3lame":
        return "mp3"
    if codec == "libopus":
        return "opus"
    return codec or ""


def _validate_dimensions(
    plan: dict[str, Any],
    input_metadata: dict[str, Any],
    output_metadata: dict[str, Any],
    issues: list[EncodeValidationIssue],
) -> None:
    intent = _intent(plan)
    output_width = int(output_metadata.get("width") or 0)
    output_height = int(output_metadata.get("height") or 0)
    input_width = int(input_metadata.get("width") or 0)
    input_height = int(input_metadata.get("height") or 0)
    max_width = _positive_int(intent.get("max_width"))
    max_height = _positive_int(intent.get("max_height"))
    if max_width and output_width > max_width + 2:
        issues.append(_fatal("width_limit_missed", f"Output width {output_width} exceeds requested maximum {max_width}."))
    if max_height and output_height > max_height + 2:
        issues.append(_fatal("height_limit_missed", f"Output height {output_height} exceeds requested maximum {max_height}."))
    if not max_width and not max_height and input_width > 0 and input_height > 0:
        if (output_width, output_height) != (input_width, input_height):
            issues.append(
                _fatal(
                    "unexpected_dimensions",
                    f"Output dimensions {output_width}x{output_height} differ from source {input_width}x{input_height}.",
                    input_width=input_width,
                    input_height=input_height,
                    output_width=output_width,
                    output_height=output_height,
                )
            )


def _validate_fps(
    plan: dict[str, Any],
    input_metadata: dict[str, Any],
    output_metadata: dict[str, Any],
    issues: list[EncodeValidationIssue],
) -> None:
    requested_fps = _expected_fps(plan)
    output_fps = _positive_float(output_metadata.get("fps"))
    input_fps = _positive_float(input_metadata.get("fps"))
    if requested_fps and output_fps:
        if abs(output_fps - requested_fps) > max(0.5, requested_fps * 0.05):
            issues.append(
                _fatal(
                    "fps_mismatch",
                    f"Output FPS {output_fps:.3f} does not match requested {requested_fps:.3f}.",
                    requested=requested_fps,
                    observed=output_fps,
                )
            )
        return
    if input_fps and output_fps and abs(output_fps - input_fps) > max(1.0, input_fps * 0.10):
        issues.append(
            _warning(
                "unexpected_fps_drift",
                f"Output FPS {output_fps:.3f} differs from source FPS {input_fps:.3f}.",
                source=input_fps,
                observed=output_fps,
            )
        )


def _expected_fps(plan: dict[str, Any]) -> float | None:
    commands = plan.get("commands") or []
    if not commands or not isinstance(commands[-1], list):
        return None
    return _positive_float(_argument_after(commands[-1], "-r"))


def _validate_duration(
    input_metadata: dict[str, Any],
    output_metadata: dict[str, Any],
    issues: list[EncodeValidationIssue],
) -> None:
    source_duration = _positive_float(input_metadata.get("duration_sec"))
    output_duration = _positive_float(output_metadata.get("duration_sec"))
    if not source_duration or not output_duration:
        return
    tolerance = max(1.0, source_duration * 0.03)
    drift = abs(output_duration - source_duration)
    if drift > tolerance:
        issues.append(
            _fatal(
                "duration_mismatch",
                f"Output duration {output_duration:.2f}s differs from source {source_duration:.2f}s.",
                source=source_duration,
                observed=output_duration,
                tolerance=tolerance,
            )
        )


def _validate_size(
    plan: dict[str, Any],
    input_metadata: dict[str, Any],
    output_size: int,
    issues: list[EncodeValidationIssue],
) -> None:
    target_size = _positive_int(plan.get("estimated_size_bytes"))
    if target_size:
        upper = int(target_size * 1.12)
        lower = int(target_size * 0.70)
        if output_size > upper:
            issues.append(
                _warning(
                    "target_size_over",
                    f"Output size {output_size} bytes is more than 12% above the requested target.",
                    target_size_bytes=target_size,
                    output_size_bytes=output_size,
                )
            )
        elif output_size < lower:
            issues.append(
                _warning(
                    "target_size_under",
                    "Output is far below the requested target size; quality may be lower than necessary.",
                    target_size_bytes=target_size,
                    output_size_bytes=output_size,
                )
            )
    input_size = _positive_int(input_metadata.get("size_bytes"))
    duration = _positive_float(input_metadata.get("duration_sec"))
    if input_size and duration and duration >= 2.0 and output_size < max(4096, int(input_size * 0.002)):
        issues.append(
            _warning(
                "suspiciously_small_output",
                "Encoded output is extremely small compared with the source; inspect quality before publishing.",
                input_size_bytes=input_size,
                output_size_bytes=output_size,
            )
        )


def _decode_command(output_path: str) -> list[str]:
    return [
        config.FFMPEG_PATH,
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-xerror",
        "-i",
        output_path,
        "-map",
        "0",
        "-f",
        "null",
        "-",
    ]


def _validate_decode(command: list[str], issues: list[EncodeValidationIssue]) -> bool:
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=max(15, int(config.ENCODE_VALIDATION_TIMEOUT_SEC)),
        )
    except subprocess.TimeoutExpired:
        issues.append(
            _fatal(
                "decode_timeout",
                f"Decode validation timed out after {int(config.ENCODE_VALIDATION_TIMEOUT_SEC)} seconds.",
            )
        )
        return True
    except OSError as exc:
        issues.append(_fatal("decode_launch_failed", f"Could not launch FFmpeg decode validation: {exc}"))
        return False
    if result.returncode != 0:
        stderr = " ".join((result.stderr or "").split())
        if len(stderr) > 300:
            stderr = stderr[:300] + "..."
        issues.append(
            _fatal(
                "decode_failed",
                f"Encoded output failed full decode validation{': ' + stderr if stderr else '.'}",
                command=" ".join(command),
            )
        )
    else:
        issues.append(_info("decode_passed", "Encoded output passed full FFmpeg decode validation."))
    return True


def _argument_after(command: list[Any], flag: str) -> str:
    try:
        index = command.index(flag)
    except ValueError:
        return ""
    if index + 1 >= len(command):
        return ""
    return str(command[index + 1]).lower()


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None
