from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FastAction:
    tool_name: str
    params: dict[str, Any]
    reason: str


_NUMBER = r"\d+(?:\.\d+)?"
_UNIT = (
    r"milliseconds?|msecs?|ms|seconds?|secs?|sec|s|minutes?|mins?|min|m|"
    r"hours?|hrs?|hr|h"
)
_COLON_TIME = r"\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?"
_TIME_TOKEN = rf"(?:{_COLON_TIME}|{_NUMBER}\s*(?:{_UNIT})?)"
_MEDIA_EXTENSIONS = r"mp4|mov|mkv|webm|avi|m4v|mpg|mpeg"

_BLOCKED_DIRECT_TRIM_RE = re.compile(
    r"\b("
    r"silence|silent|pauses?|dead\s+air|ums?|uhs?|filler|"
    r"caption|captions|subtitle|subtitles|srt|transcript|transcribe|"
    r"b[-\s]?roll|stock|visual|visuals|animation|overlay|title|text|"
    r"speed|faster|slower|merge|combine|join|export|render|"
    r"shorts?|reels?|tiktoks?|highlights?|summarize|summary"
    r")\b",
    re.IGNORECASE,
)


def detect_fast_action(user_message: str, metadata: dict[str, Any] | None = None) -> FastAction | None:
    """Return a deterministic one-tool action for high-confidence simple edits."""

    text = _normalize_instruction(user_message)
    if not text or _BLOCKED_DIRECT_TRIM_RE.search(text):
        return None

    return (
        _detect_remove_head(text)
        or _detect_remove_before(text)
        or _detect_remove_tail(text)
        or _detect_explicit_range(text)
        or _detect_from_to_end(text)
        or _detect_keep_after(text)
        or _detect_keep_first(text)
        or _detect_last_window(text, metadata or {})
    )


def _normalize_instruction(user_message: str) -> str:
    text = user_message.strip().lower()
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    text = re.sub(rf'"[^"]*(?:[\\/]|\.({_MEDIA_EXTENSIONS})\b)[^"]*"', " ", text)
    text = re.sub(rf"'[^']*(?:[\\/]|\.({_MEDIA_EXTENSIONS})\b)[^']*'", " ", text)
    text = re.sub(r"[a-z]:\\\S+", " ", text)
    text = re.sub(rf"\S+\.({_MEDIA_EXTENSIONS})\b", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _detect_remove_head(text: str) -> FastAction | None:
    pattern = rf"\b(?:remove|drop|delete|cut\s+off|trim\s+off)\s+(?:the\s+)?(?:first|opening|initial)\s+({_TIME_TOKEN})\b"
    match = re.search(pattern, text)
    if not match:
        return None
    start = _parse_time_label(match.group(1))
    if start <= 0:
        return None
    return _trim_action(start=start, end=None, reason="remove leading duration")


def _detect_remove_before(text: str) -> FastAction | None:
    pattern = rf"\b(?:remove|drop|delete|cut\s+off|trim\s+off)\s+(?:everything\s+)?(?:before|up\s+to|until)\s+({_TIME_TOKEN})\b"
    match = re.search(pattern, text)
    if not match:
        return None
    start = _parse_time_label(match.group(1))
    if start <= 0:
        return None
    return _trim_action(start=start, end=None, reason="remove content before timestamp")


def _detect_remove_tail(text: str) -> FastAction | None:
    pattern = rf"\b(?:remove|drop|delete|cut\s+off|trim\s+off)\s+(?:everything\s+)?(?:after|past)\s+({_TIME_TOKEN})\b"
    match = re.search(pattern, text)
    if not match:
        return None
    end = _parse_time_label(match.group(1))
    if end <= 0:
        return None
    return _trim_action(start=0.0, end=end, reason="remove trailing content after timestamp")


def _detect_explicit_range(text: str) -> FastAction | None:
    verb = r"(?:trim|clip|keep|use|take|extract|select)"
    pattern = rf"\b{verb}\s+(?:the\s+)?(?:clip\s+)?(?:from\s+)?({_TIME_TOKEN})\s*(?:-|to|through|thru|until)\s*({_TIME_TOKEN})\b"
    match = re.search(pattern, text)
    if not match:
        return None
    start = _parse_time_label(match.group(1))
    end = _parse_time_label(match.group(2))
    if end <= start:
        return None
    return _trim_action(start=start, end=end, reason="explicit trim range")


def _detect_from_to_end(text: str) -> FastAction | None:
    verb = r"(?:trim|clip|keep|use|take|extract|select)"
    pattern = rf"\b{verb}\s+(?:the\s+)?(?:clip\s+)?from\s+({_TIME_TOKEN})\s+(?:to|through|thru|until)\s+(?:the\s+)?end\b"
    match = re.search(pattern, text)
    if not match:
        return None
    start = _parse_time_label(match.group(1))
    if start <= 0:
        return None
    return _trim_action(start=start, end=None, reason="trim from timestamp to end")


def _detect_keep_after(text: str) -> FastAction | None:
    from_pattern = rf"\b(?:trim|clip|keep|use|take|extract|select)\s+(?:the\s+)?(?:clip\s+)?from\s+({_TIME_TOKEN})(?:\s+(?:onwards?|forward|to\s+(?:the\s+)?end))?\b"
    after_pattern = rf"\b(?:keep|use|take|extract|select)\s+(?:everything\s+)?after\s+({_TIME_TOKEN})(?:\s+(?:onwards?|forward|to\s+(?:the\s+)?end))?\b"
    match = re.search(from_pattern, text) or re.search(after_pattern, text)
    if not match:
        return None
    start = _parse_time_label(match.group(1))
    if start <= 0:
        return None
    return _trim_action(start=start, end=None, reason="keep content after timestamp")


def _detect_keep_first(text: str) -> FastAction | None:
    first_pattern = rf"\b(?:trim|clip|keep|use|take|extract|select)\s+(?:the\s+)?(?:first|opening|initial)\s+({_TIME_TOKEN})\b"
    to_pattern = rf"\b(?:trim|clip|keep|use|take|extract|select)\s+(?:it\s+)?to\s+(?:the\s+)?(?:first\s+)?({_TIME_TOKEN})\b"
    match = re.search(first_pattern, text) or re.search(to_pattern, text)
    if not match:
        return None
    end = _parse_time_label(match.group(1))
    if end <= 0:
        return None
    return _trim_action(start=0.0, end=end, reason="keep first duration")


def _detect_last_window(text: str, metadata: dict[str, Any]) -> FastAction | None:
    duration = _metadata_duration(metadata)
    if duration is None:
        return None
    keep_pattern = rf"\b(?:trim|clip|keep|use|take|extract|select)\s+(?:the\s+)?(?:last|final)\s+({_TIME_TOKEN})\b"
    remove_pattern = rf"\b(?:remove|drop|delete|cut\s+off|trim\s+off)\s+(?:the\s+)?(?:last|final)\s+({_TIME_TOKEN})\b"
    keep_match = re.search(keep_pattern, text)
    if keep_match:
        window = _parse_time_label(keep_match.group(1))
        if window <= 0:
            return None
        return _trim_action(start=max(duration - window, 0.0), end=None, reason="keep final duration")
    remove_match = re.search(remove_pattern, text)
    if remove_match:
        window = _parse_time_label(remove_match.group(1))
        end = duration - window
        if window <= 0 or end <= 0:
            return None
        return _trim_action(start=0.0, end=end, reason="remove final duration")
    return None


def _metadata_duration(metadata: dict[str, Any]) -> float | None:
    try:
        duration = float(metadata.get("duration_sec") or 0.0)
    except (TypeError, ValueError):
        return None
    return duration if duration > 0 else None


def _trim_action(*, start: float, end: float | None, reason: str) -> FastAction:
    params: dict[str, Any] = {"start": _format_seconds(start)}
    if end is not None:
        params["end"] = _format_seconds(end)
    return FastAction(tool_name="trim_clip", params=params, reason=reason)


def _parse_time_label(label: str) -> float:
    raw = label.strip().lower()
    if ":" in raw:
        parts = [float(part) for part in raw.split(":")]
        if len(parts) == 2:
            minutes, seconds = parts
            return minutes * 60 + seconds
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return hours * 3600 + minutes * 60 + seconds
        raise ValueError(f"Invalid timestamp: {label!r}")
    match = re.fullmatch(rf"({_NUMBER})\s*({_UNIT})?", raw)
    if not match:
        raise ValueError(f"Invalid timestamp: {label!r}")
    value = float(match.group(1))
    unit = (match.group(2) or "s").strip()
    if unit in {"ms", "msec", "msecs"} or unit.startswith("millisecond"):
        return value / 1000.0
    if unit in {"m", "min", "mins"} or unit.startswith("minute"):
        return value * 60.0
    if unit in {"h", "hr", "hrs"} or unit.startswith("hour"):
        return value * 3600.0
    return value


def _format_seconds(value: float) -> str:
    rounded = round(float(value), 3)
    if rounded.is_integer():
        return str(int(rounded))
    return f"{rounded:.3f}".rstrip("0").rstrip(".")
