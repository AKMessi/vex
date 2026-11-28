"""Timed text on the production frame clock."""
from __future__ import annotations

from fractions import Fraction
from html import escape, unescape
import re

from video_generation.production_timeline import _frame


MAX_SUBTITLE_BYTES = 512 * 1024
MAX_CUES = 4000
_SRT_TIME = r"[0-9]{2,8}:[0-5][0-9]:[0-5][0-9],[0-9]{3}"
_VTT_TIME = r"(?:[0-9]{2,8}:)?[0-5][0-9]:[0-5][0-9]\.[0-9]{3}"
_CUE_TAG = re.compile(
    r"</?(?:b|i|u|s|font|c|v|lang|ruby|rt)(?:[. \t][^<>]*)?>|<\d[\d:.]*>",
    re.IGNORECASE,
)


def _milliseconds(value: str) -> int:
    *hours, minutes, seconds = value.replace(",", ".").split(":")
    whole, millis = seconds.split(".")
    return 1000 * ((int(hours[0]) * 3600 if hours else 0) + int(minutes) * 60 + int(whole)) + int(millis)


def parse_captions(text: object, kind: object, *, fps: int, duration_frames: int) -> list[dict]:
    """Parse the subtitle formats into plain-text, non-overlapping production cues."""
    if kind not in ("srt", "vtt") or not isinstance(text, str):
        raise ValueError("Choose a UTF-8 .srt or .vtt subtitle file.")
    if len(text.encode("utf-8")) > MAX_SUBTITLE_BYTES or "\0" in text:
        raise ValueError("Choose a valid UTF-8 subtitle file up to 512 KiB.")
    text = text.removeprefix("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n[ \t]*\n(?:[ \t]*\n)*", text.strip("\n"))
    if kind == "vtt":
        header = blocks.pop(0)
        if "-->" in header or not re.fullmatch(r"WEBVTT(?:[ \t].*)?", header.split("\n", 1)[0]):
            raise ValueError("A WebVTT header must start with WEBVTT and end before the first cue.")
    pattern = _SRT_TIME if kind == "srt" else _VTT_TIME
    settings = r"(?:[ \t]+[^\n]+)?" if kind == "vtt" else ""
    timing = re.compile(rf"({pattern})[ \t]+-->[ \t]+({pattern}){settings}")
    cues = []
    previous_end = 0
    for block in blocks:
        lines = block.strip().split("\n")
        if kind == "vtt" and re.fullmatch(r"(?:NOTE(?:[ \t].*)?|STYLE|REGION)", lines[0]):
            continue
        if "-->" not in lines[0] and len(lines) > 1:
            lines = lines[1:]  # Cue identifier.
        match = timing.fullmatch(lines[0].strip())
        if not match:
            raise ValueError(f"Subtitle cue {len(cues) + 1} has an invalid timestamp.")
        start_ms, end_ms = map(_milliseconds, match.groups())
        start, end = (_frame(Fraction(value, 1000), fps) for value in (start_ms, end_ms))
        if end_ms <= start_ms or not previous_end <= start < end <= duration_frames:
            raise ValueError(
                f"Subtitle cue {len(cues) + 1} overlaps, falls outside the cut, or is shorter than a frame."
            )
        payload = "\n".join(unescape(_CUE_TAG.sub("", line)).strip() for line in lines[1:]).strip()
        if not payload:
            raise ValueError(f"Subtitle cue {len(cues) + 1} has no text.")
        cues.append({"start_frame": start, "duration_frames": end - start, "text": payload})
        if len(cues) > MAX_CUES:
            raise ValueError(f"Import up to {MAX_CUES:,} subtitle cues at a time.")
        previous_end = end
    if not cues:
        raise ValueError("The subtitle file contains no cues.")
    return cues


def _timestamp(frame: int, fps: int, separator: str) -> str:
    millis = (frame * 2000 + fps) // (2 * fps)
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    seconds, millis = divmod(millis, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02}{separator}{millis:03}"


def serialize_captions(cues: list[dict], fps: int, *, kind: str) -> str:
    """Serialize cues using the nearest millisecond to each frame boundary."""
    vtt = kind == "vtt"
    separator = "." if vtt else ","
    blocks = ["WEBVTT"] if vtt else []
    for index, cue in enumerate(cues, 1):
        start = _timestamp(cue["start_frame"], fps, separator)
        end = _timestamp(cue["start_frame"] + cue["duration_frames"], fps, separator)
        text = "\n".join(line.strip() for line in cue["text"].splitlines() if line.strip())
        # Escape cue markup and arrow tokens in both plain-text interchange formats.
        blocks.append(f"{index}\n{start} --> {end}\n{escape(text, quote=False)}")
    return "\n\n".join(blocks) + "\n"
