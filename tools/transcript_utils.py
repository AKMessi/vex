from __future__ import annotations

import math
import re
from pathlib import Path

from engine import parse_timestamp


def format_srt_timestamp(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def parse_srt(path: Path) -> list[dict[str, float | str]]:
    raw_text = path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return []
    blocks = re.split(r"\r?\n\r?\n", raw_text)
    segments: list[dict[str, float | str]] = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 3:
            continue
        timestamp_line = next((line for line in lines if "-->" in line), "")
        if not timestamp_line:
            continue
        start_raw, end_raw = [part.strip().replace(",", ".") for part in timestamp_line.split("-->", 1)]
        start_sec = parse_timestamp(start_raw)
        end_sec = parse_timestamp(end_raw)
        text_start = lines.index(timestamp_line) + 1
        text = " ".join(lines[text_start:]).strip()
        if text and end_sec > start_sec:
            segments.append({"start": start_sec, "end": end_sec, "text": text})
    return segments


def write_srt_segments(path: Path, segments: list[dict[str, float | str]]) -> None:
    srt_lines: list[str] = []
    for index, segment in enumerate(segments, start=1):
        srt_lines.extend(
            [
                str(index),
                (
                    f"{format_srt_timestamp(float(segment['start']))} --> "
                    f"{format_srt_timestamp(float(segment['end']))}"
                ),
                str(segment["text"]).strip(),
                "",
            ]
        )
    path.write_text("\n".join(srt_lines), encoding="utf-8")


def _wrap_caption_words(words: list[str], max_chars_per_line: int, max_lines: int) -> str:
    lines: list[str] = []
    current: list[str] = []
    remaining_words = list(words)
    while remaining_words and len(lines) < max_lines:
        word = remaining_words.pop(0)
        candidate = " ".join(current + [word]).strip()
        if current and len(candidate) > max_chars_per_line:
            lines.append(" ".join(current))
            current = [word]
            continue
        current.append(word)
    if current:
        if len(lines) < max_lines:
            lines.append(" ".join(current))
        elif lines:
            lines[-1] = f"{lines[-1]} {' '.join(current)}".strip()
    if remaining_words and lines:
        lines[-1] = f"{lines[-1]} {' '.join(remaining_words)}".strip()
    return "\n".join(line.strip() for line in lines if line.strip())


def optimize_caption_segments(
    segments: list[dict[str, float | str]],
    max_chars_per_line: int = 18,
    max_lines: int = 2,
    max_words_per_caption: int = 6,
    max_duration_sec: float = 2.4,
) -> list[dict[str, float | str]]:
    optimized: list[dict[str, float | str]] = []
    for segment in segments:
        start_sec = float(segment["start"])
        end_sec = float(segment["end"])
        text = re.sub(r"\s+", " ", str(segment["text"]).strip())
        words = [word for word in text.split(" ") if word]
        if not words or end_sec <= start_sec:
            continue
        duration = end_sec - start_sec
        caption_count = max(
            1,
            math.ceil(len(text) / float(max_chars_per_line * max_lines)),
            math.ceil(len(words) / float(max_words_per_caption)),
            math.ceil(duration / float(max_duration_sec)),
        )
        caption_count = min(caption_count, len(words))
        for index in range(caption_count):
            word_start = int(len(words) * index / caption_count)
            if index == caption_count - 1:
                word_end = len(words)
            else:
                word_end = int(len(words) * (index + 1) / caption_count)
            caption_words = words[word_start:word_end]
            if not caption_words:
                continue
            piece_start = start_sec + duration * (index / caption_count)
            piece_end = start_sec + duration * ((index + 1) / caption_count)
            optimized.append(
                {
                    "start": round(piece_start, 3),
                    "end": round(piece_end, 3),
                    "text": _wrap_caption_words(caption_words, max_chars_per_line, max_lines),
                }
            )
    return [segment for segment in optimized if str(segment["text"]).strip()]


