from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

from engine import parse_timestamp

TRANSCRIPT_ARTIFACT_NAMES = {
    "transcript.txt",
    "transcript.srt",
    "transcript.segments.json",
    "transcript.words.json",
    "transcript.sentences.json",
}


def _is_within(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath(
            [
                os.path.normcase(os.path.abspath(str(path))),
                os.path.normcase(os.path.abspath(str(root))),
            ]
        ) == os.path.normcase(os.path.abspath(str(root)))
    except ValueError:
        return False


def transcript_artifact_path(
    working_dir: str | Path,
    filename: str,
    *,
    for_write: bool = False,
) -> Path | None:
    if filename not in TRANSCRIPT_ARTIFACT_NAMES:
        raise ValueError(f"Unsupported transcript artifact name: {filename}")
    root = Path(working_dir).expanduser().resolve(strict=False)
    candidate = root / filename
    if for_write:
        root.mkdir(parents=True, exist_ok=True)
        if candidate.is_symlink():
            candidate.unlink()
        resolved = candidate.resolve(strict=False)
        if not _is_within(resolved, root):
            raise RuntimeError(f"Transcript artifact path escaped the project directory: {candidate}")
        return candidate
    if not candidate.is_file():
        return None
    try:
        resolved = candidate.resolve(strict=True)
    except OSError:
        return None
    if not _is_within(resolved, root):
        return None
    return candidate


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


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


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


def clean_transcript_text(text: str) -> str:
    collapsed = re.sub(r"\s+", " ", text).strip()
    collapsed = re.sub(r"\s+([,.;:!?])", r"\1", collapsed)
    return collapsed


def build_sentence_segments(
    words: list[dict[str, float | str]],
    fallback_segments: list[dict[str, float | str]] | None = None,
    max_words_per_sentence: int = 18,
    max_duration_sec: float = 4.5,
) -> list[dict[str, float | str]]:
    sentences: list[dict[str, float | str]] = []
    if words:
        current: list[dict[str, float | str]] = []
        for word in words:
            if word.get("start") is None or word.get("end") is None:
                continue
            current.append(word)
            word_text = str(word.get("text") or "").strip()
            current_duration = float(current[-1]["end"]) - float(current[0]["start"])
            boundary = (
                word_text.endswith((".", "?", "!", ";", ":"))
                or len(current) >= max_words_per_sentence
                or current_duration >= max_duration_sec
            )
            if not boundary:
                continue
            sentence_text = clean_transcript_text(" ".join(str(item.get("text") or "").strip() for item in current))
            if sentence_text:
                sentences.append(
                    {
                        "index": len(sentences) + 1,
                        "start": round(float(current[0]["start"]), 3),
                        "end": round(float(current[-1]["end"]), 3),
                        "text": sentence_text,
                        "word_start_index": int(current[0].get("index", len(sentences) + 1)),
                        "word_end_index": int(current[-1].get("index", len(current))),
                    }
                )
            current = []
        if current:
            sentence_text = clean_transcript_text(" ".join(str(item.get("text") or "").strip() for item in current))
            if sentence_text:
                sentences.append(
                    {
                        "index": len(sentences) + 1,
                        "start": round(float(current[0]["start"]), 3),
                        "end": round(float(current[-1]["end"]), 3),
                        "text": sentence_text,
                        "word_start_index": int(current[0].get("index", len(sentences) + 1)),
                        "word_end_index": int(current[-1].get("index", len(current))),
                    }
                )
        return sentences

    for index, segment in enumerate(fallback_segments or [], start=1):
        text = clean_transcript_text(str(segment.get("text") or ""))
        if not text:
            continue
        sentences.append(
            {
                "index": index,
                "start": round(float(segment["start"]), 3),
                "end": round(float(segment["end"]), 3),
                "text": text,
            }
        )
    return sentences


def load_transcript_bundle(working_dir: str | Path) -> dict[str, object]:
    root = Path(working_dir)
    segment_path = transcript_artifact_path(root, "transcript.segments.json")
    word_path = transcript_artifact_path(root, "transcript.words.json")
    sentence_path = transcript_artifact_path(root, "transcript.sentences.json")
    txt_path = transcript_artifact_path(root, "transcript.txt")
    srt_path = transcript_artifact_path(root, "transcript.srt")

    segments = load_json(segment_path) if segment_path is not None else parse_srt(srt_path) if srt_path is not None else []
    words = load_json(word_path) if word_path is not None else []
    sentences = (
        load_json(sentence_path)
        if sentence_path is not None
        else build_sentence_segments(
            words if isinstance(words, list) else [],
            fallback_segments=segments if isinstance(segments, list) else [],
        )
    )
    transcript_text = txt_path.read_text(encoding="utf-8").strip() if txt_path is not None else ""
    if not transcript_text and isinstance(segments, list):
        transcript_text = clean_transcript_text(
            " ".join(
                str(segment.get("text") or "").strip()
                for segment in segments
                if isinstance(segment, dict) and str(segment.get("text") or "").strip()
            )
        )
    source = "missing"
    if segment_path is not None:
        source = "audio_transcription"
    elif srt_path is not None:
        source = "existing_srt"
    elif txt_path is not None:
        source = "imported_transcript"
    usable_timed_segments = len(segments) if isinstance(segments, list) else 0
    return {
        "transcript_text": transcript_text,
        "segments": segments if isinstance(segments, list) else [],
        "words": words if isinstance(words, list) else [],
        "sentences": sentences if isinstance(sentences, list) else [],
        "source": source,
        "usable_timed_segments": usable_timed_segments,
        "has_timing": usable_timed_segments > 0,
        "paths": {
            "txt": str(root / "transcript.txt"),
            "srt": str(root / "transcript.srt"),
            "segments": str(root / "transcript.segments.json"),
            "words": str(root / "transcript.words.json"),
            "sentences": str(root / "transcript.sentences.json"),
        },
    }


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
