from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from video_generation.models import Beat, BeatGraph, ScriptPlan, TimedWord
from video_generation.script_planner import (
    estimated_script_duration,
    keyword_candidates,
    split_script_sentences,
)


BEAT_GRAPH_VERSION = "audio-first-beat-graph-v1"
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'%-]*")


def build_initial_beat_graph(
    plan: ScriptPlan,
    *,
    target_duration_sec: float,
    voice_speed: float,
) -> BeatGraph:
    duration = max(
        float(target_duration_sec or 0.0),
        estimated_script_duration(plan.narration, voice_speed=voice_speed),
        1.0,
    )
    words = estimate_timed_words(plan.narration, duration_sec=duration)
    return build_beat_graph_from_words(
        plan,
        timed_words=words,
        duration_sec=duration,
        source="estimated_words",
    )


def retime_beat_graph(
    plan: ScriptPlan,
    *,
    transcript_words: list[TimedWord],
    duration_sec: float,
    fallback: BeatGraph,
) -> BeatGraph:
    if not transcript_words:
        return BeatGraph(
            version=fallback.version,
            duration_sec=round(float(duration_sec or fallback.duration_sec), 3),
            source=fallback.source,
            beats=fallback.beats,
            words=fallback.words,
            warnings=[*fallback.warnings, "transcript had no parseable word timestamps"],
        )
    return build_beat_graph_from_words(
        plan,
        timed_words=transcript_words,
        duration_sec=max(float(duration_sec or 0.0), transcript_words[-1].end),
        source="hyperframes_transcript",
    )


def estimate_timed_words(script: str, *, duration_sec: float) -> list[TimedWord]:
    tokens = _word_tokens(script)
    if not tokens:
        return []
    weights = [max(len(token), 2) ** 0.72 for token in tokens]
    total_weight = sum(weights) or float(len(tokens))
    cursor = 0.0
    words: list[TimedWord] = []
    for token, weight in zip(tokens, weights):
        span = max(0.045, float(duration_sec) * weight / total_weight)
        start = cursor
        end = min(float(duration_sec), start + span)
        words.append(TimedWord(text=token, start=round(start, 3), end=round(end, 3)))
        cursor = end
    if words:
        words[-1] = TimedWord(
            text=words[-1].text,
            start=words[-1].start,
            end=round(float(duration_sec), 3),
            confidence=words[-1].confidence,
        )
    return words


def build_beat_graph_from_words(
    plan: ScriptPlan,
    *,
    timed_words: list[TimedWord],
    duration_sec: float,
    source: str,
) -> BeatGraph:
    sentences = split_script_sentences(plan.narration)
    if not sentences:
        sentences = [plan.narration]
    sentence_counts = [max(len(_word_tokens(sentence)), 1) for sentence in sentences]
    total_script_words = sum(sentence_counts)
    total_timed_words = len(timed_words)
    beats: list[Beat] = []
    cursor = 0
    for index, sentence in enumerate(sentences, start=1):
        share = sentence_counts[index - 1] / max(total_script_words, 1)
        take = max(1, round(share * total_timed_words)) if total_timed_words else 0
        if index == len(sentences):
            take = max(0, total_timed_words - cursor)
        segment_words = timed_words[cursor : cursor + take]
        cursor += take
        if segment_words:
            start = segment_words[0].start
            end = segment_words[-1].end
        else:
            start = (index - 1) * (float(duration_sec) / len(sentences))
            end = index * (float(duration_sec) / len(sentences))
        if end <= start:
            end = start + max(0.6, float(duration_sec) / max(len(sentences), 1))
        scene_type = classify_scene_type(sentence, index=index)
        beats.append(
            Beat(
                beat_id=f"beat_{index:02d}",
                index=index,
                start=round(max(start, 0.0), 3),
                end=round(min(max(end, start + 0.25), float(duration_sec)), 3),
                title=_beat_title(sentence, index=index),
                narration=sentence,
                caption=_caption_text(sentence),
                scene_type=scene_type,
                keywords=keyword_candidates(sentence, limit=5),
                visual_metaphor=_visual_metaphor(scene_type, sentence),
            )
        )
    beats = _repair_beat_boundaries(beats, duration_sec=float(duration_sec))
    return BeatGraph(
        version=BEAT_GRAPH_VERSION,
        duration_sec=round(float(duration_sec), 3),
        source=source,
        beats=beats,
        words=timed_words,
    )


def load_transcript_words(path: str | Path) -> list[TimedWord]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return parse_transcript_words(payload)


def parse_transcript_words(payload: Any) -> list[TimedWord]:
    words: list[TimedWord] = []
    for item in _iter_word_payloads(payload):
        text = str(item.get("word") or item.get("text") or item.get("token") or "").strip()
        if not text:
            continue
        try:
            start = float(item.get("start") if item.get("start") is not None else item.get("start_sec"))
            end = float(item.get("end") if item.get("end") is not None else item.get("end_sec"))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        confidence = item.get("confidence") or item.get("probability")
        try:
            confidence_value = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            confidence_value = None
        words.append(
            TimedWord(
                text=text,
                start=round(max(start, 0.0), 3),
                end=round(max(end, start), 3),
                confidence=confidence_value,
            )
        )
    words.sort(key=lambda item: (item.start, item.end))
    return words


def classify_scene_type(sentence: str, *, index: int) -> str:
    text = sentence.lower()
    if index == 1 or re.search(r"\b(?:why|what if|imagine|secret|problem|mistake)\b", text):
        return "hook"
    if re.search(r"\b\d+(?:\.\d+)?\s*(?:%|x|ms|seconds?|tokens?|users?|gb|mb|billion|million)\b", text):
        return "metric"
    if re.search(r"\b(?:but|instead|versus|vs|before|after|trade[- ]?off|however)\b", text):
        return "contrast"
    if re.search(r"\b(?:step|then|trace|flow|pipeline|loop|mechanism|system|route|process)\b", text):
        return "process"
    if re.search(r"\b(?:proof|because|therefore|result|takeaway|means|so)\b", text):
        return "proof"
    return "concept"


def _iter_word_payloads(payload: Any):
    if isinstance(payload, dict):
        for key in ("words", "tokens"):
            value = payload.get(key)
            if isinstance(value, list):
                yield from (item for item in value if isinstance(item, dict))
        for key in ("segments", "chunks", "transcript"):
            value = payload.get(key)
            if isinstance(value, list):
                for segment in value:
                    if isinstance(segment, dict):
                        segment_words = segment.get("words") or segment.get("tokens")
                        if isinstance(segment_words, list):
                            yield from (item for item in segment_words if isinstance(item, dict))
            elif isinstance(value, dict):
                yield from _iter_word_payloads(value)
    elif isinstance(payload, list):
        yield from (item for item in payload if isinstance(item, dict))


def _repair_beat_boundaries(beats: list[Beat], *, duration_sec: float) -> list[Beat]:
    if not beats:
        return []
    repaired: list[Beat] = []
    cursor = 0.0
    for index, beat in enumerate(beats):
        start = max(beat.start, cursor)
        end = max(beat.end, start + 0.25)
        if index == len(beats) - 1:
            end = max(end, duration_sec)
        end = min(end, duration_sec)
        repaired.append(
            Beat(
                beat_id=beat.beat_id,
                index=beat.index,
                start=round(start, 3),
                end=round(end, 3),
                title=beat.title,
                narration=beat.narration,
                caption=beat.caption,
                scene_type=beat.scene_type,
                keywords=beat.keywords,
                visual_metaphor=beat.visual_metaphor,
            )
        )
        cursor = end
    return repaired


def _beat_title(sentence: str, *, index: int) -> str:
    keywords = keyword_candidates(sentence, limit=4)
    if keywords:
        return " ".join(word.capitalize() for word in keywords)
    return f"Beat {index}"


def _caption_text(sentence: str) -> str:
    cleaned = re.sub(r"\s+", " ", sentence).strip()
    return cleaned[:150].rstrip(" ,.;:") if len(cleaned) > 150 else cleaned


def _visual_metaphor(scene_type: str, sentence: str) -> str:
    if scene_type == "hook":
        return "kinetic title lockup with a fast signal sweep"
    if scene_type == "metric":
        return "large numeric readout with calibrated bars and ticks"
    if scene_type == "contrast":
        return "split-screen comparison with a decisive center hinge"
    if scene_type == "process":
        return "moving route map where each node activates in sequence"
    if scene_type == "proof":
        return "stacked evidence layers resolving into one takeaway"
    keywords = keyword_candidates(sentence, limit=3)
    label = ", ".join(keywords) if keywords else "the idea"
    return f"abstract concept field for {label}"


def _word_tokens(text: str) -> list[str]:
    return _WORD_RE.findall(str(text or ""))
