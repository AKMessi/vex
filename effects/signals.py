from __future__ import annotations

import math
import re
from typing import Any

from broll_intelligence import semantic_keywords
from tools.transcript_utils import optimize_caption_segments


HOOK_TERMS = {
    "why",
    "how",
    "secret",
    "mistake",
    "problem",
    "watch",
    "look",
    "listen",
    "wait",
    "actually",
}

PAYOFF_TERMS = {
    "because",
    "therefore",
    "so",
    "means",
    "result",
    "turns",
    "works",
    "finally",
    "exactly",
}

CONTRAST_TERMS = {
    "but",
    "however",
    "instead",
    "although",
    "versus",
    "vs",
    "before",
    "after",
    "wrong",
    "right",
}

EMPHASIS_TERMS = {
    "important",
    "critical",
    "huge",
    "massive",
    "insane",
    "crazy",
    "breaks",
    "broken",
    "never",
    "always",
    "only",
    "best",
    "worst",
}

PROCESS_TERMS = {
    "step",
    "first",
    "second",
    "third",
    "next",
    "then",
    "flow",
    "process",
    "system",
    "pipeline",
}

FOCUS_TERMS = {
    "focus",
    "attention",
    "notice",
    "see",
    "watch",
    "look",
}

CONCLUSION_TERMS = {
    "therefore",
    "finally",
    "ultimately",
    "conclusion",
    "takeaway",
    "point",
}


def build_subtitle_cards(
    subtitle_segments: list[dict[str, Any]],
    sentence_segments: list[dict[str, Any]],
    clip_duration: float,
    *,
    words: list[dict[str, Any]] | None = None,
    scene_cuts: list[float] | None = None,
) -> list[dict[str, Any]]:
    source_segments = _subtitle_source(subtitle_segments, sentence_segments)
    scene_cuts = scene_cuts or []
    words = words or []
    cards: list[dict[str, Any]] = []
    for index, segment in enumerate(source_segments, start=1):
        start_sec = max(0.0, min(_as_float(segment.get("start"), 0.0), clip_duration))
        end_sec = max(start_sec + 0.12, min(_as_float(segment.get("end"), start_sec + 0.8), clip_duration))
        text = _clean_text(segment.get("text"))
        if not text:
            continue
        previous_text = _clean_text(source_segments[index - 2].get("text")) if index > 1 else ""
        next_text = _clean_text(source_segments[index].get("text")) if index < len(source_segments) else ""
        pause_before = 0.0
        pause_after = 0.0
        if index > 1:
            pause_before = max(0.0, start_sec - _as_float(source_segments[index - 2].get("end"), start_sec))
        if index < len(source_segments):
            pause_after = max(0.0, _as_float(source_segments[index].get("start"), end_sec) - end_sec)
        card_words = _words_for_range(words, start_sec, end_sec)
        tokens = _tokens(text)
        context = _clean_text(f"{previous_text} {text} {next_text}")
        nearest_cut, scene_distance = _nearest_scene_distance(start_sec, end_sec, scene_cuts)
        signals = {
            "question": "?" in text,
            "exclamation": "!" in text,
            "numeric_hits": len(re.findall(r"\b\d+(?:\.\d+)?(?:%|x)?\b", text)),
            "hook_hits": _marker_hits(tokens, HOOK_TERMS),
            "payoff_hits": _marker_hits(tokens, PAYOFF_TERMS),
            "contrast_hits": _marker_hits(tokens, CONTRAST_TERMS),
            "emphasis_hits": _marker_hits(tokens, EMPHASIS_TERMS),
            "process_hits": _marker_hits(tokens, PROCESS_TERMS),
            "focus_hits": _marker_hits(tokens, FOCUS_TERMS),
            "conclusion_hits": _marker_hits(tokens, CONCLUSION_TERMS),
            "pause_before": round(pause_before, 3),
            "pause_after": round(pause_after, 3),
            "word_count": len(card_words) if card_words else len(tokens),
            "words_per_second": round((len(card_words) if card_words else len(tokens)) / max(end_sec - start_sec, 0.2), 2),
            "scene_distance": round(scene_distance, 3),
            "near_scene_cut": scene_distance <= 0.45,
            "is_opening": start_sec <= 5.0,
            "is_short": end_sec - start_sec <= 0.85,
        }
        priority = _subtitle_priority(signals, text, context)
        cards.append(
            {
                "card_id": f"subtitle_card_{index:03d}",
                "start": round(start_sec, 3),
                "end": round(end_sec, 3),
                "text": text,
                "previous_text": previous_text,
                "next_text": next_text,
                "keywords": semantic_keywords(context, limit=8),
                "nearest_scene_cut": round(nearest_cut, 3) if nearest_cut is not None else None,
                "scene_distance": round(scene_distance, 3),
                "pause_before": round(pause_before, 3),
                "pause_after": round(pause_after, 3),
                "signals": signals,
                "priority": priority,
            }
        )
    return cards


def _subtitle_source(
    subtitle_segments: list[dict[str, Any]],
    sentence_segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    raw = subtitle_segments if subtitle_segments else sentence_segments
    normalized = [
        {"start": _as_float(item.get("start"), 0.0), "end": _as_float(item.get("end"), 0.0), "text": _clean_text(item.get("text"))}
        for item in raw
        if isinstance(item, dict)
    ]
    normalized = [item for item in normalized if item["text"] and item["end"] > item["start"]]
    if not normalized:
        return []
    should_optimize = any(item["end"] - item["start"] > 2.6 or len(item["text"]) > 74 for item in normalized)
    if should_optimize:
        return optimize_caption_segments(
            normalized,
            max_chars_per_line=26,
            max_lines=2,
            max_words_per_caption=9,
            max_duration_sec=2.4,
        )
    return normalized


def _subtitle_priority(signals: dict[str, Any], text: str, context: str) -> float:
    score = 18.0
    score += 18.0 if signals["is_opening"] and (signals["hook_hits"] or signals["question"]) else 0.0
    score += 11.0 if signals["question"] else 0.0
    score += 8.0 if signals["exclamation"] else 0.0
    score += min(int(signals["numeric_hits"]), 3) * 8.5
    score += min(int(signals["hook_hits"]), 3) * 7.0
    score += min(int(signals["payoff_hits"]), 3) * 6.4
    score += min(int(signals["contrast_hits"]), 3) * 7.2
    score += min(int(signals["emphasis_hits"]), 4) * 5.4
    score += min(int(signals["process_hits"]), 3) * 3.8
    score += 5.0 if float(signals["pause_after"]) >= 0.35 else 0.0
    score += 3.5 if float(signals["pause_before"]) >= 0.28 else 0.0
    score += 4.0 if signals["near_scene_cut"] else 0.0
    score += _specificity_bonus(context)
    if len(text.split()) <= 2 and not signals["numeric_hits"]:
        score -= 8.0
    if float(signals["words_per_second"]) > 5.2 and not signals["pause_after"]:
        score -= 7.0
    return round(max(0.0, min(score, 100.0)), 2)


def _specificity_bonus(text: str) -> float:
    tokens = _tokens(text)
    if not tokens:
        return 0.0
    unique = len(set(tokens))
    long_words = sum(1 for token in tokens if len(token) >= 7)
    return min(10.0, unique * 0.45 + long_words * 0.8)


def _nearest_scene_distance(start_sec: float, end_sec: float, scene_cuts: list[float]) -> tuple[float | None, float]:
    if not scene_cuts:
        return None, 999.0
    nearest = min(scene_cuts, key=lambda cut: min(abs(cut - start_sec), abs(cut - end_sec)))
    return nearest, min(abs(nearest - start_sec), abs(nearest - end_sec))


def _words_for_range(words: list[dict[str, Any]], start_sec: float, end_sec: float) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for word in words:
        word_start = _as_float(word.get("start"), -1.0)
        word_end = _as_float(word.get("end"), -1.0)
        if word_end >= start_sec - 0.02 and word_start <= end_sec + 0.02:
            selected.append(word)
    return selected


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9']+", str(text or "").lower())


def _marker_hits(tokens: list[str], markers: set[str]) -> int:
    token_set = set(tokens)
    return len(token_set & markers)


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _as_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number
