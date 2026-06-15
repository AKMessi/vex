from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from pathlib import Path

from google import genai

import config
from tools.creative_intelligence import (
    build_video_understanding_graph,
    candidate_graph_signals,
    graph_to_video_context,
)
from tools.creative_qa import evaluate_short_candidate_quality
from tools.creative_registry import record_creative_run
from engine import apply_center_punch_ins, VideoEngineError, merge, probe_video, render_vertical_short, trim
from shorts import (
    build_semantic_units,
    build_shorts_program,
    build_story_chapters,
    compile_story_proposal,
    evaluate_story_candidate,
    format_units_for_planner,
    validate_short_edit_plan,
    validate_short_render,
    validate_shorts_program,
)
from state import ProjectState, utc_now_iso
from subtitles import resolve_subtitle_style
from tools.transcript import execute as transcribe
from tools.transcript_utils import (
    load_transcript_bundle,
    optimize_caption_segments,
    parse_srt,
    transcript_artifact_path,
    write_srt_segments,
)

VIRAL_TERMS = {
    "secret",
    "mistake",
    "mistakes",
    "crazy",
    "insane",
    "wild",
    "truth",
    "hack",
    "hacks",
    "controversial",
    "future",
    "never",
    "always",
    "easy",
    "hard",
    "why",
    "how",
    "biggest",
    "best",
    "worst",
    "nobody",
    "everyone",
    "million",
    "billion",
    "percent",
    "ai",
    "agent",
    "growth",
    "viral",
    "attention",
}
EMPHASIS_TERMS = {
    "must",
    "need",
    "important",
    "surprising",
    "unexpected",
    "warning",
    "problem",
    "opportunity",
    "proof",
    "story",
    "lesson",
    "formula",
    "framework",
    "strategy",
    "system",
    "trick",
    "tip",
}
HOOK_TERMS = {
    "wait",
    "watch",
    "look",
    "listen",
    "why",
    "how",
    "secret",
    "mistake",
    "truth",
    "nobody",
    "everyone",
    "actually",
}
PAYOFF_TERMS = {
    "because",
    "therefore",
    "so",
    "means",
    "result",
    "takeaway",
    "lesson",
    "formula",
    "framework",
    "strategy",
    "system",
    "works",
    "fix",
    "solves",
    "answer",
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
    "until",
    "unless",
}
SHAREABILITY_TERMS = {
    "you",
    "your",
    "people",
    "creators",
    "founders",
    "builders",
    "teams",
    "students",
    "developers",
    "businesses",
}
FILLER_STARTERS = {"and", "so", "um", "uh", "okay", "alright", "basically", "like"}
TRAILING_FRAGMENT_TERMS = {"and", "but", "because", "so", "to", "with", "of", "the", "a", "an"}
CONTEXT_DEPENDENT_STARTERS = {
    "and",
    "also",
    "because",
    "but",
    "he",
    "here",
    "it",
    "now",
    "she",
    "so",
    "that",
    "then",
    "these",
    "they",
    "this",
    "those",
    "we",
    "which",
}
SELF_CONTAINED_STARTERS = {
    "here's",
    "heres",
    "how",
    "if",
    "imagine",
    "let",
    "look",
    "the",
    "there",
    "today",
    "wait",
    "watch",
    "what",
    "when",
    "why",
    "you",
}
SETUP_TERMS = {
    "problem",
    "mistake",
    "reason",
    "context",
    "setup",
    "first",
    "before",
    "when",
    "if",
    "imagine",
    "today",
}
LOW_SIGNAL_PHRASES = {
    "kind of",
    "sort of",
    "you know",
    "i guess",
    "maybe like",
    "something like",
}
PLATFORM_HASHTAGS = {
    "youtube_shorts": ["shorts", "youtubeshorts"],
    "tiktok": ["tiktok", "fyp"],
    "instagram_reels": ["reels", "instagramreels"],
}
VIRAL_SCORE_KEYS = ("hook_strength", "payoff", "novelty", "clarity", "shareability")
PLATFORM_PROFILES = {
    "youtube_shorts": {"ideal_duration": 34.0, "caption_style": "creator_bold"},
    "tiktok": {"ideal_duration": 27.0, "caption_style": "creator_bold"},
    "instagram_reels": {"ideal_duration": 30.0, "caption_style": "clean_pop"},
}
SHORTS_ARC_TEMPLATES = [
    {
        "name": "hook_proof_payoff",
        "roles": ["hook", "proof", "payoff"],
        "reason": "fast hook into evidence and payoff",
        "priority": 1.0,
    },
    {
        "name": "quote_context_proof_payoff",
        "roles": ["quote", "context", "proof", "payoff"],
        "reason": "memorable quote with enough context to stand alone",
        "priority": 0.94,
    },
    {
        "name": "tension_proof_payoff",
        "roles": ["tension", "proof", "payoff"],
        "reason": "conflict, evidence, resolution",
        "priority": 0.92,
    },
    {
        "name": "misconception_correction",
        "roles": ["hook", "tension", "proof", "payoff"],
        "reason": "misconception corrected with a clear takeaway",
        "priority": 0.9,
    },
    {
        "name": "setup_tension_payoff",
        "roles": ["setup", "tension", "payoff"],
        "reason": "compact setup into contrast and answer",
        "priority": 0.84,
    },
    {
        "name": "hook_context_payoff",
        "roles": ["hook", "context", "payoff"],
        "reason": "cold-viewer hook with minimum context and close",
        "priority": 0.8,
    },
    {
        "name": "proof_button",
        "roles": ["proof", "payoff", "button"],
        "reason": "specific proof ending on a shareable button",
        "priority": 0.76,
    },
]
ARC_ROLE_ALIASES = {
    "hook": {"hook", "quote", "tension"},
    "context": {"setup", "support", "proof"},
    "setup": {"setup", "support", "hook"},
    "tension": {"tension", "hook", "proof"},
    "proof": {"proof", "tension", "support"},
    "payoff": {"payoff", "proof", "quote"},
    "quote": {"quote", "hook", "payoff"},
    "button": {"payoff", "quote", "proof"},
}
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "for",
    "with", "this", "that", "these", "those", "you", "your", "our", "their",
    "from", "into", "over", "under", "about", "just", "than", "then",
    "they", "them", "have", "has", "had", "was", "were", "are", "is",
    "be", "been", "being", "what", "when", "where", "which",
}


def _safe_stem(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip().lower())
    return re.sub(r"_+", "_", cleaned).strip("_") or "short"


def _extract_json_array(raw_text: str) -> str:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("The model did not return a JSON array.")
    return cleaned[start : end + 1]


def _extract_json_object(raw_text: str) -> str:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("The model did not return a JSON object.")
    return cleaned[start : end + 1]


def _truncate(text: str, limit: int) -> str:
    collapsed = re.sub(r"\s+", " ", text).strip()
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3].rstrip() + "..."


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9']+", text.lower())


def _bounded(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(float(value), high))


def _coerce_float(
    value: object,
    default: float,
    *,
    low: float | None = None,
    high: float | None = None,
    aliases: dict[str, float] | None = None,
) -> float:
    if isinstance(value, str):
        normalized = value.strip().lower().replace("-", "_")
        if aliases and normalized in aliases:
            number = float(aliases[normalized])
        else:
            try:
                number = float(normalized)
            except (TypeError, ValueError):
                number = default
    else:
        try:
            number = float(value if value is not None else default)
        except (TypeError, ValueError):
            number = default
    if low is not None:
        number = max(low, number)
    if high is not None:
        number = min(high, number)
    return number


def _coerce_int(value: object, default: int, *, low: int = 0, high: int = 999) -> int:
    aliases = {"low": max(low, 1), "medium": default, "balanced": default, "high": high, "dense": high}
    number = _coerce_float(value, float(default), low=float(low), high=float(high), aliases={key: float(val) for key, val in aliases.items()})
    return max(low, min(int(round(number)), high))


def _normalize_short_edit_plan_fields(edit_plan: dict, candidate: dict | None = None) -> dict:
    normalized = dict(edit_plan or {})
    fallback_duration = _coerce_float(
        normalized.get("target_duration_sec"),
        _coerce_float((candidate or {}).get("duration"), 0.0, low=0.0, high=120.0),
        low=0.0,
        high=120.0,
    )
    normalized["target_duration_sec"] = fallback_duration
    normalized["quality_floor"] = _coerce_float(
        normalized.get("quality_floor"),
        56.0,
        low=50.0,
        high=90.0,
        aliases={"low": 50.0, "medium": 56.0, "balanced": 56.0, "high": 64.0, "strict": 72.0},
    )

    source_ranges: list[dict] = []
    for index, raw_range in enumerate(normalized.get("source_ranges") or [], start=1):
        if not isinstance(raw_range, dict):
            continue
        source_range = dict(raw_range)
        start = _coerce_float(source_range.get("start"), 0.0, low=0.0, high=1_000_000.0)
        end = _coerce_float(source_range.get("end"), start, low=start, high=1_000_000.0)
        duration = _coerce_float(source_range.get("duration"), max(0.0, end - start), low=0.0, high=120.0)
        source_range["index"] = _coerce_int(source_range.get("index"), index, low=1, high=99)
        source_range["start"] = round(start, 3)
        source_range["end"] = round(end, 3)
        source_range["duration"] = round(duration or max(0.0, end - start), 3)
        source_range["speed"] = round(
            _coerce_float(
                source_range.get("speed"),
                1.0,
                low=0.75,
                high=1.35,
                aliases={"slow": 0.85, "low": 0.9, "normal": 1.0, "medium": 1.0, "high": 1.15, "fast": 1.2},
            ),
            3,
        )
        source_ranges.append(source_range)
    normalized["source_ranges"] = source_ranges

    remix_policy = dict(normalized.get("remix_policy") or {})
    remix_policy["max_source_ranges"] = _coerce_int(remix_policy.get("max_source_ranges"), 6, low=1, high=12)
    normalized["remix_policy"] = remix_policy

    punch_policy = dict(normalized.get("punch_in_policy") or {})
    punch_policy["max_moments"] = _coerce_int(punch_policy.get("max_moments"), 2, low=0, high=8)
    punch_policy["min_gap_sec"] = _coerce_float(punch_policy.get("min_gap_sec"), 0.8, low=0.0, high=8.0)
    punch_policy["max_zoom"] = _coerce_float(
        punch_policy.get("max_zoom"),
        1.18,
        low=1.03,
        high=1.35,
        aliases={"low": 1.06, "medium": 1.12, "balanced": 1.12, "high": 1.18, "strong": 1.22},
    )
    normalized["punch_in_policy"] = punch_policy

    visual_policy = dict(normalized.get("visual_insert_policy") or {})
    visual_policy["max_inserts"] = _coerce_int(visual_policy.get("max_inserts"), 1, low=0, high=8)
    normalized["visual_insert_policy"] = visual_policy

    operations: list[dict] = []
    for raw_operation in normalized.get("operations") or []:
        if not isinstance(raw_operation, dict):
            continue
        operation = dict(raw_operation)
        if operation.get("start_sec") is not None:
            operation["start_sec"] = round(_coerce_float(operation.get("start_sec"), 0.0, low=0.0, high=120.0), 3)
        if operation.get("end_sec") is not None:
            default_end = _coerce_float(operation.get("start_sec"), 0.0, low=0.0, high=120.0)
            operation["end_sec"] = round(_coerce_float(operation.get("end_sec"), default_end, low=0.0, high=120.0), 3)
        params = dict(operation.get("params") or {})
        if params.get("zoom") is not None:
            params["zoom"] = round(
                _coerce_float(
                    params.get("zoom"),
                    1.12,
                    low=1.03,
                    high=1.35,
                    aliases={"low": 1.06, "medium": 1.12, "balanced": 1.12, "high": 1.18, "strong": 1.22},
                ),
                3,
            )
        for key in ("intensity", "strength"):
            if params.get(key) is not None:
                params[key] = round(
                    _coerce_float(
                        params.get(key),
                        0.5,
                        low=0.0,
                        high=1.0,
                        aliases={"low": 0.25, "medium": 0.5, "balanced": 0.5, "high": 0.8, "strong": 0.9},
                    ),
                    3,
                )
        operation["params"] = params
        operations.append(operation)
    normalized["operations"] = operations
    return normalized


def _term_hits(tokens: list[str], terms: set[str]) -> int:
    token_set = set(tokens)
    return len(token_set & terms)


def _phrase_hits(lower_text: str, phrases: set[str]) -> int:
    return sum(1 for phrase in phrases if phrase in lower_text)


def _candidate_keywords(text: str, limit: int = 10) -> list[str]:
    keywords: list[str] = []
    for token in _word_tokens(text):
        if token in STOPWORDS or len(token) < 3:
            continue
        if token not in keywords:
            keywords.append(token)
        if len(keywords) >= limit:
            break
    return keywords


def _important_tokens(text: str) -> list[str]:
    return [token for token in _word_tokens(text) if token not in STOPWORDS and len(token) >= 3]


def _keyword_counts(text: str) -> Counter[str]:
    return Counter(_important_tokens(text))


def _top_keywords(text: str, limit: int = 24) -> list[str]:
    counts = _keyword_counts(text)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))
    return [keyword for keyword, _count in ranked[:limit]]


def _top_phrases(text: str, limit: int = 12) -> list[str]:
    tokens = _important_tokens(text)
    phrases: Counter[str] = Counter()
    for first, second in zip(tokens, tokens[1:]):
        if first == second:
            continue
        phrases[f"{first} {second}"] += 1
    ranked = sorted(phrases.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))
    return [phrase for phrase, count in ranked[:limit] if count >= 1]


def _segment_text(segments: list[dict[str, float | str]]) -> str:
    return " ".join(str(segment.get("text") or "").strip() for segment in segments if str(segment.get("text") or "").strip()).strip()


def _build_video_context(transcript_text: str, segments: list[dict[str, float | str]]) -> dict[str, object]:
    cleaned_transcript = re.sub(r"\s+", " ", transcript_text or _segment_text(segments)).strip()
    segment_count = len(segments)
    opening_segments = segments[: max(1, min(6, max(segment_count // 5, 1)))] if segments else []
    closing_segments = segments[-max(1, min(6, max(segment_count // 5, 1))):] if segments else []
    opening_text = _segment_text(opening_segments)
    closing_text = _segment_text(closing_segments)
    main_keywords = _top_keywords(cleaned_transcript, limit=28)
    opening_keywords = _top_keywords(opening_text, limit=12)
    closing_keywords = _top_keywords(closing_text, limit=12)
    main_phrases = _top_phrases(cleaned_transcript, limit=14)
    full_counts = _keyword_counts(cleaned_transcript)
    repeated_keywords = [
        keyword
        for keyword, count in sorted(full_counts.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))
        if count >= 2
    ][:16]
    core_keywords = []
    for keyword in [*opening_keywords, *closing_keywords, *repeated_keywords]:
        if keyword not in core_keywords:
            core_keywords.append(keyword)
    thesis_excerpt = _truncate(opening_text or cleaned_transcript, 360)
    keyword_weights = {
        keyword: round(1.0 + (max(len(main_keywords) - index, 1) / max(len(main_keywords), 1)), 4)
        for index, keyword in enumerate(main_keywords)
    }
    for keyword in opening_keywords:
        keyword_weights[keyword] = round(float(keyword_weights.get(keyword, 1.0)) + 0.35, 4)
    for keyword in closing_keywords:
        keyword_weights[keyword] = round(float(keyword_weights.get(keyword, 1.0)) + 0.2, 4)
    duration = 0.0
    if segments:
        duration = max(float(segment["end"]) for segment in segments) - min(float(segment["start"]) for segment in segments)
    return {
        "transcript_excerpt": _truncate(cleaned_transcript, 1400),
        "thesis_excerpt": thesis_excerpt,
        "main_keywords": main_keywords,
        "main_phrases": main_phrases,
        "core_keywords": core_keywords[:24],
        "opening_keywords": opening_keywords,
        "closing_keywords": closing_keywords,
        "keyword_weights": keyword_weights,
        "duration": round(max(duration, 0.0), 3),
        "segment_count": segment_count,
    }


def _weighted_keyword_overlap(tokens: list[str], keyword_weights: dict[str, float]) -> float:
    if not tokens or not keyword_weights:
        return 0.0
    token_set = {token for token in tokens if token not in STOPWORDS and len(token) >= 3}
    if not token_set:
        return 0.0
    matched = sum(float(weight) for keyword, weight in keyword_weights.items() if keyword in token_set)
    possible = sum(sorted((float(weight) for weight in keyword_weights.values()), reverse=True)[: max(1, min(len(token_set), 12))])
    return _bounded((matched / max(possible, 0.001)) * 100.0)


def _contextual_candidate_analysis(
    text: str,
    tokens: list[str],
    *,
    video_context: dict[str, object] | None,
    hook_strength: float,
    payoff: float,
    arc_score: float,
    pause_before: float,
    pause_after: float,
    starts_with_filler: bool,
    trailing_fragment: bool,
) -> tuple[dict[str, float], list[str]]:
    keyword_weights = dict((video_context or {}).get("keyword_weights") or {})
    core_keywords = set(str(item) for item in (video_context or {}).get("core_keywords", []) if str(item).strip())
    opening_keywords = set(str(item) for item in (video_context or {}).get("opening_keywords", []) if str(item).strip())
    closing_keywords = set(str(item) for item in (video_context or {}).get("closing_keywords", []) if str(item).strip())
    token_set = {token for token in tokens if token not in STOPWORDS and len(token) >= 3}
    first_token = tokens[0] if tokens else ""
    lower = text.lower()

    contextual_importance = _weighted_keyword_overlap(tokens, keyword_weights)
    core_overlap = (
        _bounded((len(token_set & core_keywords) / max(min(len(core_keywords), len(token_set)), 1)) * 100.0)
        if core_keywords
        else contextual_importance
    )
    thesis_alignment = _bounded(
        24.0
        + contextual_importance * 0.34
        + core_overlap * 0.26
        + min(len(token_set & opening_keywords), 4) * 7.0
        + min(len(token_set & closing_keywords), 4) * 4.5
    )
    standalone_clarity = _bounded(
        42.0
        + hook_strength * 0.22
        + payoff * 0.18
        + (10.0 if first_token in SELF_CONTAINED_STARTERS else 0.0)
        + (8.0 if _term_hits(tokens[:18], SETUP_TERMS) else 0.0)
        + (7.0 if pause_before >= 0.2 else 0.0)
        - (18.0 if first_token in CONTEXT_DEPENDENT_STARTERS else 0.0)
        - (10.0 if starts_with_filler else 0.0)
        - (7.0 if _phrase_hits(lower, LOW_SIGNAL_PHRASES) else 0.0)
    )
    story_completeness = _bounded(
        32.0
        + arc_score * 0.24
        + payoff * 0.26
        + (10.0 if _term_hits(tokens[:20], SETUP_TERMS | HOOK_TERMS) else 0.0)
        + (10.0 if _term_hits(tokens[-24:], PAYOFF_TERMS | EMPHASIS_TERMS) else 0.0)
        + (8.0 if pause_after >= 0.35 else 0.0)
        - (14.0 if trailing_fragment else 0.0)
    )
    abrupt_start_penalty = _bounded(
        (18.0 if first_token in CONTEXT_DEPENDENT_STARTERS else 0.0)
        + (10.0 if starts_with_filler else 0.0)
        + (8.0 if pause_before < 0.08 and first_token not in SELF_CONTAINED_STARTERS else 0.0),
        0.0,
        34.0,
    )
    dangling_payoff_penalty = _bounded(
        (16.0 if trailing_fragment else 0.0)
        + (10.0 if payoff < 48.0 and pause_after < 0.2 else 0.0)
        + (8.0 if not text.rstrip().endswith((".", "?", "!")) and pause_after < 0.3 else 0.0),
        0.0,
        34.0,
    )
    context_dependency_penalty = _bounded(
        (14.0 if first_token in {"this", "that", "these", "those", "it", "they", "them"} else 0.0)
        + (10.0 if len(token_set & core_keywords) <= 1 and core_keywords else 0.0)
        + (8.0 if core_overlap < 20.0 and core_keywords else 0.0),
        0.0,
        34.0,
    )
    misleading_clip_penalty = _bounded(
        (12.0 if core_overlap < 18.0 and (hook_strength >= 70.0 or payoff >= 70.0) else 0.0)
        + (8.0 if len(token_set & core_keywords) == 0 and core_keywords else 0.0),
        0.0,
        26.0,
    )
    score = _bounded(
        contextual_importance * 0.3
        + standalone_clarity * 0.26
        + story_completeness * 0.24
        + thesis_alignment * 0.2
        - abrupt_start_penalty * 0.42
        - dangling_payoff_penalty * 0.36
        - context_dependency_penalty * 0.38
        - misleading_clip_penalty * 0.34,
        1.0,
        100.0,
    )
    reasons: list[str] = []
    if contextual_importance >= 58:
        reasons.append("central to the full video topic")
    if thesis_alignment >= 64:
        reasons.append("aligned with the video's main thesis")
    if standalone_clarity >= 66:
        reasons.append("clear enough without prior context")
    if story_completeness >= 66:
        reasons.append("has setup and payoff inside the clip")
    if abrupt_start_penalty >= 16:
        reasons.append("penalized for abrupt/context-dependent start")
    if dangling_payoff_penalty >= 14:
        reasons.append("penalized for weak ending/payoff")
    if misleading_clip_penalty >= 10:
        reasons.append("penalized for weak full-video alignment")
    return (
        {
            "context_score": round(score, 2),
            "contextual_importance": round(contextual_importance, 2),
            "core_topic_overlap": round(core_overlap, 2),
            "standalone_clarity": round(standalone_clarity, 2),
            "story_completeness": round(story_completeness, 2),
            "thesis_alignment": round(thesis_alignment, 2),
            "abrupt_start_penalty": round(abrupt_start_penalty, 2),
            "dangling_payoff_penalty": round(dangling_payoff_penalty, 2),
            "context_dependency_penalty": round(context_dependency_penalty, 2),
            "misleading_clip_penalty": round(misleading_clip_penalty, 2),
        },
        reasons,
    )


def _duration_fit_score(
    duration: float,
    *,
    min_duration_sec: float,
    max_duration_sec: float,
    target_platform: str,
) -> float:
    profile = PLATFORM_PROFILES.get(target_platform, PLATFORM_PROFILES["youtube_shorts"])
    ideal = max(min_duration_sec, min(float(profile["ideal_duration"]), max_duration_sec))
    tolerance = max(ideal - min_duration_sec, max_duration_sec - ideal, 8.0)
    return _bounded(100.0 * (1.0 - abs(duration - ideal) / tolerance))


def _pace_score(tokens: list[str], duration: float) -> float:
    words_per_sec = len(tokens) / max(duration, 0.1)
    if 2.0 <= words_per_sec <= 4.3:
        return 100.0
    if words_per_sec < 2.0:
        return _bounded(100.0 - (2.0 - words_per_sec) * 38.0)
    return _bounded(100.0 - (words_per_sec - 4.3) * 32.0)


def _window_pause(
    segments: list[dict[str, float | str]] | None,
    start_index: int | None,
    end_index: int | None,
) -> tuple[float, float]:
    if not segments or start_index is None or end_index is None:
        return 0.0, 0.0
    pause_before = 0.0
    pause_after = 0.0
    if start_index > 0:
        pause_before = max(0.0, float(segments[start_index]["start"]) - float(segments[start_index - 1]["end"]))
    if end_index < len(segments) - 1:
        pause_after = max(0.0, float(segments[end_index + 1]["start"]) - float(segments[end_index]["end"]))
    return round(pause_before, 3), round(pause_after, 3)


def _score_transcript_window(
    text: str,
    duration: float,
    *,
    segments: list[dict[str, float | str]] | None = None,
    start_index: int | None = None,
    end_index: int | None = None,
    min_duration_sec: float = 20.0,
    max_duration_sec: float = 45.0,
    target_platform: str = "youtube_shorts",
    video_context: dict[str, object] | None = None,
) -> tuple[float, dict[str, float | int], list[str]]:
    tokens = _word_tokens(text)
    if not tokens:
        return 0.0, {key: 1 for key in VIRAL_SCORE_KEYS} | {"overall": 1}, ["No transcript text."]
    lower = text.lower()
    opener_tokens = tokens[: min(18, len(tokens))]
    closer_tokens = tokens[-min(24, len(tokens)) :]
    opener_text = " ".join(opener_tokens)
    closer_text = " ".join(closer_tokens)
    unique_ratio = len(set(tokens)) / max(len(tokens), 1)
    stopword_ratio = sum(1 for token in tokens if token in STOPWORDS) / max(len(tokens), 1)
    numbers = len(re.findall(r"\b\d+(?:\.\d+)?%?\b", text))
    opener_numbers = len(re.findall(r"\b\d+(?:\.\d+)?%?\b", opener_text))
    viral_hits = _term_hits(tokens, VIRAL_TERMS)
    emphasis_hits = _term_hits(tokens, EMPHASIS_TERMS)
    hook_hits = _term_hits(opener_tokens, HOOK_TERMS | VIRAL_TERMS)
    payoff_hits = _term_hits(closer_tokens, PAYOFF_TERMS | EMPHASIS_TERMS)
    contrast_hits = _term_hits(tokens, CONTRAST_TERMS)
    share_hits = _term_hits(tokens, SHAREABILITY_TERMS)
    low_signal_hits = _phrase_hits(lower, LOW_SIGNAL_PHRASES)
    punctuation_hits = text.count("?") * 4 + text.count("!") * 2
    starts_with_filler = bool(tokens and tokens[0] in FILLER_STARTERS and tokens[0] not in {"why", "how"})
    trailing_fragment = bool(tokens and tokens[-1] in TRAILING_FRAGMENT_TERMS)
    pause_before, pause_after = _window_pause(segments, start_index, end_index)
    has_clean_close = text.rstrip().endswith((".", "?", "!")) or pause_after >= 0.35
    long_word_hits = sum(1 for token in set(tokens) if len(token) >= 7)
    words_per_sec = len(tokens) / max(duration, 0.1)

    duration_score = _duration_fit_score(
        duration,
        min_duration_sec=min_duration_sec,
        max_duration_sec=max_duration_sec,
        target_platform=target_platform,
    )
    pace_score = _pace_score(tokens, duration)
    specificity_score = _bounded(26.0 + min(long_word_hits, 12) * 4.2 + min(numbers, 4) * 8.0 + unique_ratio * 24.0)
    hook_strength = _bounded(
        26.0
        + hook_hits * 13.0
        + opener_numbers * 10.0
        + (18.0 if "?" in opener_text else 0.0)
        + (10.0 if any(term in opener_tokens for term in {"wait", "watch", "look", "why", "how"}) else 0.0)
        + min(punctuation_hits, 10.0)
        - (10.0 if starts_with_filler else 0.0)
    )
    payoff = _bounded(
        28.0
        + payoff_hits * 11.0
        + contrast_hits * 5.5
        + (15.0 if has_clean_close else 0.0)
        + (8.0 if pause_after >= 0.35 else 0.0)
        + min(numbers, 3) * 4.0
        - (12.0 if trailing_fragment else 0.0)
    )
    novelty = _bounded(24.0 + viral_hits * 7.8 + numbers * 8.2 + contrast_hits * 5.2 + specificity_score * 0.42)
    clarity = _bounded(
        22.0
        + duration_score * 0.24
        + pace_score * 0.26
        + unique_ratio * 24.0
        + (8.0 if pause_before >= 0.2 else 0.0)
        - stopword_ratio * 18.0
        - low_signal_hits * 7.0
    )
    shareability = _bounded(
        24.0
        + hook_strength * 0.24
        + payoff * 0.18
        + min(share_hits, 4) * 6.0
        + min(emphasis_hits, 4) * 5.0
        + min(numbers, 3) * 5.0
    )
    arc_score = _bounded(
        18.0
        + hook_strength * 0.26
        + payoff * 0.28
        + (10.0 if contrast_hits else 0.0)
        + (6.0 if pause_before >= 0.2 else 0.0)
        + (8.0 if pause_after >= 0.35 else 0.0)
    )
    context_breakdown, context_reasons = _contextual_candidate_analysis(
        text,
        tokens,
        video_context=video_context,
        hook_strength=hook_strength,
        payoff=payoff,
        arc_score=arc_score,
        pause_before=pause_before,
        pause_after=pause_after,
        starts_with_filler=starts_with_filler,
        trailing_fragment=trailing_fragment,
    )
    penalty = 0.0
    penalty += 8.0 if starts_with_filler else 0.0
    penalty += 9.0 if trailing_fragment else 0.0
    penalty += 9.0 if len(tokens) < 18 else 0.0
    penalty += 7.0 if words_per_sec > 5.4 else 0.0
    penalty += 5.0 if words_per_sec < 1.2 else 0.0
    penalty += min(low_signal_hits * 4.0, 10.0)
    base_overall = _bounded(
        hook_strength * 0.25
        + payoff * 0.22
        + novelty * 0.18
        + clarity * 0.16
        + shareability * 0.13
        + arc_score * 0.06
        - penalty,
        1.0,
        100.0,
    )
    context_enabled = bool((video_context or {}).get("main_keywords"))
    overall = (
        _bounded((base_overall * 0.66) + (float(context_breakdown["context_score"]) * 0.34), 1.0, 100.0)
        if context_enabled
        else base_overall
    )
    breakdown: dict[str, float | int] = {
        "overall": round(overall, 2),
        "hook_strength": round(hook_strength, 2),
        "payoff": round(payoff, 2),
        "novelty": round(novelty, 2),
        "clarity": round(clarity, 2),
        "shareability": round(shareability, 2),
        "arc": round(arc_score, 2),
        "duration_fit": round(duration_score, 2),
        "pace": round(pace_score, 2),
        "specificity": round(specificity_score, 2),
        "word_count": len(tokens),
        "words_per_sec": round(words_per_sec, 2),
        "numeric_hits": numbers,
        "hook_hits": hook_hits,
        "payoff_hits": payoff_hits,
        "contrast_hits": contrast_hits,
        "pause_before": pause_before,
        "pause_after": pause_after,
        "penalty": round(penalty, 2),
        **context_breakdown,
    }
    reasons: list[str] = []
    if hook_strength >= 72:
        reasons.append("strong scroll-stop hook")
    if payoff >= 70:
        reasons.append("clear payoff/closure")
    if numbers:
        reasons.append("concrete numbers increase credibility")
    if context_enabled:
        for reason in context_reasons:
            if reason not in reasons:
                reasons.append(reason)
    if novelty >= 70:
        reasons.append("specific or novel claim")
    if contrast_hits:
        reasons.append("contrast turn creates tension")
    if clarity >= 70:
        reasons.append("good short-form pacing")
    if shareability >= 72:
        reasons.append("quoteable/shareable framing")
    if not reasons:
        reasons.append("best available coherent transcript window")
    return round(overall, 2), breakdown, reasons[:4]


def _heuristic_score(text: str, duration: float) -> float:
    score, _breakdown, _reasons = _score_transcript_window(text, duration)
    return score


def _clamp_score(value: float) -> int:
    return max(1, min(int(round(float(value))), 100))


def _heuristic_viral_score_breakdown(text: str, duration: float) -> dict[str, int]:
    _score, raw_breakdown, _reasons = _score_transcript_window(text, duration)
    hook_strength = _clamp_score(raw_breakdown.get("hook_strength", 1))
    payoff = _clamp_score(raw_breakdown.get("payoff", 1))
    novelty = _clamp_score(raw_breakdown.get("novelty", 1))
    clarity = _clamp_score(raw_breakdown.get("clarity", 1))
    shareability = _clamp_score(raw_breakdown.get("shareability", 1))
    overall = _clamp_score(raw_breakdown.get("overall", 1))
    return {
        "overall": overall,
        "hook_strength": hook_strength,
        "payoff": payoff,
        "novelty": novelty,
        "clarity": clarity,
        "shareability": shareability,
    }


def _build_viral_explanations(text: str, duration: float, score_breakdown: dict[str, int]) -> list[str]:
    tokens = _word_tokens(text)
    lower = text.lower()
    numbers = len(re.findall(r"\b\d+(?:\.\d+)?%?\b", text))
    explanations: list[str] = []
    if any(term in lower for term in VIRAL_TERMS):
        explanations.append("Strong curiosity language gives the clip an immediate hook.")
    if numbers:
        explanations.append("Concrete numbers make the claim feel specific and easier to share.")
    if any(term in lower for term in EMPHASIS_TERMS):
        explanations.append("The transcript has a clear payoff or takeaway instead of vague chatter.")
    if 18.0 <= duration <= 45.0:
        explanations.append("The runtime fits short-form retention and replay behavior well.")
    if len(set(tokens)) / max(len(tokens), 1) > 0.72:
        explanations.append("The wording stays information-dense, which helps pacing and rewatch value.")
    if score_breakdown["hook_strength"] >= 80:
        explanations.append("The opener lands quickly enough to stop the scroll.")
    if score_breakdown["shareability"] >= 80:
        explanations.append("The idea is framed in a way viewers can easily quote or repost.")
    deduped: list[str] = []
    for explanation in explanations:
        if explanation not in deduped:
            deduped.append(explanation)
        if len(deduped) >= 4:
            break
    if not deduped:
        deduped.append("The clip is compact and understandable, which gives it baseline short-form potential.")
    return deduped


def _clip_transcript_text(segments: list[dict[str, float | str]]) -> str:
    return " ".join(str(segment["text"]).strip() for segment in segments if str(segment["text"]).strip()).strip()


def _fallback_viral_analysis(candidate: dict, selection: dict, clip_segments: list[dict[str, float | str]]) -> dict:
    transcript_text = _clip_transcript_text(clip_segments) or str(candidate.get("excerpt") or "")
    score_breakdown = _heuristic_viral_score_breakdown(transcript_text, float(candidate["duration"]))
    explanation = _build_viral_explanations(transcript_text, float(candidate["duration"]), score_breakdown)
    explanation.append(_truncate(str(selection.get("reason") or ""), 150)) if selection.get("reason") else None
    explanation = [item for item in explanation if item]
    return {
        "viral_score": score_breakdown,
        "viral_explanation": explanation[:4],
    }


def _normalize_viral_analysis(raw_analysis: dict, fallback: dict) -> dict:
    raw_scores = raw_analysis.get("viral_score") or {}
    normalized_scores = {
        key: _clamp_score(raw_scores.get(key, fallback["viral_score"][key]))
        for key in ("overall",) + VIRAL_SCORE_KEYS
    }
    explanation = [
        _truncate(str(item).strip(), 140)
        for item in raw_analysis.get("viral_explanation", [])
        if str(item).strip()
    ]
    if not explanation:
        explanation = list(fallback["viral_explanation"])
    return {
        "viral_score": normalized_scores,
        "viral_explanation": explanation[:4],
    }




def _format_timestamped_clip_segments(segments: list[dict[str, float | str]]) -> str:
    return "\n".join(
        f"{float(segment['start']):.2f}-{float(segment['end']):.2f}: {str(segment['text']).strip()}"
        for segment in segments
        if str(segment["text"]).strip()
    )


def _keyword_phrase(text: str, limit: int = 5) -> str:
    words: list[str] = []
    for token in _word_tokens(text):
        if token in STOPWORDS or len(token) < 3:
            continue
        if token not in words:
            words.append(token)
        if len(words) >= limit:
            break
    return " ".join(words) or _truncate(text, 50)


def _fallback_b_roll_suggestions(clip_segments: list[dict[str, float | str]]) -> list[dict]:
    suggestions: list[dict] = []
    last_end = -999.0
    for segment in clip_segments:
        text = str(segment["text"]).strip()
        if not text:
            continue
        start_sec = float(segment["start"])
        end_sec = float(segment["end"])
        if start_sec - last_end < 1.0:
            continue
        lower = text.lower()
        if any(term in lower for term in {"chart", "metric", "growth", "percent", "data", "revenue", "users"}):
            visual_type = "data_graphic"
            direction = "Show a quick chart, number card, or graph that reinforces the spoken metric."
        elif any(term in lower for term in {"tool", "app", "product", "website", "dashboard", "workflow", "agent"}):
            visual_type = "product_ui"
            direction = "Show a UI clip, dashboard screenshot, or product walkthrough beat tied to the claim."
        elif any(term in lower for term in {"customer", "team", "founder", "people", "creator", "audience"}):
            visual_type = "cutaway"
            direction = "Use a human cutaway or reaction-style insert that supports the point without covering captions."
        else:
            visual_type = "text_overlay"
            direction = "Use a quick illustrative insert or animated text card to underline the spoken takeaway."
        clip_end = min(end_sec, start_sec + 2.4)
        suggestions.append(
            {
                "start": round(max(start_sec, 0.0), 2),
                "end": round(max(clip_end, start_sec + 0.8), 2),
                "visual_type": visual_type,
                "search_query": _truncate(_keyword_phrase(text, limit=6), 70),
                "direction": _truncate(direction, 110),
                "rationale": _truncate("This beat has enough semantic density to support a reinforcing visual without distracting from the spoken payoff.", 130),
            }
        )
        last_end = clip_end
        if len(suggestions) >= 3:
            break
    if not suggestions and clip_segments:
        first = clip_segments[0]
        suggestions.append(
            {
                "start": round(float(first["start"]), 2),
                "end": round(min(float(first["end"]), float(first["start"]) + 2.0), 2),
                "visual_type": "text_overlay",
                "search_query": _truncate(_keyword_phrase(str(first["text"]), limit=6), 70),
                "direction": "Use a simple reinforcing visual or title card that clarifies the core point.",
                "rationale": "The opener is the safest place to reinforce context if no stronger B-roll beat stands out.",
            }
        )
    return suggestions


def _normalize_b_roll_suggestions(raw_suggestions: list[dict], fallback: list[dict], clip_duration: float) -> list[dict]:
    suggestions: list[dict] = []
    source = raw_suggestions or fallback
    for item in source:
        try:
            start_sec = max(0.0, min(float(item.get("start", 0.0)), clip_duration))
            end_sec = max(start_sec + 0.3, min(float(item.get("end", start_sec + 1.8)), clip_duration))
        except Exception:
            continue
        if end_sec <= start_sec:
            continue
        suggestions.append(
            {
                "start": round(start_sec, 2),
                "end": round(end_sec, 2),
                "visual_type": _truncate(str(item.get("visual_type") or "text_overlay"), 32),
                "search_query": _truncate(str(item.get("search_query") or "supporting visual"), 70),
                "direction": _truncate(str(item.get("direction") or "Add a supporting visual beat."), 110),
                "rationale": _truncate(str(item.get("rationale") or "Supports the spoken point visually."), 130),
            }
        )
        if len(suggestions) >= 4:
            break
    return suggestions or fallback[:3]


def _analyze_b_roll_with_llm(
    provider_name: str,
    model_name: str,
    candidate: dict,
    selection: dict,
    clip_segments: list[dict[str, float | str]],
    target_platform: str,
) -> list[dict]:
    fallback = _fallback_b_roll_suggestions(clip_segments)
    transcript_text = _clip_transcript_text(clip_segments) or str(candidate.get("excerpt") or "")
    timestamped_transcript = _format_timestamped_clip_segments(clip_segments)
    system_prompt = (
        "You are a short-form producer. Suggest strong B-roll beats for a short clip. "
        "Return ONLY a JSON array of up to 4 objects with keys start, end, visual_type, search_query, direction, rationale."
    )
    user_prompt = (
        f"Platform: {target_platform}\n"
        f"Candidate window: {candidate['start']:.2f}-{candidate['end']:.2f} ({candidate['duration']:.2f}s)\n"
        f"Draft title: {selection.get('title', '')}\n"
        f"Draft hook: {selection.get('hook', '')}\n\n"
        f"Transcript overview:\n{_truncate(transcript_text, 2200)}\n\n"
        f"Timestamped transcript:\n{_truncate(timestamped_transcript, 2600)}\n\n"
        "Prefer supportive visuals that reinforce the point without covering captions or feeling generic. Return JSON array only."
    )
    try:
        raw_text = _call_reasoning_model(provider_name, model_name, system_prompt, user_prompt)
        parsed = json.loads(_extract_json_array(raw_text))
    except Exception:
        return fallback
    return _normalize_b_roll_suggestions(parsed, fallback, clip_duration=float(candidate["duration"]))



def _fallback_punch_in_moments(clip_segments: list[dict[str, float | str]]) -> list[dict]:
    moments: list[dict] = []
    last_end = -999.0
    for segment in clip_segments:
        text = str(segment["text"]).strip()
        if not text:
            continue
        start_sec = float(segment["start"])
        end_sec = float(segment["end"])
        lower = text.lower()
        emphasis_score = 0
        emphasis_score += text.count("?") * 2
        emphasis_score += text.count("!")
        emphasis_score += sum(1 for term in EMPHASIS_TERMS if term in lower)
        emphasis_score += sum(1 for term in VIRAL_TERMS if term in lower)
        emphasis_score += len(re.findall(r"\b\d+(?:\.\d+)?%?\b", text))
        if emphasis_score < 2 or start_sec - last_end < 0.8:
            continue
        clip_end = min(end_sec, start_sec + 1.7)
        moments.append(
            {
                "start": round(max(start_sec, 0.0), 2),
                "end": round(max(clip_end, start_sec + 0.7), 2),
                "zoom": round(min(1.08 + emphasis_score * 0.015, 1.18), 2),
                "reason": _truncate("Emphasis-heavy line with a likely payoff beat.", 90),
            }
        )
        last_end = clip_end
        if len(moments) >= 3:
            break
    return moments


def _normalize_punch_in_moments(raw_moments: list[dict], fallback: list[dict], clip_duration: float) -> list[dict]:
    moments: list[dict] = []
    source = raw_moments or fallback
    for item in source:
        try:
            start_sec = max(0.0, min(float(item.get("start", 0.0)), clip_duration))
            end_sec = max(start_sec + 0.3, min(float(item.get("end", start_sec + 1.2)), clip_duration))
            zoom = max(1.05, min(float(item.get("zoom", 1.12)), 1.22))
        except Exception:
            continue
        if end_sec <= start_sec:
            continue
        moments.append(
            {
                "start": round(start_sec, 2),
                "end": round(end_sec, 2),
                "zoom": round(zoom, 2),
                "reason": _truncate(str(item.get("reason") or "Punch in on an emphasis beat."), 90),
            }
        )
        if len(moments) >= 4:
            break
    return moments


def _analyze_punch_in_with_llm(
    provider_name: str,
    model_name: str,
    candidate: dict,
    selection: dict,
    clip_segments: list[dict[str, float | str]],
    target_platform: str,
) -> list[dict]:
    fallback = _fallback_punch_in_moments(clip_segments)
    if not clip_segments:
        return fallback
    transcript_text = _clip_transcript_text(clip_segments) or str(candidate.get("excerpt") or "")
    timestamped_transcript = _format_timestamped_clip_segments(clip_segments)
    system_prompt = (
        "You are a short-form editor. Pick the best center punch-in moments for emphasis in a short clip. "
        "Return ONLY a JSON array of up to 4 objects with keys start, end, zoom, reason."
    )
    user_prompt = (
        f"Platform: {target_platform}\n"
        f"Candidate window: {candidate['start']:.2f}-{candidate['end']:.2f} ({candidate['duration']:.2f}s)\n"
        f"Draft title: {selection.get('title', '')}\n"
        f"Draft hook: {selection.get('hook', '')}\n\n"
        f"Transcript overview:\n{_truncate(transcript_text, 2200)}\n\n"
        f"Timestamped transcript:\n{_truncate(timestamped_transcript, 2600)}\n\n"
        "Choose moments where a subtle center punch-in would increase emphasis without feeling overedited. Return JSON array only."
    )
    try:
        raw_text = _call_reasoning_model(provider_name, model_name, system_prompt, user_prompt)
        parsed = json.loads(_extract_json_array(raw_text))
    except Exception:
        return fallback
    return _normalize_punch_in_moments(parsed, fallback, clip_duration=float(candidate["duration"]))


def _candidate_source_ranges(candidate: dict) -> list[dict]:
    raw_ranges = candidate.get("source_ranges")
    ranges: list[dict] = []
    if isinstance(raw_ranges, list):
        for index, raw_range in enumerate(raw_ranges, start=1):
            if not isinstance(raw_range, dict):
                continue
            try:
                start_sec = max(0.0, float(raw_range.get("start", 0.0)))
                end_sec = max(start_sec, float(raw_range.get("end", start_sec)))
            except (TypeError, ValueError):
                continue
            if end_sec <= start_sec:
                continue
            try:
                source_index = int(raw_range.get("index") or index)
            except (TypeError, ValueError):
                source_index = index
            ranges.append(
                {
                    "index": source_index,
                    "start": round(start_sec, 3),
                    "end": round(end_sec, 3),
                    "duration": round(end_sec - start_sec, 3),
                    "role": _normalize_source_role(str(raw_range.get("role") or "part")),
                    **_optional_source_range_fields(raw_range),
                }
            )
    if ranges:
        return ranges
    try:
        start_sec = max(0.0, float(candidate.get("start", 0.0)))
        end_sec = max(start_sec, float(candidate.get("end", start_sec)))
    except (TypeError, ValueError):
        return []
    if end_sec <= start_sec:
        return []
    return [
        {
            "index": 1,
            "start": round(start_sec, 3),
            "end": round(end_sec, 3),
            "duration": round(end_sec - start_sec, 3),
            "role": "primary",
        }
    ]


def _optional_source_range_fields(raw_range: dict) -> dict:
    optional: dict[str, object] = {}
    for key, limit in {
        "reason": 180,
        "transition": 32,
        "crop_hint": 32,
        "source_role": 32,
        "part_id": 64,
    }.items():
        value = raw_range.get(key)
        if value is not None and str(value).strip():
            optional[key] = _truncate(str(value), limit)
    speed = raw_range.get("speed")
    if speed is not None:
        try:
            optional["speed"] = round(max(0.75, min(float(speed), 1.35)), 3)
        except (TypeError, ValueError):
            pass
    unit_ids = raw_range.get("unit_ids")
    if isinstance(unit_ids, list):
        normalized_unit_ids = [
            str(unit_id).strip()
            for unit_id in unit_ids
            if str(unit_id).strip()
        ]
        if normalized_unit_ids:
            optional["unit_ids"] = normalized_unit_ids[:64]
    return optional


def _normalize_source_role(role: str) -> str:
    normalized = re.sub(r"[^a-z0-9_]+", "_", str(role or "").lower()).strip("_")
    if normalized in {"hook", "context", "setup", "tension", "proof", "payoff", "quote", "support", "button", "primary", "part"}:
        return normalized
    return "part"


def _source_ranges_duration(source_ranges: list[dict]) -> float:
    duration = 0.0
    for source_range in source_ranges:
        duration += max(0.0, float(source_range.get("end", 0.0)) - float(source_range.get("start", 0.0)))
    return round(duration, 3)


def _source_ranges_overlap_ratio(first: list[dict], second: list[dict]) -> float:
    first_duration = _source_ranges_duration(first)
    second_duration = _source_ranges_duration(second)
    if first_duration <= 0.0 or second_duration <= 0.0:
        return 0.0
    overlap = 0.0
    for first_range in first:
        first_start = float(first_range["start"])
        first_end = float(first_range["end"])
        for second_range in second:
            second_start = float(second_range["start"])
            second_end = float(second_range["end"])
            overlap += max(0.0, min(first_end, second_end) - max(first_start, second_start))
    return overlap / max(min(first_duration, second_duration), 0.001)


def _source_ranges_bounds(source_ranges: list[dict]) -> tuple[float, float]:
    if not source_ranges:
        return 0.0, 0.0
    return min(float(item["start"]) for item in source_ranges), max(float(item["end"]) for item in source_ranges)


def _overlap_ratio(first: dict, second: dict) -> float:
    return _source_ranges_overlap_ratio(_candidate_source_ranges(first), _candidate_source_ranges(second))


def _dedupe_candidates(candidates: list[dict], limit: int) -> list[dict]:
    selected: list[dict] = []
    for candidate in candidates:
        if all(_overlap_ratio(candidate, existing) < 0.68 for existing in selected):
            selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected


def _apply_creative_graph_to_candidates(
    candidates: list[dict],
    creative_graph,
    *,
    target_platform: str,
) -> None:
    for candidate in candidates:
        source_ranges = _candidate_source_ranges(candidate)
        signals = candidate_graph_signals(
            creative_graph,
            start=float(candidate.get("start") or 0.0),
            end=float(candidate.get("end") or 0.0),
            text=str(candidate.get("excerpt") or ""),
            source_ranges=source_ranges,
            target_platform=target_platform,
        )
        graph_score = float(signals.get("graph_retention_score") or 0.0) * 100.0
        continuity_risk = float(signals.get("graph_continuity_risk") or 0.0) * 100.0
        original_score = float(candidate.get("heuristic_score") or 1.0)
        blended_score = _bounded(
            (original_score * 0.84)
            + (graph_score * 0.18)
            - (continuity_risk * 0.04),
            1.0,
            100.0,
        )
        breakdown = dict(candidate.get("score_breakdown") or {})
        breakdown.update(
            {
                "creative_graph_retention_score": round(graph_score, 2),
                "creative_graph_topic_alignment": round(float(signals.get("graph_topic_alignment") or 0.0) * 100.0, 2),
                "creative_graph_visual_opportunity": round(float(signals.get("graph_visual_opportunity") or 0.0) * 100.0, 2),
                "creative_graph_continuity_risk": round(continuity_risk, 2),
            }
        )
        reasons = list(candidate.get("selection_reasons") or [])
        if graph_score >= 68.0:
            reasons.append("creative graph: high retention opportunity")
        if float(signals.get("graph_visual_opportunity") or 0.0) >= 0.56:
            reasons.append("creative graph: strong visual support opportunity")
        if float(signals.get("graph_topic_alignment") or 0.0) >= 0.58:
            reasons.append("creative graph: central to full-video thesis")
        candidate["heuristic_score"] = round(blended_score, 2)
        candidate["creative_graph_signals"] = signals
        candidate["score_breakdown"] = breakdown
        quality_report = evaluate_short_candidate_quality(
            candidate,
            creative_graph,
            target_platform=target_platform,
        ).to_dict()
        candidate["creative_quality_report"] = quality_report
        candidate["creative_quality_score"] = round(float(quality_report["score"]) * 100.0, 2)
        breakdown["creative_quality_score"] = candidate["creative_quality_score"]
        candidate["selection_reasons"] = _dedupe_text(reasons)[:8]
    candidates.sort(
        key=lambda item: (
            float(item.get("heuristic_score") or 0.0),
            float(item.get("creative_quality_score") or 0.0),
            float((item.get("score_breakdown") or {}).get("creative_graph_retention_score", 0.0)),
            float((item.get("score_breakdown") or {}).get("hook_strength", 0.0)),
            float((item.get("score_breakdown") or {}).get("payoff", 0.0)),
        ),
        reverse=True,
    )


def _build_candidates(
    segments: list[dict[str, float | str]],
    min_duration_sec: float,
    max_duration_sec: float,
    limit: int = 64,
    target_platform: str = "youtube_shorts",
    video_context: dict[str, object] | None = None,
) -> list[dict]:
    candidates: list[dict] = []
    candidate_index = 1
    for start_index in range(len(segments)):
        start_sec = float(segments[start_index]["start"])
        text_parts: list[str] = []
        for end_index in range(start_index, len(segments)):
            segment = segments[end_index]
            text_parts.append(str(segment["text"]).strip())
            end_sec = float(segment["end"])
            duration = end_sec - start_sec
            if duration > max_duration_sec and end_index > start_index:
                break
            if duration < min_duration_sec:
                continue
            text = " ".join(part for part in text_parts if part).strip()
            if len(_word_tokens(text)) < 12:
                continue
            score, breakdown, reasons = _score_transcript_window(
                text,
                duration,
                segments=segments,
                start_index=start_index,
                end_index=end_index,
                min_duration_sec=min_duration_sec,
                max_duration_sec=max_duration_sec,
                target_platform=target_platform,
                video_context=video_context,
            )
            candidates.append(
                {
                    "candidate_id": f"cand_{candidate_index:02d}",
                    "start": round(start_sec, 2),
                    "end": round(end_sec, 2),
                    "duration": round(duration, 2),
                    "excerpt": _truncate(text, 360),
                    "heuristic_score": score,
                    "score_breakdown": breakdown,
                    "selection_reasons": reasons,
                    "keywords": _candidate_keywords(text),
                }
            )
            candidate_index += 1
    if not candidates and segments:
        start_sec = float(segments[0]["start"])
        end_sec = float(segments[-1]["end"])
        text = " ".join(str(segment["text"]).strip() for segment in segments)
        score, breakdown, reasons = _score_transcript_window(
            text,
            end_sec - start_sec,
            segments=segments,
            start_index=0,
            end_index=len(segments) - 1,
            min_duration_sec=min_duration_sec,
            max_duration_sec=max_duration_sec,
            target_platform=target_platform,
            video_context=video_context,
        )
        candidates.append(
            {
                "candidate_id": "cand_01",
                "start": round(start_sec, 2),
                "end": round(end_sec, 2),
                "duration": round(end_sec - start_sec, 2),
                "excerpt": _truncate(text, 360),
                "heuristic_score": score,
                "score_breakdown": breakdown,
                "selection_reasons": reasons + ["only viable transcript window"],
                "keywords": _candidate_keywords(text),
            }
        )
    candidates.sort(
        key=lambda item: (
            float(item["heuristic_score"]),
            float((item.get("score_breakdown") or {}).get("hook_strength", 0.0)),
            float((item.get("score_breakdown") or {}).get("payoff", 0.0)),
        ),
        reverse=True,
    )
    return _dedupe_candidates(candidates, limit=limit)


def _build_remix_candidates(
    candidates: list[dict],
    segments: list[dict[str, float | str]],
    *,
    min_duration_sec: float,
    max_duration_sec: float,
    target_platform: str,
    video_context: dict[str, object] | None,
    limit: int = 24,
) -> list[dict]:
    parts = _build_remix_parts(
        segments,
        target_platform=target_platform,
        video_context=video_context,
        max_part_duration=max(6.0, min(15.0, max_duration_sec * 0.42)),
    )
    if len(parts) < 2:
        return []
    edit_arcs = _search_short_edit_arcs(
        parts,
        min_duration_sec=min_duration_sec,
        max_duration_sec=max_duration_sec,
        target_platform=target_platform,
        limit=max(limit * 3, 36),
    )

    remixes: list[dict] = []
    remix_index = 1
    for arc in edit_arcs:
        selected_parts = list(arc["parts"])
        duration = round(sum(float(part["duration"]) for part in selected_parts), 2)
        if duration < min_duration_sec or duration > max_duration_sec:
            continue
        source_ranges = _source_ranges_for_edit_arc(selected_parts, arc)
        if len(source_ranges) < 2:
            continue
        if any(_source_ranges_overlap_ratio(source_ranges, _candidate_source_ranges(existing)) >= 0.74 for existing in remixes):
            continue
        if any(_source_ranges_overlap_ratio(source_ranges, _candidate_source_ranges(existing)) >= 0.9 for existing in candidates):
            continue
        transcript_parts = [str(part["text"]).strip() for part in selected_parts if str(part.get("text") or "").strip()]
        transcript_text = " ".join(transcript_parts).strip()
        semantic_unit_by_id = {
            str(segment.get("unit_id")): segment
            for segment in segments
            if str(segment.get("unit_id") or "").strip()
        }
        selected_semantic_units = [
            semantic_unit_by_id[unit_id]
            for source_range in source_ranges
            for unit_id in source_range.get("unit_ids", [])
            if unit_id in semantic_unit_by_id
        ]
        story_critic = evaluate_story_candidate(
            transcript_text,
            source_ranges,
            selected_semantic_units,
            planner_confidence=float(arc.get("score") or 50.0),
        )
        if selected_semantic_units and not story_critic["passed"]:
            continue
        score, breakdown, reasons = _score_transcript_window(
            transcript_text,
            duration,
            min_duration_sec=min_duration_sec,
            max_duration_sec=max_duration_sec,
            target_platform=target_platform,
            video_context=video_context,
        )
        arc_score = float(arc.get("score") or 0.0)
        role_set = {str(source_range["role"]) for source_range in source_ranges}
        role_bonus = 0.0
        role_bonus += 2.0 if "hook" in role_set or "quote" in role_set else 0.0
        role_bonus += 2.0 if "proof" in role_set or "tension" in role_set else 0.0
        role_bonus += 2.0 if "payoff" in role_set or "button" in role_set else 0.0
        score = _bounded(
            (score * 0.56)
            + (arc_score * 0.24)
            + (float(story_critic["score"]) * 0.20)
            + role_bonus
            - max(len(source_ranges) - 2, 0) * 3.0,
            1.0,
            100.0,
        )
        breakdown.update(
            {
                "remix_role_bonus": round(role_bonus, 2),
                "edit_arc_score": round(arc_score, 2),
                "edit_arc_template": str(arc.get("template") or "stitched_arc"),
                "edit_arc_cohesion": round(float(arc.get("cohesion") or 0.0), 2),
                "edit_arc_continuity_risk": round(float(arc.get("continuity_risk") or 0.0), 2),
                "story_critic_score": round(float(story_critic["score"]), 2),
                "stitch_count": max(len(source_ranges) - 1, 0),
            }
        )
        start_sec, end_sec = _source_ranges_bounds(source_ranges)
        remixes.append(
            {
                "candidate_id": f"remix_{remix_index:02d}",
                "start": round(start_sec, 2),
                "end": round(end_sec, 2),
                "duration": duration,
                "composition_mode": "remix",
                "source_ranges": source_ranges,
                "source_excerpt_parts": transcript_parts,
                "remix_strategy": str(arc.get("label") or _remix_strategy_label(selected_parts)),
                "excerpt": _truncate(" ... ".join(transcript_parts), 520),
                "heuristic_score": round(score, 2),
                "score_breakdown": breakdown,
                "edit_plan_seed": {
                    "arc_template": str(arc.get("template") or "stitched_arc"),
                    "strategy": "graph_beam_search",
                    "operations": _planned_operations_for_edit_arc(source_ranges, duration),
                    "continuity_risk": round(float(arc.get("continuity_risk") or 0.0), 2),
                    "cohesion": round(float(arc.get("cohesion") or 0.0), 2),
                },
                "story_plan": {
                    "version": "shorts-story-compiler-v1",
                    "unit_ids": [
                        unit_id
                        for source_range in source_ranges
                        for unit_id in source_range.get("unit_ids", [])
                    ],
                    "critic": story_critic,
                },
                "selection_reasons": _dedupe_text(
                    [
                        f"director edit graph: {arc.get('label') or _remix_strategy_label(selected_parts)}",
                        "stitches separate high-signal beats into a typed short edit plan",
                        str(arc.get("reason") or ""),
                        *reasons,
                    ]
                )[:7],
                "keywords": _candidate_keywords(transcript_text, limit=10),
                "candidate_origin": "deterministic_story_graph",
            }
        )
        remix_index += 1
        if len(remixes) >= limit:
            break
    remixes.sort(
        key=lambda item: (
            float(item["heuristic_score"]),
            float((item.get("score_breakdown") or {}).get("edit_arc_score", 0.0)),
            float((item.get("score_breakdown") or {}).get("hook_strength", 0.0)),
            float((item.get("score_breakdown") or {}).get("payoff", 0.0)),
        ),
        reverse=True,
    )
    return remixes[:limit]


def _search_short_edit_arcs(
    parts: list[dict],
    *,
    min_duration_sec: float,
    max_duration_sec: float,
    target_platform: str,
    limit: int,
) -> list[dict]:
    ranked_parts = parts[: min(len(parts), 54)]
    arcs: list[dict] = []
    for template in SHORTS_ARC_TEMPLATES:
        beams: list[dict] = [{"parts": [], "target_roles": [], "score": 0.0}]
        for target_role in template["roles"]:
            next_beams: list[dict] = []
            role_parts = _parts_for_arc_role(ranked_parts, str(target_role))[:18]
            if not role_parts:
                role_parts = ranked_parts[:18]
            for beam in beams:
                for part in role_parts:
                    candidate_parts = [*beam["parts"], part]
                    if len({str(item.get("part_id")) for item in candidate_parts}) != len(candidate_parts):
                        continue
                    if not _remix_parts_are_distinct(candidate_parts):
                        continue
                    duration = sum(float(item["duration"]) for item in candidate_parts)
                    if duration > max_duration_sec:
                        continue
                    next_beams.append(
                        {
                            "parts": candidate_parts,
                            "target_roles": [*beam["target_roles"], str(target_role)],
                            "score": _score_edit_arc_sequence(
                                candidate_parts,
                                [*beam["target_roles"], str(target_role)],
                                template_priority=float(template.get("priority", 1.0)),
                                min_duration_sec=min_duration_sec,
                                max_duration_sec=max_duration_sec,
                                target_platform=target_platform,
                                partial=True,
                            ),
                        }
                    )
            beams = sorted(next_beams, key=lambda item: float(item["score"]), reverse=True)[:28]
            if not beams:
                break
        for beam in beams:
            selected_parts = _expand_remix_until_duration(
                list(beam["parts"]),
                ranked_parts,
                min_duration_sec=min_duration_sec,
                max_duration_sec=max_duration_sec,
            )
            duration = sum(float(item["duration"]) for item in selected_parts)
            if duration < min_duration_sec or duration > max_duration_sec:
                continue
            target_roles = list(beam["target_roles"])
            if len(selected_parts) > len(target_roles):
                target_roles.extend(str(item.get("role") or "support") for item in selected_parts[len(target_roles):])
            score = _score_edit_arc_sequence(
                selected_parts,
                target_roles,
                template_priority=float(template.get("priority", 1.0)),
                min_duration_sec=min_duration_sec,
                max_duration_sec=max_duration_sec,
                target_platform=target_platform,
            )
            continuity_risk = _edit_arc_continuity_risk(selected_parts, target_roles)
            cohesion = _edit_arc_topic_cohesion(selected_parts)
            arcs.append(
                {
                    "template": str(template["name"]),
                    "label": " -> ".join(target_roles[: len(selected_parts)]),
                    "reason": str(template.get("reason") or ""),
                    "parts": selected_parts,
                    "target_roles": target_roles[: len(selected_parts)],
                    "score": round(score, 3),
                    "continuity_risk": round(continuity_risk, 3),
                    "cohesion": round(cohesion, 3),
                }
            )
    arcs.sort(
        key=lambda item: (
            float(item["score"]),
            float(item["cohesion"]),
            -float(item["continuity_risk"]),
            -len(item["parts"]),
        ),
        reverse=True,
    )
    selected: list[dict] = []
    for arc in arcs:
        source_ranges = _source_ranges_for_edit_arc(list(arc["parts"]), arc)
        if all(_source_ranges_overlap_ratio(source_ranges, _source_ranges_for_edit_arc(list(existing["parts"]), existing)) < 0.62 for existing in selected):
            selected.append(arc)
        if len(selected) >= limit:
            break
    return selected


def _parts_for_arc_role(parts: list[dict], target_role: str) -> list[dict]:
    allowed_roles = ARC_ROLE_ALIASES.get(target_role, {target_role})
    ranked = [
        part
        for part in parts
        if str(part.get("role") or "support") in allowed_roles
    ]
    return sorted(
        ranked,
        key=lambda item: (
            _role_fit_score(str(item.get("role") or ""), target_role),
            float(item.get("score") or 0.0),
            float((item.get("score_breakdown") or {}).get("context_score", 0.0)),
        ),
        reverse=True,
    )


def _role_fit_score(source_role: str, target_role: str) -> float:
    if source_role == target_role:
        return 1.0
    if source_role in ARC_ROLE_ALIASES.get(target_role, set()):
        return 0.72
    return 0.2


def _score_edit_arc_sequence(
    parts: list[dict],
    target_roles: list[str],
    *,
    template_priority: float,
    min_duration_sec: float,
    max_duration_sec: float,
    target_platform: str,
    partial: bool = False,
) -> float:
    if not parts:
        return 0.0
    duration = sum(float(part["duration"]) for part in parts)
    average_part_score = sum(float(part.get("score") or 0.0) for part in parts) / len(parts)
    role_fit = sum(_role_fit_score(str(part.get("role") or ""), role) for part, role in zip(parts, target_roles)) / max(len(parts), 1)
    role_set = set(target_roles)
    arc_bonus = 0.0
    arc_bonus += 10.0 if "hook" in role_set or "quote" in role_set else 0.0
    arc_bonus += 9.0 if "proof" in role_set or "tension" in role_set else 0.0
    arc_bonus += 11.0 if "payoff" in role_set or "button" in role_set else 0.0
    duration_score = _duration_fit_score(
        duration,
        min_duration_sec=min_duration_sec,
        max_duration_sec=max_duration_sec,
        target_platform=target_platform,
    )
    cohesion = _edit_arc_topic_cohesion(parts)
    continuity_risk = _edit_arc_continuity_risk(parts, target_roles)
    partial_penalty = 5.0 if partial and duration < min_duration_sec else 0.0
    return _bounded(
        average_part_score * 0.38
        + role_fit * 100.0 * 0.16
        + duration_score * 0.12
        + cohesion * 0.12
        + arc_bonus
        + template_priority * 8.0
        - continuity_risk * 0.18
        - partial_penalty,
        1.0,
        100.0,
    )


def _edit_arc_topic_cohesion(parts: list[dict]) -> float:
    keyword_sets = [
        {str(keyword).lower() for keyword in part.get("keywords", []) if str(keyword).strip()}
        for part in parts
    ]
    keyword_sets = [keywords for keywords in keyword_sets if keywords]
    if len(keyword_sets) <= 1:
        return 72.0 if keyword_sets else 45.0
    shared = set.intersection(*keyword_sets)
    union = set.union(*keyword_sets)
    pair_overlap = 0.0
    pair_count = 0
    for index, first in enumerate(keyword_sets):
        for second in keyword_sets[index + 1:]:
            pair_overlap += len(first & second) / max(len(first | second), 1)
            pair_count += 1
    pair_score = (pair_overlap / max(pair_count, 1)) * 100.0
    shared_score = (len(shared) / max(len(union), 1)) * 100.0
    return _bounded(pair_score * 0.72 + shared_score * 0.28 + 18.0)


def _edit_arc_continuity_risk(parts: list[dict], target_roles: list[str]) -> float:
    if not parts:
        return 100.0
    risk = 0.0
    first_text = str(parts[0].get("text") or "")
    first_tokens = _word_tokens(first_text)
    if first_tokens and first_tokens[0] in CONTEXT_DEPENDENT_STARTERS and target_roles[0] not in {"context", "setup"}:
        risk += 22.0
    if target_roles[0] not in {"hook", "quote", "setup", "context"}:
        risk += 12.0
    if not any(role in {"payoff", "proof", "button", "quote"} for role in target_roles):
        risk += 18.0
    for previous, current in zip(parts, parts[1:]):
        previous_end = float(previous.get("end") or 0.0)
        current_start = float(current.get("start") or 0.0)
        gap = abs(current_start - previous_end)
        if gap > 90.0:
            risk += 5.0
        if current_start < previous_end:
            risk += 3.0
        current_tokens = _word_tokens(str(current.get("text") or ""))
        if current_tokens and current_tokens[0] in CONTEXT_DEPENDENT_STARTERS:
            risk += 6.0
    return _bounded(risk)


def _source_ranges_for_edit_arc(parts: list[dict], arc: dict) -> list[dict]:
    target_roles = list(arc.get("target_roles") or [])
    source_ranges: list[dict] = []
    for index, part in enumerate(parts, start=1):
        target_role = _normalize_source_role(str(target_roles[index - 1] if index - 1 < len(target_roles) else part.get("role") or "support"))
        source_role = _normalize_source_role(str(part.get("role") or target_role))
        source_ranges.append(
            {
                "index": index,
                "start": round(float(part["start"]), 3),
                "end": round(float(part["end"]), 3),
                "duration": round(float(part["duration"]), 3),
                "role": target_role,
                "source_role": source_role,
                "part_id": str(part.get("part_id") or f"part_{index:02d}"),
                "unit_ids": list(part.get("unit_ids") or []),
                "reason": _source_range_reason_for_arc(target_role, part, arc),
                "transition": "hard_cut" if index > 1 else "open",
                "speed": 1.0,
                "crop_hint": _crop_hint_for_source_role(target_role),
            }
        )
    return source_ranges


def _source_range_reason_for_arc(target_role: str, part: dict, arc: dict) -> str:
    text = str(part.get("text") or "")
    if target_role == "hook":
        return _truncate("opens with the strongest cold-viewer hook: " + text, 180)
    if target_role == "context":
        return _truncate("adds only the setup required for the edit to stand alone: " + text, 180)
    if target_role == "proof":
        return _truncate("supports the claim with specific evidence: " + text, 180)
    if target_role == "payoff":
        return _truncate("lands the short with the clearest payoff: " + text, 180)
    if target_role == "tension":
        return _truncate("creates the contrast that keeps the stitch moving: " + text, 180)
    if target_role == "button":
        return _truncate("turns the ending into a shareable final button: " + text, 180)
    return _truncate(str(arc.get("reason") or "selected by edit graph") + ": " + text, 180)


def _crop_hint_for_source_role(role: str) -> str:
    if role in {"hook", "quote"}:
        return "face_priority"
    if role in {"proof", "context"}:
        return "screen_or_center"
    return "center"


def _planned_operations_for_edit_arc(source_ranges: list[dict], duration: float) -> list[dict]:
    operations: list[dict] = []
    offset = 0.0
    for index, source_range in enumerate(source_ranges, start=1):
        start = offset
        end = offset + float(source_range.get("duration") or 0.0)
        if index > 1:
            operations.append(
                {
                    "type": "jump_cut",
                    "source_range_index": index,
                    "start_sec": round(start, 3),
                    "end_sec": round(start, 3),
                    "params": {
                        "transition": "hard_cut",
                        "reason": f"stitch into {source_range.get('role', 'part')} beat",
                    },
                }
            )
        if str(source_range.get("role") or "") in {"hook", "proof", "tension", "payoff"} and end - start >= 1.8:
            operations.append(
                {
                    "type": "punch_in",
                    "source_range_index": index,
                    "start_sec": round(start + 0.25, 3),
                    "end_sec": round(min(end, start + 2.6), 3),
                    "params": {
                        "zoom": 1.14,
                        "reason": f"emphasize {source_range.get('role')} beat",
                    },
                }
            )
        offset = end
    operations.append(
        {
            "type": "caption_emphasis",
            "start_sec": 0.0,
            "end_sec": round(duration, 3),
            "params": {"density": "fast", "reason": "keep stitched short readable"},
        }
    )
    return operations


def _build_remix_parts(
    segments: list[dict[str, float | str]],
    *,
    target_platform: str,
    video_context: dict[str, object] | None,
    max_part_duration: float,
) -> list[dict]:
    parts: list[dict] = []
    for start_index in range(len(segments)):
        text_parts: list[str] = []
        start_sec = float(segments[start_index]["start"])
        for end_index in range(start_index, min(len(segments), start_index + 4)):
            segment = segments[end_index]
            text_parts.append(str(segment["text"]).strip())
            end_sec = float(segment["end"])
            duration = end_sec - start_sec
            if duration > max_part_duration and end_index > start_index:
                break
            if duration < 2.6:
                continue
            text = " ".join(part for part in text_parts if part).strip()
            if len(_word_tokens(text)) < 5:
                continue
            score, breakdown, reasons = _score_transcript_window(
                text,
                duration,
                segments=segments,
                start_index=start_index,
                end_index=end_index,
                min_duration_sec=3.0,
                max_duration_sec=max(8.0, max_part_duration),
                target_platform=target_platform,
                video_context=video_context,
            )
            role = _classify_remix_part(text, breakdown)
            parts.append(
                {
                    "part_id": f"part_{start_index + 1:03d}_{end_index + 1:03d}",
                    "start": round(start_sec, 3),
                    "end": round(end_sec, 3),
                    "duration": round(duration, 3),
                    "text": _truncate(text, 260),
                    "role": role,
                    "score": round(score, 3),
                    "score_breakdown": breakdown,
                    "selection_reasons": reasons,
                    "keywords": _candidate_keywords(text, limit=8),
                    "unit_ids": [
                        str(item.get("unit_id"))
                        for item in segments[start_index : end_index + 1]
                        if str(item.get("unit_id") or "").strip()
                    ],
                }
            )
    parts.sort(
        key=lambda item: (
            float(item["score"]),
            float((item.get("score_breakdown") or {}).get("hook_strength", 0.0)),
            float((item.get("score_breakdown") or {}).get("payoff", 0.0)),
        ),
        reverse=True,
    )
    selected: list[dict] = []
    for part in parts:
        part_range = [{"start": part["start"], "end": part["end"]}]
        if all(_source_ranges_overlap_ratio(part_range, [{"start": existing["start"], "end": existing["end"]}]) < 0.72 for existing in selected):
            selected.append(part)
        if len(selected) >= 42:
            break
    return selected


def _classify_remix_part(text: str, breakdown: dict[str, float | int]) -> str:
    tokens = _word_tokens(text)
    if float(breakdown.get("payoff", 0.0)) >= 68 or _term_hits(tokens[-24:], PAYOFF_TERMS | EMPHASIS_TERMS) >= 1:
        return "payoff"
    if float(breakdown.get("hook_strength", 0.0)) >= 70 or _term_hits(tokens[:18], HOOK_TERMS | VIRAL_TERMS) >= 1:
        return "hook"
    if _term_hits(tokens, CONTRAST_TERMS) >= 1:
        return "tension"
    if _term_hits(tokens, PAYOFF_TERMS | EMPHASIS_TERMS) >= 1 or re.search(r"\b\d+(?:\.\d+)?%?\b", text):
        return "proof"
    if _term_hits(tokens[:18], SETUP_TERMS) >= 1:
        return "setup"
    return "support"


def _rank_remix_parts(parts: list[dict], roles: set[str]) -> list[dict]:
    return sorted(
        [part for part in parts if str(part.get("role")) in roles],
        key=lambda item: (
            float(item.get("score") or 0.0),
            float((item.get("score_breakdown") or {}).get("context_score", 0.0)),
        ),
        reverse=True,
    )


def _normalize_remix_part_sequence(parts: list[dict]) -> list[dict]:
    selected: list[dict] = []
    seen_ids: set[str] = set()
    for part in parts:
        part_id = str(part.get("part_id"))
        if part_id in seen_ids:
            continue
        selected.append(part)
        seen_ids.add(part_id)
    return selected


def _remix_parts_are_distinct(parts: list[dict]) -> bool:
    for index, first in enumerate(parts):
        first_range = [{"start": first["start"], "end": first["end"]}]
        for second in parts[index + 1:]:
            second_range = [{"start": second["start"], "end": second["end"]}]
            if _source_ranges_overlap_ratio(first_range, second_range) >= 0.32:
                return False
    return True


def _expand_remix_until_duration(
    selected_parts: list[dict],
    parts: list[dict],
    *,
    min_duration_sec: float,
    max_duration_sec: float,
) -> list[dict]:
    duration = sum(float(part["duration"]) for part in selected_parts)
    if duration >= min_duration_sec:
        return selected_parts
    expanded = list(selected_parts)
    preferred_roles = {"proof", "tension", "payoff", "support"}
    for part in parts:
        if str(part.get("role")) not in preferred_roles:
            continue
        if part in expanded:
            continue
        next_duration = duration + float(part["duration"])
        if next_duration > max_duration_sec:
            continue
        candidate_parts = [*expanded, part]
        if not _remix_parts_are_distinct(candidate_parts):
            continue
        expanded.append(part)
        duration = next_duration
        if duration >= min_duration_sec or len(expanded) >= 4:
            break
    return expanded


def _remix_strategy_label(parts: list[dict]) -> str:
    roles = [str(part.get("role") or "part") for part in parts]
    return " + ".join(_dedupe_text(roles)) or "stitched narrative"


def _apply_shorts_program_to_candidates(candidates: list[dict], program) -> None:
    plan_by_id = {plan.candidate_id: plan for plan in program.candidates}
    for candidate in candidates:
        plan = plan_by_id.get(str(candidate.get("candidate_id") or ""))
        if plan is None:
            continue
        breakdown = dict(candidate.get("score_breakdown") or {})
        original_score = float(candidate.get("heuristic_score") or breakdown.get("overall") or 1.0)
        director_score = float(plan.program_score)
        blended_score = _bounded((original_score * 0.72) + (director_score * 0.28), 1.0, 100.0)
        breakdown.update(
            {
                "director_score": round(director_score, 2),
                "arc_integrity": round(plan.arc_integrity, 2),
                "continuity_risk": round(plan.continuity_risk, 2),
                "topic_alignment": round(plan.topic_alignment, 2),
                "primary_role": plan.primary_role,
            }
        )
        candidate["heuristic_score"] = round(blended_score, 2)
        candidate["director_score"] = round(director_score, 2)
        candidate["director_plan"] = plan.to_dict()
        candidate["score_breakdown"] = breakdown
        reasons = list(candidate.get("selection_reasons") or [])
        if plan.quality_flags:
            reasons.append("director: " + ", ".join(plan.quality_flags[:3]))
        if plan.risk_flags:
            reasons.append("director risk: " + ", ".join(plan.risk_flags[:2]))
        candidate["selection_reasons"] = _dedupe_text(reasons)[:6]
    candidates.sort(
        key=lambda item: (
            float(item.get("heuristic_score") or 0.0),
            float(item.get("director_score") or 0.0),
            float((item.get("score_breakdown") or {}).get("hook_strength", 0.0)),
            float((item.get("score_breakdown") or {}).get("payoff", 0.0)),
        ),
        reverse=True,
    )


def _reconcile_selections_with_program(
    selections: list[dict],
    candidates: list[dict],
    program,
    count: int,
) -> list[dict]:
    selection_by_id = {str(selection.get("candidate_id") or ""): dict(selection) for selection in selections}
    candidate_by_id = {str(candidate.get("candidate_id") or ""): candidate for candidate in candidates}
    plan_by_id = {plan.candidate_id: plan for plan in program.candidates}
    reconciled: list[dict] = []
    for candidate_id in program.portfolio.selected_candidate_ids[:count]:
        candidate = candidate_by_id.get(candidate_id)
        if candidate is None:
            continue
        selection = selection_by_id.get(candidate_id) or _selection_from_candidate(candidate)
        plan = plan_by_id.get(candidate_id)
        if plan is not None:
            selection["score"] = round(
                _bounded((float(selection.get("score") or candidate.get("heuristic_score") or 1.0) * 0.68) + (plan.program_score * 0.32)),
                2,
            )
            selection["reason"] = _truncate(
                f"{selection.get('reason', '')} Director: {', '.join(program.portfolio.selection_reasons.get(candidate_id, [])[:3])}".strip(),
                260,
            )
        reconciled.append(selection)
    if len(reconciled) < count:
        used = {str(selection.get("candidate_id")) for selection in reconciled}
        reconciled.extend(_fallback_selections(candidates, count - len(reconciled), excluded_ids=used))
    return reconciled[:count]


def _edit_plan_dict(program, candidate_id: str) -> dict:
    edit_plan = program.edit_plans.get(candidate_id)
    return edit_plan.to_dict() if edit_plan is not None else {}


def _apply_edit_plan_to_punch_ins(moments: list[dict], edit_plan: dict) -> list[dict]:
    edit_plan = _normalize_short_edit_plan_fields(edit_plan)
    policy = dict(edit_plan.get("punch_in_policy") or {})
    if policy and not bool(policy.get("enabled", True)):
        return []
    max_moments = _coerce_int(policy.get("max_moments"), len(moments), low=0, high=8)
    if max_moments <= 0:
        return []
    min_gap = _coerce_float(policy.get("min_gap_sec"), 0.8, low=0.0, high=8.0)
    max_zoom = _coerce_float(policy.get("max_zoom"), 1.18, low=1.03, high=1.35)
    selected: list[dict] = []
    last_end = -999.0
    planned_moments = _punch_in_moments_from_edit_operations(edit_plan, max_zoom=max_zoom)
    for moment in sorted([*planned_moments, *moments], key=lambda item: _coerce_float(item.get("start"), 0.0)):
        start_sec = _coerce_float(moment.get("start"), 0.0, low=0.0, high=120.0)
        end_sec = _coerce_float(moment.get("end"), start_sec, low=start_sec, high=120.0)
        if start_sec - last_end < min_gap:
            continue
        adjusted = dict(moment)
        adjusted["zoom"] = round(
            min(
                _coerce_float(
                    moment.get("zoom"),
                    1.12,
                    low=1.03,
                    high=1.35,
                    aliases={"low": 1.06, "medium": 1.12, "high": 1.18},
                ),
                max_zoom,
            ),
            2,
        )
        selected.append(adjusted)
        last_end = end_sec
        if len(selected) >= max_moments:
            break
    return selected


def _apply_edit_plan_to_b_roll(suggestions: list[dict], edit_plan: dict) -> list[dict]:
    edit_plan = _normalize_short_edit_plan_fields(edit_plan)
    policy = dict(edit_plan.get("visual_insert_policy") or {})
    if policy and not bool(policy.get("enabled", True)):
        return []
    max_inserts = _coerce_int(policy.get("max_inserts"), len(suggestions), low=0, high=8)
    if max_inserts <= 0:
        return []
    planned_visuals = _visual_suggestions_from_edit_operations(edit_plan)
    return [*planned_visuals, *suggestions][:max_inserts]


def _punch_in_moments_from_edit_operations(edit_plan: dict, *, max_zoom: float) -> list[dict]:
    moments: list[dict] = []
    for operation in edit_plan.get("operations") or []:
        if not isinstance(operation, dict) or operation.get("type") != "punch_in":
            continue
        try:
            start_sec = max(0.0, float(operation.get("start_sec", 0.0)))
            end_sec = max(start_sec, float(operation.get("end_sec", start_sec)))
        except (TypeError, ValueError):
            continue
        if end_sec <= start_sec:
            continue
        params = dict(operation.get("params") or {})
        moments.append(
            {
                "start": round(start_sec, 3),
                "end": round(end_sec, 3),
                "zoom": round(
                    min(
                        _coerce_float(
                            params.get("zoom"),
                            1.12,
                            low=1.03,
                            high=1.35,
                            aliases={"low": 1.06, "medium": 1.12, "high": 1.18},
                        ),
                        max_zoom,
                    ),
                    2,
                ),
                "reason": _truncate(str(params.get("reason") or "edit-plan emphasis"), 140),
                "source": "shorts_edit_plan",
            }
        )
    return moments


def _visual_suggestions_from_edit_operations(edit_plan: dict) -> list[dict]:
    suggestions: list[dict] = []
    source_ranges = {
        int(source_range.get("index") or index): source_range
        for index, source_range in enumerate(edit_plan.get("source_ranges") or [], start=1)
        if isinstance(source_range, dict)
    }
    for operation in edit_plan.get("operations") or []:
        if not isinstance(operation, dict) or operation.get("type") != "auto_visual":
            continue
        try:
            start_sec = max(0.0, float(operation.get("start_sec", 0.0)))
            end_sec = max(start_sec, float(operation.get("end_sec", start_sec)))
        except (TypeError, ValueError):
            continue
        if end_sec <= start_sec:
            continue
        params = dict(operation.get("params") or {})
        source_range = source_ranges.get(int(operation.get("source_range_index") or 0), {})
        role = str(source_range.get("role") or "support")
        suggestions.append(
            {
                "start": round(start_sec, 2),
                "end": round(end_sec, 2),
                "visual_type": str((params.get("preferred_types") or [f"{role}_visual"])[0]),
                "search_query": _truncate(f"{role} visual support", 80),
                "direction": _truncate(str(params.get("reason") or "Use a contextual insert that supports the edit-plan beat."), 180),
                "source": "shorts_edit_plan",
            }
        )
    return suggestions


def _dedupe_text(items: list[str]) -> list[str]:
    deduped: list[str] = []
    for item in items:
        text = str(item).strip()
        if text and text not in deduped:
            deduped.append(text)
    return deduped


def _format_candidates_for_llm(candidates: list[dict]) -> str:
    lines: list[str] = []
    for candidate in candidates:
        source_ranges = _candidate_source_ranges(candidate)
        composition_mode = str(candidate.get("composition_mode") or ("remix" if len(source_ranges) > 1 else "single_window"))
        lines.append(
            "\n".join(
                [
                    (
                        f"{candidate['candidate_id']} | {candidate['duration']:.2f}s | "
                        f"composition={composition_mode} | source_ranges={len(source_ranges)}"
                    ),
                    f"Final assembled transcript: {candidate['excerpt']}",
                ]
            )
        )
    return "\n\n".join(lines)


def _call_reasoning_model(provider_name: str, model_name: str, system_prompt: str, user_prompt: str) -> str:
    if provider_name == "claude":
        from anthropic import Anthropic

        client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=model_name or config.CLAUDE_MODEL,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(block.text for block in response.content if getattr(block, "type", "") == "text")

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    response = client.models.generate_content(
        model=model_name or config.GEMINI_MODEL,
        contents=user_prompt,
        config=config.build_gemini_generation_config(
            system_prompt,
            model_name=model_name or config.GEMINI_MODEL,
        ),
    )
    return getattr(response, "text", "") or ""


def _plan_story_candidates_with_llm(
    *,
    provider_name: str,
    model_name: str,
    semantic_units: list[dict],
    count: int,
    min_duration_sec: float,
    max_duration_sec: float,
    target_platform: str,
    video_context: dict[str, object] | None = None,
) -> tuple[list[dict], dict]:
    chapters = build_story_chapters(semantic_units)
    if len(chapters) > 24:
        total_duration = max(
            1.0,
            float(semantic_units[-1]["end"]) - float(semantic_units[0]["start"]),
        )
        chapters = build_story_chapters(
            semantic_units,
            max_duration_sec=max(150.0, total_duration / 22.0),
            max_units=max(32, (len(semantic_units) + 21) // 22),
            overlap_units=1,
        )
    provenance: dict[str, object] = {
        "version": "shorts-story-planner-v1",
        "provider": provider_name,
        "model": model_name,
        "status": "unavailable",
        "chapter_count": len(chapters),
        "attempts": [],
        "accepted_candidate_ids": [],
        "rejected_proposals": [],
    }
    if not chapters:
        provenance["status"] = "skipped"
        provenance["reason"] = "no semantic transcript chapters"
        return [], provenance

    system_prompt = (
        "You are the top-down story editor for a production short-form video system. "
        "Read every timestamped semantic subtitle unit in the chapter before proposing clips. "
        "Find complete standalone stories, not isolated spicy lines. Each story must give a cold viewer "
        "a clear subject, a hook or tension, the minimum required explanation, and a real payoff. "
        "Prefer one contiguous span. Use multiple source_ranges only when the later range directly completes "
        "the same causal story and every range starts and ends on a complete thought. "
        "Never invent timestamps or transcript text. Reference only the exact unit_id values supplied. "
        "Return ONLY a JSON array. Each object must contain title, hook, reason, confidence, keywords, "
        "and either unit_ids or source_ranges. source_ranges must contain role, reason, and unit_ids."
    )
    planned: list[dict] = []
    candidate_index = 1
    per_chapter = max(2, min(6, count))
    for chapter in chapters:
        chapter_units = list(chapter.get("units") or [])
        attempt: dict[str, object] = {
            "chapter_id": chapter.get("chapter_id"),
            "start": chapter.get("start"),
            "end": chapter.get("end"),
            "unit_count": len(chapter_units),
            "status": "error",
        }
        user_prompt = (
            f"Platform: {target_platform}\n"
            f"Target duration: {min_duration_sec:.1f}-{max_duration_sec:.1f} seconds.\n"
            f"Propose up to {per_chapter} publishable story candidates from this chapter.\n\n"
            "Full-video thesis and topic:\n"
            f"Thesis: {_truncate(str((video_context or {}).get('thesis_excerpt') or ''), 700)}\n"
            f"Core keywords: {', '.join(str(item) for item in (video_context or {}).get('core_keywords', [])[:20])}\n"
            f"Main keywords: {', '.join(str(item) for item in (video_context or {}).get('main_keywords', [])[:20])}\n\n"
            f"Chapter {chapter.get('chapter_id')} semantic transcript:\n"
            f"{format_units_for_planner(chapter_units)}\n\n"
            "Reject ideas that begin mid-explanation, depend on unseen prior context, merely repeat a premise, "
            "or end before the answer. Confidence is 1-100 and must reflect cold-viewer comprehension, not hype."
        )
        try:
            raw_text = _call_reasoning_model(
                provider_name,
                model_name,
                system_prompt,
                user_prompt,
            )
            parsed = json.loads(_extract_json_array(raw_text))
            attempt["status"] = "completed"
            attempt["proposal_count"] = len(parsed)
        except Exception as exc:
            attempt["error"] = _truncate(f"{type(exc).__name__}: {exc}", 500)
            provenance["attempts"].append(attempt)
            continue

        accepted_in_chapter = 0
        rejected_in_chapter = 0
        for proposal in parsed[: per_chapter * 2]:
            if not isinstance(proposal, dict):
                continue
            compiled = compile_story_proposal(
                proposal,
                chapter_units,
                candidate_id=f"story_{candidate_index:03d}",
                min_duration_sec=min_duration_sec,
                max_duration_sec=max_duration_sec,
            )
            if not compiled["passed"]:
                rejected_in_chapter += 1
                provenance["rejected_proposals"].append(
                    {
                        "chapter_id": chapter.get("chapter_id"),
                        "title": _truncate(str(proposal.get("title") or "Untitled proposal"), 72),
                        "errors": list(compiled["errors"]),
                    }
                )
                continue
            candidate = dict(compiled["candidate"])
            transcript_text = str(candidate.get("excerpt") or "")
            score, breakdown, reasons = _score_transcript_window(
                transcript_text,
                float(candidate["duration"]),
                min_duration_sec=min_duration_sec,
                max_duration_sec=max_duration_sec,
                target_platform=target_platform,
                video_context=video_context,
            )
            story_plan = dict(candidate.get("story_plan") or {})
            critic = dict(story_plan.get("critic") or {})
            planner_confidence = float(story_plan.get("planner_confidence") or 50.0)
            critic_score = float(critic.get("score") or 1.0)
            stitch_count = max(len(candidate.get("source_ranges") or []) - 1, 0)
            candidate["heuristic_score"] = round(
                _bounded(
                    score * 0.62
                    + critic_score * 0.28
                    + planner_confidence * 0.10
                    - stitch_count * 3.0,
                    1.0,
                    100.0,
                ),
                2,
            )
            breakdown.update(
                {
                    "story_critic_score": round(critic_score, 2),
                    "planner_confidence": round(planner_confidence, 2),
                    "story_compiler_stitch_count": stitch_count,
                }
            )
            candidate["score_breakdown"] = breakdown
            candidate["selection_reasons"] = _dedupe_text(
                [
                    "hierarchical story planner selected a complete subtitle arc",
                    *reasons,
                    *[str(item) for item in critic.get("warnings", [])],
                ]
            )[:7]
            candidate["candidate_origin"] = "hierarchical_story_planner"
            planned.append(candidate)
            candidate_index += 1
            accepted_in_chapter += 1
        attempt["accepted_count"] = accepted_in_chapter
        attempt["rejected_count"] = rejected_in_chapter
        provenance["attempts"].append(attempt)

    planned.sort(
        key=lambda item: (
            float(item.get("heuristic_score") or 0.0),
            float((item.get("score_breakdown") or {}).get("story_critic_score", 0.0)),
            -len(item.get("source_ranges") or []),
        ),
        reverse=True,
    )
    planned = _dedupe_candidates(planned, limit=max(count * 3, count))
    provenance["accepted_candidate_ids"] = [
        str(candidate.get("candidate_id")) for candidate in planned
    ]
    provenance["accepted_count"] = len(planned)
    provenance["rejected_count"] = len(provenance["rejected_proposals"])
    if planned:
        provenance["status"] = "completed"
    elif any(attempt.get("status") == "completed" for attempt in provenance["attempts"]):
        provenance["status"] = "completed_no_valid_candidates"
    return planned, provenance


def _default_title(candidate: dict) -> str:
    words = _word_tokens(candidate["excerpt"])
    if not words:
        return "High-signal short"
    return _truncate(" ".join(words[:8]).title(), 60)


def _default_hook(candidate: dict) -> str:
    sentence = candidate["excerpt"].split(".", 1)[0].strip()
    if len(sentence) >= 12:
        return _truncate(sentence, 90)
    return _truncate(candidate["excerpt"], 90)


def _candidate_keyword_set(candidate: dict) -> set[str]:
    keywords = candidate.get("keywords") or _candidate_keywords(str(candidate.get("excerpt") or ""))
    return {str(keyword).lower() for keyword in keywords if str(keyword).strip()}


def _topic_similarity(first: dict, second: dict) -> float:
    first_keywords = _candidate_keyword_set(first)
    second_keywords = _candidate_keyword_set(second)
    if not first_keywords or not second_keywords:
        return 0.0
    return len(first_keywords & second_keywords) / max(len(first_keywords | second_keywords), 1)


def _select_diverse_candidates(
    candidates: list[dict],
    count: int,
    *,
    excluded_ids: set[str] | None = None,
    seed_candidates: list[dict] | None = None,
) -> list[dict]:
    excluded_ids = excluded_ids or set()
    selected: list[dict] = list(seed_candidates or [])
    additions: list[dict] = []
    for candidate in candidates:
        if candidate["candidate_id"] in excluded_ids:
            continue
        if not _candidate_preselection_eligible(candidate):
            continue
        if all(_overlap_ratio(candidate, existing) < 0.52 and _topic_similarity(candidate, existing) < 0.58 for existing in selected):
            selected.append(candidate)
            additions.append(candidate)
        if len(additions) >= count:
            return additions
    for candidate in candidates:
        if candidate["candidate_id"] in excluded_ids or candidate in selected or candidate in additions:
            continue
        if not _candidate_preselection_eligible(candidate):
            continue
        additions.append(candidate)
        if len(additions) >= count:
            break
    return additions


def _candidate_preselection_eligible(candidate: dict) -> bool:
    story_plan = candidate.get("story_plan")
    if isinstance(story_plan, dict):
        critic = story_plan.get("critic")
        if isinstance(critic, dict) and not bool(critic.get("passed", False)):
            return False
    breakdown = dict(candidate.get("score_breakdown") or {})
    if float(breakdown.get("abrupt_start_penalty", 0.0)) >= 22.0:
        return False
    if float(breakdown.get("dangling_payoff_penalty", 0.0)) >= 22.0:
        return False
    standalone = float(breakdown.get("standalone_clarity", 100.0))
    story = float(breakdown.get("story_completeness", 100.0))
    if standalone < 45.0 and story < 45.0:
        return False
    source_ranges = _candidate_source_ranges(candidate)
    continuity_risk = float(breakdown.get("continuity_risk", 0.0))
    if len(source_ranges) > 1 and continuity_risk >= 58.0:
        return False
    return True


def _selection_from_candidate(candidate: dict) -> dict:
    reasons = candidate.get("selection_reasons") or ["Strong deterministic transcript score."]
    story_plan = candidate.get("story_plan") if isinstance(candidate.get("story_plan"), dict) else {}
    fallback_source = (
        "story_planner_fallback"
        if candidate.get("candidate_origin") == "hierarchical_story_planner"
        else "deterministic_fallback"
    )
    return {
        "candidate_id": candidate["candidate_id"],
        "score": round(float(candidate["heuristic_score"]), 2),
        "title": _truncate(str(story_plan.get("title") or _default_title(candidate)), 72),
        "hook": _truncate(str(story_plan.get("hook") or _default_hook(candidate)), 120),
        "reason": _truncate(
            str(story_plan.get("reason") or "; ".join(str(reason) for reason in reasons)),
            220,
        ),
        "keywords": list(candidate.get("keywords") or _word_tokens(candidate["excerpt"])[:5])[:6],
        "selection_source": fallback_source,
    }


def _fallback_selections(
    candidates: list[dict],
    count: int,
    *,
    excluded_ids: set[str] | None = None,
    seed_candidates: list[dict] | None = None,
) -> list[dict]:
    selections: list[dict] = []
    for candidate in _select_diverse_candidates(
        candidates,
        count,
        excluded_ids=excluded_ids,
        seed_candidates=seed_candidates,
    ):
        selections.append(_selection_from_candidate(candidate))
    return selections


def _select_shorts_with_llm(
    provider_name: str,
    model_name: str,
    candidates: list[dict],
    transcript_text: str,
    count: int,
    min_duration_sec: float,
    max_duration_sec: float,
    target_platform: str,
    video_context: dict[str, object] | None = None,
    provenance: dict[str, object] | None = None,
) -> list[dict]:
    candidate_map = {candidate["candidate_id"]: candidate for candidate in candidates}
    model_candidates = candidates[: min(len(candidates), max(30, count * 4))]
    system_prompt = (
        "You are an independent cold-viewer critic. Choose the strongest publishable shorts from final assembled transcripts. "
        "Prioritize the clips most likely to retain a cold viewer: a fast first-line hook, concrete specificity, tension or contrast, "
        "a satisfying payoff before the end, full-video thesis alignment, and a clean standalone idea. "
        "Judge only what a viewer would hear. Do not infer quality from intended story roles or hidden edit-plan scores. "
        "A multi-range candidate must sound causally continuous despite the cuts. "
        "Diversify topics and reject near-duplicates, misleading fragments, abrupt starts, and clips that only make sense with prior context. "
        "Choose only candidate_id values from the provided candidates. "
        "Return ONLY a JSON array of objects with keys: candidate_id, score, title, hook, reason, keywords."
    )
    user_prompt = (
        f"Target platform: {target_platform}.\n"
        f"Need exactly {count} shorts. Each clip should stay between {min_duration_sec} and {max_duration_sec} seconds.\n\n"
        "Full-video context:\n"
        f"Thesis/opening: {_truncate(str((video_context or {}).get('thesis_excerpt') or ''), 650)}\n"
        f"Core repeated/opening/closing keywords: {', '.join(str(item) for item in (video_context or {}).get('core_keywords', [])[:18])}\n"
        f"Main keywords: {', '.join(str(item) for item in (video_context or {}).get('main_keywords', [])[:18])}\n"
        f"Main phrases: {', '.join(str(item) for item in (video_context or {}).get('main_phrases', [])[:10])}\n\n"
        f"Transcript overview:\n{_truncate(transcript_text, 3500)}\n\n"
        f"Blind candidate transcripts:\n{_format_candidates_for_llm(model_candidates)}\n\n"
        "Choose the best candidates for viral-style shorts based on the assembled transcript itself. "
        "Do not choose a clip just because it sounds spicy; it must be self-contained and faithful to what the full video is about. "
        "Keep titles punchy, hooks conversational, reasons concrete, and keywords platform-friendly. "
        "Return JSON array only."
    )
    raw_text = _call_reasoning_model(provider_name, model_name, system_prompt, user_prompt)
    parsed = json.loads(_extract_json_array(raw_text))
    selections: list[dict] = []
    seen_ids: set[str] = set()
    for item in parsed:
        candidate_id = str(item.get("candidate_id", "")).strip()
        if candidate_id not in candidate_map or candidate_id in seen_ids:
            continue
        seen_ids.add(candidate_id)
        candidate = candidate_map[candidate_id]
        if not _candidate_preselection_eligible(candidate):
            continue
        keywords = [str(keyword).strip() for keyword in item.get("keywords", []) if str(keyword).strip()]
        model_score = _clamp_score(item.get("score", candidate["heuristic_score"]))
        deterministic_score = float(candidate["heuristic_score"])
        blended_score = round((model_score * 0.58) + (deterministic_score * 0.42), 2)
        selections.append(
            {
                "candidate_id": candidate_id,
                "score": blended_score,
                "title": _truncate(str(item.get("title") or _default_title(candidate)), 72),
                "hook": _truncate(str(item.get("hook") or _default_hook(candidate)), 120),
                "reason": _truncate(str(item.get("reason") or "Strong transcript hook and payoff."), 220),
                "keywords": (keywords or list(candidate.get("keywords") or []))[:6],
                "selection_source": "model_candidate_tournament",
            }
        )
    selections.sort(key=lambda item: float(item["score"]), reverse=True)
    diverse: list[dict] = []
    diverse_candidates: list[dict] = []
    for selection in selections:
        candidate = candidate_map[selection["candidate_id"]]
        if all(_overlap_ratio(candidate, existing) < 0.52 and _topic_similarity(candidate, existing) < 0.62 for existing in diverse_candidates):
            diverse.append(selection)
            diverse_candidates.append(candidate)
    if len(diverse) >= count:
        if provenance is not None:
            provenance.update(
                {
                    "status": "completed",
                    "provider": provider_name,
                    "model": model_name,
                    "requested_count": count,
                    "model_selection_count": len(selections),
                    "returned_count": len(diverse[:count]),
                }
            )
        return diverse[:count]
    excluded_ids = {selection["candidate_id"] for selection in diverse}
    diverse.extend(
        _fallback_selections(
            candidates,
            count - len(diverse),
            excluded_ids=excluded_ids,
            seed_candidates=diverse_candidates,
        )
    )
    result = diverse[:count]
    if provenance is not None:
        provenance.update(
            {
                "status": "completed_with_deterministic_backfill",
                "provider": provider_name,
                "model": model_name,
                "requested_count": count,
                "model_selection_count": len(selections),
                "returned_count": len(result),
            }
        )
    return result


def _analyze_viral_score_with_llm(
    provider_name: str,
    model_name: str,
    candidate: dict,
    selection: dict,
    clip_segments: list[dict[str, float | str]],
    target_platform: str,
    video_context: dict[str, object] | None = None,
) -> dict:
    fallback = _fallback_viral_analysis(candidate, selection, clip_segments)
    transcript_text = _clip_transcript_text(clip_segments) or str(candidate.get("excerpt") or "")
    system_prompt = (
        "You are a short-form video analyst. Score a short clip for short-form virality with concrete explainability. "
        "Return ONLY a JSON object with keys viral_score and viral_explanation. "
        "viral_score must contain overall, hook_strength, payoff, novelty, clarity, and shareability as integers from 1 to 100. "
        "viral_explanation must be an array of 3 or 4 concise bullet-style strings."
    )
    user_prompt = (
        f"Platform: {target_platform}\n"
        f"Candidate window: {candidate['start']:.2f}-{candidate['end']:.2f} ({candidate['duration']:.2f}s)\n"
        f"Draft title: {selection.get('title', '')}\n"
        f"Draft hook: {selection.get('hook', '')}\n"
        f"Why selected: {selection.get('reason', '')}\n\n"
        "Full-video context:\n"
        f"Thesis/opening: {_truncate(str((video_context or {}).get('thesis_excerpt') or ''), 500)}\n"
        f"Core keywords: {', '.join(str(item) for item in (video_context or {}).get('core_keywords', [])[:12])}\n"
        f"Main keywords: {', '.join(str(item) for item in (video_context or {}).get('main_keywords', [])[:14])}\n\n"
        f"Transcript:\n{_truncate(transcript_text, 2400)}\n\n"
        "Score only self-contained clips that are faithful to the full video's topic. Return JSON only."
    )
    try:
        raw_text = _call_reasoning_model(provider_name, model_name, system_prompt, user_prompt)
        parsed = json.loads(_extract_json_object(raw_text))
    except Exception:
        return fallback
    return _normalize_viral_analysis(parsed, fallback)


def _clip_transcript_segments(
    segments: list[dict[str, float | str]],
    start_sec: float,
    end_sec: float,
) -> list[dict[str, float | str]]:
    clipped: list[dict[str, float | str]] = []
    for segment in segments:
        segment_start = float(segment["start"])
        segment_end = float(segment["end"])
        if segment_end <= start_sec or segment_start >= end_sec:
            continue
        clipped.append(
            {
                "start": round(max(segment_start, start_sec) - start_sec, 3),
                "end": round(min(segment_end, end_sec) - start_sec, 3),
                "text": str(segment["text"]).strip(),
            }
        )
    return [segment for segment in clipped if float(segment["end"]) > float(segment["start"])]


def _clip_transcript_segments_for_ranges(
    segments: list[dict[str, float | str]],
    source_ranges: list[dict],
) -> list[dict[str, float | str]]:
    stitched: list[dict[str, float | str]] = []
    timeline_offset = 0.0
    for source_range in source_ranges:
        source_start = float(source_range["start"])
        source_end = float(source_range["end"])
        range_index = int(source_range.get("index") or len(stitched) + 1)
        range_role = str(source_range.get("role") or "part")
        clipped = _clip_transcript_segments(segments, source_start, source_end)
        for segment in clipped:
            local_start = float(segment["start"])
            local_end = float(segment["end"])
            stitched.append(
                {
                    "start": round(timeline_offset + local_start, 3),
                    "end": round(timeline_offset + local_end, 3),
                    "text": str(segment["text"]).strip(),
                    "source_start": round(source_start + local_start, 3),
                    "source_end": round(source_start + local_end, 3),
                    "source_range_index": range_index,
                    "source_role": range_role,
                }
            )
        timeline_offset += max(0.0, source_end - source_start)
    return [segment for segment in stitched if float(segment["end"]) > float(segment["start"])]


def _render_candidate_raw_clip(state: ProjectState, candidate: dict, short_dir: Path) -> tuple[Path, list[dict]]:
    source_ranges = _candidate_source_ranges(candidate)
    if not source_ranges:
        raise ValueError("Candidate has no usable source ranges.")
    source_parts: list[dict] = []
    if len(source_ranges) == 1:
        source_range = source_ranges[0]
        raw_temp_path = trim(
            state.working_file,
            state.working_dir,
            float(source_range["start"]),
            float(source_range["end"]),
        )
        raw_clip_path = short_dir / "raw_clip.mp4"
        shutil.copy2(raw_temp_path, raw_clip_path)
        source_parts.append({**source_range, "path": str(raw_clip_path)})
        return raw_clip_path, source_parts

    part_paths: list[str] = []
    for index, source_range in enumerate(source_ranges, start=1):
        raw_part_path = trim(
            state.working_file,
            state.working_dir,
            float(source_range["start"]),
            float(source_range["end"]),
        )
        part_path = short_dir / f"source_part_{index:02d}.mp4"
        shutil.copy2(raw_part_path, part_path)
        part_paths.append(str(part_path))
        source_parts.append({**source_range, "path": str(part_path)})
    merged_temp_path = merge(part_paths, state.working_dir)
    raw_clip_path = short_dir / "raw_clip.mp4"
    shutil.copy2(merged_temp_path, raw_clip_path)
    return raw_clip_path, source_parts


def _fallback_short_quality_gate(
    candidate: dict,
    selection: dict,
    clip_segments: list[dict[str, float | str]],
    short_record: dict,
    video_context: dict[str, object] | None,
) -> dict:
    transcript_text = _clip_transcript_text(clip_segments) or str(candidate.get("excerpt") or "")
    duration = float(short_record.get("duration") or candidate.get("duration") or 0.0)
    score, breakdown, reasons = _score_transcript_window(
        transcript_text,
        duration,
        min_duration_sec=max(3.0, min(duration, 12.0)),
        max_duration_sec=max(duration, 12.0),
        target_platform=str(short_record.get("target_platform") or "youtube_shorts"),
        video_context=video_context,
    )
    tokens = _word_tokens(transcript_text)
    first_token = tokens[0] if tokens else ""
    source_ranges = list(short_record.get("source_ranges") or _candidate_source_ranges(candidate))
    source_range_count = len(source_ranges)
    abrupt_penalty = float(breakdown.get("abrupt_start_penalty", 0.0))
    dependency_penalty = float(breakdown.get("context_dependency_penalty", 0.0))
    payoff = float(breakdown.get("payoff", 0.0))
    standalone = float(breakdown.get("standalone_clarity", breakdown.get("clarity", 0.0)))
    story = float(breakdown.get("story_completeness", 0.0))
    topic_fit = float(breakdown.get("context_score", score))
    stitch_penalty = max(0, source_range_count - 1) * 4.0
    if source_range_count > 1:
        source_roles = {str(item.get("role") or "") for item in source_ranges}
        if "hook" not in source_roles and "quote" not in source_roles:
            stitch_penalty += 7.0
        if "payoff" not in source_roles and "proof" not in source_roles:
            stitch_penalty += 9.0
    final_score = _bounded(
        score * 0.32
        + standalone * 0.22
        + story * 0.18
        + payoff * 0.14
        + topic_fit * 0.14
        - abrupt_penalty * 0.35
        - dependency_penalty * 0.28
        - stitch_penalty,
        1.0,
        100.0,
    )
    fatal_reasons: list[str] = []
    if not transcript_text.strip():
        fatal_reasons.append("missing transcript")
    if len(tokens) < 8:
        fatal_reasons.append("too little spoken context")
    if first_token in CONTEXT_DEPENDENT_STARTERS and standalone < 58:
        fatal_reasons.append("abrupt context-dependent opener")
    if payoff < 42 and story < 52:
        fatal_reasons.append("weak payoff")
    if source_range_count > 4:
        fatal_reasons.append("too many stitched source ranges")
    passed = final_score >= 56.0 and not fatal_reasons
    verdict = "approved" if passed else "rejected"
    rejection_reason = "; ".join(fatal_reasons) if fatal_reasons else ("quality score below release threshold" if not passed else "")
    return {
        "passed": passed,
        "score": round(final_score, 2),
        "verdict": verdict,
        "abruptness": round(_bounded(100.0 - abrupt_penalty - dependency_penalty), 2),
        "standalone": round(standalone, 2),
        "payoff": round(payoff, 2),
        "topic_fit": round(topic_fit, 2),
        "stitch_continuity": round(_bounded(100.0 - stitch_penalty), 2),
        "rejection_reason": rejection_reason,
        "reasons": _dedupe_text([*fatal_reasons, *reasons])[:5],
        "model_used": "deterministic_fallback",
    }


def _preflight_short_edit_plan(
    *,
    candidate: dict,
    selection: dict,
    clip_segments: list[dict[str, float | str]],
    edit_plan: dict,
    target_platform: str,
    video_context: dict[str, object] | None,
) -> dict:
    edit_plan = _normalize_short_edit_plan_fields(edit_plan, candidate)
    plan_validation = validate_short_edit_plan(edit_plan)
    source_ranges = _candidate_source_ranges(candidate)
    quality_probe = _fallback_short_quality_gate(
        candidate,
        selection,
        clip_segments,
        {
            "duration": candidate.get("duration"),
            "source_ranges": source_ranges,
            "target_platform": target_platform,
        },
        video_context,
    )
    quality_floor = _coerce_float(edit_plan.get("quality_floor"), 56.0, low=50.0, high=90.0)
    errors = list(plan_validation.get("errors") or [])
    warnings = list(plan_validation.get("warnings") or [])
    story_plan = candidate.get("story_plan")
    story_critic = (
        dict(story_plan.get("critic") or {})
        if isinstance(story_plan, dict)
        else {}
    )
    if story_critic and not bool(story_critic.get("passed", False)):
        errors.extend(
            f"story compiler: {error}"
            for error in story_critic.get("errors", [])
            if str(error).strip()
        )
        if not story_critic.get("errors"):
            errors.append("story compiler rejected the candidate")
    warnings.extend(
        f"story compiler: {warning}"
        for warning in story_critic.get("warnings", [])
        if str(warning).strip()
    )
    if float(quality_probe.get("score") or 0.0) < quality_floor:
        errors.append(
            f"pre-render transcript quality {quality_probe.get('score')} is below floor {quality_floor:.0f}"
        )
    if not bool(quality_probe.get("passed")):
        errors.append(str(quality_probe.get("rejection_reason") or "pre-render transcript gate rejected the edit"))
    if len(source_ranges) > 1:
        roles = [str(source_range.get("role") or "") for source_range in source_ranges]
        if roles[0] not in {"hook", "quote", "setup", "context"}:
            errors.append("stitched edit opens without hook/context role")
        if not any(role in {"proof", "payoff", "button", "quote"} for role in roles):
            errors.append("stitched edit has no proof/payoff role")
    return {
        "passed": not errors,
        "errors": _dedupe_text(errors),
        "warnings": _dedupe_text(warnings),
        "quality_probe": quality_probe,
        "story_critic": story_critic,
        "plan_validation": plan_validation,
        "version": "shorts-edit-preflight-v2",
    }


def _normalize_short_quality_gate(raw_gate: dict, fallback: dict, model_name: str) -> dict:
    score = _clamp_score(raw_gate.get("score", fallback["score"]))
    passed = bool(raw_gate.get("passed", fallback["passed"])) and score >= 56
    fallback_reasons_raw = fallback.get("reasons", [])
    if isinstance(fallback_reasons_raw, str):
        fallback_reasons = [fallback_reasons_raw]
    elif isinstance(fallback_reasons_raw, list):
        fallback_reasons = fallback_reasons_raw
    else:
        fallback_reasons = []
    raw_reasons = raw_gate.get("reasons", fallback_reasons)
    if isinstance(raw_reasons, str):
        raw_reasons = [raw_reasons]
    elif not isinstance(raw_reasons, list):
        raw_reasons = []
    reasons = [
        _truncate(str(reason).strip(), 150)
        for reason in raw_reasons
        if str(reason).strip()
    ]
    rejection_reason = _truncate(
        str(raw_gate.get("rejection_reason") or fallback.get("rejection_reason") or ""),
        220,
    )
    if not passed and not rejection_reason:
        rejection_reason = "quality gate rejected the short"
    return {
        "passed": passed,
        "score": score,
        "verdict": "approved" if passed else "rejected",
        "abruptness": _clamp_score(raw_gate.get("abruptness", fallback.get("abruptness", 1))),
        "standalone": _clamp_score(raw_gate.get("standalone", fallback.get("standalone", 1))),
        "payoff": _clamp_score(raw_gate.get("payoff", fallback.get("payoff", 1))),
        "topic_fit": _clamp_score(raw_gate.get("topic_fit", fallback.get("topic_fit", 1))),
        "stitch_continuity": _clamp_score(raw_gate.get("stitch_continuity", fallback.get("stitch_continuity", 1))),
        "rejection_reason": rejection_reason,
        "reasons": (reasons or fallback_reasons)[:5],
        "model_used": model_name,
    }


def _quality_gate_with_llm(
    provider_name: str,
    model_name: str,
    candidate: dict,
    selection: dict,
    clip_segments: list[dict[str, float | str]],
    short_record: dict,
    target_platform: str,
    video_context: dict[str, object] | None = None,
) -> dict:
    fallback = _fallback_short_quality_gate(candidate, selection, clip_segments, short_record, video_context)
    transcript_text = _clip_transcript_text(clip_segments) or str(candidate.get("excerpt") or "")
    timestamped_transcript = _format_timestamped_clip_segments(clip_segments)
    source_map = ", ".join(
        f"{item.get('role', 'part')}:{float(item['start']):.2f}-{float(item['end']):.2f}"
        for item in _candidate_source_ranges(candidate)
    )
    system_prompt = (
        "You are a strict short-form release QA editor. Decide whether the rendered short is publishable. "
        "Reject abrupt fragments, stitched clips that feel incoherent, missing payoff, misleading topic cuts, and shorts that need prior context. "
        "Return ONLY a JSON object with keys passed, score, abruptness, standalone, payoff, topic_fit, stitch_continuity, verdict, reasons, rejection_reason. "
        "Scores must be integers from 1 to 100."
    )
    user_prompt = (
        f"Platform: {target_platform}\n"
        f"Title: {selection.get('title', '')}\n"
        f"Hook: {selection.get('hook', '')}\n"
        f"Source edit map: {source_map or 'single window'}\n"
        f"Duration: {short_record.get('duration')} seconds\n\n"
        "Full-video context:\n"
        f"Thesis/opening: {_truncate(str((video_context or {}).get('thesis_excerpt') or ''), 650)}\n"
        f"Core keywords: {', '.join(str(item) for item in (video_context or {}).get('core_keywords', [])[:16])}\n\n"
        "Final stitched transcript:\n"
        f"{_truncate(transcript_text, 2600)}\n\n"
        "Timestamped transcript:\n"
        f"{_truncate(timestamped_transcript, 3200)}\n\n"
        "Approve only if the short feels intentional, self-contained, and has a clean close. Return JSON only."
    )
    try:
        raw_text = _call_reasoning_model(provider_name, model_name, system_prompt, user_prompt)
        parsed = json.loads(_extract_json_object(raw_text))
    except Exception:
        return fallback
    return _normalize_short_quality_gate(parsed, fallback, model_name)


def _short_rejection_reasons(render_validation: dict, quality_gate: dict) -> list[str]:
    reasons = [str(error) for error in render_validation.get("errors", []) if str(error).strip()]
    if not quality_gate.get("passed", False):
        gate_reason = str(quality_gate.get("rejection_reason") or "").strip()
        if not gate_reason:
            gate_reason = "; ".join(str(item) for item in quality_gate.get("reasons", []) if str(item).strip())
        reasons.append(f"quality gate: {gate_reason or 'quality gate rejected the short'}")
    return reasons or ["short was rejected by release QA"]


def _hashtags(keywords: list[str], target_platform: str) -> list[str]:
    tags: list[str] = []
    seen: set[str] = set()
    for keyword in PLATFORM_HASHTAGS.get(target_platform, []):
        normalized = _safe_stem(keyword).replace("_", "")
        if normalized and normalized not in seen:
            seen.add(normalized)
            tags.append(f"#{normalized}")
    for keyword in keywords:
        normalized = _safe_stem(keyword).replace("_", "")
        if normalized and normalized not in seen:
            seen.add(normalized)
            tags.append(f"#{normalized}")
        if len(tags) >= 8:
            break
    return tags


def _bundle_readme(project_name: str, manifest: dict) -> str:
    lines = [
        "# Auto Shorts Package",
        "",
        f"Project: {project_name}",
        f"Platform profile: {manifest['target_platform']}",
        f"Subtitle style: {manifest.get('subtitle_style', 'creator_bold')}",
        f"Generated at: {manifest['created_at']}",
        f"Source video: {manifest['source_video']}",
        f"Main context keywords: {', '.join(str(item) for item in (manifest.get('video_context') or {}).get('main_keywords', [])[:12])}",
        f"Director version: {(manifest.get('shorts_program') or {}).get('version', 'legacy')}",
        "",
        f"Rendered drafts: {manifest.get('rendered_count', len(manifest['shorts']))}",
        f"Accepted shorts: {manifest.get('accepted_count', len(manifest['shorts']))}",
        f"Rejected shorts: {manifest.get('rejected_count', 0)}",
        f"Drafts: {manifest.get('drafts_dir', manifest.get('bundle_dir', ''))}",
        f"Accepted files: {manifest.get('accepted_dir', manifest.get('bundle_dir', ''))}",
        f"Rejected files: {manifest.get('rejected_dir', manifest.get('bundle_dir', ''))}",
        "",
    ]
    for item in manifest["shorts"]:
        lines.extend(
            [
                f"## {item['rank']}. {item['title']}",
                f"- Window: {item['start']}s to {item['end']}s",
                f"- Duration: {item['duration']}s",
                f"- Score: {item['score']}",
                f"- Viral score: {item['viral_score']['overall']}",
                f"- Director role: {(item.get('director_plan') or {}).get('primary_role', 'unknown')}",
                f"- Composition: {item.get('composition_mode', 'single_window')}",
                f"- Quality gate: {(item.get('quality_gate') or {}).get('verdict', 'unknown')} ({(item.get('quality_gate') or {}).get('score', 'n/a')})",
                f"- Hook: {item['hook']}",
                f"- Why it works: {item['reason']}",
                "- Viral explainability:",
            ]
        )
        for explanation in item.get("viral_explanation", []):
            lines.append(f"  - {explanation}")
        if item.get("b_roll_suggestions"):
            lines.append("- B-roll suggestions:")
            for suggestion in item["b_roll_suggestions"]:
                lines.append(
                    "  - "
                    f"{suggestion['start']}s-{suggestion['end']}s | {suggestion['visual_type']} | "
                    f"{suggestion['search_query']} | {suggestion['direction']}"
                )
        if item.get("punch_in_moments"):
            lines.append("- Punch-in moments:")
            for moment in item["punch_in_moments"]:
                lines.append(
                    "  - "
                    f"{moment['start']}s-{moment['end']}s | zoom {moment['zoom']}x | {moment['reason']}"
                )
        lines.extend([f"- Deliverable: {item['vertical_video_path']}", ""])
    if manifest.get("rejected_shorts"):
        lines.extend(["## Rejected Shorts", ""])
        for item in manifest.get("rejected_shorts", []):
            lines.extend(
                [
                    f"- {item.get('title', 'Untitled')}",
                    f"  - Stage: {item.get('stage', 'unknown')}",
                    f"  - Reason: {item.get('rejection_reason', 'short was rejected by release QA')}",
                    f"  - Draft: {item.get('draft_video_path') or item.get('draft_dir') or 'n/a'}",
                    f"  - Rejected file: {item.get('rejected_video_path') or item.get('rejected_record_path') or 'n/a'}",
                ]
            )
        lines.append("")
    if manifest.get("compilation_path"):
        lines.extend([f"Compilation: {manifest['compilation_path']}", ""])
    return "\n".join(lines)


def execute(params: dict, state: ProjectState) -> dict:
    transcript_path = transcript_artifact_path(state.working_dir, "transcript.txt")
    srt_path = transcript_artifact_path(state.working_dir, "transcript.srt")
    if transcript_path is None or srt_path is None:
        transcribe_result = transcribe({}, state)
        state = transcribe_result["updated_state"]
        if not transcribe_result["success"]:
            return {
                "success": False,
                "message": transcribe_result["message"],
                "suggestion": None,
                "updated_state": state,
                "tool_name": "create_auto_shorts",
            }
        transcript_path = transcript_artifact_path(state.working_dir, "transcript.txt")
        srt_path = transcript_artifact_path(state.working_dir, "transcript.srt")

    transcript_bundle = load_transcript_bundle(state.working_dir)
    transcript_text = str(transcript_bundle.get("transcript_text") or "").strip()
    if not transcript_text and transcript_path is not None:
        transcript_text = transcript_path.read_text(encoding="utf-8").strip()
    transcript_segments = transcript_bundle.get("segments") if isinstance(transcript_bundle.get("segments"), list) else []
    transcript_segments = transcript_segments or (parse_srt(srt_path) if srt_path is not None else [])
    transcript_words = transcript_bundle.get("words") if isinstance(transcript_bundle.get("words"), list) else []
    candidate_segments = build_semantic_units(transcript_segments, transcript_words)
    if not transcript_text or not transcript_segments:
        return {
            "success": False,
            "message": "Transcript generation succeeded, but no usable timestamped transcript segments were found.",
            "suggestion": None,
            "updated_state": state,
            "tool_name": "create_auto_shorts",
        }

    count = max(1, min(int(params.get("count", 3)), 8))
    min_duration_sec = max(12.0, float(params.get("min_duration_sec", 20.0)))
    max_duration_sec = max(min_duration_sec + 2.0, min(float(params.get("max_duration_sec", 45.0)), 90.0))
    include_compilation = bool(params.get("include_compilation", True))
    target_platform = str(params.get("target_platform", "youtube_shorts")).strip().lower()
    if target_platform not in {"youtube_shorts", "tiktok", "instagram_reels"}:
        target_platform = "youtube_shorts"
    configured_provider = (state.provider or config.PROVIDER or "gemini").strip().lower()
    provider_name = configured_provider if configured_provider in {"gemini", "claude"} else "gemini"
    model_name = state.model or (
        config.CLAUDE_MODEL if provider_name == "claude" else config.GEMINI_MODEL
    )
    reasoning_available = configured_provider in {"gemini", "claude"} and bool(
        config.ANTHROPIC_API_KEY
        if provider_name == "claude"
        else config.GEMINI_API_KEY
    )
    source_metadata = state.metadata or {}
    creative_graph = build_video_understanding_graph(
        transcript_text=transcript_text,
        segments=candidate_segments,
        metadata=source_metadata,
        quality_tier="world_class_local",
        source_context={
            "feature": "auto_shorts",
            "target_platform": target_platform,
            "requested_count": count,
        },
    )
    video_context = graph_to_video_context(creative_graph)
    default_subtitle_style = str(PLATFORM_PROFILES.get(target_platform, PLATFORM_PROFILES["youtube_shorts"])["caption_style"])
    subtitle_style = str(params.get("subtitle_style") or default_subtitle_style).strip().lower().replace("-", "_")
    try:
        resolve_subtitle_style(subtitle_style)
    except ValueError as exc:
        return {
            "success": False,
            "message": str(exc),
            "suggestion": None,
            "updated_state": state,
            "tool_name": "create_auto_shorts",
        }

    candidates = _build_candidates(
        candidate_segments,
        min_duration_sec,
        max_duration_sec,
        limit=max(64, count * 18),
        target_platform=target_platform,
        video_context=video_context,
    )
    story_planner_provenance: dict[str, object]
    if configured_provider in {"gemini", "claude"} and reasoning_available:
        try:
            story_candidates, story_planner_provenance = _plan_story_candidates_with_llm(
                provider_name=provider_name,
                model_name=model_name,
                semantic_units=candidate_segments,
                count=max(count * 2, count + 2),
                min_duration_sec=min_duration_sec,
                max_duration_sec=max_duration_sec,
                target_platform=target_platform,
                video_context=video_context,
            )
        except Exception as exc:
            story_candidates = []
            story_planner_provenance = {
                "version": "shorts-story-planner-v1",
                "provider": provider_name,
                "model": model_name,
                "status": "error",
                "error": _truncate(f"{type(exc).__name__}: {exc}", 500),
                "accepted_candidate_ids": [],
            }
    else:
        story_candidates = []
        story_planner_provenance = {
            "version": "shorts-story-planner-v1",
            "provider": configured_provider,
            "model": model_name,
            "status": "skipped",
            "reason": (
                "reasoning provider API key is not configured"
                if configured_provider in {"gemini", "claude"}
                else "configured provider does not support story planning"
            ),
            "accepted_candidate_ids": [],
        }
    remix_candidates = _build_remix_candidates(
        candidates,
        candidate_segments,
        min_duration_sec=min_duration_sec,
        max_duration_sec=max_duration_sec,
        target_platform=target_platform,
        video_context=video_context,
        limit=max(12, count * 6),
    )
    if story_candidates or remix_candidates:
        candidates = [*story_candidates, *candidates, *remix_candidates]
        candidates.sort(
            key=lambda item: (
                float(item.get("heuristic_score") or 0.0),
                3.0 if item.get("candidate_origin") == "hierarchical_story_planner" else 0.0,
                float((item.get("score_breakdown") or {}).get("hook_strength", 0.0)),
                float((item.get("score_breakdown") or {}).get("payoff", 0.0)),
            ),
            reverse=True,
        )
        candidates = _dedupe_candidates(candidates, limit=max(80, count * 22))
    _apply_creative_graph_to_candidates(
        candidates,
        creative_graph,
        target_platform=target_platform,
    )
    if not candidates:
        return {
            "success": False,
            "message": "No viable short-form clip windows were found in the transcript.",
            "suggestion": None,
            "updated_state": state,
            "tool_name": "create_auto_shorts",
        }
    shorts_program = build_shorts_program(
        transcript_text=transcript_text,
        segments=candidate_segments,
        candidates=candidates,
        selections=[],
        requested_count=min(count, len(candidates)),
        target_platform=target_platform,
        min_duration_sec=min_duration_sec,
        max_duration_sec=max_duration_sec,
        video_context=video_context,
    )
    program_validation = validate_shorts_program(shorts_program)
    if not program_validation["passed"]:
        return {
            "success": False,
            "message": "Auto shorts program failed validation: " + "; ".join(program_validation["errors"]),
            "suggestion": None,
            "updated_state": state,
            "tool_name": "create_auto_shorts",
        }
    _apply_shorts_program_to_candidates(candidates, shorts_program)

    selection_pool_size = min(len(candidates), max(count * 3, count + 2))
    selector_provenance: dict[str, object] = {
        "version": "shorts-candidate-tournament-v2",
        "status": "not_started",
    }
    try:
        if not reasoning_available:
            raise RuntimeError("reasoning provider API key is not configured")
        selections = _select_shorts_with_llm(
            provider_name=provider_name,
            model_name=model_name,
            candidates=candidates,
            transcript_text=transcript_text,
            count=selection_pool_size,
            min_duration_sec=min_duration_sec,
            max_duration_sec=max_duration_sec,
            target_platform=target_platform,
            video_context=video_context,
            provenance=selector_provenance,
        )
    except Exception as exc:
        selector_provenance.update(
            {
                "status": "error",
                "provider": provider_name,
                "model": model_name,
                "error": _truncate(f"{type(exc).__name__}: {exc}", 500),
            }
        )
        selections = []
    if not selections:
        selections = _fallback_selections(candidates, count=selection_pool_size)
        selector_provenance.update(
            {
                "status": "deterministic_fallback",
                "fallback_reason": selector_provenance.get("error")
                or "model tournament returned no valid candidates",
                "returned_count": len(selections),
            }
        )
    shorts_program = build_shorts_program(
        transcript_text=transcript_text,
        segments=candidate_segments,
        candidates=candidates,
        selections=selections,
        requested_count=selection_pool_size,
        target_platform=target_platform,
        min_duration_sec=min_duration_sec,
        max_duration_sec=max_duration_sec,
        video_context=video_context,
    )
    program_validation = validate_shorts_program(shorts_program)
    if not program_validation["passed"]:
        return {
            "success": False,
            "message": "Auto shorts program failed validation: " + "; ".join(program_validation["errors"]),
            "suggestion": None,
            "updated_state": state,
            "tool_name": "create_auto_shorts",
        }
    selections = _reconcile_selections_with_program(
        selections,
        candidates,
        shorts_program,
        count=selection_pool_size,
    )

    candidate_map = {candidate["candidate_id"]: candidate for candidate in candidates}
    timestamp_label = utc_now_iso().replace(":", "-").replace("+00:00", "Z")
    bundle_dir = Path(state.output_dir) / f"{_safe_stem(state.project_name)}_auto_shorts_{timestamp_label}"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    drafts_dir = bundle_dir / "drafts"
    accepted_dir = bundle_dir / "accepted"
    rejected_dir = bundle_dir / "rejected"
    drafts_dir.mkdir(parents=True, exist_ok=True)
    accepted_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)

    created_shorts: list[dict] = []
    rejected_shorts: list[dict] = []
    vertical_paths: list[str] = []
    failures: list[str] = []
    rendered_count = 0
    attempted_selection_count = 0

    for rank, selection in enumerate(selections, start=1):
        if len(created_shorts) >= count:
            break
        attempted_selection_count += 1
        candidate = candidate_map.get(selection["candidate_id"])
        if candidate is None:
            continue
        edit_plan = _normalize_short_edit_plan_fields(
            _edit_plan_dict(shorts_program, str(selection["candidate_id"])),
            candidate,
        )
        short_dir = drafts_dir / f"{rank:02d}_{_safe_stem(selection['title'])[:48]}"
        short_dir.mkdir(parents=True, exist_ok=True)
        try:
            source_ranges = _candidate_source_ranges(candidate)
            clip_segments = _clip_transcript_segments_for_ranges(transcript_segments, source_ranges)
            preflight = _preflight_short_edit_plan(
                candidate=candidate,
                selection=selection,
                clip_segments=clip_segments,
                edit_plan=edit_plan,
                target_platform=target_platform,
                video_context=video_context,
            )
            if not preflight["passed"]:
                rejection_reasons = [str(error) for error in preflight.get("errors", []) if str(error).strip()]
                rejection_reason = "; ".join(rejection_reasons) or "pre-render edit plan rejected short"
                failures.append(f"{selection['title']}: pre-render edit plan rejected short ({rejection_reason})")
                rejected_record = {
                    "rank": rank,
                    "selection_rank": rank,
                    "candidate_id": selection["candidate_id"],
                    "title": selection["title"],
                    "selection_source": selection.get("selection_source", "unknown"),
                    "stage": "preflight",
                    "accepted": False,
                    "rejection_reason": rejection_reason,
                    "rejection_reasons": rejection_reasons or [rejection_reason],
                    "candidate": candidate,
                    "selection": selection,
                    "edit_plan": edit_plan,
                    "preflight": preflight,
                    "draft_dir": str(short_dir),
                }
                preflight_path = short_dir / "preflight_rejected.json"
                rejected_record["metadata_path"] = str(preflight_path)
                rejected_record_path = rejected_dir / f"{rank:02d}_{_safe_stem(selection['title'])[:48]}_preflight.json"
                rejected_record["rejected_record_path"] = str(rejected_record_path)
                preflight_path.write_text(json.dumps(rejected_record, indent=2), encoding="utf-8")
                rejected_record_path.write_text(json.dumps(rejected_record, indent=2), encoding="utf-8")
                rejected_shorts.append(rejected_record)
                continue

            raw_clip_path, source_parts = _render_candidate_raw_clip(state, candidate, short_dir)

            transcript_txt_path = short_dir / "transcript.txt"
            transcript_txt_path.write_text(
                " ".join(str(segment["text"]).strip() for segment in clip_segments).strip() + "\n",
                encoding="utf-8",
            )
            caption_segments = optimize_caption_segments(clip_segments)
            captions_path = short_dir / "captions.srt"
            if caption_segments:
                write_srt_segments(captions_path, caption_segments)
                captions_arg = str(captions_path)
            else:
                captions_arg = None

            viral_analysis = _analyze_viral_score_with_llm(
                provider_name=provider_name,
                model_name=model_name,
                candidate=candidate,
                selection=selection,
                clip_segments=clip_segments,
                target_platform=target_platform,
                video_context=video_context,
            )
            b_roll_suggestions = _analyze_b_roll_with_llm(
                provider_name=provider_name,
                model_name=model_name,
                candidate=candidate,
                selection=selection,
                clip_segments=clip_segments,
                target_platform=target_platform,
            )
            b_roll_suggestions = _apply_edit_plan_to_b_roll(b_roll_suggestions, edit_plan)
            punch_in_moments = _analyze_punch_in_with_llm(
                provider_name=provider_name,
                model_name=model_name,
                candidate=candidate,
                selection=selection,
                clip_segments=clip_segments,
                target_platform=target_platform,
            )
            punch_in_moments = _apply_edit_plan_to_punch_ins(punch_in_moments, edit_plan)
            motion_input_path = apply_center_punch_ins(
                str(raw_clip_path),
                state.working_dir,
                punch_in_moments,
            )
            motion_clip_path = None
            if motion_input_path != str(raw_clip_path):
                motion_clip_path = short_dir / "punch_in_clip.mp4"
                shutil.copy2(motion_input_path, motion_clip_path)

            vertical_temp_path = render_vertical_short(
                motion_input_path,
                state.working_dir,
                srt_path=captions_arg,
                subtitle_style=subtitle_style,
            )
            vertical_video_path = short_dir / f"{rank:02d}_{_safe_stem(selection['title'])}_{target_platform}.mp4"
            shutil.copy2(vertical_temp_path, vertical_video_path)
            rendered_count += 1
            metadata = probe_video(str(vertical_video_path))
            hashtags = _hashtags(selection.get("keywords", []), target_platform)
            short_record = {
                "rank": rank,
                "selection_rank": rank,
                "title": selection["title"],
                "hook": selection["hook"],
                "reason": selection["reason"],
                "selection_source": selection.get("selection_source", "unknown"),
                "score": round(float(selection["score"]), 2),
                "viral_score": viral_analysis["viral_score"],
                "viral_explanation": viral_analysis["viral_explanation"],
                "b_roll_suggestions": b_roll_suggestions,
                "punch_in_moments": punch_in_moments,
                "motion_clip_path": str(motion_clip_path) if motion_clip_path else None,
                "start": round(float(candidate["start"]), 2),
                "end": round(float(candidate["end"]), 2),
                "duration": round(float(candidate["duration"]), 2),
                "composition_mode": str(candidate.get("composition_mode") or ("remix" if len(source_ranges) > 1 else "single_window")),
                "source_ranges": source_ranges,
                "source_parts": source_parts,
                "heuristic_score": round(float(candidate["heuristic_score"]), 2),
                "creative_graph_signals": candidate.get("creative_graph_signals", {}),
                "creative_quality_report": candidate.get("creative_quality_report", {}),
                "score_breakdown": candidate.get("score_breakdown", {}),
                "director_plan": candidate.get("director_plan", {}),
                "edit_plan": edit_plan,
                "selection_reasons": candidate.get("selection_reasons", []),
                "keywords": selection.get("keywords", []),
                "hashtags": hashtags,
                "raw_clip_path": str(raw_clip_path),
                "vertical_video_path": str(vertical_video_path),
                "draft_video_path": str(vertical_video_path),
                "draft_dir": str(short_dir),
                "captions_path": str(captions_path) if clip_segments else None,
                "transcript_path": str(transcript_txt_path),
                "resolution": f"{metadata.get('width', 0)}x{metadata.get('height', 0)}",
                "target_platform": target_platform,
            }
            short_record["preflight"] = preflight
            render_validation = validate_short_render(short_record, metadata, edit_plan)
            short_record["render_validation"] = render_validation
            quality_gate = _quality_gate_with_llm(
                provider_name=provider_name,
                model_name=model_name,
                candidate=candidate,
                selection=selection,
                clip_segments=clip_segments,
                short_record=short_record,
                target_platform=target_platform,
                video_context=video_context,
            )
            short_record["quality_gate"] = quality_gate
            if render_validation.get("errors"):
                failures.extend(f"{selection['title']}: {error}" for error in render_validation["errors"])
            if not quality_gate.get("passed", False):
                failures.append(
                    f"{selection['title']}: quality gate rejected short"
                    + (
                        f" ({quality_gate.get('rejection_reason')})"
                        if quality_gate.get("rejection_reason")
                        else ""
                    )
                )
            accepted = not render_validation.get("errors") and bool(quality_gate.get("passed", False))
            short_record["accepted"] = accepted
            if accepted:
                short_record["stage"] = "accepted"
                short_record["rank"] = len(created_shorts) + 1
                accepted_video_path = accepted_dir / vertical_video_path.name
                shutil.copy2(vertical_video_path, accepted_video_path)
                short_record["accepted_video_path"] = str(accepted_video_path)
                short_record["vertical_video_path"] = str(accepted_video_path)
            else:
                short_record["stage"] = "post_render_qa"
                rejection_reasons = _short_rejection_reasons(render_validation, quality_gate)
                rejected_video_path = rejected_dir / vertical_video_path.name
                rejected_path = short_dir / "rejected.json"
                rejected_record_path = rejected_dir / f"{rank:02d}_{_safe_stem(selection['title'])[:48]}_rejected.json"
                shutil.copy2(vertical_video_path, rejected_video_path)
                short_record["rejected_video_path"] = str(rejected_video_path)
                short_record["vertical_video_path"] = str(rejected_video_path)
                short_record["rejection_reason"] = "; ".join(rejection_reasons)
                short_record["rejection_reasons"] = rejection_reasons
                short_record["rejected_metadata_path"] = str(rejected_path)
                short_record["rejected_record_path"] = str(rejected_record_path)
            metadata_path = short_dir / "metadata.json"
            short_record["metadata_path"] = str(metadata_path)
            metadata_path.write_text(json.dumps(short_record, indent=2), encoding="utf-8")
            (short_dir / "broll_suggestions.json").write_text(
                json.dumps(b_roll_suggestions, indent=2),
                encoding="utf-8",
            )
            (short_dir / "punch_in_plan.json").write_text(
                json.dumps(punch_in_moments, indent=2),
                encoding="utf-8",
            )
            note_lines = [
                f"# {selection['title']}",
                "",
                f"Hook: {selection['hook']}",
                "",
                f"Why it works: {selection['reason']}",
                "",
                f"Viral score: {viral_analysis['viral_score']['overall']}",
                (
                    "Score breakdown: "
                    f"hook={viral_analysis['viral_score']['hook_strength']}, "
                    f"payoff={viral_analysis['viral_score']['payoff']}, "
                    f"novelty={viral_analysis['viral_score']['novelty']}, "
                    f"clarity={viral_analysis['viral_score']['clarity']}, "
                    f"shareability={viral_analysis['viral_score']['shareability']}"
                ),
                (
                    "Context fit: "
                    f"context={float((candidate.get('score_breakdown') or {}).get('context_score', 0.0)):.1f}, "
                    f"standalone={float((candidate.get('score_breakdown') or {}).get('standalone_clarity', 0.0)):.1f}, "
                    f"story={float((candidate.get('score_breakdown') or {}).get('story_completeness', 0.0)):.1f}"
                ),
                (
                    "Creative graph: "
                    f"retention={float((candidate.get('score_breakdown') or {}).get('creative_graph_retention_score', 0.0)):.1f}, "
                    f"visual={float((candidate.get('score_breakdown') or {}).get('creative_graph_visual_opportunity', 0.0)):.1f}, "
                    f"topic={float((candidate.get('score_breakdown') or {}).get('creative_graph_topic_alignment', 0.0)):.1f}"
                ),
                "",
                "Viral explainability:",
            ]
            for explanation in viral_analysis["viral_explanation"]:
                note_lines.append(f"- {explanation}")
            if b_roll_suggestions:
                note_lines.extend(["", "B-roll suggestions:"])
                for suggestion in b_roll_suggestions:
                    note_lines.append(
                        "- "
                        f"{suggestion['start']}s-{suggestion['end']}s | {suggestion['visual_type']} | "
                        f"query: {suggestion['search_query']} | {suggestion['direction']}"
                    )
            if punch_in_moments:
                note_lines.extend(["", "Punch-in moments:"])
                for moment in punch_in_moments:
                    note_lines.append(
                        "- "
                        f"{moment['start']}s-{moment['end']}s | zoom {moment['zoom']}x | {moment['reason']}"
                    )
            note_lines.extend(
                [
                    "",
                    "Quality gate:",
                    f"- Verdict: {quality_gate['verdict']} ({quality_gate['score']}/100)",
                ]
            )
            if quality_gate.get("rejection_reason"):
                note_lines.append(f"- Rejection reason: {quality_gate['rejection_reason']}")
            note_lines.extend(["", f"Suggested hashtags: {' '.join(hashtags)}"])
            (short_dir / "notes.md").write_text("\n".join(note_lines) + "\n", encoding="utf-8")
            if not accepted:
                rejected_path.write_text(json.dumps(short_record, indent=2), encoding="utf-8")
                rejected_record_path.write_text(json.dumps(short_record, indent=2), encoding="utf-8")
                rejected_shorts.append(short_record)
                continue
            vertical_paths.append(short_record["vertical_video_path"])
            created_shorts.append(short_record)
        except (ValueError, OSError, VideoEngineError) as exc:
            failures.append(f"{selection['title']}: {exc}")

    compilation_path = None
    if created_shorts and include_compilation and len(vertical_paths) > 1:
        try:
            compilation_temp_path = merge(vertical_paths, state.working_dir)
            compilation_path = bundle_dir / "all_shorts_compilation.mp4"
            shutil.copy2(compilation_temp_path, compilation_path)
        except VideoEngineError as exc:
            failures.append(f"Compilation: {exc}")

    manifest = {
        "created_at": utc_now_iso(),
        "project_id": state.project_id,
        "project_name": state.project_name,
        "source_video": state.working_file,
        "target_platform": target_platform,
        "subtitle_style": subtitle_style,
        "shorts": created_shorts,
        "candidate_count": len(candidates),
        "remix_candidate_count": len([candidate for candidate in candidates if candidate.get("composition_mode") == "remix"]),
        "story_candidate_count": len([candidate for candidate in candidates if candidate.get("candidate_origin") == "hierarchical_story_planner"]),
        "candidate_source": "semantic_transcript_units_v1",
        "semantic_unit_count": len(candidate_segments),
        "story_planner": story_planner_provenance,
        "selection_provenance": selector_provenance,
        "candidate_quality_reports": [
            candidate.get("creative_quality_report", {})
            for candidate in candidates[: min(len(candidates), 40)]
            if candidate.get("creative_quality_report")
        ],
        "video_context": video_context,
        "creative_graph": creative_graph.to_dict(),
        "creative_graph_summary": creative_graph.compact(),
        "shorts_program": shorts_program.to_dict(),
        "program_validation": program_validation,
        "quality_gate_version": "shorts-release-qa-v2",
        "bundle_dir": str(bundle_dir),
        "drafts_dir": str(drafts_dir),
        "accepted_dir": str(accepted_dir),
        "rejected_dir": str(rejected_dir),
        "rendered_count": rendered_count,
        "accepted_count": len(created_shorts),
        "rejected_count": len(rejected_shorts),
        "selected_count": len(selections),
        "requested_count": count,
        "reserve_count": max(0, len(selections) - count),
        "attempted_selection_count": attempted_selection_count,
        "rejected_shorts": rejected_shorts,
        "compilation_path": str(compilation_path) if compilation_path else None,
        "transcript_path": str(transcript_path),
        "srt_path": str(srt_path),
        "failures": failures,
    }
    manifest_path = bundle_dir / "manifest.json"
    quality_scores = [
        float((short.get("creative_quality_report") or {}).get("score"))
        for short in created_shorts
        if isinstance(short.get("creative_quality_report"), dict)
        and (short.get("creative_quality_report") or {}).get("score") is not None
    ]
    registry_result = record_creative_run(
        working_dir=state.working_dir,
        feature="auto_shorts",
        manifest_path=str(manifest_path),
        output_path=str(compilation_path) if compilation_path else None,
        graph_version=creative_graph.version,
        quality_score=(sum(quality_scores) / len(quality_scores)) if quality_scores else None,
        summary={
            "count": len(created_shorts),
            "accepted_count": len(created_shorts),
            "rejected_count": len(rejected_shorts),
            "rendered_count": rendered_count,
            "target_platform": target_platform,
            "subtitle_style": subtitle_style,
            "candidate_count": len(candidates),
        },
        artifacts={
            "bundle_dir": str(bundle_dir),
            "drafts_dir": str(drafts_dir),
            "accepted_dir": str(accepted_dir),
            "rejected_dir": str(rejected_dir),
            "short_paths": [item["vertical_video_path"] for item in created_shorts],
            "rejected_records": [item.get("rejected_record_path") for item in rejected_shorts if item.get("rejected_record_path")],
        },
    )
    manifest["creative_registry"] = registry_result
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (bundle_dir / "README.md").write_text(_bundle_readme(state.project_name, manifest) + "\n", encoding="utf-8")

    state.artifacts["latest_auto_shorts"] = {
        "created_at": manifest["created_at"],
        "manifest_path": str(manifest_path),
        "bundle_dir": str(bundle_dir),
        "drafts_dir": str(drafts_dir),
        "accepted_dir": str(accepted_dir),
        "rejected_dir": str(rejected_dir),
        "count": len(created_shorts),
        "accepted_count": len(created_shorts),
        "rejected_count": len(rejected_shorts),
        "rendered_count": rendered_count,
        "target_platform": target_platform,
        "subtitle_style": subtitle_style,
        "creative_graph_version": creative_graph.version,
        "creative_registry": registry_result,
    }
    history = list(state.artifacts.get("auto_shorts_history") or [])
    history.append(state.artifacts["latest_auto_shorts"])
    state.artifacts["auto_shorts_history"] = history[-10:]
    state.save()

    if not created_shorts:
        detail = f" Details: {'; '.join(failures)}" if failures else ""
        return {
            "success": False,
            "message": (
                "Auto shorts analysis completed, but no selected clip passed render and transcript QA. "
                f"Rendered {rendered_count} draft(s), accepted 0, rejected {len(rejected_shorts)}. "
                f"Rejected files: {rejected_dir}. Drafts: {drafts_dir}. Manifest: {manifest_path}.{detail}"
            ),
            "suggestion": None,
            "updated_state": state,
            "tool_name": "create_auto_shorts",
        }

    titles = ", ".join(item["title"] for item in created_shorts)
    failure_suffix = f" Failed extras: {'; '.join(failures)}" if failures else ""
    return {
        "success": True,
        "message": (
            f"Rendered {rendered_count} draft(s); accepted {len(created_shorts)}, rejected {len(rejected_shorts)}. "
            f"Accepted files: {accepted_dir}. Rejected files: {rejected_dir}. Drafts: {drafts_dir}. "
            f"Top picks: {titles}. Manifest: {manifest_path}.{failure_suffix}"
        ),
        "suggestion": None,
        "updated_state": state,
        "tool_name": "create_auto_shorts",
    }
