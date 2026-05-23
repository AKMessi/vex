from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from pathlib import Path

from google import genai

import config
from engine import apply_center_punch_ins, VideoEngineError, merge, probe_video, render_vertical_short, trim
from shorts import build_shorts_program, validate_short_render, validate_shorts_program
from state import ProjectState, utc_now_iso
from subtitles import resolve_subtitle_style
from tools.transcript import execute as transcribe
from tools.transcript_utils import load_transcript_bundle, optimize_caption_segments, parse_srt, write_srt_segments

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

def _overlap_ratio(first: dict, second: dict) -> float:
    overlap = max(0.0, min(first["end"], second["end"]) - max(first["start"], second["start"]))
    if overlap <= 0:
        return 0.0
    shortest = min(first["end"] - first["start"], second["end"] - second["start"])
    return overlap / max(shortest, 0.001)


def _dedupe_candidates(candidates: list[dict], limit: int) -> list[dict]:
    selected: list[dict] = []
    for candidate in candidates:
        if all(_overlap_ratio(candidate, existing) < 0.68 for existing in selected):
            selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected


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
    policy = dict(edit_plan.get("punch_in_policy") or {})
    if policy and not bool(policy.get("enabled", True)):
        return []
    max_moments = max(0, int(policy.get("max_moments", len(moments)) or 0))
    if max_moments <= 0:
        return []
    min_gap = max(0.0, float(policy.get("min_gap_sec", 0.8) or 0.8))
    max_zoom = max(1.03, float(policy.get("max_zoom", 1.18) or 1.18))
    selected: list[dict] = []
    last_end = -999.0
    for moment in sorted(moments, key=lambda item: float(item.get("start", 0.0))):
        start_sec = float(moment.get("start", 0.0))
        end_sec = float(moment.get("end", start_sec))
        if start_sec - last_end < min_gap:
            continue
        adjusted = dict(moment)
        adjusted["zoom"] = round(min(float(moment.get("zoom", 1.12) or 1.12), max_zoom), 2)
        selected.append(adjusted)
        last_end = end_sec
        if len(selected) >= max_moments:
            break
    return selected


def _apply_edit_plan_to_b_roll(suggestions: list[dict], edit_plan: dict) -> list[dict]:
    policy = dict(edit_plan.get("visual_insert_policy") or {})
    if policy and not bool(policy.get("enabled", True)):
        return []
    max_inserts = max(0, int(policy.get("max_inserts", len(suggestions)) or 0))
    if max_inserts <= 0:
        return []
    return suggestions[:max_inserts]


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
        breakdown = candidate.get("score_breakdown") or {}
        reasons = "; ".join(str(item) for item in candidate.get("selection_reasons", [])[:4])
        keywords = ", ".join(str(item) for item in candidate.get("keywords", [])[:8])
        lines.append(
            "\n".join(
                [
                    (
                        f"{candidate['candidate_id']} | {candidate['start']:.2f}-{candidate['end']:.2f} "
                        f"({candidate['duration']:.2f}s) | heuristic={candidate['heuristic_score']:.2f}"
                    ),
                    (
                        "Signals: "
                        f"hook={float(breakdown.get('hook_strength', 0.0)):.1f}, "
                        f"payoff={float(breakdown.get('payoff', 0.0)):.1f}, "
                        f"novelty={float(breakdown.get('novelty', 0.0)):.1f}, "
                        f"clarity={float(breakdown.get('clarity', 0.0)):.1f}, "
                        f"shareability={float(breakdown.get('shareability', 0.0)):.1f}, "
                        f"context={float(breakdown.get('context_score', 0.0)):.1f}, "
                        f"director={float(breakdown.get('director_score', 0.0)):.1f}, "
                        f"arc={float(breakdown.get('arc_integrity', 0.0)):.1f}, "
                        f"risk={float(breakdown.get('continuity_risk', 0.0)):.1f}, "
                        f"standalone={float(breakdown.get('standalone_clarity', 0.0)):.1f}, "
                        f"story={float(breakdown.get('story_completeness', 0.0)):.1f}, "
                        f"abrupt_penalty={float(breakdown.get('abrupt_start_penalty', 0.0)):.1f}, "
                        f"pace={float(breakdown.get('words_per_sec', 0.0)):.2f}wps"
                    ),
                    f"Why candidate exists: {reasons or 'coherent transcript window'}",
                    f"Keywords: {keywords or 'n/a'}",
                    f"Excerpt: {candidate['excerpt']}",
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
        if all(_overlap_ratio(candidate, existing) < 0.52 and _topic_similarity(candidate, existing) < 0.58 for existing in selected):
            selected.append(candidate)
            additions.append(candidate)
        if len(additions) >= count:
            return additions
    for candidate in candidates:
        if candidate["candidate_id"] in excluded_ids or candidate in selected or candidate in additions:
            continue
        additions.append(candidate)
        if len(additions) >= count:
            break
    return additions


def _selection_from_candidate(candidate: dict) -> dict:
    reasons = candidate.get("selection_reasons") or ["Strong deterministic transcript score."]
    return {
        "candidate_id": candidate["candidate_id"],
        "score": round(float(candidate["heuristic_score"]), 2),
        "title": _default_title(candidate),
        "hook": _default_hook(candidate),
        "reason": _truncate("; ".join(str(reason) for reason in reasons), 220),
        "keywords": list(candidate.get("keywords") or _word_tokens(candidate["excerpt"])[:5])[:6],
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
) -> list[dict]:
    candidate_map = {candidate["candidate_id"]: candidate for candidate in candidates}
    system_prompt = (
        "You are a short-form video strategist. Choose the most clip-worthy windows from a transcript candidate list. "
        "Prioritize the clips most likely to retain a cold viewer: a fast first-line hook, concrete specificity, tension or contrast, "
        "a satisfying payoff before the end, full-video thesis alignment, and a clean standalone idea. "
        "Diversify topics and reject near-duplicates, misleading fragments, abrupt starts, and clips that only make sense with prior context. "
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
        f"Candidate windows:\n{_format_candidates_for_llm(candidates)}\n\n"
        "Choose the best candidates for viral-style shorts. Prefer windows with high deterministic signals unless the transcript clearly proves a better clip. "
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
        return diverse
    excluded_ids = {selection["candidate_id"] for selection in diverse}
    diverse.extend(
        _fallback_selections(
            candidates,
            count - len(diverse),
            excluded_ids=excluded_ids,
            seed_candidates=diverse_candidates,
        )
    )
    return diverse[:count]


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
        f"Shorts created: {len(manifest['shorts'])}",
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
    if manifest.get("compilation_path"):
        lines.extend([f"Compilation: {manifest['compilation_path']}", ""])
    return "\n".join(lines)


def execute(params: dict, state: ProjectState) -> dict:
    transcript_path = Path(state.working_dir) / "transcript.txt"
    srt_path = Path(state.working_dir) / "transcript.srt"
    if not transcript_path.is_file() or not srt_path.is_file():
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

    transcript_bundle = load_transcript_bundle(state.working_dir)
    transcript_text = str(transcript_bundle.get("transcript_text") or "").strip() or transcript_path.read_text(encoding="utf-8").strip()
    transcript_segments = transcript_bundle.get("segments") if isinstance(transcript_bundle.get("segments"), list) else []
    transcript_segments = transcript_segments or parse_srt(srt_path)
    sentence_segments = transcript_bundle.get("sentences") if isinstance(transcript_bundle.get("sentences"), list) else []
    candidate_segments = sentence_segments or transcript_segments
    video_context = _build_video_context(transcript_text, candidate_segments)
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

    provider_name = (state.provider or config.PROVIDER or "gemini").strip().lower()
    if provider_name not in {"gemini", "claude"}:
        provider_name = "gemini"
    model_name = state.model or (
        config.CLAUDE_MODEL if provider_name == "claude" else config.GEMINI_MODEL
    )

    try:
        selections = _select_shorts_with_llm(
            provider_name=provider_name,
            model_name=model_name,
            candidates=candidates,
            transcript_text=transcript_text,
            count=min(count, len(candidates)),
            min_duration_sec=min_duration_sec,
            max_duration_sec=max_duration_sec,
            target_platform=target_platform,
            video_context=video_context,
        )
    except Exception:
        selections = []
    if not selections:
        selections = _fallback_selections(candidates, count=min(count, len(candidates)))
    shorts_program = build_shorts_program(
        transcript_text=transcript_text,
        segments=candidate_segments,
        candidates=candidates,
        selections=selections,
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
    selections = _reconcile_selections_with_program(
        selections,
        candidates,
        shorts_program,
        count=min(count, len(candidates)),
    )

    candidate_map = {candidate["candidate_id"]: candidate for candidate in candidates}
    timestamp_label = utc_now_iso().replace(":", "-").replace("+00:00", "Z")
    bundle_dir = Path(state.output_dir) / f"{_safe_stem(state.project_name)}_auto_shorts_{timestamp_label}"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    created_shorts: list[dict] = []
    vertical_paths: list[str] = []
    failures: list[str] = []

    for rank, selection in enumerate(selections, start=1):
        candidate = candidate_map.get(selection["candidate_id"])
        if candidate is None:
            continue
        edit_plan = _edit_plan_dict(shorts_program, str(selection["candidate_id"]))
        short_dir = bundle_dir / f"{rank:02d}_{_safe_stem(selection['title'])[:48]}"
        short_dir.mkdir(parents=True, exist_ok=True)
        try:
            raw_temp_path = trim(
                state.working_file,
                state.working_dir,
                float(candidate["start"]),
                float(candidate["end"]),
            )
            raw_clip_path = short_dir / "raw_clip.mp4"
            shutil.copy2(raw_temp_path, raw_clip_path)

            clip_segments = _clip_transcript_segments(
                transcript_segments,
                start_sec=float(candidate["start"]),
                end_sec=float(candidate["end"]),
            )
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
            vertical_paths.append(str(vertical_video_path))
            metadata = probe_video(str(vertical_video_path))
            hashtags = _hashtags(selection.get("keywords", []), target_platform)
            short_record = {
                "rank": rank,
                "title": selection["title"],
                "hook": selection["hook"],
                "reason": selection["reason"],
                "score": round(float(selection["score"]), 2),
                "viral_score": viral_analysis["viral_score"],
                "viral_explanation": viral_analysis["viral_explanation"],
                "b_roll_suggestions": b_roll_suggestions,
                "punch_in_moments": punch_in_moments,
                "motion_clip_path": str(motion_clip_path) if motion_clip_path else None,
                "start": round(float(candidate["start"]), 2),
                "end": round(float(candidate["end"]), 2),
                "duration": round(float(candidate["duration"]), 2),
                "heuristic_score": round(float(candidate["heuristic_score"]), 2),
                "score_breakdown": candidate.get("score_breakdown", {}),
                "director_plan": candidate.get("director_plan", {}),
                "edit_plan": edit_plan,
                "selection_reasons": candidate.get("selection_reasons", []),
                "keywords": selection.get("keywords", []),
                "hashtags": hashtags,
                "raw_clip_path": str(raw_clip_path),
                "vertical_video_path": str(vertical_video_path),
                "captions_path": str(captions_path) if clip_segments else None,
                "transcript_path": str(transcript_txt_path),
                "resolution": f"{metadata.get('width', 0)}x{metadata.get('height', 0)}",
            }
            render_validation = validate_short_render(short_record, metadata, edit_plan)
            short_record["render_validation"] = render_validation
            if render_validation.get("errors"):
                failures.extend(f"{selection['title']}: {error}" for error in render_validation["errors"])
            (short_dir / "metadata.json").write_text(json.dumps(short_record, indent=2), encoding="utf-8")
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
            note_lines.extend(["", f"Suggested hashtags: {' '.join(hashtags)}"])
            (short_dir / "notes.md").write_text("\n".join(note_lines) + "\n", encoding="utf-8")
            created_shorts.append(short_record)
        except (ValueError, VideoEngineError) as exc:
            failures.append(f"{selection['title']}: {exc}")

    if not created_shorts:
        detail = f" Details: {'; '.join(failures)}" if failures else ""
        return {
            "success": False,
            "message": f"Auto shorts analysis completed, but FFmpeg failed to render every selected clip.{detail}",
            "suggestion": None,
            "updated_state": state,
            "tool_name": "create_auto_shorts",
        }

    compilation_path = None
    if include_compilation and len(vertical_paths) > 1:
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
        "candidate_source": "sentences" if sentence_segments else "srt_segments",
        "video_context": video_context,
        "shorts_program": shorts_program.to_dict(),
        "program_validation": program_validation,
        "bundle_dir": str(bundle_dir),
        "compilation_path": str(compilation_path) if compilation_path else None,
        "transcript_path": str(transcript_path),
        "srt_path": str(srt_path),
        "failures": failures,
    }
    manifest_path = bundle_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (bundle_dir / "README.md").write_text(_bundle_readme(state.project_name, manifest) + "\n", encoding="utf-8")

    state.artifacts["latest_auto_shorts"] = {
        "created_at": manifest["created_at"],
        "manifest_path": str(manifest_path),
        "bundle_dir": str(bundle_dir),
        "count": len(created_shorts),
        "target_platform": target_platform,
        "subtitle_style": subtitle_style,
    }
    history = list(state.artifacts.get("auto_shorts_history") or [])
    history.append(state.artifacts["latest_auto_shorts"])
    state.artifacts["auto_shorts_history"] = history[-10:]
    state.save()

    titles = ", ".join(item["title"] for item in created_shorts)
    failure_suffix = f" Failed extras: {'; '.join(failures)}" if failures else ""
    return {
        "success": True,
        "message": (
            f"Created {len(created_shorts)} auto shorts in {bundle_dir}. "
            f"Top picks: {titles}. Manifest: {manifest_path}.{failure_suffix}"
        ),
        "suggestion": None,
        "updated_state": state,
        "tool_name": "create_auto_shorts",
    }
