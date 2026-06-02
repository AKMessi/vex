from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import re
from typing import Any


GRAPH_VERSION = "video-understanding-graph-v1"

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "this",
    "that",
    "these",
    "those",
    "you",
    "your",
    "our",
    "their",
    "from",
    "into",
    "over",
    "under",
    "about",
    "just",
    "than",
    "then",
    "they",
    "them",
    "have",
    "has",
    "had",
    "was",
    "were",
    "are",
    "is",
    "be",
    "been",
    "being",
    "what",
    "when",
    "where",
    "which",
    "it",
    "its",
}
HOOK_TERMS = {"wait", "watch", "look", "why", "how", "secret", "mistake", "truth", "nobody", "everyone", "actually"}
PAYOFF_TERMS = {"because", "therefore", "so", "means", "result", "takeaway", "lesson", "framework", "works", "fix", "solves", "answer"}
PROOF_TERMS = {"proof", "data", "percent", "million", "billion", "number", "evidence", "study", "metric", "measured"}
TENSION_TERMS = {"but", "however", "instead", "wrong", "break", "broken", "until", "unless", "versus", "vs", "risk", "problem"}
VISUAL_TERMS = {
    "process",
    "system",
    "pipeline",
    "framework",
    "map",
    "chart",
    "graph",
    "numbers",
    "data",
    "steps",
    "layer",
    "inside",
    "screen",
    "interface",
    "product",
    "compare",
    "before",
    "after",
}
PLATFORM_PROFILE = {
    "youtube_shorts": {"ideal_duration": 34.0, "max_caption_wps": 4.2, "opening_hook_sec": 2.4},
    "tiktok": {"ideal_duration": 27.0, "max_caption_wps": 4.6, "opening_hook_sec": 1.8},
    "instagram_reels": {"ideal_duration": 30.0, "max_caption_wps": 4.0, "opening_hook_sec": 2.2},
}


@dataclass(frozen=True)
class SemanticBeat:
    beat_id: str
    start: float
    end: float
    text: str
    role: str
    score: float
    keywords: list[str]
    signals: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetentionMoment:
    moment_id: str
    start: float
    end: float
    label: str
    reason: str
    score: float
    beat_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QualityContract:
    contract_id: str
    quality_tier: str
    priorities: list[str]
    hard_gates: list[str]
    soft_gates: list[str]
    metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VideoUnderstandingGraph:
    version: str
    duration_sec: float
    transcript_excerpt: str
    thesis_excerpt: str
    main_keywords: list[str]
    main_phrases: list[str]
    topic_weights: dict[str, float]
    semantic_beats: list[SemanticBeat]
    retention_moments: list[RetentionMoment]
    scene_cuts: list[float]
    platform_profiles: dict[str, dict[str, float]]
    quality_contract: QualityContract
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "duration_sec": round(self.duration_sec, 3),
            "transcript_excerpt": self.transcript_excerpt,
            "thesis_excerpt": self.thesis_excerpt,
            "main_keywords": list(self.main_keywords),
            "main_phrases": list(self.main_phrases),
            "topic_weights": dict(self.topic_weights),
            "semantic_beats": [beat.to_dict() for beat in self.semantic_beats],
            "retention_moments": [moment.to_dict() for moment in self.retention_moments],
            "scene_cuts": list(self.scene_cuts),
            "platform_profiles": {key: dict(value) for key, value in self.platform_profiles.items()},
            "quality_contract": self.quality_contract.to_dict(),
            "metadata": dict(self.metadata),
        }

    def compact(self, *, beat_limit: int = 16, moment_limit: int = 8) -> dict[str, Any]:
        return {
            "version": self.version,
            "duration_sec": round(self.duration_sec, 3),
            "thesis_excerpt": self.thesis_excerpt,
            "main_keywords": list(self.main_keywords[:18]),
            "main_phrases": list(self.main_phrases[:10]),
            "top_semantic_beats": [beat.to_dict() for beat in self.semantic_beats[:beat_limit]],
            "retention_moments": [moment.to_dict() for moment in self.retention_moments[:moment_limit]],
            "quality_contract": self.quality_contract.to_dict(),
        }


def build_video_understanding_graph(
    *,
    transcript_text: str,
    segments: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
    scene_cuts: list[float] | None = None,
    quality_tier: str = "world_class_local",
    source_context: dict[str, Any] | None = None,
) -> VideoUnderstandingGraph:
    metadata = dict(metadata or {})
    scene_cuts = _clean_scene_cuts(scene_cuts or [])
    duration = _duration_from_sources(segments, metadata)
    text = _clean_text(transcript_text or _segment_text(segments))
    main_keywords = _top_keywords(text, 32)
    main_phrases = _top_phrases(text, 16)
    topic_weights = _topic_weights(main_keywords, segments)
    beats = _build_semantic_beats(segments, topic_weights, duration)
    retention_moments = _build_retention_moments(beats, duration)
    contract = _quality_contract(quality_tier=quality_tier)
    graph_metadata = {
        "source_context": dict(source_context or {}),
        "segment_count": len(segments),
        "width": _as_int(metadata.get("width")),
        "height": _as_int(metadata.get("height")),
        "fps": _as_float(metadata.get("fps"), 0.0),
        "has_transcript": bool(text),
    }
    return VideoUnderstandingGraph(
        version=GRAPH_VERSION,
        duration_sec=round(duration, 3),
        transcript_excerpt=_truncate(text, 1600),
        thesis_excerpt=_thesis_excerpt(text, segments),
        main_keywords=main_keywords,
        main_phrases=main_phrases,
        topic_weights=topic_weights,
        semantic_beats=beats,
        retention_moments=retention_moments,
        scene_cuts=scene_cuts,
        platform_profiles={key: dict(value) for key, value in PLATFORM_PROFILE.items()},
        quality_contract=contract,
        metadata=graph_metadata,
    )


def graph_to_video_context(graph: VideoUnderstandingGraph) -> dict[str, Any]:
    opening_beats = [beat for beat in graph.semantic_beats if beat.start <= max(8.0, graph.duration_sec * 0.16)]
    closing_beats = [beat for beat in graph.semantic_beats if beat.start >= max(0.0, graph.duration_sec * 0.78)]
    return {
        "creative_graph_version": graph.version,
        "duration": graph.duration_sec,
        "segment_count": int(graph.metadata.get("segment_count") or len(graph.semantic_beats)),
        "transcript_excerpt": graph.transcript_excerpt,
        "thesis_excerpt": graph.thesis_excerpt,
        "main_keywords": list(graph.main_keywords),
        "main_phrases": list(graph.main_phrases),
        "core_keywords": list(dict.fromkeys([*graph.main_keywords[:18], *_beat_keywords(opening_beats), *_beat_keywords(closing_beats)]))[:24],
        "opening_keywords": _beat_keywords(opening_beats)[:12],
        "closing_keywords": _beat_keywords(closing_beats)[:12],
        "keyword_weights": dict(graph.topic_weights),
        "retention_moments": [moment.to_dict() for moment in graph.retention_moments[:8]],
        "quality_contract": graph.quality_contract.to_dict(),
    }


def candidate_graph_signals(
    graph: VideoUnderstandingGraph,
    *,
    start: float,
    end: float,
    text: str = "",
    source_ranges: list[dict[str, Any]] | None = None,
    target_platform: str = "youtube_shorts",
) -> dict[str, float | list[str]]:
    ranges = _source_ranges(start, end, source_ranges)
    overlapping = _beats_for_ranges(graph.semantic_beats, ranges)
    tokens = _tokens(text or " ".join(beat.text for beat in overlapping))
    duration = sum(max(0.0, float(item["end"]) - float(item["start"])) for item in ranges)
    profile = PLATFORM_PROFILE.get(target_platform, PLATFORM_PROFILE["youtube_shorts"])
    hook_energy = _role_energy(overlapping, {"hook", "tension", "quote"})
    payoff_energy = _role_energy(overlapping, {"payoff", "proof"})
    proof_energy = _role_energy(overlapping, {"proof"})
    tension_energy = _role_energy(overlapping, {"tension"})
    topic_alignment = _weighted_overlap(tokens, graph.topic_weights)
    visual_opportunity = _visual_opportunity(tokens, overlapping)
    retention_overlap = _retention_overlap(graph.retention_moments, ranges)
    duration_fit = _duration_fit(duration, float(profile["ideal_duration"]), low=12.0, high=90.0)
    words_per_sec = len(tokens) / max(duration, 0.1)
    pace_fit = _bounded(1.0 - abs(words_per_sec - 3.2) / float(profile["max_caption_wps"]))
    continuity_risk = _bounded(
        (0.20 if overlapping and overlapping[0].role in {"support", "payoff"} else 0.0)
        + (0.18 if payoff_energy < 0.26 else 0.0)
        + (0.14 if topic_alignment < 0.30 else 0.0)
        + (0.12 if duration < 14.0 else 0.0)
    )
    retention_score = _bounded(
        hook_energy * 0.22
        + payoff_energy * 0.20
        + proof_energy * 0.13
        + tension_energy * 0.10
        + topic_alignment * 0.16
        + visual_opportunity * 0.10
        + retention_overlap * 0.06
        + duration_fit * 0.07
        + pace_fit * 0.06
        - continuity_risk * 0.16
    )
    return {
        "graph_retention_score": round(retention_score, 4),
        "graph_hook_energy": round(hook_energy, 4),
        "graph_payoff_energy": round(payoff_energy, 4),
        "graph_proof_energy": round(proof_energy, 4),
        "graph_tension_energy": round(tension_energy, 4),
        "graph_topic_alignment": round(topic_alignment, 4),
        "graph_visual_opportunity": round(visual_opportunity, 4),
        "graph_retention_overlap": round(retention_overlap, 4),
        "graph_duration_fit": round(duration_fit, 4),
        "graph_pace_fit": round(pace_fit, 4),
        "graph_continuity_risk": round(continuity_risk, 4),
        "graph_beat_ids": [beat.beat_id for beat in overlapping[:8]],
    }


def annotate_visual_cards_with_graph(
    cards: list[dict[str, Any]],
    graph: VideoUnderstandingGraph,
) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for card in cards:
        start = _as_float(card.get("start"), 0.0)
        end = _as_float(card.get("end"), start)
        text = " ".join(
            str(card.get(key) or "")
            for key in ("sentence_text", "context_text", "headline", "emphasis_text")
        )
        signals = candidate_graph_signals(graph, start=start, end=end, text=text)
        visual_opportunity = float(signals["graph_visual_opportunity"])
        retention = float(signals["graph_retention_score"])
        topic_alignment = float(signals["graph_topic_alignment"])
        normalized = dict(card)
        normalized["creative_graph_signals"] = signals
        normalized["priority"] = round(
            _as_float(card.get("priority"), 0.0)
            + (retention * 10.0)
            + (visual_opportunity * 8.0)
            + (topic_alignment * 5.0),
            3,
        )
        annotated.append(normalized)
    return annotated


def build_color_grade_quality_contract(
    *,
    look: str,
    intensity: float,
    metadata: dict[str, Any] | None = None,
    graph: VideoUnderstandingGraph | None = None,
) -> dict[str, Any]:
    metadata = dict(metadata or {})
    dimensions = {
        "protect_skin_tones": 1.0,
        "avoid_shadow_crush": 0.92,
        "avoid_highlight_clip": 0.94,
        "temporal_consistency": 0.90,
        "preserve_brand_colors": 0.84,
        "subtitle_and_visual_legibility": 0.86,
    }
    normalized_look = str(look or "auto").strip().lower()
    if normalized_look in {"cinematic", "punchy", "vibrant"}:
        dimensions["avoid_over_saturation"] = 0.92
        dimensions["protect_skin_tones"] = 1.0
    if float(intensity) >= 1.15:
        dimensions["avoid_over_saturation"] = max(dimensions.get("avoid_over_saturation", 0.0), 0.96)
        dimensions["avoid_noise_amplification"] = 0.88
    return {
        "contract_id": "color-grade-program-v2-local",
        "graph_version": graph.version if graph else GRAPH_VERSION,
        "look": normalized_look,
        "intensity": round(float(intensity), 3),
        "quality_tier": "world_class_local",
        "hard_gates": [
            "valid_output_file",
            "audio_stream_preserved_when_present",
            "no_extreme_shadow_or_highlight_clipping",
            "shot_validation_passes_or_records_actionable_warning",
        ],
        "soft_gates": [
            "skin_tone_drift_bounded",
            "shot_to_shot_grade_continuity",
            "legible_captions_and_generated_visuals_after_grade",
            "source_that_is_already_graded_receives_light_touch",
        ],
        "metrics": dimensions,
        "media_context": {
            "width": _as_int(metadata.get("width")),
            "height": _as_int(metadata.get("height")),
            "fps": _as_float(metadata.get("fps"), 0.0),
            "duration_sec": _as_float(metadata.get("duration_sec"), 0.0),
        },
    }


def _quality_contract(*, quality_tier: str) -> QualityContract:
    return QualityContract(
        contract_id="creative-quality-v1-local",
        quality_tier=quality_tier,
        priorities=[
            "publishable_first_output",
            "source_truthfulness",
            "platform_retention",
            "visual_clarity",
            "tasteful_motion",
            "color_and_audio_integrity",
        ],
        hard_gates=[
            "no_source_file_mutation",
            "timestamped_decisions_are_replayable",
            "rendered_outputs_validate",
            "shorts_preserve_context_and_payoff",
            "visuals_do_not_cover_captions_or_required_subjects",
            "color_grade_avoids_destructive_clipping",
        ],
        soft_gates=[
            "portfolio_diversity",
            "high_signal_visual_density",
            "style_continuity",
            "platform_specific_caption_pacing",
            "explainable_model_and_deterministic_scores",
        ],
        metrics={
            "min_short_context_score": 0.66,
            "min_visual_opportunity_score": 0.48,
            "min_render_quality_score": 0.72,
            "min_color_validation_score": 0.68,
        },
    )


def _build_semantic_beats(
    segments: list[dict[str, Any]],
    topic_weights: dict[str, float],
    duration: float,
) -> list[SemanticBeat]:
    beats: list[SemanticBeat] = []
    for index, segment in enumerate(segments, start=1):
        text = _clean_text(segment.get("text"))
        if not text:
            continue
        start = _as_float(segment.get("start"), 0.0)
        end = max(start + 0.05, _as_float(segment.get("end"), start + 0.8))
        tokens = _tokens(text)
        signals = {
            "hook": _term_score(tokens, HOOK_TERMS) + (0.18 if "?" in text else 0.0),
            "payoff": _term_score(tokens, PAYOFF_TERMS),
            "proof": _term_score(tokens, PROOF_TERMS) + min(len(re.findall(r"\b\d+(?:\.\d+)?%?\b", text)) * 0.18, 0.48),
            "tension": _term_score(tokens, TENSION_TERMS),
            "visual": _visual_opportunity(tokens, []),
            "topic": _weighted_overlap(tokens, topic_weights),
            "opening": 1.0 if start <= max(4.0, duration * 0.08) else 0.0,
        }
        role = _beat_role(signals)
        score = _bounded(
            signals["hook"] * 0.20
            + signals["payoff"] * 0.20
            + signals["proof"] * 0.16
            + signals["tension"] * 0.14
            + signals["visual"] * 0.12
            + signals["topic"] * 0.14
            + signals["opening"] * 0.04
        )
        beats.append(
            SemanticBeat(
                beat_id=f"beat_{index:04d}",
                start=round(start, 3),
                end=round(end, 3),
                text=_truncate(text, 240),
                role=role,
                score=round(score, 4),
                keywords=_candidate_keywords(text, 8),
                signals={key: round(_bounded(value), 4) for key, value in signals.items()},
            )
        )
    return sorted(beats, key=lambda beat: (beat.start, beat.end))


def _build_retention_moments(beats: list[SemanticBeat], duration: float) -> list[RetentionMoment]:
    candidates = [
        beat
        for beat in beats
        if beat.score >= 0.42 or beat.role in {"hook", "proof", "tension", "payoff"}
    ]
    ranked = sorted(candidates, key=lambda beat: (beat.score, -beat.start), reverse=True)
    selected: list[SemanticBeat] = []
    for beat in ranked:
        if any(abs(beat.start - existing.start) < 3.0 for existing in selected):
            continue
        selected.append(beat)
        if len(selected) >= max(6, min(18, round(max(duration, 1.0) / 18.0))):
            break
    moments: list[RetentionMoment] = []
    for index, beat in enumerate(sorted(selected, key=lambda item: item.start), start=1):
        label = {
            "hook": "Opening hook",
            "proof": "Proof beat",
            "tension": "Tension turn",
            "payoff": "Payoff",
            "visual": "Visual explanation",
        }.get(beat.role, "High-signal beat")
        moments.append(
            RetentionMoment(
                moment_id=f"retention_{index:03d}",
                start=round(max(0.0, beat.start - 0.3), 3),
                end=round(max(beat.end, beat.start + 0.3), 3),
                label=label,
                reason=_truncate(beat.text, 120),
                score=beat.score,
                beat_ids=[beat.beat_id],
            )
        )
    return moments


def _beat_role(signals: dict[str, float]) -> str:
    ordered = sorted(
        ((key, value) for key, value in signals.items() if key not in {"topic", "opening"}),
        key=lambda item: item[1],
        reverse=True,
    )
    best_key, best_score = ordered[0] if ordered else ("support", 0.0)
    if best_score < 0.20 and signals.get("topic", 0.0) >= 0.36:
        return "support"
    if best_score < 0.20:
        return "support"
    return "visual" if best_key == "visual" else best_key


def _source_ranges(
    start: float,
    end: float,
    source_ranges: list[dict[str, Any]] | None,
) -> list[dict[str, float]]:
    ranges: list[dict[str, float]] = []
    for item in source_ranges or []:
        if not isinstance(item, dict):
            continue
        range_start = _as_float(item.get("start"), 0.0)
        range_end = _as_float(item.get("end"), range_start)
        if range_end > range_start:
            ranges.append({"start": range_start, "end": range_end})
    if ranges:
        return ranges
    return [{"start": float(start), "end": max(float(start), float(end))}]


def _beats_for_ranges(beats: list[SemanticBeat], ranges: list[dict[str, float]]) -> list[SemanticBeat]:
    selected: list[SemanticBeat] = []
    seen: set[str] = set()
    for item in ranges:
        start = item["start"]
        end = item["end"]
        for beat in beats:
            if beat.beat_id in seen:
                continue
            if beat.end > start and beat.start < end:
                selected.append(beat)
                seen.add(beat.beat_id)
    return sorted(selected, key=lambda beat: beat.start)


def _role_energy(beats: list[SemanticBeat], roles: set[str]) -> float:
    if not beats:
        return 0.0
    matching = [beat.score for beat in beats if beat.role in roles]
    if not matching:
        return 0.0
    return _bounded((sum(matching) / max(len(beats), 1)) + (max(matching) * 0.35))


def _retention_overlap(moments: list[RetentionMoment], ranges: list[dict[str, float]]) -> float:
    score = 0.0
    for item in ranges:
        start = item["start"]
        end = item["end"]
        for moment in moments:
            overlap = max(0.0, min(end, moment.end) - max(start, moment.start))
            if overlap > 0:
                score = max(score, min(moment.score + (overlap / max(moment.end - moment.start, 0.1)) * 0.18, 1.0))
    return _bounded(score)


def _visual_opportunity(tokens: list[str], beats: list[SemanticBeat]) -> float:
    if not tokens and not beats:
        return 0.0
    token_set = set(tokens)
    numeric = min(sum(1 for token in tokens if re.fullmatch(r"\d+(?:\.\d+)?%?", token)), 4) * 0.10
    visual_terms = min(len(token_set & VISUAL_TERMS), 6) * 0.09
    proof = min(len(token_set & PROOF_TERMS), 4) * 0.07
    beat_visual = max((float(beat.signals.get("visual", 0.0)) for beat in beats), default=0.0) * 0.32
    return _bounded(0.18 + numeric + visual_terms + proof + beat_visual)


def _duration_fit(duration: float, ideal: float, *, low: float, high: float) -> float:
    bounded = min(max(duration, low), high)
    tolerance = max(abs(ideal - low), abs(high - ideal), 8.0)
    return _bounded(1.0 - abs(bounded - ideal) / tolerance)


def _topic_weights(keywords: list[str], segments: list[dict[str, Any]]) -> dict[str, float]:
    opening_text = _segment_text(segments[: max(1, min(6, len(segments) // 5 or 1))])
    closing_text = _segment_text(segments[-max(1, min(6, len(segments) // 5 or 1)):]) if segments else ""
    opening_tokens = set(_tokens(opening_text))
    closing_tokens = set(_tokens(closing_text))
    weights: dict[str, float] = {}
    for index, keyword in enumerate(keywords):
        weight = 1.0 + (len(keywords) - index) / max(len(keywords), 1)
        if keyword in opening_tokens:
            weight += 0.35
        if keyword in closing_tokens:
            weight += 0.22
        weights[keyword] = round(weight, 4)
    return weights


def _weighted_overlap(tokens: list[str], topic_weights: dict[str, float]) -> float:
    token_set = {token for token in tokens if token not in STOPWORDS and len(token) >= 3}
    if not token_set or not topic_weights:
        return 0.0
    matched = sum(float(weight) for token, weight in topic_weights.items() if token in token_set)
    possible = sum(sorted((float(value) for value in topic_weights.values()), reverse=True)[: max(1, min(len(token_set), 12))])
    return _bounded(matched / max(possible, 0.001))


def _term_score(tokens: list[str], terms: set[str]) -> float:
    if not tokens:
        return 0.0
    hits = len(set(tokens) & terms)
    return _bounded(hits / 4.0)


def _beat_keywords(beats: list[SemanticBeat]) -> list[str]:
    keywords: list[str] = []
    for beat in beats:
        for keyword in beat.keywords:
            if keyword not in keywords:
                keywords.append(keyword)
    return keywords


def _clean_scene_cuts(values: list[float]) -> list[float]:
    cuts: list[float] = []
    for value in values:
        number = _as_float(value, -1.0)
        if number >= 0.0:
            cuts.append(round(number, 3))
    return sorted(set(cuts))


def _duration_from_sources(segments: list[dict[str, Any]], metadata: dict[str, Any]) -> float:
    duration = _as_float(metadata.get("duration_sec"), 0.0)
    if duration > 0:
        return duration
    if not segments:
        return 0.0
    starts = [_as_float(segment.get("start"), 0.0) for segment in segments]
    ends = [_as_float(segment.get("end"), 0.0) for segment in segments]
    return max(0.0, max(ends) - min(starts))


def _thesis_excerpt(text: str, segments: list[dict[str, Any]]) -> str:
    if segments:
        opening = _segment_text(segments[: max(1, min(8, len(segments)))])
        if opening:
            return _truncate(opening, 520)
    return _truncate(text, 520)


def _segment_text(segments: list[dict[str, Any]]) -> str:
    return " ".join(_clean_text(segment.get("text")) for segment in segments if _clean_text(segment.get("text"))).strip()


def _candidate_keywords(text: str, limit: int) -> list[str]:
    keywords: list[str] = []
    for token in _tokens(text):
        if token in STOPWORDS or len(token) < 3:
            continue
        if token not in keywords:
            keywords.append(token)
        if len(keywords) >= limit:
            break
    return keywords


def _top_keywords(text: str, limit: int) -> list[str]:
    counts: dict[str, int] = {}
    for token in _tokens(text):
        if token in STOPWORDS or len(token) < 3:
            continue
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))
    return [keyword for keyword, _count in ranked[:limit]]


def _top_phrases(text: str, limit: int) -> list[str]:
    tokens = [token for token in _tokens(text) if token not in STOPWORDS and len(token) >= 3]
    counts: dict[str, int] = {}
    for first, second in zip(tokens, tokens[1:]):
        if first == second:
            continue
        phrase = f"{first} {second}"
        counts[phrase] = counts.get(phrase, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))
    return [phrase for phrase, _count in ranked[:limit]]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9']+", str(text or "").lower())


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _truncate(text: str, limit: int) -> str:
    cleaned = _clean_text(text)
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(limit - 3, 0)].rstrip(" ,.;:-") + "..."


def _bounded(value: float, low: float = 0.0, high: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return low
    if math.isnan(number) or math.isinf(number):
        return low
    return max(low, min(number, high))


def _as_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


__all__ = [
    "GRAPH_VERSION",
    "QualityContract",
    "RetentionMoment",
    "SemanticBeat",
    "VideoUnderstandingGraph",
    "annotate_visual_cards_with_graph",
    "build_color_grade_quality_contract",
    "build_video_understanding_graph",
    "candidate_graph_signals",
    "graph_to_video_context",
]
