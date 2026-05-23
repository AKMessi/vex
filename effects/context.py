from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EffectContext:
    clip_duration: float
    scene_cuts: list[float]
    blocked_ranges: list[tuple[float, float]]
    timeline: dict[str, Any]
    card_contexts: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "clip_duration": round(float(self.clip_duration), 3),
            "scene_cuts": [round(float(cut), 3) for cut in self.scene_cuts],
            "blocked_ranges": [
                [round(float(start), 3), round(float(end), 3)]
                for start, end in self.blocked_ranges
            ],
            "timeline": dict(self.timeline),
            "card_contexts": {
                card_id: dict(context)
                for card_id, context in self.card_contexts.items()
            },
        }


def build_effect_context(
    cards: list[dict[str, Any]],
    *,
    clip_duration: float,
    scene_cuts: list[float] | None = None,
    blocked_ranges: list[tuple[float, float]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> EffectContext:
    duration = max(0.0, float(clip_duration or 0.0))
    scene_cuts = _valid_scene_cuts(scene_cuts or [], duration)
    blocked_ranges = _valid_ranges(blocked_ranges or [], duration)
    scenes = _scene_ranges(duration, scene_cuts)
    timeline = _timeline_summary(cards, duration, scene_cuts, blocked_ranges, metadata or {})
    card_contexts: dict[str, dict[str, Any]] = {}
    for index, card in enumerate(cards):
        card_id = str(card.get("card_id") or f"card_{index:03d}")
        start_sec = _as_float(card.get("start"), 0.0)
        end_sec = _as_float(card.get("end"), start_sec)
        center = (start_sec + end_sec) / 2.0
        scene_start, scene_end = _scene_for_time(center, scenes)
        scene_distance = _nearest_cut_distance(start_sec, end_sec, scene_cuts)
        local_density = _local_card_density(cards, center)
        signals = dict(card.get("signals") or {})
        words_per_second = _as_float(signals.get("words_per_second"), 0.0)
        pause_before = _as_float(signals.get("pause_before"), _as_float(card.get("pause_before"), 0.0))
        pause_after = _as_float(signals.get("pause_after"), _as_float(card.get("pause_after"), 0.0))
        phase = _timeline_phase(start_sec, duration, signals)
        role = _narrative_role(card, phase)
        pacing = _pacing_class(words_per_second, local_density, pause_before, pause_after)
        scene_stability = _scene_stability(
            scene_start,
            scene_end,
            scene_distance=scene_distance,
            local_density=local_density,
            words_per_second=words_per_second,
        )
        risk = _visual_risk(
            start_sec,
            end_sec,
            scene_distance=scene_distance,
            local_density=local_density,
            words_per_second=words_per_second,
            blocked_ranges=blocked_ranges,
            signals=signals,
        )
        context_score = _context_score(card, role=role, phase=phase, scene_stability=scene_stability, risk=risk, pacing=pacing)
        card_contexts[card_id] = {
            "card_index": index,
            "phase": phase,
            "narrative_role": role,
            "pacing": pacing,
            "local_card_density": round(local_density, 3),
            "words_per_second": round(words_per_second, 3),
            "scene_start": round(scene_start, 3),
            "scene_end": round(scene_end, 3),
            "scene_duration": round(max(scene_end - scene_start, 0.0), 3),
            "scene_distance": round(scene_distance, 3),
            "scene_stability": round(scene_stability, 3),
            "visual_risk": round(risk, 3),
            "context_score": round(context_score, 2),
            "recommended_effect": _recommended_effect(role, pacing=pacing, scene_stability=scene_stability, risk=risk, signals=signals),
            "recommended_duration_sec": round(_recommended_duration(role, pacing=pacing, scene_stability=scene_stability), 3),
            "safe_start": round(max(scene_start + 0.08, 0.0), 3),
            "safe_end": round(min(scene_end - 0.08, duration), 3),
        }
    return EffectContext(
        clip_duration=duration,
        scene_cuts=scene_cuts,
        blocked_ranges=blocked_ranges,
        timeline=timeline,
        card_contexts=card_contexts,
    )


def annotate_cards_with_context(cards: list[dict[str, Any]], context: EffectContext | dict[str, Any] | None) -> list[dict[str, Any]]:
    if context is None:
        return [dict(card) for card in cards]
    card_contexts = _card_contexts(context)
    annotated: list[dict[str, Any]] = []
    for card in cards:
        item = dict(card)
        card_id = str(item.get("card_id") or "")
        card_context = dict(card_contexts.get(card_id) or {})
        if card_context:
            item["raw_priority"] = item.get("priority")
            item["priority"] = card_context.get("context_score", item.get("priority", 0.0))
            item["effect_context"] = card_context
        annotated.append(item)
    return annotated


def _timeline_summary(
    cards: list[dict[str, Any]],
    duration: float,
    scene_cuts: list[float],
    blocked_ranges: list[tuple[float, float]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    word_rates = [
        _as_float((card.get("signals") or {}).get("words_per_second"), 0.0)
        for card in cards
        if _as_float((card.get("signals") or {}).get("words_per_second"), 0.0) > 0
    ]
    avg_wps = sum(word_rates) / len(word_rates) if word_rates else 0.0
    scene_count = len(scene_cuts) + 1 if duration > 0 else 0
    average_scene_duration = duration / scene_count if scene_count else duration
    cards_per_minute = len(cards) / max(duration / 60.0, 0.1)
    if cards_per_minute >= 24 or avg_wps >= 4.6:
        rhythm = "fast"
        recommended_cooldown = 2.85
    elif cards_per_minute <= 10 and avg_wps <= 2.7:
        rhythm = "slow"
        recommended_cooldown = 1.75
    else:
        rhythm = "balanced"
        recommended_cooldown = 2.25
    return {
        "planner": "contextual_effect_director_v1",
        "card_count": len(cards),
        "cards_per_minute": round(cards_per_minute, 3),
        "average_words_per_second": round(avg_wps, 3),
        "rhythm": rhythm,
        "scene_count": scene_count,
        "average_scene_duration": round(average_scene_duration, 3),
        "blocked_range_count": len(blocked_ranges),
        "recommended_cooldown_sec": recommended_cooldown,
        "width": int(metadata.get("width") or 0),
        "height": int(metadata.get("height") or 0),
        "fps": _as_float(metadata.get("fps"), 0.0),
    }


def _valid_scene_cuts(scene_cuts: list[float], duration: float) -> list[float]:
    valid = sorted(
        {
            round(cut, 3)
            for cut in (_as_float(item, -1.0) for item in scene_cuts)
            if 0.08 < cut < max(duration - 0.08, 0.08)
        }
    )
    return valid


def _valid_ranges(ranges: list[tuple[float, float]], duration: float) -> list[tuple[float, float]]:
    valid: list[tuple[float, float]] = []
    for start_sec, end_sec in ranges:
        start = max(0.0, min(_as_float(start_sec, 0.0), duration))
        end = max(start, min(_as_float(end_sec, start), duration))
        if end > start:
            valid.append((round(start, 3), round(end, 3)))
    return valid


def _scene_ranges(duration: float, scene_cuts: list[float]) -> list[tuple[float, float]]:
    boundaries = [0.0, *scene_cuts, max(duration, 0.0)]
    scenes: list[tuple[float, float]] = []
    for start_sec, end_sec in zip(boundaries, boundaries[1:]):
        if end_sec > start_sec:
            scenes.append((start_sec, end_sec))
    return scenes or [(0.0, max(duration, 0.0))]


def _scene_for_time(value: float, scenes: list[tuple[float, float]]) -> tuple[float, float]:
    for start_sec, end_sec in scenes:
        if start_sec <= value <= end_sec:
            return start_sec, end_sec
    return scenes[-1]


def _nearest_cut_distance(start_sec: float, end_sec: float, scene_cuts: list[float]) -> float:
    if not scene_cuts:
        return 999.0
    return min(min(abs(cut - start_sec), abs(cut - end_sec)) for cut in scene_cuts)


def _local_card_density(cards: list[dict[str, Any]], center: float) -> float:
    window = 5.0
    nearby = 0
    for card in cards:
        start_sec = _as_float(card.get("start"), 0.0)
        end_sec = _as_float(card.get("end"), start_sec)
        candidate_center = (start_sec + end_sec) / 2.0
        if abs(candidate_center - center) <= window:
            nearby += 1
    return nearby / (window * 2.0)


def _timeline_phase(start_sec: float, duration: float, signals: dict[str, Any]) -> str:
    if start_sec <= min(6.0, duration * 0.12):
        return "opening"
    if signals.get("conclusion_hits") or signals.get("payoff_hits"):
        return "payoff"
    if duration and start_sec >= duration * 0.84:
        return "closing"
    if duration and start_sec >= duration * 0.62:
        return "resolution"
    return "development"


def _narrative_role(card: dict[str, Any], phase: str) -> str:
    signals = dict(card.get("signals") or {})
    if int(signals.get("numeric_hits") or 0) > 0:
        return "proof"
    if int(signals.get("contrast_hits") or 0) > 0:
        return "contrast"
    if int(signals.get("process_hits") or 0) > 0:
        return "process"
    if bool(signals.get("question")) or int(signals.get("hook_hits") or 0) > 0 or phase == "opening":
        return "hook"
    if int(signals.get("conclusion_hits") or 0) > 0 or int(signals.get("payoff_hits") or 0) > 0 or phase == "closing":
        return "payoff"
    if int(signals.get("focus_hits") or 0) > 0:
        return "focus"
    if int(signals.get("emphasis_hits") or 0) > 0:
        return "emphasis"
    return "support"


def _pacing_class(words_per_second: float, local_density: float, pause_before: float, pause_after: float) -> str:
    if words_per_second >= 4.8 or local_density >= 0.72:
        return "fast"
    if words_per_second <= 2.4 and max(pause_before, pause_after) >= 0.32:
        return "breathing"
    return "balanced"


def _scene_stability(
    scene_start: float,
    scene_end: float,
    *,
    scene_distance: float,
    local_density: float,
    words_per_second: float,
) -> float:
    scene_duration = max(scene_end - scene_start, 0.0)
    score = 0.52
    score += min(scene_duration / 8.0, 0.32)
    score += min(scene_distance / 1.4, 0.24)
    score -= max(local_density - 0.45, 0.0) * 0.24
    score -= max(words_per_second - 4.2, 0.0) * 0.035
    return max(0.0, min(score, 1.0))


def _visual_risk(
    start_sec: float,
    end_sec: float,
    *,
    scene_distance: float,
    local_density: float,
    words_per_second: float,
    blocked_ranges: list[tuple[float, float]],
    signals: dict[str, Any],
) -> float:
    risk = 0.0
    risk += 0.38 if scene_distance <= 0.22 else 0.16 if scene_distance <= 0.5 else 0.0
    risk += max(local_density - 0.48, 0.0) * 0.46
    risk += max(words_per_second - 4.4, 0.0) * 0.08
    risk += 0.18 if _overlaps_ranges(start_sec - 0.18, end_sec + 0.18, blocked_ranges) else 0.0
    risk += 0.12 if signals.get("is_short") else 0.0
    return max(0.0, min(risk, 1.0))


def _context_score(
    card: dict[str, Any],
    *,
    role: str,
    phase: str,
    scene_stability: float,
    risk: float,
    pacing: str,
) -> float:
    base = _as_float(card.get("priority"), 0.0)
    role_bonus = {
        "hook": 7.0,
        "contrast": 8.0,
        "proof": 7.5,
        "process": 4.5,
        "payoff": 6.5,
        "focus": 3.0,
        "emphasis": 4.0,
        "support": 0.0,
    }.get(role, 0.0)
    phase_bonus = 4.0 if phase in {"opening", "payoff", "closing"} else 0.0
    stability_bonus = (scene_stability - 0.5) * 10.0
    pacing_penalty = 4.5 if pacing == "fast" else 0.0
    risk_penalty = risk * 12.0
    return max(0.0, min(base + role_bonus + phase_bonus + stability_bonus - pacing_penalty - risk_penalty, 100.0))


def _recommended_effect(
    role: str,
    *,
    pacing: str,
    scene_stability: float,
    risk: float,
    signals: dict[str, Any],
) -> str:
    if risk >= 0.58:
        return "impact_pulse" if int(signals.get("numeric_hits") or 0) else "punch_in"
    if pacing == "breathing" and scene_stability >= 0.62:
        if role in {"process", "support", "focus"}:
            return "slow_push"
        if role == "payoff":
            return "punch_out"
    if role == "contrast":
        return "snap_reframe" if scene_stability >= 0.55 else "punch_in"
    if role == "proof":
        return "impact_pulse"
    if role == "hook":
        return "punch_in"
    if role == "payoff":
        return "punch_out"
    if role == "focus":
        return "micro_pan" if scene_stability >= 0.55 else "punch_in"
    if role == "process":
        return "slow_push"
    return "micro_pan" if pacing != "fast" else "punch_in"


def _recommended_duration(role: str, *, pacing: str, scene_stability: float) -> float:
    base = {
        "hook": 1.25,
        "contrast": 1.05,
        "proof": 0.86,
        "process": 1.9,
        "payoff": 1.55,
        "focus": 1.35,
        "emphasis": 0.95,
        "support": 1.1,
    }.get(role, 1.1)
    if pacing == "fast":
        base *= 0.82
    elif pacing == "breathing":
        base *= 1.18
    if scene_stability >= 0.72:
        base *= 1.08
    return max(0.55, min(base, 2.6))


def _card_contexts(context: EffectContext | dict[str, Any]) -> dict[str, dict[str, Any]]:
    if isinstance(context, EffectContext):
        return context.card_contexts
    raw = context.get("card_contexts") if isinstance(context, dict) else {}
    return raw if isinstance(raw, dict) else {}


def _overlaps_ranges(start_sec: float, end_sec: float, ranges: list[tuple[float, float]]) -> bool:
    return any(start_sec < blocked_end and end_sec > blocked_start for blocked_start, blocked_end in ranges)


def _as_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number
