from __future__ import annotations

import math
from typing import Any

from effects.normalizer import normalize_effects
from effects.presets import density_profile, intensity_value, params_for_effect
from effects.schema import EffectInstance, EffectPlan


def plan_subtitle_effects(
    cards: list[dict[str, Any]],
    clip_duration: float,
    *,
    max_effects: int = 12,
    density: str = "medium",
    intensity: object = "medium",
    include_style_effects: bool = True,
    subtitle_position: str = "bottom",
    blocked_ranges: list[tuple[float, float]] | None = None,
) -> EffectPlan:
    profile = density_profile(density)
    strength = intensity_value(intensity)
    budget = _effect_budget(clip_duration, max_effects=max_effects, per_minute=profile["per_minute"])
    threshold = float(profile["threshold"])
    ranked = sorted(cards, key=lambda item: (float(item.get("priority") or 0.0), -float(item.get("start") or 0.0)), reverse=True)
    candidates: list[EffectInstance] = []
    used_specials: set[str] = set()
    for card in ranked:
        score = float(card.get("priority") or 0.0)
        if score < threshold:
            continue
        effect_type = _choose_effect_type(card, used_specials=used_specials)
        if effect_type in {"freeze_accent", "subtle_shake"}:
            used_specials.add(effect_type)
        start_sec, end_sec = _effect_window(card, effect_type, clip_duration)
        params, modifiers = params_for_effect(
            effect_type,
            card,
            intensity=strength,
            subtitle_position=subtitle_position,
            include_style_effects=include_style_effects,
        )
        candidates.append(
            EffectInstance(
                effect_id="",
                effect_type=effect_type,
                start=start_sec,
                end=end_sec,
                priority=round(score / 100.0, 3),
                source_card_id=str(card.get("card_id") or ""),
                reason=_reason_for_card(card, effect_type),
                params=params,
                modifiers=modifiers,
                signals=dict(card.get("signals") or {}),
            )
        )
        if len(candidates) >= budget * 3:
            break

    normalized = normalize_effects(
        candidates,
        clip_duration=clip_duration,
        cooldown_sec=float(profile["cooldown"]),
        blocked_ranges=blocked_ranges,
    )[:budget]
    return EffectPlan(
        effects=normalized,
        metadata={
            "density": density,
            "intensity": strength,
            "include_style_effects": include_style_effects,
            "subtitle_position": subtitle_position,
            "candidate_card_count": len(cards),
            "selected_effect_count": len(normalized),
        },
    )


def _effect_budget(clip_duration: float, *, max_effects: int, per_minute: float) -> int:
    requested = max(1, min(int(max_effects or 12), 32))
    duration_budget = max(1, int(math.ceil(max(clip_duration, 1.0) / 60.0 * per_minute)))
    return min(requested, duration_budget)


def _choose_effect_type(card: dict[str, Any], *, used_specials: set[str]) -> str:
    signals = dict(card.get("signals") or {})
    text = str(card.get("text") or "").lower()
    if (
        ("wait" in text or "look" in text or "moment" in text)
        and float(signals.get("pause_after") or 0.0) >= 0.28
        and "freeze_accent" not in used_specials
    ):
        return "freeze_accent"
    if (
        any(term in text for term in ("break", "broken", "crazy", "insane", "wrong"))
        and "subtle_shake" not in used_specials
    ):
        return "subtle_shake"
    if int(signals.get("contrast_hits") or 0) > 0:
        return "snap_reframe"
    if int(signals.get("numeric_hits") or 0) > 0 or int(signals.get("emphasis_hits") or 0) >= 2:
        return "impact_pulse"
    if int(signals.get("process_hits") or 0) > 0:
        return "slow_push"
    if int(signals.get("conclusion_hits") or 0) > 0:
        return "punch_out"
    if bool(signals.get("question")) or bool(signals.get("is_opening")) or int(signals.get("hook_hits") or 0) > 0:
        return "punch_in"
    if int(signals.get("focus_hits") or 0) > 0:
        return "micro_pan"
    return "micro_pan"


def _effect_window(card: dict[str, Any], effect_type: str, clip_duration: float) -> tuple[float, float]:
    start_sec = float(card.get("start") or 0.0)
    end_sec = float(card.get("end") or start_sec + 0.8)
    if effect_type in {"impact_pulse", "subtle_shake", "freeze_accent"}:
        pre, post = 0.04, 0.12
    elif effect_type in {"slow_push", "micro_pan"}:
        pre, post = 0.12, 0.28
    else:
        pre, post = 0.08, 0.2
    return max(0.0, start_sec - pre), min(clip_duration, end_sec + post)


def _reason_for_card(card: dict[str, Any], effect_type: str) -> str:
    signals = dict(card.get("signals") or {})
    reasons: list[str] = []
    if signals.get("is_opening"):
        reasons.append("opening hook")
    if signals.get("question"):
        reasons.append("question subtitle")
    if int(signals.get("numeric_hits") or 0):
        reasons.append("numeric claim")
    if int(signals.get("contrast_hits") or 0):
        reasons.append("contrast turn")
    if int(signals.get("payoff_hits") or 0):
        reasons.append("payoff language")
    if int(signals.get("emphasis_hits") or 0):
        reasons.append("emphasis words")
    if float(signals.get("pause_after") or 0.0) >= 0.35:
        reasons.append("pause after subtitle")
    if not reasons:
        reasons.append("high-scoring subtitle beat")
    return f"{effect_type} selected for " + ", ".join(reasons[:3])
