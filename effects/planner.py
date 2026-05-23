from __future__ import annotations

import math
from typing import Any

from effects.context import EffectContext, annotate_cards_with_context
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
    subtitle_highlight_enabled: bool = False,
    blocked_ranges: list[tuple[float, float]] | None = None,
    effect_context: EffectContext | dict[str, Any] | None = None,
) -> EffectPlan:
    profile = density_profile(density)
    strength = intensity_value(intensity)
    budget = _effect_budget(clip_duration, max_effects=max_effects, per_minute=profile["per_minute"])
    threshold = float(profile["threshold"])
    cards = annotate_cards_with_context(cards, effect_context)
    context_timeline = _context_timeline(effect_context)
    contextual_cooldown = max(float(profile["cooldown"]), float(context_timeline.get("recommended_cooldown_sec") or 0.0))
    ranked = sorted(cards, key=lambda item: (float(item.get("priority") or 0.0), -float(item.get("start") or 0.0)), reverse=True)
    selected_cards, fallback_used = _select_candidate_cards(ranked, threshold=threshold, budget=budget)
    candidates: list[EffectInstance] = []
    used_specials: set[str] = set()
    for card in selected_cards:
        score = float(card.get("priority") or 0.0)
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
            subtitle_highlight_enabled=subtitle_highlight_enabled,
        )
        params, modifiers = _apply_contextual_motion_profile(
            params,
            modifiers,
            card,
            effect_type=effect_type,
            intensity=strength,
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
        cooldown_sec=contextual_cooldown,
        blocked_ranges=blocked_ranges,
    )[:budget]
    return EffectPlan(
        effects=normalized,
        metadata={
            "planner": "contextual_subtitle_effects_v2" if effect_context else "subtitle_effects_v1",
            "density": density,
            "intensity": strength,
            "include_style_effects": include_style_effects,
            "subtitle_position": subtitle_position,
            "subtitle_highlight_enabled": subtitle_highlight_enabled,
            "timeline_rhythm": context_timeline.get("rhythm", "unknown"),
            "recommended_cooldown_sec": round(contextual_cooldown, 3),
            "candidate_card_count": len(cards),
            "eligible_card_count": len(selected_cards),
            "fallback_used": fallback_used,
            "threshold": threshold,
            "selected_effect_count": len(normalized),
        },
    )


def _effect_budget(clip_duration: float, *, max_effects: int, per_minute: float) -> int:
    requested = max(1, min(int(max_effects or 12), 32))
    duration_budget = max(1, int(math.ceil(max(clip_duration, 1.0) / 60.0 * per_minute)))
    return min(requested, duration_budget)


def _select_candidate_cards(
    ranked: list[dict[str, Any]],
    *,
    threshold: float,
    budget: int,
) -> tuple[list[dict[str, Any]], bool]:
    eligible = [card for card in ranked if float(card.get("priority") or 0.0) >= threshold]
    if eligible:
        return eligible[: max(budget * 3, budget)], False
    if not ranked:
        return [], False
    best_score = float(ranked[0].get("priority") or 0.0)
    fallback_floor = max(12.0, min(best_score, threshold) - 8.0)
    fallback = [
        card
        for card in ranked
        if float(card.get("priority") or 0.0) >= fallback_floor
    ]
    if not fallback:
        fallback = ranked[:1]
    return fallback[: max(budget * 2, 1)], True


def _choose_effect_type(card: dict[str, Any], *, used_specials: set[str]) -> str:
    signals = dict(card.get("signals") or {})
    context = dict(card.get("effect_context") or {})
    recommended = str(context.get("recommended_effect") or "").strip()
    visual_risk = _as_float(context.get("visual_risk"), 0.0)
    scene_stability = _as_float(context.get("scene_stability"), 0.5)
    if recommended:
        if recommended in {"subtle_shake", "freeze_accent"} and recommended in used_specials:
            return "punch_in"
        if visual_risk >= 0.52 and recommended in {"subtle_shake", "snap_reframe", "micro_pan"}:
            return "punch_in"
        if scene_stability < 0.42 and recommended in {"slow_push", "micro_pan"}:
            return "punch_in"
        return recommended
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
    context = dict(card.get("effect_context") or {})
    recommended_duration = _as_float(context.get("recommended_duration_sec"), 0.0)
    if effect_type in {"impact_pulse", "subtle_shake", "freeze_accent"}:
        pre, post = 0.04, 0.12
    elif effect_type in {"slow_push", "micro_pan"}:
        pre, post = 0.12, 0.28
    else:
        pre, post = 0.08, 0.2
    planned_start = max(0.0, start_sec - pre)
    planned_end = min(clip_duration, end_sec + post)
    if recommended_duration > 0.0:
        center = (start_sec + end_sec) / 2.0
        planned_start = max(0.0, center - recommended_duration / 2.0)
        planned_end = min(clip_duration, planned_start + recommended_duration)
    safe_start = _as_float(context.get("safe_start"), 0.0)
    safe_end = _as_float(context.get("safe_end"), clip_duration)
    if safe_end - safe_start >= 0.55:
        planned_start = max(planned_start, safe_start)
        planned_end = min(planned_end, safe_end)
        if planned_end - planned_start < 0.55:
            center = max(safe_start, min((start_sec + end_sec) / 2.0, safe_end))
            planned_start = max(safe_start, center - 0.275)
            planned_end = min(safe_end, planned_start + 0.55)
    return planned_start, planned_end


def _reason_for_card(card: dict[str, Any], effect_type: str) -> str:
    signals = dict(card.get("signals") or {})
    context = dict(card.get("effect_context") or {})
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
    if context.get("narrative_role"):
        reasons.append(f"{context['narrative_role']} beat in {context.get('phase', 'timeline')} phase")
    if _as_float(context.get("scene_stability"), 0.0) >= 0.68:
        reasons.append("stable scene window")
    if _as_float(context.get("visual_risk"), 0.0) >= 0.48:
        reasons.append("risk-aware restrained motion")
    if not reasons:
        reasons.append("high-scoring subtitle beat")
    return f"{effect_type} selected for " + ", ".join(reasons[:3])


def _apply_contextual_motion_profile(
    params: dict[str, Any],
    modifiers: list[str],
    card: dict[str, Any],
    *,
    effect_type: str,
    intensity: float,
) -> tuple[dict[str, Any], list[str]]:
    context = dict(card.get("effect_context") or {})
    if not context:
        return params, modifiers
    adjusted = dict(params)
    next_modifiers = list(modifiers)
    visual_risk = _as_float(context.get("visual_risk"), 0.0)
    scene_stability = _as_float(context.get("scene_stability"), 0.5)
    pacing = str(context.get("pacing") or "balanced")
    role = str(context.get("narrative_role") or "support")

    scale = _as_float(adjusted.get("max_scale"), 1.08)
    if visual_risk >= 0.5 or pacing == "fast":
        scale *= 0.88
        next_modifiers = [modifier for modifier in next_modifiers if modifier != "flash_accent"]
    elif scene_stability >= 0.68 and role in {"hook", "payoff", "process"}:
        scale *= 1.08
    adjusted["max_scale"] = round(max(1.025, min(scale, 1.24)), 3)

    if effect_type in {"punch_in", "punch_out", "slow_push", "micro_pan", "snap_reframe"}:
        adjusted["motion_shape"] = "ease_hold" if pacing != "fast" else "soft_peak"
    adjusted["context_role"] = role
    adjusted["context_phase"] = str(context.get("phase") or "timeline")
    adjusted["context_risk"] = round(visual_risk, 3)
    adjusted["context_scene_stability"] = round(scene_stability, 3)
    adjusted["motion_smoothing"] = "contextual"
    if effect_type == "subtle_shake" and visual_risk >= 0.35:
        adjusted["shake_amplitude"] = round(max(0.02, _as_float(adjusted.get("shake_amplitude"), 0.05) * 0.72), 3)
    if intensity <= 0.85 and "flash_accent" in next_modifiers:
        next_modifiers.remove("flash_accent")
    return adjusted, next_modifiers


def _context_timeline(context: EffectContext | dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(context, EffectContext):
        return dict(context.timeline)
    if isinstance(context, dict) and isinstance(context.get("timeline"), dict):
        return dict(context["timeline"])
    return {}


def _as_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number
