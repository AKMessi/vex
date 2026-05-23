from __future__ import annotations

from typing import Any

from effects.schema import CAMERA_EFFECT_TYPES, STYLE_EFFECT_TYPES, EffectInstance, normalize_effect_type


TYPE_DURATION_LIMITS = {
    "impact_pulse": (0.45, 1.35),
    "snap_reframe": (0.55, 1.65),
    "subtle_shake": (0.35, 0.95),
    "freeze_accent": (0.35, 0.85),
    "punch_in": (0.65, 2.4),
    "punch_out": (0.65, 2.4),
    "slow_push": (1.25, 4.0),
    "micro_pan": (0.9, 3.2),
    "subtitle_highlight": (0.45, 2.4),
    "flash_accent": (0.25, 0.8),
    "vignette": (0.65, 2.8),
    "focus_blur": (0.55, 1.8),
}


def normalize_effects(
    effects: list[EffectInstance],
    *,
    clip_duration: float,
    cooldown_sec: float,
    blocked_ranges: list[tuple[float, float]] | None = None,
) -> list[EffectInstance]:
    blocked_ranges = blocked_ranges or []
    cleaned: list[EffectInstance] = []
    for effect in effects:
        effect_type = normalize_effect_type(effect.effect_type)
        if effect_type not in CAMERA_EFFECT_TYPES and effect_type not in STYLE_EFFECT_TYPES:
            continue
        start_sec = max(0.0, min(float(effect.start), clip_duration))
        end_sec = max(start_sec + 0.08, min(float(effect.end), clip_duration))
        start_sec, end_sec = _clamp_duration(effect_type, start_sec, end_sec, clip_duration)
        if end_sec <= start_sec or _overlaps_any(start_sec, end_sec, blocked_ranges):
            continue
        params = _normalize_params(effect.params)
        modifiers = [
            modifier
            for modifier in (normalize_effect_type(item) for item in effect.modifiers)
            if modifier in STYLE_EFFECT_TYPES and modifier != effect_type
        ][:3]
        cleaned.append(
            EffectInstance(
                effect_id=effect.effect_id,
                effect_type=effect_type,
                start=round(start_sec, 3),
                end=round(end_sec, 3),
                priority=max(0.0, min(float(effect.priority), 1.0)),
                source_card_id=effect.source_card_id,
                reason=effect.reason,
                params=params,
                modifiers=modifiers,
                signals=dict(effect.signals),
            )
        )

    selected: list[EffectInstance] = []
    for effect in sorted(cleaned, key=lambda item: (item.priority, -item.start), reverse=True):
        if any(_ranges_overlap(effect.start - cooldown_sec, effect.end + cooldown_sec, kept.start, kept.end) for kept in selected):
            continue
        selected.append(effect)
    selected = sorted(selected, key=lambda item: item.start)
    resequenced: list[EffectInstance] = []
    for index, effect in enumerate(selected, start=1):
        resequenced.append(
            EffectInstance(
                effect_id=f"effect_{index:03d}",
                effect_type=effect.effect_type,
                start=effect.start,
                end=effect.end,
                priority=effect.priority,
                source_card_id=effect.source_card_id,
                reason=effect.reason,
                params=effect.params,
                modifiers=effect.modifiers,
                signals=effect.signals,
            )
        )
    return resequenced


def _clamp_duration(effect_type: str, start_sec: float, end_sec: float, clip_duration: float) -> tuple[float, float]:
    min_duration, max_duration = TYPE_DURATION_LIMITS.get(effect_type, (0.45, 2.4))
    duration = end_sec - start_sec
    if duration < min_duration:
        pad = (min_duration - duration) / 2.0
        start_sec = max(0.0, start_sec - pad)
        end_sec = min(clip_duration, end_sec + pad)
    if end_sec - start_sec > max_duration:
        center = (start_sec + end_sec) / 2.0
        start_sec = max(0.0, center - max_duration / 2.0)
        end_sec = min(clip_duration, start_sec + max_duration)
    return start_sec, end_sec


def _normalize_params(params: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(params)
    normalized["max_scale"] = round(max(1.0, min(_as_float(normalized.get("max_scale"), 1.08), 1.32)), 3)
    normalized["anchor_x"] = round(max(0.18, min(_as_float(normalized.get("anchor_x"), 0.5), 0.82)), 3)
    normalized["anchor_y"] = round(max(0.18, min(_as_float(normalized.get("anchor_y"), 0.5), 0.82)), 3)
    normalized["shake_amplitude"] = round(max(0.0, min(_as_float(normalized.get("shake_amplitude"), 0.06), 0.12)), 3)
    if str(normalized.get("subtitle_position") or "bottom") not in {"bottom", "center", "top"}:
        normalized["subtitle_position"] = "bottom"
    if str(normalized.get("motion_shape") or "") not in {"ease_hold", "soft_peak", "soft_hold", "ease_in_out_hold"}:
        normalized.pop("motion_shape", None)
    return normalized


def _overlaps_any(start_sec: float, end_sec: float, ranges: list[tuple[float, float]]) -> bool:
    return any(_ranges_overlap(start_sec, end_sec, blocked_start, blocked_end) for blocked_start, blocked_end in ranges)


def _ranges_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> bool:
    return start_a < end_b and end_a > start_b


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
