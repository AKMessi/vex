from __future__ import annotations

from typing import Any

from effects.schema import normalize_effect_type


DENSITY_PROFILES = {
    "low": {"per_minute": 3.0, "threshold": 62.0, "cooldown": 4.0},
    "medium": {"per_minute": 6.0, "threshold": 52.0, "cooldown": 2.4},
    "high": {"per_minute": 10.0, "threshold": 42.0, "cooldown": 1.25},
}

INTENSITY_VALUES = {
    "subtle": 0.72,
    "low": 0.72,
    "medium": 1.0,
    "balanced": 1.0,
    "high": 1.18,
    "strong": 1.28,
}


def density_profile(value: str) -> dict[str, float]:
    return dict(DENSITY_PROFILES.get(str(value or "medium").strip().lower(), DENSITY_PROFILES["medium"]))


def intensity_value(value: object) -> float:
    if isinstance(value, (int, float)):
        return max(0.45, min(float(value), 1.35))
    return INTENSITY_VALUES.get(str(value or "medium").strip().lower(), 1.0)


def params_for_effect(
    effect_type: str,
    card: dict[str, Any],
    *,
    intensity: float,
    subtitle_position: str,
    include_style_effects: bool,
    subtitle_highlight_enabled: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    normalized = normalize_effect_type(effect_type)
    params = {
        "easing": "ease_out_cubic",
        "anchor_x": 0.5,
        "anchor_y": 0.5,
        "subtitle_position": subtitle_position if subtitle_position in {"bottom", "center", "top"} else "bottom",
        "subtitle_highlight_enabled": bool(subtitle_highlight_enabled),
    }
    if normalized == "punch_in":
        params["max_scale"] = round(1.08 + 0.055 * intensity, 3)
    elif normalized == "punch_out":
        params["max_scale"] = round(1.07 + 0.045 * intensity, 3)
    elif normalized == "slow_push":
        params["max_scale"] = round(1.055 + 0.045 * intensity, 3)
    elif normalized == "impact_pulse":
        params["max_scale"] = round(1.045 + 0.055 * intensity, 3)
    elif normalized == "micro_pan":
        params["max_scale"] = round(1.045 + 0.035 * intensity, 3)
        params["pan_axis"] = "x"
    elif normalized == "snap_reframe":
        params["max_scale"] = round(1.075 + 0.045 * intensity, 3)
        params["pan_axis"] = "x"
    elif normalized == "subtle_shake":
        params["max_scale"] = round(1.045 + 0.035 * intensity, 3)
        params["shake_amplitude"] = round(0.055 + 0.025 * intensity, 3)
    elif normalized == "freeze_accent":
        params["max_scale"] = round(1.035 + 0.03 * intensity, 3)
    else:
        params["max_scale"] = 1.0

    modifiers = style_modifiers_for_card(
        card,
        effect_type=normalized,
        include_style_effects=include_style_effects,
        subtitle_highlight_enabled=subtitle_highlight_enabled,
    )
    return params, modifiers


def style_modifiers_for_card(
    card: dict[str, Any],
    *,
    effect_type: str,
    include_style_effects: bool,
    subtitle_highlight_enabled: bool = False,
) -> list[str]:
    if not include_style_effects:
        return []
    signals = dict(card.get("signals") or {})
    modifiers: list[str] = []
    if subtitle_highlight_enabled and effect_type not in {"subtitle_highlight", "flash_accent"}:
        modifiers.append("subtitle_highlight")
    if int(signals.get("numeric_hits") or 0) or int(signals.get("hook_hits") or 0) or int(signals.get("emphasis_hits") or 0) >= 2:
        modifiers.append("flash_accent")
    if int(signals.get("payoff_hits") or 0) or int(signals.get("conclusion_hits") or 0):
        modifiers.append("vignette")
    if int(signals.get("focus_hits") or 0) and effect_type not in {"subtle_shake", "freeze_accent"}:
        modifiers.append("focus_blur")
    deduped: list[str] = []
    for modifier in modifiers:
        if modifier not in deduped:
            deduped.append(modifier)
        if len(deduped) >= 3:
            break
    return deduped
