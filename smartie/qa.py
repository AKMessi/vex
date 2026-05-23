from __future__ import annotations

import math
from typing import Any

from effects.schema import EffectPlan


def validate_smartie_effect_plan(
    plan: EffectPlan,
    *,
    duration: float | None = None,
    max_scale: float = 2.5,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    smart_zoom_effects = [effect for effect in plan.effects if effect.effect_type == "smart_zoom_segment"]

    previous_end = -1.0
    previous_scale = 1.0
    for index, effect in enumerate(smart_zoom_effects, start=1):
        label = effect.effect_id or f"smart_zoom_segment[{index}]"
        start = _finite_float(effect.start)
        end = _finite_float(effect.end)
        if start is None or end is None:
            errors.append(f"{label} has non-finite timestamps.")
            continue
        if start < 0:
            errors.append(f"{label} starts before 0s.")
        if end <= start:
            errors.append(f"{label} ends before or at its start time.")
        if duration is not None and duration > 0 and end > duration + 0.05:
            errors.append(f"{label} ends after the source duration.")
        if previous_end > start + 0.001:
            warnings.append(f"{label} overlaps the previous Smartie zoom segment.")
        previous_end = max(previous_end, end)

        focus_x = _finite_float(effect.params.get("focus_x"))
        focus_y = _finite_float(effect.params.get("focus_y"))
        target_scale = _finite_float(effect.params.get("target_scale", effect.params.get("max_scale", 1.0)))
        if focus_x is None or focus_y is None:
            errors.append(f"{label} has non-finite focus coordinates.")
        elif not (0.0 <= focus_x <= 1.0 and 0.0 <= focus_y <= 1.0):
            errors.append(f"{label} focus coordinates must be normalized between 0 and 1.")
        if target_scale is None:
            errors.append(f"{label} has a non-finite target scale.")
            continue
        if target_scale < 1.0:
            errors.append(f"{label} target scale cannot be below 1.0.")
        if target_scale > max_scale:
            errors.append(f"{label} target scale {target_scale:.2f} exceeds the allowed {max_scale:.2f}.")
        if abs(target_scale - previous_scale) > 0.85:
            warnings.append(f"{label} has a large zoom jump from the previous Smartie segment.")
        previous_scale = target_scale

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "smart_zoom_segments": len(smart_zoom_effects),
    }


def _finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None
