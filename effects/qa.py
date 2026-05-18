from __future__ import annotations

from typing import Any

from effects.schema import EffectPlan


def validate_effect_output(
    source_metadata: dict[str, Any],
    output_metadata: dict[str, Any],
    plan: EffectPlan,
) -> dict[str, Any]:
    warnings: list[str] = []
    source_duration = _as_float(source_metadata.get("duration_sec"), 0.0)
    output_duration = _as_float(output_metadata.get("duration_sec"), 0.0)
    duration_delta = abs(source_duration - output_duration)
    tolerance = max(0.18, source_duration * 0.006)
    source_width = int(source_metadata.get("width") or 0)
    source_height = int(source_metadata.get("height") or 0)
    output_width = int(output_metadata.get("width") or 0)
    output_height = int(output_metadata.get("height") or 0)
    if duration_delta > tolerance:
        warnings.append(f"Output duration changed by {duration_delta:.3f}s.")
    if source_width and output_width and source_width != output_width:
        warnings.append(f"Output width changed from {source_width} to {output_width}.")
    if source_height and output_height and source_height != output_height:
        warnings.append(f"Output height changed from {source_height} to {output_height}.")
    if not plan.effects:
        warnings.append("No effects were rendered.")
    return {
        "passed": not warnings,
        "warnings": warnings,
        "source_duration_sec": round(source_duration, 3),
        "output_duration_sec": round(output_duration, 3),
        "duration_delta_sec": round(duration_delta, 3),
        "effect_count": len(plan.effects),
        "compiler_version": plan.compiler_version,
    }


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
