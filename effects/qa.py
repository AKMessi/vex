from __future__ import annotations

import math
from typing import Any

from effects.motion import validate_motion_plan_payload
from effects.schema import EffectPlan


def validate_effect_plan(
    plan: EffectPlan,
    *,
    clip_duration: float,
    scene_cuts: list[float] | None = None,
    blocked_ranges: list[tuple[float, float]] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    duration = _as_float(clip_duration, 0.0)
    scene_cuts = [cut for cut in (_as_float(item, -1.0) for item in (scene_cuts or [])) if cut >= 0.0]
    blocked_ranges = blocked_ranges or []
    effects = sorted(plan.effects, key=lambda effect: float(effect.start))
    previous_end = -1.0
    special_counts: dict[str, int] = {}
    for index, effect in enumerate(effects, start=1):
        label = effect.effect_id or f"effect[{index}]"
        start = _finite_float(effect.start)
        end = _finite_float(effect.end)
        if start is None or end is None:
            errors.append(f"{label} has non-finite timing.")
            continue
        if start < 0.0:
            errors.append(f"{label} starts before 0s.")
        if end <= start:
            errors.append(f"{label} ends before or at its start time.")
        if duration > 0 and end > duration + 0.02:
            errors.append(f"{label} ends after the source duration.")
        if previous_end >= 0 and start < previous_end + 0.18:
            warnings.append(f"{label} starts too close to the previous effect.")
        previous_end = max(previous_end, end)
        if _overlaps_ranges(start, end, blocked_ranges):
            errors.append(f"{label} overlaps a blocked overlay range.")
        if _nearest_cut_distance(start, end, scene_cuts) <= 0.06:
            warnings.append(f"{label} starts or ends directly on a scene cut.")
        scale = _finite_float(effect.params.get("max_scale", effect.params.get("target_scale", 1.0)))
        if scale is None:
            errors.append(f"{label} has a non-finite scale value.")
        elif scale > 1.32:
            warnings.append(f"{label} uses an aggressive scale value of {scale:.2f}.")
        if len(effect.modifiers) > 3:
            warnings.append(f"{label} has more than three style modifiers.")
        if effect.effect_type in {"subtle_shake", "freeze_accent", "flash_accent"}:
            special_counts[effect.effect_type] = special_counts.get(effect.effect_type, 0) + 1

    if duration > 0:
        effects_per_minute = len(effects) / max(duration / 60.0, 0.1)
        if effects_per_minute > 12.0:
            warnings.append(f"Effect density is high at {effects_per_minute:.1f} effects per minute.")
    for effect_type, count in special_counts.items():
        if count > 2:
            warnings.append(f"{effect_type} appears {count} times; production plans should use it sparingly.")
    motion_plan = plan.metadata.get("motion_plan") if isinstance(plan.metadata, dict) else None
    motion_report = None
    if isinstance(motion_plan, dict):
        motion_report = validate_motion_plan_payload(motion_plan).to_dict()
        errors.extend(str(item) for item in motion_report.get("issues") or [])
        warnings.extend(str(item) for item in motion_report.get("warnings") or [])

    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "effect_count": len(effects),
        "compiler_version": plan.compiler_version,
        "motion_director": motion_report,
    }


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


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _overlaps_ranges(start_sec: float, end_sec: float, ranges: list[tuple[float, float]]) -> bool:
    return any(start_sec < blocked_end and end_sec > blocked_start for blocked_start, blocked_end in ranges)


def _nearest_cut_distance(start_sec: float, end_sec: float, scene_cuts: list[float]) -> float:
    if not scene_cuts:
        return 999.0
    return min(min(abs(cut - start_sec), abs(cut - end_sec)) for cut in scene_cuts)
