from __future__ import annotations

import math
from typing import Any

from shorts.director import EDIT_OPERATION_TYPES, SOURCE_RANGE_ROLES, ShortEditPlan, ShortsProgram


def validate_shorts_program(program: ShortsProgram) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    candidate_ids = {candidate.candidate_id for candidate in program.candidates}
    if not program.candidates:
        errors.append("Shorts program has no candidate plans.")
    if not program.moments:
        warnings.append("Shorts program has no moment graph.")
    for candidate_id in program.portfolio.selected_candidate_ids:
        if candidate_id not in candidate_ids:
            errors.append(f"Portfolio selected unknown candidate {candidate_id}.")
    if len(program.portfolio.selected_candidate_ids) < min(program.portfolio.target_count, len(candidate_ids)):
        warnings.append("Portfolio could not satisfy the requested short count.")
    for plan in program.candidates:
        if plan.duration <= 0:
            errors.append(f"{plan.candidate_id} has non-positive duration.")
        if not plan.source_ranges:
            errors.append(f"{plan.candidate_id} has no source ranges.")
        source_duration = 0.0
        for index, source_range in enumerate(plan.source_ranges, start=1):
            start = _float(source_range.get("start"))
            end = _float(source_range.get("end"))
            if start < 0 or end < 0:
                errors.append(f"{plan.candidate_id} source range {index} has a negative timestamp.")
            if end <= start:
                errors.append(f"{plan.candidate_id} source range {index} ends before it starts.")
            source_duration += max(0.0, end - start)
        if source_duration and abs(source_duration - plan.duration) > max(0.3, plan.duration * 0.03):
            warnings.append(f"{plan.candidate_id} source range duration differs from planned duration.")
        if plan.composition_mode == "remix" and len(plan.source_ranges) < 2:
            warnings.append(f"{plan.candidate_id} is marked remix with fewer than two source ranges.")
        if plan.continuity_risk >= 70:
            warnings.append(f"{plan.candidate_id} has high continuity risk.")
        if plan.arc_integrity < 34:
            warnings.append(f"{plan.candidate_id} has weak narrative arc integrity.")
        edit_plan = program.edit_plans.get(plan.candidate_id)
        if edit_plan is None:
            errors.append(f"{plan.candidate_id} has no typed edit plan.")
            continue
        edit_validation = validate_short_edit_plan(edit_plan)
        errors.extend(f"{plan.candidate_id}: {error}" for error in edit_validation["errors"])
        warnings.extend(f"{plan.candidate_id}: {warning}" for warning in edit_validation["warnings"])
    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "candidate_count": len(program.candidates),
        "moment_count": len(program.moments),
        "selected_count": len(program.portfolio.selected_candidate_ids),
        "version": "shorts-director-v3",
    }


def validate_short_edit_plan(edit_plan: ShortEditPlan | dict[str, Any]) -> dict[str, Any]:
    payload = edit_plan.to_dict() if isinstance(edit_plan, ShortEditPlan) else dict(edit_plan or {})
    errors: list[str] = []
    warnings: list[str] = []
    source_ranges = list(payload.get("source_ranges") or [])
    operations = list(payload.get("operations") or [])
    target_duration = _finite_float(payload.get("target_duration_sec"))
    if not source_ranges:
        errors.append("edit plan has no source ranges")
    if target_duration <= 0:
        errors.append("edit plan has non-positive target duration")
    max_source_ranges = _int((payload.get("remix_policy") or {}).get("max_source_ranges")) or 6
    if len(source_ranges) > max_source_ranges:
        errors.append(f"edit plan has {len(source_ranges)} source ranges, max is {max_source_ranges}")

    total_duration = 0.0
    range_indices: set[int] = set()
    for position, source_range in enumerate(source_ranges, start=1):
        if not isinstance(source_range, dict):
            errors.append(f"source range {position} is not an object")
            continue
        start = _finite_float(source_range.get("start"))
        end = _finite_float(source_range.get("end"))
        duration = _finite_float(source_range.get("duration"))
        role = str(source_range.get("role") or "")
        index = _int(source_range.get("index")) or position
        if index in range_indices:
            errors.append(f"source range {position} duplicates index {index}")
        range_indices.add(index)
        if start < 0 or end < 0:
            errors.append(f"source range {position} has negative timestamps")
        if end <= start:
            errors.append(f"source range {position} ends before it starts")
        computed_duration = max(0.0, end - start)
        if duration and abs(duration - computed_duration) > 0.05:
            warnings.append(f"source range {position} duration differs from timestamps")
        total_duration += computed_duration
        if role not in SOURCE_RANGE_ROLES:
            errors.append(f"source range {position} has unsupported role {role!r}")
        speed = _finite_float(source_range.get("speed"), default=1.0)
        if speed < 0.75 or speed > 1.35:
            errors.append(f"source range {position} speed {speed:.2f} is outside supported bounds")

    if target_duration > 0 and total_duration > 0 and abs(total_duration - target_duration) > max(0.35, target_duration * 0.04):
        warnings.append("edit plan source duration differs from target duration")

    for position, operation in enumerate(operations, start=1):
        if not isinstance(operation, dict):
            errors.append(f"operation {position} is not an object")
            continue
        op_type = str(operation.get("type") or "")
        if op_type not in EDIT_OPERATION_TYPES:
            errors.append(f"operation {position} has unsupported type {op_type!r}")
        source_range_index = operation.get("source_range_index")
        if source_range_index is not None and (_int(source_range_index) not in range_indices):
            errors.append(f"operation {position} references unknown source range {source_range_index}")
        start_sec = operation.get("start_sec")
        end_sec = operation.get("end_sec")
        if start_sec is not None:
            start = _finite_float(start_sec)
            if start < 0:
                errors.append(f"operation {position} has negative start")
            if target_duration > 0 and start > target_duration + 0.05:
                errors.append(f"operation {position} starts after target duration")
        if end_sec is not None:
            end = _finite_float(end_sec)
            if end < 0:
                errors.append(f"operation {position} has negative end")
            if target_duration > 0 and end > target_duration + 0.05:
                errors.append(f"operation {position} ends after target duration")
        if start_sec is not None and end_sec is not None:
            start = _finite_float(start_sec)
            end = _finite_float(end_sec)
            if op_type != "jump_cut" and end <= start:
                errors.append(f"operation {position} ends before it starts")
    first_role = str(source_ranges[0].get("role") or "") if source_ranges and isinstance(source_ranges[0], dict) else ""
    if len(source_ranges) > 1 and first_role not in {"hook", "setup", "quote", "context"}:
        warnings.append("stitched edit does not open with a hook/context beat")
    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "source_range_count": len(source_ranges),
        "operation_count": len(operations),
        "target_duration_sec": round(target_duration, 3) if target_duration else None,
    }


def validate_short_render(
    short_record: dict[str, Any],
    metadata: dict[str, Any],
    edit_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    width = _int(metadata.get("width"))
    height = _int(metadata.get("height"))
    duration = _float(metadata.get("duration_sec"))
    expected_duration = _float(short_record.get("duration"))
    if width and height:
        if width != 1080 or height != 1920:
            errors.append(f"Rendered short is {width}x{height}, expected 1080x1920.")
    else:
        warnings.append("Rendered short metadata did not include usable resolution.")
    if duration and expected_duration and abs(duration - expected_duration) > max(0.35, expected_duration * 0.015):
        warnings.append(f"Rendered duration differs from planned window by {abs(duration - expected_duration):.2f}s.")
    if edit_plan:
        remix_policy = dict(edit_plan.get("remix_policy") or {})
        source_ranges = list(short_record.get("source_ranges") or [])
        if remix_policy.get("enabled") and len(source_ranges) < 2:
            warnings.append("Director remix policy expected multiple source ranges.")
        max_source_ranges = _int(remix_policy.get("max_source_ranges"))
        if max_source_ranges and len(source_ranges) > max_source_ranges:
            errors.append(f"Source range count {len(source_ranges)} exceeds remix policy {max_source_ranges}.")
        policy = dict(edit_plan.get("punch_in_policy") or {})
        max_moments = _int(policy.get("max_moments"))
        punch_in_count = len(short_record.get("punch_in_moments") or [])
        if max_moments and punch_in_count > max_moments:
            warnings.append(f"Punch-in count {punch_in_count} exceeds director policy {max_moments}.")
        if edit_plan.get("caption_density") == "fast" and not short_record.get("captions_path"):
            warnings.append("Director expected fast captions, but no captions path was recorded.")
    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "width": width,
        "height": height,
        "duration_sec": round(duration, 3) if duration else None,
    }


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value if value is not None else default)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number
