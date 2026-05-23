from __future__ import annotations

from typing import Any

from shorts.director import ShortsProgram


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
        if plan.continuity_risk >= 70:
            warnings.append(f"{plan.candidate_id} has high continuity risk.")
        if plan.arc_integrity < 34:
            warnings.append(f"{plan.candidate_id} has weak narrative arc integrity.")
    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "candidate_count": len(program.candidates),
        "moment_count": len(program.moments),
        "selected_count": len(program.portfolio.selected_candidate_ids),
        "version": "shorts-director-v2",
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
