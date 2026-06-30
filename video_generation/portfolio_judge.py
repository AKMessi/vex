from __future__ import annotations

from collections import Counter
from typing import Any

from video_generation.director import DirectorPackage, generic_script_patterns_found
from video_generation.models import BeatGraph, ScriptPlan, VideoGenerationRequest


PORTFOLIO_JUDGE_VERSION = "hyperframes-video-portfolio-judge-v1"


def judge_generation_portfolio(
    *,
    request: VideoGenerationRequest,
    plan: ScriptPlan,
    beat_graph: BeatGraph,
    cinematic_plan: Any | None,
    motion_plan: Any | None,
    director_package: DirectorPackage | None,
    video_skill_graph: Any | None = None,
) -> dict[str, Any]:
    issues: list[str] = []
    warnings: list[str] = []
    accepted_count = int(getattr(cinematic_plan, "accepted_count", 0) or 0)
    beat_count = len(beat_graph.beats)
    semantic_coverage = accepted_count / max(beat_count, 1)
    policy = dict(director_package.portfolio_policy) if director_package else {}
    min_semantic = float(policy.get("min_semantic_coverage") or 0.82)
    if semantic_coverage < min_semantic:
        issues.append("portfolio_semantic_coverage_below_director_floor")

    medium_families = _medium_families(cinematic_plan)
    medium_run = _longest_run(medium_families)
    max_run = int(policy.get("max_repeated_medium_run") or 2)
    if medium_run > max_run:
        issues.append("portfolio_repeats_visual_medium_too_often")
    elif len(set(medium_families)) < max(2, min(beat_count, 3)) and beat_count >= 3:
        warnings.append("portfolio_visual_medium_variety_low")

    signatures = _world_signatures(cinematic_plan)
    duplicate_signatures = [
        signature
        for signature, count in Counter(signatures).items()
        if signature and count > 1
    ]
    if duplicate_signatures and bool(policy.get("require_distinct_world_signatures", True)):
        issues.append("portfolio_duplicate_visual_world_signatures")

    tournament_reports = _tournament_reports(cinematic_plan)
    if bool(policy.get("require_tournament_records", True)) and len(tournament_reports) < accepted_count:
        issues.append("portfolio_missing_beat_tournament_records")
    failed_tournaments = [
        report
        for report in tournament_reports
        if not bool(report.get("passed", False))
    ]
    if failed_tournaments:
        issues.append("portfolio_contains_failed_beat_tournament")
    low_margin = [
        report
        for report in tournament_reports
        if 0 < float(report.get("selected_score") or 0.0) < 0.68
    ]
    if low_margin:
        warnings.append("portfolio_contains_low_margin_variant_selection")

    generic_patterns = generic_script_patterns_found(plan.narration)
    if generic_patterns:
        issues.append("portfolio_script_contains_generic_generation_patterns")

    native_count = int(getattr(motion_plan, "native_composition_count", 0) or 0)
    native_coverage = native_count / max(beat_count, 1)
    min_native = float(policy.get("min_native_motion_coverage") or 0.82)
    if beat_count >= 2 and native_coverage < min_native:
        issues.append("portfolio_native_motion_coverage_below_director_floor")

    cue_count = int(getattr(motion_plan, "audio_cue_count", 0) or 0)
    if request.generate_audio and cue_count < max(1, beat_count):
        warnings.append("portfolio_audio_motion_cues_too_sparse")

    contract_count = len(getattr(director_package, "beat_contracts", []) or [])
    if director_package is None or contract_count < beat_count:
        issues.append("portfolio_director_contract_incomplete")

    skill_graph_payload = (
        video_skill_graph.to_dict()
        if hasattr(video_skill_graph, "to_dict")
        else dict(video_skill_graph or {})
    )
    if skill_graph_payload:
        skill_coverage = float(skill_graph_payload.get("coverage") or 0.0)
        min_skill_coverage = float(
            (skill_graph_payload.get("portfolio_constraints") or {}).get("min_skill_coverage")
            or 0.78
        )
        if not bool(skill_graph_payload.get("passed")):
            issues.append("portfolio_video_skill_graph_failed")
        if skill_coverage < min_skill_coverage:
            issues.append("portfolio_video_skill_coverage_below_floor")
        assignments = [
            item
            for item in skill_graph_payload.get("beat_assignments") or []
            if isinstance(item, dict)
        ]
        if len(assignments) < beat_count:
            issues.append("portfolio_video_skill_assignments_incomplete")
        if not skill_graph_payload.get("production_skill_id"):
            issues.append("portfolio_video_production_skill_missing")
        if beat_count >= 3 and len({str(item.get("skill_id") or "") for item in assignments}) < 2:
            warnings.append("portfolio_video_skill_variety_low")
    elif beat_count >= 2:
        issues.append("portfolio_video_skill_graph_missing")

    score = 1.0
    score -= min(len(issues) * 0.18, 0.72)
    score -= min(len(warnings) * 0.035, 0.18)
    score += min(semantic_coverage * 0.04, 0.04)
    if len(set(medium_families)) >= min(beat_count, 3):
        score += 0.03
    score = round(max(0.0, min(score, 1.0)), 4)
    return {
        "version": PORTFOLIO_JUDGE_VERSION,
        "passed": not issues,
        "score": score,
        "issues": _unique(issues),
        "warnings": _unique(warnings),
        "evidence": {
            "beat_count": beat_count,
            "accepted_count": accepted_count,
            "semantic_coverage": round(semantic_coverage, 4),
            "native_motion_coverage": round(native_coverage, 4),
            "audio_cue_count": cue_count,
            "medium_families": medium_families,
            "longest_medium_run": medium_run,
            "world_signature_count": len(signatures),
            "duplicate_world_signatures": duplicate_signatures[:5],
            "tournament_count": len(tournament_reports),
            "failed_tournament_count": len(failed_tournaments),
            "low_margin_tournament_count": len(low_margin),
            "generic_script_patterns": generic_patterns,
            "director_contract_count": contract_count,
            "video_skill_graph": {
                "version": skill_graph_payload.get("version") if skill_graph_payload else "",
                "passed": bool(skill_graph_payload.get("passed")) if skill_graph_payload else False,
                "production_skill_id": skill_graph_payload.get("production_skill_id") if skill_graph_payload else "",
                "coverage": skill_graph_payload.get("coverage") if skill_graph_payload else 0.0,
                "assignment_count": skill_graph_payload.get("assignment_count") if skill_graph_payload else 0,
            },
        },
    }


def _medium_families(cinematic_plan: Any | None) -> list[str]:
    result: list[str] = []
    for item in getattr(cinematic_plan, "beat_compositions", []) or []:
        metadata = dict(getattr(item, "metadata", {}) or {})
        world = dict(metadata.get("visual_world_program") or metadata.get("visual_world") or {})
        medium = str(world.get("medium_family") or "").strip()
        if medium:
            result.append(medium)
    return result


def _world_signatures(cinematic_plan: Any | None) -> list[str]:
    result: list[str] = []
    for item in getattr(cinematic_plan, "beat_compositions", []) or []:
        metadata = dict(getattr(item, "metadata", {}) or {})
        world = dict(metadata.get("visual_world_program") or {})
        signature = str(world.get("world_signature") or "").strip()
        if signature:
            result.append(signature)
    return result


def _tournament_reports(cinematic_plan: Any | None) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for item in getattr(cinematic_plan, "beat_compositions", []) or []:
        report = dict(getattr(item, "tournament", {}) or {})
        if report:
            reports.append(report)
    return reports


def _longest_run(values: list[str]) -> int:
    longest = 0
    current = 0
    previous = None
    for value in values:
        if value == previous:
            current += 1
        else:
            previous = value
            current = 1
        longest = max(longest, current)
    return longest


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
