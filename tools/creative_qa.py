from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any

from tools.creative_intelligence import VideoUnderstandingGraph, candidate_graph_signals


@dataclass(frozen=True)
class CreativeQualityReport:
    report_id: str
    subject_type: str
    score: float
    passed: bool
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["score"] = round(float(self.score), 4)
        payload["metrics"] = {key: round(float(value), 4) for key, value in self.metrics.items()}
        return payload


def evaluate_short_candidate_quality(
    candidate: dict[str, Any],
    graph: VideoUnderstandingGraph,
    *,
    target_platform: str,
) -> CreativeQualityReport:
    start = _as_float(candidate.get("start"), 0.0)
    end = _as_float(candidate.get("end"), start)
    source_ranges = candidate.get("source_ranges") if isinstance(candidate.get("source_ranges"), list) else None
    signals = dict(
        candidate.get("creative_graph_signals")
        or candidate_graph_signals(
            graph,
            start=start,
            end=end,
            text=str(candidate.get("excerpt") or ""),
            source_ranges=source_ranges,
            target_platform=target_platform,
        )
    )
    breakdown = dict(candidate.get("score_breakdown") or {})
    duration = _as_float(candidate.get("duration"), max(end - start, 0.0))
    story = _scale100(breakdown.get("story_completeness"))
    standalone = _scale100(breakdown.get("standalone_clarity"))
    hook = _scale100(breakdown.get("hook_strength"))
    payoff = _scale100(breakdown.get("payoff"))
    graph_retention = _scale01(signals.get("graph_retention_score"))
    topic = _scale01(signals.get("graph_topic_alignment"))
    continuity_risk = _scale01(signals.get("graph_continuity_risk"))
    duration_fit = _scale01(signals.get("graph_duration_fit"))
    visual = _scale01(signals.get("graph_visual_opportunity"))
    score = _bounded(
        graph_retention * 0.22
        + story * 0.18
        + standalone * 0.14
        + hook * 0.13
        + payoff * 0.13
        + topic * 0.08
        + duration_fit * 0.07
        + visual * 0.05
        - continuity_risk * 0.16
    )
    issues: list[str] = []
    warnings: list[str] = []
    if duration < 12.0:
        issues.append("short_candidate_too_short_for_standalone_publish")
    if standalone < 0.42:
        issues.append("weak_standalone_clarity")
    if payoff < 0.36 and graph_retention < 0.42:
        issues.append("weak_payoff_or_retention")
    if continuity_risk >= 0.58:
        issues.append("high_graph_continuity_risk")
    if hook < 0.36:
        warnings.append("opening_hook_is_not_strong")
    if story < 0.48:
        warnings.append("story_arc_may_need_manual_review")
    passed = not issues and score >= 0.55
    return CreativeQualityReport(
        report_id=f"short_candidate_quality:{candidate.get('candidate_id', 'unknown')}",
        subject_type="short_candidate",
        score=score,
        passed=passed,
        issues=issues,
        warnings=warnings,
        metrics={
            "graph_retention": graph_retention,
            "story_completeness": story,
            "standalone_clarity": standalone,
            "hook_strength": hook,
            "payoff": payoff,
            "topic_alignment": topic,
            "duration_fit": duration_fit,
            "visual_opportunity": visual,
            "continuity_risk": continuity_risk,
        },
        evidence={
            "candidate_id": candidate.get("candidate_id"),
            "graph_beat_ids": signals.get("graph_beat_ids", []),
            "selection_reasons": list(candidate.get("selection_reasons") or []),
        },
    )


def evaluate_visual_plan_quality(
    plan: list[dict[str, Any]],
    graph: VideoUnderstandingGraph,
    *,
    max_visuals: int,
) -> CreativeQualityReport:
    if not plan:
        return CreativeQualityReport(
            report_id="visual_plan_quality",
            subject_type="visual_plan",
            score=0.0,
            passed=False,
            issues=["visual_plan_empty"],
            metrics={},
            evidence={"graph_version": graph.version},
        )
    signals = [dict(item.get("creative_graph_signals") or {}) for item in plan]
    opportunity_scores = [_scale01(item.get("graph_visual_opportunity")) for item in signals]
    retention_scores = [_scale01(item.get("graph_retention_score")) for item in signals]
    topic_scores = [_scale01(item.get("graph_topic_alignment")) for item in signals]
    spacing_score = _visual_spacing_score(plan)
    coverage_score = min(len(plan) / max(int(max_visuals), 1), 1.0)
    renderer_confidence = sum(1 for item in plan if str(item.get("renderer_hint") or item.get("renderer") or "").strip()) / max(len(plan), 1)
    score = _bounded(
        _avg(opportunity_scores) * 0.28
        + _avg(retention_scores) * 0.18
        + _avg(topic_scores) * 0.15
        + spacing_score * 0.17
        + coverage_score * 0.12
        + renderer_confidence * 0.10
    )
    issues: list[str] = []
    warnings: list[str] = []
    if _avg(opportunity_scores) < 0.34:
        issues.append("visual_plan_low_semantic_opportunity")
    if spacing_score < 0.45:
        warnings.append("visual_plan_has_dense_or_clustered_timing")
    if coverage_score < 0.34 and max_visuals >= 3:
        warnings.append("visual_plan_under_uses_available_budget")
    if renderer_confidence < 0.5:
        warnings.append("visual_plan_lacks_renderer_confidence")
    return CreativeQualityReport(
        report_id="visual_plan_quality",
        subject_type="visual_plan",
        score=score,
        passed=not issues and score >= 0.52,
        issues=issues,
        warnings=warnings,
        metrics={
            "average_visual_opportunity": _avg(opportunity_scores),
            "average_retention": _avg(retention_scores),
            "average_topic_alignment": _avg(topic_scores),
            "spacing_score": spacing_score,
            "coverage_score": coverage_score,
            "renderer_confidence": renderer_confidence,
        },
        evidence={
            "graph_version": graph.version,
            "visual_ids": [item.get("visual_id") for item in plan],
            "card_ids": [item.get("card_id") for item in plan],
        },
    )


def evaluate_color_grade_quality(plan: dict[str, Any]) -> CreativeQualityReport:
    validation = dict(plan.get("validation") or {})
    manifest = dict(plan.get("manifest") or {})
    adjustments = dict(plan.get("adjustments") or {})
    validation_score = _scale01(validation.get("score"), default=0.68)
    selected_score = _scale01(adjustments.get("average_selected_score"), default=validation_score)
    real_preview_fraction = _scale01(adjustments.get("real_preview_fraction"), default=0.0)
    correction_strength = _scale01(adjustments.get("correction_strength"), default=0.5)
    warnings = [str(item) for item in (plan.get("warnings") or []) if str(item).strip()]
    warnings.extend(str(item) for item in (validation.get("warnings") or []) if str(item).strip())
    shot_validation = dict(validation.get("shot_validation") or {})
    shot_score = _scale01(shot_validation.get("score"), default=validation_score)
    warning_penalty = min(len(warnings) * 0.045, 0.22)
    score = _bounded(
        validation_score * 0.36
        + selected_score * 0.25
        + shot_score * 0.18
        + real_preview_fraction * 0.08
        + (1.0 - min(correction_strength, 1.0)) * 0.05
        + (0.08 if manifest.get("shot_count") else 0.0)
        - warning_penalty
    )
    issues: list[str] = []
    if validation and not validation.get("passed", False):
        issues.append("color_grade_output_validation_failed")
    if shot_validation and not shot_validation.get("passed", False):
        issues.append("color_grade_shot_validation_failed")
    if validation_score < 0.58:
        issues.append("color_grade_validation_score_below_floor")
    return CreativeQualityReport(
        report_id="color_grade_quality",
        subject_type="color_grade",
        score=score,
        passed=not issues and score >= 0.62,
        issues=issues,
        warnings=warnings,
        metrics={
            "validation_score": validation_score,
            "selected_candidate_score": selected_score,
            "shot_validation_score": shot_score,
            "real_preview_fraction": real_preview_fraction,
            "correction_strength": correction_strength,
            "warning_penalty": warning_penalty,
        },
        evidence={
            "resolved_look": plan.get("resolved_look"),
            "filter_graph": plan.get("filter_graph"),
            "manifest_mode": manifest.get("mode"),
            "shot_count": manifest.get("shot_count"),
        },
    )


def _visual_spacing_score(plan: list[dict[str, Any]]) -> float:
    starts = sorted(_as_float(item.get("start"), 0.0) for item in plan)
    if len(starts) <= 1:
        return 1.0
    gaps = [second - first for first, second in zip(starts, starts[1:])]
    too_close = sum(1 for gap in gaps if gap < 1.4)
    return _bounded(1.0 - (too_close / max(len(gaps), 1)))


def _avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return _bounded(sum(values) / len(values))


def _scale100(value: Any, *, default: float = 0.0) -> float:
    number = _as_float(value, default)
    if number > 1.0:
        number /= 100.0
    return _bounded(number)


def _scale01(value: Any, *, default: float = 0.0) -> float:
    number = _as_float(value, default)
    if number > 1.5:
        number /= 100.0
    return _bounded(number)


def _bounded(value: float, low: float = 0.0, high: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return low
    if math.isnan(number) or math.isinf(number):
        return low
    return max(low, min(number, high))


def _as_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number


__all__ = [
    "CreativeQualityReport",
    "evaluate_color_grade_quality",
    "evaluate_short_candidate_quality",
    "evaluate_visual_plan_quality",
]
