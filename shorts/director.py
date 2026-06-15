from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any


STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "for",
    "with", "this", "that", "these", "those", "you", "your", "our", "their",
    "from", "into", "over", "under", "about", "just", "than", "then",
    "they", "them", "have", "has", "had", "was", "were", "are", "is",
    "be", "been", "being", "what", "when", "where", "which", "it", "its",
}
HOOK_TERMS = {"wait", "watch", "look", "why", "how", "secret", "mistake", "truth", "nobody", "everyone"}
SETUP_TERMS = {"problem", "mistake", "reason", "context", "first", "before", "when", "if", "imagine"}
TENSION_TERMS = {"but", "however", "instead", "wrong", "break", "broken", "until", "unless", "versus", "vs"}
PROOF_TERMS = {"because", "proof", "data", "percent", "million", "billion", "number", "result", "evidence"}
PAYOFF_TERMS = {"therefore", "so", "takeaway", "lesson", "system", "framework", "works", "fix", "solves", "answer"}
QUOTE_TERMS = {"never", "always", "only", "best", "worst", "important", "critical", "surprising"}
PLATFORM_IDEAL_DURATIONS = {
    "youtube_shorts": 34.0,
    "tiktok": 27.0,
    "instagram_reels": 30.0,
}
SOURCE_RANGE_ROLES = {
    "hook",
    "context",
    "setup",
    "tension",
    "proof",
    "payoff",
    "quote",
    "support",
    "button",
    "primary",
    "part",
}
EDIT_OPERATION_TYPES = {
    "jump_cut",
    "punch_in",
    "caption_emphasis",
    "auto_visual",
    "speed_ramp",
    "silence_trim",
    "hold_frame",
}


@dataclass(frozen=True)
class VideoContextGraph:
    duration: float
    segment_count: int
    transcript_excerpt: str
    thesis_excerpt: str
    main_keywords: list[str]
    core_keywords: list[str]
    main_phrases: list[str]
    topic_weights: dict[str, float]
    phases: list[dict[str, Any]]
    source_context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration": round(self.duration, 3),
            "segment_count": self.segment_count,
            "transcript_excerpt": self.transcript_excerpt,
            "thesis_excerpt": self.thesis_excerpt,
            "main_keywords": list(self.main_keywords),
            "core_keywords": list(self.core_keywords),
            "main_phrases": list(self.main_phrases),
            "topic_weights": dict(self.topic_weights),
            "phases": [dict(phase) for phase in self.phases],
            "source_context": dict(self.source_context),
        }


@dataclass(frozen=True)
class MomentNode:
    moment_id: str
    start: float
    end: float
    text: str
    moment_type: str
    phase: str
    score: float
    keywords: list[str]
    signals: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "moment_id": self.moment_id,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
            "moment_type": self.moment_type,
            "phase": self.phase,
            "score": round(self.score, 3),
            "keywords": list(self.keywords),
            "signals": dict(self.signals),
        }


@dataclass(frozen=True)
class ShortCandidatePlan:
    candidate_id: str
    start: float
    end: float
    duration: float
    composition_mode: str
    source_ranges: list[dict[str, Any]]
    moment_ids: list[str]
    arc_roles: list[str]
    primary_role: str
    program_score: float
    hook_moment_id: str | None
    payoff_moment_id: str | None
    continuity_risk: float
    arc_integrity: float
    topic_alignment: float
    standalone_score: float
    quality_flags: list[str]
    risk_flags: list[str]
    edit_strategy: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "composition_mode": self.composition_mode,
            "source_ranges": [dict(source_range) for source_range in self.source_ranges],
            "moment_ids": list(self.moment_ids),
            "arc_roles": list(self.arc_roles),
            "primary_role": self.primary_role,
            "program_score": round(self.program_score, 3),
            "hook_moment_id": self.hook_moment_id,
            "payoff_moment_id": self.payoff_moment_id,
            "continuity_risk": round(self.continuity_risk, 3),
            "arc_integrity": round(self.arc_integrity, 3),
            "topic_alignment": round(self.topic_alignment, 3),
            "standalone_score": round(self.standalone_score, 3),
            "quality_flags": list(self.quality_flags),
            "risk_flags": list(self.risk_flags),
            "edit_strategy": dict(self.edit_strategy),
        }


@dataclass(frozen=True)
class ShortsPortfolioPlan:
    target_count: int
    selected_candidate_ids: list[str]
    rejected_candidate_ids: list[str]
    diversity_score: float
    coverage: dict[str, Any]
    selection_reasons: dict[str, list[str]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_count": self.target_count,
            "selected_candidate_ids": list(self.selected_candidate_ids),
            "rejected_candidate_ids": list(self.rejected_candidate_ids),
            "diversity_score": round(self.diversity_score, 3),
            "coverage": dict(self.coverage),
            "selection_reasons": {
                candidate_id: list(reasons)
                for candidate_id, reasons in self.selection_reasons.items()
            },
        }


@dataclass(frozen=True)
class ShortSourceRangePlan:
    index: int
    start: float
    end: float
    duration: float
    role: str
    reason: str = ""
    transition: str = "hard_cut"
    speed: float = 1.0
    crop_hint: str = "center"
    unit_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "role": self.role,
            "reason": self.reason,
            "transition": self.transition,
            "speed": round(self.speed, 3),
            "crop_hint": self.crop_hint,
            "unit_ids": list(self.unit_ids),
        }


@dataclass(frozen=True)
class ShortOperationPlan:
    operation_id: str
    op_type: str
    source_range_index: int | None = None
    start_sec: float | None = None
    end_sec: float | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "operation_id": self.operation_id,
            "type": self.op_type,
            "params": dict(self.params),
        }
        if self.source_range_index is not None:
            payload["source_range_index"] = self.source_range_index
        if self.start_sec is not None:
            payload["start_sec"] = round(self.start_sec, 3)
        if self.end_sec is not None:
            payload["end_sec"] = round(self.end_sec, 3)
        return payload


@dataclass(frozen=True)
class ShortEditPlan:
    candidate_id: str
    target_platform: str
    target_duration_sec: float
    source_ranges: list[ShortSourceRangePlan]
    operations: list[ShortOperationPlan]
    arc_template: str
    selection_strategy: str
    framing_mode: str
    caption_density: str
    caption_style_hint: str
    remix_policy: dict[str, Any]
    punch_in_policy: dict[str, Any]
    visual_insert_policy: dict[str, Any]
    intro_hold_sec: float
    outro_hold_sec: float
    risk_level: str
    qa_checks: list[str]
    quality_floor: float = 56.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "target_platform": self.target_platform,
            "target_duration_sec": round(self.target_duration_sec, 3),
            "source_ranges": [source_range.to_dict() for source_range in self.source_ranges],
            "operations": [operation.to_dict() for operation in self.operations],
            "arc_template": self.arc_template,
            "selection_strategy": self.selection_strategy,
            "framing_mode": self.framing_mode,
            "caption_density": self.caption_density,
            "caption_style_hint": self.caption_style_hint,
            "remix_policy": dict(self.remix_policy),
            "punch_in_policy": dict(self.punch_in_policy),
            "visual_insert_policy": dict(self.visual_insert_policy),
            "intro_hold_sec": round(self.intro_hold_sec, 3),
            "outro_hold_sec": round(self.outro_hold_sec, 3),
            "risk_level": self.risk_level,
            "qa_checks": list(self.qa_checks),
            "quality_floor": round(self.quality_floor, 3),
        }


@dataclass(frozen=True)
class ShortsProgram:
    video_context: VideoContextGraph
    moments: list[MomentNode]
    candidates: list[ShortCandidatePlan]
    portfolio: ShortsPortfolioPlan
    edit_plans: dict[str, ShortEditPlan]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": "shorts-director-v4",
            "video_context": self.video_context.to_dict(),
            "moments": [moment.to_dict() for moment in self.moments],
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "portfolio": self.portfolio.to_dict(),
            "edit_plans": {
                candidate_id: edit_plan.to_dict()
                for candidate_id, edit_plan in self.edit_plans.items()
            },
            "metadata": dict(self.metadata),
        }


def build_shorts_program(
    *,
    transcript_text: str,
    segments: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    selections: list[dict[str, Any]] | None = None,
    requested_count: int,
    target_platform: str,
    min_duration_sec: float,
    max_duration_sec: float,
    video_context: dict[str, Any] | None = None,
) -> ShortsProgram:
    context_graph = _build_context_graph(transcript_text, segments, video_context or {})
    moments = _build_moments(segments, context_graph)
    candidate_plans = _build_candidate_plans(
        candidates,
        moments,
        context_graph,
        target_platform=target_platform,
        min_duration_sec=min_duration_sec,
        max_duration_sec=max_duration_sec,
    )
    portfolio = _solve_portfolio(
        candidate_plans,
        candidates,
        requested_count=requested_count,
        seed_ids=[
            str(selection.get("candidate_id"))
            for selection in (selections or [])
            if str(selection.get("candidate_id") or "").strip()
        ],
        seed_sources={
            str(selection.get("candidate_id")): str(
                selection.get("selection_source") or "unknown"
            )
            for selection in (selections or [])
            if str(selection.get("candidate_id") or "").strip()
        },
    )
    edit_plans = {
        plan.candidate_id: _build_edit_plan(plan, target_platform=target_platform)
        for plan in candidate_plans
    }
    return ShortsProgram(
        video_context=context_graph,
        moments=moments,
        candidates=candidate_plans,
        portfolio=portfolio,
        edit_plans=edit_plans,
        metadata={
            "requested_count": requested_count,
            "target_platform": target_platform,
            "min_duration_sec": round(float(min_duration_sec), 3),
            "max_duration_sec": round(float(max_duration_sec), 3),
        },
    )


def _build_context_graph(
    transcript_text: str,
    segments: list[dict[str, Any]],
    video_context: dict[str, Any],
) -> VideoContextGraph:
    text = _clean_text(transcript_text or _segment_text(segments))
    duration = _segments_duration(segments) or _as_float(video_context.get("duration"), 0.0)
    main_keywords = _strings(video_context.get("main_keywords")) or _top_keywords(text, 28)
    core_keywords = _strings(video_context.get("core_keywords")) or main_keywords[:18]
    main_phrases = _strings(video_context.get("main_phrases")) or _top_phrases(text, 12)
    topic_weights = {
        keyword: round(1.0 + (len(main_keywords) - index) / max(len(main_keywords), 1), 4)
        for index, keyword in enumerate(main_keywords)
    }
    for keyword in core_keywords:
        topic_weights[keyword] = round(float(topic_weights.get(keyword, 1.0)) + 0.35, 4)
    return VideoContextGraph(
        duration=duration,
        segment_count=len(segments),
        transcript_excerpt=_truncate(text, 1400),
        thesis_excerpt=_truncate(str(video_context.get("thesis_excerpt") or text), 500),
        main_keywords=main_keywords,
        core_keywords=core_keywords[:24],
        main_phrases=main_phrases[:14],
        topic_weights=topic_weights,
        phases=_phase_ranges(duration),
        source_context=dict(video_context),
    )


def _build_moments(segments: list[dict[str, Any]], context: VideoContextGraph) -> list[MomentNode]:
    moments: list[MomentNode] = []
    for index, segment in enumerate(segments, start=1):
        start = _as_float(segment.get("start"), 0.0)
        end = max(start + 0.05, _as_float(segment.get("end"), start + 0.8))
        text = _clean_text(segment.get("text"))
        if not text:
            continue
        tokens = _tokens(text)
        moment_type, signals = _classify_moment(tokens, text, start=start, duration=context.duration)
        keywords = _candidate_keywords(text, limit=8)
        topic_alignment = _weighted_overlap(tokens, context.topic_weights)
        score = _bounded(
            28.0
            + signals["hook_hits"] * 8.0
            + signals["payoff_hits"] * 8.0
            + signals["proof_hits"] * 7.0
            + signals["tension_hits"] * 6.0
            + topic_alignment * 0.22
            + min(len(tokens), 28) * 0.7,
        )
        moments.append(
            MomentNode(
                moment_id=f"moment_{index:03d}",
                start=round(start, 3),
                end=round(end, 3),
                text=_truncate(text, 220),
                moment_type=moment_type,
                phase=_phase_for_time(start, context.duration),
                score=round(score, 3),
                keywords=keywords,
                signals={**signals, "topic_alignment": round(topic_alignment, 3)},
            )
        )
    return moments


def _build_candidate_plans(
    candidates: list[dict[str, Any]],
    moments: list[MomentNode],
    context: VideoContextGraph,
    *,
    target_platform: str,
    min_duration_sec: float,
    max_duration_sec: float,
) -> list[ShortCandidatePlan]:
    plans: list[ShortCandidatePlan] = []
    for candidate in candidates:
        source_ranges = _source_ranges(candidate)
        start = min((source_range["start"] for source_range in source_ranges), default=_as_float(candidate.get("start"), 0.0))
        end = max((source_range["end"] for source_range in source_ranges), default=_as_float(candidate.get("end"), start))
        duration = _source_ranges_duration(source_ranges) or max(0.0, end - start)
        composition_mode = str(candidate.get("composition_mode") or ("remix" if len(source_ranges) > 1 else "single_window"))
        inside = _moments_for_source_ranges(moments, source_ranges)
        roles = _ordered_unique(moment.moment_type for moment in inside)
        breakdown = dict(candidate.get("score_breakdown") or {})
        tokens = _tokens(str(candidate.get("excerpt") or ""))
        topic_alignment = _weighted_overlap(tokens, context.topic_weights)
        standalone = _as_float(breakdown.get("standalone_clarity"), _as_float(breakdown.get("clarity"), 45.0))
        hook_moment = _best_moment(inside, {"hook", "setup", "quote"})
        payoff_moment = _best_moment(inside, {"payoff", "proof", "quote"})
        arc_integrity = _arc_integrity(roles, hook_moment=hook_moment, payoff_moment=payoff_moment, duration=duration)
        continuity_risk = _continuity_risk(candidate, breakdown, roles, duration, min_duration_sec, max_duration_sec)
        base_score = _as_float(candidate.get("heuristic_score"), _as_float(breakdown.get("overall"), 1.0))
        duration_fit = _duration_fit(duration, target_platform, min_duration_sec, max_duration_sec)
        program_score = _bounded(
            base_score * 0.42
            + arc_integrity * 0.22
            + topic_alignment * 0.16
            + standalone * 0.12
            + duration_fit * 0.08
            - continuity_risk * 0.26,
            1.0,
            100.0,
        )
        quality_flags = _quality_flags(roles, arc_integrity, topic_alignment, standalone, duration_fit)
        risk_flags = _risk_flags(candidate, breakdown, roles, continuity_risk)
        primary_role = _primary_role(roles, inside)
        plans.append(
            ShortCandidatePlan(
                candidate_id=str(candidate.get("candidate_id") or ""),
                start=round(start, 3),
                end=round(end, 3),
                duration=round(duration, 3),
                composition_mode=composition_mode,
                source_ranges=source_ranges,
                moment_ids=[moment.moment_id for moment in inside],
                arc_roles=roles,
                primary_role=primary_role,
                program_score=round(program_score, 3),
                hook_moment_id=hook_moment.moment_id if hook_moment else None,
                payoff_moment_id=payoff_moment.moment_id if payoff_moment else None,
                continuity_risk=round(continuity_risk, 3),
                arc_integrity=round(arc_integrity, 3),
                topic_alignment=round(topic_alignment, 3),
                standalone_score=round(standalone, 3),
                quality_flags=quality_flags,
                risk_flags=risk_flags,
                edit_strategy=_candidate_edit_strategy(
                    primary_role,
                    continuity_risk,
                    arc_integrity,
                    duration,
                    composition_mode=composition_mode,
                    source_range_count=len(source_ranges),
                    edit_plan_seed=dict(candidate.get("edit_plan_seed") or {}),
                ),
            )
        )
    plans.sort(key=lambda plan: (plan.program_score, plan.arc_integrity, -plan.continuity_risk), reverse=True)
    return plans


def _solve_portfolio(
    plans: list[ShortCandidatePlan],
    candidates: list[dict[str, Any]],
    *,
    requested_count: int,
    seed_ids: list[str],
    seed_sources: dict[str, str],
) -> ShortsPortfolioPlan:
    candidate_by_id = {str(candidate.get("candidate_id")): candidate for candidate in candidates}
    plan_by_id = {plan.candidate_id: plan for plan in plans}
    selected: list[ShortCandidatePlan] = []
    reasons: dict[str, list[str]] = {}
    for candidate_id in seed_ids:
        plan = plan_by_id.get(candidate_id)
        candidate = candidate_by_id.get(candidate_id, {})
        if (
            plan
            and _portfolio_seed_eligible(plan, candidate)
            and _portfolio_compatible(plan, selected, candidate_by_id)
        ):
            selected.append(plan)
            selection_source = seed_sources.get(candidate_id, "unknown")
            reasons[plan.candidate_id] = [
                f"kept from {selection_source}",
                *_portfolio_reasons(plan),
            ]
        if len(selected) >= requested_count:
            break
    for plan in plans:
        if len(selected) >= requested_count:
            break
        if plan in selected:
            continue
        if (
            _portfolio_seed_eligible(plan, candidate_by_id.get(plan.candidate_id, {}))
            and _portfolio_compatible(plan, selected, candidate_by_id)
        ):
            selected.append(plan)
            reasons[plan.candidate_id] = _portfolio_reasons(plan)
    selected_ids = [plan.candidate_id for plan in selected[:requested_count]]
    rejected_ids = [plan.candidate_id for plan in plans if plan.candidate_id not in set(selected_ids)]
    selected_roles = [plan.primary_role for plan in selected[:requested_count]]
    selected_topics = [_candidate_keywords(str(candidate_by_id.get(plan.candidate_id, {}).get("excerpt") or ""), 5) for plan in selected[:requested_count]]
    diversity_score = _portfolio_diversity(selected[:requested_count], candidate_by_id)
    return ShortsPortfolioPlan(
        target_count=requested_count,
        selected_candidate_ids=selected_ids,
        rejected_candidate_ids=rejected_ids,
        diversity_score=round(diversity_score, 3),
        coverage={
            "roles": selected_roles,
            "role_count": len(set(selected_roles)),
            "topic_keywords": selected_topics,
            "average_program_score": round(
                sum(plan.program_score for plan in selected[:requested_count]) / max(len(selected[:requested_count]), 1),
                3,
            ),
        },
        selection_reasons=reasons,
    )


def _portfolio_seed_eligible(
    plan: ShortCandidatePlan,
    candidate: dict[str, Any],
) -> bool:
    story_plan = candidate.get("story_plan")
    if isinstance(story_plan, dict):
        critic = story_plan.get("critic")
        if isinstance(critic, dict) and not bool(critic.get("passed", False)):
            return False
    if plan.continuity_risk >= 58:
        return False
    severe_risks = {"abrupt_start", "dangling_end"}
    return not bool(severe_risks & set(plan.risk_flags))


def _build_edit_plan(plan: ShortCandidatePlan, *, target_platform: str) -> ShortEditPlan:
    risk_level = "high" if plan.continuity_risk >= 58 else "medium" if plan.continuity_risk >= 34 else "low"
    fast_caption = plan.duration <= 24 or plan.primary_role in {"proof", "quote"}
    is_remix = plan.composition_mode != "single_window" or len(plan.source_ranges) > 1
    max_punch_ins = 1 if risk_level == "high" else 2 if plan.duration <= 32 else 3
    max_visuals = 0 if risk_level == "high" else 1 if plan.duration <= 28 else 2
    if plan.primary_role in {"proof", "process"} and risk_level != "high":
        max_visuals = max(max_visuals, 2)
    if is_remix and risk_level != "high":
        max_visuals = max(max_visuals, 1)
    source_range_plans = _source_range_plans(plan)
    operations = _operation_plans(
        plan,
        source_range_plans,
        max_punch_ins=max_punch_ins,
        max_visuals=max_visuals,
        risk_level=risk_level,
    )
    return ShortEditPlan(
        candidate_id=plan.candidate_id,
        target_platform=target_platform,
        target_duration_sec=plan.duration,
        source_ranges=source_range_plans,
        operations=operations,
        arc_template=str(plan.edit_strategy.get("arc_template") or _arc_template_for_plan(plan)),
        selection_strategy="edit_graph" if is_remix else "continuous_window",
        framing_mode="stitched_center_stage_safe" if is_remix else "center_stage_safe",
        caption_density="fast" if fast_caption else "balanced",
        caption_style_hint="creator_bold" if target_platform in {"youtube_shorts", "tiktok"} else "clean_pop",
        remix_policy={
            "enabled": is_remix,
            "composition_mode": plan.composition_mode,
            "source_range_count": len(plan.source_ranges),
            "max_source_ranges": 4,
            "min_part_duration_sec": 3.0,
            "join_style": "hard_cut_story_order",
            "preserve_source_audio": True,
            "requires_transcript_gate": True,
        },
        punch_in_policy={
            "enabled": risk_level != "high",
            "max_moments": max_punch_ins,
            "min_gap_sec": 1.1 if risk_level == "low" else 1.6,
            "max_zoom": 1.18 if risk_level == "low" else 1.12,
        },
        visual_insert_policy={
            "enabled": max_visuals > 0,
            "max_inserts": max_visuals,
            "prefer_types": _visual_types_for_role(plan.primary_role),
        },
        intro_hold_sec=0.18 if plan.hook_moment_id else 0.0,
        outro_hold_sec=0.24 if plan.payoff_moment_id else 0.08,
        risk_level=risk_level,
        qa_checks=[
            "vertical_resolution",
            "audio_preserved",
            "caption_file_present",
            "duration_within_platform_window",
            "punch_in_count_within_policy",
            "typed_edit_plan_validation",
            "pre_render_transcript_quality_gate",
            "final_transcript_quality_gate",
            *(
                [
                    "stitched_transcript_continuity",
                    "source_range_count_within_policy",
                ]
                if is_remix
                else []
            ),
        ],
        quality_floor=58.0 if is_remix else 56.0,
    )


def _source_range_plans(plan: ShortCandidatePlan) -> list[ShortSourceRangePlan]:
    ranges: list[ShortSourceRangePlan] = []
    for index, source_range in enumerate(plan.source_ranges, start=1):
        start = _as_float(source_range.get("start"), 0.0)
        end = max(start, _as_float(source_range.get("end"), start))
        role = _safe_role(str(source_range.get("role") or "part"))
        reason = _truncate(str(source_range.get("reason") or _source_range_reason(role, index)), 180)
        speed = _bounded(_as_float(source_range.get("speed"), 1.0), 0.75, 1.35)
        ranges.append(
            ShortSourceRangePlan(
                index=int(source_range.get("index") or index),
                start=round(start, 3),
                end=round(end, 3),
                duration=round(max(0.0, end - start), 3),
                role=role,
                reason=reason,
                transition=_truncate(str(source_range.get("transition") or ("hard_cut" if index > 1 else "open")), 32),
                speed=round(speed, 3),
                crop_hint=_truncate(str(source_range.get("crop_hint") or _crop_hint_for_role(role)), 32),
                unit_ids=[
                    str(unit_id).strip()
                    for unit_id in source_range.get("unit_ids", [])
                    if str(unit_id).strip()
                ][:64]
                if isinstance(source_range.get("unit_ids"), list)
                else [],
            )
        )
    return ranges


def _operation_plans(
    plan: ShortCandidatePlan,
    source_ranges: list[ShortSourceRangePlan],
    *,
    max_punch_ins: int,
    max_visuals: int,
    risk_level: str,
) -> list[ShortOperationPlan]:
    operations: list[ShortOperationPlan] = []
    timeline_offsets: dict[int, tuple[float, float]] = {}
    offset = 0.0
    previous_range: ShortSourceRangePlan | None = None
    for source_range in source_ranges:
        local_start = offset
        local_end = offset + source_range.duration
        timeline_offsets[source_range.index] = (local_start, local_end)
        if previous_range is not None:
            operations.append(
                ShortOperationPlan(
                    operation_id=f"{plan.candidate_id}_cut_{source_range.index:02d}",
                    op_type="jump_cut",
                    source_range_index=source_range.index,
                    start_sec=local_start,
                    end_sec=local_start,
                    params={
                        "from_previous_role": previous_range.role,
                        "to_role": source_range.role,
                        "transition": source_range.transition,
                    },
                )
            )
        offset = local_end
        previous_range = source_range

    punch_roles = {"hook", "quote", "proof", "tension", "payoff"}
    punch_count = 0
    for source_range in source_ranges:
        if punch_count >= max_punch_ins or risk_level == "high":
            break
        if source_range.role not in punch_roles:
            continue
        local_start, local_end = timeline_offsets[source_range.index]
        if local_end - local_start < 1.8:
            continue
        operations.append(
            ShortOperationPlan(
                operation_id=f"{plan.candidate_id}_punch_{source_range.index:02d}",
                op_type="punch_in",
                source_range_index=source_range.index,
                start_sec=round(local_start + 0.25, 3),
                end_sec=round(min(local_end, local_start + 2.6), 3),
                params={
                    "zoom": 1.16 if risk_level == "low" else 1.1,
                    "reason": f"emphasize {source_range.role} beat",
                },
            )
        )
        punch_count += 1

    visual_roles = {"proof", "tension", "context"}
    visual_count = 0
    for source_range in source_ranges:
        if visual_count >= max_visuals or risk_level == "high":
            break
        if source_range.role not in visual_roles:
            continue
        local_start, local_end = timeline_offsets[source_range.index]
        if local_end - local_start < 3.0:
            continue
        operations.append(
            ShortOperationPlan(
                operation_id=f"{plan.candidate_id}_visual_{source_range.index:02d}",
                op_type="auto_visual",
                source_range_index=source_range.index,
                start_sec=round(local_start + 0.35, 3),
                end_sec=round(min(local_end, local_start + 4.8), 3),
                params={
                    "mode": "contextual_support",
                    "preferred_types": _visual_types_for_role(source_range.role),
                    "reason": f"support {source_range.role} with a visual insert",
                },
            )
        )
        visual_count += 1

    operations.append(
        ShortOperationPlan(
            operation_id=f"{plan.candidate_id}_captions",
            op_type="caption_emphasis",
            start_sec=0.0,
            end_sec=round(plan.duration, 3),
            params={
                "density": "fast" if plan.duration <= 24 else "balanced",
                "emphasize_roles": [source_range.role for source_range in source_ranges if source_range.role in punch_roles],
            },
        )
    )
    return operations


def _portfolio_compatible(
    plan: ShortCandidatePlan,
    selected: list[ShortCandidatePlan],
    candidate_by_id: dict[str, dict[str, Any]],
) -> bool:
    for existing in selected:
        if _source_ranges_overlap_ratio(plan.source_ranges, existing.source_ranges) >= 0.5:
            return False
        if plan.primary_role == existing.primary_role and len(selected) >= 1:
            if _topic_similarity(
                candidate_by_id.get(plan.candidate_id, {}),
                candidate_by_id.get(existing.candidate_id, {}),
            ) >= 0.52:
                return False
    return True


def _portfolio_reasons(plan: ShortCandidatePlan) -> list[str]:
    reasons = [f"{plan.primary_role} arc", f"program score {plan.program_score:.1f}"]
    if plan.arc_integrity >= 70:
        reasons.append("strong setup/payoff structure")
    if plan.topic_alignment >= 55:
        reasons.append("aligned with the full video topic")
    if plan.continuity_risk <= 28:
        reasons.append("low continuity risk")
    return reasons[:4]


def _portfolio_diversity(
    selected: list[ShortCandidatePlan],
    candidate_by_id: dict[str, dict[str, Any]],
) -> float:
    if len(selected) <= 1:
        return 100.0 if selected else 0.0
    role_score = len({plan.primary_role for plan in selected}) / max(len(selected), 1) * 100.0
    pair_scores: list[float] = []
    for index, first in enumerate(selected):
        for second in selected[index + 1:]:
            pair_scores.append(
                100.0
                - (_topic_similarity(candidate_by_id.get(first.candidate_id, {}), candidate_by_id.get(second.candidate_id, {})) * 100.0)
            )
    topic_score = sum(pair_scores) / max(len(pair_scores), 1)
    return _bounded(role_score * 0.46 + topic_score * 0.54)


def _classify_moment(tokens: list[str], text: str, *, start: float, duration: float) -> tuple[str, dict[str, Any]]:
    lower = text.lower()
    signals = {
        "hook_hits": _hits(tokens, HOOK_TERMS) + (1 if "?" in text else 0),
        "setup_hits": _hits(tokens, SETUP_TERMS),
        "tension_hits": _hits(tokens, TENSION_TERMS),
        "proof_hits": _hits(tokens, PROOF_TERMS) + len(re.findall(r"\b\d+(?:\.\d+)?%?\b", text)),
        "payoff_hits": _hits(tokens, PAYOFF_TERMS),
        "quote_hits": _hits(tokens, QUOTE_TERMS) + (1 if "!" in text else 0),
        "starts_contextual": bool(tokens and tokens[0] in {"and", "but", "so", "this", "that", "it", "they"}),
        "phase": _phase_for_time(start, duration),
    }
    if signals["hook_hits"] >= 2 or start <= min(5.0, max(duration * 0.12, 0.0)):
        moment_type = "hook"
    elif signals["proof_hits"] >= 2:
        moment_type = "proof"
    elif signals["tension_hits"] >= 1:
        moment_type = "tension"
    elif signals["payoff_hits"] >= 1 or signals["phase"] in {"resolution", "closing"}:
        moment_type = "payoff"
    elif signals["setup_hits"] >= 1:
        moment_type = "setup"
    elif signals["quote_hits"] >= 2 or any(phrase in lower for phrase in ("the truth", "the mistake", "the lesson")):
        moment_type = "quote"
    else:
        moment_type = "support"
    return moment_type, signals


def _arc_integrity(
    roles: list[str],
    *,
    hook_moment: MomentNode | None,
    payoff_moment: MomentNode | None,
    duration: float,
) -> float:
    score = 18.0
    if hook_moment:
        score += 22.0
    if payoff_moment:
        score += 24.0
    if "tension" in roles or "proof" in roles:
        score += 16.0
    if "setup" in roles:
        score += 10.0
    if len(roles) >= 3:
        score += 10.0
    if duration < 14.0:
        score -= 12.0
    return _bounded(score)


def _continuity_risk(
    candidate: dict[str, Any],
    breakdown: dict[str, Any],
    roles: list[str],
    duration: float,
    min_duration_sec: float,
    max_duration_sec: float,
) -> float:
    excerpt = str(candidate.get("excerpt") or "")
    tokens = _tokens(excerpt)
    risk = 0.0
    risk += _as_float(breakdown.get("abrupt_start_penalty"), 0.0) * 0.55
    risk += _as_float(breakdown.get("dangling_payoff_penalty"), 0.0) * 0.45
    risk += _as_float(breakdown.get("context_dependency_penalty"), 0.0) * 0.45
    risk += 14.0 if tokens and tokens[0] in {"and", "but", "so", "this", "that", "it", "they"} else 0.0
    risk += 10.0 if roles and roles[0] not in {"hook", "setup", "quote"} else 0.0
    risk += 12.0 if not any(role in roles for role in {"payoff", "proof", "quote"}) else 0.0
    risk += 8.0 if duration < min_duration_sec + 1.0 or duration > max_duration_sec - 1.0 else 0.0
    return _bounded(risk)


def _quality_flags(
    roles: list[str],
    arc_integrity: float,
    topic_alignment: float,
    standalone: float,
    duration_fit: float,
) -> list[str]:
    flags: list[str] = []
    if arc_integrity >= 70:
        flags.append("complete_arc")
    if any(role in roles for role in {"hook", "quote"}):
        flags.append("strong_open")
    if any(role in roles for role in {"payoff", "proof"}):
        flags.append("clear_payoff")
    if topic_alignment >= 55:
        flags.append("on_topic")
    if standalone >= 66:
        flags.append("standalone")
    if duration_fit >= 72:
        flags.append("platform_duration_fit")
    return flags or ["usable"]


def _risk_flags(candidate: dict[str, Any], breakdown: dict[str, Any], roles: list[str], continuity_risk: float) -> list[str]:
    flags: list[str] = []
    if continuity_risk >= 45:
        flags.append("continuity_risk")
    if _as_float(breakdown.get("abrupt_start_penalty"), 0.0) >= 14:
        flags.append("abrupt_start")
    if _as_float(breakdown.get("dangling_payoff_penalty"), 0.0) >= 14:
        flags.append("dangling_end")
    if _as_float(breakdown.get("misleading_clip_penalty"), 0.0) >= 10:
        flags.append("weak_topic_fit")
    if not any(role in roles for role in {"payoff", "proof", "quote"}):
        flags.append("weak_payoff")
    return flags


def _candidate_edit_strategy(
    primary_role: str,
    continuity_risk: float,
    arc_integrity: float,
    duration: float,
    *,
    composition_mode: str,
    source_range_count: int,
    edit_plan_seed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    edit_plan_seed = edit_plan_seed or {}
    return {
        "primary_role": primary_role,
        "composition_mode": composition_mode,
        "source_range_count": source_range_count,
        "arc_template": str(edit_plan_seed.get("arc_template") or ""),
        "arc_strategy": str(edit_plan_seed.get("strategy") or ""),
        "pace": "fast" if duration <= 24 else "balanced",
        "motion_intensity": "restrained" if continuity_risk >= 40 else "medium" if arc_integrity >= 62 else "subtle",
        "needs_context_title": continuity_risk >= 42,
        "needs_stitch_review": composition_mode != "single_window" or source_range_count > 1,
        "best_visual_support": _visual_types_for_role(primary_role),
        "planned_operations": list(edit_plan_seed.get("operations") or []),
    }


def _visual_types_for_role(primary_role: str) -> list[str]:
    if primary_role == "proof":
        return ["data_graphic", "metric_callout"]
    if primary_role == "process":
        return ["process_diagram", "product_ui"]
    if primary_role == "tension":
        return ["comparison", "contrast_card"]
    if primary_role == "hook":
        return ["title_card", "keyword_pop"]
    return ["text_overlay", "supporting_cutaway"]


def _safe_role(role: str) -> str:
    normalized = re.sub(r"[^a-z0-9_]+", "_", str(role or "").lower()).strip("_")
    return normalized if normalized in SOURCE_RANGE_ROLES else "part"


def _source_range_reason(role: str, index: int) -> str:
    if role == "hook":
        return "opens the short with the strongest attention beat"
    if role == "context":
        return "adds only the context needed for a cold viewer"
    if role == "proof":
        return "supports the claim with evidence or specificity"
    if role == "payoff":
        return "closes the short with a satisfying takeaway"
    if role == "tension":
        return "creates contrast that improves retention"
    if role == "quote":
        return "uses a concise memorable line"
    if role == "button":
        return "adds a final shareable button"
    return f"source beat {index}"


def _crop_hint_for_role(role: str) -> str:
    if role in {"hook", "quote"}:
        return "face_priority"
    if role in {"proof", "context"}:
        return "screen_or_center"
    if role == "payoff":
        return "center"
    return "center"


def _arc_template_for_plan(plan: ShortCandidatePlan) -> str:
    if plan.composition_mode == "single_window" and len(plan.source_ranges) <= 1:
        return "continuous_window"
    roles = [source_range.get("role", "part") for source_range in plan.source_ranges]
    return " -> ".join(str(role) for role in roles if role) or "stitched_arc"


def _source_ranges(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    raw_ranges = candidate.get("source_ranges")
    ranges: list[dict[str, Any]] = []
    if isinstance(raw_ranges, list):
        for index, raw_range in enumerate(raw_ranges, start=1):
            if not isinstance(raw_range, dict):
                continue
            start = _as_float(raw_range.get("start"), 0.0)
            end = _as_float(raw_range.get("end"), start)
            if end <= start:
                continue
            try:
                source_index = int(raw_range.get("index") or index)
            except (TypeError, ValueError):
                source_index = index
            ranges.append(
                {
                    "index": source_index,
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "duration": round(end - start, 3),
                    "role": _safe_role(str(raw_range.get("role") or "part")),
                    "reason": _truncate(str(raw_range.get("reason") or ""), 180),
                    "transition": _truncate(str(raw_range.get("transition") or ("hard_cut" if index > 1 else "open")), 32),
                    "speed": _bounded(_as_float(raw_range.get("speed"), 1.0), 0.75, 1.35),
                    "crop_hint": _truncate(str(raw_range.get("crop_hint") or ""), 32),
                    "unit_ids": [
                        str(unit_id).strip()
                        for unit_id in raw_range.get("unit_ids", [])
                        if str(unit_id).strip()
                    ][:64]
                    if isinstance(raw_range.get("unit_ids"), list)
                    else [],
                }
            )
    if ranges:
        return ranges
    start = _as_float(candidate.get("start"), 0.0)
    end = _as_float(candidate.get("end"), start)
    if end <= start:
        return []
    return [
        {
            "index": 1,
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(end - start, 3),
            "role": "primary",
        }
    ]


def _source_ranges_duration(source_ranges: list[dict[str, Any]]) -> float:
    return round(
        sum(max(0.0, _as_float(source_range.get("end"), 0.0) - _as_float(source_range.get("start"), 0.0)) for source_range in source_ranges),
        3,
    )


def _moments_for_source_ranges(moments: list[MomentNode], source_ranges: list[dict[str, Any]]) -> list[MomentNode]:
    selected: list[MomentNode] = []
    seen_ids: set[str] = set()
    for source_range in source_ranges:
        start = _as_float(source_range.get("start"), 0.0)
        end = _as_float(source_range.get("end"), start)
        for moment in moments:
            if moment.moment_id in seen_ids:
                continue
            if moment.end > start and moment.start < end:
                selected.append(moment)
                seen_ids.add(moment.moment_id)
    return selected


def _source_ranges_overlap_ratio(first: list[dict[str, Any]], second: list[dict[str, Any]]) -> float:
    first_duration = _source_ranges_duration(first)
    second_duration = _source_ranges_duration(second)
    if first_duration <= 0 or second_duration <= 0:
        return 0.0
    overlap = 0.0
    for first_range in first:
        first_start = _as_float(first_range.get("start"), 0.0)
        first_end = _as_float(first_range.get("end"), first_start)
        for second_range in second:
            second_start = _as_float(second_range.get("start"), 0.0)
            second_end = _as_float(second_range.get("end"), second_start)
            overlap += max(0.0, min(first_end, second_end) - max(first_start, second_start))
    return overlap / max(min(first_duration, second_duration), 0.001)


def _best_moment(moments: list[MomentNode], roles: set[str]) -> MomentNode | None:
    candidates = [moment for moment in moments if moment.moment_type in roles]
    if not candidates:
        return None
    return max(candidates, key=lambda moment: (moment.score, -moment.start))


def _primary_role(roles: list[str], moments: list[MomentNode]) -> str:
    if not roles:
        return "support"
    ranked_roles = sorted(
        roles,
        key=lambda role: (
            {"hook": 7, "proof": 6, "tension": 5, "payoff": 4, "quote": 4, "setup": 3, "support": 1}.get(role, 0),
            sum(moment.score for moment in moments if moment.moment_type == role),
        ),
        reverse=True,
    )
    return ranked_roles[0]


def _phase_ranges(duration: float) -> list[dict[str, Any]]:
    if duration <= 0:
        return []
    return [
        {"phase": "opening", "start": 0.0, "end": round(duration * 0.16, 3)},
        {"phase": "development", "start": round(duration * 0.16, 3), "end": round(duration * 0.62, 3)},
        {"phase": "resolution", "start": round(duration * 0.62, 3), "end": round(duration * 0.86, 3)},
        {"phase": "closing", "start": round(duration * 0.86, 3), "end": round(duration, 3)},
    ]


def _phase_for_time(start: float, duration: float) -> str:
    if duration <= 0:
        return "unknown"
    position = start / duration
    if position <= 0.16:
        return "opening"
    if position <= 0.62:
        return "development"
    if position <= 0.86:
        return "resolution"
    return "closing"


def _duration_fit(duration: float, target_platform: str, min_duration_sec: float, max_duration_sec: float) -> float:
    ideal = max(min_duration_sec, min(PLATFORM_IDEAL_DURATIONS.get(target_platform, 30.0), max_duration_sec))
    tolerance = max(ideal - min_duration_sec, max_duration_sec - ideal, 8.0)
    return _bounded(100.0 * (1.0 - abs(duration - ideal) / tolerance))


def _weighted_overlap(tokens: list[str], topic_weights: dict[str, float]) -> float:
    token_set = {token for token in tokens if token not in STOPWORDS and len(token) >= 3}
    if not token_set or not topic_weights:
        return 0.0
    matched = sum(float(weight) for token, weight in topic_weights.items() if token in token_set)
    possible = sum(sorted((float(value) for value in topic_weights.values()), reverse=True)[: max(1, min(len(token_set), 12))])
    return _bounded((matched / max(possible, 0.001)) * 100.0)


def _topic_similarity(first: dict[str, Any], second: dict[str, Any]) -> float:
    first_keywords = set(_candidate_keywords(str(first.get("excerpt") or ""), 10))
    second_keywords = set(_candidate_keywords(str(second.get("excerpt") or ""), 10))
    if not first_keywords or not second_keywords:
        return 0.0
    return len(first_keywords & second_keywords) / max(len(first_keywords | second_keywords), 1)


def _range_overlap_ratio(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    overlap = max(0.0, min(end_a, end_b) - max(start_a, start_b))
    if overlap <= 0:
        return 0.0
    return overlap / max(min(end_a - start_a, end_b - start_b), 0.001)


def _ordered_unique(items: Any) -> list[str]:
    result: list[str] = []
    for item in items:
        value = str(item or "").strip()
        if value and value not in result:
            result.append(value)
    return result


def _segment_text(segments: list[dict[str, Any]]) -> str:
    return " ".join(_clean_text(segment.get("text")) for segment in segments if _clean_text(segment.get("text"))).strip()


def _segments_duration(segments: list[dict[str, Any]]) -> float:
    if not segments:
        return 0.0
    starts = [_as_float(segment.get("start"), 0.0) for segment in segments]
    ends = [_as_float(segment.get("end"), 0.0) for segment in segments]
    return max(0.0, max(ends) - min(starts))


def _candidate_keywords(text: str, limit: int = 10) -> list[str]:
    keywords: list[str] = []
    for token in _tokens(text):
        if token in STOPWORDS or len(token) < 3:
            continue
        if token not in keywords:
            keywords.append(token)
        if len(keywords) >= limit:
            break
    return keywords


def _top_keywords(text: str, limit: int) -> list[str]:
    counts: dict[str, int] = {}
    for token in _tokens(text):
        if token in STOPWORDS or len(token) < 3:
            continue
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))
    return [keyword for keyword, _count in ranked[:limit]]


def _top_phrases(text: str, limit: int) -> list[str]:
    tokens = [token for token in _tokens(text) if token not in STOPWORDS and len(token) >= 3]
    counts: dict[str, int] = {}
    for first, second in zip(tokens, tokens[1:]):
        phrase = f"{first} {second}"
        counts[phrase] = counts.get(phrase, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))
    return [phrase for phrase, _count in ranked[:limit]]


def _hits(tokens: list[str], terms: set[str]) -> int:
    token_set = set(tokens)
    return len(token_set & terms)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9']+", str(text or "").lower())


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = str(item).strip().lower()
        if text and text not in result:
            result.append(text)
    return result


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _truncate(text: str, limit: int) -> str:
    cleaned = _clean_text(text)
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def _bounded(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(float(value), high))


def _as_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number
