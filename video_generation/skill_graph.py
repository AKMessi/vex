from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from video_generation.director import DirectorPackage
from video_generation.models import Beat, BeatGraph, ScriptPlan, VideoGenerationRequest
from video_generation.script_planner import keyword_candidates
from vex_hyperframes.skill_pack import retrieve_skill_slices


VIDEO_GENERATION_SKILL_GRAPH_VERSION = "video-generation-skill-graph-v1"


@dataclass(frozen=True)
class VideoProductionSkill:
    skill_id: str
    title: str
    priority: float
    route_terms: tuple[str, ...]
    preferred_arc: tuple[str, ...]
    visual_language: tuple[str, ...]
    portfolio_constraints: dict[str, Any]
    reject_rules: tuple[str, ...]
    renderer_policy: tuple[str, ...] = ("hyperframes",)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BeatSceneSkill:
    scene_type: str
    skill_id: str
    title: str
    hyperframes_skill_id: str
    stage_families: tuple[str, ...]
    proof_encodings: tuple[str, ...]
    visual_world_mediums: tuple[str, ...]
    required_slots: tuple[str, ...]
    renderer_route: str
    qa_floor: float
    motion_technique: str
    camera_move: str
    effect_stack: tuple[str, ...]
    candidate_order: tuple[str, ...]
    reject_rules: tuple[str, ...]
    anti_patterns: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BeatSkillAssignment:
    version: str
    beat_id: str
    index: int
    passed: bool
    arc_role: str
    skill_id: str
    skill_title: str
    scene_type: str
    renderer_route: str
    score: float
    required_slots: list[str]
    missing_slots: list[str]
    slot_values: dict[str, Any]
    semantic_frame: dict[str, Any]
    required_labels: list[str]
    metric_facts: list[dict[str, Any]]
    stage_families: list[str]
    proof_encodings: list[str]
    visual_world_mediums: list[str]
    skill_slices: list[dict[str, Any]]
    candidate_order: list[str]
    qa_floor: float
    motion_technique: str
    camera_move: str
    effect_stack: list[str]
    transition_intent: str
    continuity_subject: str
    continuity_key: str
    reject_rules: list[str]
    anti_patterns: list[str]
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VideoSkillGraph:
    version: str
    passed: bool
    score: float
    production_skill: dict[str, Any]
    request_classification: dict[str, Any]
    arc_contract: dict[str, Any]
    continuity_ledger: dict[str, Any]
    portfolio_constraints: dict[str, Any]
    beat_assignments: list[BeatSkillAssignment]
    warnings: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    prompt_block: str = ""

    @property
    def production_skill_id(self) -> str:
        return str(self.production_skill.get("skill_id") or "")

    @property
    def assignment_count(self) -> int:
        return len(self.beat_assignments)

    @property
    def accepted_count(self) -> int:
        return sum(1 for item in self.beat_assignments if item.passed)

    @property
    def coverage(self) -> float:
        return round(self.accepted_count / max(self.assignment_count, 1), 4)

    def assignment_for(self, beat_id: str) -> BeatSkillAssignment | None:
        for item in self.beat_assignments:
            if item.beat_id == beat_id:
                return item
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "passed": self.passed,
            "score": round(float(self.score), 4),
            "production_skill": dict(self.production_skill),
            "production_skill_id": self.production_skill_id,
            "request_classification": dict(self.request_classification),
            "arc_contract": dict(self.arc_contract),
            "continuity_ledger": dict(self.continuity_ledger),
            "portfolio_constraints": dict(self.portfolio_constraints),
            "assignment_count": self.assignment_count,
            "accepted_count": self.accepted_count,
            "coverage": self.coverage,
            "beat_assignments": [item.to_dict() for item in self.beat_assignments],
            "warnings": list(self.warnings),
            "issues": list(self.issues),
            "prompt_block": self.prompt_block,
        }


PRODUCTION_SKILLS: tuple[VideoProductionSkill, ...] = (
    VideoProductionSkill(
        skill_id="architecture-demo",
        title="Architecture Demo And System Lifecycle",
        priority=0.98,
        route_terms=("api", "service", "gateway", "worker", "renderer", "planner", "pipeline", "database", "queue"),
        preferred_arc=("hook", "system_boundary", "request_route", "mechanism", "proof", "payoff"),
        visual_language=("explicit service boundaries", "single request token", "return path", "ownership labels"),
        portfolio_constraints={
            "min_skill_coverage": 0.86,
            "require_request_token": True,
            "max_same_scene_type_run": 2,
        },
        reject_rules=(
            "Reject invented infrastructure, unlabeled boxes, or bidirectional arrows without request/response meaning.",
            "Every service, queue, worker, or renderer must be named by the prompt or script.",
        ),
    ),
    VideoProductionSkill(
        skill_id="metric-proof-video",
        title="Metric Proof And Measured Change",
        priority=0.97,
        route_terms=("%", " ms", "x", "latency", "speed", "tokens", "users", "gb", "mb", "score", "conversion"),
        preferred_arc=("hook", "baseline", "intervention", "measurement", "proof", "payoff"),
        visual_language=("source-backed metrics", "registered before/after states", "intervention trace"),
        portfolio_constraints={
            "min_skill_coverage": 0.86,
            "forbid_unverified_numbers": True,
            "max_same_scene_type_run": 2,
        },
        reject_rules=(
            "Reject any displayed number that does not appear in source evidence.",
            "Reject floating metrics with no evidence structure or intervention path.",
        ),
    ),
    VideoProductionSkill(
        skill_id="product-walkthrough",
        title="Grounded Product Or Interface Walkthrough",
        priority=0.96,
        route_terms=("ui", "interface", "screen", "dashboard", "editor", "click", "open", "retry", "log", "button"),
        preferred_arc=("hook", "state", "action", "feedback", "result", "payoff"),
        visual_language=("stable interface surface", "focus region", "action trace", "result feedback"),
        portfolio_constraints={
            "min_skill_coverage": 0.84,
            "prefer_source_ui": True,
            "forbid_fake_interface_states": True,
        },
        reject_rules=(
            "Reject imaginary controls, progress percentages, status rows, logs, or dashboards.",
            "Keep the action target and resulting state in one stable interface context.",
        ),
    ),
    VideoProductionSkill(
        skill_id="decision-story",
        title="Decision Gate And Guardrail Story",
        priority=0.94,
        route_terms=("if", "otherwise", "decision", "guardrail", "confidence", "branch", "choose", "gate", "review"),
        preferred_arc=("hook", "condition", "low_branch", "high_branch", "selected_route", "payoff"),
        visual_language=("single decision gate", "exclusive active branch", "protected downstream constraint"),
        portfolio_constraints={
            "min_skill_coverage": 0.82,
            "require_exclusive_branches": True,
            "max_same_scene_type_run": 2,
        },
        reject_rules=(
            "Reject generic yes/no branches.",
            "Reject both branches appearing active at the same beat.",
        ),
    ),
    VideoProductionSkill(
        skill_id="process-trace",
        title="Process Trace And Handoff Choreography",
        priority=0.92,
        route_terms=("then", "step", "process", "flow", "route", "loop", "handoff", "mechanism", "sequence"),
        preferred_arc=("hook", "input", "route", "handoff", "result", "payoff"),
        visual_language=("source-ordered steps", "persistent traveler", "completion trace"),
        portfolio_constraints={
            "min_skill_coverage": 0.82,
            "require_step_order": True,
            "max_same_scene_type_run": 2,
        },
        reject_rules=(
            "Reject disconnected numbered cards.",
            "Reject process visuals where all steps appear simultaneously.",
        ),
    ),
    VideoProductionSkill(
        skill_id="narrative-proof",
        title="Narrative Proof And Recovery Arc",
        priority=0.88,
        route_terms=("failed", "recovered", "before", "after", "turn", "story", "proof", "payoff", "finally"),
        preferred_arc=("hook", "setup", "turn", "evidence", "payoff"),
        visual_language=("persistent subject", "turning point", "trajectory change", "resolved hold"),
        portfolio_constraints={
            "min_skill_coverage": 0.78,
            "require_persistent_subject": True,
            "max_same_scene_type_run": 2,
        },
        reject_rules=(
            "Reject unrelated subjects across setup, turn, and payoff.",
            "Reject a turning point that does not alter visible trajectory.",
        ),
    ),
    VideoProductionSkill(
        skill_id="technical-explainer",
        title="Technical Explainer And Visible Mechanism",
        priority=0.84,
        route_terms=("attention", "retrieval", "generation", "model", "token", "context", "signal", "evidence", "compute"),
        preferred_arc=("hook", "problem", "mechanism", "proof", "payoff"),
        visual_language=("cause trace", "state transform", "semantic route", "proof marker"),
        portfolio_constraints={
            "min_skill_coverage": 0.78,
            "require_mechanism": True,
            "max_same_scene_type_run": 2,
        },
        reject_rules=(
            "Reject concepts shown only as title cards.",
            "Reject visuals that could be swapped with generic boxes without losing meaning.",
        ),
    ),
    VideoProductionSkill(
        skill_id="quote-manifesto",
        title="Quote-Led Manifesto",
        priority=0.7,
        route_terms=("quote", "phrase", "says", "line", "manifesto"),
        preferred_arc=("hook", "phrase", "meaning", "proof", "payoff"),
        visual_language=("exact quote", "decisive phrase emphasis", "readable final hold"),
        portfolio_constraints={
            "min_skill_coverage": 0.72,
            "preserve_exact_quote": True,
            "max_same_scene_type_run": 2,
        },
        reject_rules=(
            "Reject paraphrased quoted language.",
            "Reject word clouds or fabricated keywords.",
        ),
    ),
)


SCENE_SKILLS: dict[str, BeatSceneSkill] = {
    item.scene_type: item
    for item in (
        BeatSceneSkill(
            scene_type="metric_intervention",
            skill_id="metric-story",
            title="Measured Intervention Proof",
            hyperframes_skill_id="hyperframes-metric-story",
            stage_families=("semantic_metric",),
            proof_encodings=("linear_trace", "split_register", "layered_flow"),
            visual_world_mediums=("data_sculpture", "diagrammatic_system"),
            required_slots=("metric_facts", "before_state", "after_state", "intervention"),
            renderer_route="hyperframes",
            qa_floor=0.82,
            motion_technique="particle_sculpture_orbit",
            camera_move="slow_orbital_push",
            effect_stack=("tracked_metric", "intervention_gate", "proof_lock"),
            candidate_order=("skill", "causal", "matched_transform", "process", "quote"),
            reject_rules=("Do not invent thresholds or synthetic deltas.",),
            anti_patterns=("floating metric", "fake dashboard", "random bars"),
        ),
        BeatSceneSkill(
            scene_type="metric_delta",
            skill_id="metric-story",
            title="Measured Before/After Delta",
            hyperframes_skill_id="hyperframes-metric-story",
            stage_families=("semantic_metric",),
            proof_encodings=("split_register", "linear_trace", "layered_flow"),
            visual_world_mediums=("data_sculpture", "diagrammatic_system"),
            required_slots=("metric_facts", "before_state", "after_state"),
            renderer_route="hyperframes",
            qa_floor=0.8,
            motion_technique="particle_sculpture_orbit",
            camera_move="slow_orbital_push",
            effect_stack=("matched_state_geometry", "measurement_ticks", "delta_annotation"),
            candidate_order=("skill", "matched_transform", "causal", "quote", "process"),
            reject_rules=("Do not imply intervention causality unless the source names it.",),
            anti_patterns=("unlabeled chart scale", "decorative sparkline"),
        ),
        BeatSceneSkill(
            scene_type="metric_proof",
            skill_id="metric-story",
            title="Metric Evidence Proof",
            hyperframes_skill_id="hyperframes-metric-story",
            stage_families=("semantic_metric",),
            proof_encodings=("layered_flow", "linear_trace", "radial_evidence"),
            visual_world_mediums=("data_sculpture",),
            required_slots=("metric_facts", "result"),
            renderer_route="hyperframes",
            qa_floor=0.78,
            motion_technique="particle_sculpture_orbit",
            camera_move="slow_orbital_push",
            effect_stack=("evidence_geometry", "proof_marker", "resolved_hold"),
            candidate_order=("skill", "causal", "quote", "matched_transform", "process"),
            reject_rules=("Keep the number attached to one visible evidence structure.",),
            anti_patterns=("giant isolated number", "metric wallpaper"),
        ),
        BeatSceneSkill(
            scene_type="architecture_flow",
            skill_id="architecture-flow",
            title="Architecture Flow",
            hyperframes_skill_id="hyperframes-architecture-flow",
            stage_families=("semantic_architecture",),
            proof_encodings=("linear_trace", "layered_flow", "split_register"),
            visual_world_mediums=("diagrammatic_system", "product_interface"),
            required_slots=("steps", "route_token", "service_boundaries"),
            renderer_route="hyperframes",
            qa_floor=0.8,
            motion_technique="routed_system_trace",
            camera_move="guided_center_push",
            effect_stack=("route_draw", "node_pulse", "scanline"),
            candidate_order=("skill", "process", "causal", "matched_transform", "quote"),
            reject_rules=("Do not add unnamed services, queues, databases, or workers.",),
            anti_patterns=("network wallpaper", "unlabeled boxes"),
        ),
        BeatSceneSkill(
            scene_type="guided_process",
            skill_id="route-choreography",
            title="Guided Process Route",
            hyperframes_skill_id="hyperframes-route-choreography",
            stage_families=("semantic_route",),
            proof_encodings=("linear_trace", "layered_flow", "radial_evidence"),
            visual_world_mediums=("diagrammatic_system", "spatial_metaphor"),
            required_slots=("steps",),
            renderer_route="hyperframes",
            qa_floor=0.76,
            motion_technique="routed_system_trace",
            camera_move="guided_center_push",
            effect_stack=("route_draw", "completion_trace", "handoff_bridge"),
            candidate_order=("skill", "process", "causal", "matched_transform", "quote"),
            reject_rules=("Preserve one traveler through ordered handoffs.",),
            anti_patterns=("disconnected cards", "all steps revealed at once"),
        ),
        BeatSceneSkill(
            scene_type="causal_intervention",
            skill_id="causal-spine",
            title="Causal Mechanism Spine",
            hyperframes_skill_id="hyperframes-causal-spine",
            stage_families=("semantic_causal",),
            proof_encodings=("layered_flow", "focal_gate", "linear_trace"),
            visual_world_mediums=("spatial_metaphor", "diagrammatic_system"),
            required_slots=("problem", "mechanism", "result"),
            renderer_route="hyperframes",
            qa_floor=0.8,
            motion_technique="semantic_motion_stage",
            camera_move="guided_center_push",
            effect_stack=("mechanism_chamber", "cause_trace", "result_lock"),
            candidate_order=("skill", "causal", "process", "matched_transform", "quote"),
            reject_rules=("Do not skip the mechanism and jump from problem to result.",),
            anti_patterns=("static cause/effect cards", "decorative arrows"),
        ),
        BeatSceneSkill(
            scene_type="matched_state_transform",
            skill_id="matched-transform",
            title="Matched State Transform",
            hyperframes_skill_id="hyperframes-matched-transform",
            stage_families=("semantic_transform",),
            proof_encodings=("split_register", "focal_gate", "linear_trace"),
            visual_world_mediums=("spatial_metaphor", "editorial_collage"),
            required_slots=("before_state", "after_state"),
            renderer_route="hyperframes",
            qa_floor=0.77,
            motion_technique="pseudo_3d_arc_motion",
            camera_move="split_reveal_slide",
            effect_stack=("matched_geometry", "difference_highlight", "constraint_anchor"),
            candidate_order=("skill", "matched_transform", "causal", "quote", "process"),
            reject_rules=("Keep before and after geometry registered.",),
            anti_patterns=("static versus cards", "generic improvement glow"),
        ),
        BeatSceneSkill(
            scene_type="grounded_interface_walkthrough",
            skill_id="grounded-interface",
            title="Grounded Interface Walkthrough",
            hyperframes_skill_id="hyperframes-grounded-interface",
            stage_families=("semantic_interface",),
            proof_encodings=("focal_gate", "linear_trace", "split_register"),
            visual_world_mediums=("product_interface", "source_media_composite"),
            required_slots=("screen", "action", "result"),
            renderer_route="hyperframes",
            qa_floor=0.82,
            motion_technique="ui_camera_push_focus",
            camera_move="interface_push_in",
            effect_stack=("focus_ring", "cursor_trace", "feedback_path"),
            candidate_order=("skill", "causal", "matched_transform", "process", "quote"),
            reject_rules=("Never invent controls, logs, status rows, or percentages.",),
            anti_patterns=("fake dashboards", "unrelated UI chrome"),
        ),
        BeatSceneSkill(
            scene_type="decision_branch",
            skill_id="decision-gate",
            title="Decision Gate",
            hyperframes_skill_id="hyperframes-decision-gate",
            stage_families=("semantic_decision",),
            proof_encodings=("focal_gate", "split_register", "linear_trace"),
            visual_world_mediums=("diagrammatic_system", "spatial_metaphor"),
            required_slots=("decision", "low_branch", "high_branch"),
            renderer_route="hyperframes",
            qa_floor=0.8,
            motion_technique="routed_system_trace",
            camera_move="guided_center_push",
            effect_stack=("decision_gate", "exclusive_branch", "guardrail"),
            candidate_order=("skill", "causal", "matched_transform", "process", "quote"),
            reject_rules=("Only one branch may appear active at a time.",),
            anti_patterns=("generic yes/no", "traffic-light decoration"),
        ),
        BeatSceneSkill(
            scene_type="narrative_progression",
            skill_id="narrative-continuity",
            title="Narrative Progression",
            hyperframes_skill_id="hyperframes-narrative-continuity",
            stage_families=("semantic_narrative",),
            proof_encodings=("linear_trace", "layered_flow", "radial_evidence"),
            visual_world_mediums=("editorial_collage", "spatial_metaphor"),
            required_slots=("setup", "intervention", "result"),
            renderer_route="hyperframes",
            qa_floor=0.74,
            motion_technique="parallax_editorial_collage",
            camera_move="paper_parallax_pan",
            effect_stack=("story_arc", "turn_marker", "resolved_hold"),
            candidate_order=("skill", "matched_transform", "causal", "quote", "process"),
            reject_rules=("Keep one persistent subject across setup, turn, and payoff.",),
            anti_patterns=("generic filmstrip", "new subject per beat"),
        ),
        BeatSceneSkill(
            scene_type="evidence_backed_quote",
            skill_id="exact-quote",
            title="Evidence-Backed Quote",
            hyperframes_skill_id="hyperframes-exact-quote",
            stage_families=("semantic_quote",),
            proof_encodings=("linear_trace", "focal_gate"),
            visual_world_mediums=("kinetic_typography", "editorial_collage"),
            required_slots=("exact_quote",),
            renderer_route="hyperframes",
            qa_floor=0.72,
            motion_technique="per_word_kinetic_type",
            camera_move="type_scale_snap",
            effect_stack=("phrase_segments", "emphasis_lock", "resolved_hold"),
            candidate_order=("skill", "quote", "matched_transform", "causal", "process"),
            reject_rules=("Preserve exact source language.",),
            anti_patterns=("word clouds", "fabricated keywords"),
        ),
    )
}


def build_video_skill_graph(
    *,
    request: VideoGenerationRequest,
    plan: ScriptPlan,
    beat_graph: BeatGraph,
    director_package: DirectorPackage | None = None,
) -> VideoSkillGraph:
    request_text = _clean(" ".join([request.prompt, request.script, plan.narration, plan.design_direction]))
    production_skill, classification = _select_production_skill(
        request=request,
        plan=plan,
        beat_graph=beat_graph,
        request_text=request_text,
    )
    assignments = [
        _assignment_for_beat(
            beat,
            request=request,
            plan=plan,
            beat_graph=beat_graph,
            production_skill=production_skill,
            director_package=director_package,
        )
        for beat in beat_graph.beats
    ]
    continuity = _continuity_ledger(assignments, request_text=request_text)
    arc_contract = _arc_contract(production_skill, assignments, beat_graph=beat_graph)
    issues: list[str] = []
    warnings: list[str] = []
    coverage = sum(1 for item in assignments if item.passed) / max(len(assignments), 1)
    min_coverage = float(production_skill.portfolio_constraints.get("min_skill_coverage") or 0.78)
    if coverage < min_coverage:
        issues.append("video_skill_graph_coverage_below_production_floor")
    if len(assignments) < len(beat_graph.beats):
        issues.append("video_skill_graph_missing_beat_assignments")
    if _longest_scene_run(assignments) > int(production_skill.portfolio_constraints.get("max_same_scene_type_run") or 2):
        warnings.append("video_skill_graph_repeats_scene_type_run")
    if not any(item.arc_role == "hook" for item in assignments) and len(assignments) >= 2:
        warnings.append("video_skill_graph_missing_hook_role")
    score = _graph_score(assignments, coverage=coverage, warnings=warnings, issues=issues)
    graph = VideoSkillGraph(
        version=VIDEO_GENERATION_SKILL_GRAPH_VERSION,
        passed=not issues,
        score=score,
        production_skill=production_skill.to_dict(),
        request_classification=classification,
        arc_contract=arc_contract,
        continuity_ledger=continuity,
        portfolio_constraints=dict(production_skill.portfolio_constraints),
        beat_assignments=assignments,
        warnings=warnings,
        issues=issues,
    )
    return VideoSkillGraph(
        version=graph.version,
        passed=graph.passed,
        score=graph.score,
        production_skill=graph.production_skill,
        request_classification=graph.request_classification,
        arc_contract=graph.arc_contract,
        continuity_ledger=graph.continuity_ledger,
        portfolio_constraints=graph.portfolio_constraints,
        beat_assignments=graph.beat_assignments,
        warnings=graph.warnings,
        issues=graph.issues,
        prompt_block=skill_graph_prompt_block(graph),
    )


def skill_graph_prompt_block(graph: VideoSkillGraph | dict[str, Any]) -> str:
    payload = graph.to_dict() if hasattr(graph, "to_dict") else dict(graph or {})
    production = dict(payload.get("production_skill") or {})
    lines = [
        "Video Generation Skill Graph:",
        f"- Production skill: {production.get('skill_id', '')} - {production.get('title', '')}.",
        "- Treat beat skill assignments as hard architecture, not suggestions.",
        "- Fill visible copy only from prompt/script/narration evidence. Do not invent metrics, UI states, services, branches, or outcomes.",
        "- Reroute or reject a beat when its required slots cannot be grounded.",
    ]
    for item in payload.get("beat_assignments") or []:
        if not isinstance(item, dict):
            continue
        labels = ", ".join(str(label) for label in item.get("required_labels") or [] if str(label))[:180]
        lines.append(
            (
                f"- {item.get('beat_id')}: role={item.get('arc_role')} "
                f"skill={item.get('skill_id')} scene={item.get('scene_type')} "
                f"renderer={item.get('renderer_route')} labels={labels}"
            )
        )
    return "\n".join(lines)


def assignment_payload(graph: VideoSkillGraph | dict[str, Any] | None, beat_id: str) -> dict[str, Any]:
    if graph is None:
        return {}
    if hasattr(graph, "assignment_for"):
        assignment = graph.assignment_for(beat_id)  # type: ignore[attr-defined]
        return assignment.to_dict() if assignment is not None else {}
    payload = dict(graph or {})
    for item in payload.get("beat_assignments") or []:
        if isinstance(item, dict) and item.get("beat_id") == beat_id:
            return dict(item)
    return {}


def _select_production_skill(
    *,
    request: VideoGenerationRequest,
    plan: ScriptPlan,
    beat_graph: BeatGraph,
    request_text: str,
) -> tuple[VideoProductionSkill, dict[str, Any]]:
    normalized = _normalize(request_text)
    scene_counts: dict[str, int] = {}
    for beat in beat_graph.beats:
        scene_counts[beat.scene_type] = scene_counts.get(beat.scene_type, 0) + 1
    scored: list[tuple[float, VideoProductionSkill, list[str]]] = []
    for skill in PRODUCTION_SKILLS:
        reasons: list[str] = []
        hits = sum(1 for term in skill.route_terms if _term_hit(term, normalized))
        score = skill.priority + min(hits * 0.055, 0.33)
        if skill.skill_id == "metric-proof-video" and _has_content_metric(request_text):
            score += 0.12
            reasons.append("numeric_source_evidence")
        if skill.skill_id == "architecture-demo" and _architecture_term_count(request_text) >= 2:
            score += 0.16
            reasons.append("multiple_architecture_terms")
        if skill.skill_id == "decision-story" and _decision_parts(request_text):
            score += 0.14
            reasons.append("explicit_decision_language")
        if skill.skill_id == "product-walkthrough" and _ui_term_count(request_text) >= 2:
            score += 0.14
            reasons.append("interface_language")
        if skill.skill_id == "process-trace" and scene_counts.get("process", 0) >= 1:
            score += 0.08
            reasons.append("process_beats_present")
        if skill.skill_id == "technical-explainer" and not reasons:
            score += 0.02
            reasons.append("default_technical_showrunner")
        if hits:
            reasons.append(f"route_term_hits:{hits}")
        scored.append((score, skill, reasons))
    score, selected, reasons = max(scored, key=lambda item: (item[0], item[1].priority, item[1].skill_id))
    return selected, {
        "version": "video-production-skill-classifier-v1",
        "selected_skill_id": selected.skill_id,
        "score": round(min(score, 1.0), 4),
        "reasons": reasons,
        "scene_counts": scene_counts,
        "aspect": request.aspect,
        "style": request.style,
        "has_script": bool(request.script),
        "generate_audio": request.generate_audio,
    }


def _assignment_for_beat(
    beat: Beat,
    *,
    request: VideoGenerationRequest,
    plan: ScriptPlan,
    beat_graph: BeatGraph,
    production_skill: VideoProductionSkill,
    director_package: DirectorPackage | None,
) -> BeatSkillAssignment:
    contract = director_package.beat_contract(beat.beat_id) if director_package else None
    # Beat-level grounding must not treat request instructions such as
    # "12 seconds" or "portrait" as content facts for every scene.
    source_text = _clean(" ".join([beat.narration, beat.title, " ".join(beat.keywords)]))
    labels = _required_labels(beat, source_text=source_text, contract=contract)
    metrics = _metric_facts(source_text)
    target_scene = _target_scene_type(
        beat,
        source_text=source_text,
        labels=labels,
        metrics=metrics,
        production_skill=production_skill,
    )
    scene_skill = SCENE_SKILLS[target_scene]
    semantic_frame, slot_values = _semantic_frame_for(
        scene_skill.scene_type,
        beat=beat,
        labels=labels,
        metrics=metrics,
        source_text=source_text,
        contract=contract,
    )
    missing = [
        slot
        for slot in scene_skill.required_slots
        if not _has_slot(slot_values.get(slot))
    ]
    arc_role = _arc_role_for(beat, beat_graph=beat_graph, target_scene=target_scene)
    score = _assignment_score(
        scene_skill,
        labels=labels,
        metrics=metrics,
        missing_slots=missing,
        source_text=source_text,
        arc_role=arc_role,
    )
    passed = not missing and score >= 0.56 and bool(semantic_frame)
    reasons = [
        f"production_skill:{production_skill.skill_id}",
        f"scene_skill:{scene_skill.skill_id}",
        f"arc_role:{arc_role}",
    ]
    if passed:
        reasons.append("beat_skill_assignment_passed")
    warnings: list[str] = []
    if missing:
        warnings.extend(f"missing_slot:{item}" for item in missing)
    if not labels:
        warnings.append("no_source_grounded_labels")
    slices = retrieve_skill_slices(
        scene_skill.stage_families[0],
        scene_type=scene_skill.scene_type,
        limit=2,
    )
    continuity_subject = _continuity_subject(labels, source_text=source_text)
    return BeatSkillAssignment(
        version=VIDEO_GENERATION_SKILL_GRAPH_VERSION,
        beat_id=beat.beat_id,
        index=beat.index,
        passed=passed,
        arc_role=arc_role,
        skill_id=scene_skill.skill_id,
        skill_title=scene_skill.title,
        scene_type=scene_skill.scene_type,
        renderer_route=scene_skill.renderer_route,
        score=round(score, 4),
        required_slots=list(scene_skill.required_slots),
        missing_slots=missing,
        slot_values=slot_values,
        semantic_frame=semantic_frame,
        required_labels=labels,
        metric_facts=metrics,
        stage_families=list(scene_skill.stage_families),
        proof_encodings=list(scene_skill.proof_encodings),
        visual_world_mediums=list(scene_skill.visual_world_mediums),
        skill_slices=[item.to_dict() for item in slices],
        candidate_order=list(scene_skill.candidate_order),
        qa_floor=scene_skill.qa_floor,
        motion_technique=scene_skill.motion_technique,
        camera_move=scene_skill.camera_move,
        effect_stack=list(scene_skill.effect_stack),
        transition_intent=_transition_intent(arc_role, scene_skill.scene_type, is_last=beat.index >= len(beat_graph.beats)),
        continuity_subject=continuity_subject,
        continuity_key=_safe_key(continuity_subject or scene_skill.scene_type),
        reject_rules=[*production_skill.reject_rules, *scene_skill.reject_rules],
        anti_patterns=list(scene_skill.anti_patterns),
        reasons=reasons,
        warnings=warnings,
    )


def _target_scene_type(
    beat: Beat,
    *,
    source_text: str,
    labels: list[str],
    metrics: list[dict[str, Any]],
    production_skill: VideoProductionSkill,
) -> str:
    decision = _decision_parts(source_text)
    before_after = _before_after_pair(source_text)
    steps = _step_labels(source_text, labels=labels)
    if production_skill.skill_id == "decision-story" and decision:
        return "decision_branch"
    if production_skill.skill_id == "product-walkthrough" and _ui_term_count(source_text) >= 1 and len(labels) >= 2:
        return "grounded_interface_walkthrough"
    if metrics:
        if before_after and _has_intervention(source_text):
            return "metric_intervention"
        if before_after or len(metrics) >= 2:
            return "metric_delta"
        return "metric_proof"
    if production_skill.skill_id == "architecture-demo" and len(steps) >= 2:
        return "architecture_flow"
    if decision:
        return "decision_branch"
    if len(steps) >= 3 and _architecture_term_count(source_text) >= 2:
        return "architecture_flow"
    if len(steps) >= 2 or beat.scene_type == "process":
        return "guided_process"
    if before_after:
        return "matched_state_transform"
    if _ui_term_count(source_text) >= 2 and len(labels) >= 2:
        return "grounded_interface_walkthrough"
    if production_skill.skill_id == "narrative-proof" and len(labels) >= 3:
        return "narrative_progression"
    if len(labels) >= 3 and (beat.scene_type in {"proof", "contrast"} or _causal_language(source_text)):
        return "causal_intervention"
    if beat.index == 1 and len(_words(beat.narration)) <= 14:
        return "evidence_backed_quote"
    if len(labels) >= 2:
        return "causal_intervention"
    return "evidence_backed_quote"


def _semantic_frame_for(
    scene_type: str,
    *,
    beat: Beat,
    labels: list[str],
    metrics: list[dict[str, Any]],
    source_text: str,
    contract: Any | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    label0 = labels[0] if labels else _short_sentence(beat.narration, max_words=6)
    label1 = labels[1] if len(labels) > 1 else label0
    label2 = labels[2] if len(labels) > 2 else label1
    label_last = labels[-1] if labels else label0
    pair = _before_after_pair(source_text)
    decision = _decision_parts(source_text)
    steps = _step_labels(source_text, labels=labels)
    if scene_type in {"metric_intervention", "metric_delta", "metric_proof"}:
        before = pair[0] if pair else label0
        after = pair[1] if pair else label_last
        intervention = _intervention_label(source_text, labels=labels) or label1
        frame = {
            "before_state": before,
            "after_state": after,
            "intervention": intervention,
            "result": after,
            "viewer_takeaway": _short_sentence(beat.narration, max_words=10),
        }
        if scene_type == "metric_proof":
            frame["mechanism"] = label1
        slots = {
            "metric_facts": metrics,
            "before_state": before,
            "after_state": after,
            "intervention": intervention,
            "result": after,
        }
        return _grounded_frame(frame, source_text), slots
    if scene_type == "architecture_flow":
        route_steps = steps or labels[:4]
        frame = {
            "steps": route_steps,
            "mechanism": route_steps[0] if route_steps else label0,
            "result": route_steps[-1] if route_steps else label_last,
            "viewer_takeaway": _short_sentence(beat.narration, max_words=10),
        }
        slots = {
            "steps": route_steps,
            "service_boundaries": route_steps,
            "route_token": label0,
            "result": frame["result"],
        }
        return _grounded_frame(frame, source_text, allow_lists=True), slots
    if scene_type == "guided_process":
        route_steps = steps or labels[:4]
        frame = {
            "steps": route_steps,
            "mechanism": route_steps[0] if route_steps else label0,
            "result": route_steps[-1] if route_steps else label_last,
            "viewer_takeaway": _short_sentence(beat.narration, max_words=10),
        }
        slots = {"steps": route_steps, "result": frame["result"]}
        return _grounded_frame(frame, source_text, allow_lists=True), slots
    if scene_type == "decision_branch" and decision:
        condition, low_branch, high_branch = decision
        frame = {
            "decision": condition,
            "low_branch": low_branch,
            "high_branch": high_branch,
            "result": high_branch,
            "viewer_takeaway": _short_sentence(beat.narration, max_words=10),
        }
        slots = {
            "decision": condition,
            "low_branch": low_branch,
            "high_branch": high_branch,
            "result": high_branch,
        }
        return _grounded_frame(frame, source_text), slots
    if scene_type == "matched_state_transform":
        before = pair[0] if pair else label0
        after = pair[1] if pair else label_last
        frame = {
            "before_state": before,
            "after_state": after,
            "preserved_constraint": _contract_relation(contract) or label1,
            "viewer_takeaway": after,
        }
        slots = {"before_state": before, "after_state": after, "constraint": frame["preserved_constraint"]}
        return _grounded_frame(frame, source_text), slots
    if scene_type == "grounded_interface_walkthrough":
        action = _intervention_label(source_text, labels=labels) or label1
        result = label_last
        screen = label0
        frame = {
            "screen": screen,
            "action": action,
            "intervention": action,
            "result": result,
            "viewer_takeaway": result,
        }
        slots = {"screen": screen, "action": action, "result": result}
        return _grounded_frame(frame, source_text), slots
    if scene_type == "narrative_progression":
        frame = {
            "setup": label0,
            "intervention": label1,
            "result": label_last,
            "payoff": label_last,
            "viewer_takeaway": _short_sentence(beat.narration, max_words=10),
        }
        slots = {"setup": label0, "intervention": label1, "result": label_last}
        return _grounded_frame(frame, source_text), slots
    if scene_type == "causal_intervention":
        frame = {
            "problem": label0,
            "mechanism": label1,
            "intervention": label2 if label2 != label1 else label1,
            "result": label_last,
            "viewer_takeaway": label_last,
        }
        slots = {"problem": label0, "mechanism": label1, "intervention": frame["intervention"], "result": label_last}
        return _grounded_frame(frame, source_text), slots
    quote = _short_sentence(beat.narration, max_words=12)
    return {"exact_quote": quote, "viewer_takeaway": quote}, {"exact_quote": quote}


def _required_labels(beat: Beat, *, source_text: str, contract: Any | None) -> list[str]:
    labels: list[str] = []
    if contract is not None:
        labels.extend(str(item) for item in getattr(contract, "required_objects", []) or [])
    labels.extend(beat.keywords)
    labels.extend(_clause_labels(beat.narration, limit=5))
    result: list[str] = []
    source_key = _normalize(source_text)
    for label in labels:
        cleaned = _label(label)
        if not cleaned or _too_generic(cleaned):
            continue
        tokens = [token for token in _words(cleaned) if len(token) >= 3]
        if tokens and not any(token in source_key for token in tokens):
            continue
        if cleaned.lower() not in {item.lower() for item in result}:
            result.append(cleaned)
        if len(result) >= 7:
            break
    if not result:
        result.extend(_label(item) for item in keyword_candidates(source_text, limit=4))
        result = [item for item in result if item and not _too_generic(item)]
    return result[:7]


def _metric_facts(source_text: str) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for index, match in enumerate(
        re.finditer(r"\b\d+(?:\.\d+)?\s*(?:%|x|ms|seconds?|tokens?|users?|gb|mb|billion|million)?\b", source_text, flags=re.IGNORECASE),
        start=1,
    ):
        value = re.sub(r"\s+", " ", match.group(0)).strip()
        facts.append(
            {
                "fact_id": f"video_metric_{index:02d}",
                "label": value,
                "value": value,
                "unit": _metric_unit(value),
                "source": "beat_text",
                "grounding": "source_text",
            }
        )
        if len(facts) >= 4:
            break
    return facts


def _metric_unit(value: str) -> str:
    match = re.search(r"(?:%|x|ms|seconds?|tokens?|users?|gb|mb|billion|million)\b", value, flags=re.IGNORECASE)
    return match.group(0).lower() if match else "count"


def _arc_role_for(beat: Beat, *, beat_graph: BeatGraph, target_scene: str) -> str:
    if beat.index <= 1:
        return "hook"
    if beat.index >= len(beat_graph.beats):
        return "payoff"
    if target_scene in {"metric_delta", "metric_intervention", "metric_proof"}:
        return "proof"
    if target_scene in {"guided_process", "architecture_flow"}:
        return "mechanism"
    if target_scene == "decision_branch":
        return "decision"
    if target_scene == "matched_state_transform":
        return "contrast"
    if target_scene == "grounded_interface_walkthrough":
        return "action"
    if target_scene == "narrative_progression":
        return "turn"
    return beat.scene_type or "concept"


def _assignment_score(
    scene_skill: BeatSceneSkill,
    *,
    labels: list[str],
    metrics: list[dict[str, Any]],
    missing_slots: list[str],
    source_text: str,
    arc_role: str,
) -> float:
    score = 0.58
    score += min(len(labels), 4) * 0.045
    score += min(len(metrics), 2) * 0.05
    score += 0.04 if arc_role in {"hook", "mechanism", "proof", "payoff", "decision"} else 0.0
    score += 0.04 if any(term in _normalize(source_text) for term in scene_skill.skill_id.split("-")) else 0.0
    score -= len(missing_slots) * 0.14
    if _too_generic(" ".join(labels[:2])):
        score -= 0.08
    return max(0.0, min(score, 1.0))


def _arc_contract(
    production_skill: VideoProductionSkill,
    assignments: list[BeatSkillAssignment],
    *,
    beat_graph: BeatGraph,
) -> dict[str, Any]:
    roles = [item.arc_role for item in assignments]
    return {
        "version": "video-skill-arc-contract-v1",
        "production_skill_id": production_skill.skill_id,
        "preferred_arc": list(production_skill.preferred_arc),
        "actual_arc": roles,
        "beat_count": len(beat_graph.beats),
        "has_hook": "hook" in roles,
        "has_mechanism": any(role in roles for role in {"mechanism", "action", "decision"}),
        "has_proof": any(role in roles for role in {"proof", "contrast"}),
        "has_payoff": "payoff" in roles,
    }


def _continuity_ledger(assignments: list[BeatSkillAssignment], *, request_text: str) -> dict[str, Any]:
    subjects = _unique(
        [
            item.continuity_subject
            for item in assignments
            if item.continuity_subject
        ],
        limit=12,
    )
    recurring = [
        subject
        for subject in subjects
        if sum(1 for item in assignments if item.continuity_subject == subject) > 1
    ]
    return {
        "version": "video-continuity-ledger-v1",
        "primary_subject": subjects[0] if subjects else _label(" ".join(keyword_candidates(request_text, limit=2))),
        "subjects": subjects,
        "recurring_subjects": recurring,
        "scene_types": [item.scene_type for item in assignments],
        "renderer_routes": sorted({item.renderer_route for item in assignments if item.renderer_route}),
        "world_memory_policy": "carry recent visual-world fingerprints and avoid duplicate signatures",
        "object_identity_policy": "preserve named subjects across transforms, routes, and payoffs",
    }


def _graph_score(
    assignments: list[BeatSkillAssignment],
    *,
    coverage: float,
    warnings: list[str],
    issues: list[str],
) -> float:
    score = 0.54 + coverage * 0.34
    if assignments:
        score += sum(item.score for item in assignments) / len(assignments) * 0.12
    score -= min(len(issues) * 0.18, 0.5)
    score -= min(len(warnings) * 0.035, 0.14)
    return round(max(0.0, min(score, 1.0)), 4)


def _longest_scene_run(assignments: list[BeatSkillAssignment]) -> int:
    longest = 0
    current = 0
    previous = ""
    for item in assignments:
        if item.scene_type == previous:
            current += 1
        else:
            previous = item.scene_type
            current = 1
        longest = max(longest, current)
    return longest


def _transition_intent(arc_role: str, scene_type: str, *, is_last: bool) -> str:
    if is_last:
        return "resolve_to_readable_final_hold"
    if arc_role == "hook":
        return "cold_open_snap_into_first_mechanism"
    if scene_type in {"matched_state_transform", "decision_branch"}:
        return "semantic_morph_preserving_subject_identity"
    if scene_type in {"guided_process", "architecture_flow"}:
        return "route_handoff_to_next_named_object"
    return "cinematic_cut_with_continuity_anchor"


def _grounded_frame(frame: dict[str, Any], source_text: str, *, allow_lists: bool = False) -> dict[str, Any]:
    source_key = _normalize(source_text)
    grounded: dict[str, Any] = {}
    for key, value in frame.items():
        if isinstance(value, list):
            values = [
                _label(item)
                for item in value
                if _label(item) and _value_grounded(_label(item), source_key)
            ]
            if values or allow_lists:
                grounded[key] = values
            continue
        cleaned = _label(value) if key != "viewer_takeaway" else _clean(value)
        if cleaned and (key == "viewer_takeaway" or _value_grounded(cleaned, source_key)):
            grounded[key] = cleaned
    return grounded


def _value_grounded(value: str, source_key: str) -> bool:
    tokens = [token for token in _words(value) if len(token) >= 3]
    if not tokens:
        return bool(value.strip())
    return any(token in source_key for token in tokens)


def _decision_parts(source_text: str) -> tuple[str, str, str] | None:
    cleaned = _clean(source_text)
    patterns = (
        r"\bif\s+(?P<condition>.+?),?\s+(?P<low>.+?);\s*(?:otherwise|else)\s+(?P<high>.+?)(?:[.!?]|$)",
        r"\bif\s+(?P<condition>.+?),?\s+(?P<low>.+?)\s+(?:otherwise|else)\s+(?P<high>.+?)(?:[.!?]|$)",
        r"\b(?P<condition>[^.!?]{3,90}?)\s+(?:routes?|branches?|chooses?)\s+(?P<low>[^.!?]{3,90}?)\s+(?:or|otherwise)\s+(?P<high>[^.!?]{3,90}?)(?:[.!?]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if not match:
            continue
        condition = _label(match.group("condition"))
        low = _label(match.group("low"))
        high = _label(match.group("high"))
        if condition and low and high:
            return condition, low, high
    return None


def _before_after_pair(source_text: str) -> tuple[str, str] | None:
    cleaned = _clean(source_text)
    patterns = (
        r"\bfrom\s+(?P<before>.+?)\s+to\s+(?P<after>.+?)(?:[,.;]|$)",
        r"\b(?:turns?|transforms?|converts?|compresses?)\s+(?P<before>.+?)\s+into\s+(?P<after>.+?)(?:\s+by\b|[,.;]|$)",
        r"\bbefore\s+(?P<before>.+?)\s+(?:after|then)\s+(?P<after>.+?)(?:[,.;]|$)",
        r"(?P<before>[^.!?:;]{3,90}?)\s+becomes?\s+(?P<after>.+?)(?:[,.;]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if not match:
            continue
        before = _label(match.group("before"))
        after = _label(match.group("after"))
        if before and after and _normalize(before) != _normalize(after):
            return before, after
    return None


def _step_labels(source_text: str, *, labels: list[str]) -> list[str]:
    explicit = [
        _label(item)
        for item in re.split(r"\b(?:then|next|finally|after that|and then)\b|[;:]", source_text, flags=re.IGNORECASE)
        if len(_words(item)) >= 2
    ]
    explicit = [item for item in explicit if item and not _too_generic(item)]
    if len(explicit) >= 2:
        return _unique(explicit, limit=5)
    if len(labels) >= 2:
        return labels[:5]
    return []


def _intervention_label(source_text: str, *, labels: list[str]) -> str:
    match = re.search(
        r"\b(?:by|using|with|after|enable|enables|apply|applies|retry|open|click)\s+(?P<label>[^.!?,;]{3,90})",
        source_text,
        flags=re.IGNORECASE,
    )
    if match:
        label = _label(match.group("label"))
        if label:
            return label
    return labels[1] if len(labels) > 1 else ""


def _contract_relation(contract: Any | None) -> str:
    if contract is None:
        return ""
    return _label(getattr(contract, "required_relation", "") or "")


def _continuity_subject(labels: list[str], *, source_text: str) -> str:
    for label in labels:
        if len(_words(label)) <= 4 and not _too_generic(label):
            return label
    candidates = keyword_candidates(source_text, limit=2)
    return " ".join(candidates[:2]) if candidates else ""


def _has_intervention(source_text: str) -> bool:
    return bool(re.search(r"\b(?:after|by|using|enable|enabled|apply|intervention|with)\b", source_text, flags=re.IGNORECASE))


def _causal_language(source_text: str) -> bool:
    return bool(re.search(r"\b(?:because|therefore|so|causes?|leads?|result|proof|means)\b", source_text, flags=re.IGNORECASE))


def _architecture_term_count(source_text: str) -> int:
    return len(set(re.findall(r"\b(?:api|service|gateway|planner|renderer|database|queue|worker|pipeline|request|response|runtime|client|server)\b", source_text.lower())))


def _ui_term_count(source_text: str) -> int:
    return len(set(re.findall(r"\b(?:ui|interface|screen|dashboard|editor|click|button|log|retry|panel|modal|control)\b", source_text.lower())))


def _term_hit(term: str, normalized: str) -> bool:
    token = _normalize(term)
    if not token:
        return False
    if token == "x":
        return bool(re.search(r"\b\d+(?:\.\d+)?x\b", normalized))
    if token == "%":
        return token in normalized
    return bool(re.search(rf"\b{re.escape(token)}\b", normalized))


def _has_content_metric(source_text: str) -> bool:
    return bool(
        re.search(
            r"\b\d+(?:\.\d+)?\s*(?:%|x|ms|tokens?|users?|gb|mb|billion|million)\b",
            source_text,
            flags=re.IGNORECASE,
        )
        or re.search(r"\b(?:latency|throughput|conversion|score|speedup)\b", source_text, flags=re.IGNORECASE)
    )


def _clause_labels(text: str, *, limit: int) -> list[str]:
    labels: list[str] = []
    for piece in re.split(r"(?<=[.!?])\s+|[,;:]", text):
        label = _label(piece)
        if label and not _too_generic(label):
            labels.append(label)
        if len(labels) >= limit:
            break
    return labels


def _short_sentence(text: str, *, max_words: int) -> str:
    sentence = re.split(r"(?<=[.!?])\s+", _clean(text))[0]
    words = sentence.split()
    if len(words) > max_words:
        sentence = " ".join(words[:max_words])
    return sentence.strip(" ,.;:") or "The mechanism becomes visible"


def _label(value: Any) -> str:
    cleaned = _clean(value)
    cleaned = re.sub(
        r"^(?:first|second|third|then|finally|now|once|the|a|an|and|so|because)\b[:,]?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\b(?:the|a|an)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.;:-")
    words = cleaned.split()
    if len(words) > 6:
        cleaned = " ".join(words[:6])
    return cleaned[:90].strip(" ,.;:-")


def _too_generic(value: str) -> bool:
    normalized = _normalize(value)
    if not normalized:
        return True
    generic = {
        "action",
        "after",
        "better",
        "clear",
        "concept",
        "context",
        "input",
        "output",
        "result",
        "simple",
        "signal",
        "system",
        "useful",
        "video",
        "workflow",
    }
    return normalized in generic


def _has_slot(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _unique(values: list[str], *, limit: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = _clean(value)
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
        if len(result) >= limit:
            break
    return result


def _safe_key(value: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "-", str(value or "").lower()).strip("-") or "subject"


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("\ufeff", "")).strip(" ,.;:-")


def _normalize(value: Any) -> str:
    return re.sub(r"[^a-z0-9%+./-]+", " ", str(value or "").lower()).strip()


def _words(value: Any) -> list[str]:
    return re.findall(r"[a-z0-9%+./-]+", str(value or "").lower())


__all__ = [
    "VIDEO_GENERATION_SKILL_GRAPH_VERSION",
    "BeatSceneSkill",
    "BeatSkillAssignment",
    "VideoProductionSkill",
    "VideoSkillGraph",
    "assignment_payload",
    "build_video_skill_graph",
    "skill_graph_prompt_block",
]
