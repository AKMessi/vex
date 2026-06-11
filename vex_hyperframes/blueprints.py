from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from visual_explanation import VisualExplanationIR


@dataclass(frozen=True)
class HyperframesBlueprint:
    blueprint_id: str
    scene_type: str
    stage_family: str
    required_roles: tuple[str, ...]
    minimum_objects: int
    layout_thesis: str
    motion_spine: str
    dynamic_devices: tuple[str, ...]
    anti_patterns: tuple[str, ...]
    selection_tags: tuple[str, ...] = ()
    priority: float = 0.8

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BlueprintSelection:
    passed: bool
    blueprint: HyperframesBlueprint | None
    score: float
    reasons: list[str] = field(default_factory=list)
    missing_roles: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "blueprint": self.blueprint.to_dict() if self.blueprint else None,
            "score": round(float(self.score), 3),
            "reasons": list(self.reasons),
            "missing_roles": list(self.missing_roles),
        }


def _blueprint(
    blueprint_id: str,
    scene_type: str,
    stage_family: str,
    required_roles: tuple[str, ...],
    minimum_objects: int,
    layout_thesis: str,
    motion_spine: str,
    dynamic_devices: tuple[str, ...],
    anti_patterns: tuple[str, ...],
    selection_tags: tuple[str, ...] = (),
    priority: float = 0.8,
) -> HyperframesBlueprint:
    return HyperframesBlueprint(
        blueprint_id=blueprint_id,
        scene_type=scene_type,
        stage_family=stage_family,
        required_roles=required_roles,
        minimum_objects=minimum_objects,
        layout_thesis=layout_thesis,
        motion_spine=motion_spine,
        dynamic_devices=dynamic_devices,
        anti_patterns=anti_patterns,
        selection_tags=selection_tags,
        priority=priority,
    )


CURATED_BLUEPRINTS: tuple[HyperframesBlueprint, ...] = (
    _blueprint(
        "metric_delta_bridge",
        "metric_delta",
        "semantic_metric",
        ("metric",),
        2,
        "Keep the measured value attached to registered before and after states.",
        "A measurement bridge transforms the old state into the new state.",
        ("matched_state_geometry", "tracked_metric", "measurement_ticks", "resolved_hold"),
        ("synthetic bars", "unlabeled percentages", "dashboard filler"),
        ("before", "after", "change"),
        0.96,
    ),
    _blueprint(
        "metric_delta_axis",
        "metric_delta",
        "semantic_metric",
        ("metric",),
        2,
        "Use one evidence axis with both source-backed states and a readable delta.",
        "The marker travels from the first measured state to the second.",
        ("evidence_axis", "moving_marker", "delta_annotation", "final_comparison"),
        ("random sparkline", "decorative chart", "invented scale"),
        ("latency", "time", "performance"),
        0.9,
    ),
    _blueprint(
        "metric_intervention_trace",
        "metric_intervention",
        "semantic_metric",
        ("metric", "intervention"),
        3,
        "Place the intervention on the causal path between two measured states.",
        "The active intervention changes the tracked measurement in view.",
        ("before_metric", "intervention_gate", "after_metric", "cause_trace"),
        ("detached metric cards", "unearned improvement", "fabricated threshold"),
        ("intervention", "enable", "after"),
        0.98,
    ),
    _blueprint(
        "metric_intervention_threshold",
        "metric_intervention",
        "semantic_metric",
        ("metric", "intervention"),
        3,
        "Show the measured signal crossing a source-backed threshold after intervention.",
        "A threshold line reacts only when the intervention becomes active.",
        ("threshold_line", "tracked_signal", "intervention_switch", "proof_lock"),
        ("invented threshold", "fake gauge", "unlabeled trend"),
        ("threshold", "limit", "target"),
        0.86,
    ),
    _blueprint(
        "metric_proof_spine",
        "metric_proof",
        "semantic_metric",
        ("metric",),
        1,
        "Tie the hero number to one concrete evidence structure instead of floating typography.",
        "Evidence assembles first; the metric locks only after proof is visible.",
        ("evidence_geometry", "tracked_value", "proof_marker", "final_lock"),
        ("giant isolated number", "fake dashboard", "random bars"),
        ("proof", "measured", "evidence"),
        0.94,
    ),
    _blueprint(
        "metric_proof_ladder",
        "metric_proof",
        "semantic_metric",
        ("metric",),
        1,
        "Accumulate source-backed proof in a vertical hierarchy that resolves into the metric.",
        "Each proof rung contributes to one final measured claim.",
        ("proof_rungs", "progressive_highlight", "metric_resolution"),
        ("equal-weight cards", "generic proof labels", "decorative ranking"),
        ("sequence", "proof", "accumulate"),
        0.84,
    ),
    _blueprint(
        "causal_mechanism_reveal",
        "causal_intervention",
        "semantic_causal",
        ("problem", "mechanism"),
        3,
        "Keep cause, mechanism, intervention, and result on one persistent causal spine.",
        "A signal propagates through the mechanism and visibly changes after intervention.",
        ("cause_source", "mechanism_chamber", "intervention_gate", "result_state"),
        ("cause/effect cards", "unrelated icons", "decorative arrows"),
        ("mechanism", "because", "intervention"),
        0.98,
    ),
    _blueprint(
        "causal_counterfactual_split",
        "causal_intervention",
        "semantic_causal",
        ("problem", "intervention", "result"),
        3,
        "Branch from one cause into untreated and intervened outcomes using matched geometry.",
        "The intervention redirects the same signal into a different observable state.",
        ("shared_input", "counterfactual_branch", "intervention_gate", "matched_outcomes"),
        ("before/after prose cards", "unsupported second outcome", "ambiguous branch"),
        ("without", "with", "instead"),
        0.9,
    ),
    _blueprint(
        "causal_direct_trace",
        "causal_intervention",
        "semantic_causal",
        ("problem", "result"),
        2,
        "Keep the grounded cause and effect on one trace when the source does not name a separate intervention.",
        "The source state emits a signal that resolves into the observed effect.",
        ("cause_source", "direct_trace", "result_state", "resolved_hold"),
        ("invented mechanism", "decorative causal cards", "unsupported intervention"),
        ("because", "leads", "causes", "therefore"),
        0.86,
    ),
    _blueprint(
        "guided_process_route",
        "guided_process",
        "semantic_route",
        ("mechanism",),
        2,
        "Arrange concrete process steps along one directional route with completed-state memory.",
        "A traveler activates each step and leaves a visible completion trail.",
        ("route_path", "traveler", "step_activation", "completion_trace"),
        ("disconnected cards", "simultaneous reveals", "generic numbered list"),
        ("then", "next", "process"),
        0.97,
    ),
    _blueprint(
        "guided_process_handoff",
        "guided_process",
        "semantic_route",
        ("mechanism", "result"),
        3,
        "Emphasize ownership transfer by changing the route treatment at the handoff.",
        "The active token changes owner while preserving its identity.",
        ("ownership_zones", "handoff_bridge", "identity_preserving_token", "result_lock"),
        ("new token after handoff", "ambiguous ownership", "static checklist"),
        ("handoff", "route", "human"),
        0.93,
    ),
    _blueprint(
        "architecture_service_lifecycle",
        "architecture_flow",
        "semantic_architecture",
        ("problem", "mechanism"),
        3,
        "Use service boundaries and one request token to explain the complete lifecycle.",
        "The request token crosses explicit service boundaries in source order.",
        ("service_nodes", "request_token", "boundary_crossings", "return_path"),
        ("network wallpaper", "unlabeled boxes", "bidirectional motion without meaning"),
        ("api", "service", "request"),
        0.98,
    ),
    _blueprint(
        "architecture_layered_pipeline",
        "architecture_flow",
        "semantic_architecture",
        ("mechanism",),
        3,
        "Expose system layers while retaining a single end-to-end route.",
        "A vertical depth shift reveals responsibility at each layer.",
        ("layer_planes", "active_route", "ownership_labels", "response_trace"),
        ("isometric decoration", "fake infrastructure", "too many nodes"),
        ("layer", "pipeline", "stage"),
        0.88,
    ),
    _blueprint(
        "matched_state_morph",
        "matched_state_transform",
        "semantic_transform",
        ("problem", "result"),
        2,
        "Register equivalent objects spatially so the transformation itself explains the difference.",
        "Persistent objects morph while preserved constraints remain fixed.",
        ("matched_geometry", "state_morph", "constraint_anchor", "difference_highlight"),
        ("static versus cards", "unmatched layouts", "full sentence duplication"),
        ("old", "new", "before", "after"),
        0.98,
    ),
    _blueprint(
        "matched_state_constraint",
        "matched_state_transform",
        "semantic_transform",
        ("problem", "result", "constraint"),
        3,
        "Make the preserved constraint a fixed anchor while all changed elements transform around it.",
        "The unchanged anchor proves what the new workflow still guarantees.",
        ("constraint_anchor", "before_cluster", "after_cluster", "change_trace"),
        ("hidden invariant", "generic improvement glow", "unexplained replacement"),
        ("keep", "preserve", "constraint", "validation"),
        0.99,
    ),
    _blueprint(
        "interface_state_trace",
        "grounded_interface_walkthrough",
        "semantic_interface",
        ("intervention", "result"),
        2,
        "Show only named interface states and connect the user action to its result.",
        "A focus beam and cursor trace move through the source-backed UI states.",
        ("interface_surface", "focus_region", "action_trace", "result_feedback"),
        ("invented controls", "fake percentages", "generic dashboard rows"),
        ("editor", "screen", "interface", "log"),
        0.99,
    ),
    _blueprint(
        "interface_action_result",
        "grounded_interface_walkthrough",
        "semantic_interface",
        ("intervention", "result"),
        2,
        "Keep the action target and resulting state visible in one stable interface composition.",
        "The interaction creates a visible feedback path from control to result.",
        ("action_target", "interaction_pulse", "feedback_trace", "result_state"),
        ("carousel of screens", "unrelated UI chrome", "imaginary metrics"),
        ("click", "open", "retry", "focus"),
        0.94,
    ),
    _blueprint(
        "decision_quality_gate",
        "decision_branch",
        "semantic_decision",
        ("decision", "branch_low", "branch_high"),
        3,
        "Center the quality gate and make both source-backed outcomes inspectable.",
        "One signal reaches the gate, then only the selected branch activates.",
        ("decision_gate", "low_branch", "high_branch", "active_route"),
        ("simultaneous active branches", "generic yes/no", "unlabeled condition"),
        ("confidence", "review", "otherwise"),
        0.99,
    ),
    _blueprint(
        "decision_guardrail_route",
        "decision_branch",
        "semantic_decision",
        ("decision", "branch_low", "branch_high"),
        3,
        "Treat the decision as a guardrail that protects a named downstream constraint.",
        "The unsafe branch stops visibly while the safe branch continues.",
        ("guardrail", "stop_state", "continue_state", "constraint_badge"),
        ("traffic-light decoration", "missing condition", "ambiguous safe path"),
        ("protect", "prevent", "gate"),
        0.92,
    ),
    _blueprint(
        "narrative_recovery_arc",
        "narrative_progression",
        "semantic_narrative",
        ("setup", "intervention", "result"),
        3,
        "Use persistent story objects across failure, diagnosis, and recovery.",
        "The same system travels through a visible turning point into recovery.",
        ("setup_state", "turning_point", "persistent_subject", "payoff_state"),
        ("generic filmstrip", "new subject per beat", "decorative timeline"),
        ("failed", "traced", "recovered", "launch"),
        0.98,
    ),
    _blueprint(
        "narrative_turning_point",
        "narrative_progression",
        "semantic_narrative",
        ("setup", "intervention", "result"),
        3,
        "Hold the setup and payoff at opposite ends of an arc dominated by one decisive turn.",
        "The turning point changes trajectory rather than merely adding another card.",
        ("story_arc", "turn_marker", "trajectory_change", "payoff_lock"),
        ("equal beats", "static chronology", "generic setup/payoff labels"),
        ("turn", "bottleneck", "diagnosis"),
        0.9,
    ),
    _blueprint(
        "quote_phrase_assembly",
        "evidence_backed_quote",
        "semantic_quote",
        ("quote",),
        1,
        "Assemble exact source language into one readable phrase with one decisive emphasis.",
        "Phrase segments converge and lock on the retained idea.",
        ("exact_quote", "phrase_segments", "emphasis_lock", "resolved_hold"),
        ("paraphrased quote", "decorative keywords", "unrelated attribution"),
        ("quote", "exact", "phrase"),
        0.96,
    ),
    _blueprint(
        "quote_semantic_breakdown",
        "evidence_backed_quote",
        "semantic_quote",
        ("quote",),
        1,
        "Preserve the exact quote while briefly isolating its source-backed semantic clauses.",
        "The quote stays intact as clauses receive sequential emphasis.",
        ("quote_line", "clause_focus", "relationship_marker", "full_quote_return"),
        ("word cloud", "fabricated keywords", "permanent fragmentation"),
        ("path", "prototype", "clarity"),
        0.9,
    ),
)


def rank_blueprints(
    ir: VisualExplanationIR,
    spec: dict[str, Any] | None = None,
    *,
    limit: int | None = None,
) -> list[BlueprintSelection]:
    if ir.render_policy != "render":
        return []
    spec = dict(spec or {})
    roles = {item.role for item in ir.objects}
    source = " ".join(
        [
            ir.thesis,
            ir.takeaway,
            " ".join(ir.required_labels),
            str(spec.get("sentence_text") or ""),
            str(spec.get("context_text") or ""),
        ]
    ).lower()
    candidates: list[tuple[float, HyperframesBlueprint, list[str]]] = []
    for blueprint in CURATED_BLUEPRINTS:
        if blueprint.scene_type != ir.scene_type:
            continue
        missing = [role for role in blueprint.required_roles if role not in roles]
        if missing or len(ir.objects) < blueprint.minimum_objects:
            continue
        tag_hits = sum(1 for tag in blueprint.selection_tags if tag in source)
        role_coverage = len(blueprint.required_roles) / max(len(roles), 1)
        score = blueprint.priority + min(tag_hits * 0.035, 0.14) + min(role_coverage * 0.08, 0.08)
        candidates.append((score, blueprint, []))
    ranked = sorted(candidates, key=lambda item: (item[0], item[1].blueprint_id), reverse=True)
    selections = [
        BlueprintSelection(
            passed=True,
            blueprint=blueprint,
            score=min(score, 1.0),
            reasons=[
                *reasons,
                f"ranked_for_scene_type:{ir.scene_type}",
                f"blueprint_rank:{index + 1}",
            ],
        )
        for index, (score, blueprint, reasons) in enumerate(ranked)
    ]
    if limit is None:
        return selections
    return selections[: max(0, int(limit))]


def select_blueprint(ir: VisualExplanationIR, spec: dict[str, Any] | None = None) -> BlueprintSelection:
    ranked = rank_blueprints(ir, spec, limit=1)
    if ranked:
        selection = ranked[0]
        return BlueprintSelection(
            passed=True,
            blueprint=selection.blueprint,
            score=selection.score,
            reasons=[
                reason
                for reason in selection.reasons
                if not reason.startswith("ranked_for_scene_type:")
            ]
            + [f"selected_for_scene_type:{ir.scene_type}"],
        )

    roles = {item.role for item in ir.objects}
    expected = [
        blueprint
        for blueprint in CURATED_BLUEPRINTS
        if blueprint.scene_type == ir.scene_type
    ]
    missing_roles = sorted(
        {
            role
            for blueprint in expected
            for role in blueprint.required_roles
            if role not in roles
        }
    )
    if ir.render_policy != "render":
        return BlueprintSelection(
            passed=False,
            blueprint=None,
            score=0.0,
            reasons=["visual_explanation_ir_rejected"],
        )
    return BlueprintSelection(
        passed=False,
        blueprint=None,
        score=0.0,
        reasons=["no_blueprint_satisfies_grounded_role_prerequisites"],
        missing_roles=missing_roles,
    )


__all__ = [
    "BlueprintSelection",
    "CURATED_BLUEPRINTS",
    "HyperframesBlueprint",
    "rank_blueprints",
    "select_blueprint",
]
