from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any

from vex_hyperframes.compiler import CompiledHyperframesPlan, compile_hyperframes_plan
from vex_hyperframes.skill_pack import retrieve_skill_slices


AUTO_VISUAL_SKILL_GRAPH_VERSION = "auto-visual-skill-graph-v1"


@dataclass(frozen=True)
class AutoVisualSkill:
    skill_id: str
    title: str
    scene_types: tuple[str, ...]
    preferred_templates: tuple[str, ...]
    renderer_hint: str
    composition_mode: str
    required_slots: tuple[str, ...]
    optional_slots: tuple[str, ...]
    blueprint_tags: tuple[str, ...]
    proof_encodings: tuple[str, ...]
    visual_world_mediums: tuple[str, ...]
    qa_floor: float
    reject_rules: tuple[str, ...]
    anti_patterns: tuple[str, ...]
    priority: float = 0.8

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SkillRoutingDecision:
    version: str
    card_id: str
    passed: bool
    skill_id: str
    skill_title: str
    scene_type: str
    score: float
    renderer_hint: str
    composition_mode: str
    preferred_template: str
    allowed_templates: list[str]
    required_slots: list[str]
    optional_slots: list[str]
    slot_values: dict[str, Any]
    skill_slices: list[dict[str, Any]]
    blueprint_priors: list[dict[str, Any]]
    proof_encodings: list[str]
    visual_world_mediums: list[str]
    qa_floor: float
    plan_seed: dict[str, Any]
    preflight: dict[str, Any]
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    reject_rules: list[str] = field(default_factory=list)
    anti_patterns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_CORE_REQUIRED_SLOTS = (
    "card_id",
    "evidence_text",
    "scene_type",
    "headline",
    "required_labels",
    "required_objects",
    "required_relations",
)


AUTO_VISUAL_SKILLS: tuple[AutoVisualSkill, ...] = (
    AutoVisualSkill(
        skill_id="metric-story",
        title="Metric Story And Measured Proof",
        scene_types=("metric_delta", "metric_intervention", "metric_proof"),
        preferred_templates=("data_journey", "proof_sequence", "data_pulse"),
        renderer_hint="hyperframes",
        composition_mode="replace",
        required_slots=(*_CORE_REQUIRED_SLOTS, "metric_facts"),
        optional_slots=("intervention", "before_state", "after_state", "threshold"),
        blueprint_tags=("metric", "axis", "proof", "threshold"),
        proof_encodings=("linear_trace", "split_register", "layered_flow"),
        visual_world_mediums=("data_sculpture", "diagrammatic_technical_system"),
        qa_floor=0.78,
        reject_rules=(
            "Reject if any displayed number is absent from source evidence.",
            "Reject if the metric floats without a visible evidence structure.",
        ),
        anti_patterns=("fake gauges", "random bars", "unlabeled chart scales"),
        priority=0.96,
    ),
    AutoVisualSkill(
        skill_id="causal-spine",
        title="Causal Mechanism Spine",
        scene_types=("causal_intervention",),
        preferred_templates=("causal_chain", "mechanism_blueprint", "problem_solution"),
        renderer_hint="hyperframes",
        composition_mode="replace",
        required_slots=(*_CORE_REQUIRED_SLOTS, "cause", "effect"),
        optional_slots=("intervention", "counterfactual", "mechanism"),
        blueprint_tags=("cause", "mechanism", "intervention", "counterfactual"),
        proof_encodings=("layered_flow", "focal_gate", "linear_trace"),
        visual_world_mediums=("spatial_metaphor", "diagrammatic_technical_system"),
        qa_floor=0.8,
        reject_rules=(
            "Reject if the mechanism is skipped and the scene becomes static cause/effect cards.",
            "Reject if an unsupported second outcome is invented.",
        ),
        anti_patterns=("decorative arrows", "unrelated icons", "static cause/effect cards"),
        priority=0.95,
    ),
    AutoVisualSkill(
        skill_id="route-choreography",
        title="Process Route Choreography",
        scene_types=("guided_process",),
        preferred_templates=("signal_network", "kinetic_route", "pipeline_xray"),
        renderer_hint="hyperframes",
        composition_mode="replace",
        required_slots=(*_CORE_REQUIRED_SLOTS, "steps"),
        optional_slots=("traveler", "handoff", "completion_state"),
        blueprint_tags=("route", "handoff", "process", "pipeline"),
        proof_encodings=("linear_trace", "layered_flow", "radial_evidence"),
        visual_world_mediums=("diagrammatic_technical_system", "spatial_metaphor"),
        qa_floor=0.76,
        reject_rules=(
            "Reject if steps appear simultaneously without sequence.",
            "Reject if the moving token loses identity between handoffs.",
        ),
        anti_patterns=("disconnected numbered cards", "simultaneous reveals"),
        priority=0.9,
    ),
    AutoVisualSkill(
        skill_id="architecture-flow",
        title="Architecture And Service Lifecycle",
        scene_types=("architecture_flow",),
        preferred_templates=("pipeline_xray", "signal_network", "mechanism_blueprint"),
        renderer_hint="hyperframes",
        composition_mode="replace",
        required_slots=(*_CORE_REQUIRED_SLOTS, "service_boundaries", "route_token"),
        optional_slots=("return_path", "ownership_labels"),
        blueprint_tags=("api", "service", "pipeline", "layer"),
        proof_encodings=("linear_trace", "split_register", "layered_flow"),
        visual_world_mediums=("diagrammatic_technical_system", "product_interface"),
        qa_floor=0.8,
        reject_rules=(
            "Reject if infrastructure not present in evidence is added.",
            "Reject if request and response direction are ambiguous.",
        ),
        anti_patterns=("network wallpaper", "fake infrastructure", "unlabeled boxes"),
        priority=0.93,
    ),
    AutoVisualSkill(
        skill_id="matched-transform",
        title="Matched State Transform",
        scene_types=("matched_state_transform",),
        preferred_templates=("spotlight_compare", "contrast_ladder", "problem_solution"),
        renderer_hint="hyperframes",
        composition_mode="replace",
        required_slots=(*_CORE_REQUIRED_SLOTS, "before_state", "after_state"),
        optional_slots=("preserved_constraint", "difference_highlight"),
        blueprint_tags=("before", "after", "constraint", "morph"),
        proof_encodings=("split_register", "focal_gate", "linear_trace"),
        visual_world_mediums=("spatial_metaphor", "editorial_collage"),
        qa_floor=0.77,
        reject_rules=(
            "Reject if before and after states are not source backed.",
            "Reject if the invariant named by the source disappears.",
        ),
        anti_patterns=("static versus cards", "unmatched layouts", "generic improvement glow"),
        priority=0.92,
    ),
    AutoVisualSkill(
        skill_id="grounded-interface",
        title="Grounded Interface Walkthrough",
        scene_types=("grounded_interface_walkthrough",),
        preferred_templates=("interface_cascade", "mechanism_blueprint"),
        renderer_hint="hyperframes",
        composition_mode="replace",
        required_slots=(*_CORE_REQUIRED_SLOTS, "screen", "action", "result"),
        optional_slots=("source_asset_grounding", "focus_region", "feedback_path"),
        blueprint_tags=("interface", "screen", "action", "result"),
        proof_encodings=("focal_gate", "linear_trace", "split_register"),
        visual_world_mediums=("product_interface", "grounded_source_media_composite"),
        qa_floor=0.82,
        reject_rules=(
            "Reject if imaginary UI controls, states, metrics, or notifications are introduced.",
            "Prefer a source frame when the source recording contains the named interface state.",
        ),
        anti_patterns=("fake dashboards", "generic UI rows", "carousel of unrelated screens"),
        priority=0.98,
    ),
    AutoVisualSkill(
        skill_id="decision-gate",
        title="Decision Gate And Guardrail",
        scene_types=("decision_branch",),
        preferred_templates=("decision_tree", "decision_matrix"),
        renderer_hint="hyperframes",
        composition_mode="replace",
        required_slots=(*_CORE_REQUIRED_SLOTS, "decision", "branch_low", "branch_high"),
        optional_slots=("constraint", "selected_route"),
        blueprint_tags=("decision", "branch", "gate", "guardrail"),
        proof_encodings=("focal_gate", "split_register", "linear_trace"),
        visual_world_mediums=("diagrammatic_technical_system", "spatial_metaphor"),
        qa_floor=0.8,
        reject_rules=(
            "Reject if both branches appear active at the same beat.",
            "Reject if the condition becomes generic yes/no copy.",
        ),
        anti_patterns=("traffic-light decoration", "generic yes/no branches"),
        priority=0.96,
    ),
    AutoVisualSkill(
        skill_id="narrative-continuity",
        title="Narrative Progression And Recovery",
        scene_types=("narrative_progression",),
        preferred_templates=("narrative_arc", "timeline_filmstrip"),
        renderer_hint="hyperframes",
        composition_mode="replace",
        required_slots=(*_CORE_REQUIRED_SLOTS, "setup", "turn", "payoff"),
        optional_slots=("persistent_subject", "trajectory_change"),
        blueprint_tags=("setup", "turn", "payoff", "recovery"),
        proof_encodings=("linear_trace", "layered_flow", "radial_evidence"),
        visual_world_mediums=("editorial_collage", "spatial_metaphor"),
        qa_floor=0.74,
        reject_rules=(
            "Reject if each beat introduces an unrelated subject.",
            "Reject if the turning point does not visibly alter trajectory.",
        ),
        anti_patterns=("generic filmstrip", "equal-weight beats", "reset spatial frame"),
        priority=0.84,
    ),
    AutoVisualSkill(
        skill_id="partition-compression",
        title="Token Partition And Compression",
        scene_types=("set_partition",),
        preferred_templates=("data_journey", "mechanism_blueprint"),
        renderer_hint="hyperframes",
        composition_mode="replace",
        required_slots=(*_CORE_REQUIRED_SLOTS, "input_count", "group_size", "group_count"),
        optional_slots=("membership_rule", "compression_result"),
        blueprint_tags=("compression", "tokens", "blocks"),
        proof_encodings=("split_register", "linear_trace", "layered_flow"),
        visual_world_mediums=("data_sculpture", "diagrammatic_technical_system"),
        qa_floor=0.82,
        reject_rules=(
            "Reject if arithmetic cannot be proven from source evidence.",
            "Reject if token membership is hidden behind a generic metric card.",
        ),
        anti_patterns=("floating numbers", "generic before/after cards"),
        priority=0.98,
    ),
    AutoVisualSkill(
        skill_id="exact-quote",
        title="Evidence-Backed Quote Direction",
        scene_types=("evidence_backed_quote",),
        preferred_templates=("quote_breakdown", "ribbon_quote"),
        renderer_hint="hyperframes",
        composition_mode="replace",
        required_slots=(*_CORE_REQUIRED_SLOTS, "exact_quote"),
        optional_slots=("emphasis_phrase", "source_context"),
        blueprint_tags=("quote", "phrase", "exact"),
        proof_encodings=("linear_trace", "focal_gate"),
        visual_world_mediums=("kinetic_typography", "editorial_collage"),
        qa_floor=0.72,
        reject_rules=(
            "Reject if quoted language is paraphrased.",
            "Reject if the quote is used as filler for a non-memorable line.",
        ),
        anti_patterns=("word clouds", "fabricated keywords", "over-animated prose"),
        priority=0.7,
    ),
)


_SKILL_BY_SCENE = {
    scene_type: skill
    for skill in AUTO_VISUAL_SKILLS
    for scene_type in skill.scene_types
}


def route_visual_skill(
    card: dict[str, Any],
    *,
    available_renderers: list[dict[str, Any]] | None = None,
    prefer_premium: bool = False,
    force_fullscreen: bool = False,
) -> SkillRoutingDecision:
    card_id = _clean(card.get("card_id")) or "visual"
    compiled, compile_error = _compile_card(card)
    preflight = _preflight_payload(card, compiled=compiled, compile_error=compile_error)
    scene_type = str(preflight.get("scene_type") or "")
    skill = _SKILL_BY_SCENE.get(scene_type)
    warnings: list[str] = []
    reasons: list[str] = []
    if compile_error:
        warnings.append(f"semantic_preflight_error:{compile_error}")
    if not scene_type:
        warnings.append("missing_scene_type")
    if not bool(preflight.get("passed")):
        reasons.append("semantic_preflight_rejected")
    if skill is None:
        reasons.append("no_skill_for_scene_type")
        return _rejected_decision(
            card_id=card_id,
            scene_type=scene_type,
            preflight=preflight,
            reasons=reasons,
            warnings=warnings,
        )

    slot_values = _slot_values_for(card, compiled=compiled, scene_type=scene_type)
    missing_slots = _missing_required_slots(skill, slot_values)
    if missing_slots:
        reasons.extend(f"missing_slot:{slot}" for slot in missing_slots)
    renderer_hint = _renderer_hint_for(skill, available_renderers)
    if renderer_hint != skill.renderer_hint:
        warnings.append(f"preferred_renderer_unavailable:{skill.renderer_hint}")
    composition_mode = "replace" if (prefer_premium or force_fullscreen) else skill.composition_mode
    template = _preferred_template_for(skill, card, slot_values)
    score = _score_skill_route(card, skill, preflight=preflight, missing_slots=missing_slots)
    slices = retrieve_skill_slices(
        f"semantic_{_semantic_family(scene_type)}",
        scene_type=scene_type,
        blueprint_id=str(preflight.get("selected_blueprint_id") or ""),
        limit=2,
    )
    plan_seed = _plan_seed(
        card,
        skill=skill,
        slot_values=slot_values,
        template=template,
        renderer_hint=renderer_hint,
        composition_mode=composition_mode,
    )
    passed = bool(preflight.get("passed")) and not missing_slots and score >= 0.55
    if passed:
        reasons.append("skill_route_passed")
    return SkillRoutingDecision(
        version=AUTO_VISUAL_SKILL_GRAPH_VERSION,
        card_id=card_id,
        passed=passed,
        skill_id=skill.skill_id,
        skill_title=skill.title,
        scene_type=scene_type,
        score=round(score, 4),
        renderer_hint=renderer_hint,
        composition_mode=composition_mode,
        preferred_template=template,
        allowed_templates=list(skill.preferred_templates),
        required_slots=list(skill.required_slots),
        optional_slots=list(skill.optional_slots),
        slot_values=slot_values,
        skill_slices=[item.to_dict() for item in slices],
        blueprint_priors=list(preflight.get("blueprint_priors") or []),
        proof_encodings=list(skill.proof_encodings),
        visual_world_mediums=list(skill.visual_world_mediums),
        qa_floor=skill.qa_floor,
        plan_seed=plan_seed,
        preflight=preflight,
        reasons=reasons,
        warnings=warnings,
        reject_rules=list(skill.reject_rules),
        anti_patterns=list(skill.anti_patterns),
    )


def apply_visual_skill_graph(
    cards: list[dict[str, Any]],
    *,
    available_renderers: list[dict[str, Any]] | None = None,
    prefer_premium: bool = False,
    force_fullscreen: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for card in cards:
        decision = route_visual_skill(
            dict(card),
            available_renderers=available_renderers,
            prefer_premium=prefer_premium,
            force_fullscreen=force_fullscreen,
        )
        payload = decision.to_dict()
        normalized = dict(card)
        normalized["auto_visual_skill"] = payload
        if decision.passed:
            normalized["suggested_renderer"] = decision.renderer_hint
            normalized["suggested_composition"] = decision.composition_mode
            normalized["skill_template"] = decision.preferred_template
            normalized["skill_plan_seed"] = dict(decision.plan_seed)
        else:
            rejected.append(payload)
        enriched.append(normalized)
        decisions.append(payload)
    accepted = [item for item in decisions if bool(item.get("passed"))]
    return enriched, {
        "version": AUTO_VISUAL_SKILL_GRAPH_VERSION,
        "input_count": len(cards),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "average_score": round(
            sum(float(item.get("score") or 0.0) for item in decisions)
            / max(len(decisions), 1),
            4,
        ),
        "decisions": decisions,
        "rejected": rejected[:20],
        "skill_counts": _skill_counts(accepted),
        "prompt_block": skill_graph_prompt_block({"decisions": decisions}),
    }


def skill_seed_plan(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seeds: list[dict[str, Any]] = []
    for card in cards:
        skill = dict(card.get("auto_visual_skill") or {})
        seed = dict(skill.get("plan_seed") or card.get("skill_plan_seed") or {})
        if seed and bool(skill.get("passed", True)):
            seeds.append(seed)
    return seeds


def skill_graph_prompt_block(report: dict[str, Any]) -> str:
    decisions = [
        dict(item)
        for item in report.get("decisions") or []
        if isinstance(item, dict)
    ]
    lines = [
        "Auto Visuals Skill Graph:",
        "- Treat the selected skill as the visual architecture. Do not invent another template family.",
        "- Fill missing copy slots only from the card evidence. Do not invent metrics, UI states, entities, branches, or outcomes.",
        "- If the skill reject rules cannot be satisfied, return no visual for that card.",
    ]
    for item in decisions[:24]:
        if not bool(item.get("passed")):
            continue
        slots = dict(item.get("slot_values") or {})
        labels = "; ".join(str(value) for value in slots.get("required_labels") or [] if str(value))[:220]
        lines.append(
            (
                f"- {item.get('card_id')}: skill={item.get('skill_id')} "
                f"scene={item.get('scene_type')} template={item.get('preferred_template')} "
                f"renderer={item.get('renderer_hint')} mode={item.get('composition_mode')} "
                f"required_labels={labels}"
            )
        )
    return "\n".join(lines)


def _rejected_decision(
    *,
    card_id: str,
    scene_type: str,
    preflight: dict[str, Any],
    reasons: list[str],
    warnings: list[str],
) -> SkillRoutingDecision:
    return SkillRoutingDecision(
        version=AUTO_VISUAL_SKILL_GRAPH_VERSION,
        card_id=card_id,
        passed=False,
        skill_id="",
        skill_title="",
        scene_type=scene_type,
        score=0.0,
        renderer_hint="auto",
        composition_mode="replace",
        preferred_template="",
        allowed_templates=[],
        required_slots=[],
        optional_slots=[],
        slot_values={},
        skill_slices=[],
        blueprint_priors=[],
        proof_encodings=[],
        visual_world_mediums=[],
        qa_floor=0.0,
        plan_seed={},
        preflight=preflight,
        reasons=reasons,
        warnings=warnings,
        reject_rules=[],
        anti_patterns=[],
    )


def _compile_card(card: dict[str, Any]) -> tuple[CompiledHyperframesPlan | None, str]:
    try:
        return compile_hyperframes_plan(_preflight_spec(card)), ""
    except Exception as exc:  # noqa: BLE001
        return None, _clean(str(exc))[:240]


def _preflight_spec(card: dict[str, Any]) -> dict[str, Any]:
    start = _as_float(card.get("start"), 0.0)
    end = _as_float(card.get("end"), start + 3.0)
    return {
        "visual_id": _clean(card.get("card_id")) or "visual",
        "card_id": _clean(card.get("card_id")) or "visual",
        "sentence_text": _clean(card.get("sentence_text") or card.get("source_sentence_text")),
        "context_text": _clean(card.get("planning_context_text") or card.get("context_text")),
        "semantic_frame": dict(card.get("semantic_frame") or {}),
        "metric_facts": list(card.get("metric_facts") or []),
        "visual_type_hint": _clean(card.get("visual_type_hint")),
        "duration": max(1.0, min(end - start if end > start else 3.0, 8.0)),
        "composition_mode": "replace",
        "required_labels": list(card.get("required_labels") or []),
    }


def _preflight_payload(
    card: dict[str, Any],
    *,
    compiled: CompiledHyperframesPlan | None,
    compile_error: str,
) -> dict[str, Any]:
    existing = dict(card.get("opportunity_preflight") or {})
    if compiled is None:
        return {
            **existing,
            "passed": False,
            "scene_type": str(existing.get("scene_type") or ""),
            "issues": [compile_error] if compile_error else list(existing.get("issues") or []),
        }
    blueprint_priors = []
    ranked = getattr(compiled.proof_tournament, "ranked_blueprints", None)
    if isinstance(ranked, list):
        blueprint_priors = [dict(item) for item in ranked if isinstance(item, dict)]
    if not blueprint_priors and compiled.blueprint_selection.blueprint is not None:
        blueprint_priors = [
            {
                "blueprint_id": compiled.blueprint_selection.blueprint.blueprint_id,
                "stage_family": compiled.blueprint_selection.blueprint.stage_family,
                "score": compiled.blueprint_selection.score,
            }
        ]
    contract = compiled.production_contract
    return {
        **existing,
        "passed": bool(compiled.passed),
        "raw_preflight_passed": bool(compiled.passed),
        "strict_preflight_passed": bool(compiled.passed),
        "scene_type": compiled.ir.scene_type,
        "render_policy": compiled.ir.render_policy,
        "issues": list(compiled.issues),
        "rejection_reasons": list(compiled.ir.rejection_reasons),
        "semantic_signature": contract.semantic_signature if contract else "",
        "required_labels": list(contract.required_labels) if contract else list(compiled.ir.required_labels),
        "required_relation_ids": list(contract.required_relation_ids) if contract else [],
        "selected_blueprint_id": (
            compiled.blueprint_selection.blueprint.blueprint_id
            if compiled.blueprint_selection.blueprint
            else ""
        ),
        "selected_stage_family": (
            compiled.blueprint_selection.blueprint.stage_family
            if compiled.blueprint_selection.blueprint
            else ""
        ),
        "claim_graph_signature": compiled.claim_graph.graph_signature,
        "proof_tournament_signature": compiled.proof_tournament.tournament_signature,
        "proof_candidate_count": len(compiled.proof_tournament.programs),
        "blueprint_priors": blueprint_priors,
    }


def _slot_values_for(
    card: dict[str, Any],
    *,
    compiled: CompiledHyperframesPlan | None,
    scene_type: str,
) -> dict[str, Any]:
    semantic = dict(card.get("semantic_frame") or {})
    source_text = _clean(
        " ".join(
            str(card.get(key) or "")
            for key in ("sentence_text", "context_text", "planning_context_text")
        )
    )
    ir = compiled.ir if compiled is not None else None
    contract = compiled.production_contract if compiled is not None else None
    labels = (
        list(contract.required_labels)
        if contract is not None
        else _string_list((card.get("opportunity_preflight") or {}).get("required_labels"))
    )
    if not labels and ir is not None:
        labels = list(ir.required_labels)
    objects = [item.to_dict() for item in ir.objects] if ir is not None else []
    relations = [item.to_dict() for item in ir.relations] if ir is not None else []
    facts = [item.to_dict() for item in ir.facts] if ir is not None else list(card.get("metric_facts") or [])
    steps = _semantic_list(semantic.get("steps"))
    if not steps:
        steps = [str(item.get("label") or "") for item in objects if str(item.get("label") or "")]
    metric_facts = list(card.get("metric_facts") or [])
    executable_model = dict((ir.metadata if ir is not None else {}).get("executable_model") or {})
    values: dict[str, Any] = {
        "card_id": _clean(card.get("card_id")),
        "evidence_text": source_text,
        "scene_type": scene_type,
        "headline": _headline_for(card, ir=ir, labels=labels),
        "deck": _deck_for(card, ir=ir, labels=labels),
        "required_labels": _unique(labels, limit=10),
        "required_objects": objects,
        "required_relations": relations,
        "grounded_facts": facts,
        "steps": _unique(steps, limit=5),
        "metric_facts": metric_facts,
        "before_state": _first_present(semantic, "before_state", "problem", "setup"),
        "after_state": _first_present(semantic, "after_state", "result", "payoff", "viewer_takeaway"),
        "intervention": _first_present(semantic, "intervention", "action", "turn"),
        "cause": _first_present(semantic, "cause", "problem", "mechanism"),
        "effect": _first_present(semantic, "effect", "result", "viewer_takeaway"),
        "mechanism": _first_present(semantic, "mechanism", "mental_model"),
        "screen": _first_present(semantic, "screen", "interface", "focus"),
        "action": _first_present(semantic, "action", "intervention"),
        "result": _first_present(semantic, "result", "viewer_takeaway", "after_state"),
        "decision": _first_present(semantic, "decision", "condition"),
        "branch_low": _first_present(semantic, "low_branch", "branch_low"),
        "branch_high": _first_present(semantic, "high_branch", "branch_high"),
        "constraint": _first_present(semantic, "constraint", "preserved_constraint"),
        "setup": _first_present(semantic, "setup", "before_state"),
        "turn": _first_present(semantic, "turn", "intervention"),
        "payoff": _first_present(semantic, "payoff", "result", "after_state"),
        "exact_quote": _first_present(semantic, "exact_quote") or _clean(card.get("sentence_text")),
        "service_boundaries": _unique(steps, limit=5),
        "route_token": _first_present(semantic, "input") or (labels[0] if labels else ""),
        "input_count": executable_model.get("input_count"),
        "group_size": executable_model.get("group_size"),
        "group_count": executable_model.get("group_count"),
    }
    return {key: value for key, value in values.items() if _has_value(value)}


def _missing_required_slots(skill: AutoVisualSkill, slots: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for slot in skill.required_slots:
        if slot in {"required_relations"} and slots.get("scene_type") == "evidence_backed_quote":
            continue
        if not _has_value(slots.get(slot)):
            missing.append(slot)
    return missing


def _renderer_hint_for(
    skill: AutoVisualSkill,
    available_renderers: list[dict[str, Any]] | None,
) -> str:
    available = {
        str(item.get("name") or "").strip().lower()
        for item in (available_renderers or [])
        if bool(item.get("available"))
    }
    if not available or skill.renderer_hint in available:
        return skill.renderer_hint
    if "hyperframes" in available:
        return "hyperframes"
    if "ffmpeg" in available and skill.skill_id == "exact-quote":
        return "ffmpeg"
    return "auto"


def _preferred_template_for(
    skill: AutoVisualSkill,
    card: dict[str, Any],
    slots: dict[str, Any],
) -> str:
    text = str(slots.get("evidence_text") or "").lower()
    if skill.skill_id == "metric-story":
        if re.search(r"\b(?:pulse|spike|threshold|live|feedback)\b", text):
            return "data_pulse"
        if len(slots.get("metric_facts") or []) >= 2:
            return "proof_sequence"
    if skill.skill_id == "route-choreography":
        if re.search(r"\b(?:pipeline|stack|layer|hidden|inside)\b", text):
            return "pipeline_xray"
        if len(slots.get("steps") or []) <= 3:
            return "kinetic_route"
    if skill.skill_id == "architecture-flow":
        if re.search(r"\b(?:layer|pipeline|stage)\b", text):
            return "pipeline_xray"
    if skill.skill_id == "matched-transform":
        if slots.get("constraint"):
            return "contrast_ladder"
    if skill.skill_id == "exact-quote" and len(_words(slots.get("exact_quote"))) <= 9:
        return "ribbon_quote"
    return skill.preferred_templates[0]


def _plan_seed(
    card: dict[str, Any],
    *,
    skill: AutoVisualSkill,
    slot_values: dict[str, Any],
    template: str,
    renderer_hint: str,
    composition_mode: str,
) -> dict[str, Any]:
    headline = _copy(slot_values.get("headline"), max_words=6, max_chars=42)
    deck = _copy(slot_values.get("deck"), max_words=9, max_chars=58)
    steps = [_copy(item, max_words=5, max_chars=34) for item in slot_values.get("steps") or []]
    steps = [item for item in steps if item]
    labels = [_copy(item, max_words=5, max_chars=34) for item in slot_values.get("required_labels") or []]
    labels = [item for item in labels if item]
    metric_facts = list(slot_values.get("metric_facts") or [])
    emphasis = ""
    if metric_facts:
        first = metric_facts[0] if isinstance(metric_facts[0], dict) else {}
        emphasis = _clean(first.get("value") or first.get("label"))
    if not emphasis:
        emphasis = labels[0] if labels else headline
    left_detail = _copy(slot_values.get("before_state") or slot_values.get("branch_low"), max_words=6, max_chars=48)
    right_detail = _copy(slot_values.get("after_state") or slot_values.get("branch_high"), max_words=6, max_chars=48)
    return {
        "card_id": _clean(card.get("card_id")),
        "template": template,
        "renderer_hint": renderer_hint,
        "composition_mode": composition_mode,
        "style_pack": _clean(card.get("style_pack")) or "signal_lab",
        "headline": headline,
        "deck": deck,
        "emphasis_text": emphasis,
        "supporting_lines": labels[1:4] or steps[:3],
        "steps": steps or labels[:4],
        "keywords": _string_list(card.get("keywords"))[:4] or labels[:4],
        "quote_text": _copy(slot_values.get("exact_quote") or headline, max_words=12, max_chars=96),
        "left_label": "Before" if skill.skill_id != "decision-gate" else "Low confidence",
        "right_label": "After" if skill.skill_id != "decision-gate" else "Continue",
        "left_detail": left_detail,
        "right_detail": right_detail,
        "footer_text": deck,
        "position": "center",
        "scale": 1.0,
        "motion_preset": _motion_preset(skill.skill_id),
        "background_motif": _background_motif(skill.skill_id),
        "layout_variant": _layout_variant(template),
        "rationale": (
            f"Skill graph selected {skill.title} because the source compiled "
            f"as {slot_values.get('scene_type')} with grounded slots."
        ),
        "confidence": max(0.68, min(0.94, skill.priority)),
    }


def _score_skill_route(
    card: dict[str, Any],
    skill: AutoVisualSkill,
    *,
    preflight: dict[str, Any],
    missing_slots: list[str],
) -> float:
    base = _bounded((card.get("opportunity_contract") or {}).get("score"), 0.62)
    if not base:
        base = _bounded(card.get("visualizability"), 0.58)
    score = base * 0.45 + skill.priority * 0.25
    score += min(int(preflight.get("proof_candidate_count") or 0), 4) * 0.035
    score += 0.08 if preflight.get("semantic_signature") else 0.0
    score -= len(missing_slots) * 0.12
    if not bool(preflight.get("passed")):
        score -= 0.25
    return max(0.0, min(1.0, score))


def _semantic_family(scene_type: str) -> str:
    return {
        "architecture_flow": "architecture",
        "causal_intervention": "causal",
        "decision_branch": "decision",
        "evidence_backed_quote": "quote",
        "grounded_interface_walkthrough": "interface",
        "guided_process": "route",
        "matched_state_transform": "transform",
        "metric_delta": "metric",
        "metric_intervention": "metric",
        "metric_proof": "metric",
        "narrative_progression": "narrative",
        "set_partition": "partition",
    }.get(scene_type, scene_type)


def _skill_counts(decisions: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in decisions:
        skill_id = str(item.get("skill_id") or "unknown")
        counts[skill_id] = counts.get(skill_id, 0) + 1
    return counts


def _headline_for(card: dict[str, Any], *, ir: Any, labels: list[str]) -> str:
    if ir is not None and _clean(getattr(ir, "thesis", "")):
        return _clean(getattr(ir, "thesis", ""))
    semantic = dict(card.get("semantic_frame") or {})
    return (
        _first_present(semantic, "viewer_takeaway", "result", "after_state")
        or (labels[0] if labels else "")
        or _clean(card.get("headline"))
        or _clean(card.get("sentence_text"))
    )


def _deck_for(card: dict[str, Any], *, ir: Any, labels: list[str]) -> str:
    if ir is not None and _clean(getattr(ir, "takeaway", "")):
        return _clean(getattr(ir, "takeaway", ""))
    semantic = dict(card.get("semantic_frame") or {})
    return (
        _first_present(semantic, "mental_model", "effect", "context")
        or (labels[1] if len(labels) > 1 else "")
        or _clean(card.get("context_text"))
    )


def _motion_preset(skill_id: str) -> str:
    return {
        "metric-story": "kinetic_pop",
        "causal-spine": "diagram_draw",
        "route-choreography": "diagram_draw",
        "architecture-flow": "diagram_draw",
        "matched-transform": "focus_shift",
        "grounded-interface": "focus_shift",
        "decision-gate": "diagram_draw",
        "narrative-continuity": "story_sweep",
        "partition-compression": "kinetic_pop",
        "exact-quote": "type_sweep",
    }.get(skill_id, "diagram_draw")


def _background_motif(skill_id: str) -> str:
    return {
        "metric-story": "rings",
        "causal-spine": "grid",
        "route-choreography": "grid",
        "architecture-flow": "grid",
        "matched-transform": "bands",
        "grounded-interface": "beams",
        "decision-gate": "grid",
        "narrative-continuity": "constellation",
        "partition-compression": "rings",
        "exact-quote": "constellation",
    }.get(skill_id, "grid")


def _layout_variant(template: str) -> str:
    return {
        "causal_chain": "cause_effect_trace",
        "data_journey": "arc_stage",
        "data_pulse": "data_pulse",
        "decision_matrix": "criteria_grid",
        "decision_tree": "branching_tree",
        "interface_cascade": "cascade_focus",
        "kinetic_route": "route_curve",
        "mechanism_blueprint": "mechanism_blueprint",
        "narrative_arc": "story_arc",
        "pipeline_xray": "pipeline_xray",
        "proof_sequence": "evidence_chain",
        "quote_breakdown": "quote_deconstruction",
        "ribbon_quote": "ribbon_sweep",
        "signal_network": "network_sweep",
        "spotlight_compare": "spotlight_stage",
        "timeline_filmstrip": "filmstrip_timeline",
        "contrast_ladder": "before_after_ladder",
        "problem_solution": "pivot_cards",
    }.get(template, "hero_split")


def _copy(value: Any, *, max_words: int, max_chars: int) -> str:
    cleaned = _clean(value)
    if not cleaned:
        return ""
    words = cleaned.split()
    clipped = " ".join(words[:max_words]).strip()
    if len(clipped) > max_chars:
        clipped = clipped[:max_chars].rstrip(" ,.;:-")
    return clipped


def _first_present(payload: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            value = "; ".join(str(item) for item in value if str(item).strip())
        cleaned = _clean(value)
        if cleaned:
            return cleaned
    return ""


def _semantic_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [
            _clean(item)
            for item in re.split(r"[;|]\s*|\n+", value)
            if _clean(item)
        ]
    return _string_list(value)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value if _clean(item)]
    if value is None:
        return []
    cleaned = _clean(value)
    return [cleaned] if cleaned else []


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


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _words(value: Any) -> list[str]:
    return re.findall(r"[a-z0-9%+./-]+", str(value or "").lower())


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip(" ,.;:-")


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bounded(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(number, 1.0))


__all__ = [
    "AUTO_VISUAL_SKILL_GRAPH_VERSION",
    "AUTO_VISUAL_SKILLS",
    "AutoVisualSkill",
    "SkillRoutingDecision",
    "apply_visual_skill_graph",
    "route_visual_skill",
    "skill_graph_prompt_block",
    "skill_seed_plan",
]
