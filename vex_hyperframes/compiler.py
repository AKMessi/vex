from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from visual_explanation import (
    VisualExplanationIR,
    build_visual_explanation_ir,
    validate_visual_explanation_ir,
    visual_explanation_ir_from_dict,
    visual_explanation_ir_signature,
)
from vex_hyperframes.authoring import build_bespoke_program
from vex_hyperframes.blueprints import BlueprintSelection, rank_blueprints, select_blueprint
from vex_hyperframes.claim_graph import (
    VisualClaimGraph,
    VisualClaimGraphValidation,
    build_visual_claim_graph,
    validate_visual_claim_graph,
)
from vex_hyperframes.production_contract import (
    HyperframesProductionContract,
    build_production_contract,
)
from vex_hyperframes.proof_program import (
    VisualProofTournament,
    VisualProofTournamentValidation,
    build_visual_proof_tournament,
    validate_visual_proof_tournament,
)
from vex_hyperframes.storyboard import (
    StoryboardPanel,
    StoryboardReview,
    build_storyboard,
    review_storyboard,
)
from vex_hyperframes.scene_program import build_scene_program
from vex_hyperframes.visual_world import build_visual_world_program
from vex_visuals.creative_direction import compile_creative_direction


@dataclass(frozen=True)
class CompiledHyperframesPlan:
    passed: bool
    ir: VisualExplanationIR
    storyboard: list[StoryboardPanel]
    storyboard_review: StoryboardReview
    claim_graph: VisualClaimGraph
    claim_graph_validation: VisualClaimGraphValidation
    blueprint_selection: BlueprintSelection
    production_contract: HyperframesProductionContract | None
    proof_tournament: VisualProofTournament
    proof_tournament_validation: VisualProofTournamentValidation
    renderer_spec: dict[str, Any]
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "ir": self.ir.to_dict(),
            "storyboard": [item.to_dict() for item in self.storyboard],
            "storyboard_review": self.storyboard_review.to_dict(),
            "claim_graph": self.claim_graph.to_dict(),
            "claim_graph_validation": self.claim_graph_validation.to_dict(),
            "blueprint_selection": self.blueprint_selection.to_dict(),
            "production_contract": (
                self.production_contract.to_dict()
                if self.production_contract
                else None
            ),
            "proof_tournament": self.proof_tournament.to_dict(),
            "proof_tournament_validation": self.proof_tournament_validation.to_dict(),
            "renderer_spec": dict(self.renderer_spec),
            "issues": list(self.issues),
        }


def compile_hyperframes_plan(spec: dict[str, Any]) -> CompiledHyperframesPlan:
    supplied_ir = dict(spec.get("visual_explanation_ir") or {})
    ir = (
        visual_explanation_ir_from_dict(supplied_ir)
        if supplied_ir
        else build_visual_explanation_ir(spec)
    )
    ir_validation = validate_visual_explanation_ir(ir)
    storyboard = build_storyboard(ir)
    review = review_storyboard(ir, storyboard)
    claim_graph = build_visual_claim_graph(ir)
    claim_graph_validation = validate_visual_claim_graph(claim_graph)
    ranked_selections = rank_blueprints(ir, spec)
    selection = select_blueprint(ir, spec)
    tournament = build_visual_proof_tournament(
        ir,
        ranked_selections,
        storyboard,
        review,
        claim_graph,
        claim_graph_validation,
        candidate_count=_candidate_count(spec),
    )
    tournament_validation = validate_visual_proof_tournament(tournament)
    contract = None
    issues: list[str] = []
    expected_ir_signature = str(
        (spec.get("opportunity_contract") or {}).get(
            "visual_explanation_ir_signature"
        )
        or (spec.get("opportunity_preflight") or {}).get(
            "visual_explanation_ir_signature"
        )
        or ""
    )
    if expected_ir_signature and visual_explanation_ir_signature(ir) != expected_ir_signature:
        issues.append("visual_explanation_ir_signature_mismatch")
    if not ir_validation.passed:
        issues.extend(ir_validation.errors)
    if ir.render_policy != "render":
        issues.extend(ir.rejection_reasons)
    if not review.passed:
        issues.extend(review.fatal_issues)
    if not claim_graph_validation.passed:
        issues.extend(claim_graph_validation.errors)
    if not selection.passed or selection.blueprint is None:
        issues.extend(selection.reasons)
        issues.extend(f"missing_role:{role}" for role in selection.missing_roles)
    if not tournament_validation.passed:
        issues.extend(tournament_validation.errors)
    if selection.blueprint is not None:
        contract = build_production_contract(
            ir,
            selection.blueprint,
            storyboard,
            review,
            claim_graph,
            claim_graph_validation,
        )
        if not contract.passed:
            issues.extend(contract.issues)
    passed = (
        ir_validation.passed
        and ir.render_policy == "render"
        and review.passed
        and claim_graph_validation.passed
        and selection.passed
        and contract is not None
        and contract.passed
        and tournament_validation.passed
        and "visual_explanation_ir_signature_mismatch" not in issues
    )
    renderer_spec = (
        _renderer_spec(
            spec,
            ir,
            storyboard,
            selection,
            contract,
            tournament,
        )
        if passed
        else {}
    )
    return CompiledHyperframesPlan(
        passed=passed,
        ir=ir,
        storyboard=storyboard,
        storyboard_review=review,
        claim_graph=claim_graph,
        claim_graph_validation=claim_graph_validation,
        blueprint_selection=selection,
        production_contract=contract,
        proof_tournament=tournament,
        proof_tournament_validation=tournament_validation,
        renderer_spec=renderer_spec,
        issues=_unique(issues, limit=20),
    )


def _renderer_spec(
    spec: dict[str, Any],
    ir: VisualExplanationIR,
    storyboard: list[StoryboardPanel],
    selection: BlueprintSelection,
    contract: HyperframesProductionContract | None,
    tournament: VisualProofTournament,
) -> dict[str, Any]:
    blueprint = selection.blueprint
    assert blueprint is not None
    assert contract is not None
    object_payloads = [item.to_dict() for item in ir.objects]
    labels = [item.label for item in ir.objects]
    facts = [item.to_dict() for item in ir.facts]
    semantic_frame = dict(spec.get("semantic_frame") or {})
    proof_programs: list[dict[str, Any]] = []
    visual_world_history = [
        dict(item)
        for item in spec.get("visual_world_history") or []
        if isinstance(item, dict)
    ]
    for index, item in enumerate(tournament.programs):
        scene_program = build_scene_program(
            ir,
            contract.visual_claim_graph,
            storyboard,
            blueprint_id=item.blueprint_id,
            proof_program_id=item.program_id,
            proof_encoding=item.encoding_family,
            semantic_signature=str(
                item.production_contract.get("semantic_signature") or ""
            ),
        )
        visual_world = build_visual_world_program(
            ir,
            scene_program,
            proof_program_id=item.program_id,
            proof_encoding=item.encoding_family,
            variant_index=index,
            spec={**spec, "visual_world_history": visual_world_history},
        )
        creative_direction = compile_creative_direction(
            spec,
            scene_type=ir.scene_type,
            scene_family=_scene_family(ir.scene_type),
            objects=[item.to_dict() for item in scene_program.elements],
            relations=[item.to_dict() for item in scene_program.relations],
            width=int(spec.get("width") or 1280),
            height=int(spec.get("height") or 720),
            variant_index=index,
            visual_world=visual_world.to_dict(),
        )
        visual_world_history.append(visual_world.fingerprint.to_dict())
        proof_programs.append(
            {
                **item.renderer_overlay(),
                "scene_program_v2": scene_program.to_dict(),
                "visual_world_program": visual_world.to_dict(),
                "creative_direction_program": creative_direction.to_dict(),
            }
        )
    renderer_spec = {
        **dict(spec),
        "template": blueprint.stage_family,
        "semantic_blueprint_id": blueprint.blueprint_id,
        "visual_explanation_ir": ir.to_dict(),
        "hyperframes_storyboard": [item.to_dict() for item in storyboard],
        "hyperframes_production_contract": contract.to_dict(),
        "visual_proof_tournament": tournament.to_dict(),
        "visual_proof_programs": proof_programs,
        "visual_claim_graph": contract.visual_claim_graph,
        "headline": str(ir.metadata.get("display_title") or ir.thesis or labels[0]),
        "deck": ir.takeaway,
        "steps": labels,
        "supporting_lines": labels[1:],
        "semantic_objects": object_payloads,
        "grounded_facts": facts,
        "semantic_frame": semantic_frame,
        "qa_contract": {
            **dict(spec.get("qa_contract") or {}),
            "semantic_signature": contract.semantic_signature,
            "required_labels": list(contract.required_labels),
            "required_object_ids": list(contract.required_object_ids),
            "required_relation_ids": list(contract.required_relation_ids),
            "proof_questions": list(contract.proof_questions),
            "required_motion": list(contract.required_motion),
            "screenshot_test": contract.screenshot_test,
            "quality_floor": contract.quality_floor,
        },
    }
    if tournament.programs:
        primary_program = tournament.programs[0]
        renderer_spec.update(
            {
                "proof_program_id": primary_program.program_id,
                "proof_strategy_id": primary_program.strategy_id,
                "proof_encoding": primary_program.encoding_family,
                "proof_relation_mode": primary_program.relation_mode,
                "proof_program": primary_program.to_dict(),
                "scene_program_v2": renderer_spec["visual_proof_programs"][0][
                    "scene_program_v2"
                ],
                "visual_world_program": renderer_spec["visual_proof_programs"][0][
                    "visual_world_program"
                ],
                "creative_direction_program": renderer_spec["visual_proof_programs"][0][
                    "creative_direction_program"
                ],
            }
        )
    authoring_mode = str(
        spec.get("hyperframes_authoring_mode") or spec.get("authoring_mode") or ""
    ).strip().lower()
    if authoring_mode == "bespoke":
        renderer_spec["bespoke_scene_program"] = build_bespoke_program(
            ir,
            blueprint_id=blueprint.blueprint_id,
            variant_index=int(spec.get("variant_index") or 0),
        ).to_dict()
    return renderer_spec


def _candidate_count(spec: dict[str, Any]) -> int:
    value = spec.get(
        "hyperframes_proof_candidate_count",
        spec.get("hyperframes_variant_count", 4),
    )
    try:
        count = int(value)
    except (TypeError, ValueError):
        count = 4
    return max(1, min(count, 8))


def _scene_family(scene_type: str) -> str:
    if scene_type.startswith("metric_"):
        return "metric"
    if scene_type in {"matched_state_transform", "decision_branch"}:
        return "contrast"
    if scene_type == "narrative_progression":
        return "timeline"
    if scene_type == "grounded_interface_walkthrough":
        return "interface"
    if scene_type == "evidence_backed_quote":
        return "emphasis"
    return "mechanism"


def _unique(values: list[str], *, limit: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = str(value or "").strip()
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
        if len(result) >= limit:
            break
    return result


__all__ = [
    "CompiledHyperframesPlan",
    "compile_hyperframes_plan",
]
