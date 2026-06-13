from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from visual_explanation import VisualExplanationIR
from vex_hyperframes.blueprints import BlueprintSelection
from vex_hyperframes.claim_graph import VisualClaimGraph, VisualClaimGraphValidation
from vex_hyperframes.production_contract import build_production_contract
from vex_hyperframes.storyboard import StoryboardPanel, StoryboardReview


PROOF_TOURNAMENT_VERSION = "hyperframes-proof-tournament-v1"
PROOF_ENCODING_FAMILIES = frozenset(
    {
        "focal_gate",
        "layered_flow",
        "linear_trace",
        "radial_evidence",
        "split_register",
    }
)


@dataclass(frozen=True)
class ProofEncoding:
    strategy_id: str
    encoding_family: str
    rationale: str
    relation_mode: str


@dataclass(frozen=True)
class VisualProofProgram:
    program_id: str
    blueprint_id: str
    stage_family: str
    strategy_id: str
    encoding_family: str
    rationale: str
    relation_mode: str
    structural_prior: float
    required_relation_ids: list[str]
    production_contract: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["structural_prior"] = round(float(self.structural_prior), 3)
        return payload

    def renderer_overlay(self) -> dict[str, Any]:
        contract = dict(self.production_contract)
        return {
            "proof_program_id": self.program_id,
            "proof_strategy_id": self.strategy_id,
            "proof_encoding": self.encoding_family,
            "proof_relation_mode": self.relation_mode,
            "proof_program": self.to_dict(),
            "semantic_blueprint_id": self.blueprint_id,
            "template": self.stage_family,
            "hyperframes_production_contract": contract,
            "visual_claim_graph": dict(contract.get("visual_claim_graph") or {}),
            "qa_contract": {
                "semantic_signature": contract.get("semantic_signature"),
                "required_labels": list(contract.get("required_labels") or []),
                "required_object_ids": list(contract.get("required_object_ids") or []),
                "required_relation_ids": list(contract.get("required_relation_ids") or []),
                "proof_questions": list(contract.get("proof_questions") or []),
                "required_motion": list(contract.get("required_motion") or []),
                "screenshot_test": contract.get("screenshot_test"),
                "quality_floor": contract.get("quality_floor"),
            },
        }


@dataclass(frozen=True)
class VisualProofTournament:
    version: str
    visual_id: str
    scene_type: str
    programs: list[VisualProofProgram]
    tournament_signature: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "visual_id": self.visual_id,
            "scene_type": self.scene_type,
            "programs": [item.to_dict() for item in self.programs],
            "tournament_signature": self.tournament_signature,
        }


@dataclass(frozen=True)
class VisualProofTournamentValidation:
    passed: bool
    candidate_count: int
    distinct_blueprints: int
    distinct_encodings: int
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SCENE_ENCODINGS: dict[str, tuple[ProofEncoding, ...]] = {
    "set_partition": (
        ProofEncoding("token_groups", "split_register", "Keep original tokens registered against their compressed groups.", "matched_state"),
        ProofEncoding("grouping_trace", "linear_trace", "Trace tokens into fixed-size groups before resolving the block count.", "ordered_trace"),
        ProofEncoding("compression_gate", "focal_gate", "Make the group size the explicit compression operator.", "comparison_gate"),
        ProofEncoding("membership_layers", "layered_flow", "Preserve token membership while compressed blocks assemble.", "constraint_layers"),
    ),
    "metric_delta": (
        ProofEncoding("matched_register", "split_register", "Align measured states so the delta is directly inspectable.", "matched_state"),
        ProofEncoding("evidence_axis", "linear_trace", "Place both measurements on one source-backed evidence axis.", "ordered_trace"),
        ProofEncoding("delta_focus", "focal_gate", "Resolve the comparison through one explicit delta gate.", "comparison_gate"),
        ProofEncoding("evidence_orbit", "radial_evidence", "Keep supporting states attached to the measured claim.", "evidence_convergence"),
    ),
    "metric_intervention": (
        ProofEncoding("causal_trace", "linear_trace", "Keep intervention and measured response on one causal path.", "causal_sequence"),
        ProofEncoding("threshold_gate", "focal_gate", "Make the intervention the visible gate between measured states.", "intervention_gate"),
        ProofEncoding("counterfactual_register", "split_register", "Compare untreated and intervened states with matched geometry.", "counterfactual"),
        ProofEncoding("mechanism_layers", "layered_flow", "Expose the intervention mechanism without detaching the metric.", "layered_causality"),
    ),
    "metric_proof": (
        ProofEncoding("evidence_spine", "linear_trace", "Accumulate evidence before locking the hero metric.", "evidence_sequence"),
        ProofEncoding("proof_ladder", "layered_flow", "Build a hierarchy of evidence that resolves into the claim.", "evidence_accumulation"),
        ProofEncoding("focal_metric", "focal_gate", "Center one metric and require every proof object to support it.", "claim_gate"),
        ProofEncoding("evidence_orbit", "radial_evidence", "Converge source-backed proof around the measured claim.", "evidence_convergence"),
    ),
    "causal_intervention": (
        ProofEncoding("mechanism_trace", "linear_trace", "Keep cause, mechanism, intervention, and result on one path.", "causal_sequence"),
        ProofEncoding("counterfactual_split", "split_register", "Compare the same input with and without intervention.", "counterfactual"),
        ProofEncoding("mechanism_chamber", "focal_gate", "Make the mechanism the decisive transformation gate.", "mechanism_gate"),
        ProofEncoding("causal_layers", "layered_flow", "Separate causal responsibilities while preserving direction.", "layered_causality"),
    ),
    "guided_process": (
        ProofEncoding("process_route", "linear_trace", "Preserve source order along one readable route.", "ordered_trace"),
        ProofEncoding("handoff_zones", "split_register", "Make ownership transfer visible across matched zones.", "handoff"),
        ProofEncoding("progressive_stack", "layered_flow", "Retain completed-state memory as the process advances.", "progressive_disclosure"),
        ProofEncoding("active_step_gate", "focal_gate", "Focus each beat on the currently active process decision.", "step_gate"),
    ),
    "architecture_flow": (
        ProofEncoding("service_route", "linear_trace", "Track one request through explicit service boundaries.", "ordered_trace"),
        ProofEncoding("layered_pipeline", "layered_flow", "Expose responsibility by layer while preserving end-to-end flow.", "layered_causality"),
        ProofEncoding("boundary_handoff", "split_register", "Make boundary crossings and ownership changes inspectable.", "handoff"),
        ProofEncoding("system_hub", "radial_evidence", "Converge dependent services around the grounded request lifecycle.", "dependency_convergence"),
    ),
    "matched_state_transform": (
        ProofEncoding("matched_morph", "split_register", "Register equivalent objects so the transformation explains itself.", "matched_state"),
        ProofEncoding("constraint_anchor", "focal_gate", "Keep the preserved constraint fixed while state changes around it.", "constraint_gate"),
        ProofEncoding("transformation_trace", "linear_trace", "Show source state, change, and result in one ordered path.", "ordered_trace"),
        ProofEncoding("transformation_layers", "layered_flow", "Separate changing and preserved properties into explicit layers.", "constraint_layers"),
    ),
    "grounded_interface_walkthrough": (
        ProofEncoding("source_focus", "focal_gate", "Anchor annotations to the real interface evidence.", "source_focus"),
        ProofEncoding("action_result", "split_register", "Match the grounded action with its resulting interface state.", "matched_state"),
        ProofEncoding("focus_trace", "linear_trace", "Trace the user action through the interface in source order.", "ordered_trace"),
        ProofEncoding("interface_layers", "layered_flow", "Separate source surface, action, and feedback without fabricating UI.", "source_layers"),
    ),
    "decision_branch": (
        ProofEncoding("centered_gate", "focal_gate", "Make the decision rule the visible center of both outcomes.", "decision_gate"),
        ProofEncoding("branch_comparison", "split_register", "Compare mutually exclusive branches with matched treatment.", "exclusive_branches"),
        ProofEncoding("guardrail_route", "linear_trace", "Trace evidence through the gate to the selected branch.", "ordered_trace"),
        ProofEncoding("decision_layers", "layered_flow", "Separate evidence, rule, and outcomes into an inspectable hierarchy.", "decision_layers"),
    ),
    "narrative_progression": (
        ProofEncoding("recovery_arc", "linear_trace", "Make setup, turn, and payoff one continuous trajectory.", "ordered_trace"),
        ProofEncoding("turning_point", "focal_gate", "Make the decisive turn visibly alter the trajectory.", "turn_gate"),
        ProofEncoding("state_trajectory", "split_register", "Contrast the state before and after the turning point.", "matched_state"),
        ProofEncoding("story_layers", "layered_flow", "Retain setup context while the turning point resolves.", "narrative_layers"),
    ),
    "evidence_backed_quote": (
        ProofEncoding("phrase_assembly", "linear_trace", "Assemble exact source language in reading order.", "ordered_trace"),
        ProofEncoding("clause_focus", "split_register", "Isolate source-backed clauses without losing the full quote.", "clause_comparison"),
        ProofEncoding("quote_lock", "focal_gate", "Resolve every emphasis back into the exact quote.", "quote_gate"),
        ProofEncoding("semantic_orbit", "radial_evidence", "Attach source-backed concepts to the intact quoted phrase.", "evidence_convergence"),
    ),
}


def build_visual_proof_tournament(
    ir: VisualExplanationIR,
    selections: list[BlueprintSelection],
    panels: list[StoryboardPanel],
    review: StoryboardReview,
    claim_graph: VisualClaimGraph,
    claim_graph_validation: VisualClaimGraphValidation,
    *,
    candidate_count: int = 4,
) -> VisualProofTournament:
    encodings = _SCENE_ENCODINGS.get(ir.scene_type, ())
    valid_selections = [
        item for item in selections if item.passed and item.blueprint is not None
    ]
    count = max(1, min(int(candidate_count), 8))
    combinations: list[tuple[BlueprintSelection, ProofEncoding]] = []
    seen: set[tuple[str, str]] = set()

    for index, selection in enumerate(valid_selections):
        if not encodings:
            break
        encoding = encodings[index % len(encodings)]
        key = (selection.blueprint.blueprint_id, encoding.strategy_id)
        if key not in seen:
            combinations.append((selection, encoding))
            seen.add(key)
    used_strategies = {encoding.strategy_id for _, encoding in combinations}
    for encoding in encodings:
        if len(combinations) >= count:
            break
        if encoding.strategy_id in used_strategies or not valid_selections:
            continue
        selection = valid_selections[len(combinations) % len(valid_selections)]
        key = (selection.blueprint.blueprint_id, encoding.strategy_id)
        if key not in seen:
            combinations.append((selection, encoding))
            seen.add(key)
            used_strategies.add(encoding.strategy_id)
    for encoding in encodings:
        for selection in valid_selections:
            key = (selection.blueprint.blueprint_id, encoding.strategy_id)
            if key in seen:
                continue
            combinations.append((selection, encoding))
            seen.add(key)
            if len(combinations) >= count:
                break
        if len(combinations) >= count:
            break

    required_relation_ids = [
        item.relation_id for item in claim_graph.relations if item.required
    ]
    programs: list[VisualProofProgram] = []
    for index, (selection, encoding) in enumerate(combinations[:count]):
        blueprint = selection.blueprint
        assert blueprint is not None
        contract = build_production_contract(
            ir,
            blueprint,
            panels,
            review,
            claim_graph,
            claim_graph_validation,
        )
        program_id = (
            f"proof_{index + 1:02d}_{blueprint.blueprint_id}_{encoding.strategy_id}"
        )
        programs.append(
            VisualProofProgram(
                program_id=program_id,
                blueprint_id=blueprint.blueprint_id,
                stage_family=blueprint.stage_family,
                strategy_id=encoding.strategy_id,
                encoding_family=encoding.encoding_family,
                rationale=encoding.rationale,
                relation_mode=encoding.relation_mode,
                structural_prior=min(
                    1.0,
                    selection.score
                    + max(0.0, 0.04 - index * 0.008),
                ),
                required_relation_ids=required_relation_ids,
                production_contract=contract.to_dict(),
            )
        )
    payload = {
        "version": PROOF_TOURNAMENT_VERSION,
        "visual_id": ir.visual_id,
        "scene_type": ir.scene_type,
        "programs": [item.to_dict() for item in programs],
    }
    signature = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    return VisualProofTournament(
        version=PROOF_TOURNAMENT_VERSION,
        visual_id=ir.visual_id,
        scene_type=ir.scene_type,
        programs=programs,
        tournament_signature=signature,
    )


def validate_visual_proof_tournament(
    tournament: VisualProofTournament | dict[str, Any],
) -> VisualProofTournamentValidation:
    payload = (
        tournament.to_dict()
        if isinstance(tournament, VisualProofTournament)
        else dict(tournament)
    )
    programs = list(payload.get("programs") or [])
    errors: list[str] = []
    program_ids = [str(item.get("program_id") or "") for item in programs]
    blueprints = {str(item.get("blueprint_id") or "") for item in programs}
    encodings = {str(item.get("encoding_family") or "") for item in programs}
    if not programs:
        errors.append("proof_tournament_has_no_candidates")
    if len(program_ids) != len(set(program_ids)) or any(not item for item in program_ids):
        errors.append("proof_tournament_program_ids_invalid")
    if not encodings.issubset(PROOF_ENCODING_FAMILIES):
        errors.append("proof_tournament_has_unknown_encoding")
    for item in programs:
        contract = dict(item.get("production_contract") or {})
        if contract.get("blueprint_id") != item.get("blueprint_id"):
            errors.append("proof_program_contract_blueprint_mismatch")
        if not contract.get("semantic_signature") or not contract.get("passed"):
            errors.append("proof_program_contract_invalid")
    signature_payload = {
        "version": payload.get("version"),
        "visual_id": payload.get("visual_id"),
        "scene_type": payload.get("scene_type"),
        "programs": programs,
    }
    expected_signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    if payload.get("tournament_signature") != expected_signature:
        errors.append("proof_tournament_signature_mismatch")
    return VisualProofTournamentValidation(
        passed=not errors,
        candidate_count=len(programs),
        distinct_blueprints=len(blueprints - {""}),
        distinct_encodings=len(encodings - {""}),
        errors=_unique(errors),
    )


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


__all__ = [
    "PROOF_ENCODING_FAMILIES",
    "PROOF_TOURNAMENT_VERSION",
    "ProofEncoding",
    "VisualProofProgram",
    "VisualProofTournament",
    "VisualProofTournamentValidation",
    "build_visual_proof_tournament",
    "validate_visual_proof_tournament",
]
