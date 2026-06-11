from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from visual_explanation import VisualExplanationIR
from vex_hyperframes.blueprints import HyperframesBlueprint
from vex_hyperframes.claim_graph import (
    VisualClaimGraph,
    VisualClaimGraphValidation,
)
from vex_hyperframes.storyboard import StoryboardPanel, StoryboardReview


@dataclass(frozen=True)
class HyperframesProductionContract:
    contract_version: str
    visual_id: str
    scene_type: str
    blueprint_id: str
    thesis: str
    viewer_question: str
    takeaway: str
    required_labels: list[str]
    required_object_ids: list[str]
    required_relation_ids: list[str]
    proof_questions: list[dict[str, Any]]
    visual_claim_graph: dict[str, Any]
    claim_graph_signature: str
    required_motion: list[str]
    required_devices: list[str]
    screenshot_test: str
    forbidden_content: list[str]
    semantic_signature: str
    quality_floor: float
    passed: bool
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["quality_floor"] = round(float(self.quality_floor), 3)
        return payload


def build_production_contract(
    ir: VisualExplanationIR,
    blueprint: HyperframesBlueprint,
    panels: list[StoryboardPanel],
    review: StoryboardReview,
    claim_graph: VisualClaimGraph,
    claim_graph_validation: VisualClaimGraphValidation,
) -> HyperframesProductionContract:
    issues: list[str] = []
    object_ids = [item.object_id for item in ir.objects]
    motion = _unique([panel.visual_change for panel in panels], limit=8)
    if ir.render_policy != "render":
        issues.append("visual_explanation_ir_rejected")
    if not review.passed:
        issues.extend(review.fatal_issues)
    if not claim_graph_validation.passed:
        issues.extend(claim_graph_validation.errors)
    if not ir.required_labels:
        issues.append("no_required_visible_labels")
    if len(motion) < 2:
        issues.append("insufficient_semantic_motion")
    if any(pattern.lower() in {label.lower() for label in ir.required_labels} for pattern in ir.forbidden_content):
        issues.append("required_copy_conflicts_with_forbidden_content")
    signature_payload = {
        "visual_id": ir.visual_id,
        "scene_type": ir.scene_type,
        "blueprint_id": blueprint.blueprint_id,
        "facts": [item.to_dict() for item in ir.facts],
        "objects": [item.to_dict() for item in ir.objects],
        "beats": [item.to_dict() for item in ir.beats],
        "claim_graph": claim_graph.to_dict(),
    }
    semantic_signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    return HyperframesProductionContract(
        contract_version="hyperframes-production-v3",
        visual_id=ir.visual_id,
        scene_type=ir.scene_type,
        blueprint_id=blueprint.blueprint_id,
        thesis=ir.thesis,
        viewer_question=ir.viewer_question,
        takeaway=ir.takeaway,
        required_labels=list(ir.required_labels),
        required_object_ids=object_ids,
        required_relation_ids=[
            item.relation_id for item in claim_graph.relations if item.required
        ],
        proof_questions=[item.to_dict() for item in claim_graph.questions],
        visual_claim_graph=claim_graph.to_dict(),
        claim_graph_signature=claim_graph.graph_signature,
        required_motion=motion,
        required_devices=list(blueprint.dynamic_devices),
        screenshot_test=(
            panels[-1].final_frame_requirement
            if panels
            else "The final frame must communicate the source-backed relationship without transcript help."
        ),
        forbidden_content=_unique(
            [*ir.forbidden_content, *blueprint.anti_patterns],
            limit=16,
        ),
        semantic_signature=semantic_signature,
        quality_floor=_quality_floor(ir, blueprint),
        passed=not issues,
        issues=_unique(issues, limit=12),
    )


def production_contract_prompt_block(
    contract: HyperframesProductionContract | dict[str, Any],
) -> str:
    payload = contract.to_dict() if isinstance(contract, HyperframesProductionContract) else dict(contract)
    return "\n".join(
        [
            "HyperFrames production contract:",
            f"- Scene type: {payload.get('scene_type')}",
            f"- Blueprint: {payload.get('blueprint_id')}",
            f"- Thesis: {payload.get('thesis')}",
            f"- Viewer question: {payload.get('viewer_question')}",
            f"- Takeaway: {payload.get('takeaway')}",
            "- Required labels: " + "; ".join(payload.get("required_labels") or []),
            "- Required relations: "
            + "; ".join(payload.get("required_relation_ids") or []),
            "- Blind proof questions: "
            + "; ".join(
                str(item.get("prompt") or "")
                for item in payload.get("proof_questions") or []
                if isinstance(item, dict)
            ),
            "- Required motion: " + "; ".join(payload.get("required_motion") or []),
            "- Required devices: " + "; ".join(payload.get("required_devices") or []),
            f"- Screenshot test: {payload.get('screenshot_test')}",
            "- Forbidden: " + "; ".join(payload.get("forbidden_content") or []),
            f"- Semantic signature: {payload.get('semantic_signature')}",
            f"- Claim graph signature: {payload.get('claim_graph_signature')}",
        ]
    )


def _quality_floor(ir: VisualExplanationIR, blueprint: HyperframesBlueprint) -> float:
    floor = 0.78
    if ir.composition_mode == "replace":
        floor += 0.03
    if ir.scene_type in {
        "architecture_flow",
        "causal_intervention",
        "grounded_interface_walkthrough",
        "metric_intervention",
    }:
        floor += 0.02
    floor += max(0.0, blueprint.priority - 0.9) * 0.1
    return min(round(floor, 3), 0.86)


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
    "HyperframesProductionContract",
    "build_production_contract",
    "production_contract_prompt_block",
]
