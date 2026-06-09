from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from visual_explanation import (
    VisualExplanationIR,
    build_visual_explanation_ir,
    validate_visual_explanation_ir,
)
from vex_hyperframes.authoring import build_bespoke_program
from vex_hyperframes.blueprints import BlueprintSelection, select_blueprint
from vex_hyperframes.production_contract import (
    HyperframesProductionContract,
    build_production_contract,
)
from vex_hyperframes.storyboard import (
    StoryboardPanel,
    StoryboardReview,
    build_storyboard,
    review_storyboard,
)


@dataclass(frozen=True)
class CompiledHyperframesPlan:
    passed: bool
    ir: VisualExplanationIR
    storyboard: list[StoryboardPanel]
    storyboard_review: StoryboardReview
    blueprint_selection: BlueprintSelection
    production_contract: HyperframesProductionContract | None
    renderer_spec: dict[str, Any]
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "ir": self.ir.to_dict(),
            "storyboard": [item.to_dict() for item in self.storyboard],
            "storyboard_review": self.storyboard_review.to_dict(),
            "blueprint_selection": self.blueprint_selection.to_dict(),
            "production_contract": (
                self.production_contract.to_dict()
                if self.production_contract
                else None
            ),
            "renderer_spec": dict(self.renderer_spec),
            "issues": list(self.issues),
        }


def compile_hyperframes_plan(spec: dict[str, Any]) -> CompiledHyperframesPlan:
    ir = build_visual_explanation_ir(spec)
    ir_validation = validate_visual_explanation_ir(ir)
    storyboard = build_storyboard(ir)
    review = review_storyboard(ir, storyboard)
    selection = select_blueprint(ir, spec)
    contract = None
    issues: list[str] = []
    if not ir_validation.passed:
        issues.extend(ir_validation.errors)
    if ir.render_policy != "render":
        issues.extend(ir.rejection_reasons)
    if not review.passed:
        issues.extend(review.fatal_issues)
    if not selection.passed or selection.blueprint is None:
        issues.extend(selection.reasons)
        issues.extend(f"missing_role:{role}" for role in selection.missing_roles)
    if selection.blueprint is not None:
        contract = build_production_contract(ir, selection.blueprint, storyboard, review)
        if not contract.passed:
            issues.extend(contract.issues)
    passed = (
        ir_validation.passed
        and ir.render_policy == "render"
        and review.passed
        and selection.passed
        and contract is not None
        and contract.passed
    )
    renderer_spec = _renderer_spec(spec, ir, storyboard, selection, contract) if passed else {}
    return CompiledHyperframesPlan(
        passed=passed,
        ir=ir,
        storyboard=storyboard,
        storyboard_review=review,
        blueprint_selection=selection,
        production_contract=contract,
        renderer_spec=renderer_spec,
        issues=_unique(issues, limit=20),
    )


def _renderer_spec(
    spec: dict[str, Any],
    ir: VisualExplanationIR,
    storyboard: list[StoryboardPanel],
    selection: BlueprintSelection,
    contract: HyperframesProductionContract | None,
) -> dict[str, Any]:
    blueprint = selection.blueprint
    assert blueprint is not None
    assert contract is not None
    object_payloads = [item.to_dict() for item in ir.objects]
    labels = [item.label for item in ir.objects]
    facts = [item.to_dict() for item in ir.facts]
    semantic_frame = dict(spec.get("semantic_frame") or {})
    renderer_spec = {
        **dict(spec),
        "template": blueprint.stage_family,
        "semantic_blueprint_id": blueprint.blueprint_id,
        "visual_explanation_ir": ir.to_dict(),
        "hyperframes_storyboard": [item.to_dict() for item in storyboard],
        "hyperframes_production_contract": contract.to_dict(),
        "headline": ir.thesis or labels[0],
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
            "required_motion": list(contract.required_motion),
            "screenshot_test": contract.screenshot_test,
            "quality_floor": contract.quality_floor,
        },
    }
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
