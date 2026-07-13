from __future__ import annotations

from dataclasses import replace

from tests.test_visual_communication_contract import _ir
from tests.test_visual_verifier import _payload
from vex_visuals.communication_contract import build_communication_contract
from vex_visuals.generative_authoring import compile_open_visual_program_for_spec
from vex_visuals.repair import (
    RepairLevel,
    apply_visual_repair,
    assess_repair_improvement,
    plan_visual_repair,
)
from vex_visuals.verifier import VisualQualityState, evaluate_verifier_payload


def _compiled_spec() -> dict:
    spec, result = compile_open_visual_program_for_spec(
        {"visual_id": "visual_001", "duration": 4.8},
        ir=_ir(),
        width=1920,
        height=1080,
        fps=60,
        enable_model_authoring=False,
        candidate_count=4,
    )
    assert result.passed
    return spec


def _degraded_report():
    contract = build_communication_contract(_ir())
    payload = _payload(contract)
    payload["answers"] = {}
    payload["design"] = {
        "hierarchy": 0.55,
        "composition": 0.58,
        "typography": 0.62,
        "density": 0.55,
        "polish": 0.6,
        "originality": 0.52,
        "issues": ["Weak hierarchy"],
    }
    payload["temporal"] = {
        "causal_readability": 0.5,
        "meaningful_motion": 0.48,
        "smoothness": 0.76,
        "sequence_legibility": 0.5,
        "settling": 0.52,
        "issues": ["Motion is decorative"],
    }
    return evaluate_verifier_payload(payload, contract)


def test_repair_plan_uses_typed_semantic_composition_and_motion_operations() -> None:
    plan = plan_visual_repair(_degraded_report(), _compiled_spec(), round_index=1)
    levels = {item.level for item in plan.operations}

    assert plan.requires_concept_regeneration
    assert RepairLevel.SEMANTIC_ENCODING in levels
    assert RepairLevel.COMPOSITION in levels
    assert RepairLevel.MOTION_CAUSALITY in levels
    assert plan.signature


def test_verified_candidate_can_request_bounded_alternate_concept_search() -> None:
    contract = build_communication_contract(_ir())
    verified = evaluate_verifier_payload(_payload(contract), contract)

    plan = plan_visual_repair(
        verified,
        _compiled_spec(),
        round_index=1,
        explore_alternate=True,
    )

    assert [item.operation for item in plan.operations] == [
        "promote_alternate_concept"
    ]


def test_semantic_repair_promotes_alternate_program_and_keeps_schema_valid() -> None:
    spec = _compiled_spec()
    original_id = spec["open_visual_program"]["program_id"]
    plan = plan_visual_repair(_degraded_report(), spec, round_index=1)

    result = apply_visual_repair(spec, plan, ir=_ir())

    assert result.passed
    assert result.changed
    assert result.validation["passed"]
    assert result.promoted_program_id
    assert result.promoted_program_id != original_id
    assert result.spec["open_visual_program"]["signature"]


def test_copy_repair_replaces_conversational_filler_with_grounded_fact() -> None:
    spec = _compiled_spec()
    title = next(
        item
        for item in spec["open_visual_program"]["elements"]
        if item["role"] == "title"
    )
    title["text"] = "Okay, so now"
    # Re-signing occurs inside the repair application.
    report = _degraded_report()
    plan = plan_visual_repair(report, spec, round_index=1)

    result = apply_visual_repair(spec, plan, ir=_ir())
    repaired_title = next(
        item
        for item in result.spec["open_visual_program"]["elements"]
        if item["role"] == "title"
    )

    assert result.validation["passed"]
    assert repaired_title["text"] != "Okay, so now"
    assert "replace_weak_copy" in {
        item.operation for item in plan.operations
    }


def test_monotonic_assessment_accepts_verified_improvement_and_rejects_regression() -> None:
    contract = build_communication_contract(_ir())
    before = _degraded_report()
    after = evaluate_verifier_payload(_payload(contract), contract)

    improved = assess_repair_improvement(before, after)
    regressed = assess_repair_improvement(
        after,
        replace(
            after,
            state=VisualQualityState.DEGRADED,
            publishable=False,
            score=0.6,
            semantic=replace(after.semantic, score=0.4, passed=False),
        ),
    )

    assert improved.accepted
    assert improved.state_improved
    assert not regressed.accepted
    assert "semantic_quality_regressed" in regressed.reasons
