from __future__ import annotations

from vex_hyperframes.compiler import compile_hyperframes_plan
from vex_hyperframes.counterexamples import VisualCounterexample
from vex_hyperframes.patches import (
    VisualPatchOperation,
    VisualPatchSet,
    apply_visual_patch_set,
    plan_visual_patches,
    validate_patch_set,
)


def test_patch_planner_strengthens_exact_failed_relation() -> None:
    plan = compile_hyperframes_plan(_process_spec())
    program = plan.renderer_spec["scene_program_v2"]
    relation = program["relations"][0]
    counterexample = VisualCounterexample(
        counterexample_id="blind_01_missing_relation",
        critic="blind",
        issue_type="missing_relation",
        severity="hard_failure",
        summary="Route is not decodable.",
        expected="Visible route",
        observed="No route",
        confidence=0.96,
        relation_ids=[relation["relation_id"]],
        evidence_ids=list(relation["evidence_ids"]),
        allowed_repairs=["strengthen_relation", "swap_proof_encoding"],
    )

    patch_set = plan_visual_patches(
        [counterexample],
        scene_program=program,
        round_index=1,
    )
    application = apply_visual_patch_set(
        patch_set,
        scene_program=program,
        ir=plan.ir.to_dict(),
        claim_graph=plan.claim_graph.to_dict(),
    )

    assert application.passed is True
    repaired = next(
        item
        for item in application.scene_program["relations"]
        if item["relation_id"] == relation["relation_id"]
    )
    assert repaired["strength"] == 1.0
    assert repaired["reveal_fraction"] < relation["reveal_fraction"]
    assert repaired["evidence_ids"] == relation["evidence_ids"]
    assert (
        application.scene_program["graph_signature"]
        == program["graph_signature"]
    )
    assert (
        application.scene_program["program_signature"]
        != program["program_signature"]
    )


def test_patch_application_reflows_layout_without_changing_copy() -> None:
    plan = compile_hyperframes_plan(_process_spec())
    program = plan.renderer_spec["scene_program_v2"]
    counterexample = VisualCounterexample(
        counterexample_id="design_01_density",
        critic="design",
        issue_type="density",
        severity="error",
        summary="Too dense",
        expected="Readable grouping",
        observed="Crowded",
        confidence=0.9,
        allowed_repairs=["change_layout_family"],
    )

    patch_set = plan_visual_patches(
        [counterexample],
        scene_program=program,
        round_index=1,
    )
    application = apply_visual_patch_set(
        patch_set,
        scene_program=program,
        ir=plan.ir.to_dict(),
        claim_graph=plan.claim_graph.to_dict(),
    )

    assert application.passed is True
    assert (
        application.scene_program["layout_family"]
        != program["layout_family"]
    )
    assert [
        item["text"] for item in application.scene_program["elements"]
    ] == [item["text"] for item in program["elements"]]


def test_patch_validator_rejects_arbitrary_operation_and_unsigned_patch() -> None:
    plan = compile_hyperframes_plan(_process_spec())
    program = plan.renderer_spec["scene_program_v2"]
    patch_set = VisualPatchSet(
        version="hyperframes-visual-patch-v1",
        patch_id="bad_patch",
        base_program_id=program["program_id"],
        base_program_signature=program["program_signature"],
        operations=[
            VisualPatchOperation(
                operation_id="patch_op_01",
                operation="rewrite_arbitrary_html",
                target_ids=[],
                parameters={"html": "<script>alert(1)</script>"},
                counterexample_ids=["counterexample_01"],
                evidence_ids=[],
            )
        ],
        patch_signature="tampered",
    )

    validation = validate_patch_set(
        patch_set,
        scene_program=program,
    )

    assert validation.passed is False
    assert "visual_patch_signature_mismatch" in validation.errors
    assert any(
        issue.startswith("unsupported_patch_operation")
        for issue in validation.errors
    )


def test_patch_application_rejects_grounded_copy_mutation() -> None:
    plan = compile_hyperframes_plan(_process_spec())
    program = plan.renderer_spec["scene_program_v2"]
    counterexample = VisualCounterexample(
        counterexample_id="design_01_hierarchy",
        critic="design",
        issue_type="hierarchy",
        severity="warning",
        summary="Flat hierarchy",
        expected="Clear focus",
        observed="Uniform scale",
        confidence=0.8,
        allowed_repairs=["strengthen_hierarchy"],
    )
    patch_set = plan_visual_patches(
        [counterexample],
        scene_program=program,
        round_index=1,
    )
    payload = patch_set.to_dict()
    tampered_program = {
        **program,
        "elements": [dict(item) for item in program["elements"]],
    }
    tampered_program["elements"][0]["text"] = "Invented 99% result"

    application = apply_visual_patch_set(
        payload,
        scene_program=tampered_program,
        ir=plan.ir.to_dict(),
        claim_graph=plan.claim_graph.to_dict(),
    )

    assert application.passed is False
    assert (
        "scene_program_signature_invalid_before_patch"
        in application.patch_validation.errors
    )


def test_source_asset_counterexample_reroutes_without_inventing_asset() -> None:
    plan = compile_hyperframes_plan(_process_spec())
    program = plan.renderer_spec["scene_program_v2"]
    counterexample = VisualCounterexample(
        counterexample_id="grounded_01_source",
        critic="grounded",
        issue_type="source_asset_required",
        severity="hard_failure",
        summary="Real source frame required",
        expected="Grounded UI",
        observed="No source",
        confidence=1.0,
        allowed_repairs=["bind_source_asset", "reroute_renderer"],
    )

    patch_set = plan_visual_patches(
        [counterexample],
        scene_program=program,
        round_index=1,
    )
    application = apply_visual_patch_set(
        patch_set,
        scene_program=program,
        ir=plan.ir.to_dict(),
        claim_graph=plan.claim_graph.to_dict(),
    )

    assert application.passed is True
    assert application.disposition == "reroute"
    assert application.spec_updates["reroute_renderer"] == "ffmpeg_asset"
    assert "source_asset_grounding" not in application.spec_updates


def _process_spec() -> dict:
    return {
        "visual_id": "patch_process",
        "sentence_text": (
            "The request is classified, checked against policy, then sent to a human."
        ),
        "context_text": "The handoff prevents unsupported answers.",
        "semantic_frame": {
            "steps": ["Classify request", "Check policy", "Send to human"],
            "result": "Prevent unsupported answers",
        },
        "required_labels": [
            "Classify request",
            "Check policy",
            "Send to human",
            "Prevent unsupported answers",
        ],
        "duration": 4.0,
        "composition_mode": "replace",
    }
