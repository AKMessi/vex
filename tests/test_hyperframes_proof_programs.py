from __future__ import annotations

import json
from pathlib import Path

from vex_hyperframes.compiler import compile_hyperframes_plan
from vex_hyperframes.composer import build_composition
from vex_hyperframes.proof_program import validate_visual_proof_tournament
from vex_hyperframes.variants import build_variants


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "hyperframes_semantic_cases.json"


def test_compiler_builds_signed_structural_tournaments_for_renderable_corpus() -> None:
    cases = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    for case in cases:
        if case["expected_action"] != "render":
            continue
        plan = compile_hyperframes_plan(_spec_from_case(case))

        assert plan.passed is True, (case["case_id"], plan.issues)
        assert plan.proof_tournament_validation.passed is True
        assert len(plan.proof_tournament.programs) == 4
        assert plan.proof_tournament_validation.distinct_encodings >= 3
        assert plan.renderer_spec["visual_proof_tournament"]["tournament_signature"]
        assert len(plan.renderer_spec["visual_proof_programs"]) == 4


def test_proof_programs_bind_distinct_blueprints_to_distinct_contracts() -> None:
    plan = compile_hyperframes_plan(
        _spec_from_case(_case("causal_passive_learning"))
    )
    programs = plan.proof_tournament.programs
    signatures_by_blueprint: dict[str, set[str]] = {}
    for program in programs:
        signatures_by_blueprint.setdefault(program.blueprint_id, set()).add(
            str(program.production_contract["semantic_signature"])
        )

    assert len(signatures_by_blueprint) >= 2
    assert len(
        {
            next(iter(signatures))
            for signatures in signatures_by_blueprint.values()
        }
    ) == len(signatures_by_blueprint)


def test_variants_execute_proof_programs_instead_of_cosmetic_indices() -> None:
    plan = compile_hyperframes_plan(
        _spec_from_case(_case("process_support_handoff"))
    )
    variants = build_variants(plan.renderer_spec, default_count=1)

    assert len(variants) == 4
    assert len({item.variant_id for item in variants}) == 4
    assert len({item.spec["proof_encoding"] for item in variants}) >= 3
    assert all(item.spec["proof_program_id"] == item.variant_id for item in variants)
    assert all(item.spec["hyperframes_production_contract"] for item in variants)


def test_composer_stamps_executable_proof_encoding_and_metadata() -> None:
    plan = compile_hyperframes_plan(
        _spec_from_case(_case("decision_quality_gate"))
    )
    variant = build_variants(plan.renderer_spec)[1]
    composition = build_composition(
        variant.spec,
        width=1280,
        height=720,
        fps=30,
    )
    encoding_class = variant.spec["proof_encoding"].replace("_", "-")

    assert f"proof-encoding-{encoding_class}" in composition.html
    assert f'data-proof-program-id="{variant.variant_id}"' in composition.html
    assert composition.metadata["proof_program_id"] == variant.variant_id
    assert composition.metadata["stage"]["proof_encoding"] == variant.spec["proof_encoding"]


def test_tournament_signature_rejects_candidate_tampering() -> None:
    plan = compile_hyperframes_plan(
        _spec_from_case(_case("architecture_request_lifecycle"))
    )
    payload = plan.proof_tournament.to_dict()
    payload["programs"][0]["encoding_family"] = "split_register"

    validation = validate_visual_proof_tournament(payload)

    assert validation.passed is False
    assert "proof_tournament_signature_mismatch" in validation.errors


def _case(case_id: str) -> dict:
    return next(
        item
        for item in json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        if item["case_id"] == case_id
    )


def _spec_from_case(case: dict) -> dict:
    semantic_frame = dict(case.get("semantic_frame") or {})
    required = list(case.get("required_labels") or [])
    if case.get("expected_scene_type") in {"guided_process", "architecture_flow"}:
        semantic_frame["steps"] = required
    if case.get("expected_scene_type") == "grounded_interface_walkthrough":
        semantic_frame["action"] = semantic_frame.get("action") or required[-2]
        semantic_frame["result"] = semantic_frame.get("result") or required[-1]
    if case.get("expected_scene_type") == "metric_delta":
        semantic_frame["before_state"] = (
            semantic_frame.get("before_state") or required[0]
        )
        semantic_frame["after_state"] = (
            semantic_frame.get("after_state") or required[-1]
        )
    return {
        "visual_id": case["case_id"],
        "sentence_text": case["transcript"],
        "context_text": case["context"],
        "semantic_frame": semantic_frame,
        "metric_facts": case.get("metric_facts") or [],
        "required_labels": required,
        "visual_type_hint": (
            "product_ui"
            if case.get("expected_scene_type")
            == "grounded_interface_walkthrough"
            else ""
        ),
        "duration": 4.0,
        "composition_mode": "replace",
    }
