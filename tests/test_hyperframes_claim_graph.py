from __future__ import annotations

import json
from pathlib import Path

from visual_explanation import build_visual_explanation_ir
from vex_hyperframes.claim_graph import (
    CLAIM_GRAPH_VERSION,
    build_visual_claim_graph,
    validate_visual_claim_graph,
)
from vex_hyperframes.compiler import compile_hyperframes_plan


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "hyperframes_semantic_cases.json"


def test_claim_graph_builds_relations_and_questions_for_renderable_corpus() -> None:
    cases = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    for case in cases:
        if case["expected_action"] != "render":
            continue
        ir = build_visual_explanation_ir(_spec_from_case(case))
        graph = build_visual_claim_graph(ir)
        validation = validate_visual_claim_graph(graph)

        assert graph.version == CLAIM_GRAPH_VERSION
        assert validation.passed is True, (
            case["case_id"],
            validation.errors,
            graph.to_dict(),
        )
        assert graph.graph_signature
        assert graph.nodes
        assert graph.questions
        if case["expected_scene_type"] not in {
            "evidence_backed_quote",
            "metric_proof",
        }:
            assert graph.relations
            assert validation.relation_node_coverage >= 0.66


def test_decision_claim_graph_preserves_both_exclusive_branches() -> None:
    case = _case("decision_quality_gate")
    graph = build_visual_claim_graph(
        build_visual_explanation_ir(_spec_from_case(case))
    )

    relation_types = {item.relation_type for item in graph.relations}
    answers = {item.expected_answer for item in graph.questions}

    assert relation_types == {"branches_to_low", "branches_to_high"}
    assert {"Request review", "Continue rendering"}.issubset(answers)


def test_compiler_binds_claim_graph_into_signed_production_contract() -> None:
    plan = compile_hyperframes_plan(
        _spec_from_case(_case("causal_passive_learning"))
    )

    assert plan.passed is True
    assert plan.claim_graph_validation.passed is True
    assert plan.production_contract is not None
    assert plan.production_contract.claim_graph_signature
    assert plan.production_contract.required_relation_ids
    assert plan.production_contract.proof_questions
    assert (
        plan.renderer_spec["visual_claim_graph"]["graph_signature"]
        == plan.production_contract.claim_graph_signature
    )
    assert plan.renderer_spec["qa_contract"]["required_relation_ids"]


def test_claim_graph_signature_detects_contract_tampering() -> None:
    graph = build_visual_claim_graph(
        build_visual_explanation_ir(
            _spec_from_case(_case("process_support_handoff"))
        )
    ).to_dict()
    graph["relations"][0]["target_id"] = graph["relations"][-1]["target_id"]

    validation = validate_visual_claim_graph(graph)

    assert validation.passed is False
    assert "claim_graph_signature_mismatch" in validation.errors


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
