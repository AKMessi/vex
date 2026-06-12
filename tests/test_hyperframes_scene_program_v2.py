from __future__ import annotations

import json
from pathlib import Path

from vex_hyperframes.capture import build_adaptive_capture_plan, build_render_trace
from vex_hyperframes.compiler import compile_hyperframes_plan
from vex_hyperframes.composer import build_composition
from vex_hyperframes.scene_program import (
    SCENE_PROGRAM_VERSION,
    compile_scene_stage,
    validate_scene_program,
)
from vex_hyperframes.variants import build_variants


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "hyperframes_semantic_cases.json"


def test_scene_program_v2_preserves_claim_graph_relations_and_evidence() -> None:
    plan = compile_hyperframes_plan(_spec(_case("causal_passive_learning")))
    program = plan.renderer_spec["visual_proof_programs"][0]["scene_program_v2"]
    graph = plan.claim_graph.to_dict()

    validation = validate_scene_program(
        program,
        ir=plan.ir,
        claim_graph=graph,
    )

    assert validation.passed is True, validation.errors
    assert program["version"] == SCENE_PROGRAM_VERSION
    assert validation.object_coverage == 1.0
    assert validation.relation_coverage == 1.0
    assert {
        item["relation_id"] for item in program["relations"]
    } == {
        item["relation_id"]
        for item in graph["relations"]
        if item["required"]
    }
    assert all(item["evidence_ids"] for item in program["relations"])


def test_scene_program_v2_compiles_traceable_safe_html() -> None:
    plan = compile_hyperframes_plan(_spec(_case("architecture_request_lifecycle")))
    variant = build_variants(plan.renderer_spec)[0]
    program = variant.spec["scene_program_v2"]

    compiled = compile_scene_stage(
        program,
        ir=plan.ir,
        claim_graph=plan.claim_graph,
    )
    composition = build_composition(
        variant.spec,
        width=1280,
        height=720,
        fps=30,
    )

    assert compiled.metadata["object_coverage"] == 1.0
    assert compiled.metadata["relation_coverage"] == 1.0
    assert 'data-element-id="' in compiled.html
    assert 'data-relation-id="' in compiled.html
    assert 'data-evidence-ids="' in compiled.html
    assert "https://" not in compiled.html
    assert composition.metadata["stage"]["generation_mode"] == "typed_scene_program_v2"
    assert composition.metadata["scene_program_v2"]["program_signature"]


def test_scene_program_v2_rejects_unsigned_layout_mutation() -> None:
    plan = compile_hyperframes_plan(_spec(_case("decision_quality_gate")))
    program = dict(plan.renderer_spec["scene_program_v2"])
    program["elements"] = [dict(item) for item in program["elements"]]
    program["elements"][0]["x"] = 0.99

    validation = validate_scene_program(
        program,
        ir=plan.ir,
        claim_graph=plan.claim_graph,
    )

    assert validation.passed is False
    assert "scene_program_signature_mismatch" in validation.errors
    assert any(
        issue.startswith("element_exceeds_horizontal_bounds")
        for issue in validation.errors
    )


def test_adaptive_capture_plan_covers_relations_beats_and_final_hold() -> None:
    plan = compile_hyperframes_plan(_spec(_case("process_support_handoff")))
    program = plan.renderer_spec["scene_program_v2"]

    captures = build_adaptive_capture_plan(
        storyboard=[item.to_dict() for item in plan.storyboard],
        scene_program=program,
        max_frames=8,
    )
    trace = build_render_trace(
        scene_program=program,
        capture_plan=captures,
        frame_paths=[
            Path(f"/tmp/frame_{index:02d}.png")
            for index in range(len(captures))
        ],
        duration_sec=4.0,
    )

    assert captures[-1].reason == "final_resolved_hold"
    assert any(item.reason == "required_relation_visible" for item in captures)
    assert any(item.reason == "semantic_beat_resolved" for item in captures)
    assert len(captures) <= 8
    assert trace["program_signature"] == program["program_signature"]
    assert all(item["frame_path"] for item in trace["captures"])
    assert {item["relation_id"] for item in trace["relations"]} == {
        item["relation_id"] for item in program["relations"]
    }


def _case(case_id: str) -> dict:
    return next(
        item
        for item in json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        if item["case_id"] == case_id
    )


def _spec(case: dict) -> dict:
    semantic_frame = dict(case.get("semantic_frame") or {})
    required = list(case.get("required_labels") or [])
    if case.get("expected_scene_type") in {"guided_process", "architecture_flow"}:
        semantic_frame["steps"] = required
    if case.get("expected_scene_type") == "grounded_interface_walkthrough":
        semantic_frame["action"] = semantic_frame.get("action") or required[-2]
        semantic_frame["result"] = semantic_frame.get("result") or required[-1]
    return {
        "visual_id": case["case_id"],
        "sentence_text": case["transcript"],
        "context_text": case["context"],
        "semantic_frame": semantic_frame,
        "metric_facts": case.get("metric_facts") or [],
        "required_labels": required,
        "duration": 4.0,
        "composition_mode": "replace",
    }
