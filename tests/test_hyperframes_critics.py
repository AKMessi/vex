from __future__ import annotations

from vex_hyperframes.capture import build_adaptive_capture_plan, build_render_trace
from vex_hyperframes.compiler import compile_hyperframes_plan
from vex_hyperframes.counterexamples import parse_counterexamples
from vex_hyperframes.critics import (
    build_blind_critic_report,
    build_local_design_critic,
    build_local_grounded_critic,
    run_visual_critics,
)


def test_counterexample_parser_rejects_unknown_ids_and_repairs() -> None:
    plan = compile_hyperframes_plan(_process_spec())
    program = plan.renderer_spec["scene_program_v2"]
    trace = _trace(plan, program)

    parsed = parse_counterexamples(
        [
            {
                "issue_type": "missing_relation",
                "severity": "hard_failure",
                "summary": "Relation is unclear",
                "expected": "Readable route",
                "observed": "No route",
                "relation_ids": [
                    program["relations"][0]["relation_id"],
                    "relation_invented",
                ],
                "element_ids": ["element_invented"],
                "allowed_repairs": [
                    "strengthen_relation",
                    "rewrite_arbitrary_html",
                ],
                "confidence": 0.9,
            }
        ],
        critic="grounded",
        scene_program=program,
        render_trace=trace,
    )

    assert len(parsed) == 1
    assert parsed[0].relation_ids == [
        program["relations"][0]["relation_id"]
    ]
    assert parsed[0].element_ids == []
    assert parsed[0].allowed_repairs == ["strengthen_relation"]
    assert parsed[0].evidence_ids


def test_blind_critic_maps_failures_back_to_scene_ids() -> None:
    plan = compile_hyperframes_plan(_process_spec())
    program = plan.renderer_spec["scene_program_v2"]
    trace = _trace(plan, program)
    graph = plan.claim_graph.to_dict()
    missing_node = graph["nodes"][0]
    missing_relation = graph["relations"][0]

    report = build_blind_critic_report(
        {
            "available": True,
            "passed": False,
            "score": 0.32,
            "notes": "Could not decode route.",
            "missing_labels": [missing_node["label"]],
            "missing_relation_ids": [missing_relation["relation_id"]],
            "thesis_score": 0.2,
            "decoded_claim": {
                "thesis": "Several unrelated cards.",
                "unsupported_visual_claims": [],
            },
        },
        production_contract=plan.production_contract.to_dict(),
        scene_program=program,
        render_trace=trace,
    )

    assert report.passed is False
    assert any(item.element_ids for item in report.counterexamples)
    assert any(
        item.relation_ids == [missing_relation["relation_id"]]
        for item in report.counterexamples
    )
    assert any(
        item.issue_type == "ambiguous_thesis"
        for item in report.counterexamples
    )


def test_grounded_critic_requires_real_source_for_interface_visual() -> None:
    plan = compile_hyperframes_plan(_interface_spec())
    program = plan.renderer_spec["scene_program_v2"]
    trace = _trace(plan, program)

    report = build_local_grounded_critic(
        production_contract=plan.production_contract.to_dict(),
        visual_explanation_ir=plan.ir.to_dict(),
        scene_program=program,
        render_trace=trace,
        source_asset_grounding={},
    )

    assert report.passed is False
    assert any(
        item.issue_type == "source_asset_required"
        and item.severity == "hard_failure"
        for item in report.counterexamples
    )


def test_grounded_critic_rejects_unembedded_interface_asset() -> None:
    plan = compile_hyperframes_plan(_interface_spec())
    program = plan.renderer_spec["scene_program_v2"]
    trace = _trace(plan, program)

    report = build_local_grounded_critic(
        production_contract=plan.production_contract.to_dict(),
        visual_explanation_ir=plan.ir.to_dict(),
        scene_program=program,
        render_trace=trace,
        source_asset_grounding={
            "asset_path": "/approved/interface.png",
            "embedded": False,
        },
    )

    assert report.passed is False
    assert any(
        item.issue_type == "source_asset_required"
        for item in report.counterexamples
    )


def test_design_critic_emits_typed_hierarchy_counterexample() -> None:
    plan = compile_hyperframes_plan(_process_spec())
    program = {
        **plan.renderer_spec["scene_program_v2"],
        "elements": [
            {**item, "emphasis": 0.5}
            for item in plan.renderer_spec["scene_program_v2"]["elements"]
        ],
    }
    trace = _trace(plan, program)

    report = build_local_design_critic(
        scene_program=program,
        render_trace=trace,
        quality_report={"passed": True, "score": 0.91, "issues": []},
    )

    assert any(
        item.issue_type == "hierarchy"
        for item in report.counterexamples
    )
    assert all(
        set(item.allowed_repairs)
        for item in report.counterexamples
    )


def test_critic_bundle_runs_deterministically_without_vision_key(
    monkeypatch,
) -> None:
    import config

    plan = compile_hyperframes_plan(_process_spec())
    program = plan.renderer_spec["scene_program_v2"]
    trace = _trace(plan, program)
    monkeypatch.setattr(config, "GEMINI_API_KEY", None)

    bundle = run_visual_critics(
        [],
        production_contract=plan.production_contract.to_dict(),
        visual_explanation_ir=plan.ir.to_dict(),
        scene_program=program,
        render_trace=trace,
        quality_report={
            "passed": False,
            "score": 0.62,
            "issues": [
                "The render is too static for the selected motion intensity."
            ],
        },
        vision_report={"available": False, "notes": "No key."},
    )

    assert bundle.grounded.available is True
    assert bundle.design.available is True
    assert bundle.blind.available is False
    assert bundle.counterexamples
    assert bundle.score > 0.0


def _trace(plan, program: dict) -> dict:
    captures = build_adaptive_capture_plan(
        storyboard=[item.to_dict() for item in plan.storyboard],
        scene_program=program,
    )
    return build_render_trace(
        scene_program=program,
        capture_plan=captures,
        frame_paths=[],
        duration_sec=4.0,
    )


def _process_spec() -> dict:
    return {
        "visual_id": "critic_process",
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


def _interface_spec() -> dict:
    return {
        "visual_id": "critic_interface",
        "sentence_text": (
            "The editor highlights the failed shot, opens its render log, and "
            "lets the user retry only that shot."
        ),
        "context_text": (
            "The source recording contains the actual editor interface."
        ),
        "semantic_frame": {
            "screen": "Editor interface",
            "focus": "Failed shot",
            "action": "Open render log",
            "result": "Retry that shot",
        },
        "required_labels": [
            "Failed shot",
            "Render log",
            "Retry that shot",
        ],
        "visual_type_hint": "product_ui",
        "duration": 4.0,
        "composition_mode": "replace",
    }
