from __future__ import annotations

import json
from pathlib import Path

from vex_hyperframes.blueprints import CURATED_BLUEPRINTS
from vex_hyperframes.compiler import compile_hyperframes_plan
from vex_hyperframes.production_contract import production_contract_prompt_block


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "hyperframes_semantic_cases.json"


def test_curated_blueprint_library_has_multiple_strict_options_per_scene_type() -> None:
    counts: dict[str, int] = {}
    for blueprint in CURATED_BLUEPRINTS:
        counts[blueprint.scene_type] = counts.get(blueprint.scene_type, 0) + 1
        assert blueprint.required_roles
        assert blueprint.dynamic_devices
        assert blueprint.anti_patterns
        assert blueprint.stage_family.startswith("semantic_")

    assert len(CURATED_BLUEPRINTS) >= 22
    assert all(count >= 2 for count in counts.values())


def test_semantic_compiler_builds_production_contracts_for_golden_corpus() -> None:
    cases = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    for case in cases:
        plan = compile_hyperframes_plan(_spec_from_case(case))
        if case["expected_action"] == "reject":
            assert plan.passed is False
            assert plan.renderer_spec == {}
            assert plan.issues
            continue

        assert plan.passed is True, (case["case_id"], plan.issues, plan.to_dict())
        assert plan.production_contract is not None
        assert plan.production_contract.semantic_signature
        assert plan.production_contract.quality_floor >= 0.78
        assert plan.blueprint_selection.blueprint is not None
        assert plan.renderer_spec["template"].startswith("semantic_")
        assert plan.renderer_spec["qa_contract"]["required_labels"]
        assert len(plan.storyboard) >= 2
        prompt = production_contract_prompt_block(plan.production_contract)
        assert "Screenshot test:" in prompt
        assert "Semantic signature:" in prompt


def test_compiler_rejects_blueprint_when_grounded_roles_are_missing() -> None:
    plan = compile_hyperframes_plan(
        {
            "visual_id": "incomplete_decision",
            "sentence_text": "Transcript confidence determines whether rendering continues.",
            "context_text": "No low confidence path is specified.",
            "semantic_frame": {
                "decision": "Transcript confidence",
                "high_branch": "Continue rendering",
            },
            "duration": 3.0,
        }
    )

    assert plan.passed is False
    assert plan.renderer_spec == {}


def _spec_from_case(case: dict) -> dict:
    semantic_frame = dict(case.get("semantic_frame") or {})
    required = list(case.get("required_labels") or [])
    if case.get("expected_scene_type") in {"guided_process", "architecture_flow"}:
        semantic_frame["steps"] = required
    if case.get("expected_scene_type") == "grounded_interface_walkthrough":
        semantic_frame["action"] = semantic_frame.get("action") or required[-2]
        semantic_frame["result"] = semantic_frame.get("result") or required[-1]
    if case.get("expected_scene_type") == "metric_delta":
        semantic_frame["before_state"] = semantic_frame.get("before_state") or required[0]
        semantic_frame["after_state"] = semantic_frame.get("after_state") or required[-1]
    return {
        "visual_id": case["case_id"],
        "sentence_text": case["transcript"],
        "context_text": case["context"],
        "semantic_frame": semantic_frame,
        "metric_facts": case.get("metric_facts") or [],
        "required_labels": required,
        "visual_type_hint": (
            "product_ui"
            if case.get("expected_scene_type") == "grounded_interface_walkthrough"
            else ""
        ),
        "duration": 4.0,
        "composition_mode": "replace",
    }
