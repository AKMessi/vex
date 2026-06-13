from __future__ import annotations

import json
from pathlib import Path

import pytest

from vex_hyperframes.blueprints import CURATED_BLUEPRINTS
from vex_hyperframes.compiler import compile_hyperframes_plan
from vex_hyperframes.composer import build_composition
from vex_hyperframes.evaluation import visible_text_from_html
from vex_hyperframes.production_contract import production_contract_prompt_block
from vex_hyperframes.validator import validate_composition_html


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


def test_semantic_composer_builds_valid_grounded_scenes_for_golden_corpus() -> None:
    cases = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    for case in cases:
        if case["expected_action"] != "render":
            continue
        plan = compile_hyperframes_plan(_spec_from_case(case))
        assert plan.passed is True, (case["case_id"], plan.issues)

        composition = build_composition(
            plan.renderer_spec,
            width=1280,
            height=720,
            fps=30,
        )
        validation = validate_composition_html(
            composition.html,
            expected_width=1280,
            expected_height=720,
            expected_duration=4.0,
        )
        visible_copy = visible_text_from_html(composition.html)

        assert validation.valid, (case["case_id"], validation.errors)
        assert composition.metadata["semantic_signature"]
        assert composition.metadata["semantic_blueprint_id"]
        assert composition.metadata["stage"]["stage_family"].startswith("semantic_")
        assert all(
            forbidden.lower() not in visible_copy.lower()
            for forbidden in case.get("forbidden_labels") or []
        )


def test_semantic_interface_never_fabricates_progress_percentages() -> None:
    case = next(
        item
        for item in json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        if item["case_id"] == "interface_real_states"
    )
    plan = compile_hyperframes_plan(_spec_from_case(case))
    composition = build_composition(plan.renderer_spec, width=1280, height=720, fps=30)
    visible_copy = visible_text_from_html(composition.html)

    assert "82%" not in visible_copy
    assert "87%" not in visible_copy
    assert "92%" not in visible_copy
    assert composition.metadata["stage"]["synthetic_metrics"] == 0


def test_composer_rejects_unknown_templates_instead_of_quote_fallback() -> None:
    with pytest.raises(ValueError, match="Unsupported HyperFrames template"):
        build_composition(
            {
                "visual_id": "unknown",
                "template": "generic_magic_diagram",
                "duration": 3.0,
            },
            width=1280,
            height=720,
            fps=30,
        )


def test_compiler_replaces_failed_attention_cards_with_partition_geometry() -> None:
    plan = compile_hyperframes_plan(
        {
            "visual_id": "visual_001",
            "sentence_text": (
                "Full attention would need 32 square that is 102 for comparisons."
            ),
            "context_text": (
                "Full attention would need 32 square that is 102 for comparisons. "
                "After compression, 4 is to 1, 8 compressed blocks remain."
            ),
            "headline": (
                "Ground the spoken claim in concrete evidence the viewer can track"
            ),
            "semantic_frame": {
                "before_state": "Full attention would need 32",
                "after_state": "After compression 4",
                "effect": "Full attention would need 32 square",
                "mental_model": (
                    "Ground the spoken claim in concrete evidence the viewer can track."
                ),
                "viewer_takeaway": "Full attention would need 32 square",
            },
            "metric_facts": [
                {
                    "value": "32",
                    "label": "Full attention would need 32 square that",
                },
                {
                    "value": "102",
                    "label": "Full attention would need 32 square that",
                },
            ],
            "duration": 4.34,
            "composition_mode": "replace",
        }
    )

    assert plan.passed is True
    assert plan.ir.scene_type == "set_partition"
    assert plan.renderer_spec["template"] == "semantic_partition"
    assert plan.renderer_spec["headline"] == "32 tokens become 8 blocks"
    assert "102" not in " ".join(plan.renderer_spec["steps"])
    composition = build_composition(
        plan.renderer_spec,
        width=1280,
        height=720,
        fps=30,
    )
    visible_copy = visible_text_from_html(composition.html)

    assert composition.metadata["stage"]["generation_mode"] == (
        "executable_partition_geometry"
    )
    assert "32 original tokens" in visible_copy
    assert "4 tokens per block" in visible_copy
    assert "8 compressed blocks" in visible_copy
    assert "102" not in visible_copy


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
