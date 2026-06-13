from __future__ import annotations

import json
from pathlib import Path

from visual_explanation import (
    VISUAL_EXPLANATION_VERSION,
    build_visual_explanation_ir,
    validate_visual_explanation_ir,
    visual_explanation_prompt_block,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "hyperframes_semantic_cases.json"


def test_visual_explanation_ir_builds_grounded_contracts_for_golden_corpus() -> None:
    cases = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    rendered = 0
    rejected = 0

    for case in cases:
        spec = _spec_from_case(case)
        ir = build_visual_explanation_ir(spec)
        validation = validate_visual_explanation_ir(ir)

        assert ir.version == VISUAL_EXPLANATION_VERSION
        assert validation.passed is True, (case["case_id"], validation.errors, ir.to_dict())
        if case["expected_action"] == "render":
            rendered += 1
            assert ir.render_policy == "render", (case["case_id"], ir.rejection_reasons)
            assert ir.scene_type == case["expected_scene_type"]
            assert validation.grounded_fact_ratio >= 0.95
            assert len(ir.objects) >= 1
            assert len(ir.beats) >= 2
            assert set(case["required_labels"]).issubset(set(ir.required_labels))
        else:
            rejected += 1
            assert ir.render_policy == "reject"
            assert ir.scene_type == "none"
            assert ir.rejection_reasons

    assert rendered >= 10
    assert rejected >= 2


def test_visual_explanation_ir_rejects_ungrounded_metric() -> None:
    ir = build_visual_explanation_ir(
        {
            "visual_id": "invented_metric",
            "sentence_text": "The interface becomes easier to use.",
            "context_text": "No measured percentage was provided.",
            "headline": "Interface improvement",
            "metric_facts": [{"value": "92%", "label": "Success score"}],
            "semantic_frame": {
                "viewer_takeaway": "Interface improvement",
            },
            "duration": 3.0,
        }
    )
    validation = validate_visual_explanation_ir(ir)

    assert ir.render_policy == "reject"
    assert "numeric_fact_lacks_source_provenance" in ir.rejection_reasons
    assert validation.passed is True
    assert validation.invented_numeric_facts == ["92%"]


def test_visual_explanation_ir_references_valid_facts_and_objects() -> None:
    ir = build_visual_explanation_ir(
        {
            "visual_id": "process",
            "sentence_text": "The request is classified, checked against policy, then sent to a human.",
            "context_text": "The handoff prevents unsupported answers.",
            "semantic_frame": {
                "steps": ["Classify request", "Check policy", "Send to human"],
                "result": "Prevent unsupported answers",
            },
            "duration": 4.0,
        }
    )
    validation = validate_visual_explanation_ir(ir)
    prompt = visual_explanation_prompt_block(ir)

    assert validation.passed is True
    assert ir.render_policy == "render"
    assert ir.scene_type == "guided_process"
    assert "Required visible objects" in prompt
    assert "Required motion beats" in prompt
    assert "Classify request" in prompt


def test_visual_explanation_ir_salvages_consistent_partition_from_noisy_asr() -> None:
    ir = build_visual_explanation_ir(_failed_attention_bundle_spec())
    validation = validate_visual_explanation_ir(ir)

    assert validation.passed is True
    assert ir.render_policy == "render"
    assert ir.scene_type == "set_partition"
    assert ir.rejection_reasons == []
    assert ir.relations
    assert ir.metadata["executable_model"]["valid"] is True
    assert ir.metadata["executable_model"]["input_count"] == 32
    assert ir.metadata["executable_model"]["group_size"] == 4
    assert ir.metadata["executable_model"]["group_count"] == 8
    assert "102" not in " ".join(ir.required_labels)
    assert "Ground the spoken claim" not in ir.thesis


def test_visual_explanation_ir_emits_relation_level_provenance() -> None:
    cases = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    case = next(
        item for item in cases if item["case_id"] == "data_threshold_latency"
    )
    ir = build_visual_explanation_ir(_spec_from_case(case))

    assert ir.render_policy == "render"
    assert ir.metadata["executable_model"]["valid"] is True
    assert ir.relations
    assert all(item.provenance == "executable_visual_model" for item in ir.relations)
    assert all(item.evidence_ids for item in ir.relations)


def _failed_attention_bundle_spec() -> dict:
    return {
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
            "intuition_mode": "metric_proof",
            "before_state": "Full attention would need 32",
            "after_state": "After compression 4",
            "cause": "After compression",
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


def _spec_from_case(case: dict) -> dict:
    semantic_frame = dict(case.get("semantic_frame") or {})
    expected = list(case.get("required_labels") or [])
    if case.get("expected_scene_type") in {"guided_process", "architecture_flow"}:
        semantic_frame["steps"] = expected
    if case.get("expected_scene_type") == "grounded_interface_walkthrough":
        semantic_frame["action"] = semantic_frame.get("action") or expected[-2]
        semantic_frame["result"] = semantic_frame.get("result") or expected[-1]
    if case.get("expected_scene_type") == "metric_delta":
        semantic_frame["before_state"] = semantic_frame.get("before_state") or expected[0]
        semantic_frame["after_state"] = semantic_frame.get("after_state") or expected[-1]
    return {
        "visual_id": case["case_id"],
        "sentence_text": case["transcript"],
        "context_text": case["context"],
        "semantic_frame": semantic_frame,
        "metric_facts": case.get("metric_facts") or [],
        "required_labels": expected,
        "visual_type_hint": "product_ui" if case.get("expected_scene_type") == "grounded_interface_walkthrough" else "",
        "duration": 4.0,
        "composition_mode": "replace",
    }
