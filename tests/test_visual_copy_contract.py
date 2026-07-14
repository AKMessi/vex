from __future__ import annotations

import copy

from visual_copy_contract import (
    display_copy_issues,
    metric_value_is_visual_measure,
    validate_visual_copy_contract,
)
from visual_opportunity import _grounded_semantic_frame
from visual_explanation import build_visual_explanation_ir
from visual_intelligence import _extract_metric_facts
from visual_skill_graph import _copy
from vex_visuals.open_visual_program import (
    build_open_visual_program_candidates,
    sign_open_visual_program,
    validate_open_visual_program,
)


FAILED_BUNDLE_SOURCE = (
    "about the hype of gated Delta net Quen 3.6 and 3.7 models are heavily "
    "based on the Gated Delta net architecture. So I'm going to explain you "
    "gated Delta net and shot in a very simple manner Standard attention is "
    "powerful But it's like a student who reads everything"
)


def test_model_version_identifiers_are_not_promoted_to_metrics() -> None:
    assert _extract_metric_facts(FAILED_BUNDLE_SOURCE) == []
    assert not metric_value_is_visual_measure(
        "3.6",
        "About the hype of gated Delta net Quen",
        FAILED_BUNDLE_SOURCE,
    )


def test_failed_bundle_copy_is_rejected_before_rendering() -> None:
    ir = build_visual_explanation_ir(
        {
            "visual_id": "failed_bundle_visual_001",
            "sentence_text": FAILED_BUNDLE_SOURCE,
            "context_text": FAILED_BUNDLE_SOURCE,
            "display_title": "Attention Is All You Need",
            "semantic_frame": {
                "viewer_takeaway": "About the hype of gated Delta net Quen",
            },
            "metric_facts": [
                {"value": "3.6", "label": "About the hype of gated Delta net Quen"},
                {"value": "3.7", "label": "About the hype of gated Delta net Quen"},
            ],
            "duration": 7.0,
        }
    )

    contract = ir.metadata["visual_copy_contract"]
    assert ir.render_policy == "reject"
    assert ir.scene_type == "none"
    assert all(item["text"] != "Attention Is All You Need" for item in contract["items"])
    assert not any(item.fact_type == "metric" for item in ir.facts)


def test_dangling_asr_steps_cannot_become_visible_copy() -> None:
    source = (
        "every new page you add multiplies the cost As context grows but It "
        "needs unlimited infinite compute gated Delta net takes a"
    )
    ir = build_visual_explanation_ir(
        {
            "visual_id": "failed_bundle_visual_002",
            "sentence_text": source,
            "context_text": source,
            "semantic_frame": {
                "steps": [
                    "every new page you add multiplies the cost As",
                    "but It needs unlimited infinite compute",
                    "gated Delta net takes a",
                ],
                "viewer_takeaway": "gated Delta net takes a",
            },
            "duration": 6.0,
        }
    )

    assert ir.render_policy == "reject"
    assert "fragmented_semantic_labels" in ir.rejection_reasons
    assert "visual_copy_contract_rejected" in ir.rejection_reasons
    assert display_copy_issues("gated Delta net takes a", role="label")


def test_open_program_copy_permissions_are_binding_local() -> None:
    ir = build_visual_explanation_ir(
        {
            "visual_id": "binding_local",
            "sentence_text": (
                "The request enters the API, authentication checks policy, and "
                "the renderer returns the response."
            ),
            "context_text": "Each stage owns one transition.",
            "semantic_frame": {
                "steps": ["Request enters API", "Authentication checks policy", "Renderer returns response"],
                "result": "Renderer returns response",
            },
            "duration": 4.0,
        }
    )
    payload = ir.to_dict()
    programs = build_open_visual_program_candidates(
        payload,
        visual_id="binding_local",
        width=1920,
        height=1080,
        duration_sec=4.0,
        fps=60,
        candidate_count=1,
    )
    assert programs
    program = copy.deepcopy(programs[0])
    title = next(item for item in program["elements"] if item["element_id"] == "title")
    object_copy = next(
        item
        for item in payload["metadata"]["visual_copy_contract"]["items"]
        if item["role"] == "object_label" and item["binding_id"] != title["binding"]["id"]
    )
    title["text"] = object_copy["text"]
    program = sign_open_visual_program(program)

    validation = validate_open_visual_program(program, ir=payload)

    assert not validation.passed
    assert "ungrounded_element_copy:title" in validation.errors


def test_copy_fitting_never_emits_a_truncated_fragment() -> None:
    long_copy = "Absolutely not so the gated Delta net paper fixed it by replacing"

    assert _copy(long_copy, max_words=6, max_chars=42) == ""
    assert _copy("Write only the correction", max_words=6, max_chars=42) == (
        "Write only the correction"
    )


def test_visual_copy_contract_fails_closed_after_mutation() -> None:
    ir = build_visual_explanation_ir(
        {
            "visual_id": "signed_copy",
            "sentence_text": "The planner routes the request to the renderer.",
            "context_text": "The renderer returns the finished frame.",
            "semantic_frame": {
                "steps": [
                    "planner routes the request",
                    "renderer returns the finished frame",
                ],
                "result": "renderer returns the finished frame",
            },
            "duration": 3.0,
        }
    )
    contract = copy.deepcopy(ir.metadata["visual_copy_contract"])
    assert validate_visual_copy_contract(contract) == []

    contract["items"][0]["text"] = "Invented hardcoded label"

    assert "visual_copy_contract_signature_invalid" in validate_visual_copy_contract(
        contract
    )


def test_failed_bundle_asr_fragments_are_rejected_as_visible_copy() -> None:
    assert "trailing_fragment" in display_copy_issues(
        "write only that difference that's",
        role="title",
    )
    assert "asr_discourse_splice" in display_copy_issues(
        "write gate two different vectors No correlation",
    )
    assert "placeholder_noun" in display_copy_issues("then the node pad thing")


def test_failed_bundle_planner_keeps_only_complete_causal_steps() -> None:
    correction_source = (
        "it's wrong He writes only the correction The Delta the difference not "
        "the whole thing again This makes the memory precise Read what's currently "
        "stored for this key compute the difference to the new value write only "
        "that difference that's"
    )
    correction = _grounded_semantic_frame([], correction_source, [])

    assert correction["steps"] == [
        "makes the memory precise",
        "Read what's currently stored for this key",
    ]
    assert correction["result"] == "compute the difference to the new value"
    assert "that's" not in str(correction)

    noisy_gate_source = (
        "gate erase gate and write gate two different vectors No correlation "
        "between them And then the node pad thing and then the output tokens "
        "decay gate channel wise fade each key channel forgets at its own Learned "
        "rate topic shifted old notes dim automatically Erase"
    )
    assert _grounded_semantic_frame([], noisy_gate_source, []) == {}
