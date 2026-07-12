from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from vex_remotion.compiler import compile_remotion_scene_program
from vex_remotion.qa import evaluate_remotion_render


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "hyperframes_semantic_cases.json"


def _grounded_process_spec() -> dict:
    return {
        "visual_id": "process_001",
        "renderer_hint": "remotion",
        "template": "signal_network",
        "visual_intent_type": "mechanism",
        "headline": "Agents coordinate through feedback",
        "sentence_text": (
            "The planner selects a tool, the tool acts, and memory updates "
            "after the result."
        ),
        "context_text": "Agents coordinate by routing each result back into memory.",
        "semantic_frame": {
            "intuition_mode": "process_route",
            "steps": [
                "planner selects a tool",
                "the tool acts",
                "memory updates",
            ],
            "viewer_takeaway": "Agents coordinate by routing each result back into memory.",
        },
        "duration": 3.2,
        "style_pack": "product_ui",
    }


def test_grounded_process_compiles_to_signed_relation_preserving_program() -> None:
    result = compile_remotion_scene_program(
        _grounded_process_spec(),
        width=1920,
        height=1080,
        fps=30,
    )

    assert result.passed
    assert result.program is not None
    assert result.program.scene_family == "mechanism"
    assert result.program.grounding_mode == "transcript_evidence"
    assert result.program.semantic_score >= 0.9
    assert len(result.program.signature) == 64
    labels = [node.label for node in result.program.nodes]
    assert labels[:3] == [
        "planner selects a tool",
        "the tool acts",
        "memory updates",
    ]
    assert labels == [
        "planner selects a tool",
        "the tool acts",
        "memory updates",
    ]
    assert len(result.program.edges) == 2
    assert all(edge.provenance != "layout_sequence" for edge in result.program.edges)
    assert result.program.open_visual_program["signature"]
    assert result.program.open_visual_tournament["selected_program_id"]
    assert result.program.open_visual_program["elements"]


def test_scene_family_constraints_reject_metric_without_grounded_number() -> None:
    spec = _grounded_process_spec()
    spec["template"] = "metric_callout"
    spec["visual_intent_type"] = "data_proof"

    result = compile_remotion_scene_program(
        spec,
        width=1920,
        height=1080,
        fps=30,
    )

    assert result.passed
    assert result.program is not None
    assert result.program.scene_family == "mechanism"
    metric = next(item for item in result.candidate_scores if item["family"] == "metric")
    assert "no_grounded_metric" in metric["hard_violations"]


def test_invented_metric_is_rejected_before_render() -> None:
    spec = {
        **_grounded_process_spec(),
        "template": "metric_callout",
        "visual_intent_type": "data_proof",
        "metric_facts": [{"value": "99%", "label": "accuracy"}],
    }

    result = compile_remotion_scene_program(
        spec,
        width=1920,
        height=1080,
        fps=30,
    )

    assert not result.passed
    assert any(
        "numeric" in issue or "metric" in issue
        for issue in result.errors
    )


def test_portrait_program_selects_vertical_layout() -> None:
    result = compile_remotion_scene_program(
        _grounded_process_spec(),
        width=1080,
        height=1920,
        fps=30,
    )

    assert result.passed
    assert result.program is not None
    assert result.program.layout.orientation == "portrait"
    assert result.program.layout.variant == "vertical_flow"
    assert result.program.layout.content_columns == 1


def test_golden_semantic_corpus_selects_supported_scene_families() -> None:
    expected_families = {
        "metric_delta": "metric",
        "metric_intervention": "metric",
        "causal_intervention": "mechanism",
        "guided_process": "mechanism",
        "architecture_flow": "mechanism",
        "matched_state_transform": "contrast",
        "decision_branch": "contrast",
        "narrative_progression": "timeline",
        "grounded_interface_walkthrough": "interface",
        "evidence_backed_quote": "emphasis",
    }
    cases = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    for case in cases:
        result = compile_remotion_scene_program(
            {
                "visual_id": case["case_id"],
                "sentence_text": case["transcript"],
                "context_text": case["context"],
                "semantic_frame": case.get("semantic_frame") or {},
                "metric_facts": case.get("metric_facts") or [],
                "required_labels": case.get("required_labels") or [],
                "visual_type_hint": (
                    "product_ui"
                    if case.get("expected_scene_type") == "grounded_interface_walkthrough"
                    else ""
                ),
                "duration": 4.0,
                "composition_mode": "replace",
            },
            width=1280,
            height=720,
            fps=30,
        )

        if case["expected_action"] == "reject":
            assert not result.passed, case["case_id"]
            continue
        assert result.passed, (case["case_id"], result.errors)
        assert result.program is not None
        assert result.program.scene_family == expected_families[case["expected_scene_type"]]
        assert result.program.semantic_score >= 0.9
        assert all(node.value != "10% %" for node in result.program.nodes)
        if case["case_id"] == "contrast_manual_automation":
            assert "Validation gate" in result.program.annotations


def test_render_qa_accepts_motion_and_rejects_blank_frames(
    monkeypatch,
    tmp_path: Path,
) -> None:  # noqa: ANN001
    program_result = compile_remotion_scene_program(
        _grounded_process_spec(),
        width=1280,
        height=720,
        fps=30,
    )
    assert program_result.program is not None
    program = program_result.program.to_dict()

    def animated_extract(_video, output, *, time_sec):  # noqa: ANN001
        frame = np.full((180, 320, 3), (12, 20, 28), dtype=np.uint8)
        offset = int(time_sec * 24) % 90
        frame[35:145, 30 + offset : 170 + offset] = (238, 241, 245)
        frame[58:122, 55 + offset : 110 + offset] = (225, 29, 72)
        Image.fromarray(frame).save(output)
        return True, ""

    monkeypatch.setattr("vex_remotion.qa._extract_frame", animated_extract)
    good = evaluate_remotion_render(
        tmp_path / "good.mp4",
        program,
        job_dir=tmp_path / "good",
    )
    assert good.passed
    assert good.score >= 0.58
    assert len(good.metrics["sample_fractions"]) >= 5
    assert good.metrics["maximum_motion_area"] > 0.0

    def blank_extract(_video, output, *, time_sec):  # noqa: ANN001, ARG001
        Image.new("RGB", (320, 180), color=(12, 20, 28)).save(output)
        return True, ""

    monkeypatch.setattr("vex_remotion.qa._extract_frame", blank_extract)
    blank = evaluate_remotion_render(
        tmp_path / "blank.mp4",
        program,
        job_dir=tmp_path / "blank",
    )
    assert not blank.passed
    assert "remotion_render_final_frame_is_visually_empty" in blank.issues
    assert "remotion_render_has_no_meaningful_motion" in blank.issues
