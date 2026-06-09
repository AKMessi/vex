from __future__ import annotations

import json
from pathlib import Path

import pytest

from vex_hyperframes.authoring import (
    build_bespoke_program,
    compile_bespoke_stage,
    validate_bespoke_program,
)
from vex_hyperframes.compiler import compile_hyperframes_plan
from vex_hyperframes.composer import build_composition
from vex_hyperframes.safety import validate_authored_html_safety
from vex_hyperframes.validator import validate_composition_html


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "hyperframes_semantic_cases.json"


def test_typed_bespoke_program_compiles_grounded_deterministic_html() -> None:
    case = _case("causal_passive_learning")
    plan = compile_hyperframes_plan(_spec(case))
    assert plan.blueprint_selection.blueprint is not None
    program = build_bespoke_program(
        plan.ir,
        blueprint_id=plan.blueprint_selection.blueprint.blueprint_id,
    )
    validation = validate_bespoke_program(program, plan.ir)
    compiled = compile_bespoke_stage(program, plan.ir)

    assert validation.passed is True
    assert validation.grounded_copy_ratio == 1.0
    assert compiled.metadata["primitive_count"] == len(plan.ir.objects)
    assert "Tutorial watching hides gaps" in compiled.html
    assert "https://" not in compiled.html
    assert "<script" not in compiled.html
    assert "onclick=" not in compiled.html


def test_bespoke_program_rejects_copy_not_present_in_grounded_ir() -> None:
    case = _case("interface_real_states")
    plan = compile_hyperframes_plan(_spec(case))
    assert plan.blueprint_selection.blueprint is not None
    payload = build_bespoke_program(
        plan.ir,
        blueprint_id=plan.blueprint_selection.blueprint.blueprint_id,
    ).to_dict()
    payload["primitives"][0]["text"] = "92% success score"

    validation = validate_bespoke_program(payload, plan.ir)

    assert validation.passed is False
    assert any(
        issue.startswith("primitive_copy_is_not_grounded")
        for issue in validation.errors
    )
    with pytest.raises(ValueError, match="Unsafe bespoke scene program"):
        compile_bespoke_stage(payload, plan.ir)


@pytest.mark.parametrize(
    ("fragment", "expected_error"),
    [
        (
            '<style>.x{color:red}</style><iframe src="https://example.com"></iframe>',
            "forbidden_html_tag:iframe",
        ),
        (
            '<style>.x{color:red}</style><div onclick="fetch(\'https://example.com\')">x</div>',
            "event_handler_attribute:onclick",
        ),
        (
            "<style>@import 'https://example.com/x.css';</style><div>x</div>",
            "remote_url",
        ),
        (
            "<style>.x{color:red}</style><script>eval('1')</script>",
            "forbidden_html_tag:script",
        ),
    ],
)
def test_authored_html_safety_rejects_executable_or_remote_content(
    fragment: str,
    expected_error: str,
) -> None:
    report = validate_authored_html_safety(fragment)

    assert report.safe is False
    assert expected_error in report.errors


def test_semantic_compiler_can_emit_bespoke_scene_program_for_composer() -> None:
    case = _case("architecture_request_lifecycle")
    spec = {**_spec(case), "hyperframes_authoring_mode": "bespoke"}
    plan = compile_hyperframes_plan(spec)

    assert plan.passed is True
    assert plan.renderer_spec["bespoke_scene_program"]["version"].endswith("-v1")
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

    assert validation.valid, validation.errors
    assert composition.metadata["stage"]["generation_mode"] == "typed_bespoke_scene_program"
    assert composition.metadata["stage"]["grounded_copy_ratio"] == 1.0


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
        "visual_type_hint": (
            "product_ui"
            if case.get("expected_scene_type") == "grounded_interface_walkthrough"
            else ""
        ),
        "duration": 4.0,
        "composition_mode": "replace",
    }
