from __future__ import annotations

import json
from pathlib import Path

from vex_hyperframes.composer import build_composition
from vex_hyperframes.compiler import compile_hyperframes_plan
from vex_hyperframes.validator import validate_composition_html
from vex_hyperframes.variants import build_variants


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "hyperframes_semantic_cases.json"


def test_visual_world_composition_replaces_global_card_and_grid_renderer() -> None:
    plan = compile_hyperframes_plan(_spec(_case("causal_passive_learning")))
    variant = build_variants(plan.renderer_spec)[0]
    composition = build_composition(
        variant.spec,
        width=1280,
        height=720,
        fps=30,
    )
    report = validate_composition_html(
        composition.html,
        expected_width=1280,
        expected_height=720,
        expected_duration=4.0,
    )

    assert report.valid, report.errors
    assert composition.metadata["stage"]["generation_mode"] == (
        "typed_visual_world_program"
    )
    assert "visual-world-canvas" in composition.html
    assert 'id="hf-bg-grid"' not in composition.html
    assert "scene-v2-element" not in composition.html
    assert composition.metadata["stage"]["world_signature"]
    assert composition.metadata["stage"]["object_coverage"] == 1.0


def test_proof_tournament_executes_visually_distinct_medium_compilers() -> None:
    plan = compile_hyperframes_plan(_spec(_case("causal_passive_learning")))
    variants = build_variants(plan.renderer_spec)
    markers: set[str] = set()

    for variant in variants:
        composition = build_composition(
            variant.spec,
            width=1280,
            height=720,
            fps=30,
        )
        medium = variant.spec["visual_world_program"]["medium_family"]
        markers.add(medium)
        assert f'data-medium-family="{medium}"' in composition.html
        assert (
            composition.metadata["visual_world_program"]["medium_family"]
            == medium
        )

    assert markers == {
        "data_sculpture",
        "editorial_collage",
        "kinetic_typography",
        "spatial_metaphor",
    }


def test_visual_world_preserves_traceable_object_relation_and_evidence_ids() -> None:
    plan = compile_hyperframes_plan(
        _spec(_case("architecture_request_lifecycle"))
    )
    variant = build_variants(plan.renderer_spec)[0]
    composition = build_composition(
        variant.spec,
        width=1280,
        height=720,
        fps=30,
    )

    assert 'data-element-id="' in composition.html
    assert 'data-object-id="' in composition.html
    assert 'data-evidence-ids="' in composition.html
    assert 'data-relation-id="' in composition.html
    assert "https://" not in composition.html
    assert "requestAnimationFrame" not in composition.html


def test_interface_world_uses_product_surfaces_only_for_interface_semantics() -> None:
    plan = compile_hyperframes_plan(_spec(_case("interface_real_states")))
    product_variant = next(
        item
        for item in build_variants(plan.renderer_spec)
        if item.spec["visual_world_program"]["medium_family"]
        == "product_interface"
    )
    composition = build_composition(
        product_variant.spec,
        width=1280,
        height=720,
        fps=30,
    )

    assert "vw-ui-shell" in composition.html
    assert product_variant.spec["visual_world_program"]["card_policy"] == "allowed"
    assert composition.metadata["stage"]["panel_ratio_target"] > 0.4


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
            if case.get("expected_scene_type")
            == "grounded_interface_walkthrough"
            else ""
        ),
        "duration": 4.0,
        "composition_mode": "replace",
    }
