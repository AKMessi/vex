from __future__ import annotations

from vex_hyperframes import build_design_ir
from vex_hyperframes.composer import build_composition


def test_design_ir_selects_data_proof_for_metric_visuals() -> None:
    ir = build_design_ir(
        {
            "template": "data_journey",
            "visual_type_hint": "data_graphic",
            "headline": "Retention Doubles",
            "supporting_lines": ["Build first", "Study blockers"],
            "duration": 3.2,
            "importance": 0.86,
        },
        width=1920,
        height=1080,
        fps=30,
    )

    assert ir.archetype == "metric_proof"
    assert ir.art_direction.direction_id == "data_proof"
    assert ir.motion_intensity == "high"
    assert ir.safe_margin_px >= 70
    assert ir.subtitle_safe_px >= 100


def test_design_ir_rotates_art_direction_by_variant() -> None:
    base_spec = {
        "template": "signal_network",
        "visual_type_hint": "process",
        "headline": "Feedback Loop",
        "steps": ["Build", "Get stuck", "Target study"],
        "duration": 2.8,
    }

    first = build_design_ir(base_spec, width=1920, height=1080, fps=30, variant_index=0)
    second = build_design_ir(base_spec, width=1920, height=1080, fps=30, variant_index=1)

    assert first.art_direction.direction_id == "system_flow"
    assert second.art_direction.direction_id == "premium_explainer"
    assert first.design_id != second.design_id


def test_composition_metadata_carries_design_ir_and_root_classes() -> None:
    composition = build_composition(
        {
            "visual_id": "visual_design",
            "template": "interface_cascade",
            "visual_type_hint": "product_ui",
            "headline": "Agent Dashboard",
            "steps": ["Plan", "Render", "Verify"],
            "duration": 2.8,
            "hyperframes_variant_index": 1,
        },
        width=1280,
        height=720,
        fps=24,
    )

    assert composition.metadata["design_ir"]["art_direction"]["direction_id"] == "premium_explainer"
    assert "archetype-interface_cascade" in composition.html
    assert "data-composition-id=\"vex-visual_design\"" in composition.html
