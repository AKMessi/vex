from __future__ import annotations

from google.genai import types

import config
from renderers.manim_renderer import (
    _can_accept_blueprint_compiler_quality,
    _compiler_validation_report,
    _minimum_blueprint_compiler_quality,
)
from vex_manim.layout import LayoutCanvas, build_layout_spec
from vex_manim.blueprint import build_scene_blueprints
from vex_manim.briefs import build_scene_brief
from vex_manim.layout_qa import LayoutReport, analyze_layout_snapshot
from vex_manim.qa import PreviewReport, QualityReport
from vex_manim.visual_ir import (
    build_storyboard_frames,
    build_visual_explanation_ir,
    critique_storyboard,
)


def test_gemma_config_disables_thinking_and_sdk_afc() -> None:
    tool = types.Tool(function_declarations=[])

    generation_config = config.build_gemini_generation_config(
        "system",
        model_name="gemma-4-31b-it",
        tools=[tool],
    )

    assert generation_config.thinking_config is None
    assert generation_config.automatic_function_calling is not None
    assert generation_config.automatic_function_calling.disable is True


def test_gemini_config_keeps_thinking_disabled_only_for_gemma() -> None:
    generation_config = config.build_gemini_generation_config(
        "system",
        model_name="gemini-2.5-flash",
    )

    assert generation_config.thinking_config is not None
    assert generation_config.thinking_config.thinking_budget == 0


def test_visual_ir_storyboard_uses_semantic_misconception_frame() -> None:
    spec = _comparison_spec()
    brief = build_scene_brief(spec, width=1920, height=1080, fps=30, latex_available=False)
    blueprints = build_scene_blueprints(brief, limit=3)

    ir = build_visual_explanation_ir(spec, brief, blueprints[0])
    frames = build_storyboard_frames(ir, brief, blueprints[0])
    critique = critique_storyboard(ir, frames, brief, blueprints[0])

    assert ir.scene_type == "before_after_morph"
    assert ir.misconception == "Tutorial binge"
    assert ir.correct_model == "Build then study"
    assert len(frames) == 3
    assert critique.passed


def test_layout_qa_counts_diagram_roles_as_motion_structure() -> None:
    spec = _comparison_spec()
    brief = build_scene_brief(spec, width=1920, height=1080, fps=30, latex_available=False)
    snapshot = {
        "frame": {"left": -7.11, "right": 7.11, "top": 4.0, "bottom": -4.0},
        "safe_bounds": {"left": -6.54, "right": 6.54, "top": 3.6, "bottom": -2.64},
        "registered_count": 4,
        "registered": [
            _box("panel_a", role="panel", left=-5.0, right=-3.7, top=1.5, bottom=0.2, panel_like=True),
            _box("panel_b", role="panel", left=-1.0, right=0.3, top=1.5, bottom=0.2, panel_like=True),
            _box("panel_c", role="panel", left=3.7, right=5.0, top=1.5, bottom=0.2, panel_like=True),
            _box("motion_spine", role="diagram", left=-3.6, right=3.6, top=0.15, bottom=-0.15),
        ],
        "guardrail_actions": [],
    }

    report = analyze_layout_snapshot(snapshot, brief)

    assert not any("instead of a clearer motion system" in issue for issue in report.issues)
    assert not any("boxed editorial cards" in issue for issue in report.issues)


def test_layout_canvas_tracks_output_aspect_ratio() -> None:
    landscape = LayoutCanvas.from_dimensions(1920, 1080)
    vertical = LayoutCanvas.from_dimensions(1080, 1920)
    square = LayoutCanvas.from_dimensions(1080, 1080)

    assert landscape.aspect_class == "landscape"
    assert round(landscape.frame_width, 3) == 14.222
    assert vertical.aspect_class == "vertical"
    assert round(vertical.frame_width, 3) == 4.5
    assert square.aspect_class == "square"
    assert round(square.frame_width, 3) == 8.0
    assert landscape.safe_bottom > landscape.bottom
    assert vertical.safe_right < vertical.right


def test_layout_specs_are_safe_and_non_overlapping_for_core_families() -> None:
    templates = [
        "data_journey",
        "signal_network",
        "kinetic_route",
        "spotlight_compare",
        "interface_cascade",
        "ribbon_quote",
    ]
    dimensions = [(1920, 1080), (1080, 1920), (1080, 1080)]

    for width, height in dimensions:
        for template in templates:
            spec = {
                **_comparison_spec(),
                "template": template,
                "visual_id": f"{template}_{width}x{height}",
            }
            brief = build_scene_brief(spec, width=width, height=height, fps=30, latex_available=False)
            layout = build_layout_spec(brief)
            safe = layout.canvas.safe_bounds()
            checked_slots = [
                slot
                for slot in layout.slots.values()
                if slot.slot_id != "full" and not slot.allow_overlap
            ]

            assert layout.aspect_class in {"landscape", "vertical", "square"}
            assert checked_slots
            for slot in checked_slots:
                assert slot.left >= safe["left"] - 1e-6
                assert slot.right <= safe["right"] + 1e-6
                assert slot.top <= safe["top"] + 1e-6
                assert slot.bottom >= safe["bottom"] - 1e-6
                assert slot.inner_width >= 0.4
                assert slot.inner_height >= 0.36
            for index, first in enumerate(checked_slots):
                for second in checked_slots[index + 1 :]:
                    assert not first.intersects(second, padding=0.01), (
                        template,
                        width,
                        height,
                        first.slot_id,
                        second.slot_id,
                    )


def test_near_threshold_blueprint_compiler_quality_can_recover() -> None:
    spec = _comparison_spec()
    brief = build_scene_brief(spec, width=1920, height=1080, fps=30, latex_available=False)
    blueprint = build_scene_blueprints(brief, limit=3)[0]
    validation = _compiler_validation_report(brief, blueprint, "class GeneratedScene: pass")
    min_quality = _minimum_blueprint_compiler_quality(brief)
    quality = QualityReport(
        passed=False,
        score=min_quality - 0.032,
        issues=["The composition still reads like boxed editorial cards rather than a bespoke animation."],
        preview=PreviewReport(preview_video_path="", duration_sec=brief.duration_sec),
        layout=LayoutReport(
            passed=False,
            score=min_quality - 0.152,
            issues=["The composition still reads like boxed editorial cards rather than a bespoke animation."],
            registered_count=4,
            action_count=0,
        ),
    )

    assert _can_accept_blueprint_compiler_quality(brief, validation, quality, min_quality)


def test_near_threshold_blueprint_compiler_rejects_severe_layout() -> None:
    spec = _comparison_spec()
    brief = build_scene_brief(spec, width=1920, height=1080, fps=30, latex_available=False)
    blueprint = build_scene_blueprints(brief, limit=3)[0]
    validation = _compiler_validation_report(brief, blueprint, "class GeneratedScene: pass")
    min_quality = _minimum_blueprint_compiler_quality(brief)
    quality = QualityReport(
        passed=False,
        score=min_quality - 0.032,
        issues=["comparison_after_state extends outside the safe frame and may clip on screen."],
        preview=PreviewReport(preview_video_path="", duration_sec=brief.duration_sec),
        layout=LayoutReport(
            passed=False,
            score=min_quality - 0.152,
            issues=["comparison_after_state extends outside the safe frame and may clip on screen."],
            registered_count=4,
            action_count=0,
        ),
    )

    assert not _can_accept_blueprint_compiler_quality(brief, validation, quality, min_quality)


def _comparison_spec() -> dict:
    spec = {
        "visual_id": "visual_smoke",
        "template": "spotlight_compare",
        "renderer_hint": "manim",
        "composition_mode": "replace",
        "headline": "Build First Study Later",
        "deck": "Inverted learning loop",
        "sentence_text": "You do not learn hard things by watching tutorials for ten hours.",
        "context_text": "Pick a small project, get stuck, then study exactly what blocks you.",
        "left_detail": "Tutorial binge",
        "right_detail": "Build then study",
        "supporting_lines": ["Get stuck", "Targeted study", "Build first"],
        "duration": 5.0,
        "importance": 0.82,
        "semantic_frame": {
            "intuition_mode": "misconception_flip",
            "mental_model": "Active building exposes the exact gaps passive watching hides.",
            "viewer_takeaway": "Build first, study the blocker.",
            "before_state": "Tutorial binge",
            "after_state": "Build then study",
            "cause": "Passive watching hides gaps",
            "effect": "Getting stuck reveals the next lesson",
            "story_window": "Tutorials feel productive until a project exposes the missing skill.",
        },
    }
    return spec


def _box(
    name: str,
    *,
    role: str,
    left: float,
    right: float,
    top: float,
    bottom: float,
    panel_like: bool = False,
) -> dict:
    return {
        "name": name,
        "role": role,
        "class_name": "RoundedRectangle" if panel_like else "VMobject",
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom,
        "width": right - left,
        "height": top - bottom,
        "center_x": (left + right) / 2,
        "center_y": (top + bottom) / 2,
        "text_based": False,
        "panel_like": panel_like,
        "connector_like": False,
    }
