from __future__ import annotations

from renderers import resolve_renderer
from renderers.remotion_renderer import RemotionRenderer, _build_input_props
from tools.auto_visuals import _compile_hyperframes_specs, _rendered_visual_quality_for_spec
from renderers.base import RenderedAsset


def _spec() -> dict:
    return {
        "visual_id": "visual_001",
        "card_id": "card_001",
        "template": "signal_network",
        "renderer_hint": "remotion",
        "visual_intent_type": "mechanism",
        "visual_type_hint": "process",
        "composition_mode": "replace",
        "headline": "Agents Coordinate",
        "deck": "Planner, tools, and memory move in sequence",
        "supporting_lines": ["Planner selects", "Tool acts", "Memory updates"],
        "steps": ["Plan", "Act", "Observe"],
        "keywords": ["planner", "tools", "memory"],
        "duration": 3.2,
        "importance": 0.8,
        "auto_visuals_director": {
            "director_score": 72,
            "copy_alignment": 0.62,
        },
    }


def test_remotion_renderer_resolves_as_strict_backend() -> None:
    renderer, reason = resolve_renderer(
        _spec(),
        preferred="remotion",
        allow_unavailable=True,
    )

    assert renderer.name == "remotion"
    assert "remotion was explicitly preferred" in reason


def test_remotion_input_props_preserve_structured_visual_data() -> None:
    props = _build_input_props(
        {
            **_spec(),
            "metric_facts": [{"value": "42%", "label": "accuracy"}],
            "visual_beats": [{"text": "Planner routes the job"}],
        },
        width=1920,
        height=1080,
        fps=30,
    )

    assert props["width"] == 1920
    assert props["height"] == 1080
    assert props["spec"]["metric_facts"][0]["value"] == "42%"
    assert props["spec"]["visual_beats"][0]["text"] == "Planner routes the job"


def test_hyperframes_compiler_bypasses_remotion_specs() -> None:
    plan, report = _compile_hyperframes_specs([_spec()])

    assert plan == [_spec()]
    assert report["compiled_count"] == 0
    assert report["accepted_count"] == 1
    assert report["estimated_render_count"] == 1


def test_remotion_rendered_quality_uses_renderer_metadata() -> None:
    asset = RenderedAsset(
        asset_path="visual.mp4",
        width=1920,
        height=1080,
        duration_sec=3.2,
        renderer="remotion",
        job_dir=".",
        script_path="entry.jsx",
        metadata={"quality_score": 0.72, "quality_passed": True},
    )

    qa = _rendered_visual_quality_for_spec(_spec(), asset)

    assert qa.renderer == "remotion"
    assert qa.passed
    assert qa.score >= 0.6


def test_remotion_renderer_scores_dom_explainer_specs() -> None:
    renderer = RemotionRenderer()

    assert renderer.supports(_spec())
    assert renderer.score_spec(_spec()) > 1.0
