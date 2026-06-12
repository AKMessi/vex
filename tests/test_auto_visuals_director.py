from __future__ import annotations

from renderers.base import RenderedAsset
from tools.auto_visuals import (
    SOURCE_FRAME_SAMPLE_HEIGHT,
    SOURCE_FRAME_SAMPLE_WIDTH,
    _analyze_tiny_rgb_frame,
    _apply_auto_visuals_director_v3,
    _creative_outcome_signals,
    _final_auto_visuals_qa,
    _rendered_visual_quality_for_spec,
)


def test_source_frame_analysis_scores_empty_flat_frames_as_visual_opportunities() -> None:
    raw = bytes([8, 8, 8] * SOURCE_FRAME_SAMPLE_WIDTH * SOURCE_FRAME_SAMPLE_HEIGHT)

    analysis = _analyze_tiny_rgb_frame(raw, time_sec=4.25)

    assert analysis.visual_need > 0.8
    assert analysis.source_richness < 0.1
    assert "source_frame_dark" in analysis.warnings
    assert "source_frame_flat" in analysis.warnings


def test_auto_visuals_director_reroutes_non_math_manim_to_hyperframes() -> None:
    plan, report = _apply_auto_visuals_director_v3(
        [_visual_spec(renderer_hint="manim", template="keyword_stack")],
        [_visual_card()],
        renderer_name="auto",
        capabilities=_capabilities(),
        force_fullscreen=True,
        max_visuals=4,
    )

    assert report["accepted_count"] == 1
    assert plan[0]["renderer_hint"] == "hyperframes"
    assert plan[0]["composition_mode"] == "replace"
    assert "manim_rerouted_non_math_visual" in plan[0]["auto_visuals_director"]["warnings"]


def test_auto_visuals_director_keeps_strict_manim_separate_from_hyperframes() -> None:
    plan, report = _apply_auto_visuals_director_v3(
        [_visual_spec(renderer_hint="manim", template="keyword_stack")],
        [_visual_card()],
        renderer_name="manim",
        capabilities=[
            {"name": "manim", "available": True, "supported_templates": ["keyword_stack"]},
        ],
        force_fullscreen=True,
        max_visuals=4,
    )

    assert plan == []
    assert report["rejected_count"] == 1
    assert report["rejected"][0]["renderer_policy"] == "manim"


def test_auto_visuals_director_uses_set_quality_instead_of_earliest_candidate() -> None:
    early_spec = _visual_spec(
        visual_id="visual_early",
        card_id="card_early",
        start=1.0,
        end=4.0,
        confidence=0.58,
    )
    strong_spec = _visual_spec(
        visual_id="visual_strong",
        card_id="card_strong",
        start=8.0,
        end=11.0,
        confidence=0.98,
    )
    early_card = {
        **_visual_card(),
        "card_id": "card_early",
        "visualizability": 0.55,
        "creative_graph_signals": {
            "graph_visual_opportunity": 0.52,
            "graph_topic_alignment": 0.58,
        },
    }
    strong_card = {
        **_visual_card(),
        "card_id": "card_strong",
        "visualizability": 0.96,
        "creative_graph_signals": {
            "graph_visual_opportunity": 0.98,
            "graph_topic_alignment": 0.96,
        },
    }

    plan, report = _apply_auto_visuals_director_v3(
        [early_spec, strong_spec],
        [early_card, strong_card],
        renderer_name="auto",
        capabilities=_capabilities(),
        force_fullscreen=True,
        max_visuals=1,
    )

    assert [item["visual_id"] for item in plan] == ["visual_001"]
    assert plan[0]["card_id"] == "card_strong"
    assert report["set_optimization"]["selected"][0]["candidate_id"] == "visual_strong"


def test_rendered_visual_qa_rejects_failed_hyperframes_variant() -> None:
    asset = RenderedAsset(
        asset_path="/tmp/visual.mp4",
        width=1920,
        height=1080,
        duration_sec=3.0,
        renderer="hyperframes",
        job_dir="/tmp/job",
        script_path="/tmp/job/scene.html",
        metadata={
            "variant_selection": {
                "selected_quality_score": 0.42,
                "selected_quality_passed": False,
                "selected_variant_id": "variant_01",
            }
        },
    )

    report = _rendered_visual_quality_for_spec(_visual_spec(), asset)

    assert report.passed is False
    assert "hyperframes_variant_quality_below_floor" in report.issues


def test_rendered_visual_qa_rejects_failed_independent_final_judge() -> None:
    asset = RenderedAsset(
        asset_path="/tmp/visual.mp4",
        width=1920,
        height=1080,
        duration_sec=3.0,
        renderer="hyperframes",
        job_dir="/tmp/job",
        script_path="/tmp/job/scene.html",
        metadata={
            "variant_selection": {
                "selected_quality_score": 0.92,
                "selected_quality_passed": True,
                "selected_variant_id": "proof_01",
            },
            "semantic_qa": {"passed": True, "score": 0.92},
            "visual_critics": {
                "passed": True,
                "score": 0.9,
                "hard_failure_count": 0,
            },
            "final_independent_verdict": {
                "passed": False,
                "score": 0.54,
                "issues": ["The relation remains visually ambiguous."],
            },
        },
    )

    report = _rendered_visual_quality_for_spec(_visual_spec(), asset)

    assert report.passed is False
    assert "hyperframes_independent_final_judge_failed" in report.issues


def test_rendered_visual_qa_rejects_non_math_manim_visuals() -> None:
    spec = {
        **_visual_spec(renderer_hint="manim", template="keyword_stack"),
        "visual_intent_type": "concept",
    }
    asset = RenderedAsset(
        asset_path="/tmp/visual.mp4",
        width=1920,
        height=1080,
        duration_sec=3.0,
        renderer="manim",
        job_dir="/tmp/job",
        script_path="/tmp/job/scene.py",
        metadata={"quality_score": 0.91, "scene_generation_mode": "blueprint_compiler"},
    )

    report = _rendered_visual_quality_for_spec(spec, asset)

    assert report.passed is False
    assert "manim_render_not_matched_to_math_or_formula_context" in report.issues


def test_final_auto_visuals_qa_adds_soft_transitions_for_replacement_visuals() -> None:
    overlays, report = _final_auto_visuals_qa(
        [
            {
                "visual_id": "visual_001",
                "start": 2.0,
                "end": 5.0,
                "compose_mode": "replace",
                "rendered_visual_qa": {"passed": True, "score": 0.82},
                "auto_visuals_director": {
                    "visual_need": 0.74,
                    "source_richness": 0.24,
                },
            }
        ],
        clip_duration=12.0,
    )

    assert report["accepted_count"] == 1
    assert overlays[0]["transition_in"]["kind"] == "soft_dissolve"
    assert overlays[0]["transition_out"]["kind"] == "soft_dissolve"


def test_final_auto_visuals_qa_keeps_stronger_overlapping_visual() -> None:
    overlays, report = _final_auto_visuals_qa(
        [
            {
                "visual_id": "visual_early",
                "start": 2.0,
                "end": 5.0,
                "compose_mode": "replace",
                "rendered_visual_qa": {"passed": True, "score": 0.62},
                "auto_visuals_director": {
                    "director_score": 68.0,
                    "copy_alignment": 0.7,
                    "visual_need": 0.7,
                    "source_richness": 0.2,
                },
            },
            {
                "visual_id": "visual_strong",
                "start": 3.0,
                "end": 6.0,
                "compose_mode": "replace",
                "rendered_visual_qa": {"passed": True, "score": 0.94},
                "auto_visuals_director": {
                    "director_score": 91.0,
                    "copy_alignment": 0.9,
                    "visual_need": 0.82,
                    "source_richness": 0.14,
                },
            },
        ],
        clip_duration=12.0,
    )

    assert [item["visual_id"] for item in overlays] == ["visual_strong"]
    assert report["set_optimization"]["rejected"][0]["candidate_id"] == "visual_early"


def test_creative_outcome_signals_capture_all_renderer_tournament_contenders() -> None:
    signals = _creative_outcome_signals(
        [
            {
                "visual_id": "visual_001",
                "visual_intent_type": "mechanism",
                "template": "mechanism_blueprint",
            }
        ],
        [
            {
                "visual_id": "visual_001",
                "renderer": "ffmpeg",
                "score": 0.9,
                "passed": True,
                "renderer_tournament": {
                    "attempts": [
                        {
                            "renderer": "hyperframes",
                            "rendered": True,
                            "qa": {
                                "renderer": "hyperframes",
                                "score": 0.54,
                                "passed": False,
                            },
                        },
                        {
                            "renderer": "ffmpeg",
                            "rendered": True,
                            "qa": {
                                "renderer": "ffmpeg",
                                "score": 0.9,
                                "passed": True,
                            },
                        },
                    ]
                },
            }
        ],
        [{"visual_id": "visual_001"}],
    )

    assert [item["renderer"] for item in signals] == ["hyperframes", "ffmpeg"]
    assert signals[0]["qa_passed"] is False
    assert signals[1]["qa_passed"] is True
    assert signals[0]["published"] is False
    assert signals[1]["published"] is True


def _visual_spec(**overrides: object) -> dict[str, object]:
    spec: dict[str, object] = {
        "visual_id": "visual_001",
        "card_id": "card_001",
        "start": 2.0,
        "end": 5.0,
        "template": "mechanism_blueprint",
        "renderer_hint": "auto",
        "composition_mode": "replace",
        "headline": "Measurement Pipeline",
        "deck": "Quality improves only when every pass feeds a measurable signal.",
        "emphasis_text": "Quality loop",
        "supporting_lines": ["Measure each pass", "Promote the cleanest signal"],
        "steps": ["Plan", "Measure", "Improve"],
        "keywords": ["measurement", "pipeline", "quality"],
        "sentence_text": "The system has no measurement pipeline, so quality never improves.",
        "context_text": "The fix is a measurement pipeline that turns production feedback into quality.",
        "confidence": 0.92,
        "rationale": "Show the production loop as a mechanism, not a generic card.",
        "auto_visuals_director": {
            "director_score": 82.0,
            "copy_alignment": 0.84,
            "warnings": [],
        },
        "visual_intent_type": "mechanism",
    }
    spec.update(overrides)
    return spec


def _visual_card() -> dict[str, object]:
    return {
        "card_id": "card_001",
        "start": 2.0,
        "end": 5.0,
        "sentence_text": "The system has no measurement pipeline, so quality never improves.",
        "context_text": "The fix is a measurement pipeline that turns production feedback into quality.",
        "keywords": ["measurement", "pipeline", "quality"],
        "visualizability": 0.86,
        "generic_penalty": 0.04,
        "replace_safety": 0.78,
        "visual_type_hint": "process",
        "creative_graph_signals": {
            "graph_visual_opportunity": 0.9,
            "graph_topic_alignment": 0.86,
        },
        "source_frame_analysis": {
            "visual_need": 0.76,
            "source_richness": 0.22,
            "source_type": "talking_head_or_simple",
        },
    }


def _capabilities() -> list[dict[str, object]]:
    return [
        {"name": "hyperframes", "available": True, "supported_templates": ["mechanism_blueprint"]},
        {"name": "manim", "available": True, "supported_templates": ["keyword_stack"]},
        {"name": "ffmpeg", "available": True, "supported_templates": ["quote_focus"]},
    ]
