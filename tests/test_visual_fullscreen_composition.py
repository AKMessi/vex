from __future__ import annotations

from pathlib import Path

from engine import _normalize_visual_overlays
from visual_intelligence import fallback_visual_plan


def test_premium_generated_visual_plan_forces_fullscreen_replace() -> None:
    plan = fallback_visual_plan(
        [_visual_card()],
        clip_duration=12.0,
        max_visuals=1,
        min_visual_sec=2.2,
        max_visual_sec=4.0,
        scene_cuts=[],
        available_renderers=[
            {"name": "hyperframes", "available": True, "supported_templates": []},
            {"name": "manim", "available": True, "supported_templates": []},
            {"name": "ffmpeg", "available": True, "supported_templates": []},
        ],
        prefer_premium=True,
    )

    assert len(plan) == 1
    visual = plan[0]
    assert visual["composition_mode"] == "replace"
    assert visual["position"] == "center"
    assert visual["scale"] == 1.0
    assert visual["renderer_hint"] == "hyperframes"
    assert visual["require_generated_scene"] is False


def test_force_fullscreen_overlay_metadata_overrides_corner_pip(tmp_path: Path) -> None:
    asset_path = tmp_path / "visual.mp4"
    asset_path.write_bytes(b"placeholder")

    normalized = _normalize_visual_overlays(
        [
            {
                "start": 1.0,
                "end": 3.0,
                "asset_path": str(asset_path),
                "compose_mode": "picture_in_picture",
                "position": "bottom_right",
                "scale": 0.42,
                "force_fullscreen": True,
            }
        ],
        duration=8.0,
        width=1920,
        height=1080,
    )

    assert len(normalized) == 1
    overlay = normalized[0]
    assert overlay["compose_mode"] == "replace"
    assert overlay["position"] == "center"
    assert overlay["scale"] == 1.0
    assert overlay["margin"] == 0


def _visual_card() -> dict:
    return {
        "card_id": "visual_card_001",
        "start": 2.0,
        "end": 5.0,
        "sentence_text": "The system moves from scattered effort to a focused feedback loop.",
        "context_text": "A visual route should show the before state collapsing into a tighter process.",
        "previous_text": "",
        "next_text": "",
        "keywords": ["scattered effort", "feedback loop", "focused process"],
        "visual_type_hint": "process",
        "word_count": 12,
        "words_per_second": 3.0,
        "pause_before": 0.4,
        "pause_after": 0.5,
        "nearest_scene_cut": None,
        "scene_distance": 2.5,
        "sentence_numeric_hits": 0,
        "numeric_hits": 0,
        "sentence_process_cues": 0.8,
        "process_cues": 0.8,
        "sentence_contrast_cues": 0.45,
        "contrast_cues": 0.45,
        "generic_penalty": 0.1,
        "concrete_hits": 2.0,
        "proper_nouns": 0,
        "replace_safety": 0.2,
        "visualizability": 0.7,
        "semantic_frame": {
            "intuition_mode": "process_route",
            "intuition_role": "core_mechanism",
            "intuition_payoff": 0.82,
            "before_state": "Scattered effort",
            "after_state": "Focused feedback loop",
            "viewer_takeaway": "Focus the loop.",
        },
        "intuition_mode": "process_route",
        "intuition_role": "core_mechanism",
        "intuition_payoff": 0.82,
        "novelty_key": "focused-feedback-loop",
        "suggested_composition": "picture_in_picture",
        "style_pack": "editorial_clean",
        "suggested_renderer": "ffmpeg",
        "priority": 86.0,
    }
