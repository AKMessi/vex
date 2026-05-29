from __future__ import annotations

from pathlib import Path

from engine import _normalize_visual_overlays
from visual_intelligence import (
    _count_numbers,
    _extract_metric_facts,
    _normalize_visual_plan,
    _split_fragments,
    build_visual_context_cards,
    fallback_visual_plan,
)


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


def test_premium_visual_plan_backfills_multiple_contextual_beats() -> None:
    cards = []
    for index, start in enumerate((1.0, 8.0, 15.0, 22.0, 29.0), start=1):
        card = {**_visual_card()}
        card.update(
            {
                "card_id": f"visual_card_{index:03d}",
                "start": start,
                "end": start + 2.2,
                "sentence_text": f"Step {index} turns noisy context into a clearer production signal.",
                "context_text": f"The visual should show production signal {index} moving through the workflow.",
                "keywords": [f"signal {index}", "workflow", "production"],
                "novelty_key": f"signal-{index}",
                "priority": 92.0 - index,
            }
        )
        card["semantic_frame"] = {
            **dict(card["semantic_frame"]),
            "viewer_takeaway": f"Production signal {index}",
            "after_state": f"Clear signal {index}",
        }
        cards.append(card)

    plan = fallback_visual_plan(
        cards,
        clip_duration=40.0,
        max_visuals=4,
        min_visual_sec=2.2,
        max_visual_sec=4.2,
        scene_cuts=[],
        available_renderers=[
            {"name": "hyperframes", "available": True, "supported_templates": []},
            {"name": "ffmpeg", "available": True, "supported_templates": []},
        ],
        prefer_premium=True,
    )

    assert len(plan) == 4
    assert all(item["composition_mode"] == "replace" for item in plan)
    assert len({item["card_id"] for item in plan}) == 4


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


def test_visual_plan_preserves_blender_3d_template_when_available() -> None:
    plan = _normalize_visual_plan(
        [
            {
                "card_id": "visual_card_001",
                "template": "data_tunnel",
                "renderer_hint": "auto",
                "composition_mode": "replace",
                "headline": "Neural Network",
                "confidence": 0.92,
            }
        ],
        [_visual_card()],
        clip_duration=12.0,
        max_visuals=1,
        min_visual_sec=2.2,
        max_visual_sec=4.0,
        scene_cuts=[],
        available_renderers=[
            {"name": "blender", "available": True, "supported_templates": ["data_tunnel"]},
            {"name": "hyperframes", "available": True, "supported_templates": []},
            {"name": "ffmpeg", "available": True, "supported_templates": []},
        ],
    )

    assert plan[0]["template"] == "data_tunnel"
    assert plan[0]["renderer_hint"] == "blender"
    assert plan[0]["composition_mode"] == "replace"
    assert plan[0]["camera_motion"] == "orbit"


def test_visual_plan_downgrades_blender_template_when_unavailable() -> None:
    plan = _normalize_visual_plan(
        [
            {
                "card_id": "visual_card_001",
                "template": "screen_pointer_3d",
                "renderer_hint": "blender",
                "composition_mode": "overlay",
                "headline": "Look here",
                "confidence": 0.92,
            }
        ],
        [_visual_card()],
        clip_duration=12.0,
        max_visuals=1,
        min_visual_sec=2.2,
        max_visual_sec=4.0,
        scene_cuts=[],
        available_renderers=[
            {"name": "blender", "available": False, "supported_templates": []},
            {"name": "ffmpeg", "available": True, "supported_templates": []},
        ],
    )

    assert plan[0]["template"] == "quote_focus"
    assert plan[0]["renderer_hint"] != "blender"


def test_metric_detection_ignores_version_fragments_and_preserves_units() -> None:
    text = "DeepSeek V4 Pro requires 10 % of the KV cache compared to V3.8."

    assert _count_numbers("DeepSeek V4 Pro compared to V3.8") == 0
    assert _count_numbers(text) == 1
    assert _extract_metric_facts(text)[0]["value"] == "10%"
    assert _extract_metric_facts("Compute jumps at 1 million tokens.")[0]["value"] == "1 million"
    assert _split_fragments(text) == [
        "DeepSeek V4 Pro requires 10 % of the KV cache compared to V3.8"
    ]


def test_visual_cards_merge_transcript_fragments_and_skip_bridge_lines() -> None:
    sentences = [
        {"start": 100.62, "end": 104.96, "text": "So deep seek, introduce something that is very interesting hybrid attention."},
        {"start": 105.34, "end": 109.04, "text": "We'll talk about this in detail in the video later."},
        {"start": 109.3, "end": 110.9, "text": "And this is one interesting insight."},
        {"start": 110.9, "end": 115.62, "text": "Deep seek V4 Pro just requires 10 % of the KV"},
        {"start": 115.62, "end": 118.4, "text": "cache compared to V3.8."},
        {"start": 119.3, "end": 120.88, "text": "Did it not reduce 10 %?"},
        {"start": 121.68, "end": 126.42, "text": "It just requires 10 % of the KV cache required by deep seek V3"},
    ]
    transcript_segments = [
        {
            "start": sentences[0]["start"],
            "end": sentences[-1]["end"],
            "text": " ".join(str(item["text"]) for item in sentences),
        }
    ]

    cards = build_visual_context_cards(sentences, transcript_segments, 140.0)

    assert not any("talk about this" in card["sentence_text"].lower() for card in cards)
    kv_card = next(card for card in cards if "V4 Pro just requires" in card["sentence_text"])
    question_card = next(card for card in cards if "Did it not reduce" in card["sentence_text"])
    assert kv_card["sentence_text"].endswith("cache compared to V3.8.")
    assert kv_card["metric_facts"][0]["value"] == "10%"
    assert kv_card["numeric_hits"] == 1
    assert kv_card["novelty_key"] == question_card["novelty_key"]

    plan = fallback_visual_plan(
        cards,
        clip_duration=140.0,
        max_visuals=3,
        min_visual_sec=2.2,
        max_visual_sec=4.8,
        scene_cuts=[],
        available_renderers=[{"name": "hyperframes", "available": True, "supported_templates": []}],
        prefer_premium=True,
    )

    kv_visuals = [item for item in plan if item.get("headline") == "10% KV cache"]
    assert len(kv_visuals) == 1
    assert kv_visuals[0]["emphasis_text"] == "10%"
    assert "8" not in kv_visuals[0].get("supporting_lines", [])
    assert not any("talk about this" in " ".join(item.get("supporting_lines", [])).lower() for item in plan)


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
