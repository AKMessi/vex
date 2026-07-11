from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from vex_visuals import (
    compile_creative_direction,
    evaluate_frame_aesthetics,
    validate_creative_direction,
)


def _objects() -> list[dict]:
    return [
        {"object_id": "planner", "role": "mechanism", "emphasis": 0.65},
        {"object_id": "tool", "role": "mechanism", "emphasis": 0.6},
        {"object_id": "result", "role": "result", "emphasis": 0.9},
    ]


def _direction(*, family: str = "mechanism", width: int = 1280, height: int = 720):
    return compile_creative_direction(
        {
            "visual_id": "visual_001",
            "duration": 3.2,
            "video_design_bible": {
                "palette_sequence": [
                    {
                        "background": "#090A0C",
                        "panel_fill": "#15171C",
                        "panel_stroke": "#D8FF3E",
                        "text_primary": "#F8F8F2",
                        "text_secondary": "#B9BCC4",
                        "accent": "#D8FF3E",
                        "accent_secondary": "#58E6FF",
                        "glow": "#8B5CF6",
                    }
                ]
            },
        },
        scene_type="guided_process",
        scene_family=family,
        objects=_objects(),
        relations=[{"relation_id": "planner_to_tool"}],
        width=width,
        height=height,
    )


def test_creative_direction_is_signed_semantic_and_deterministic() -> None:
    first = _direction()
    second = _direction()

    assert first == second
    assert first.medium_family == "spatial_metaphor"
    assert first.focal_object_ids == ["result"]
    assert first.composition["layout_grammar"] == "guided_route"
    assert first.choreography["phases"][-1]["phase"] == "hold"
    assert first.art_direction["palette"]["accent"] == "#D8FF3E"
    assert validate_creative_direction(first).passed


def test_creative_direction_selects_medium_and_layout_by_semantics() -> None:
    metric = _direction(family="metric")
    emphasis = _direction(family="emphasis", width=720, height=1280)

    assert metric.medium_family == "data_sculpture"
    assert metric.composition["layout_grammar"] == "asymmetric_monument"
    assert emphasis.medium_family == "kinetic_typography"
    assert emphasis.orientation == "portrait"
    assert emphasis.composition["layout_grammar"] == "typographic_lockup"


def test_creative_direction_rejects_tampering() -> None:
    payload = _direction().to_dict()
    payload["medium_family"] = "kinetic_typography"

    validation = validate_creative_direction(payload)

    assert not validation.passed
    assert "creative_direction_signature_mismatch" in validation.errors


def test_aesthetic_critic_rewards_hierarchy_and_rejects_flat_frames(tmp_path: Path) -> None:
    direction = _direction().to_dict()
    good_paths: list[Path] = []
    for index, offset in enumerate((0, 18, 24), start=1):
        frame = np.full((180, 320, 3), (9, 12, 16), dtype=np.uint8)
        frame[28:74, 26:142] = (248, 248, 242)
        frame[92:150, 128 + offset : 278 + offset] = (20, 184, 166)
        frame[102:140, 150 + offset : 238 + offset] = (216, 255, 62)
        path = tmp_path / f"good_{index}.png"
        Image.fromarray(frame).save(path)
        good_paths.append(path)

    good = evaluate_frame_aesthetics(good_paths, direction)

    assert good.score >= 0.5
    assert good.metrics["dominant_color_count"] >= 3

    flat_paths: list[Path] = []
    for index in range(3):
        path = tmp_path / f"flat_{index}.png"
        Image.fromarray(np.full((180, 320, 3), 24, dtype=np.uint8)).save(path)
        flat_paths.append(path)

    flat = evaluate_frame_aesthetics(flat_paths, direction)

    assert not flat.passed
    assert "aesthetic_visual_hierarchy_is_too_flat" in flat.issues
