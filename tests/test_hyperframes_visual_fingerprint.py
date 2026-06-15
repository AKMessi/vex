from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np

from vex_hyperframes.qa import (
    build_rendered_visual_fingerprint,
    visual_fingerprint_distance,
)


def test_rendered_visual_fingerprint_distinguishes_different_color_fields(
    tmp_path: Path,
) -> None:
    red_path = tmp_path / "red.png"
    blue_path = tmp_path / "blue.png"
    iio.imwrite(
        red_path,
        np.full((90, 160, 3), [240, 24, 32], dtype=np.uint8),
    )
    iio.imwrite(
        blue_path,
        np.full((90, 160, 3), [16, 62, 230], dtype=np.uint8),
    )

    red = build_rendered_visual_fingerprint(
        [red_path],
        visual_world_program={
            "medium_family": "editorial_collage",
            "background_mode": "paper_registration",
        },
    )
    blue = build_rendered_visual_fingerprint(
        [blue_path],
        visual_world_program={
            "medium_family": "data_sculpture",
            "background_mode": "radial_data_field",
        },
    )

    assert red["available"] is True
    assert blue["available"] is True
    assert red["signature"] != blue["signature"]
    assert visual_fingerprint_distance(red, blue) > 0.2


def test_rendered_visual_fingerprint_is_stable_for_same_frames(
    tmp_path: Path,
) -> None:
    frame_path = tmp_path / "frame.png"
    image = np.zeros((90, 160, 3), dtype=np.uint8)
    image[:, :80] = [245, 235, 220]
    image[:, 80:] = [20, 30, 40]
    iio.imwrite(frame_path, image)

    first = build_rendered_visual_fingerprint([frame_path])
    second = build_rendered_visual_fingerprint([frame_path])

    assert first == second
    assert visual_fingerprint_distance(first, second) == 0.0
