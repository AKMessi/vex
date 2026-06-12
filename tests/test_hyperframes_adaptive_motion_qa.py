from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np

from vex_hyperframes.semantic_qa import inspect_animation_frames


def test_adaptive_motion_qa_accepts_small_sequential_semantic_changes(
    tmp_path: Path,
) -> None:
    paths: list[Path] = []
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    for index in range(6):
        current = frame.copy()
        for item in range(index + 1):
            left = 12 + item * 48
            current[70:100, left : left + 34] = 220
        path = tmp_path / f"frame_{index:02d}.png"
        iio.imwrite(path, current)
        paths.append(path)
    final_hold = tmp_path / "frame_06.png"
    iio.imwrite(final_hold, iio.imread(paths[-1]))
    paths.append(final_hold)

    report = inspect_animation_frames(paths)

    assert report.passed is True
    assert report.active_transition_count >= 2
    assert report.final_hold_delta == 0.0


def test_adaptive_motion_qa_still_rejects_nearly_static_frames(
    tmp_path: Path,
) -> None:
    paths: list[Path] = []
    for index, value in enumerate((20, 20, 21, 21, 21), start=1):
        frame = np.full((180, 320, 3), value, dtype=np.uint8)
        path = tmp_path / f"static_{index:02d}.png"
        iio.imwrite(path, frame)
        paths.append(path)

    report = inspect_animation_frames(paths)

    assert report.passed is False
    assert "animation_is_effectively_static" in report.issues
