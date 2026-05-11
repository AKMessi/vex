from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np

from vex_hyperframes.qa import analyze_hyperframes_quality
from vex_hyperframes.variants import build_variants, select_best_variant


def test_build_variants_stamps_variant_identity() -> None:
    variants = build_variants({"visual_id": "v1", "template": "signal_network"}, default_count=3)

    assert [variant.variant_id for variant in variants] == ["variant_01", "variant_02", "variant_03"]
    assert [variant.spec["hyperframes_variant_index"] for variant in variants] == [0, 1, 2]


def test_select_best_variant_prefers_highest_quality_score() -> None:
    selected = select_best_variant(
        [
            {"variant_id": "variant_01", "asset_path": "a.mp4", "qa": {"score": 0.74, "passed": False}},
            {"variant_id": "variant_02", "asset_path": "b.mp4", "qa": {"score": 0.91, "passed": True}},
            {"variant_id": "variant_03", "render_error": "failed"},
        ]
    )

    assert selected is not None
    assert selected["variant_id"] == "variant_02"


def test_quality_analysis_flags_blank_low_motion_frame(tmp_path: Path) -> None:
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    frame_path = tmp_path / "blank.png"
    iio.imwrite(frame_path, frame)

    report = analyze_hyperframes_quality(
        video_path=tmp_path / "blank.mp4",
        html="<div>Short title</div>",
        frame_paths=[frame_path],
        theme={"background": "#000000"},
        design_ir={"motion_intensity": "high"},
        min_score=0.78,
    )

    assert not report.passed
    assert report.mean_occupancy == 0.0
    assert any("sparse" in issue for issue in report.issues)
    assert any("static" in issue for issue in report.issues)
