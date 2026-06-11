from __future__ import annotations

from pathlib import Path

from tools.composite_qa import FRAME_BYTES, evaluate_visual_composite


def test_composite_qa_accepts_matching_fullscreen_visual(tmp_path: Path) -> None:
    source = _file(tmp_path / "source.mp4")
    output = _file(tmp_path / "output.mp4")
    asset = _file(tmp_path / "asset.mp4")
    frame = bytes([32, 96, 180] * (FRAME_BYTES // 3))

    report = evaluate_visual_composite(
        str(source),
        str(output),
        [_overlay(asset)],
        source_metadata=_metadata(has_audio=True),
        output_metadata=_metadata(has_audio=True),
        frame_extractor=lambda _path, _time: frame,
    )

    assert report.passed is True
    assert report.samples[0].similarity == 1.0


def test_composite_qa_rejects_missing_rendered_visual(tmp_path: Path) -> None:
    source = _file(tmp_path / "source.mp4")
    output = _file(tmp_path / "output.mp4")
    asset = _file(tmp_path / "asset.mp4")
    asset_frame = bytes([240, 80, 20] * (FRAME_BYTES // 3))
    output_frame = bytes([12, 12, 12] * (FRAME_BYTES // 3))

    report = evaluate_visual_composite(
        str(source),
        str(output),
        [_overlay(asset)],
        source_metadata=_metadata(has_audio=False),
        output_metadata=_metadata(has_audio=False),
        frame_extractor=lambda path, _time: (
            asset_frame if path == str(asset) else output_frame
        ),
    )

    assert report.passed is False
    assert "composite_visual_sample_failed" in report.issues
    assert "composite_visual_not_present_as_rendered" in report.samples[0].issues


def test_composite_qa_rejects_audio_loss_and_timing_drift(tmp_path: Path) -> None:
    source = _file(tmp_path / "source.mp4")
    output = _file(tmp_path / "output.mp4")

    report = evaluate_visual_composite(
        str(source),
        str(output),
        [],
        source_metadata=_metadata(has_audio=True, duration=10.0),
        output_metadata=_metadata(has_audio=False, duration=8.0),
        frame_extractor=lambda _path, _time: None,
    )

    assert report.passed is False
    assert "composite_dropped_source_audio" in report.issues
    assert "composite_duration_drift" in report.issues


def test_composite_qa_rejects_missing_asset_path(tmp_path: Path) -> None:
    source = _file(tmp_path / "source.mp4")
    output = _file(tmp_path / "output.mp4")
    missing = tmp_path / "missing.mp4"

    report = evaluate_visual_composite(
        str(source),
        str(output),
        [_overlay(missing)],
        source_metadata=_metadata(has_audio=False),
        output_metadata=_metadata(has_audio=False),
        frame_extractor=lambda _path, _time: None,
    )

    assert report.passed is False
    assert report.samples[0].issues == ["composite_qa_asset_missing"]


def _overlay(asset_path: Path) -> dict[str, object]:
    return {
        "visual_id": "visual_001",
        "start": 2.0,
        "end": 5.0,
        "compose_mode": "replace",
        "asset_path": str(asset_path),
    }


def _metadata(*, has_audio: bool, duration: float = 10.0) -> dict[str, object]:
    return {
        "duration_sec": duration,
        "fps": 30.0,
        "width": 1920,
        "height": 1080,
        "has_audio": has_audio,
        "size_bytes": 1000,
    }


def _file(path: Path) -> Path:
    path.write_bytes(b"media")
    return path
