from __future__ import annotations

import subprocess
from pathlib import Path

import encode_validator
from encode_validator import validate_encode_output


def test_validate_encode_output_accepts_clean_mp4(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    output_path = tmp_path / "encoded.mp4"
    output_path.write_bytes(b"0" * 20_000)
    captured: dict[str, object] = {}

    monkeypatch.setattr(encode_validator, "probe_video", lambda _path: _metadata())
    monkeypatch.setattr(encode_validator.config, "ENCODE_VALIDATION_TIMEOUT_SEC", 42)

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        captured["timeout"] = kwargs["timeout"]
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(encode_validator.subprocess, "run", fake_run)

    report = validate_encode_output(_plan(output_path))

    assert report.ok
    assert report.decode_checked
    assert captured["timeout"] == 42
    assert "-xerror" in captured["command"]
    assert report.to_dict()["warnings"] == []
    assert report.to_dict()["format"] == "mov,mp4,m4a,3gp,3g2,mj2"


def test_validate_encode_output_rejects_decode_failure(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    output_path = tmp_path / "encoded.mp4"
    output_path.write_bytes(b"0" * 20_000)

    monkeypatch.setattr(encode_validator, "probe_video", lambda _path: _metadata())

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 1, "", "Invalid data found when processing input")

    monkeypatch.setattr(encode_validator.subprocess, "run", fake_run)

    report = validate_encode_output(_plan(output_path))

    assert not report.ok
    assert _codes(report) == {"decode_failed"}
    assert "failed full decode validation" in report.fatal_errors[0]


def test_validate_encode_output_rejects_wrong_container_missing_audio_and_duration(
    monkeypatch,
    tmp_path: Path,
) -> None:  # noqa: ANN001
    output_path = tmp_path / "encoded.mp4"
    output_path.write_bytes(b"0" * 20_000)
    output_metadata = _metadata(
        format="matroska,webm",
        has_audio=False,
        audio_codec=None,
        duration_sec=7.0,
    )

    monkeypatch.setattr(encode_validator, "probe_video", lambda _path: output_metadata)
    monkeypatch.setattr(
        encode_validator.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 0, "", ""),
    )

    report = validate_encode_output(_plan(output_path))

    assert not report.ok
    assert {"container_mismatch", "audio_missing", "duration_mismatch"}.issubset(_codes(report))


def test_validate_encode_output_warns_when_target_size_is_far_under(
    monkeypatch,
    tmp_path: Path,
) -> None:  # noqa: ANN001
    output_path = tmp_path / "encoded.mp4"
    output_path.write_bytes(b"0" * 20_000)

    monkeypatch.setattr(encode_validator, "probe_video", lambda _path: _metadata())
    monkeypatch.setattr(
        encode_validator.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 0, "", ""),
    )

    report = validate_encode_output(_plan(output_path, estimated_size_bytes=100_000))

    assert report.ok
    assert "target_size_under" in _codes(report)
    assert report.warnings == ["Output is far below the requested target size; quality may be lower than necessary."]


def _plan(output_path: Path, *, estimated_size_bytes: int | None = None) -> dict[str, object]:
    return {
        "input_path": "input.mov",
        "output_path": str(output_path),
        "intent": {
            "target_format": "mp4",
            "strip_audio": False,
        },
        "source_metadata": _metadata(format="mov,mp4,m4a,3gp,3g2,mj2"),
        "commands": [
            [
                "ffmpeg",
                "-i",
                "input.mov",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-y",
                str(output_path),
            ]
        ],
        "estimated_size_bytes": estimated_size_bytes,
    }


def _metadata(**overrides: object) -> dict[str, object]:
    metadata: dict[str, object] = {
        "duration_sec": 10.0,
        "fps": 30.0,
        "width": 1280,
        "height": 720,
        "codec": "h264",
        "profile": "High",
        "pix_fmt": "yuv420p",
        "video_bit_rate": 2_000_000,
        "has_audio": True,
        "audio_codec": "aac",
        "audio_bit_rate": 128_000,
        "audio_channels": 2,
        "size_bytes": 100_000,
        "format": "mov,mp4,m4a,3gp,3g2,mj2",
    }
    metadata.update(overrides)
    return metadata


def _codes(report: encode_validator.EncodeValidationReport) -> set[str]:
    return {issue.code for issue in report.issues if issue.severity != "info"}
