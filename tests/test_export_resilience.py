from __future__ import annotations

import subprocess
from pathlib import Path

import engine


class _FakeProcess:
    def __init__(self, lines: list[str], returncode: int) -> None:
        self.stderr = lines
        self._returncode = returncode

    def wait(self) -> int:
        return self._returncode


def test_export_retries_libx264_memory_failure_with_low_memory_settings(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    commands: list[list[str]] = []
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "out.mp4"
    input_path.write_bytes(b"source")

    def fake_popen(command, stderr=None, stdout=None, text=None):  # noqa: ANN001
        commands.append(list(command))
        command_output = Path(command[-1])
        if len(commands) == 1:
            command_output.write_bytes(b"broken")
            return _FakeProcess(
                [
                    "x264 : malloc of size 7088704 failed\n",
                    "Error submitting video frame to the encoder\n",
                ],
                1,
            )
        assert not output_path.exists()
        command_output.write_bytes(b"ok")
        return _FakeProcess(["frame=10 time=00:00:01.00\n"], 0)

    monkeypatch.setattr(engine, "probe_video", lambda _path: _metadata())
    monkeypatch.setattr(engine.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        engine.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 0, "", ""),
    )

    saved = engine.export(
        str(input_path),
        str(output_path),
        {
            "resolution": "1920x1080",
            "video_codec": "libx264",
            "audio_codec": "aac",
            "video_bitrate": "8000k",
            "audio_bitrate": "192k",
            "format": "mp4",
        },
    )

    assert saved == str(output_path)
    assert output_path.read_bytes() == b"ok"
    assert len(commands) == 2
    retry = commands[1]
    assert retry[retry.index("-threads") + 1] == "2"
    assert retry[retry.index("-preset") + 1] == "veryfast"
    assert retry[retry.index("-x264-params") + 1] == "rc-lookahead=10:sync-lookahead=0:sliced-threads=1"
    assert "-pix_fmt" in retry
    assert retry[retry.index("-pix_fmt") + 1] == "yuv420p"


def test_export_does_not_retry_unrelated_ffmpeg_failure(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    commands: list[list[str]] = []
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"source")

    def fake_popen(command, stderr=None, stdout=None, text=None):  # noqa: ANN001
        commands.append(list(command))
        return _FakeProcess(["No such filter: definitely_missing\n"], 1)

    monkeypatch.setattr(engine, "probe_video", lambda _path: _metadata())
    monkeypatch.setattr(engine.subprocess, "Popen", fake_popen)

    try:
        engine.export(
            str(input_path),
            str(tmp_path / "out.mp4"),
            {
                "resolution": "1920x1080",
                "video_codec": "libx264",
                "audio_codec": "aac",
                "video_bitrate": "8000k",
                "audio_bitrate": "192k",
                "format": "mp4",
            },
        )
    except engine.VideoEngineError as exc:
        assert "No such filter" in str(exc)
    else:
        raise AssertionError("Unrelated FFmpeg failures should not be retried or hidden.")

    assert len(commands) == 1


def test_export_preserves_existing_output_when_validation_fails(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "out.mp4"
    input_path.write_bytes(b"source")
    output_path.write_bytes(b"known-good")

    def fake_popen(command, stderr=None, stdout=None, text=None):  # noqa: ANN001
        Path(command[-1]).write_bytes(b"corrupt")
        return _FakeProcess(["frame=10 time=00:00:01.00\n"], 0)

    monkeypatch.setattr(engine, "probe_video", lambda _path: _metadata())
    monkeypatch.setattr(engine.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        engine.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 1, "", "Invalid data found"),
    )

    try:
        engine.export(
            str(input_path),
            str(output_path),
            {
                "resolution": "1920x1080",
                "video_codec": "libx264",
                "audio_codec": "aac",
                "video_bitrate": "8000k",
                "audio_bitrate": "192k",
                "format": "mp4",
            },
        )
    except engine.VideoEngineError as exc:
        assert "decode validation failed" in str(exc)
    else:
        raise AssertionError("Corrupt exports must fail validation.")

    assert output_path.read_bytes() == b"known-good"
    assert not list(tmp_path.glob("*.tmp.mp4"))


def test_export_rejects_audio_only_preset_when_source_has_no_audio(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"source")
    monkeypatch.setattr(engine, "probe_video", lambda _path: _metadata(has_audio=False, audio_codec=None))

    try:
        engine.export(
            str(input_path),
            str(tmp_path / "out.mp3"),
            {
                "audio_only": True,
                "audio_codec": "libmp3lame",
                "audio_bitrate": "256k",
                "format": "mp3",
            },
        )
    except engine.VideoEngineError as exc:
        assert "source has no audio" in str(exc)
    else:
        raise AssertionError("Audio-only export should fail before FFmpeg when the source has no audio.")


def test_export_command_preserves_aspect_ratio_for_scaled_presets() -> None:
    command = engine._build_export_command(  # noqa: SLF001
        "input.mp4",
        "output.mp4",
        {
            "resolution": "1080x1920",
            "video_codec": "libx264",
            "audio_codec": "aac",
            "format": "mp4",
        },
    )

    filter_graph = command[command.index("-vf") + 1]
    assert "force_original_aspect_ratio=decrease" in filter_graph
    assert "pad=1080:1920" in filter_graph
    assert "setsar=1" in filter_graph


def _metadata(**overrides: object) -> dict[str, object]:
    metadata: dict[str, object] = {
        "duration_sec": 2.0,
        "fps": 30.0,
        "width": 1920,
        "height": 1080,
        "codec": "h264",
        "profile": "High",
        "pix_fmt": "yuv420p",
        "video_bit_rate": 8_000_000,
        "has_audio": True,
        "audio_codec": "aac",
        "audio_bit_rate": 128_000,
        "audio_channels": 2,
        "size_bytes": 1024,
        "format": "mov,mp4,m4a,3gp,3g2,mj2",
    }
    metadata.update(overrides)
    return metadata
