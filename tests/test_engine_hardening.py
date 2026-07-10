from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import engine
from engine import VideoEngineError


def test_concat_file_line_escapes_single_quotes() -> None:
    line = engine._ffconcat_file_line("C:/video/O'Brien clip.mp4")

    assert line.startswith("file '")
    assert "O'\\''Brien clip.mp4" in line


def test_adjust_speed_omits_audio_filter_when_source_has_no_audio(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    commands: list[list[str]] = []
    monkeypatch.setattr(engine, "probe_video", lambda _path: _metadata(has_audio=False))
    monkeypatch.setattr(engine, "_run_command", lambda command, _message: commands.append(command))

    engine.adjust_speed("input.mp4", str(tmp_path), 1.5, None, None)

    command = commands[0]
    assert "[0:a]" not in " ".join(command)
    assert "-an" in command
    assert "[a]" not in command


def test_adjust_segment_speed_uses_video_only_concat_when_source_has_no_audio(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    commands: list[list[str]] = []
    monkeypatch.setattr(engine, "probe_video", lambda _path: _metadata(has_audio=False))
    monkeypatch.setattr(engine, "_run_command", lambda command, _message: commands.append(command))

    engine.adjust_speed("input.mp4", str(tmp_path), 1.5, 1.0, 3.0)

    command_text = " ".join(commands[0])
    assert "[0:a]" not in command_text
    assert "concat=n=3:v=1:a=0[v]" in command_text
    assert "-an" in commands[0]


def test_fade_in_omits_audio_filter_when_source_has_no_audio(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    commands: list[list[str]] = []
    monkeypatch.setattr(engine, "probe_video", lambda _path: _metadata(has_audio=False))
    monkeypatch.setattr(engine, "_run_command", lambda command, _message: commands.append(command))

    engine.fade_in("input.mp4", str(tmp_path), 0.5)

    command = commands[0]
    assert "-af" not in command
    assert "-an" in command


def test_extract_audio_returns_clear_error_for_video_without_audio(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    monkeypatch.setattr(engine, "probe_video", lambda _path: _metadata(has_audio=False))

    try:
        engine.extract_audio("input.mp4", str(tmp_path), "mp3")
    except VideoEngineError as exc:
        assert "no audio stream" in str(exc)
    else:
        raise AssertionError("extract_audio should fail clearly when the input has no audio stream.")


def test_replace_audio_with_mix_falls_back_to_replacement_when_original_has_no_audio(
    monkeypatch,
    tmp_path: Path,
) -> None:  # noqa: ANN001
    commands: list[list[str]] = []
    monkeypatch.setattr(engine, "probe_video", lambda _path: _metadata(has_audio=False))
    monkeypatch.setattr(engine, "_run_command", lambda command, _message: commands.append(command))

    engine.replace_audio("video.mp4", "new-audio.wav", str(tmp_path), mix=True, mix_ratio=0.4)

    command = commands[0]
    assert "-filter_complex" not in command
    assert command[command.index("-map") + 1] == "0:v"
    assert "1:a" in command


def test_mute_segment_noops_when_source_has_no_audio(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    monkeypatch.setattr(engine, "probe_video", lambda _path: _metadata(has_audio=False))
    called = False

    def fail_run_command(_command: list[str], _message: str) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(engine, "_run_command", fail_run_command)

    output_path = engine.mute_segment("input.mp4", str(tmp_path), 1.0, 2.0)

    assert output_path == "input.mp4"
    assert not called


def test_probe_video_uses_bounded_noninteractive_ffprobe(monkeypatch) -> None:  # noqa: ANN001
    calls: list[tuple[list[str], dict[str, object]]] = []
    payload = {
        "format": {"duration": "10", "size": "1024", "format_name": "mp4"},
        "streams": [
            {
                "codec_type": "video",
                "avg_frame_rate": "30/1",
                "width": 1920,
                "height": 1080,
                "codec_name": "h264",
            }
        ],
    }

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    monkeypatch.setattr(engine.subprocess, "run", fake_run)
    monkeypatch.setattr(engine.config, "FFMPEG_PROBE_TIMEOUT_SEC", 17)

    metadata = engine.probe_video("input.mp4")

    assert metadata["duration_sec"] == 10.0
    assert calls[0][0][-1] == "input.mp4"
    assert calls[0][1]["timeout"] == 17
    assert calls[0][1]["stdin"] is subprocess.DEVNULL


def test_probe_video_translates_timeout_to_engine_error(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(engine.config, "FFMPEG_PROBE_TIMEOUT_SEC", 5)
    monkeypatch.setattr(
        engine.subprocess,
        "run",
        lambda command, **_kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(command, 5)
        ),
    )

    with pytest.raises(VideoEngineError, match="timed out after 5 seconds"):
        engine.probe_video("input.mp4")


def test_probe_video_rejects_invalid_ffprobe_json(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(
        engine.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 0, "not-json", ""),
    )

    with pytest.raises(VideoEngineError, match="invalid JSON"):
        engine.probe_video("input.mp4")


def _metadata(*, has_audio: bool) -> dict[str, object]:
    return {
        "duration_sec": 10.0,
        "fps": 30.0,
        "width": 1920,
        "height": 1080,
        "codec": "h264",
        "has_audio": has_audio,
        "size_bytes": 1024,
        "format": "mov,mp4",
    }
