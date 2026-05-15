from __future__ import annotations

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
    output_path = tmp_path / "out.mp4"

    def fake_popen(command, stderr=None, stdout=None, text=None):  # noqa: ANN001
        commands.append(list(command))
        if len(commands) == 1:
            output_path.write_bytes(b"broken")
            return _FakeProcess(
                [
                    "x264 : malloc of size 7088704 failed\n",
                    "Error submitting video frame to the encoder\n",
                ],
                1,
            )
        assert not output_path.exists()
        output_path.write_bytes(b"ok")
        return _FakeProcess(["frame=10 time=00:00:01.00\n"], 0)

    monkeypatch.setattr(engine, "probe_video", lambda _path: {"duration_sec": 2.0})
    monkeypatch.setattr(engine.subprocess, "Popen", fake_popen)

    saved = engine.export(
        "input.mp4",
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

    def fake_popen(command, stderr=None, stdout=None, text=None):  # noqa: ANN001
        commands.append(list(command))
        return _FakeProcess(["No such filter: definitely_missing\n"], 1)

    monkeypatch.setattr(engine, "probe_video", lambda _path: {"duration_sec": 2.0})
    monkeypatch.setattr(engine.subprocess, "Popen", fake_popen)

    try:
        engine.export(
            "input.mp4",
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
