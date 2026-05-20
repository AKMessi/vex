from __future__ import annotations

from pathlib import Path

import encode_planner
from state import ProjectState, utc_now_iso
from tools import audio, encode, export as export_tool, merge, subtitles
from tools.path_security import TRUSTED_OUTPUT_PATH_TOKEN


def test_audio_extract_rejects_model_output_outside_safe_roots(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    state = _state(tmp_path)
    blocked_target = tmp_path / "outside.mp3"
    called = False

    def fake_extract_audio(*_args: object, **_kwargs: object) -> str:
        nonlocal called
        called = True
        return str(tmp_path / "temp.mp3")

    monkeypatch.setattr(audio, "extract_audio", fake_extract_audio)

    result = audio.execute_extract(
        {"format": "mp3", "output_path": str(blocked_target)},
        state,
    )

    assert not result["success"]
    assert "must stay inside" in result["message"]
    assert not called
    assert not blocked_target.exists()


def test_export_rejects_model_output_outside_safe_roots(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    state = _state(tmp_path)
    blocked_target = tmp_path / "outside.mp4"
    exported_paths: list[str] = []

    monkeypatch.setattr(export_tool, "load_presets", lambda: {"web": {"format": "mp4"}})
    monkeypatch.setattr(export_tool, "estimate_output_size", lambda *_args: 1024)
    monkeypatch.setattr(export_tool, "check_disk_space", lambda *_args: True)
    monkeypatch.setattr(export_tool, "export", lambda _src, dst, _preset: exported_paths.append(dst) or dst)

    result = export_tool.execute(
        {"preset_name": "web", "output_path": str(blocked_target)},
        state,
    )

    assert not result["success"]
    assert "must stay inside" in result["message"]
    assert exported_paths == []
    assert not blocked_target.exists()


def test_export_tool_handles_missing_preset_name(tmp_path: Path) -> None:
    state = _state(tmp_path)

    result = export_tool.execute({}, state)

    assert not result["success"]
    assert "Missing export preset name" in result["message"]


def test_export_tool_rejects_non_object_custom_settings(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    state = _state(tmp_path)
    monkeypatch.setattr(export_tool, "load_presets", lambda: {"web": {"format": "mp4"}})

    result = export_tool.execute({"preset_name": "web", "custom_settings": "bad"}, state)

    assert not result["success"]
    assert "custom_settings must be an object" in result["message"]


def test_export_tool_reports_size_estimate_errors(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    state = _state(tmp_path)
    monkeypatch.setattr(export_tool, "load_presets", lambda: {"web": {"format": "mp4", "video_bitrate": "bad"}})
    monkeypatch.setattr(export_tool, "estimate_output_size", lambda *_args: (_ for _ in ()).throw(ValueError("bad bitrate")))

    result = export_tool.execute({"preset_name": "web"}, state)

    assert not result["success"]
    assert "Could not estimate export size" in result["message"]


def test_plan_encode_rejects_untrusted_output_outside_project(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    monkeypatch.setattr(encode_planner, "available_ffmpeg_encoders", lambda: {"libx264", "aac"})
    state = _state(tmp_path)

    result = encode.execute_plan(
        {"raw_request": "convert to mp4", "output_path": str(tmp_path / "outside.mp4")},
        state,
    )

    assert not result["success"]
    assert "must stay inside" in result["message"]
    assert "pending_encode" not in state.artifacts


def test_replace_audio_rejects_model_input_outside_project(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    state = _state(tmp_path)
    outside_audio = tmp_path / "outside.wav"
    outside_audio.write_bytes(b"audio")
    called = False

    def fake_replace_audio(*_args: object, **_kwargs: object) -> str:
        nonlocal called
        called = True
        return str(Path(state.working_dir) / "replaced.mp4")

    monkeypatch.setattr(audio, "replace_audio", fake_replace_audio)

    result = audio.execute_replace({"audio_path": str(outside_audio)}, state)

    assert not result["success"]
    assert "must stay inside" in result["message"]
    assert not called


def test_merge_rejects_model_input_outside_project(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    state = _state(tmp_path)
    outside_clip = tmp_path / "outside.mp4"
    outside_clip.write_bytes(b"video")
    calls: list[list[str]] = []

    def fake_merge(paths: list[str], _working_dir: str) -> str:
        calls.append(paths)
        return str(Path(state.working_dir) / "merged.mp4")

    monkeypatch.setattr(merge, "merge", fake_merge)

    result = merge.execute({"file_paths": [str(outside_clip)]}, state)

    assert not result["success"]
    assert "must stay inside" in result["message"]
    assert calls == []


def test_burn_subtitles_rejects_model_input_outside_project(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    state = _state(tmp_path)
    outside_srt = tmp_path / "outside.srt"
    outside_srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
    called = False

    def fake_burn_subtitles(*_args: object, **_kwargs: object) -> str:
        nonlocal called
        called = True
        return str(Path(state.working_dir) / "subtitled.mp4")

    monkeypatch.setattr(subtitles, "burn_subtitles", fake_burn_subtitles)

    result = subtitles.execute({"srt_path": str(outside_srt)}, state)

    assert not result["success"]
    assert "must stay inside" in result["message"]
    assert not called


def test_project_input_path_allows_source_parent_files(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    state = _state(tmp_path)
    project_audio = Path(state.working_dir) / "voiceover.wav"
    project_audio.write_bytes(b"audio")

    monkeypatch.setattr(audio, "replace_audio", lambda *_args, **_kwargs: str(Path(state.working_dir) / "replaced.mp4"))
    monkeypatch.setattr(audio, "probe_video", lambda _path: _metadata(size_bytes=1024))

    result = audio.execute_replace({"audio_path": str(project_audio)}, state)

    assert result["success"]
    assert state.timeline[-1]["params"]["audio_path"] == str(project_audio.resolve())


def test_trusted_encode_output_path_can_target_cli_destination(tmp_path: Path) -> None:
    destination = tmp_path / "cli-output.mp4"

    plan = encode_planner.build_encode_plan(
        str(tmp_path / "working.mov"),
        str(tmp_path / "project" / "out"),
        "Encode Test",
        {
            "raw_request": "convert to mp4",
            "output_path": str(destination),
            "_trusted_output_path_token": TRUSTED_OUTPUT_PATH_TOKEN,
        },
        metadata=_metadata(size_bytes=12),
        available_encoders={"libx264", "aac"},
    )

    assert plan.output_path == str(destination.resolve())


def _state(tmp_path: Path) -> ProjectState:
    source = tmp_path / "project" / "working.mov"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source-video")
    now = utc_now_iso()
    return ProjectState(
        project_id="test-project",
        project_name="Output Path Test",
        created_at=now,
        updated_at=now,
        source_files=[str(source)],
        working_file=str(source),
        working_dir=str(tmp_path / "project"),
        output_dir=str(tmp_path / "project" / "out"),
        metadata=_metadata(size_bytes=source.stat().st_size),
        provider="test",
        model="test-model",
    )


def _metadata(*, size_bytes: int) -> dict[str, object]:
    return {
        "duration_sec": 10.0,
        "fps": 30.0,
        "width": 1920,
        "height": 1080,
        "codec": "h264",
        "video_bit_rate": 8_000_000,
        "has_audio": True,
        "audio_codec": "aac",
        "audio_bit_rate": 128_000,
        "audio_channels": 2,
        "size_bytes": size_bytes,
        "format": "mov,mp4,m4a,3gp,3g2,mj2",
    }
