from __future__ import annotations

from pathlib import Path

import pytest

from engine import VideoEngineError
from state import ProjectState, utc_now_iso
from tools import undo


def test_rebuild_timeline_rejects_stored_audio_path_outside_project(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    state = _state(tmp_path)
    outside_audio = tmp_path / "outside.wav"
    outside_audio.write_bytes(b"audio")
    state.timeline = [
        {
            "op": "replace_audio",
            "params": {
                "audio_path": str(outside_audio),
                "mix_with_original": False,
                "mix_ratio": 0.5,
            },
        }
    ]
    monkeypatch.setattr(undo, "probe_video", lambda _path: _metadata())

    with pytest.raises(VideoEngineError, match="must stay inside"):
        undo.rebuild_timeline(state)


def test_rebuild_timeline_rejects_stored_visual_manifest_outside_project(tmp_path: Path) -> None:
    state = _state(tmp_path)
    outside_manifest = tmp_path / "outside.json"
    outside_manifest.write_text('{"overlays": []}', encoding="utf-8")
    state.timeline = [
        {
            "op": "add_auto_visuals",
            "params": {"manifest_path": str(outside_manifest)},
        }
    ]

    with pytest.raises(VideoEngineError, match="must stay inside"):
        undo.rebuild_timeline(state)


def _state(tmp_path: Path) -> ProjectState:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    source = project_dir / "source.mp4"
    source.write_bytes(b"source")
    now = utc_now_iso()
    return ProjectState(
        project_id="replay-hardening",
        project_name="Replay Hardening",
        created_at=now,
        updated_at=now,
        source_files=[str(source)],
        working_file=str(project_dir / "working.mp4"),
        working_dir=str(project_dir),
        output_dir=str(project_dir / "out"),
        metadata=_metadata(),
        provider="test",
        model="test-model",
    )


def _metadata() -> dict[str, object]:
    return {
        "duration_sec": 10.0,
        "fps": 30.0,
        "width": 1920,
        "height": 1080,
        "codec": "h264",
        "has_audio": True,
        "size_bytes": 1024,
        "format": "mov,mp4",
    }
