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


def test_undo_rolls_back_timeline_when_rebuild_fails(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    state = _state(tmp_path)
    state.working_file = state.source_files[0]
    state.apply_operation({"op": "trim_clip", "params": {"start": 0.0, "end": 5.0}})
    original_timeline = state.capture_snapshot()["timeline"]

    monkeypatch.setattr(
        undo,
        "rebuild_timeline",
        lambda _state: (_ for _ in ()).throw(VideoEngineError("rebuild failed")),
    )

    result = undo.execute_undo({}, state)

    assert result["success"] is False
    assert state.timeline == original_timeline
    assert state.redo_stack == []
    assert state.working_file == state.source_files[0]


def test_redo_rolls_back_timeline_when_rebuild_fails(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    state = _state(tmp_path)
    state.working_file = state.source_files[0]
    state.apply_operation({"op": "trim_clip", "params": {"start": 0.0, "end": 5.0}})
    state.undo()
    original_redo = state.capture_snapshot()["redo_stack"]

    monkeypatch.setattr(
        undo,
        "rebuild_timeline",
        lambda _state: (_ for _ in ()).throw(VideoEngineError("rebuild failed")),
    )

    result = undo.execute_redo({}, state)

    assert result["success"] is False
    assert state.timeline == []
    assert state.redo_stack == original_redo


def test_refresh_rolls_back_removed_operations_when_rebuild_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:  # noqa: ANN001
    state = _state(tmp_path)
    state.working_file = state.source_files[0]
    state.artifacts["latest_auto_visuals"] = {"manifest_path": "old.json"}
    state.apply_operation({"op": "add_auto_visuals", "params": {"overlays": []}})
    state.apply_operation({"op": "trim_clip", "params": {"start": 0.0, "end": 5.0}})
    original = state.capture_snapshot()
    monkeypatch.setattr(
        undo,
        "rebuild_timeline",
        lambda _state: (_ for _ in ()).throw(VideoEngineError("rebuild failed")),
    )

    with pytest.raises(VideoEngineError, match="rebuild failed"):
        undo.refresh_generated_overlay_ops(state, remove_ops={"add_auto_visuals"})

    assert state.timeline == original["timeline"]
    assert state.artifacts == original["artifacts"]
    assert state.working_file == original["working_file"]


def test_rebuild_timeline_replays_manual_visual_asset(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    state = _state(tmp_path)
    asset = Path(state.working_dir) / "visual.mp4"
    asset.write_bytes(b"visual")
    output = Path(state.working_dir) / "rebuilt.mp4"
    calls: list[list[dict]] = []
    state.timeline = [
        {
            "op": "add_visual_asset",
            "params": {
                "overlays": [
                    {"asset_path": str(asset), "start": 1.0, "end": 2.0}
                ]
            },
        }
    ]

    def fake_apply(_input: str, _working_dir: str, overlays: list[dict]) -> str:
        calls.append(overlays)
        output.write_bytes(b"rebuilt")
        return str(output)

    monkeypatch.setattr(undo, "apply_visual_overlays", fake_apply)
    monkeypatch.setattr(undo, "probe_video", lambda _path: _metadata())

    undo.rebuild_timeline(state)

    assert calls[0][0]["asset_path"] == str(asset.resolve())
    assert state.working_file == str(output)


def test_rebuild_timeline_replays_song_mix(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    state = _state(tmp_path)
    song = Path(state.working_dir) / "song.wav"
    song.write_bytes(b"song")
    output = Path(state.working_dir) / "song-rebuilt.mp4"
    calls: list[dict[str, object]] = []
    state.timeline = [
        {
            "op": "add_song",
            "params": {
                "song_path": str(song),
                "mix_plan": {"video_duration_sec": 10.0, "placements": [{"start": 0, "end": 3}]},
            },
        }
    ]

    def fake_add(video_path: str, song_path: str, working_dir: str, mix_plan: dict) -> str:
        calls.append(
            {
                "video_path": video_path,
                "song_path": song_path,
                "working_dir": working_dir,
                "mix_plan": mix_plan,
            }
        )
        output.write_bytes(b"rebuilt")
        return str(output)

    monkeypatch.setattr(undo, "add_song_to_video", fake_add)
    monkeypatch.setattr(undo, "probe_video", lambda _path: _metadata())

    undo.rebuild_timeline(state)

    assert calls[0]["song_path"] == str(song.resolve())
    assert state.working_file == str(output)


def test_rebuild_timeline_rejects_unknown_operation(tmp_path: Path) -> None:
    state = _state(tmp_path)
    state.timeline = [{"op": "future_mutation", "params": {}}]

    with pytest.raises(VideoEngineError, match="not replayable"):
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
