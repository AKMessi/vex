from __future__ import annotations

from pathlib import Path

import engine
import prompts
import tools.song as song_tool
from intent_compiler import compile_intent
from state import ProjectState, utc_now_iso
from tools import TOOL_CONTRACTS
from tools.song_director import build_song_mix_plan


def test_song_mix_plan_selects_voiceover_bed_and_looping() -> None:
    plan = build_song_mix_plan(
        params={"song_path": "music.mp3"},
        source_metadata=_metadata(duration=20.0, has_audio=True),
        song_metadata=_song_metadata(duration=4.0),
    )

    assert plan.selected_skill_id == "voiceover_bed"
    assert plan.preserve_original_audio is True
    assert plan.ducking_enabled is True
    assert plan.loop_song is True
    assert plan.placements[0].start == 0.0
    assert plan.placements[0].end == 20.0


def test_song_mix_plan_uses_soundtrack_for_silent_video() -> None:
    plan = build_song_mix_plan(
        params={"mode": "auto"},
        source_metadata=_metadata(duration=12.0, has_audio=False),
        song_metadata=_song_metadata(duration=30.0),
    )

    assert plan.selected_skill_id == "silent_video_soundtrack"
    assert plan.preserve_original_audio is False
    assert plan.ducking_enabled is False
    assert plan.music_volume > 0.3


def test_song_mix_plan_supports_intro_outro_bookends() -> None:
    plan = build_song_mix_plan(
        params={"mode": "intro_outro", "duration": 5},
        source_metadata=_metadata(duration=30.0, has_audio=True),
        song_metadata=_song_metadata(duration=8.0),
    )

    assert plan.selected_skill_id == "intro_outro_sting"
    assert [(item.placement_id, item.start, item.end) for item in plan.placements] == [
        ("intro", 0.0, 5.0),
        ("outro", 25.0, 30.0),
    ]


def test_engine_add_song_builds_auditable_ducked_filtergraph(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    commands: list[list[str]] = []
    output_path = tmp_path / "mixed.mp4"
    filtergraph_path = tmp_path / "filtergraph.txt"
    plan = build_song_mix_plan(
        params={"mode": "background"},
        source_metadata=_metadata(duration=10.0, has_audio=True),
        song_metadata=_song_metadata(duration=2.0),
    )

    monkeypatch.setattr(engine, "probe_video", lambda _path: _metadata(duration=10.0, has_audio=True))
    monkeypatch.setattr(engine, "_unique_path", lambda _working_dir, _suffix: str(output_path))
    monkeypatch.setattr(engine, "_run_command", lambda command, _message: commands.append(command))

    result = engine.add_song_to_video(
        "video.mp4",
        "song.mp3",
        str(tmp_path),
        plan.to_dict(),
        filtergraph_path=str(filtergraph_path),
    )

    assert result == str(output_path)
    command = commands[0]
    assert "-stream_loop" in command
    assert command[command.index("-map") + 1] == "0:v:0"
    filter_graph = filtergraph_path.read_text(encoding="utf-8")
    assert "sidechaincompress" in filter_graph
    assert "amix=inputs=2" in filter_graph
    assert "loudnorm" in filter_graph
    assert "adelay=0:all=1" in filter_graph


def test_add_song_tool_promotes_only_passing_mix(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    state = _state(tmp_path)
    song_path = Path(state.working_dir) / "track.wav"
    song_path.write_bytes(b"song")
    output_path = Path(state.working_dir) / "mixed.mp4"

    def fake_add_song_to_video(*_args: object, filtergraph_path: str | None = None, **_kwargs: object) -> str:
        output_path.write_bytes(b"mixed-video")
        if filtergraph_path:
            Path(filtergraph_path).write_text("[mixed]anull[a]", encoding="utf-8")
        return str(output_path)

    def fake_probe(path: str) -> dict[str, object]:
        if Path(path) == song_path:
            return _song_metadata(duration=5.0)
        return _metadata(duration=10.0, has_audio=True, size_bytes=output_path.stat().st_size if output_path.exists() else 10)

    monkeypatch.setattr(song_tool, "add_song_to_video", fake_add_song_to_video)
    monkeypatch.setattr(song_tool, "probe_video", fake_probe)
    monkeypatch.setattr(
        song_tool,
        "evaluate_song_mix_output",
        lambda **_kwargs: {"passed": True, "score": 0.93, "issues": [], "warnings": [], "evidence": {}},
    )

    result = song_tool.execute({"song_path": str(song_path), "mode": "background"}, state)

    assert result["success"] is True
    assert result["asset_id"].startswith("asset_")
    assert state.working_file == str(output_path.resolve())
    assert state.timeline[-1]["op"] == "add_song"
    assert state.timeline[-1]["params"]["selected_skill_id"] == "voiceover_bed"
    assert Path(state.artifacts["latest_added_song"]["manifest_path"]).is_file()


def test_add_song_rejects_model_input_outside_project(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    state = _state(tmp_path)
    outside_song = tmp_path.parent / "outside-song.wav"
    outside_song.write_bytes(b"song")
    called = False

    def fake_add_song_to_video(*_args: object, **_kwargs: object) -> str:
        nonlocal called
        called = True
        return str(Path(state.working_dir) / "mixed.mp4")

    monkeypatch.setattr(song_tool, "add_song_to_video", fake_add_song_to_video)

    result = song_tool.execute({"song_path": str(outside_song)}, state)

    assert not result["success"]
    assert "must stay inside" in result["message"]
    assert not called


def test_add_song_is_registered_for_tools_prompts_and_intent() -> None:
    assert "add_song" in TOOL_CONTRACTS
    assert TOOL_CONTRACTS["add_song"].replayable
    assert any(schema["name"] == "add_song" for schema in prompts.TOOL_SCHEMAS)

    plan = compile_intent("add background music assets/beat.mp3 from 5s to 9s at 20% volume", _state(Path("D:/tmp/vex-song-test")))

    assert plan is not None
    step = plan.steps[0]
    assert step.tool == "add_song"
    assert step.params == {
        "song_path": "assets/beat.mp3",
        "mode": "segment",
        "start": "5",
        "end": "9",
        "volume": 0.2,
    }


def _state(tmp_path: Path) -> ProjectState:
    project_dir = tmp_path / "project"
    project_dir.mkdir(parents=True, exist_ok=True)
    working = project_dir / "working.mp4"
    working.write_bytes(b"source-video")
    now = utc_now_iso()
    return ProjectState(
        project_id="test-project",
        project_name="Song Test",
        created_at=now,
        updated_at=now,
        source_files=[str(working)],
        working_file=str(working),
        working_dir=str(project_dir),
        output_dir=str(project_dir / "out"),
        metadata=_metadata(duration=10.0, has_audio=True, size_bytes=working.stat().st_size),
        provider="test",
        model="test-model",
    )


def _metadata(*, duration: float, has_audio: bool, size_bytes: int = 1024) -> dict[str, object]:
    return {
        "duration_sec": duration,
        "fps": 30.0,
        "width": 1920,
        "height": 1080,
        "codec": "h264",
        "video_bit_rate": 8_000_000,
        "has_audio": has_audio,
        "audio_codec": "aac" if has_audio else None,
        "audio_bit_rate": 128_000 if has_audio else 0,
        "audio_channels": 2 if has_audio else 0,
        "size_bytes": size_bytes,
        "format": "mov,mp4,m4a,3gp,3g2,mj2",
    }


def _song_metadata(*, duration: float) -> dict[str, object]:
    return {
        "duration_sec": duration,
        "fps": 0.0,
        "width": 0,
        "height": 0,
        "codec": "unknown",
        "has_audio": True,
        "audio_codec": "pcm_s16le",
        "audio_bit_rate": 705_600,
        "audio_channels": 2,
        "size_bytes": 4096,
        "format": "wav",
    }
