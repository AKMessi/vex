from __future__ import annotations

from pathlib import Path

import pytest

from state import ProjectState, utc_now_iso
from tools import audio, auto_visuals, merge, speed, transitions


@pytest.mark.parametrize("params", [{}, {"factor": "fast"}, {"factor": float("inf")}])
def test_adjust_speed_rejects_invalid_factor_without_rendering(monkeypatch, tmp_path: Path, params: dict) -> None:  # noqa: ANN001
    state = _state(tmp_path)
    monkeypatch.setattr(speed, "adjust_speed", lambda *_args: pytest.fail("render should not run"))

    result = speed.execute(params, state)

    assert result["success"] is False


def test_adjust_speed_supports_end_only_segment(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    state = _state(tmp_path)
    output = str(Path(state.working_dir) / "speed.mp4")
    monkeypatch.setattr(speed, "adjust_speed", lambda *_args: output)
    monkeypatch.setattr(speed, "probe_video", lambda _path: _metadata())

    result = speed.execute({"factor": 1.5, "end": "5"}, state)

    assert result["success"] is True
    assert "segment start to 5" in result["message"]


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"type": "fade_in", "duration": 0, "position": "start"},
        {"type": "wipe", "duration": 1, "position": "start"},
        {"type": "fade_in", "duration": 1, "position": "middle"},
    ],
)
def test_transition_rejects_invalid_params_without_rendering(monkeypatch, tmp_path: Path, params: dict) -> None:  # noqa: ANN001
    state = _state(tmp_path)
    monkeypatch.setattr(transitions, "fade_in", lambda *_args: pytest.fail("render should not run"))
    monkeypatch.setattr(transitions, "fade_out", lambda *_args: pytest.fail("render should not run"))

    result = transitions.execute(params, state)

    assert result["success"] is False


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"file_paths": []},
        {"file_paths": "clip.mp4"},
        {"file_paths": [""]},
    ],
)
def test_merge_rejects_invalid_path_lists_without_rendering(monkeypatch, tmp_path: Path, params: dict) -> None:  # noqa: ANN001
    state = _state(tmp_path)
    monkeypatch.setattr(merge, "merge", lambda *_args: pytest.fail("render should not run"))

    result = merge.execute(params, state)

    assert result["success"] is False


@pytest.mark.parametrize(
    "params",
    [
        {"mix_with_original": "false"},
        {"mix_ratio": -0.1},
        {"mix_ratio": 1.1},
        {"mix_ratio": float("nan")},
    ],
)
def test_replace_audio_rejects_invalid_mix_settings(monkeypatch, tmp_path: Path, params: dict) -> None:  # noqa: ANN001
    state = _state(tmp_path)
    audio_path = Path(state.working_dir) / "song.wav"
    audio_path.write_bytes(b"audio")
    params = {"audio_path": str(audio_path), **params}
    monkeypatch.setattr(audio, "replace_audio", lambda *_args, **_kwargs: pytest.fail("render should not run"))

    result = audio.execute_replace(params, state)

    assert result["success"] is False


def test_mutating_tool_restores_state_when_operation_save_fails(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    state = _state(tmp_path)
    original_file = state.working_file
    original_metadata = state.metadata.copy()
    output = str(Path(state.working_dir) / "speed.mp4")
    monkeypatch.setattr(speed, "adjust_speed", lambda *_args: output)
    monkeypatch.setattr(speed, "probe_video", lambda _path: {**_metadata(), "duration_sec": 5.0})
    monkeypatch.setattr(
        state,
        "apply_operation",
        lambda _op: (_ for _ in ()).throw(OSError("disk full")),
    )

    result = speed.execute({"factor": 2.0}, state)

    assert result["success"] is False
    assert "disk full" in result["message"]
    assert state.working_file == original_file
    assert state.metadata == original_metadata
    assert state.timeline == []


def test_audio_tools_return_failures_for_invalid_boundary_input(tmp_path: Path) -> None:
    state = _state(tmp_path)

    extract_result = audio.execute_extract({"format": "flac"}, state)
    mute_result = audio.execute_mute({"start": "0"}, state)

    assert extract_result["success"] is False
    assert mute_result["success"] is False


def test_auto_visuals_restores_state_when_directed_execution_returns_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:  # noqa: ANN001
    state = _state(tmp_path)
    original_file = state.working_file
    monkeypatch.setattr(
        auto_visuals,
        "_directed_hyperframes_specs_from_params",
        lambda _params: [{"visual_id": "directed-1"}],
    )
    monkeypatch.setattr(auto_visuals, "_manual_visual_specs_from_params", lambda _params: [])

    def fail_after_mutation(*_args: object, **_kwargs: object) -> dict[str, object]:
        state.working_file = str(Path(state.working_dir) / "failed.mp4")
        state.artifacts["partial"] = True
        return {
            "success": False,
            "message": "directed render rejected",
            "updated_state": state,
            "tool_name": "add_auto_visuals",
        }

    monkeypatch.setattr(
        auto_visuals,
        "_execute_directed_hyperframes_specs",
        fail_after_mutation,
    )

    result = auto_visuals.execute({}, state)

    assert result["success"] is False
    assert state.working_file == original_file
    assert "partial" not in state.artifacts


def _state(tmp_path: Path) -> ProjectState:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    source = project_dir / "source.mp4"
    source.write_bytes(b"video")
    now = utc_now_iso()
    return ProjectState(
        project_id="tool-transaction-test",
        project_name="Tool Transaction Test",
        created_at=now,
        updated_at=now,
        source_files=[str(source)],
        working_file=str(source),
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
        "size_bytes": 5,
        "format": "mov,mp4",
    }
