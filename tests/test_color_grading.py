from __future__ import annotations

from pathlib import Path

import numpy as np

import color_grading
import engine
import tools.color_grade as color_grade_tool
import tools.undo as undo_tool
from state import ProjectState, utc_now_iso


def test_color_grade_plan_corrects_underexposed_low_saturation_frames() -> None:
    frame = _gradient_frame(red_scale=0.92, green_scale=0.90, blue_scale=0.88, low=32, high=118)

    plan = color_grading.build_color_grade_plan_from_frames([frame], look="natural", intensity=1.0)

    assert plan.adjustments["brightness"] > 0
    assert plan.adjustments["contrast"] > 1.0
    assert plan.adjustments["saturation"] > 1.0
    assert "lutrgb=" in plan.filter_graph
    assert "eq=" in plan.filter_graph
    assert plan.filter_graph.endswith("format=yuv420p")


def test_color_grade_plan_reduces_blue_cast_with_bounded_white_balance() -> None:
    frame = _gradient_frame(red_scale=0.72, green_scale=0.84, blue_scale=1.24, low=64, high=190)

    plan = color_grading.build_color_grade_plan_from_frames([frame], look="auto", intensity=1.0)

    assert plan.resolved_look == "natural"
    assert plan.adjustments["red_gain"] > plan.adjustments["blue_gain"]
    assert 0.88 <= plan.adjustments["blue_gain"] <= 1.12
    assert 0.88 <= plan.adjustments["red_gain"] <= 1.12


def test_apply_color_grade_uses_rebuildable_ffmpeg_filter(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    commands: list[list[str]] = []
    monkeypatch.setattr(engine, "_run_command", lambda command, _message: commands.append(command))

    output_path = engine.apply_color_grade("input.mp4", str(tmp_path), "eq=contrast=1.05,format=yuv420p")

    command = commands[0]
    assert command[command.index("-vf") + 1] == "eq=contrast=1.05,format=yuv420p"
    assert "0:a?" in command
    assert "-pix_fmt" in command
    assert output_path.endswith(".mp4")


def test_color_grade_tool_records_filter_for_timeline_rebuild(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    plan = {
        "resolved_look": "cinematic",
        "filter_graph": "format=rgb24,eq=contrast=1.05,format=yuv420p",
        "adjustments": {
            "brightness": 0.01,
            "contrast": 1.05,
            "saturation": 1.08,
            "red_gain": 1.01,
            "green_gain": 1.0,
            "blue_gain": 0.99,
        },
        "analysis": {"sample_count": 5},
        "warnings": [],
    }
    monkeypatch.setattr(color_grade_tool, "auto_color_grade", lambda *_args, **_kwargs: (str(tmp_path / "graded.mp4"), plan))
    monkeypatch.setattr(color_grade_tool, "probe_video", lambda _path: _metadata())
    state = _state(tmp_path)

    result = color_grade_tool.execute({"look": "cinematic", "intensity": 0.8, "sample_count": 5}, state)

    assert result["success"] is True
    assert state.working_file == str(tmp_path / "graded.mp4")
    assert state.timeline[-1]["op"] == "auto_color_grade"
    assert state.timeline[-1]["params"]["filter_graph"] == plan["filter_graph"]
    assert state.artifacts["latest_auto_color_grade"]["resolved_look"] == "cinematic"


def test_rebuild_timeline_reuses_stored_color_grade_filter(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        undo_tool,
        "apply_color_grade",
        lambda input_path, _working_dir, filter_graph: calls.append((input_path, filter_graph)) or str(tmp_path / "rebuilt.mp4"),
    )
    monkeypatch.setattr(undo_tool, "probe_video", lambda _path: _metadata())
    state = _state(tmp_path)
    state.source_files = [str(source)]
    state.timeline = [
        {
            "op": "auto_color_grade",
            "params": {"filter_graph": "format=rgb24,eq=contrast=1.05,format=yuv420p"},
            "description": "Applied color grade",
        }
    ]

    undo_tool.rebuild_timeline(state)

    assert calls == [(str(source), "format=rgb24,eq=contrast=1.05,format=yuv420p")]
    assert state.working_file == str(tmp_path / "rebuilt.mp4")


def _gradient_frame(
    *,
    red_scale: float,
    green_scale: float,
    blue_scale: float,
    low: int,
    high: int,
) -> np.ndarray:
    base = np.linspace(low, high, 64, dtype=np.float32).reshape(8, 8)
    frame = np.stack(
        [
            np.clip(base * red_scale, 0, 255),
            np.clip(base * green_scale, 0, 255),
            np.clip(base * blue_scale, 0, 255),
        ],
        axis=2,
    )
    return frame.astype(np.uint8)


def _state(tmp_path: Path) -> ProjectState:
    now = utc_now_iso()
    return ProjectState(
        project_id="test-color-grade",
        project_name="Test Color Grade",
        created_at=now,
        updated_at=now,
        source_files=[str(tmp_path / "source.mp4")],
        working_file=str(tmp_path / "working.mp4"),
        working_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
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
