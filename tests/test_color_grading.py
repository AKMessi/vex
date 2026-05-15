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
    assert 0.78 <= plan.adjustments["blue_gain"] <= 1.12
    assert 0.88 <= plan.adjustments["red_gain"] <= 1.30


def test_white_balance_prefers_neutral_midtones_over_saturated_regions() -> None:
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    frame[:, :40] = np.array([190, 42, 38], dtype=np.uint8)
    frame[:, 40:] = np.array([30, 62, 210], dtype=np.uint8)
    frame[24:56, 24:56] = np.array([92, 94, 112], dtype=np.uint8)

    plan = color_grading.build_color_grade_plan_from_frames([frame], look="natural", intensity=1.0)

    assert plan.analysis.neutral_pixel_fraction > 0.05
    assert plan.analysis.white_balance_confidence > 0.4
    assert plan.adjustments["red_gain"] > plan.adjustments["blue_gain"]
    assert plan.adjustments["green_gain"] > plan.adjustments["blue_gain"]


def test_cinematic_grade_adds_bounded_levels_and_curve_when_source_is_flat() -> None:
    frame = _gradient_frame(red_scale=1.0, green_scale=1.0, blue_scale=1.0, low=70, high=165)

    plan = color_grading.build_color_grade_plan_from_frames([frame], look="cinematic", intensity=1.0)

    assert 0.0 <= plan.adjustments["level_input_black"] <= 0.08
    assert 0.90 <= plan.adjustments["level_input_white"] <= 1.0
    assert plan.adjustments["curve_shadow"] < 0.25
    assert plan.adjustments["curve_highlight"] > 0.75
    assert "colorlevels=" in plan.filter_graph
    assert "curves=" in plan.filter_graph


def test_good_balanced_source_gets_subtle_auto_correction() -> None:
    frame = _balanced_color_frame()

    plan = color_grading.build_color_grade_plan_from_frames([frame], look="auto", intensity=1.0)

    assert plan.adjustments["overall_need"] < 0.25
    assert plan.adjustments["correction_strength"] < 0.75
    assert abs(plan.adjustments["brightness"]) < 0.03
    assert 0.94 <= plan.adjustments["red_gain"] <= 1.06
    assert 0.94 <= plan.adjustments["blue_gain"] <= 1.06


def test_severely_underexposed_blue_cast_source_gets_strong_correction() -> None:
    frame = _gradient_frame(red_scale=0.72, green_scale=0.82, blue_scale=1.25, low=8, high=62)

    plan = color_grading.build_color_grade_plan_from_frames([frame], look="auto", intensity=1.0)

    assert plan.adjustments["overall_need"] > 0.65
    assert plan.adjustments["correction_strength"] > 1.1
    assert plan.adjustments["brightness"] > 0.12
    assert plan.adjustments["contrast"] > 1.25
    assert plan.adjustments["level_input_white"] < 0.92
    assert plan.adjustments["red_gain"] > 1.12
    assert plan.adjustments["blue_gain"] < 0.9


def test_shot_aware_plan_grades_bad_shot_more_aggressively_than_good_shot() -> None:
    balanced = _balanced_color_frame()
    underexposed_blue = _gradient_frame(red_scale=0.72, green_scale=0.82, blue_scale=1.25, low=8, high=62)

    plan = color_grading.build_shot_aware_color_grade_plan_from_shots(
        [
            (0.0, 4.0, [balanced, balanced]),
            (4.0, 8.0, [underexposed_blue, underexposed_blue]),
        ],
        look="auto",
        intensity=1.0,
    )

    assert plan.render_mode == "filter_complex"
    assert "concat=n=2:v=1:a=0[vout]" in plan.filter_graph
    assert plan.manifest is not None
    first_shot, second_shot = plan.manifest.shots
    assert first_shot.correction_need < 0.30
    assert second_shot.correction_need > 0.65
    assert second_shot.selected_adjustments["brightness"] > first_shot.selected_adjustments["brightness"] + 0.08
    assert second_shot.selected_adjustments["red_gain"] > second_shot.selected_adjustments["blue_gain"]
    assert second_shot.selected_candidate_id != "source-0"


def test_shot_aware_candidate_selection_keeps_balanced_single_shot_subtle() -> None:
    frame = _balanced_color_frame()

    plan = color_grading.build_shot_aware_color_grade_plan_from_shots(
        [(0.0, 3.0, [frame, frame])],
        look="auto",
        intensity=1.0,
    )

    assert plan.render_mode == "vf"
    assert plan.manifest is not None
    shot = plan.manifest.shots[0]
    assert shot.selected_adjustments["overall_need"] < 0.25
    assert abs(shot.selected_adjustments["brightness"]) < 0.03
    assert shot.selected_score >= 0.65
    assert len(shot.candidates) >= 2
    assert any(candidate.candidate_id == "source-0" for candidate in shot.candidates)


def test_cinematic_grade_protects_good_skin_tone_source_from_overgrading() -> None:
    frame = _balanced_skin_frame()

    plan = color_grading.build_shot_aware_color_grade_plan_from_shots(
        [(0.0, 3.0, [frame, frame])],
        look="cinematic",
        intensity=1.0,
    )

    assert plan.manifest is not None
    shot = plan.manifest.shots[0]
    adjustments = shot.selected_adjustments
    assert shot.correction_need < 0.35
    assert adjustments["style_strength"] < 0.35
    assert 0.985 <= adjustments["saturation"] <= 1.055
    assert 0.985 <= adjustments["red_gain"] <= 1.025
    assert 0.985 <= adjustments["blue_gain"] <= 1.025


def test_color_grade_validation_flags_extreme_clipping() -> None:
    frame = np.full((32, 32, 3), 255, dtype=np.uint8)

    validation = color_grading.validate_color_grade_analysis(color_grading.analyze_frames([frame]))

    assert validation["passed"] is False
    assert validation["score"] < 0.9
    assert any("highlight" in warning for warning in validation["warnings"])


def test_apply_color_grade_uses_rebuildable_ffmpeg_filter(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    commands: list[list[str]] = []
    monkeypatch.setattr(engine, "_run_command", lambda command, _message: commands.append(command))

    output_path = engine.apply_color_grade("input.mp4", str(tmp_path), "eq=contrast=1.05,format=yuv420p")

    command = commands[0]
    assert command[command.index("-vf") + 1] == "eq=contrast=1.05,format=yuv420p"
    assert "0:a?" in command
    assert "-pix_fmt" in command
    assert output_path.endswith(".mp4")


def test_apply_color_grade_supports_shot_aware_filter_complex(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    commands: list[list[str]] = []
    monkeypatch.setattr(engine, "_run_command", lambda command, _message: commands.append(command))
    filter_complex = "[0:v]trim=start=0:end=1,setpts=PTS-STARTPTS,format=yuv420p[v0];[v0]concat=n=1:v=1:a=0[vout]"

    output_path = engine.apply_color_grade(
        "input.mp4",
        str(tmp_path),
        filter_complex,
        render_mode="filter_complex",
        output_label="[vout]",
    )

    command = commands[0]
    assert command[command.index("-filter_complex") + 1] == filter_complex
    assert command[command.index("-map") + 1] == "[vout]"
    assert "0:a?" in command
    assert "-shortest" in command
    assert output_path.endswith(".mp4")


def test_color_grade_tool_records_filter_for_timeline_rebuild(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    plan = {
        "resolved_look": "cinematic",
        "filter_graph": "format=rgb24,eq=contrast=1.05,format=yuv420p",
        "render_mode": "vf",
        "output_label": "",
        "adjustments": {
            "brightness": 0.01,
            "contrast": 1.05,
            "saturation": 1.08,
            "red_gain": 1.01,
            "green_gain": 1.0,
            "blue_gain": 0.99,
        },
        "analysis": {"sample_count": 5},
        "manifest": None,
        "validation": {"passed": True, "score": 0.97, "warnings": [], "analysis": {}},
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
    assert state.timeline[-1]["params"]["render_mode"] == "vf"
    assert state.timeline[-1]["params"]["validation"]["score"] == 0.97
    assert state.artifacts["latest_auto_color_grade"]["resolved_look"] == "cinematic"


def test_rebuild_timeline_reuses_stored_color_grade_filter(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        undo_tool,
        "apply_color_grade",
        lambda input_path, _working_dir, filter_graph, **_kwargs: calls.append((input_path, filter_graph)) or str(tmp_path / "rebuilt.mp4"),
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


def _balanced_color_frame() -> np.ndarray:
    y_coords, x_coords = np.mgrid[0:96, 0:96].astype(np.float32)
    x_coords /= 95
    y_coords /= 95
    base = 0.20 + (0.62 * ((x_coords + y_coords) / 2))
    red = np.clip(base + 0.10 * np.sin(x_coords * 6.28), 0, 1)
    green = np.clip(base + 0.08 * np.sin((y_coords * 6.28) + 1.2), 0, 1)
    blue = np.clip(base + 0.09 * np.sin(((x_coords - y_coords) * 6.28) + 2.1), 0, 1)
    return (np.stack([red, green, blue], axis=2) * 255).astype(np.uint8)


def _balanced_skin_frame() -> np.ndarray:
    y_coords, x_coords = np.mgrid[0:96, 0:96].astype(np.float32)
    x_coords /= 95
    y_coords /= 95
    base = 0.42 + (0.25 * ((x_coords + y_coords) / 2))
    red = np.clip(base + 0.16, 0, 1)
    green = np.clip(base + 0.06, 0, 1)
    blue = np.clip(base - 0.03, 0, 1)
    frame = np.stack([red, green, blue], axis=2)
    frame[10:32, 10:86] = np.array([0.55, 0.56, 0.55], dtype=np.float32)
    frame[64:86, 16:80] = np.array([0.18, 0.19, 0.20], dtype=np.float32)
    return (np.clip(frame, 0, 1) * 255).astype(np.uint8)


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
