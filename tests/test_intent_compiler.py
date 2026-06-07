from __future__ import annotations

from pathlib import Path

import pytest

from intent_compiler import compile_intent
from state import ProjectState, utc_now_iso


def test_compiles_trim_and_export_chain_without_llm() -> None:
    plan = compile_intent("trim the first 30 seconds and export it for instagram", _state())

    assert plan is not None
    assert plan.requires_llm is False
    assert [step.tool for step in plan.steps] == ["trim_clip", "export_video"]
    assert plan.steps[0].params == {"start": "0", "end": "30"}
    assert plan.steps[1].params == {"preset_name": "instagram_reels"}


def test_compiles_common_direct_commands() -> None:
    cases = {
        "remove silent pauses": ("trim_silence", {"aggressiveness": "medium"}),
        "speed it up by 1.5x": ("adjust_speed", {"factor": 1.5}),
        "mute from 0:10 to 0:20": ("mute_segment", {"start": "10", "end": "20"}),
        "show video metadata": ("get_video_info", {}),
        "extract audio as wav": ("extract_audio", {"format": "wav"}),
        "make 3 youtube shorts": ("create_auto_shorts", {"count": 3, "target_platform": "youtube_shorts"}),
        "add 2 generated visuals": (
            "add_auto_visuals",
            {
                "force_fullscreen": True,
                "max_visuals": 2,
                "requested_count": 2,
                "coverage_policy": "target_count",
            },
        ),
        "add generated visuals with hyperframes": ("add_auto_visuals", {"force_fullscreen": True, "renderer": "hyperframes"}),
        "add generated visuals with hyperframes and manim": ("add_auto_visuals", {"force_fullscreen": True, "renderer": "both"}),
        "add 4 auto effects": ("add_auto_effects", {"max_effects": 4, "density": "medium"}),
        "add subtitle-aware auto effects": ("add_auto_effects", {"density": "medium"}),
        "use add_auto_effects": ("add_auto_effects", {"density": "medium"}),
        "auto color grade this video": ("auto_color_grade", {}),
        "give it a subtle cinematic look": ("auto_color_grade", {"look": "cinematic", "intensity": 0.65}),
        "make the colors pop": ("auto_color_grade", {"look": "vibrant"}),
        "convert this mov file to mp4 and compress it without losing much quality": (
            "plan_encode",
            {"raw_request": "convert this mov file to mp4 and compress it without losing much quality"},
        ),
        "scale this video to 1080p": ("upscale_video", {"resolution": "1920x1080", "scale_mode": "fit"}),
    }

    for command, (tool, params) in cases.items():
        plan = compile_intent(command, _state())

        assert plan is not None, command
        assert len(plan.steps) == 1
        assert plan.steps[0].tool == tool
        assert plan.steps[0].params == params


def test_explicit_auto_counts_use_target_coverage_policy() -> None:
    plan = compile_intent("add 15 B-roll clips", _state())

    assert plan is not None
    step = plan.steps[0]
    assert step.tool == "add_auto_broll"
    assert step.params["max_overlays"] == 15
    assert step.params["requested_count"] == 15
    assert step.params["coverage_policy"] == "target_count"


def test_interval_broll_request_estimates_requested_count_from_duration() -> None:
    plan = compile_intent("add B-roll about game development every 45 seconds", _state())

    assert plan is not None
    step = plan.steps[0]
    assert step.tool == "add_auto_broll"
    assert step.params["interval_sec"] == 45.0
    assert step.params["max_overlays"] == 3
    assert step.params["requested_count"] == 3
    assert step.params["coverage_policy"] == "target_count"


def test_auto_visual_density_request_is_exposed_to_tool() -> None:
    plan = compile_intent("add dense generated visuals with hyperframes", _state())

    assert plan is not None
    step = plan.steps[0]
    assert step.tool == "add_auto_visuals"
    assert step.params["density"] == "dense"
    assert step.params["renderer"] == "hyperframes"


def test_compiles_manual_visual_asset_command() -> None:
    plan = compile_intent("use this html animation assets/visual.html from 00:12 to 00:16", _state())

    assert plan is not None
    step = plan.steps[0]
    assert step.tool == "add_visual_asset"
    assert step.params == {
        "asset_path": "assets/visual.html",
        "start": "12",
        "end": "16",
        "composition_mode": "replace",
    }


def test_subtitle_command_compiles_transcribe_then_burn_when_srt_missing(tmp_path: Path) -> None:
    state = _state(tmp_path)
    plan = compile_intent("add subtitles at the bottom", state)

    assert plan is not None
    assert [step.tool for step in plan.steps] == ["transcribe_video", "burn_subtitles"]
    assert plan.steps[1].params == {"position": "bottom"}


def test_subtitle_command_skips_transcribe_when_srt_exists(tmp_path: Path) -> None:
    state = _state(tmp_path)
    (tmp_path / "transcript.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")

    plan = compile_intent("add subtitles", state)

    assert plan is not None
    assert [step.tool for step in plan.steps] == ["burn_subtitles"]


def test_subtitle_command_transcribes_when_srt_is_symlink_outside_project(tmp_path: Path) -> None:
    state = _state(tmp_path)
    outside_srt = tmp_path.parent / "outside-transcript.srt"
    outside_srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nOutside\n", encoding="utf-8")
    try:
        (tmp_path / "transcript.srt").symlink_to(outside_srt)
    except OSError as exc:
        pytest.skip(f"Symlink creation is unavailable: {exc}")

    plan = compile_intent("add subtitles", state)

    assert plan is not None
    assert [step.tool for step in plan.steps] == ["transcribe_video", "burn_subtitles"]


def test_subtitle_command_detects_style_preset(tmp_path: Path) -> None:
    state = _state(tmp_path)
    (tmp_path / "transcript.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")

    plan = compile_intent("add cinematic subtitles at the top", state)

    assert plan is not None
    assert [step.tool for step in plan.steps] == ["burn_subtitles"]
    assert plan.steps[0].params == {"position": "top", "style": "cinematic"}


def test_compiles_typed_blender_3d_visual_command() -> None:
    plan = compile_intent(
        "add a rotating 3d title saying Matrix Multiplication at 00:18",
        _state(),
    )

    assert plan is not None
    step = plan.steps[0]
    assert step.tool == "add_auto_visuals"
    assert step.params["renderer"] == "blender"
    assert step.params["force_fullscreen"] is True
    spec = step.params["manual_visual_specs"][0]
    assert spec["template"] == "three_d_title"
    assert spec["start"] == "18"
    assert spec["end"] == "22"
    assert spec["object_motion"] == "spin_y"
    assert spec["headline"] == "matrix multiplication"


def test_compiles_blender_overlay_and_model_asset_command() -> None:
    plan = compile_intent(
        "add a 3d product spin overlay using assets/model.glb from 5s to 9s",
        _state(),
    )

    assert plan is not None
    spec = plan.steps[0].params["manual_visual_specs"][0]
    assert plan.steps[0].params["force_fullscreen"] is False
    assert spec["template"] == "product_model_spin"
    assert spec["composition_mode"] == "overlay"
    assert spec["asset_path"] == "assets/model.glb"
    assert spec["start"] == "5"
    assert spec["end"] == "9"


def test_compiles_blender_triggered_data_tunnel_command() -> None:
    plan = compile_intent("add a cinematic glowing chip animation when i say GPU", _state())

    assert plan is not None
    spec = plan.steps[0].params["manual_visual_specs"][0]
    assert spec["template"] == "data_tunnel"
    assert spec["trigger_text"] == "gpu"
    assert "start" not in spec
    assert "end" not in spec


def test_ambiguous_instruction_falls_back_to_llm() -> None:
    assert compile_intent("make this video more professional", _state()) is None


def _state(tmp_path: Path | None = None) -> ProjectState:
    base = tmp_path or Path("D:/tmp/vex-test")
    now = utc_now_iso()
    return ProjectState(
        project_id="test-project",
        project_name="Test Project",
        created_at=now,
        updated_at=now,
        source_files=[str(base / "source.mp4")],
        working_file=str(base / "working.mp4"),
        working_dir=str(base),
        output_dir=str(base / "out"),
        metadata={"duration_sec": 120.0, "width": 1920, "height": 1080, "fps": 30.0},
        provider="test",
        model="test-model",
    )
