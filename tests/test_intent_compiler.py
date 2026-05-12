from __future__ import annotations

from pathlib import Path

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
        "add 2 generated visuals": ("add_auto_visuals", {"force_fullscreen": True, "max_visuals": 2}),
        "add generated visuals with hyperframes": ("add_auto_visuals", {"force_fullscreen": True, "renderer": "hyperframes"}),
        "convert this mov file to mp4 and compress it without losing much quality": (
            "plan_encode",
            {"raw_request": "convert this mov file to mp4 and compress it without losing much quality"},
        ),
    }

    for command, (tool, params) in cases.items():
        plan = compile_intent(command, _state())

        assert plan is not None, command
        assert len(plan.steps) == 1
        assert plan.steps[0].tool == tool
        assert plan.steps[0].params == params


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
