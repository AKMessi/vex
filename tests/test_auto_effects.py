from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

import engine
import tools.auto_effects as auto_effects_tool
from effects.compiler import build_effect_filter_graph
from effects.context import build_effect_context
from effects.motion import direct_effect_plan, validate_motion_plan_payload
from effects.planner import plan_subtitle_effects
from effects.preview import validate_effect_preview
from effects.qa import validate_effect_plan
from effects.schema import EffectInstance, EffectPlan
from effects.signals import build_subtitle_cards
from intent_compiler import compile_intent
from state import ProjectState, utc_now_iso


def test_subtitle_cards_score_emphasis_signals() -> None:
    cards = build_subtitle_cards(
        [
            {"start": 0.0, "end": 1.2, "text": "Why does this break?"},
            {"start": 1.7, "end": 3.0, "text": "Because 80% of the pipeline is wrong."},
        ],
        [],
        6.0,
        scene_cuts=[1.65],
    )

    assert len(cards) == 2
    assert cards[0]["signals"]["question"] is True
    assert cards[0]["priority"] >= 50
    assert cards[1]["signals"]["numeric_hits"] == 1
    assert cards[1]["signals"]["pause_before"] >= 0.4


def test_planner_selects_subtitle_aware_effects_and_style_modifiers() -> None:
    cards = build_subtitle_cards(
        [
            {"start": 0.2, "end": 1.1, "text": "Wait, look at this number."},
            {"start": 2.0, "end": 3.1, "text": "But this is where everything breaks."},
            {"start": 6.0, "end": 7.2, "text": "Finally, this is the takeaway."},
        ],
        [],
        12.0,
    )

    plan = plan_subtitle_effects(
        cards,
        12.0,
        max_effects=8,
        density="high",
        include_style_effects=True,
        subtitle_highlight_enabled=True,
    )

    assert plan.effects
    assert any(effect.effect_type in {"freeze_accent", "impact_pulse", "snap_reframe", "subtle_shake"} for effect in plan.effects)
    assert any("subtitle_highlight" in effect.modifiers for effect in plan.effects)
    assert plan.to_dict()["compiler_version"] == "effects-ffmpeg-v1"


def test_planner_falls_back_to_best_subtitle_when_medium_threshold_is_too_strict() -> None:
    cards = build_subtitle_cards(
        [
            {"start": 3.0, "end": 4.0, "text": "This part keeps moving forward."},
            {"start": 8.0, "end": 9.1, "text": "The setup continues here."},
        ],
        [],
        12.0,
    )

    assert max(float(card["priority"]) for card in cards) < 52.0
    plan = plan_subtitle_effects(cards, 12.0, max_effects=4, density="medium")

    assert plan.effects
    assert plan.metadata["fallback_used"] is True
    assert plan.effects[0].source_card_id


def test_effect_context_models_video_phase_pacing_and_scene_risk() -> None:
    cards = build_subtitle_cards(
        [
            {"start": 0.2, "end": 1.2, "text": "Why does this matter?"},
            {"start": 4.0, "end": 5.4, "text": "Then the pipeline becomes repeatable."},
            {"start": 10.2, "end": 11.0, "text": "Finally, this is the takeaway."},
        ],
        [],
        14.0,
        scene_cuts=[1.25, 8.5],
    )

    context = build_effect_context(
        cards,
        clip_duration=14.0,
        scene_cuts=[1.25, 8.5],
        blocked_ranges=[(6.0, 7.0)],
        metadata={"width": 1920, "height": 1080, "fps": 30.0},
    )

    first_context = context.card_contexts[cards[0]["card_id"]]
    last_context = context.card_contexts[cards[-1]["card_id"]]

    assert context.timeline["planner"] == "contextual_effect_director_v1"
    assert context.timeline["scene_count"] == 3
    assert first_context["phase"] == "opening"
    assert first_context["narrative_role"] == "hook"
    assert first_context["visual_risk"] > 0.0
    assert last_context["phase"] in {"payoff", "closing"}


def test_contextual_planner_uses_video_context_for_restrained_motion() -> None:
    cards = build_subtitle_cards(
        [
            {"start": 0.4, "end": 1.2, "text": "Wait, this is where everything breaks."},
            {"start": 4.0, "end": 5.2, "text": "Because 80% of the pipeline is wrong."},
            {"start": 8.2, "end": 9.4, "text": "Finally, this is the takeaway."},
        ],
        [],
        12.0,
        scene_cuts=[1.22, 7.8],
    )
    context = build_effect_context(cards, clip_duration=12.0, scene_cuts=[1.22, 7.8])

    plan = plan_subtitle_effects(
        cards,
        12.0,
        max_effects=8,
        density="high",
        intensity="high",
        include_style_effects=True,
        effect_context=context,
    )

    assert plan.effects
    assert plan.metadata["planner"] == "contextual_subtitle_effects_v2"
    assert plan.metadata["timeline_rhythm"] in {"balanced", "fast", "slow"}
    assert "subtle_shake" not in {effect.effect_type for effect in plan.effects}
    assert any(effect.params.get("motion_smoothing") == "contextual" for effect in plan.effects)
    assert any("context_role" in effect.params for effect in plan.effects)


def test_motion_director_builds_bounded_camera_plan() -> None:
    cards = build_subtitle_cards(
        [
            {"start": 0.4, "end": 1.2, "text": "Wait, this number changes everything."},
            {"start": 4.0, "end": 5.2, "text": "But this is where the workflow breaks."},
            {"start": 8.2, "end": 9.4, "text": "Finally, this is the takeaway."},
        ],
        [],
        12.0,
        scene_cuts=[1.3, 7.8],
    )
    context = build_effect_context(cards, clip_duration=12.0, scene_cuts=[1.3, 7.8])
    plan = plan_subtitle_effects(
        cards,
        12.0,
        max_effects=8,
        density="high",
        intensity="high",
        include_style_effects=True,
        effect_context=context,
    )

    directed, motion_plan = direct_effect_plan(
        plan,
        clip_duration=12.0,
        width=1920,
        height=1080,
    )

    payload = directed.to_dict()
    assert payload["metadata"]["motion_plan"]["version"] == "motion-director-v1"
    assert motion_plan.qa.passed
    assert motion_plan.camera_segments
    assert all(segment.target_scale <= motion_plan.max_scale for segment in motion_plan.camera_segments)
    assert all(effect.params["motion_director_version"] == "motion-director-v1" for effect in directed.effects)


def test_motion_plan_compiler_uses_directed_non_additive_scale() -> None:
    plan = EffectPlan(
        effects=[
            EffectInstance(
                effect_id="effect_001",
                effect_type="punch_in",
                start=0.5,
                end=1.5,
                priority=0.9,
                params={"max_scale": 1.22, "subtitle_position": "bottom"},
            ),
            EffectInstance(
                effect_id="effect_002",
                effect_type="micro_pan",
                start=1.0,
                end=2.0,
                priority=0.8,
                params={"max_scale": 1.2, "subtitle_position": "bottom"},
            ),
        ],
        metadata={"density": "high", "intensity": 1.2},
    )
    directed, _motion_plan = direct_effect_plan(plan, clip_duration=4.0, width=1920, height=1080)

    graph = build_effect_filter_graph(
        directed,
        duration=4.0,
        width=1920,
        height=1080,
        fps=30.0,
        has_audio=True,
    )

    assert "motion-director-v1" not in graph
    assert "max(" in graph
    assert "crop=1920:1080" in graph
    assert "1+0.22000" not in graph


def test_motion_plan_validation_rejects_unsafe_payload() -> None:
    report = validate_motion_plan_payload(
        {
            "version": "motion-director-v1",
            "max_scale": 1.12,
            "max_pan_offset": 0.05,
            "max_zoom_velocity": 0.08,
            "camera_segments": [
                {
                    "effect_id": "effect_bad",
                    "start": 0.5,
                    "end": 1.0,
                    "target_scale": 1.3,
                    "anchor_x": 0.95,
                    "anchor_y": 0.5,
                    "pan_x": 0.2,
                    "pan_y": 0.0,
                }
            ],
        }
    )

    assert report.passed is False
    assert any("exceeds directed max scale" in issue for issue in report.issues)
    assert any("unsafe camera anchor" in issue for issue in report.issues)


def test_preview_qa_uses_raw_frame_samples_without_files(monkeypatch) -> None:  # noqa: ANN001
    import effects.preview as preview_module

    gray = bytes([80, 80, 80] * preview_module.PREVIEW_WIDTH * preview_module.PREVIEW_HEIGHT)
    bright = bytes([92, 92, 92] * preview_module.PREVIEW_WIDTH * preview_module.PREVIEW_HEIGHT)
    calls: list[tuple[str, float]] = []

    def fake_extract(path: str, time_sec: float) -> bytes:
        calls.append((path, time_sec))
        return bright if path == "output.mp4" else gray

    monkeypatch.setattr(preview_module, "_extract_tiny_frame", fake_extract)
    plan = EffectPlan(
        effects=[
            EffectInstance(
                effect_id="effect_001",
                effect_type="punch_in",
                start=0.5,
                end=1.5,
                priority=0.8,
                params={"max_scale": 1.08},
            )
        ]
    )

    report = validate_effect_preview("source.mp4", "output.mp4", plan)

    assert report["passed"] is True
    assert report["sample_count"] == 1
    assert len(calls) == 2


def test_effect_plan_validation_rejects_blocked_ranges_and_warns_on_rhythm() -> None:
    plan = EffectPlan(
        effects=[
            EffectInstance(
                effect_id="effect_001",
                effect_type="punch_in",
                start=0.5,
                end=1.2,
                priority=0.8,
                params={"max_scale": 1.1},
            ),
            EffectInstance(
                effect_id="effect_002",
                effect_type="punch_in",
                start=1.25,
                end=1.9,
                priority=0.75,
                params={"max_scale": 1.1},
            ),
        ]
    )

    validation = validate_effect_plan(
        plan,
        clip_duration=4.0,
        scene_cuts=[1.2],
        blocked_ranges=[(1.3, 1.6)],
    )

    assert validation["passed"] is False
    assert any("blocked overlay range" in error for error in validation["errors"])
    assert any("too close" in warning for warning in validation["warnings"])


def test_compiler_builds_single_pass_audio_graph() -> None:
    plan = EffectPlan(
        effects=[
            EffectInstance(
                effect_id="effect_001",
                effect_type="impact_pulse",
                start=1.0,
                end=2.0,
                priority=0.9,
                params={"max_scale": 1.12, "subtitle_position": "bottom", "subtitle_highlight_enabled": True},
                modifiers=["flash_accent", "subtitle_highlight"],
            )
        ]
    )

    graph = build_effect_filter_graph(plan, duration=4.0, width=1920, height=1080, fps=30.0, has_audio=True)

    assert "[0:v]fps=30" in graph
    assert "split=" not in graph
    assert "concat=" not in graph
    assert "[0:a]" not in graph
    assert graph.endswith("[v]")
    assert "sin(PI*((t-1.00000)" in graph
    assert "drawbox=x=0:y=0" in graph
    assert "drawbox=x=iw*0.08" in graph


def test_compiler_builds_video_only_freeze_graph() -> None:
    plan = EffectPlan(
        effects=[
            EffectInstance(
                effect_id="effect_001",
                effect_type="freeze_accent",
                start=0.5,
                end=1.0,
                priority=0.9,
                params={"max_scale": 1.04},
                modifiers=[],
            )
        ]
    )

    graph = build_effect_filter_graph(plan, duration=2.0, width=1280, height=720, fps=24.0, has_audio=False)

    assert "[0:a]" not in graph
    assert "loop=loop=" not in graph
    assert "crop=1280:720" in graph
    assert graph.endswith("[v]")


def test_compiler_does_not_draw_subtitle_highlight_unless_enabled() -> None:
    plan = EffectPlan(
        effects=[
            EffectInstance(
                effect_id="effect_001",
                effect_type="punch_in",
                start=0.5,
                end=1.2,
                priority=0.8,
                params={"max_scale": 1.09, "subtitle_position": "bottom"},
                modifiers=["subtitle_highlight"],
            )
        ]
    )

    disabled_graph = build_effect_filter_graph(plan, duration=2.0, width=1280, height=720, fps=24.0, has_audio=False)
    enabled_plan = EffectPlan(
        effects=[
            EffectInstance(
                effect_id="effect_001",
                effect_type="punch_in",
                start=0.5,
                end=1.2,
                priority=0.8,
                params={"max_scale": 1.09, "subtitle_position": "bottom", "subtitle_highlight_enabled": True},
                modifiers=["subtitle_highlight"],
            )
        ]
    )
    enabled_graph = build_effect_filter_graph(enabled_plan, duration=2.0, width=1280, height=720, fps=24.0, has_audio=False)

    assert "drawbox=x=iw*0.08" not in disabled_graph
    assert "drawbox=x=iw*0.08" in enabled_graph


def test_apply_timed_effects_omits_audio_map_when_source_has_no_audio(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    commands: list[list[str]] = []
    monkeypatch.setattr(engine, "probe_video", lambda _path: _metadata(has_audio=False))
    monkeypatch.setattr(engine, "_run_command", lambda command, _message: commands.append(command))

    plan = EffectPlan(
        effects=[
            EffectInstance(
                effect_id="effect_001",
                effect_type="punch_in",
                start=0.5,
                end=1.4,
                priority=0.8,
                params={"max_scale": 1.1},
            )
        ]
    )
    output = engine.apply_timed_effects("input.mp4", str(tmp_path), plan.to_dict())

    assert output.endswith(".mp4")
    command = commands[0]
    assert "-an" in command
    assert "[a]" not in command


def test_apply_timed_effects_maps_source_audio_without_filtering(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    commands: list[list[str]] = []
    monkeypatch.setattr(engine, "probe_video", lambda _path: _metadata(has_audio=True))
    monkeypatch.setattr(engine, "_run_command", lambda command, _message: commands.append(command))

    plan = EffectPlan(
        effects=[
            EffectInstance(
                effect_id="effect_001",
                effect_type="punch_in",
                start=0.5,
                end=1.4,
                priority=0.8,
                params={"max_scale": 1.1},
            )
        ]
    )
    engine.apply_timed_effects("input.mp4", str(tmp_path), plan.to_dict())

    command = commands[0]
    assert "0:a?" in command
    assert "[a]" not in command


def test_auto_effects_can_move_trailing_subtitles_behind_effects(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    state = _state_with_dirs(tmp_path)
    state.timeline = [
        {"op": "trim_clip", "params": {"start": "0"}, "description": "Trimmed"},
        {
            "op": "burn_subtitles",
            "params": {
                "srt_path": str(tmp_path / "transcript.srt"),
                "font_size": 24,
                "font_color": "white",
                "outline_color": "black",
                "position": "bottom",
            },
            "description": "Burned subtitles",
            "result_file": str(tmp_path / "subtitled.mp4"),
        },
    ]
    calls: list[str] = []

    def fake_rebuild(project_state: ProjectState, *, timeline_override=None) -> None:  # noqa: ANN001
        calls.append("rebuild")
        project_state.working_file = str(tmp_path / "rebuilt.mp4")
        project_state.metadata = _metadata(has_audio=True)

    monkeypatch.setattr(auto_effects_tool, "rebuild_timeline", fake_rebuild)

    popped = auto_effects_tool._pop_trailing_subtitle_ops(state)

    assert calls == ["rebuild"]
    assert [op["op"] for op in popped] == ["burn_subtitles"]
    assert [op["op"] for op in state.timeline] == ["trim_clip"]
    assert state.working_file.endswith("rebuilt.mp4")


def test_auto_effects_reapplies_popped_subtitles(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    state = _state_with_dirs(tmp_path)
    state.working_file = str(tmp_path / "effected.mp4")
    (tmp_path / "transcript.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
    subtitle_op = {
        "op": "burn_subtitles",
        "params": {
            "srt_path": str(tmp_path / "transcript.srt"),
            "font_size": 28,
            "font_color": "yellow",
            "outline_color": "black",
            "position": "top",
        },
        "description": "Burned subtitles",
        "result_file": str(tmp_path / "old.mp4"),
    }
    burn_calls: list[dict[str, object]] = []

    def fake_burn_subtitles(input_path: str, working_dir: str, **kwargs: object) -> str:
        burn_calls.append({"input_path": input_path, "working_dir": working_dir, **kwargs})
        return str(tmp_path / "resubtitled.mp4")

    monkeypatch.setattr(auto_effects_tool, "burn_subtitles", fake_burn_subtitles)
    monkeypatch.setattr(auto_effects_tool, "probe_video", lambda _path: _metadata(has_audio=True))

    auto_effects_tool._reapply_subtitle_ops(state, [subtitle_op])

    assert burn_calls[0]["input_path"].endswith("effected.mp4")
    assert burn_calls[0]["font_size"] == 28
    assert burn_calls[0]["position"] == "top"
    assert state.working_file.endswith("resubtitled.mp4")
    assert state.timeline[-1]["op"] == "burn_subtitles"
    assert state.timeline[-1]["result_file"].endswith("resubtitled.mp4")


def test_auto_effects_rejects_reapplying_subtitles_outside_project(tmp_path: Path) -> None:
    state = _state_with_dirs(tmp_path)
    outside_srt = tmp_path.parent / "outside.srt"
    outside_srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
    subtitle_op = {
        "op": "burn_subtitles",
        "params": {"srt_path": str(outside_srt)},
        "description": "Burned subtitles",
    }

    with pytest.raises(engine.VideoEngineError, match="must stay inside"):
        auto_effects_tool._reapply_subtitle_ops(state, [subtitle_op])


def test_auto_effects_restores_trailing_subtitles_when_planning_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:  # noqa: ANN001
    state = _state_with_dirs(tmp_path)
    state.timeline = [
        {
            "op": "burn_subtitles",
            "params": {"srt_path": str(tmp_path / "transcript.srt")},
            "result_file": state.working_file,
        }
    ]
    state.save()
    monkeypatch.setattr(auto_effects_tool, "rebuild_timeline", lambda _state: None)
    monkeypatch.setattr(
        auto_effects_tool,
        "_ensure_transcript_bundle",
        lambda _state: (_ for _ in ()).throw(RuntimeError("planning failed")),
    )

    result = auto_effects_tool.execute({"refresh_existing": False}, state)

    assert result["success"] is False
    assert [op["op"] for op in state.timeline] == ["burn_subtitles"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="FFmpeg is required")
def test_apply_timed_effects_renders_valid_synthetic_video(tmp_path: Path) -> None:
    input_path = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x90:rate=15:duration=3.0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(input_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    plan = EffectPlan(
        effects=[
            EffectInstance(
                effect_id="effect_001",
                effect_type="impact_pulse",
                start=0.15,
                end=0.65,
                priority=0.9,
                params={"max_scale": 1.1, "subtitle_position": "bottom", "subtitle_highlight_enabled": True},
                modifiers=["flash_accent", "subtitle_highlight"],
            ),
            EffectInstance(
                effect_id="effect_002",
                effect_type="micro_pan",
                start=0.9,
                end=1.45,
                priority=0.8,
                params={"max_scale": 1.08},
                modifiers=["vignette"],
            ),
            EffectInstance(
                effect_id="effect_003",
                effect_type="subtle_shake",
                start=1.75,
                end=2.25,
                priority=0.75,
                params={"max_scale": 1.06, "shake_amplitude": 0.045},
                modifiers=["focus_blur"],
            ),
        ]
    )

    output_path = engine.apply_timed_effects(str(input_path), str(tmp_path), plan.to_dict())
    metadata = engine.probe_video(output_path)

    assert Path(output_path).is_file()
    assert int(metadata["width"]) == 160
    assert int(metadata["height"]) == 90
    assert abs(float(metadata["duration_sec"]) - 3.0) < 0.18


def test_intent_compiles_auto_effects_without_llm() -> None:
    plan = compile_intent("add strong subtitle-aware zoom effects", _state())

    assert plan is not None
    assert len(plan.steps) == 1
    assert plan.steps[0].tool == "add_auto_effects"
    assert plan.steps[0].params == {"density": "high", "intensity": "high"}


def _metadata(*, has_audio: bool) -> dict[str, object]:
    return {
        "duration_sec": 4.0,
        "fps": 30.0,
        "width": 1920,
        "height": 1080,
        "codec": "h264",
        "has_audio": has_audio,
        "size_bytes": 1024,
        "format": "mov,mp4",
    }


def _state() -> ProjectState:
    now = utc_now_iso()
    return ProjectState(
        project_id="test-project",
        project_name="Test Project",
        created_at=now,
        updated_at=now,
        source_files=["source.mp4"],
        working_file="working.mp4",
        working_dir=".",
        output_dir="out",
        metadata={"duration_sec": 120.0, "width": 1920, "height": 1080, "fps": 30.0},
        provider="test",
        model="test-model",
    )


def _state_with_dirs(tmp_path: Path) -> ProjectState:
    state = _state()
    state.working_dir = str(tmp_path)
    state.output_dir = str(tmp_path / "out")
    state.source_files = [str(tmp_path / "source.mp4")]
    state.working_file = str(tmp_path / "working.mp4")
    return state
