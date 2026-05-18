from __future__ import annotations

from pathlib import Path

import engine
from effects.compiler import build_effect_filter_graph
from effects.planner import plan_subtitle_effects
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

    plan = plan_subtitle_effects(cards, 12.0, max_effects=8, density="high", include_style_effects=True)

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


def test_compiler_builds_single_pass_audio_graph() -> None:
    plan = EffectPlan(
        effects=[
            EffectInstance(
                effect_id="effect_001",
                effect_type="impact_pulse",
                start=1.0,
                end=2.0,
                priority=0.9,
                params={"max_scale": 1.12, "subtitle_position": "bottom"},
                modifiers=["flash_accent", "subtitle_highlight"],
            )
        ]
    )

    graph = build_effect_filter_graph(plan, duration=4.0, width=1920, height=1080, fps=30.0, has_audio=True)

    assert "[0:v]split=3" in graph
    assert "[0:a]asplit=3" in graph
    assert "concat=n=3:v=1:a=1[v][a]" in graph
    assert "sin(PI*t/" in graph
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
    assert "select='eq(n\\,0)'" in graph
    assert "loop=loop=" in graph
    assert "concat=n=3:v=1:a=0[v]" in graph


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
