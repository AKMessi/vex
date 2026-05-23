from __future__ import annotations

import json
from pathlib import Path

import pytest

from effects.compiler import build_effect_filter_graph
from effects.schema import EffectInstance, EffectPlan
from smartie import SmartieBundleError, load_smartie_bundle, plan_smartie_attention_effects, validate_smartie_effect_plan
from tools.import_smartie import import_smartie_bundle


def test_bundle_loader_accepts_required_files_and_optional_files_are_optional(tmp_path: Path) -> None:
    bundle_dir = _smartie_bundle(tmp_path)

    bundle = load_smartie_bundle(bundle_dir)

    assert bundle.source_video.name == "recording.webm"
    assert bundle.smartie_metadata_path is None
    assert bundle.preview_thumbnails_dir is None
    assert bundle.manifest.duration_sec == 8.0
    assert bundle.manifest.width == 1920
    assert bundle.attention_points
    assert bundle.attention_points[0].x == 0.5
    assert bundle.attention_points[0].y == 0.5


def test_bundle_loader_requires_attention_timeline(tmp_path: Path) -> None:
    bundle_dir = _smartie_bundle(tmp_path)
    (bundle_dir / "attention.timeline.json").unlink()

    with pytest.raises(SmartieBundleError, match="attention.timeline.json"):
        load_smartie_bundle(bundle_dir)


def test_import_smartie_creates_project_manifest_without_render(tmp_path: Path) -> None:
    bundle_dir = _smartie_bundle(tmp_path)
    project_dir = tmp_path / "vex-smartie-project"

    state, result = import_smartie_bundle(
        bundle_dir,
        project=str(project_dir),
        render=False,
        probe_video_fn=lambda _path: _metadata(),
    )

    assert state.project_name == "Smartie Demo"
    assert Path(state.working_file).is_file()
    assert state.source_files == [str((bundle_dir / "recording.webm").resolve())]
    assert state.artifacts["smartie"]["attention_event_count"] == 5
    assert state.artifacts["latest_smartie_import"]["effect_count"] == result["effect_count"]
    assert result["effect_count"] >= 1
    assert Path(result["manifest_path"]).is_file()
    assert Path(result["plan_path"]).is_file()
    assert not result["rendered"]


def test_attention_planner_emits_deterministic_smart_zoom_segments(tmp_path: Path) -> None:
    bundle = load_smartie_bundle(_smartie_bundle(tmp_path))

    plan = plan_smartie_attention_effects(bundle, duration=8.0, width=1920, height=1080, fps=30.0)
    second_plan = plan_smartie_attention_effects(bundle, duration=8.0, width=1920, height=1080, fps=30.0)

    assert plan.to_dict() == second_plan.to_dict()
    assert plan.effects
    assert {effect.effect_type for effect in plan.effects} == {"smart_zoom_segment"}
    assert all(effect.end - effect.start >= 0.8 for effect in plan.effects)
    assert all(1.0 <= float(effect.params["target_scale"]) <= 2.5 for effect in plan.effects)
    assert validate_smartie_effect_plan(plan, duration=8.0)["ok"] is True


def test_compiler_accepts_smartie_zoom_segments() -> None:
    plan = EffectPlan(
        effects=[
            EffectInstance(
                effect_id="smart_zoom_001",
                effect_type="smart_zoom_segment",
                start=1.0,
                end=2.4,
                priority=0.9,
                reason="Smartie click focus",
                params={
                    "focus_x": 0.72,
                    "focus_y": 0.38,
                    "target_scale": 1.42,
                    "easing": "ease_in_out_hold",
                    "confidence": 0.9,
                },
            )
        ],
        source="smartie_attention",
    )

    loaded = EffectPlan.from_dict(plan.to_dict())
    graph = build_effect_filter_graph(loaded, duration=5.0, width=1920, height=1080, fps=30.0, has_audio=True)

    assert "crop=1920:1080" in graph
    assert "0.72000*in_w-out_w/2" in graph
    assert "0.38000*in_h-out_h/2" in graph
    assert "min(1\\,min(max(0" in graph
    assert "[0:a]" not in graph
    assert graph.endswith("[v]")


def _smartie_bundle(tmp_path: Path) -> Path:
    bundle_dir = tmp_path / "smartie-bundle"
    bundle_dir.mkdir()
    (bundle_dir / "recording.webm").write_bytes(b"fake webm bytes")
    (bundle_dir / "manifest.json").write_text(
        json.dumps(
            {
                "title": "Smartie Demo",
                "recording": "recording.webm",
                "duration_sec": 8.0,
                "fps": 30.0,
                "width": 1920,
                "height": 1080,
            }
        ),
        encoding="utf-8",
    )
    (bundle_dir / "attention.timeline.json").write_text(
        json.dumps(
            {
                "events": [
                    {"time": 1.0, "x": 960, "y": 540, "type": "move", "confidence": 0.45},
                    {"time": 1.25, "x": 1010, "y": 548, "type": "dwell", "confidence": 0.7},
                    {"time": 2.8, "x": 1420, "y": 420, "type": "click", "confidence": 0.85},
                    {"time": 4.4, "x": 1500, "y": 438, "type": "keyboard", "confidence": 0.75},
                    {"time": 6.2, "x": 780, "y": 700, "type": "attention", "confidence": 0.6},
                ]
            }
        ),
        encoding="utf-8",
    )
    return bundle_dir


def _metadata() -> dict[str, object]:
    return {
        "duration_sec": 8.0,
        "fps": 30.0,
        "width": 1920,
        "height": 1080,
        "codec": "vp9",
        "has_audio": True,
        "size_bytes": 128,
        "format": "matroska,webm",
    }
