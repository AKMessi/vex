from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np

import config
from renderers.hyperframes_renderer import _build_bounded_repair_variant
from vex_hyperframes.compiler import compile_hyperframes_plan
from vex_hyperframes.semantic_qa import (
    analyze_hyperframes_semantics,
    inspect_animation_frames,
)
from vex_hyperframes.variants import select_best_variant
from vex_hyperframes.variants import build_variants
from vex_hyperframes.vision_qa import critique_hyperframes_frames


def test_animation_inspection_requires_meaningful_changes_and_final_hold(
    tmp_path: Path,
) -> None:
    frames = _moving_frames(tmp_path)

    report = inspect_animation_frames(frames)

    assert report.passed is True
    assert report.active_transition_count == 2
    assert report.final_hold_delta == 0.0
    assert report.score >= 0.58


def test_semantic_qa_passes_grounded_labels_objects_and_motion(tmp_path: Path) -> None:
    frames = _moving_frames(tmp_path)
    report = analyze_hyperframes_semantics(
        html="<main><b>Classify request</b><b>Check policy</b><b>Human handoff</b></main>",
        frame_paths=frames,
        production_contract={
            "scene_type": "guided_process",
            "required_labels": ["Classify request", "Check policy", "Human handoff"],
            "quality_floor": 0.78,
        },
        visual_explanation_ir={
            "objects": [
                {"object_id": "object_01", "label": "Classify request"},
                {"object_id": "object_02", "label": "Check policy"},
                {"object_id": "object_03", "label": "Human handoff"},
            ]
        },
        storyboard=[
            {"phase": "establish", "visual_change": "Reveal request"},
            {"phase": "explain", "visual_change": "Trace policy"},
            {"phase": "resolve", "visual_change": "Complete handoff"},
        ],
        stage_metadata={
            "visible_labels": ["Classify request", "Check policy", "Human handoff"]
        },
        qa_mode="local",
    )

    assert report.passed is True
    assert report.label_coverage == 1.0
    assert report.object_coverage == 1.0
    assert report.screenshot_test_passed is True
    assert report.repair_action == "keep"


def test_semantic_qa_reports_exact_missing_copy_and_bounded_repair(
    tmp_path: Path,
) -> None:
    frames = _moving_frames(tmp_path)
    report = analyze_hyperframes_semantics(
        html="<main><b>Classify request</b><b>Human handoff</b></main>",
        frame_paths=frames,
        production_contract={
            "scene_type": "guided_process",
            "required_labels": ["Classify request", "Check policy", "Human handoff"],
            "quality_floor": 0.78,
        },
        visual_explanation_ir={
            "objects": [
                {"object_id": "object_01", "label": "Classify request"},
                {"object_id": "object_02", "label": "Check policy"},
                {"object_id": "object_03", "label": "Human handoff"},
            ]
        },
        storyboard=[
            {"phase": "establish"},
            {"phase": "resolve"},
        ],
        stage_metadata={
            "visible_labels": ["Classify request", "Human handoff"]
        },
        qa_mode="local",
    )

    assert report.passed is False
    assert report.missing_labels == ["Check policy"]
    assert "object_02" in report.missing_objects
    assert report.repair_action == "repair_grounded_copy_placement"
    assert "Check policy" in report.repair_directives[0]


def test_hybrid_qa_warns_when_vision_is_unavailable_but_strict_mode_fails(
    tmp_path: Path,
) -> None:
    frames = _moving_frames(tmp_path)
    kwargs = {
        "html": "<main><b>Measured result</b></main>",
        "frame_paths": frames,
        "production_contract": {
            "scene_type": "metric_proof",
            "required_labels": ["Measured result"],
            "quality_floor": 0.72,
        },
        "visual_explanation_ir": {
            "objects": [{"object_id": "object_01", "label": "Measured result"}]
        },
        "storyboard": [{"phase": "establish"}, {"phase": "resolve"}],
        "stage_metadata": {"visible_labels": ["Measured result"]},
        "vision_report": {"available": False, "passed": None, "score": None},
    }

    hybrid = analyze_hyperframes_semantics(**kwargs, qa_mode="hybrid")
    strict = analyze_hyperframes_semantics(**kwargs, qa_mode="vision")

    assert hybrid.passed is True
    assert "vision_qa_unavailable_local_semantic_qa_used" in hybrid.warnings
    assert strict.passed is False
    assert "strict_vision_qa_unavailable" in strict.hard_failures


def test_vision_failure_recommends_renderer_reroute(tmp_path: Path) -> None:
    frames = _moving_frames(tmp_path)
    report = analyze_hyperframes_semantics(
        html="<main><b>Failed shot</b><b>Render log</b><b>Retry that shot</b></main>",
        frame_paths=frames,
        production_contract={
            "scene_type": "grounded_interface_walkthrough",
            "required_labels": ["Failed shot", "Render log", "Retry that shot"],
            "quality_floor": 0.78,
        },
        visual_explanation_ir={
            "objects": [
                {"object_id": "object_01", "label": "Failed shot"},
                {"object_id": "object_02", "label": "Render log"},
                {"object_id": "object_03", "label": "Retry that shot"},
            ]
        },
        storyboard=[{"phase": "establish"}, {"phase": "resolve"}],
        stage_metadata={
            "visible_labels": ["Failed shot", "Render log", "Retry that shot"]
        },
        qa_mode="hybrid",
        vision_report={
            "available": True,
            "passed": False,
            "score": 0.42,
            "semantic_issues": ["The interface state is not recognizable."],
            "repair_directives": ["Use the captured editor frame."],
        },
    )

    assert report.passed is False
    assert report.repair_action == "reroute_renderer"
    assert report.reroute_renderer == "ffmpeg_asset"


def test_vision_qa_skips_without_credentials(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    frame = _moving_frames(tmp_path)[0]
    monkeypatch.setattr(config, "HYPERFRAMES_ENABLE_VISION_QA", True)
    monkeypatch.setattr(config, "GEMINI_API_KEY", None)

    report = critique_hyperframes_frames(
        [frame],
        production_contract={"required_labels": ["Measured result"]},
        storyboard=[],
    )

    assert report.available is False
    assert report.passed is None
    assert "GEMINI_API_KEY" in report.notes


def test_variant_selection_refuses_all_failed_candidates() -> None:
    selected = select_best_variant(
        [
            {
                "variant_id": "variant_01",
                "asset_path": "a.mp4",
                "qa": {"score": 0.94, "passed": False},
            },
            {
                "variant_id": "variant_02",
                "asset_path": "b.mp4",
                "qa": {"score": 0.91, "passed": False},
            },
        ]
    )

    assert selected is None


def test_bounded_repair_switches_to_typed_bespoke_without_changing_ir() -> None:
    plan = compile_hyperframes_plan(
        {
            "visual_id": "repair_process",
            "sentence_text": "The request is classified, checked against policy, then sent to a human.",
            "context_text": "The handoff prevents unsupported answers.",
            "semantic_frame": {
                "steps": ["Classify request", "Check policy", "Send to human"],
                "result": "Prevent unsupported answers",
            },
            "duration": 4.0,
        }
    )
    variant = build_variants(plan.renderer_spec, default_count=2)[1]

    repaired = _build_bounded_repair_variant(
        variant,
        "repair_object_coverage",
    )

    assert repaired.spec["hyperframes_repair_action"] == "repair_object_coverage"
    assert repaired.variant_id.endswith("_repair")
    assert repaired.variant_id != variant.variant_id
    assert repaired.spec["bespoke_scene_program"]["version"].endswith("-v1")
    assert repaired.spec["visual_explanation_ir"] == variant.spec["visual_explanation_ir"]


def _moving_frames(tmp_path: Path) -> list[Path]:
    paths: list[Path] = []
    positions = [8, 54, 100, 100]
    for index, left in enumerate(positions, start=1):
        frame = np.zeros((120, 180, 3), dtype=np.uint8)
        frame[34:86, left : left + 48, :] = 255
        path = tmp_path / f"frame_{index:02d}.png"
        iio.imwrite(path, frame)
        paths.append(path)
    return paths
