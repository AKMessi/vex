from __future__ import annotations

import numpy as np
import pytest

import color_grading
from color_grading_evaluator import evaluate_masked_perceptual_grade


def test_masked_perceptual_evaluator_penalizes_skin_hue_drift() -> None:
    source = _skin_patch_frame()
    shifted = source.astype(np.float32).copy()
    shifted[..., 0] = np.clip(shifted[..., 0] * 1.22, 0, 255)
    shifted[..., 1] = np.clip(shifted[..., 1] * 0.90, 0, 255)
    shifted = shifted.astype(np.uint8)

    evaluation = evaluate_masked_perceptual_grade(
        [source],
        [shifted],
        need=0.12,
        source_quality=0.92,
    )

    assert evaluation.penalty > 0.015
    assert evaluation.breakdown["skin_delta"] > evaluation.breakdown["skin_delta_allowance"]
    assert evaluation.breakdown["skin_delta_penalty"] > 0.0


def test_candidate_scoring_uses_real_preview_frames_when_available(monkeypatch) -> None:  # noqa: ANN001
    source = _skin_patch_frame()
    damaged_preview = source.astype(np.float32).copy()
    damaged_preview[..., 2] = np.clip(damaged_preview[..., 2] * 1.35, 0, 255)
    damaged_preview = damaged_preview.astype(np.uint8)

    def fake_preview(_input_path, _metadata, timestamps, _filter_graph, **_kwargs):  # noqa: ANN001
        return [damaged_preview for _timestamp in timestamps]

    monkeypatch.setattr(color_grading, "render_color_grade_preview_frames", fake_preview)
    analysis = color_grading.analyze_frames([source, source])

    candidates = color_grading._generate_grade_candidates(  # noqa: SLF001
        [source, source],
        requested_look="cinematic",
        grade_intensity=1.0,
        candidate_count=2,
        overall_need=0.12,
        profile=color_grading.LOOK_PROFILES["cinematic"],
        before_analysis=analysis,
        sample_timestamps=[0.5, 1.5],
        preview_input_path="input.mp4",
        preview_metadata={"width": 96, "height": 96},
        real_preview_frame_count=1,
    )

    assert candidates
    assert all(candidate.score_breakdown["real_preview_used"] == 1.0 for candidate in candidates)
    assert all(candidate.score_breakdown["preview_fallback_used"] == 0.0 for candidate in candidates)
    assert any(candidate.score_breakdown["masked_perceptual_penalty"] > 0.0 for candidate in candidates)


def test_candidate_preview_fallback_is_recorded_when_ffmpeg_preview_fails(monkeypatch) -> None:  # noqa: ANN001
    source = _skin_patch_frame()

    def fail_preview(_input_path, _metadata, _timestamps, _filter_graph, **_kwargs):  # noqa: ANN001
        raise color_grading.ColorGradePlanningError("preview unavailable")

    monkeypatch.setattr(color_grading, "render_color_grade_preview_frames", fail_preview)
    analysis = color_grading.analyze_frames([source, source])

    candidates = color_grading._generate_grade_candidates(  # noqa: SLF001
        [source, source],
        requested_look="cinematic",
        grade_intensity=1.0,
        candidate_count=2,
        overall_need=0.12,
        profile=color_grading.LOOK_PROFILES["cinematic"],
        before_analysis=analysis,
        sample_timestamps=[0.5, 1.5],
        preview_input_path="input.mp4",
        preview_metadata={"width": 96, "height": 96},
        real_preview_frame_count=1,
    )

    assert candidates
    assert all(candidate.score_breakdown["real_preview_used"] == 0.0 for candidate in candidates)
    assert all(candidate.score_breakdown["simulated_preview_used"] == 1.0 for candidate in candidates)
    assert all(candidate.score_breakdown["preview_fallback_used"] == 1.0 for candidate in candidates)


def test_shot_plan_manifest_reports_simulated_preview_evaluation() -> None:
    source = _skin_patch_frame()

    plan = color_grading.build_shot_aware_color_grade_plan_from_shots(
        [(0.0, 2.0, [source, source])],
        look="cinematic",
        intensity=1.0,
        candidate_count=2,
    )

    assert plan.manifest is not None
    preview_evaluation = plan.manifest.preview_evaluation
    assert preview_evaluation["mode"] == "simulated"
    assert preview_evaluation["frames_per_candidate"] == 2
    assert preview_evaluation["total_candidate_count"] == len(plan.manifest.shots[0].candidates)
    assert preview_evaluation["real_candidate_count"] == 0
    assert preview_evaluation["simulated_candidate_count"] == preview_evaluation["total_candidate_count"]
    assert plan.adjustments["real_preview_fraction"] == pytest.approx(0.0)
    assert plan.adjustments["simulated_preview_fraction"] == pytest.approx(1.0)


def _skin_patch_frame() -> np.ndarray:
    frame = np.full((96, 96, 3), np.array([128, 124, 118], dtype=np.uint8))
    frame[12:84, 14:82] = np.array([186, 132, 104], dtype=np.uint8)
    frame[6:16, 6:90] = np.array([92, 94, 95], dtype=np.uint8)
    return frame
