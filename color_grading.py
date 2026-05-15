from __future__ import annotations

import subprocess
import re
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

import config


SUPPORTED_COLOR_GRADE_LOOKS = (
    "auto",
    "natural",
    "vibrant",
    "cinematic",
    "warm",
    "cool",
    "documentary",
    "punchy",
)

LOOK_ALIASES = {
    "neutral": "natural",
    "clean": "natural",
    "balanced": "natural",
    "cinema": "cinematic",
    "film": "cinematic",
    "filmic": "cinematic",
    "pop": "vibrant",
    "color_pop": "vibrant",
    "colour_pop": "vibrant",
    "high_contrast": "punchy",
}

LOOK_PROFILES: dict[str, dict[str, Any]] = {
    "natural": {
        "contrast": 1.015,
        "saturation": 1.025,
        "target_luma": 0.48,
        "target_span": 0.66,
        "target_saturation": 0.32,
        "rgb_gain": (1.0, 1.0, 1.0),
        "level_strength": 0.34,
        "curve_strength": 0.18,
        "balance": {},
    },
    "vibrant": {
        "contrast": 1.055,
        "saturation": 1.13,
        "target_luma": 0.49,
        "target_span": 0.68,
        "target_saturation": 0.38,
        "rgb_gain": (1.01, 1.005, 0.995),
        "level_strength": 0.42,
        "curve_strength": 0.36,
        "balance": {},
    },
    "cinematic": {
        "contrast": 1.075,
        "saturation": 1.055,
        "target_luma": 0.46,
        "target_span": 0.70,
        "target_saturation": 0.34,
        "rgb_gain": (1.01, 0.998, 1.012),
        "level_strength": 0.44,
        "curve_strength": 0.48,
        "balance": {"bs": 0.018, "rm": 0.006, "rh": 0.018, "bh": -0.014},
    },
    "warm": {
        "contrast": 1.025,
        "saturation": 1.045,
        "target_luma": 0.49,
        "target_span": 0.66,
        "target_saturation": 0.33,
        "rgb_gain": (1.035, 1.01, 0.975),
        "level_strength": 0.36,
        "curve_strength": 0.22,
        "balance": {"rm": 0.01, "bm": -0.008, "rh": 0.012, "bh": -0.01},
    },
    "cool": {
        "contrast": 1.03,
        "saturation": 1.035,
        "target_luma": 0.48,
        "target_span": 0.66,
        "target_saturation": 0.32,
        "rgb_gain": (0.982, 0.998, 1.038),
        "level_strength": 0.36,
        "curve_strength": 0.22,
        "balance": {"rs": -0.008, "bs": 0.012, "bm": 0.01},
    },
    "documentary": {
        "contrast": 1.025,
        "saturation": 1.0,
        "target_luma": 0.50,
        "target_span": 0.64,
        "target_saturation": 0.30,
        "rgb_gain": (1.004, 1.004, 0.996),
        "level_strength": 0.28,
        "curve_strength": 0.12,
        "balance": {},
    },
    "punchy": {
        "contrast": 1.105,
        "saturation": 1.095,
        "target_luma": 0.48,
        "target_span": 0.72,
        "target_saturation": 0.36,
        "rgb_gain": (1.008, 1.0, 0.998),
        "level_strength": 0.50,
        "curve_strength": 0.55,
        "balance": {},
    },
}


class ColorGradePlanningError(ValueError):
    pass


@dataclass(frozen=True)
class ColorGradeAnalysis:
    sample_count: int
    luma_mean: float
    luma_median: float
    luma_std: float
    luma_p01: float
    luma_p05: float
    luma_p95: float
    luma_p99: float
    luma_span: float
    saturation_mean: float
    red_mean: float
    green_mean: float
    blue_mean: float
    black_clip_fraction: float
    white_clip_fraction: float
    midtone_fraction: float
    neutral_pixel_fraction: float
    skin_pixel_fraction: float
    white_balance_confidence: float
    frame_quality_mean: float
    skipped_frame_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ColorGradeShotRange:
    index: int
    start_sec: float
    end_sec: float
    duration_sec: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ColorGradeCandidate:
    candidate_id: str
    label: str
    intensity: float
    filter_graph: str
    adjustments: dict[str, float]
    score: float
    base_score: float
    continuity_penalty: float
    after_analysis: ColorGradeAnalysis
    score_breakdown: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "label": self.label,
            "intensity": self.intensity,
            "filter_graph": self.filter_graph,
            "adjustments": dict(self.adjustments),
            "score": self.score,
            "base_score": self.base_score,
            "continuity_penalty": self.continuity_penalty,
            "after_analysis": self.after_analysis.to_dict(),
            "score_breakdown": dict(self.score_breakdown),
        }


@dataclass(frozen=True)
class ColorGradeShotDecision:
    index: int
    start_sec: float
    end_sec: float
    duration_sec: float
    sample_timestamps: list[float]
    analysis: ColorGradeAnalysis
    correction_need: float
    confidence: float
    selected_candidate_id: str
    selected_filter_graph: str
    selected_adjustments: dict[str, float]
    selected_score: float
    candidates: list[ColorGradeCandidate]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "duration_sec": self.duration_sec,
            "sample_timestamps": list(self.sample_timestamps),
            "analysis": self.analysis.to_dict(),
            "correction_need": self.correction_need,
            "confidence": self.confidence,
            "selected_candidate_id": self.selected_candidate_id,
            "selected_filter_graph": self.selected_filter_graph,
            "selected_adjustments": dict(self.selected_adjustments),
            "selected_score": self.selected_score,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class ColorGradeManifest:
    mode: str
    requested_look: str
    resolved_look: str
    intensity: float
    render_mode: str
    output_label: str
    scene_threshold: float
    max_shots: int
    shot_count: int
    candidate_count: int
    filter_graph: str
    shots: list[ColorGradeShotDecision]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "requested_look": self.requested_look,
            "resolved_look": self.resolved_look,
            "intensity": self.intensity,
            "render_mode": self.render_mode,
            "output_label": self.output_label,
            "scene_threshold": self.scene_threshold,
            "max_shots": self.max_shots,
            "shot_count": self.shot_count,
            "candidate_count": self.candidate_count,
            "filter_graph": self.filter_graph,
            "warnings": list(self.warnings),
            "shots": [shot.to_dict() for shot in self.shots],
        }


@dataclass(frozen=True)
class _ShotSample:
    shot_range: ColorGradeShotRange
    frames: list[np.ndarray]
    timestamps: list[float]


@dataclass(frozen=True)
class ColorGradePlan:
    requested_look: str
    resolved_look: str
    intensity: float
    filter_graph: str
    adjustments: dict[str, float]
    analysis: ColorGradeAnalysis
    warnings: list[str]
    render_mode: str = "vf"
    output_label: str = ""
    manifest: ColorGradeManifest | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_look": self.requested_look,
            "resolved_look": self.resolved_look,
            "intensity": self.intensity,
            "filter_graph": self.filter_graph,
            "adjustments": dict(self.adjustments),
            "analysis": self.analysis.to_dict(),
            "warnings": list(self.warnings),
            "render_mode": self.render_mode,
            "output_label": self.output_label,
            "manifest": self.manifest.to_dict() if self.manifest else None,
        }


def normalize_color_grade_look(value: str | None) -> str:
    raw = (value or "auto").strip().lower().replace("-", "_").replace(" ", "_")
    normalized = LOOK_ALIASES.get(raw, raw)
    if normalized not in SUPPORTED_COLOR_GRADE_LOOKS:
        supported = ", ".join(SUPPORTED_COLOR_GRADE_LOOKS)
        raise ColorGradePlanningError(f"Unsupported color grade look {value!r}. Supported looks: {supported}.")
    return normalized


def build_color_grade_plan(
    input_path: str,
    metadata: dict[str, Any],
    *,
    look: str = "auto",
    intensity: float = 1.0,
    sample_count: int = 9,
) -> ColorGradePlan:
    frames = sample_video_frames(input_path, metadata, sample_count=sample_count)
    return build_color_grade_plan_from_frames(frames, look=look, intensity=intensity)


def build_shot_aware_color_grade_plan(
    input_path: str,
    metadata: dict[str, Any],
    *,
    look: str = "auto",
    intensity: float = 1.0,
    sample_count: int = 9,
    mode: str = "auto",
    max_shots: int = 18,
    candidate_count: int = 4,
    scene_threshold: float = 0.25,
) -> ColorGradePlan:
    normalized_mode = _normalize_grading_mode(mode)
    if normalized_mode == "fast_global":
        return build_color_grade_plan(input_path, metadata, look=look, intensity=intensity, sample_count=sample_count)

    requested_look = normalize_color_grade_look(look)
    grade_intensity = _validate_intensity(intensity)
    max_shots = max(1, min(int(max_shots or 18), 64))
    candidate_count = max(2, min(int(candidate_count or 4), 5))
    scene_threshold = _clamp(scene_threshold, 0.08, 0.55)
    try:
        shot_ranges = detect_video_shots(
            input_path,
            metadata,
            max_shots=max_shots,
            scene_threshold=scene_threshold,
        )
        samples = [
            _sample_shot(
                input_path,
                metadata,
                shot_range,
                sample_count=max(3, min(int(sample_count or 9), 6)),
            )
            for shot_range in shot_ranges
        ]
        return _build_shot_aware_plan_from_samples(
            samples,
            requested_look=requested_look,
            grade_intensity=grade_intensity,
            candidate_count=candidate_count,
            scene_threshold=scene_threshold,
            max_shots=max_shots,
            fallback_mode=normalized_mode,
        )
    except (ColorGradePlanningError, OSError, subprocess.SubprocessError) as exc:
        if normalized_mode == "shot_aware":
            raise
        fallback = build_color_grade_plan(input_path, metadata, look=requested_look, intensity=grade_intensity, sample_count=sample_count)
        warning = f"Shot-aware grading fell back to global grading: {exc}"
        return replace(fallback, warnings=[*fallback.warnings, warning])


def build_shot_aware_color_grade_plan_from_shots(
    shot_frames: list[tuple[float, float, list[np.ndarray]]],
    *,
    look: str = "auto",
    intensity: float = 1.0,
    candidate_count: int = 4,
) -> ColorGradePlan:
    if not shot_frames:
        raise ColorGradePlanningError("Cannot build a shot-aware color grade without shot samples.")
    samples: list[_ShotSample] = []
    for index, (start_sec, end_sec, frames) in enumerate(shot_frames):
        start = max(float(start_sec), 0.0)
        end = max(float(end_sec), start + 0.001)
        normalized_frames = list(frames)
        if not normalized_frames:
            raise ColorGradePlanningError(f"Shot {index} does not contain any sample frames.")
        timestamps = _sample_timestamps_for_range(start, end, len(normalized_frames))
        samples.append(
            _ShotSample(
                shot_range=ColorGradeShotRange(
                    index=index,
                    start_sec=round(start, 5),
                    end_sec=round(end, 5),
                    duration_sec=round(end - start, 5),
                ),
                frames=normalized_frames,
                timestamps=timestamps,
            )
        )
    return _build_shot_aware_plan_from_samples(
        samples,
        requested_look=normalize_color_grade_look(look),
        grade_intensity=_validate_intensity(intensity),
        candidate_count=max(2, min(int(candidate_count or 4), 5)),
        scene_threshold=0.0,
        max_shots=len(samples),
        fallback_mode="shot_aware",
    )


def build_color_grade_plan_from_frames(
    frames: list[np.ndarray],
    *,
    look: str = "auto",
    intensity: float = 1.0,
) -> ColorGradePlan:
    requested_look = normalize_color_grade_look(look)
    resolved_look = "natural" if requested_look == "auto" else requested_look
    grade_intensity = _validate_intensity(intensity)
    analysis = analyze_frames(frames)
    profile = LOOK_PROFILES[resolved_look]
    diagnostics = _correction_diagnostics(analysis, profile)
    correction_strength = float(diagnostics["correction_strength"]) * grade_intensity
    style_strength = grade_intensity
    if requested_look == "auto":
        style_strength = min(style_strength, 0.65)
    if requested_look in {"auto", "natural"} and diagnostics["overall_need"] < 0.22:
        style_strength *= 0.45

    luma_span = max(analysis.luma_span, 0.001)
    exposure_offset = float(profile["target_luma"]) - analysis.luma_median
    brightness_limit = 0.045 + (0.145 * diagnostics["exposure_error"])
    brightness = _clamp(
        exposure_offset * (0.20 + (0.24 * diagnostics["exposure_error"])),
        -brightness_limit,
        brightness_limit,
    )
    gamma_limit = 0.04 + (0.14 * diagnostics["exposure_error"])
    gamma = 1.0 + _clamp(
        exposure_offset * (0.16 + (0.20 * diagnostics["exposure_error"])),
        -gamma_limit,
        gamma_limit,
    )
    contrast_limit = 0.10 + (0.26 * diagnostics["contrast_error"])
    contrast_auto = 1.0 + _clamp(
        (float(profile["target_span"]) - luma_span) * (0.36 + (0.48 * diagnostics["contrast_error"])),
        -0.16,
        contrast_limit,
    )
    saturation_limit = 0.12 + (0.32 * diagnostics["saturation_error"])
    saturation_auto = 1.0 + _clamp(
        (float(profile["target_saturation"]) - analysis.saturation_mean)
        * (0.52 + (0.52 * diagnostics["saturation_error"])),
        -0.22,
        saturation_limit,
    )
    if analysis.white_clip_fraction > 0.015 and brightness > 0.0:
        brightness *= _clamp(1.0 - analysis.white_clip_fraction * 18.0, 0.20, 1.0)
    if analysis.black_clip_fraction > 0.02 and brightness < 0.0:
        brightness *= _clamp(1.0 - analysis.black_clip_fraction * 16.0, 0.20, 1.0)
    if analysis.white_clip_fraction + analysis.black_clip_fraction > 0.06:
        contrast_auto = min(contrast_auto, 1.04)
    if analysis.skin_pixel_fraction > 0.10 and saturation_auto > 1.0:
        saturation_auto = 1.0 + ((saturation_auto - 1.0) * 0.78)

    contrast = _blend_identity(contrast_auto, correction_strength) * _blend_identity(float(profile["contrast"]), style_strength)
    saturation = _blend_identity(saturation_auto, correction_strength) * _blend_identity(float(profile["saturation"]), style_strength)
    brightness = _clamp(brightness * correction_strength, -0.24, 0.24)
    gamma = _blend_identity(gamma, correction_strength)

    auto_gains = _white_balance_gains(analysis, max_delta=0.10 + (0.16 * diagnostics["cast_error"]))
    profile_gains = tuple(float(value) for value in profile["rgb_gain"])
    skin_guard = 0.82 if analysis.skin_pixel_fraction > 0.10 else 1.0
    wb_strength = (
        0.22
        + (0.55 * analysis.white_balance_confidence)
        + (0.28 * diagnostics["cast_error"])
    ) * correction_strength * skin_guard
    red_gain = _clamp(_blend_identity(auto_gains[0], wb_strength) * _blend_identity(profile_gains[0], style_strength), 0.78, 1.30)
    green_gain = _clamp(_blend_identity(auto_gains[1], wb_strength) * _blend_identity(profile_gains[1], style_strength), 0.78, 1.30)
    blue_gain = _clamp(_blend_identity(auto_gains[2], wb_strength) * _blend_identity(profile_gains[2], style_strength), 0.78, 1.30)

    color_balance = {
        key: round(float(value) * style_strength, 5)
        for key, value in dict(profile.get("balance") or {}).items()
        if abs(float(value) * style_strength) >= 0.0005
    }
    level_input_black, level_input_white = _level_inputs(
        analysis,
        float(profile["level_strength"]) * max(correction_strength, style_strength),
    )
    curve_shadow, curve_highlight = _curve_points(
        analysis,
        float(profile["curve_strength"]) * max(correction_strength, style_strength),
    )
    adjustments = {
        "brightness": round(brightness, 5),
        "contrast": round(_clamp(contrast, 0.78, 1.46), 5),
        "saturation": round(_clamp(saturation, 0.68, 1.72), 5),
        "gamma": round(_clamp(gamma, 0.78, 1.24), 5),
        "red_gain": round(red_gain, 5),
        "green_gain": round(green_gain, 5),
        "blue_gain": round(blue_gain, 5),
        "level_input_black": round(level_input_black, 5),
        "level_input_white": round(level_input_white, 5),
        "curve_shadow": round(curve_shadow, 5),
        "curve_highlight": round(curve_highlight, 5),
        "overall_need": round(float(diagnostics["overall_need"]), 5),
        "correction_strength": round(correction_strength, 5),
        "style_strength": round(style_strength, 5),
        "exposure_error": round(float(diagnostics["exposure_error"]), 5),
        "contrast_error": round(float(diagnostics["contrast_error"]), 5),
        "saturation_error": round(float(diagnostics["saturation_error"]), 5),
        "cast_error": round(float(diagnostics["cast_error"]), 5),
        "clip_risk": round(float(diagnostics["clip_risk"]), 5),
        **{f"colorbalance_{key}": value for key, value in color_balance.items()},
    }
    filter_graph = build_filter_graph(adjustments, color_balance)
    return ColorGradePlan(
        requested_look=requested_look,
        resolved_look=resolved_look,
        intensity=grade_intensity,
        filter_graph=filter_graph,
        adjustments=adjustments,
        analysis=analysis,
        warnings=_analysis_warnings(analysis, grade_intensity),
    )


def detect_video_shots(
    input_path: str,
    metadata: dict[str, Any],
    *,
    max_shots: int = 18,
    scene_threshold: float = 0.25,
    min_shot_duration: float = 0.65,
) -> list[ColorGradeShotRange]:
    duration = max(float(metadata.get("duration_sec") or 0.0), 0.0)
    max_shots = max(1, min(int(max_shots or 18), 64))
    if duration <= 0.0 or duration < min_shot_duration * 1.4:
        return _build_shot_ranges([], duration, max_shots=max_shots, min_shot_duration=min_shot_duration)

    command = [
        config.FFMPEG_PATH,
        "-hide_banner",
        "-i",
        input_path,
        "-filter:v",
        f"select='gt(scene,{_fmt(_clamp(scene_threshold, 0.08, 0.55))})',showinfo",
        "-f",
        "null",
        "-",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    except (subprocess.TimeoutExpired, OSError):
        return _build_shot_ranges([], duration, max_shots=max_shots, min_shot_duration=min_shot_duration)
    output = f"{result.stderr}\n{result.stdout}"
    boundaries = [
        float(match.group(1))
        for match in re.finditer(r"pts_time:([0-9]+(?:\.[0-9]+)?)", output)
    ]
    return _build_shot_ranges(
        boundaries,
        duration,
        max_shots=max_shots,
        min_shot_duration=min_shot_duration,
    )


def build_shot_filter_complex(shots: list[ColorGradeShotDecision]) -> str:
    if not shots:
        raise ColorGradePlanningError("Cannot build a shot-aware filter graph without shots.")
    if len(shots) == 1:
        return shots[0].selected_filter_graph
    segments: list[str] = []
    concat_inputs: list[str] = []
    for shot in shots:
        label = f"v{shot.index}"
        trim = f"[0:v]trim=start={_fmt(shot.start_sec)}:end={_fmt(shot.end_sec)},setpts=PTS-STARTPTS"
        segments.append(f"{trim},{shot.selected_filter_graph}[{label}]")
        concat_inputs.append(f"[{label}]")
    segments.append(f"{''.join(concat_inputs)}concat=n={len(shots)}:v=1:a=0[vout]")
    return ";".join(segments)


def validate_color_grade_output_by_shots(
    output_path: str,
    metadata: dict[str, Any],
    shots: list[dict[str, Any]],
    *,
    sample_count: int = 3,
) -> dict[str, Any]:
    shot_results: list[dict[str, Any]] = []
    for raw_shot in shots:
        try:
            frames, timestamps = sample_video_frames_for_range(
                output_path,
                metadata,
                float(raw_shot.get("start_sec", 0.0)),
                float(raw_shot.get("end_sec", 0.0)),
                sample_count=max(1, min(int(sample_count or 3), 4)),
            )
            validation = validate_color_grade_analysis(analyze_frames(frames))
            shot_results.append(
                {
                    "index": int(raw_shot.get("index", len(shot_results))),
                    "start_sec": float(raw_shot.get("start_sec", 0.0)),
                    "end_sec": float(raw_shot.get("end_sec", 0.0)),
                    "sample_timestamps": timestamps,
                    **validation,
                }
            )
        except (ColorGradePlanningError, OSError, ValueError) as exc:
            shot_results.append(
                {
                    "index": int(raw_shot.get("index", len(shot_results))),
                    "start_sec": float(raw_shot.get("start_sec", 0.0)),
                    "end_sec": float(raw_shot.get("end_sec", 0.0)),
                    "passed": False,
                    "score": 0.0,
                    "warnings": [f"Could not validate shot output: {exc}"],
                    "analysis": {},
                }
            )
    if not shot_results:
        return {"passed": False, "score": 0.0, "warnings": ["No shots were available for validation."], "shots": []}
    score = round(float(np.mean([float(shot.get("score", 0.0)) for shot in shot_results])), 4)
    warnings = [
        f"Shot {shot['index']}: {warning}"
        for shot in shot_results
        for warning in shot.get("warnings", [])
    ]
    return {
        "passed": all(bool(shot.get("passed")) for shot in shot_results) and score >= 0.68,
        "score": score,
        "warnings": warnings,
        "shots": shot_results,
    }


def _build_shot_aware_plan_from_samples(
    samples: list[_ShotSample],
    *,
    requested_look: str,
    grade_intensity: float,
    candidate_count: int,
    scene_threshold: float,
    max_shots: int,
    fallback_mode: str,
) -> ColorGradePlan:
    if not samples:
        raise ColorGradePlanningError("Shot-aware grading could not find usable shot samples.")
    resolved_look = "natural" if requested_look == "auto" else requested_look
    profile = LOOK_PROFILES[resolved_look]
    decisions: list[ColorGradeShotDecision] = []
    all_frames: list[np.ndarray] = []
    warnings: list[str] = []
    previous_selection: ColorGradeCandidate | None = None
    previous_analysis: ColorGradeAnalysis | None = None

    for sample in samples:
        if not sample.frames:
            raise ColorGradePlanningError(f"Shot {sample.shot_range.index} did not produce usable sample frames.")
        analysis = analyze_frames(sample.frames)
        diagnostics = _correction_diagnostics(analysis, profile)
        candidates = _generate_grade_candidates(
            sample.frames,
            requested_look=requested_look,
            grade_intensity=grade_intensity,
            candidate_count=candidate_count,
            overall_need=float(diagnostics["overall_need"]),
            profile=profile,
            before_analysis=analysis,
        )
        ranked_candidates = _rank_candidates_for_shot(candidates, previous_selection, previous_analysis, analysis)
        selected = ranked_candidates[0]
        shot_warnings = _analysis_warnings(analysis, selected.intensity)
        decision = ColorGradeShotDecision(
            index=sample.shot_range.index,
            start_sec=sample.shot_range.start_sec,
            end_sec=sample.shot_range.end_sec,
            duration_sec=sample.shot_range.duration_sec,
            sample_timestamps=[round(timestamp, 5) for timestamp in sample.timestamps],
            analysis=analysis,
            correction_need=round(float(diagnostics["overall_need"]), 5),
            confidence=round(_shot_confidence(analysis), 5),
            selected_candidate_id=selected.candidate_id,
            selected_filter_graph=selected.filter_graph,
            selected_adjustments=dict(selected.adjustments),
            selected_score=selected.score,
            candidates=ranked_candidates,
            warnings=shot_warnings,
        )
        decisions.append(decision)
        all_frames.extend(sample.frames)
        previous_selection = selected
        previous_analysis = analysis
        warnings.extend(f"Shot {decision.index}: {warning}" for warning in shot_warnings)

    render_mode = "filter_complex" if len(decisions) > 1 else "vf"
    output_label = "[vout]" if render_mode == "filter_complex" else ""
    filter_graph = build_shot_filter_complex(decisions)
    aggregate_analysis = analyze_frames(all_frames)
    aggregate_adjustments = _aggregate_selected_adjustments(decisions)
    aggregate_adjustments["shot_count"] = float(len(decisions))
    aggregate_adjustments["candidate_count"] = float(candidate_count)
    aggregate_adjustments["average_selected_score"] = round(
        float(np.mean([decision.selected_score for decision in decisions])),
        5,
    )
    aggregate_adjustments["max_shot_correction_need"] = round(
        max(decision.correction_need for decision in decisions),
        5,
    )
    manifest = ColorGradeManifest(
        mode=fallback_mode,
        requested_look=requested_look,
        resolved_look=resolved_look,
        intensity=grade_intensity,
        render_mode=render_mode,
        output_label=output_label,
        scene_threshold=round(scene_threshold, 5),
        max_shots=max_shots,
        shot_count=len(decisions),
        candidate_count=candidate_count,
        filter_graph=filter_graph,
        shots=decisions,
        warnings=warnings,
    )
    return ColorGradePlan(
        requested_look=requested_look,
        resolved_look=resolved_look,
        intensity=grade_intensity,
        filter_graph=filter_graph,
        adjustments=aggregate_adjustments,
        analysis=aggregate_analysis,
        warnings=warnings,
        render_mode=render_mode,
        output_label=output_label,
        manifest=manifest,
    )


def _generate_grade_candidates(
    frames: list[np.ndarray],
    *,
    requested_look: str,
    grade_intensity: float,
    candidate_count: int,
    overall_need: float,
    profile: dict[str, Any],
    before_analysis: ColorGradeAnalysis,
) -> list[ColorGradeCandidate]:
    candidates: list[ColorGradeCandidate] = []
    for label, candidate_intensity in _candidate_intensity_specs(overall_need, grade_intensity, candidate_count):
        plan = build_color_grade_plan_from_frames(frames, look=requested_look, intensity=candidate_intensity)
        preview_frames = _simulate_grade_frames(frames, plan.adjustments)
        after_analysis = analyze_frames(preview_frames)
        base_score, score_breakdown = _score_candidate_analysis(
            before_analysis,
            after_analysis,
            plan.adjustments,
            profile,
        )
        candidate_id = f"{label}-{_fmt(candidate_intensity)}"
        candidates.append(
            ColorGradeCandidate(
                candidate_id=candidate_id,
                label=label,
                intensity=round(candidate_intensity, 5),
                filter_graph=plan.filter_graph,
                adjustments=dict(plan.adjustments),
                score=round(base_score, 5),
                base_score=round(base_score, 5),
                continuity_penalty=0.0,
                after_analysis=after_analysis,
                score_breakdown=score_breakdown,
            )
        )
    return candidates


def _rank_candidates_for_shot(
    candidates: list[ColorGradeCandidate],
    previous_selection: ColorGradeCandidate | None,
    previous_analysis: ColorGradeAnalysis | None,
    current_analysis: ColorGradeAnalysis,
) -> list[ColorGradeCandidate]:
    ranked: list[ColorGradeCandidate] = []
    for candidate in candidates:
        continuity_penalty = 0.0
        if previous_selection is not None and previous_analysis is not None:
            continuity_penalty = _continuity_penalty(
                previous_selection.adjustments,
                candidate.adjustments,
                previous_analysis,
                current_analysis,
            )
        score = round(_clamp(candidate.base_score - continuity_penalty, 0.0, 1.0), 5)
        ranked.append(
            replace(
                candidate,
                score=score,
                continuity_penalty=round(continuity_penalty, 5),
                score_breakdown={
                    **candidate.score_breakdown,
                    "continuity_penalty": round(continuity_penalty, 5),
                    "final_score": score,
                },
            )
        )
    return sorted(ranked, key=lambda item: item.score, reverse=True)


def analyze_frames(frames: list[np.ndarray]) -> ColorGradeAnalysis:
    analyzed_frames: list[dict[str, np.ndarray | float]] = []
    skipped = 0
    for frame in frames:
        prepared = _normalize_frame(frame)
        if prepared is None:
            skipped += 1
            continue
        luma = _luma(prepared.reshape(-1, 3))
        mean_luma = float(np.mean(luma))
        luma_std = float(np.std(luma))
        if mean_luma < 0.018 or mean_luma > 0.985 or luma_std < 0.004:
            skipped += 1
            continue
        pixels = prepared.reshape(-1, 3)
        max_channel = np.max(pixels, axis=1)
        min_channel = np.min(pixels, axis=1)
        saturation = np.where(max_channel > 0.05, (max_channel - min_channel) / np.maximum(max_channel, 1e-6), 0.0)
        p05 = float(np.percentile(luma, 5))
        p95 = float(np.percentile(luma, 95))
        clip_fraction = float(np.mean((luma <= 0.018) | (luma >= 0.985)))
        midtone_fraction = float(np.mean((luma >= 0.10) & (luma <= 0.90)))
        contrast_score = _clamp((p95 - p05) / 0.48, 0.12, 1.25)
        clip_score = _clamp(1.0 - (clip_fraction * 2.8), 0.18, 1.0)
        midtone_score = _clamp(midtone_fraction / 0.72, 0.25, 1.15)
        quality = _clamp(contrast_score * clip_score * midtone_score, 0.08, 1.25)
        analyzed_frames.append(
            {
                "pixels": pixels,
                "luma": luma,
                "saturation": saturation,
                "weight": quality,
            }
        )

    if not analyzed_frames:
        for frame in frames:
            prepared = _normalize_frame(frame)
            if prepared is None:
                continue
            pixels = prepared.reshape(-1, 3)
            luma = _luma(pixels)
            max_channel = np.max(pixels, axis=1)
            min_channel = np.min(pixels, axis=1)
            saturation = np.where(max_channel > 0.05, (max_channel - min_channel) / np.maximum(max_channel, 1e-6), 0.0)
            analyzed_frames.append(
                {
                    "pixels": pixels,
                    "luma": luma,
                    "saturation": saturation,
                    "weight": 0.18,
                }
            )
    if not analyzed_frames:
        raise ColorGradePlanningError("Could not analyze video color because no usable sample frames were decoded.")

    pixels = np.concatenate([frame["pixels"] for frame in analyzed_frames], axis=0)
    luma = np.concatenate([frame["luma"] for frame in analyzed_frames], axis=0)
    saturation = np.concatenate([frame["saturation"] for frame in analyzed_frames], axis=0)
    weights = np.concatenate(
        [
            np.full(len(frame["luma"]), float(frame["weight"]), dtype=np.float32)
            for frame in analyzed_frames
        ],
        axis=0,
    )
    midtone_mask = (luma >= 0.08) & (luma <= 0.92)
    skin_mask = _skin_tone_mask(pixels, luma, saturation)
    neutral_mask = _neutral_pixel_mask(pixels, luma, saturation, skin_mask)
    neutral_weight = float(np.sum(weights[neutral_mask])) if np.any(neutral_mask) else 0.0
    total_weight = float(np.sum(weights)) or 1.0
    if neutral_weight / total_weight >= 0.015:
        channel_pixels = pixels[neutral_mask]
        channel_weights = weights[neutral_mask]
        white_balance_confidence = _clamp((neutral_weight / total_weight) / 0.12, 0.18, 1.0)
    else:
        channel_pixels = pixels[midtone_mask] if np.any(midtone_mask) else pixels
        channel_weights = weights[midtone_mask] if np.any(midtone_mask) else weights
        white_balance_confidence = 0.28
    channel_means = _weighted_channel_mean(channel_pixels, channel_weights)
    luma_p01 = _weighted_percentile(luma, weights, 1)
    luma_p05 = _weighted_percentile(luma, weights, 5)
    luma_p95 = _weighted_percentile(luma, weights, 95)
    luma_p99 = _weighted_percentile(luma, weights, 99)
    return ColorGradeAnalysis(
        sample_count=len(analyzed_frames),
        luma_mean=round(_weighted_average(luma, weights), 5),
        luma_median=round(_weighted_percentile(luma, weights, 50), 5),
        luma_std=round(float(np.sqrt(_weighted_average((luma - _weighted_average(luma, weights)) ** 2, weights))), 5),
        luma_p01=round(luma_p01, 5),
        luma_p05=round(luma_p05, 5),
        luma_p95=round(luma_p95, 5),
        luma_p99=round(luma_p99, 5),
        luma_span=round(luma_p95 - luma_p05, 5),
        saturation_mean=round(_weighted_average(saturation, weights), 5),
        red_mean=round(float(channel_means[0]), 5),
        green_mean=round(float(channel_means[1]), 5),
        blue_mean=round(float(channel_means[2]), 5),
        black_clip_fraction=round(_weighted_average((luma <= 0.018).astype(np.float32), weights), 5),
        white_clip_fraction=round(_weighted_average((luma >= 0.985).astype(np.float32), weights), 5),
        midtone_fraction=round(_weighted_average(midtone_mask.astype(np.float32), weights), 5),
        neutral_pixel_fraction=round(neutral_weight / total_weight, 5),
        skin_pixel_fraction=round(_weighted_average(skin_mask.astype(np.float32), weights), 5),
        white_balance_confidence=round(float(white_balance_confidence), 5),
        frame_quality_mean=round(float(np.mean([float(frame["weight"]) for frame in analyzed_frames])), 5),
        skipped_frame_count=skipped,
    )


def sample_video_frames(
    input_path: str,
    metadata: dict[str, Any],
    *,
    sample_count: int = 9,
    max_dimension: int = 160,
) -> list[np.ndarray]:
    count = max(1, min(int(sample_count or 9), 15))
    width = int(metadata.get("width") or 0)
    height = int(metadata.get("height") or 0)
    duration = max(float(metadata.get("duration_sec") or 0.0), 0.0)
    if width <= 0 or height <= 0:
        raise ColorGradePlanningError("Cannot sample frames because video dimensions are missing.")
    timestamps = _sample_timestamps(duration, count)
    decoded = _sample_frames_at_timestamps(
        input_path,
        width,
        height,
        timestamps,
        max_dimension=max_dimension,
    )
    frames = [frame for _timestamp, frame in decoded]
    if not frames:
        raise ColorGradePlanningError("FFmpeg did not return any sample frames for color analysis.")
    return frames


def sample_video_frames_for_range(
    input_path: str,
    metadata: dict[str, Any],
    start_sec: float,
    end_sec: float,
    *,
    sample_count: int = 4,
    max_dimension: int = 160,
) -> tuple[list[np.ndarray], list[float]]:
    count = max(1, min(int(sample_count or 4), 8))
    width = int(metadata.get("width") or 0)
    height = int(metadata.get("height") or 0)
    if width <= 0 or height <= 0:
        raise ColorGradePlanningError("Cannot sample shot frames because video dimensions are missing.")
    timestamps = _sample_timestamps_for_range(float(start_sec), float(end_sec), count)
    decoded = _sample_frames_at_timestamps(
        input_path,
        width,
        height,
        timestamps,
        max_dimension=max_dimension,
    )
    frames = [frame for _timestamp, frame in decoded]
    decoded_timestamps = [round(timestamp, 5) for timestamp, _frame in decoded]
    if not frames:
        raise ColorGradePlanningError("FFmpeg did not return any sample frames for shot color analysis.")
    return frames, decoded_timestamps


def _sample_frames_at_timestamps(
    input_path: str,
    width: int,
    height: int,
    timestamps: list[float],
    *,
    max_dimension: int,
) -> list[tuple[float, np.ndarray]]:
    sample_width, sample_height = _sample_dimensions(width, height, max_dimension=max_dimension)
    decoded: list[tuple[float, np.ndarray]] = []
    for timestamp in timestamps:
        command = [
            config.FFMPEG_PATH,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            input_path,
            "-frames:v",
            "1",
            "-vf",
            f"scale={sample_width}:{sample_height}:flags=bilinear,format=rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        try:
            result = subprocess.run(command, capture_output=True, timeout=30)
        except subprocess.TimeoutExpired:
            continue
        expected_size = sample_width * sample_height * 3
        if result.returncode != 0 or len(result.stdout) < expected_size:
            continue
        frame = np.frombuffer(result.stdout[:expected_size], dtype=np.uint8).reshape((sample_height, sample_width, 3))
        decoded.append((float(timestamp), frame))
    return decoded


def build_filter_graph(adjustments: dict[str, float], color_balance: dict[str, float] | None = None) -> str:
    red_gain = _fmt(adjustments["red_gain"])
    green_gain = _fmt(adjustments["green_gain"])
    blue_gain = _fmt(adjustments["blue_gain"])
    filters = [
        "format=rgb24",
        (
            "lutrgb="
            f"r='clip(val*{red_gain},0,255)':"
            f"g='clip(val*{green_gain},0,255)':"
            f"b='clip(val*{blue_gain},0,255)'"
        ),
    ]
    level_input_black = float(adjustments.get("level_input_black", 0.0) or 0.0)
    level_input_white = float(adjustments.get("level_input_white", 1.0) or 1.0)
    if level_input_black > 0.001 or level_input_white < 0.999:
        filters.append(
            "colorlevels="
            f"rimin={_fmt(level_input_black)}:"
            f"gimin={_fmt(level_input_black)}:"
            f"bimin={_fmt(level_input_black)}:"
            f"rimax={_fmt(level_input_white)}:"
            f"gimax={_fmt(level_input_white)}:"
            f"bimax={_fmt(level_input_white)}:"
            "preserve=lum"
        )
    filters.append(
        (
            "eq="
            f"brightness={_fmt(adjustments['brightness'])}:"
            f"contrast={_fmt(adjustments['contrast'])}:"
            f"saturation={_fmt(adjustments['saturation'])}:"
            f"gamma={_fmt(adjustments['gamma'])}:"
            "gamma_weight=0.85"
        )
    )
    curve_shadow = float(adjustments.get("curve_shadow", 0.25) or 0.25)
    curve_highlight = float(adjustments.get("curve_highlight", 0.75) or 0.75)
    if abs(curve_shadow - 0.25) >= 0.003 or abs(curve_highlight - 0.75) >= 0.003:
        filters.append(
            "curves="
            f"master='0/0 0.25/{_fmt(curve_shadow)} 0.75/{_fmt(curve_highlight)} 1/1':"
            "interp=pchip"
        )
    balance = dict(color_balance or {})
    if balance:
        allowed = ("rs", "gs", "bs", "rm", "gm", "bm", "rh", "gh", "bh")
        options = [f"{key}={_fmt(balance[key])}" for key in allowed if key in balance]
        options.append("pl=1")
        filters.append("colorbalance=" + ":".join(options))
    filters.append("format=yuv420p")
    return ",".join(filters)


def validate_color_grade_output(
    output_path: str,
    metadata: dict[str, Any],
    *,
    sample_count: int = 5,
) -> dict[str, Any]:
    frames = sample_video_frames(
        output_path,
        metadata,
        sample_count=max(3, min(int(sample_count or 5), 7)),
    )
    analysis = analyze_frames(frames)
    return validate_color_grade_analysis(analysis)


def validate_color_grade_analysis(analysis: ColorGradeAnalysis) -> dict[str, Any]:
    warnings: list[str] = []
    penalty = 0.0
    if analysis.black_clip_fraction > 0.10:
        warnings.append("Output still has heavy crushed-shadow clipping.")
        penalty += min((analysis.black_clip_fraction - 0.10) * 2.2, 0.25)
    elif analysis.black_clip_fraction > 0.045:
        warnings.append("Output has noticeable shadow clipping.")
        penalty += min((analysis.black_clip_fraction - 0.045) * 1.4, 0.12)
    if analysis.white_clip_fraction > 0.10:
        warnings.append("Output still has heavy clipped-highlight clipping.")
        penalty += min((analysis.white_clip_fraction - 0.10) * 2.2, 0.25)
    elif analysis.white_clip_fraction > 0.045:
        warnings.append("Output has noticeable highlight clipping.")
        penalty += min((analysis.white_clip_fraction - 0.045) * 1.4, 0.12)
    if analysis.luma_median < 0.18:
        warnings.append("Output remains very dark after grading.")
        penalty += min((0.18 - analysis.luma_median) * 1.1, 0.18)
    elif analysis.luma_median > 0.82:
        warnings.append("Output remains very bright after grading.")
        penalty += min((analysis.luma_median - 0.82) * 1.1, 0.18)
    if analysis.luma_span < 0.20:
        warnings.append("Output remains low contrast after grading.")
        penalty += min((0.20 - analysis.luma_span) * 0.8, 0.14)
    if analysis.saturation_mean > 0.78:
        warnings.append("Output saturation is high enough to risk unnatural color.")
        penalty += min((analysis.saturation_mean - 0.78) * 0.35, 0.08)
    if analysis.frame_quality_mean < 0.16:
        warnings.append("Output validation confidence is low because sampled frames were weak.")
        penalty += 0.08
    score = round(_clamp(1.0 - penalty, 0.0, 1.0), 4)
    return {
        "passed": score >= 0.68,
        "score": score,
        "warnings": warnings,
        "analysis": analysis.to_dict(),
    }


def _normalize_grading_mode(value: str | None) -> str:
    raw = (value or "auto").strip().lower().replace("-", "_").replace(" ", "_")
    if raw in {"auto", "shot_aware", "scene_aware"}:
        return "shot_aware" if raw == "scene_aware" else raw
    if raw in {"fast", "global", "fast_global", "legacy"}:
        return "fast_global"
    supported = "auto, shot_aware, fast_global"
    raise ColorGradePlanningError(f"Unsupported auto color grading mode {value!r}. Supported modes: {supported}.")


def _build_shot_ranges(
    boundaries: list[float],
    duration: float,
    *,
    max_shots: int,
    min_shot_duration: float,
) -> list[ColorGradeShotRange]:
    clean_boundaries: list[float] = []
    last_boundary = 0.0
    for boundary in sorted(set(round(float(item), 3) for item in boundaries)):
        if boundary <= 0.0 or boundary >= duration:
            continue
        if boundary - last_boundary < min_shot_duration:
            continue
        if duration - boundary < min_shot_duration:
            continue
        clean_boundaries.append(boundary)
        last_boundary = boundary

    points = [0.0, *clean_boundaries, max(duration, 0.001)]
    ranges = [
        ColorGradeShotRange(
            index=index,
            start_sec=round(points[index], 5),
            end_sec=round(points[index + 1], 5),
            duration_sec=round(points[index + 1] - points[index], 5),
        )
        for index in range(len(points) - 1)
        if points[index + 1] - points[index] > 0.001
    ]
    if not ranges:
        ranges = [ColorGradeShotRange(index=0, start_sec=0.0, end_sec=round(max(duration, 0.001), 5), duration_sec=round(max(duration, 0.001), 5))]
    return _merge_shot_ranges_to_limit(ranges, max_shots)


def _merge_shot_ranges_to_limit(ranges: list[ColorGradeShotRange], max_shots: int) -> list[ColorGradeShotRange]:
    merged = list(ranges)
    while len(merged) > max_shots:
        merge_index = min(
            range(len(merged) - 1),
            key=lambda index: merged[index].duration_sec + merged[index + 1].duration_sec,
        )
        first = merged[merge_index]
        second = merged[merge_index + 1]
        combined = ColorGradeShotRange(
            index=first.index,
            start_sec=first.start_sec,
            end_sec=second.end_sec,
            duration_sec=round(second.end_sec - first.start_sec, 5),
        )
        merged = [*merged[:merge_index], combined, *merged[merge_index + 2 :]]
    return [
        ColorGradeShotRange(
            index=index,
            start_sec=shot.start_sec,
            end_sec=shot.end_sec,
            duration_sec=round(shot.end_sec - shot.start_sec, 5),
        )
        for index, shot in enumerate(merged)
    ]


def _sample_shot(
    input_path: str,
    metadata: dict[str, Any],
    shot_range: ColorGradeShotRange,
    *,
    sample_count: int,
) -> _ShotSample:
    frames, timestamps = sample_video_frames_for_range(
        input_path,
        metadata,
        shot_range.start_sec,
        shot_range.end_sec,
        sample_count=sample_count,
    )
    return _ShotSample(shot_range=shot_range, frames=frames, timestamps=timestamps)


def _candidate_intensity_specs(
    overall_need: float,
    grade_intensity: float,
    candidate_count: int,
) -> list[tuple[str, float]]:
    if grade_intensity <= 0.0:
        return [("identity", 0.0)]
    need = _clamp(overall_need, 0.0, 1.0)
    if need < 0.20:
        raw_specs = [("protect", 0.42), ("subtle", 0.65), ("balanced", 0.92), ("styled", 1.08)]
    elif need < 0.55:
        raw_specs = [("protect", 0.58), ("balanced", 0.92), ("assertive", 1.18), ("rescue", 1.36)]
    else:
        raw_specs = [("guarded", 0.78), ("balanced", 1.05), ("assertive", 1.32), ("rescue", 1.50), ("max_rescue", 1.68)]

    specs: list[tuple[str, float]] = []
    seen: set[int] = set()
    for label, multiplier in raw_specs:
        candidate_intensity = _clamp(grade_intensity * multiplier, 0.0, 1.5)
        key = int(round(candidate_intensity * 1000))
        if key in seen:
            continue
        seen.add(key)
        specs.append((label, candidate_intensity))
        if len(specs) >= candidate_count:
            break
    return specs or [("balanced", _clamp(grade_intensity, 0.0, 1.5))]


def _simulate_grade_frames(frames: list[np.ndarray], adjustments: dict[str, float]) -> list[np.ndarray]:
    simulated: list[np.ndarray] = []
    red_gain = float(adjustments.get("red_gain", 1.0))
    green_gain = float(adjustments.get("green_gain", 1.0))
    blue_gain = float(adjustments.get("blue_gain", 1.0))
    level_black = float(adjustments.get("level_input_black", 0.0) or 0.0)
    level_white = float(adjustments.get("level_input_white", 1.0) or 1.0)
    brightness = float(adjustments.get("brightness", 0.0))
    contrast = float(adjustments.get("contrast", 1.0))
    saturation = float(adjustments.get("saturation", 1.0))
    gamma = max(float(adjustments.get("gamma", 1.0)), 0.05)
    curve_shadow = float(adjustments.get("curve_shadow", 0.25) or 0.25)
    curve_highlight = float(adjustments.get("curve_highlight", 0.75) or 0.75)

    for frame in frames:
        prepared = _normalize_frame(frame)
        if prepared is None:
            continue
        rgb = prepared.copy()
        rgb *= np.array([red_gain, green_gain, blue_gain], dtype=np.float32)
        rgb = np.clip(rgb, 0.0, 1.0)
        if level_black > 0.001 or level_white < 0.999:
            span = max(level_white - level_black, 0.01)
            rgb = np.clip((rgb - level_black) / span, 0.0, 1.0)
        rgb = np.clip(((rgb - 0.5) * contrast) + 0.5 + brightness, 0.0, 1.0)
        rgb = np.power(rgb, 1.0 / gamma)
        luma = (
            (rgb[..., 0] * 0.2126)
            + (rgb[..., 1] * 0.7152)
            + (rgb[..., 2] * 0.0722)
        )
        rgb = np.clip(luma[..., None] + ((rgb - luma[..., None]) * saturation), 0.0, 1.0)
        if abs(curve_shadow - 0.25) >= 0.003 or abs(curve_highlight - 0.75) >= 0.003:
            rgb = np.interp(rgb, [0.0, 0.25, 0.75, 1.0], [0.0, curve_shadow, curve_highlight, 1.0])
        simulated.append((np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8))
    if not simulated:
        raise ColorGradePlanningError("Could not simulate color grade candidates because no frames were usable.")
    return simulated


def _score_candidate_analysis(
    before: ColorGradeAnalysis,
    after: ColorGradeAnalysis,
    adjustments: dict[str, float],
    profile: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    before_quality, before_breakdown = _analysis_quality_score(before, profile)
    after_quality, after_breakdown = _analysis_quality_score(after, profile)
    improvement = after_quality - before_quality
    clipping_increase = max(
        0.0,
        (after.black_clip_fraction + after.white_clip_fraction)
        - (before.black_clip_fraction + before.white_clip_fraction),
    )
    need = float(adjustments.get("overall_need", 0.0))
    change_magnitude = _adjustment_magnitude(adjustments)
    low_need_penalty = (1.0 - _clamp(need, 0.0, 1.0)) * min(change_magnitude * 0.055, 0.16)
    clipping_penalty = min(clipping_increase * 1.8, 0.20)
    skin_penalty = 0.0
    if before.skin_pixel_fraction > 0.10 and float(adjustments.get("saturation", 1.0)) > 1.20:
        skin_penalty = min((float(adjustments.get("saturation", 1.0)) - 1.20) * 0.18, 0.06)
    score = _clamp(
        after_quality
        + (0.35 * improvement)
        - low_need_penalty
        - clipping_penalty
        - skin_penalty,
        0.0,
        1.0,
    )
    return round(score, 5), {
        "before_quality": round(before_quality, 5),
        "after_quality": round(after_quality, 5),
        "improvement": round(improvement, 5),
        "exposure_score": round(after_breakdown["exposure"], 5),
        "contrast_score": round(after_breakdown["contrast"], 5),
        "saturation_score": round(after_breakdown["saturation"], 5),
        "cast_score": round(after_breakdown["cast"], 5),
        "clip_score": round(after_breakdown["clip"], 5),
        "low_need_penalty": round(low_need_penalty, 5),
        "clipping_penalty": round(clipping_penalty, 5),
        "skin_penalty": round(skin_penalty, 5),
        "change_magnitude": round(change_magnitude, 5),
        "source_exposure_score": round(before_breakdown["exposure"], 5),
    }


def _analysis_quality_score(analysis: ColorGradeAnalysis, profile: dict[str, Any]) -> tuple[float, dict[str, float]]:
    target_luma = float(profile["target_luma"])
    target_span = float(profile["target_span"])
    target_saturation = float(profile["target_saturation"])
    exposure = _clamp(1.0 - (abs(target_luma - analysis.luma_median) / 0.36), 0.0, 1.0)
    if analysis.luma_span < target_span:
        contrast = _clamp(analysis.luma_span / max(target_span, 0.001), 0.0, 1.0)
    else:
        contrast = _clamp(1.0 - ((analysis.luma_span - target_span) / 0.38), 0.0, 1.0)
    saturation = _clamp(1.0 - (abs(target_saturation - analysis.saturation_mean) / 0.42), 0.0, 1.0)
    cast = _clamp(1.0 - _color_cast_score(analysis), 0.0, 1.0)
    clip = _clamp(1.0 - ((analysis.black_clip_fraction + analysis.white_clip_fraction) / 0.14), 0.0, 1.0)
    midtone = _clamp(analysis.midtone_fraction / 0.72, 0.0, 1.0)
    quality = _clamp(
        (0.29 * exposure)
        + (0.24 * contrast)
        + (0.16 * saturation)
        + (0.18 * cast)
        + (0.09 * clip)
        + (0.04 * midtone),
        0.0,
        1.0,
    )
    return quality, {
        "exposure": exposure,
        "contrast": contrast,
        "saturation": saturation,
        "cast": cast,
        "clip": clip,
        "midtone": midtone,
    }


def _adjustment_magnitude(adjustments: dict[str, float]) -> float:
    gain_delta = max(
        abs(float(adjustments.get("red_gain", 1.0)) - 1.0),
        abs(float(adjustments.get("green_gain", 1.0)) - 1.0),
        abs(float(adjustments.get("blue_gain", 1.0)) - 1.0),
    )
    return (
        abs(float(adjustments.get("brightness", 0.0))) / 0.24
        + abs(float(adjustments.get("contrast", 1.0)) - 1.0) / 0.46
        + abs(float(adjustments.get("saturation", 1.0)) - 1.0) / 0.72
        + abs(float(adjustments.get("gamma", 1.0)) - 1.0) / 0.24
        + gain_delta / 0.30
    )


def _continuity_penalty(
    previous_adjustments: dict[str, float],
    current_adjustments: dict[str, float],
    previous_analysis: ColorGradeAnalysis,
    current_analysis: ColorGradeAnalysis,
) -> float:
    similarity = _analysis_similarity(previous_analysis, current_analysis)
    if similarity <= 0.0:
        return 0.0
    delta = (
        abs(float(previous_adjustments.get("brightness", 0.0)) - float(current_adjustments.get("brightness", 0.0))) / 0.24
        + abs(float(previous_adjustments.get("contrast", 1.0)) - float(current_adjustments.get("contrast", 1.0))) / 0.46
        + abs(float(previous_adjustments.get("saturation", 1.0)) - float(current_adjustments.get("saturation", 1.0))) / 0.72
        + abs(float(previous_adjustments.get("red_gain", 1.0)) - float(current_adjustments.get("red_gain", 1.0))) / 0.30
        + abs(float(previous_adjustments.get("blue_gain", 1.0)) - float(current_adjustments.get("blue_gain", 1.0))) / 0.30
    )
    return _clamp(similarity * min(delta / 5.0, 1.0) * 0.10, 0.0, 0.14)


def _analysis_similarity(previous: ColorGradeAnalysis, current: ColorGradeAnalysis) -> float:
    luma_delta = abs(previous.luma_median - current.luma_median) / 0.42
    span_delta = abs(previous.luma_span - current.luma_span) / 0.55
    saturation_delta = abs(previous.saturation_mean - current.saturation_mean) / 0.42
    cast_delta = abs(_color_cast_score(previous) - _color_cast_score(current)) / 0.65
    return _clamp(1.0 - ((0.36 * luma_delta) + (0.24 * span_delta) + (0.20 * saturation_delta) + (0.20 * cast_delta)), 0.0, 1.0)


def _shot_confidence(analysis: ColorGradeAnalysis) -> float:
    sample_score = _clamp(analysis.sample_count / 4.0, 0.15, 1.0)
    quality_score = _clamp(analysis.frame_quality_mean / 0.72, 0.15, 1.0)
    neutral_bonus = _clamp(analysis.neutral_pixel_fraction / 0.08, 0.0, 1.0) * 0.18
    return _clamp((0.48 * sample_score) + (0.40 * quality_score) + neutral_bonus, 0.0, 1.0)


def _aggregate_selected_adjustments(decisions: list[ColorGradeShotDecision]) -> dict[str, float]:
    keys = sorted(
        {
            key
            for decision in decisions
            for key, value in decision.selected_adjustments.items()
            if isinstance(value, (int, float))
        }
    )
    total_duration = sum(max(decision.duration_sec, 0.001) for decision in decisions) or 1.0
    aggregate: dict[str, float] = {}
    for key in keys:
        weighted = sum(
            float(decision.selected_adjustments.get(key, 0.0)) * max(decision.duration_sec, 0.001)
            for decision in decisions
        )
        aggregate[key] = round(weighted / total_duration, 5)
    return aggregate


def _weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    if total <= 0.0:
        return float(np.mean(values))
    return float(np.sum(values * weights) / total)


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    if values.size == 0:
        return 0.0
    if weights.size != values.size or float(np.sum(weights)) <= 0.0:
        return float(np.percentile(values, percentile))
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    cutoff = (float(percentile) / 100.0) * float(cumulative[-1])
    index = int(np.searchsorted(cumulative, cutoff, side="left"))
    return float(sorted_values[min(max(index, 0), len(sorted_values) - 1)])


def _weighted_channel_mean(pixels: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if pixels.size == 0:
        return np.array([0.5, 0.5, 0.5], dtype=np.float64)
    if weights.size != pixels.shape[0] or float(np.sum(weights)) <= 0.0:
        return np.mean(pixels, axis=0)
    normalized = weights / float(np.sum(weights))
    return np.sum(pixels * normalized[:, None], axis=0)


def _neutral_pixel_mask(
    pixels: np.ndarray,
    luma: np.ndarray,
    saturation: np.ndarray,
    skin_mask: np.ndarray,
) -> np.ndarray:
    channel_spread = np.max(pixels, axis=1) - np.min(pixels, axis=1)
    return (
        (luma >= 0.16)
        & (luma <= 0.86)
        & (saturation <= 0.20)
        & (channel_spread <= 0.11)
        & ~skin_mask
    )


def _skin_tone_mask(pixels: np.ndarray, luma: np.ndarray, saturation: np.ndarray) -> np.ndarray:
    red = pixels[:, 0]
    green = pixels[:, 1]
    blue = pixels[:, 2]
    return (
        (luma >= 0.18)
        & (luma <= 0.88)
        & (saturation >= 0.12)
        & (saturation <= 0.68)
        & (red > green)
        & (green > blue * 0.72)
        & ((red - green) >= 0.018)
        & ((red - green) <= 0.28)
        & ((red - blue) >= 0.045)
    )


def _correction_diagnostics(analysis: ColorGradeAnalysis, profile: dict[str, Any]) -> dict[str, float]:
    target_luma = float(profile["target_luma"])
    target_span = float(profile["target_span"])
    target_saturation = float(profile["target_saturation"])
    exposure_error = _clamp(abs(target_luma - analysis.luma_median) / 0.28, 0.0, 1.0)
    if analysis.luma_span < target_span:
        contrast_error = _clamp((target_span - analysis.luma_span) / max(target_span, 0.001), 0.0, 1.0)
    else:
        contrast_error = _clamp((analysis.luma_span - target_span) / 0.45, 0.0, 1.0) * 0.65
    saturation_error = _clamp(abs(target_saturation - analysis.saturation_mean) / 0.34, 0.0, 1.0)
    cast_error = _color_cast_score(analysis)
    clip_risk = _clamp((analysis.black_clip_fraction + analysis.white_clip_fraction) / 0.12, 0.0, 1.0)
    overall_need = _clamp(
        (0.34 * exposure_error)
        + (0.27 * contrast_error)
        + (0.17 * saturation_error)
        + (0.17 * cast_error)
        + (0.05 * clip_risk),
        0.0,
        1.0,
    )
    correction_strength = _clamp(0.28 + (1.38 * (overall_need ** 0.85)), 0.28, 1.65)
    if analysis.frame_quality_mean < 0.20:
        correction_strength *= 0.80
    elif analysis.frame_quality_mean < 0.34:
        correction_strength *= 0.90
    return {
        "exposure_error": exposure_error,
        "contrast_error": contrast_error,
        "saturation_error": saturation_error,
        "cast_error": cast_error,
        "clip_risk": clip_risk,
        "overall_need": overall_need,
        "correction_strength": _clamp(correction_strength, 0.22, 1.65),
    }


def _color_cast_score(analysis: ColorGradeAnalysis) -> float:
    channels = np.array(
        [
            max(analysis.red_mean, 0.025),
            max(analysis.green_mean, 0.025),
            max(analysis.blue_mean, 0.025),
        ],
        dtype=np.float64,
    )
    neutral = float(np.mean(channels))
    if neutral <= 0.0:
        return 0.0
    relative = channels / neutral
    return _clamp(float(np.max(np.abs(relative - 1.0))) / 0.32, 0.0, 1.0)


def _white_balance_gains(analysis: ColorGradeAnalysis, *, max_delta: float = 0.16) -> tuple[float, float, float]:
    channels = np.array(
        [
            max(analysis.red_mean, 0.025),
            max(analysis.green_mean, 0.025),
            max(analysis.blue_mean, 0.025),
        ],
        dtype=np.float64,
    )
    neutral = float(np.mean(channels))
    gains = neutral / channels
    delta = _clamp(max_delta, 0.08, 0.28)
    return tuple(float(_clamp(value, 1.0 - delta, 1.0 + delta)) for value in gains)


def _level_inputs(analysis: ColorGradeAnalysis, strength: float) -> tuple[float, float]:
    level_strength = _clamp(strength, 0.0, 1.0)
    if level_strength <= 0.0:
        return 0.0, 1.0
    clip_guard = _clamp(1.0 - ((analysis.black_clip_fraction + analysis.white_clip_fraction) * 7.0), 0.22, 1.0)
    contrast_deficit = _clamp((0.68 - analysis.luma_span) / 0.68, 0.0, 1.0)
    black_target = _clamp(analysis.luma_p01 * 0.82, 0.0, 0.20)
    if analysis.luma_p99 < 0.62:
        white_target = _clamp(analysis.luma_p99 + 0.24, 0.52, 0.82)
    elif analysis.luma_p99 < 0.88:
        white_target = _clamp(analysis.luma_p99 + 0.08, 0.72, 0.95)
    else:
        white_target = 1.0 - _clamp((1.0 - analysis.luma_p99) * 0.60, 0.0, 0.045)
    amount = _clamp(level_strength * (0.35 + 0.65 * contrast_deficit) * clip_guard, 0.0, 1.0)
    black = _clamp(black_target * amount, 0.0, 0.18)
    white = _clamp(1.0 - ((1.0 - white_target) * amount), 0.52, 1.0)
    if white - black < 0.86:
        white = min(1.0, black + 0.86)
        black = max(0.0, white - 0.86)
    return black, white


def _curve_points(analysis: ColorGradeAnalysis, strength: float) -> tuple[float, float]:
    curve_strength = _clamp(strength, 0.0, 1.0)
    if curve_strength <= 0.0:
        return 0.25, 0.75
    contrast_deficit = _clamp((0.70 - analysis.luma_span) / 0.70, 0.0, 1.0)
    clipping_guard = _clamp(1.0 - ((analysis.black_clip_fraction + analysis.white_clip_fraction) * 8.0), 0.25, 1.0)
    amount = curve_strength * (0.55 + 0.45 * contrast_deficit) * clipping_guard
    shadow = _clamp(0.25 - (0.070 * amount), 0.18, 0.25)
    highlight = _clamp(0.75 + (0.085 * amount), 0.75, 0.84)
    return shadow, highlight


def _analysis_warnings(analysis: ColorGradeAnalysis, intensity: float) -> list[str]:
    warnings: list[str] = []
    if analysis.sample_count < 3:
        warnings.append("Only a few usable frames were available, so the grade is conservative.")
    if analysis.luma_p01 <= 0.015:
        warnings.append("The source has clipped or near-black shadows that grading cannot fully recover.")
    if analysis.luma_p99 >= 0.985:
        warnings.append("The source has clipped or near-white highlights that grading cannot fully recover.")
    if analysis.neutral_pixel_fraction < 0.015:
        warnings.append("Few neutral midtone pixels were available, so white balance was intentionally conservative.")
    if analysis.skin_pixel_fraction > 0.10:
        warnings.append("Skin-tone-like pixels were detected, so color balance and saturation changes were guarded.")
    if analysis.frame_quality_mean < 0.25:
        warnings.append("Sampled frames had low color-analysis confidence; the grade was bounded to avoid artifacts.")
    if intensity > 1.2:
        warnings.append("High intensity can create a stylized look and may amplify noise or compression artifacts.")
    return warnings


def _normalize_frame(frame: np.ndarray) -> np.ndarray | None:
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] < 3 or array.shape[0] <= 0 or array.shape[1] <= 0:
        return None
    rgb = array[..., :3].astype(np.float32, copy=False)
    if float(np.nanmax(rgb)) > 1.5:
        rgb = rgb / 255.0
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(rgb, 0.0, 1.0)


def _luma(pixels: np.ndarray) -> np.ndarray:
    return pixels[:, 0] * 0.2126 + pixels[:, 1] * 0.7152 + pixels[:, 2] * 0.0722


def _sample_dimensions(width: int, height: int, *, max_dimension: int) -> tuple[int, int]:
    if width >= height:
        sample_width = max(2, int(max_dimension))
        sample_height = max(2, int(round(height * sample_width / width)))
    else:
        sample_height = max(2, int(max_dimension))
        sample_width = max(2, int(round(width * sample_height / height)))
    return sample_width, sample_height


def _sample_timestamps(duration: float, sample_count: int) -> list[float]:
    if duration <= 0.0:
        return [0.0]
    if sample_count <= 1:
        return [max(duration * 0.5, 0.0)]
    guard = min(max(duration * 0.08, 0.15), 1.5)
    start = min(guard, max(duration * 0.10, 0.0))
    end = max(duration - guard, start)
    if end <= start:
        start = 0.0
        end = duration
    span = max(end - start, 0.0)
    return [start + span * ((index + 0.5) / sample_count) for index in range(sample_count)]


def _sample_timestamps_for_range(start_sec: float, end_sec: float, sample_count: int) -> list[float]:
    start = max(float(start_sec), 0.0)
    end = max(float(end_sec), start + 0.001)
    count = max(1, int(sample_count or 1))
    duration = end - start
    if count <= 1:
        return [start + (duration * 0.5)]
    guard = min(max(duration * 0.10, 0.04), 0.50)
    sample_start = start + guard
    sample_end = end - guard
    if sample_end <= sample_start:
        sample_start = start
        sample_end = end
    span = max(sample_end - sample_start, 0.0)
    return [sample_start + span * ((index + 0.5) / count) for index in range(count)]


def _validate_intensity(value: float) -> float:
    try:
        intensity = float(value)
    except (TypeError, ValueError) as exc:
        raise ColorGradePlanningError("Color grade intensity must be a number between 0.0 and 1.5.") from exc
    if intensity < 0.0 or intensity > 1.5:
        raise ColorGradePlanningError("Color grade intensity must be between 0.0 and 1.5.")
    return intensity


def _blend_identity(value: float, amount: float) -> float:
    return 1.0 + ((float(value) - 1.0) * amount)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(float(value), maximum))


def _fmt(value: float) -> str:
    rendered = f"{float(value):.5f}".rstrip("0").rstrip(".")
    return rendered if rendered not in {"", "-0"} else "0"


__all__ = [
    "ColorGradeAnalysis",
    "ColorGradeCandidate",
    "ColorGradeManifest",
    "ColorGradePlan",
    "ColorGradePlanningError",
    "ColorGradeShotDecision",
    "ColorGradeShotRange",
    "SUPPORTED_COLOR_GRADE_LOOKS",
    "analyze_frames",
    "build_color_grade_plan",
    "build_color_grade_plan_from_frames",
    "build_filter_graph",
    "build_shot_aware_color_grade_plan",
    "build_shot_aware_color_grade_plan_from_shots",
    "build_shot_filter_complex",
    "detect_video_shots",
    "normalize_color_grade_look",
    "sample_video_frames",
    "sample_video_frames_for_range",
    "validate_color_grade_analysis",
    "validate_color_grade_output",
    "validate_color_grade_output_by_shots",
]
