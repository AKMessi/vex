from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
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
class ColorGradePlan:
    requested_look: str
    resolved_look: str
    intensity: float
    filter_graph: str
    adjustments: dict[str, float]
    analysis: ColorGradeAnalysis
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_look": self.requested_look,
            "resolved_look": self.resolved_look,
            "intensity": self.intensity,
            "filter_graph": self.filter_graph,
            "adjustments": dict(self.adjustments),
            "analysis": self.analysis.to_dict(),
            "warnings": list(self.warnings),
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
    sample_width, sample_height = _sample_dimensions(width, height, max_dimension=max_dimension)
    timestamps = _sample_timestamps(duration, count)
    frames: list[np.ndarray] = []
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
        frames.append(frame)
    if not frames:
        raise ColorGradePlanningError("FFmpeg did not return any sample frames for color analysis.")
    return frames


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
    "ColorGradePlan",
    "ColorGradePlanningError",
    "SUPPORTED_COLOR_GRADE_LOOKS",
    "analyze_frames",
    "build_color_grade_plan",
    "build_color_grade_plan_from_frames",
    "build_filter_graph",
    "normalize_color_grade_look",
    "sample_video_frames",
    "validate_color_grade_analysis",
    "validate_color_grade_output",
]
