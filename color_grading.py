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
        "balance": {},
    },
    "vibrant": {
        "contrast": 1.055,
        "saturation": 1.13,
        "target_luma": 0.49,
        "target_span": 0.68,
        "target_saturation": 0.38,
        "rgb_gain": (1.01, 1.005, 0.995),
        "balance": {},
    },
    "cinematic": {
        "contrast": 1.075,
        "saturation": 1.055,
        "target_luma": 0.46,
        "target_span": 0.70,
        "target_saturation": 0.34,
        "rgb_gain": (1.01, 0.998, 1.012),
        "balance": {"bs": 0.018, "rm": 0.006, "rh": 0.018, "bh": -0.014},
    },
    "warm": {
        "contrast": 1.025,
        "saturation": 1.045,
        "target_luma": 0.49,
        "target_span": 0.66,
        "target_saturation": 0.33,
        "rgb_gain": (1.035, 1.01, 0.975),
        "balance": {"rm": 0.01, "bm": -0.008, "rh": 0.012, "bh": -0.01},
    },
    "cool": {
        "contrast": 1.03,
        "saturation": 1.035,
        "target_luma": 0.48,
        "target_span": 0.66,
        "target_saturation": 0.32,
        "rgb_gain": (0.982, 0.998, 1.038),
        "balance": {"rs": -0.008, "bs": 0.012, "bm": 0.01},
    },
    "documentary": {
        "contrast": 1.025,
        "saturation": 1.0,
        "target_luma": 0.50,
        "target_span": 0.64,
        "target_saturation": 0.30,
        "rgb_gain": (1.004, 1.004, 0.996),
        "balance": {},
    },
    "punchy": {
        "contrast": 1.105,
        "saturation": 1.095,
        "target_luma": 0.48,
        "target_span": 0.72,
        "target_saturation": 0.36,
        "rgb_gain": (1.008, 1.0, 0.998),
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
    luma_p01: float
    luma_p05: float
    luma_p95: float
    luma_p99: float
    saturation_mean: float
    red_mean: float
    green_mean: float
    blue_mean: float
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
    sample_count: int = 7,
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

    luma_span = max(analysis.luma_p95 - analysis.luma_p05, 0.001)
    brightness = _clamp((float(profile["target_luma"]) - analysis.luma_median) * 0.18, -0.075, 0.075)
    gamma = 1.0 + _clamp((float(profile["target_luma"]) - analysis.luma_median) * 0.16, -0.055, 0.075)
    contrast_auto = 1.0 + _clamp((float(profile["target_span"]) - luma_span) * 0.30, -0.10, 0.16)
    saturation_auto = 1.0 + _clamp((float(profile["target_saturation"]) - analysis.saturation_mean) * 0.45, -0.08, 0.18)

    contrast = _blend_identity(contrast_auto * float(profile["contrast"]), grade_intensity)
    saturation = _blend_identity(saturation_auto * float(profile["saturation"]), grade_intensity)
    brightness *= grade_intensity
    gamma = _blend_identity(gamma, grade_intensity)

    auto_gains = _white_balance_gains(analysis)
    profile_gains = tuple(float(value) for value in profile["rgb_gain"])
    red_gain = _clamp(_blend_identity(auto_gains[0], 0.68 * grade_intensity) * _blend_identity(profile_gains[0], grade_intensity), 0.88, 1.12)
    green_gain = _clamp(_blend_identity(auto_gains[1], 0.68 * grade_intensity) * _blend_identity(profile_gains[1], grade_intensity), 0.88, 1.12)
    blue_gain = _clamp(_blend_identity(auto_gains[2], 0.68 * grade_intensity) * _blend_identity(profile_gains[2], grade_intensity), 0.88, 1.12)

    color_balance = {
        key: round(float(value) * grade_intensity, 5)
        for key, value in dict(profile.get("balance") or {}).items()
        if abs(float(value) * grade_intensity) >= 0.0005
    }
    adjustments = {
        "brightness": round(brightness, 5),
        "contrast": round(_clamp(contrast, 0.88, 1.24), 5),
        "saturation": round(_clamp(saturation, 0.82, 1.42), 5),
        "gamma": round(_clamp(gamma, 0.88, 1.16), 5),
        "red_gain": round(red_gain, 5),
        "green_gain": round(green_gain, 5),
        "blue_gain": round(blue_gain, 5),
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
    normalized_frames: list[np.ndarray] = []
    skipped = 0
    for frame in frames:
        prepared = _normalize_frame(frame)
        if prepared is None:
            skipped += 1
            continue
        luma = _luma(prepared.reshape(-1, 3))
        if float(np.mean(luma)) < 0.018 or float(np.mean(luma)) > 0.985 or float(np.std(luma)) < 0.004:
            skipped += 1
            continue
        normalized_frames.append(prepared)

    if not normalized_frames:
        normalized_frames = [prepared for frame in frames if (prepared := _normalize_frame(frame)) is not None]
    if not normalized_frames:
        raise ColorGradePlanningError("Could not analyze video color because no usable sample frames were decoded.")

    pixels = np.concatenate([frame.reshape(-1, 3) for frame in normalized_frames], axis=0)
    luma = _luma(pixels)
    midtone_mask = (luma >= 0.08) & (luma <= 0.92)
    midtone_pixels = pixels[midtone_mask] if np.any(midtone_mask) else pixels
    max_channel = np.max(pixels, axis=1)
    min_channel = np.min(pixels, axis=1)
    saturation = np.where(max_channel > 0.05, (max_channel - min_channel) / np.maximum(max_channel, 1e-6), 0.0)
    channel_means = np.mean(midtone_pixels, axis=0)
    return ColorGradeAnalysis(
        sample_count=len(normalized_frames),
        luma_mean=round(float(np.mean(luma)), 5),
        luma_median=round(float(np.median(luma)), 5),
        luma_p01=round(float(np.percentile(luma, 1)), 5),
        luma_p05=round(float(np.percentile(luma, 5)), 5),
        luma_p95=round(float(np.percentile(luma, 95)), 5),
        luma_p99=round(float(np.percentile(luma, 99)), 5),
        saturation_mean=round(float(np.mean(saturation)), 5),
        red_mean=round(float(channel_means[0]), 5),
        green_mean=round(float(channel_means[1]), 5),
        blue_mean=round(float(channel_means[2]), 5),
        skipped_frame_count=skipped,
    )


def sample_video_frames(
    input_path: str,
    metadata: dict[str, Any],
    *,
    sample_count: int = 7,
    max_dimension: int = 160,
) -> list[np.ndarray]:
    count = max(1, min(int(sample_count or 7), 15))
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
        (
            "eq="
            f"brightness={_fmt(adjustments['brightness'])}:"
            f"contrast={_fmt(adjustments['contrast'])}:"
            f"saturation={_fmt(adjustments['saturation'])}:"
            f"gamma={_fmt(adjustments['gamma'])}:"
            "gamma_weight=0.85"
        ),
    ]
    balance = dict(color_balance or {})
    if balance:
        allowed = ("rs", "gs", "bs", "rm", "gm", "bm", "rh", "gh", "bh")
        options = [f"{key}={_fmt(balance[key])}" for key in allowed if key in balance]
        options.append("pl=1")
        filters.append("colorbalance=" + ":".join(options))
    filters.append("format=yuv420p")
    return ",".join(filters)


def _white_balance_gains(analysis: ColorGradeAnalysis) -> tuple[float, float, float]:
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
    return tuple(float(_clamp(value, 0.86, 1.16)) for value in gains)


def _analysis_warnings(analysis: ColorGradeAnalysis, intensity: float) -> list[str]:
    warnings: list[str] = []
    if analysis.sample_count < 3:
        warnings.append("Only a few usable frames were available, so the grade is conservative.")
    if analysis.luma_p01 <= 0.015:
        warnings.append("The source has clipped or near-black shadows that grading cannot fully recover.")
    if analysis.luma_p99 >= 0.985:
        warnings.append("The source has clipped or near-white highlights that grading cannot fully recover.")
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
]
