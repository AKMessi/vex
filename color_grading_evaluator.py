from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MaskedPerceptualEvaluation:
    penalty: float
    reward: float
    breakdown: dict[str, float]


def evaluate_masked_perceptual_grade(
    before_frames: list[np.ndarray],
    after_frames: list[np.ndarray],
    *,
    need: float,
    source_quality: float,
) -> MaskedPerceptualEvaluation:
    frame_metrics: list[dict[str, float]] = []
    for before_frame, after_frame in zip(before_frames, after_frames):
        before = _normalize_frame(before_frame)
        after = _normalize_frame(after_frame)
        if before is None or after is None:
            continue
        before, after = _align_frames(before, after)
        frame_metrics.append(_frame_perceptual_metrics(before, after))

    if not frame_metrics:
        return MaskedPerceptualEvaluation(
            penalty=0.0,
            reward=0.0,
            breakdown={
                "perceptual_frames": 0.0,
                "perceptual_penalty": 0.0,
                "perceptual_reward": 0.0,
            },
        )

    averaged = _average_metrics(frame_metrics)
    bounded_need = _clamp(need, 0.0, 1.0)
    protected_source = source_quality > 0.84 and bounded_need < 0.38
    global_allowance = 0.020 + (0.135 * bounded_need)
    skin_allowance = 0.016 + (0.085 * bounded_need)
    neutral_allowance = 0.010 + (0.050 * bounded_need)
    edge_allowance = 0.020 + (0.095 * bounded_need)
    if protected_source:
        global_allowance *= 0.72
        skin_allowance *= 0.62
        neutral_allowance *= 0.70
        edge_allowance *= 0.72

    global_penalty = max(averaged["oklab_delta"] - global_allowance, 0.0) * 0.55
    skin_penalty = max(averaged["skin_delta"] - skin_allowance, 0.0) * (0.95 + averaged["skin_fraction"])
    neutral_penalty = max(averaged["neutral_drift"] - neutral_allowance, 0.0) * 0.90
    edge_penalty = max(averaged["edge_luma_delta"] - edge_allowance, 0.0) * 0.30
    shadow_penalty = max(averaged["shadow_crush"] - 0.012, 0.0) * 1.20
    highlight_penalty = max(averaged["highlight_blowout"] - 0.010, 0.0) * 1.15
    saturation_penalty = max(averaged["skin_saturation_gain"] - (0.020 + 0.055 * bounded_need), 0.0) * 0.75
    preservation_penalty = min(
        global_penalty
        + skin_penalty
        + neutral_penalty
        + edge_penalty
        + shadow_penalty
        + highlight_penalty
        + saturation_penalty,
        0.32,
    )

    rescue_reward = 0.0
    if bounded_need >= 0.45:
        rescue_reward = min(
            max(averaged["shadow_lift"] - averaged["shadow_crush"], 0.0) * 0.20
            + max(averaged["neutral_drift_reduction"], 0.0) * 0.25,
            0.045,
        )

    breakdown = {
        **averaged,
        "perceptual_frames": float(len(frame_metrics)),
        "perceptual_penalty": round(preservation_penalty, 5),
        "perceptual_reward": round(rescue_reward, 5),
        "global_delta_allowance": round(global_allowance, 5),
        "skin_delta_allowance": round(skin_allowance, 5),
        "neutral_delta_allowance": round(neutral_allowance, 5),
        "global_delta_penalty": round(global_penalty, 5),
        "skin_delta_penalty": round(skin_penalty, 5),
        "neutral_drift_penalty": round(neutral_penalty, 5),
        "edge_delta_penalty": round(edge_penalty, 5),
        "shadow_preservation_penalty": round(shadow_penalty, 5),
        "highlight_preservation_penalty": round(highlight_penalty, 5),
        "skin_saturation_penalty": round(saturation_penalty, 5),
    }
    return MaskedPerceptualEvaluation(
        penalty=round(preservation_penalty, 5),
        reward=round(rescue_reward, 5),
        breakdown=breakdown,
    )


def classify_source_from_analysis(analysis: Any) -> dict[str, float]:
    luma_span = float(getattr(analysis, "luma_span", 0.0))
    luma_median = float(getattr(analysis, "luma_median", 0.0))
    saturation = float(getattr(analysis, "saturation_mean", 0.0))
    black_clip = float(getattr(analysis, "black_clip_fraction", 0.0))
    white_clip = float(getattr(analysis, "white_clip_fraction", 0.0))
    neutral = float(getattr(analysis, "neutral_pixel_fraction", 0.0))
    skin = float(getattr(analysis, "skin_pixel_fraction", 0.0))
    quality = float(getattr(analysis, "frame_quality_mean", 0.0))

    already_graded = _clamp(
        (0.34 if 0.38 <= luma_median <= 0.60 else 0.0)
        + (0.28 if 0.55 <= luma_span <= 0.92 else 0.0)
        + (0.20 if 0.22 <= saturation <= 0.48 else 0.0)
        + (0.18 if quality >= 0.70 else 0.0)
        - min((black_clip + white_clip) * 1.8, 0.22),
        0.0,
        1.0,
    )
    flat_or_log = _clamp(
        (0.42 if luma_span < 0.42 else 0.0)
        + (0.30 if saturation < 0.24 else 0.0)
        + (0.16 if 0.28 <= luma_median <= 0.68 else 0.0)
        + (0.12 if black_clip + white_clip < 0.012 else 0.0),
        0.0,
        1.0,
    )
    low_light = _clamp(
        (0.48 if luma_median < 0.30 else 0.0)
        + min(black_clip / 0.09, 1.0) * 0.34
        + (0.18 if luma_span > 0.72 else 0.0),
        0.0,
        1.0,
    )
    screen_like = _clamp(
        (0.30 if neutral > 0.16 else 0.0)
        + (0.24 if saturation < 0.26 else 0.0)
        + (0.24 if luma_span > 0.72 else 0.0)
        + (0.22 if skin < 0.04 else 0.0),
        0.0,
        1.0,
    )
    return {
        "source_already_graded": round(already_graded, 5),
        "source_flat_or_log": round(flat_or_log, 5),
        "source_low_light": round(low_light, 5),
        "source_screen_like": round(screen_like, 5),
        "source_skin_sensitive": round(_clamp(skin / 0.18, 0.0, 1.0), 5),
    }


def _frame_perceptual_metrics(before: np.ndarray, after: np.ndarray) -> dict[str, float]:
    before_luma = _luma(before)
    after_luma = _luma(after)
    before_sat = _saturation(before)
    after_sat = _saturation(after)
    before_oklab = _rgb_to_oklab(before)
    after_oklab = _rgb_to_oklab(after)
    delta = np.linalg.norm(after_oklab - before_oklab, axis=2)
    masks = _masks(before, before_luma, before_sat)

    before_neutral_chroma = np.linalg.norm(before_oklab[..., 1:3], axis=2)
    after_neutral_chroma = np.linalg.norm(after_oklab[..., 1:3], axis=2)
    before_cast = _masked_mean(before_neutral_chroma, masks["neutral"])
    after_cast = _masked_mean(after_neutral_chroma, masks["neutral"])
    return {
        "oklab_delta": _masked_mean(delta, masks["valid"]),
        "skin_delta": _masked_mean(delta, masks["skin"]),
        "skin_fraction": float(np.mean(masks["skin"])),
        "neutral_drift": max(after_cast - before_cast, 0.0),
        "neutral_drift_reduction": max(before_cast - after_cast, 0.0),
        "shadow_crush": _masked_mean(np.maximum(before_luma - after_luma, 0.0), masks["shadow"]),
        "shadow_lift": _masked_mean(np.maximum(after_luma - before_luma, 0.0), masks["shadow"]),
        "highlight_blowout": _masked_mean(np.maximum(after_luma - before_luma, 0.0), masks["highlight"]),
        "edge_luma_delta": _masked_mean(np.abs(after_luma - before_luma), masks["edge"]),
        "skin_saturation_gain": _masked_mean(np.maximum(after_sat - before_sat, 0.0), masks["skin"]),
        "colorful_saturation_gain": _masked_mean(np.maximum(after_sat - before_sat, 0.0), masks["colorful"]),
    }


def _masks(rgb: np.ndarray, luma: np.ndarray, saturation: np.ndarray) -> dict[str, np.ndarray]:
    red = rgb[..., 0]
    green = rgb[..., 1]
    blue = rgb[..., 2]
    channel_spread = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    skin = (
        (luma >= 0.18)
        & (luma <= 0.88)
        & (saturation >= 0.10)
        & (saturation <= 0.72)
        & (red > green)
        & (green > blue * 0.70)
        & ((red - green) >= 0.014)
        & ((red - green) <= 0.32)
        & ((red - blue) >= 0.040)
    )
    neutral = (
        (luma >= 0.14)
        & (luma <= 0.88)
        & (saturation <= 0.22)
        & (channel_spread <= 0.12)
        & ~skin
    )
    edge = _edge_mask(luma)
    return {
        "valid": np.ones(luma.shape, dtype=bool),
        "skin": skin,
        "neutral": neutral,
        "shadow": luma <= 0.18,
        "highlight": luma >= 0.82,
        "colorful": saturation >= 0.45,
        "edge": edge & (luma >= 0.08) & (luma <= 0.94),
    }


def _edge_mask(luma: np.ndarray) -> np.ndarray:
    horizontal = np.zeros_like(luma)
    vertical = np.zeros_like(luma)
    horizontal[:, 1:] = np.abs(luma[:, 1:] - luma[:, :-1])
    vertical[1:, :] = np.abs(luma[1:, :] - luma[:-1, :])
    gradient = np.maximum(horizontal, vertical)
    threshold = max(float(np.percentile(gradient, 88)), 0.035)
    return gradient >= threshold


def _rgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    linear = _srgb_to_linear(rgb)
    red = linear[..., 0]
    green = linear[..., 1]
    blue = linear[..., 2]
    lms_l = 0.4122214708 * red + 0.5363325363 * green + 0.0514459929 * blue
    lms_m = 0.2119034982 * red + 0.6806995451 * green + 0.1073969566 * blue
    lms_s = 0.0883024619 * red + 0.2817188376 * green + 0.6299787005 * blue
    l_ = np.cbrt(np.maximum(lms_l, 0.0))
    m_ = np.cbrt(np.maximum(lms_m, 0.0))
    s_ = np.cbrt(np.maximum(lms_s, 0.0))
    return np.stack(
        [
            0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
            1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
            0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
        ],
        axis=2,
    )


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _average_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for item in items for key in item})
    return {
        key: round(float(np.mean([item.get(key, 0.0) for item in items])), 5)
        for key in keys
    }


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    if values.size == 0 or mask.size == 0 or not bool(np.any(mask)):
        return 0.0
    return float(np.mean(values[mask]))


def _normalize_frame(frame: np.ndarray) -> np.ndarray | None:
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] < 3 or array.shape[0] <= 0 or array.shape[1] <= 0:
        return None
    rgb = array[..., :3].astype(np.float32, copy=False)
    if float(np.nanmax(rgb)) > 1.5:
        rgb = rgb / 255.0
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(rgb, 0.0, 1.0)


def _align_frames(before: np.ndarray, after: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height = min(before.shape[0], after.shape[0])
    width = min(before.shape[1], after.shape[1])
    return before[:height, :width, :3], after[:height, :width, :3]


def _luma(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def _saturation(rgb: np.ndarray) -> np.ndarray:
    maximum = np.max(rgb, axis=2)
    minimum = np.min(rgb, axis=2)
    return np.where(maximum > 0.05, (maximum - minimum) / np.maximum(maximum, 1e-6), 0.0)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(float(value), maximum))


__all__ = [
    "MaskedPerceptualEvaluation",
    "classify_source_from_analysis",
    "evaluate_masked_perceptual_grade",
]
