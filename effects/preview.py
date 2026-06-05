from __future__ import annotations

import math
import subprocess
from typing import Any

import config
from effects.schema import EffectPlan


PREVIEW_WIDTH = 48
PREVIEW_HEIGHT = 27


def validate_effect_preview(
    source_path: str,
    output_path: str,
    plan: EffectPlan,
    *,
    max_samples: int = 12,
) -> dict[str, Any]:
    sample_times = _sample_times(plan, max_samples=max_samples)
    if not sample_times:
        return {
            "passed": True,
            "errors": [],
            "warnings": ["No effect preview samples were requested."],
            "samples": [],
            "sample_count": 0,
        }
    errors: list[str] = []
    warnings: list[str] = []
    samples: list[dict[str, Any]] = []
    missing = 0
    for time_sec in sample_times:
        source_raw = _extract_tiny_frame(source_path, time_sec)
        output_raw = _extract_tiny_frame(output_path, time_sec)
        if source_raw is None or output_raw is None:
            missing += 1
            continue
        source_stats = _frame_stats(source_raw)
        output_stats = _frame_stats(output_raw)
        luma_delta = abs(output_stats["brightness"] - source_stats["brightness"])
        contrast_delta = abs(output_stats["contrast"] - source_stats["contrast"])
        sample = {
            "time_sec": round(time_sec, 3),
            "source": source_stats,
            "output": output_stats,
            "brightness_delta": round(luma_delta, 4),
            "contrast_delta": round(contrast_delta, 4),
        }
        samples.append(sample)
        if output_stats["brightness"] < 0.025 and output_stats["contrast"] < 0.01:
            errors.append(f"Preview frame at {time_sec:.2f}s is effectively black.")
        if output_stats["brightness"] > 0.96 and output_stats["contrast"] < 0.018:
            errors.append(f"Preview frame at {time_sec:.2f}s is effectively white.")
        if luma_delta > 0.34:
            warnings.append(f"Preview brightness changes sharply at {time_sec:.2f}s.")
        if contrast_delta > 0.28:
            warnings.append(f"Preview contrast changes sharply at {time_sec:.2f}s.")
    if missing == len(sample_times):
        errors.append("Could not sample any rendered effect preview frames.")
    elif missing:
        warnings.append(f"Could not sample {missing} effect preview frame{'s' if missing != 1 else ''}.")
    score = 1.0 - len(errors) * 0.28 - len(warnings) * 0.045
    return {
        "passed": not errors,
        "score": round(max(0.0, min(score, 1.0)), 4),
        "errors": errors,
        "warnings": warnings[:12],
        "samples": samples,
        "sample_count": len(samples),
        "missing_sample_count": missing,
        "preview_size": [PREVIEW_WIDTH, PREVIEW_HEIGHT],
    }


def _sample_times(plan: EffectPlan, *, max_samples: int) -> list[float]:
    times: list[float] = []
    for effect in plan.effects:
        start = _as_float(effect.start, 0.0)
        end = _as_float(effect.end, start)
        if end <= start:
            continue
        center = (start + end) / 2.0
        times.append(center)
        if effect.duration >= 1.1:
            times.append(start + (end - start) * 0.24)
    unique: list[float] = []
    for time_sec in sorted(times):
        if not unique or abs(time_sec - unique[-1]) > 0.18:
            unique.append(time_sec)
        if len(unique) >= max(1, max_samples):
            break
    return unique


def _extract_tiny_frame(video_path: str, time_sec: float) -> bytes | None:
    command = [
        config.FFMPEG_PATH,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(float(time_sec), 0.0):.3f}",
        "-i",
        video_path,
        "-frames:v",
        "1",
        "-vf",
        (
            f"scale={PREVIEW_WIDTH}:{PREVIEW_HEIGHT}:"
            "force_original_aspect_ratio=decrease,"
            f"pad={PREVIEW_WIDTH}:{PREVIEW_HEIGHT}:"
            "(ow-iw)/2:(oh-ih)/2:color=black,format=rgb24"
        ),
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    try:
        result = subprocess.run(command, capture_output=True, timeout=8, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    expected = PREVIEW_WIDTH * PREVIEW_HEIGHT * 3
    if result.returncode != 0 or len(result.stdout) < expected:
        return None
    return result.stdout[:expected]


def _frame_stats(raw: bytes) -> dict[str, float]:
    pixels = [
        raw[index : index + 3]
        for index in range(0, len(raw), 3)
        if index + 2 < len(raw)
    ]
    if not pixels:
        return {"brightness": 0.0, "contrast": 0.0, "colorfulness": 0.0}
    lumas = [
        (0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2]) / 255.0
        for pixel in pixels
    ]
    brightness = sum(lumas) / len(lumas)
    variance = sum((value - brightness) ** 2 for value in lumas) / len(lumas)
    spreads = [
        (max(pixel[0], pixel[1], pixel[2]) - min(pixel[0], pixel[1], pixel[2])) / 255.0
        for pixel in pixels
    ]
    return {
        "brightness": round(brightness, 4),
        "contrast": round(math.sqrt(max(variance, 0.0)), 4),
        "colorfulness": round(sum(spreads) / len(spreads), 4),
    }


def _as_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number
