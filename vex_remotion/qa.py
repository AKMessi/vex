from __future__ import annotations

import json
import math
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from vex_visuals.aesthetic_critic import evaluate_frame_aesthetics

import config


REMOTION_RENDER_QA_VERSION = "remotion-render-qa-v2"


@dataclass(frozen=True)
class RemotionRenderQA:
    version: str
    passed: bool
    score: float
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    frame_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_remotion_render(
    video_path: Path | str,
    program: dict[str, Any],
    *,
    job_dir: Path | str,
) -> RemotionRenderQA:
    video = Path(video_path)
    output_dir = Path(job_dir) / "remotion_qa_frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    duration = max(_as_float(program.get("duration_sec"), 3.0), 0.5)
    frame_paths: list[Path] = []
    extraction_errors: list[str] = []
    for index, fraction in enumerate((0.14, 0.5, 0.82), start=1):
        frame_path = output_dir / f"frame_{index:02d}_{int(fraction * 100):02d}.png"
        ok, reason = _extract_frame(video, frame_path, time_sec=duration * fraction)
        if ok:
            frame_paths.append(frame_path)
        else:
            extraction_errors.append(reason or f"frame_{index}_extraction_failed")

    issues: list[str] = []
    warnings: list[str] = []
    if len(frame_paths) < 3:
        issues.append("remotion_qa_could_not_extract_required_frames")
        warnings.extend(extraction_errors[:3])
        return RemotionRenderQA(
            version=REMOTION_RENDER_QA_VERSION,
            passed=False,
            score=0.0,
            issues=issues,
            warnings=warnings,
            metrics={"extracted_frame_count": len(frame_paths)},
            frame_paths=[str(path) for path in frame_paths],
        )

    frames = [_load_frame(path) for path in frame_paths]
    aesthetic_report = evaluate_frame_aesthetics(
        frame_paths,
        dict(program.get("creative_direction") or {}),
    )
    contrasts = [_contrast(frame) for frame in frames]
    occupancies = [_occupancy(frame) for frame in frames]
    entropies = [_entropy(frame) for frame in frames]
    motion_deltas = [_motion_delta(left, right) for left, right in zip(frames, frames[1:])]
    contract = dict(program.get("quality_contract") or {})
    min_occupancy = _as_float(contract.get("min_occupancy"), 0.06)
    max_occupancy = _as_float(contract.get("max_occupancy"), 0.86)
    min_contrast = _as_float(contract.get("min_contrast"), 0.075)
    min_motion = _as_float(contract.get("min_motion_delta"), 0.0035)
    final_occupancy = occupancies[-1]
    median_contrast = float(np.median(contrasts))
    max_motion = max(motion_deltas or [0.0])
    final_entropy = entropies[-1]
    if median_contrast < min_contrast:
        issues.append("remotion_render_has_insufficient_visual_contrast")
    if final_occupancy < min_occupancy:
        issues.append("remotion_render_final_frame_is_visually_empty")
    if final_occupancy > max_occupancy:
        issues.append("remotion_render_final_frame_is_overcrowded")
    if max_motion < min_motion:
        issues.append("remotion_render_has_no_meaningful_motion")
    if final_entropy < 0.1:
        issues.append("remotion_render_final_frame_has_low_information_density")
    elif final_entropy < 2.2:
        warnings.append("remotion_render_final_frame_has_low_information_density")
    if occupancies[0] > final_occupancy + 0.2:
        warnings.append("remotion_visual_hierarchy_regressed_during_reveal")

    semantic_score = max(0.0, min(_as_float(program.get("semantic_score"), 0.0), 1.0))
    contrast_score = max(0.0, min(median_contrast / max(min_contrast * 2.0, 0.001), 1.0))
    occupancy_center = (min_occupancy + max_occupancy) / 2.0
    occupancy_radius = max((max_occupancy - min_occupancy) / 2.0, 0.01)
    occupancy_score = max(0.0, 1.0 - abs(final_occupancy - occupancy_center) / occupancy_radius)
    motion_score = max(0.0, min(max_motion / max(min_motion * 4.0, 0.001), 1.0))
    entropy_score = max(0.0, min(final_entropy / 6.5, 1.0))
    score = (
        semantic_score * 0.32
        + contrast_score * 0.16
        + occupancy_score * 0.14
        + motion_score * 0.11
        + entropy_score * 0.07
        + aesthetic_report.score * 0.2
    )
    if not aesthetic_report.passed:
        issues.extend(aesthetic_report.issues)
    warnings.extend(aesthetic_report.warnings)
    if score < 0.58:
        issues.append("remotion_render_quality_score_below_publishable_floor")
    metrics = {
        "semantic_score": round(semantic_score, 4),
        "contrast_by_frame": [round(item, 4) for item in contrasts],
        "occupancy_by_frame": [round(item, 4) for item in occupancies],
        "entropy_by_frame": [round(item, 4) for item in entropies],
        "motion_delta_by_pair": [round(item, 4) for item in motion_deltas],
        "median_contrast": round(median_contrast, 4),
        "final_occupancy": round(final_occupancy, 4),
        "maximum_motion_delta": round(max_motion, 4),
        "final_entropy": round(final_entropy, 4),
        "contract": contract,
        "aesthetic_critic": aesthetic_report.to_dict(),
    }
    result = RemotionRenderQA(
        version=REMOTION_RENDER_QA_VERSION,
        passed=not issues,
        score=round(max(0.0, min(score, 1.0)), 4),
        issues=issues,
        warnings=warnings,
        metrics=metrics,
        frame_paths=[str(path) for path in frame_paths],
    )
    (Path(job_dir) / "remotion_qa.json").write_text(
        json.dumps(result.to_dict(), indent=2),
        encoding="utf-8",
    )
    return result


def _extract_frame(
    video_path: Path,
    output_path: Path,
    *,
    time_sec: float,
) -> tuple[bool, str]:
    command = [
        config.FFMPEG_PATH,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(time_sec, 0.0):.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-vf",
        "scale=480:-2:force_original_aspect_ratio=decrease",
        "-update",
        "1",
        "-y",
        str(output_path),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    return result.returncode == 0 and output_path.is_file(), (result.stderr or "").strip()


def _load_frame(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32)


def _luminance(frame: np.ndarray) -> np.ndarray:
    return frame[..., 0] * 0.2126 + frame[..., 1] * 0.7152 + frame[..., 2] * 0.0722


def _contrast(frame: np.ndarray) -> float:
    return float(np.std(_luminance(frame)) / 255.0)


def _occupancy(frame: np.ndarray) -> float:
    height, width, _ = frame.shape
    patch = max(2, min(height, width) // 30)
    corners = np.concatenate(
        [
            frame[:patch, :patch].reshape(-1, 3),
            frame[:patch, -patch:].reshape(-1, 3),
            frame[-patch:, :patch].reshape(-1, 3),
            frame[-patch:, -patch:].reshape(-1, 3),
        ],
        axis=0,
    )
    background = np.median(corners, axis=0)
    distance = np.sqrt(np.sum((frame - background) ** 2, axis=2))
    return float(np.mean(distance > 30.0))


def _motion_delta(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return 0.0
    return float(np.mean(np.abs(left - right)) / 255.0)


def _entropy(frame: np.ndarray) -> float:
    values = np.clip(_luminance(frame), 0, 255).astype(np.uint8)
    histogram = np.bincount(values.ravel(), minlength=256).astype(np.float64)
    probabilities = histogram / max(float(histogram.sum()), 1.0)
    probabilities = probabilities[probabilities > 0]
    return float(-np.sum(probabilities * np.log2(probabilities))) if len(probabilities) else 0.0


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default
