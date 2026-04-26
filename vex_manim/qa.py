from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np

import config
from vex_manim.briefs import SceneBrief
from vex_manim.validator import ValidationReport


def _hex_to_rgb(value: str) -> np.ndarray:
    cleaned = str(value or "#000000").strip().lstrip("#")
    if len(cleaned) != 6:
        cleaned = "000000"
    return np.array([int(cleaned[index : index + 2], 16) / 255.0 for index in (0, 2, 4)], dtype=np.float32)


@dataclass
class PreviewFrameStats:
    path: str
    contrast: float
    occupancy: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PreviewReport:
    preview_video_path: str
    duration_sec: float
    frame_stats: list[PreviewFrameStats] = field(default_factory=list)
    mean_contrast: float = 0.0
    mean_occupancy: float = 0.0
    motion_delta: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "preview_video_path": self.preview_video_path,
            "duration_sec": self.duration_sec,
            "frame_stats": [item.to_dict() for item in self.frame_stats],
            "mean_contrast": self.mean_contrast,
            "mean_occupancy": self.mean_occupancy,
            "motion_delta": self.motion_delta,
        }


@dataclass
class QualityReport:
    passed: bool
    score: float
    issues: list[str]
    preview: PreviewReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "score": self.score,
            "issues": list(self.issues),
            "preview": self.preview.to_dict(),
        }

    def feedback_lines(self) -> list[str]:
        return list(self.issues)


def extract_preview_frames(
    preview_video_path: str,
    output_dir: Path,
    *,
    duration_sec: float,
    frame_count: int = 3,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if duration_sec <= 0:
        return []
    fractions = [0.2, 0.55, 0.85][:frame_count]
    frame_paths: list[Path] = []
    for index, fraction in enumerate(fractions, start=1):
        timestamp = max(0.0, duration_sec * fraction)
        target = output_dir / f"frame_{index:02d}.png"
        command = [
            config.FFMPEG_PATH,
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            preview_video_path,
            "-frames:v",
            "1",
            "-update",
            "1",
            "-y",
            str(target),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0 and target.is_file():
            frame_paths.append(target)
    return frame_paths


def _frame_stats(frame_path: Path, background_rgb: np.ndarray) -> PreviewFrameStats:
    image = iio.imread(frame_path).astype(np.float32) / 255.0
    rgb = image[..., :3]
    luminance = rgb.mean(axis=2)
    contrast = float(np.std(luminance) * 255.0)
    distance = np.abs(rgb - background_rgb.reshape(1, 1, 3)).mean(axis=2)
    occupancy = float(np.mean(distance > 0.08))
    return PreviewFrameStats(path=str(frame_path), contrast=round(contrast, 3), occupancy=round(occupancy, 4))


def analyze_preview(
    preview_video_path: str,
    preview_duration_sec: float,
    frame_paths: list[Path],
    *,
    theme: dict[str, str],
) -> PreviewReport:
    background_rgb = _hex_to_rgb(theme.get("background", "#000000"))
    stats = [_frame_stats(path, background_rgb) for path in frame_paths if path.is_file()]
    motion_delta = 0.0
    if len(frame_paths) >= 2:
        frames = [iio.imread(path).astype(np.float32) / 255.0 for path in frame_paths if path.is_file()]
        deltas: list[float] = []
        for first, second in zip(frames, frames[1:]):
            deltas.append(float(np.mean(np.abs(first[..., :3] - second[..., :3]))))
        motion_delta = round(sum(deltas) / max(len(deltas), 1), 5)
    return PreviewReport(
        preview_video_path=preview_video_path,
        duration_sec=round(preview_duration_sec, 3),
        frame_stats=stats,
        mean_contrast=round(sum(item.contrast for item in stats) / max(len(stats), 1), 3),
        mean_occupancy=round(sum(item.occupancy for item in stats) / max(len(stats), 1), 4),
        motion_delta=motion_delta,
    )


def evaluate_generated_scene_quality(
    brief: SceneBrief,
    validation: ValidationReport,
    preview: PreviewReport,
) -> QualityReport:
    issues: list[str] = []
    target_duration = float(brief.render_constraints.get("target_duration_sec") or brief.duration_sec or 0.0)
    duration_delta = abs(preview.duration_sec - target_duration)
    if duration_delta > 1.1:
        issues.append(
            f"Preview duration drifted from the target by {duration_delta:.2f}s; tighten the animation pacing."
        )
    if preview.mean_contrast < 18.0:
        issues.append("The preview frames look too low-contrast; add stronger hierarchy, lighting, or contrast.")
    if preview.mean_occupancy < 0.035:
        issues.append("The preview looks too sparse; use more of the frame and add visual structure.")
    if brief.animation_intensity in {"medium", "high"} and preview.motion_delta < 0.018:
        issues.append("The scene is too static for the requested intensity; add transforms, camera motion, or evolving geometry.")
    profile = validation.profile
    if brief.scene_family in {"system_map", "timeline_journey", "comparison_morph"} and "MovingCameraScene" not in profile.advanced_features:
        issues.append("This scene family should leverage camera choreography; add guided reframing or punch-ins.")
    if brief.scene_family in {"metric_story", "dashboard_build"} and not any(
        feature in profile.advanced_features for feature in {"ValueTracker", "Axes", "BarChart", "TransformMatchingShapes"}
    ):
        issues.append("The data scene does not leverage Manim's dynamic strengths; use trackers, charts, or morphs.")
    if len(profile.advanced_features) == 0 and len(profile.primitive_features) >= 3:
        issues.append("The composition still reads like boxes-and-text; use richer Manim features to avoid a generic look.")
    score = 1.0
    score -= min(len(issues) * 0.16, 0.8)
    score += min(len(profile.advanced_features), 5) * 0.03
    score += min(preview.mean_occupancy, 0.18) * 0.35
    score = round(max(0.0, min(score, 1.0)), 3)
    return QualityReport(passed=not issues, score=score, issues=issues, preview=preview)
