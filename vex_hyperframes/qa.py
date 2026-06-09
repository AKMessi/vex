from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np

import config


@dataclass
class HyperframesFrameStats:
    path: str
    contrast: float
    occupancy: float
    edge_occupancy: float
    dead_space: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HyperframesQualityReport:
    passed: bool
    score: float
    issues: list[str] = field(default_factory=list)
    frame_stats: list[HyperframesFrameStats] = field(default_factory=list)
    mean_contrast: float = 0.0
    mean_occupancy: float = 0.0
    mean_edge_occupancy: float = 0.0
    mean_dead_space: float = 0.0
    motion_delta: float = 0.0
    vision_score: float | None = None
    vision_notes: str = ""
    semantic_score: float | None = None
    semantic_passed: bool | None = None
    semantic_issues: list[str] = field(default_factory=list)
    animation_score: float | None = None
    repair_action: str = "keep"
    reroute_renderer: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "score": self.score,
            "issues": list(self.issues),
            "frame_stats": [item.to_dict() for item in self.frame_stats],
            "mean_contrast": self.mean_contrast,
            "mean_occupancy": self.mean_occupancy,
            "mean_edge_occupancy": self.mean_edge_occupancy,
            "mean_dead_space": self.mean_dead_space,
            "motion_delta": self.motion_delta,
            "vision_score": self.vision_score,
            "vision_notes": self.vision_notes,
            "semantic_score": self.semantic_score,
            "semantic_passed": self.semantic_passed,
            "semantic_issues": list(self.semantic_issues),
            "animation_score": self.animation_score,
            "repair_action": self.repair_action,
            "reroute_renderer": self.reroute_renderer,
        }


def extract_quality_frames(
    video_path: str | Path,
    output_dir: Path,
    *,
    duration_sec: float,
    frame_count: int = 3,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if duration_sec <= 0:
        return []
    fractions = [0.18, 0.52, 0.82, 0.94][: max(1, min(frame_count, 4))]
    frame_paths: list[Path] = []
    for index, fraction in enumerate(fractions, start=1):
        target = output_dir / f"qa_frame_{index:02d}.png"
        command = [
            config.FFMPEG_PATH,
            "-ss",
            f"{max(duration_sec * fraction, 0.0):.3f}",
            "-i",
            str(video_path),
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


def _hex_to_rgb(value: str) -> np.ndarray:
    cleaned = str(value or "#000000").strip().lstrip("#")
    if len(cleaned) != 6:
        cleaned = "000000"
    return np.array([int(cleaned[index : index + 2], 16) / 255.0 for index in (0, 2, 4)], dtype=np.float32)


def _frame_stats(frame_path: Path, background_rgb: np.ndarray) -> HyperframesFrameStats:
    image = iio.imread(frame_path).astype(np.float32) / 255.0
    rgb = image[..., :3]
    luminance = rgb.mean(axis=2)
    distance = np.abs(rgb - background_rgb.reshape(1, 1, 3)).mean(axis=2)
    occupancy_mask = distance > 0.075
    foreground_mask = distance > 0.18
    height, width = occupancy_mask.shape
    edge_band_x = max(1, int(width * 0.06))
    edge_band_y = max(1, int(height * 0.08))
    edge_mask = np.zeros_like(occupancy_mask, dtype=bool)
    edge_mask[:, :edge_band_x] = True
    edge_mask[:, -edge_band_x:] = True
    edge_mask[:edge_band_y, :] = True
    edge_mask[-edge_band_y:, :] = True
    occupancy = float(np.mean(occupancy_mask))
    edge_occupancy = float(np.mean(foreground_mask[edge_mask]))
    dead_space = float(np.mean(distance < 0.045))
    return HyperframesFrameStats(
        path=str(frame_path),
        contrast=round(float(np.std(luminance) * 255.0), 3),
        occupancy=round(occupancy, 4),
        edge_occupancy=round(edge_occupancy, 4),
        dead_space=round(dead_space, 4),
    )


def _motion_delta(frame_paths: list[Path]) -> float:
    if len(frame_paths) < 2:
        return 0.0
    frames = [iio.imread(path).astype(np.float32) / 255.0 for path in frame_paths if path.is_file()]
    deltas: list[float] = []
    for first, second in zip(frames, frames[1:]):
        deltas.append(float(np.mean(np.abs(first[..., :3] - second[..., :3]))))
    return round(sum(deltas) / max(len(deltas), 1), 5)


def _text_overflow_risk(html: str) -> list[str]:
    issues: list[str] = []
    visible_html = re.sub(
        r"<head\b[^>]*>.*?</head>",
        "",
        html,
        flags=re.DOTALL | re.IGNORECASE,
    )
    visible_html = re.sub(
        r"<(?:script|style)\b[^>]*>.*?</(?:script|style)>",
        "",
        visible_html,
        flags=re.DOTALL | re.IGNORECASE,
    )
    text_nodes = re.findall(r">(.*?)<", visible_html, flags=re.DOTALL)
    visible = [
        re.sub(r"\s+", " ", item).strip()
        for item in text_nodes
        if re.sub(r"\s+", " ", item).strip() and not item.strip().startswith(("{", "const ", "function "))
    ]
    longest_word = max((len(word) for text in visible for word in re.findall(r"\S+", text)), default=0)
    total_words = sum(len(re.findall(r"[A-Za-z0-9%+.-]+", text)) for text in visible)
    if longest_word > 26:
        issues.append("A visible text token is very long and may overflow in compressed renders.")
    if total_words > 42:
        issues.append("The slide carries too much visible copy for a premium motion insert.")
    return issues


def analyze_hyperframes_quality(
    *,
    video_path: str | Path,
    html: str,
    frame_paths: list[Path],
    theme: dict[str, str],
    design_ir: dict[str, Any],
    min_score: float,
    vision_report: dict[str, Any] | None = None,
    semantic_report: dict[str, Any] | None = None,
) -> HyperframesQualityReport:
    background_rgb = _hex_to_rgb(theme.get("background", "#000000"))
    stats = [_frame_stats(path, background_rgb) for path in frame_paths if path.is_file()]
    mean_contrast = round(sum(item.contrast for item in stats) / max(len(stats), 1), 3)
    mean_occupancy = round(sum(item.occupancy for item in stats) / max(len(stats), 1), 4)
    mean_edge_occupancy = round(sum(item.edge_occupancy for item in stats) / max(len(stats), 1), 4)
    mean_dead_space = round(sum(item.dead_space for item in stats) / max(len(stats), 1), 4)
    motion_delta = _motion_delta(frame_paths)
    issues = _text_overflow_risk(html)
    motion_intensity = str(design_ir.get("motion_intensity") or "medium")
    if not stats:
        issues.append("No preview frames could be extracted for visual QA.")
    if mean_contrast < 19.0:
        issues.append("The render is too low-contrast for a premium explanatory slide.")
    if mean_occupancy < 0.08:
        issues.append("The composition is too sparse and does not use enough of the frame.")
    if mean_dead_space > 0.82:
        issues.append("The frame has excessive dead space relative to authored visual structure.")
    if mean_edge_occupancy > 0.22:
        issues.append("Important visual content appears too close to frame edges.")
    if motion_intensity in {"medium", "high"} and motion_delta < 0.012:
        issues.append("The render is too static for the selected motion intensity.")
    vision_score = None
    vision_notes = ""
    if vision_report:
        try:
            vision_score = float(vision_report.get("score"))
        except (TypeError, ValueError):
            vision_score = None
        vision_notes = str(vision_report.get("notes") or "")
        if vision_score is not None and vision_score < 0.68:
            issues.append("Vision critique judged the visual design below the premium threshold.")
    semantic_score = None
    semantic_passed = None
    semantic_issues: list[str] = []
    animation_score = None
    repair_action = "keep"
    reroute_renderer = ""
    if semantic_report:
        try:
            semantic_score = float(semantic_report.get("score"))
        except (TypeError, ValueError):
            semantic_score = None
        semantic_passed = bool(semantic_report.get("passed"))
        semantic_issues = [
            str(item)
            for item in [
                *list(semantic_report.get("hard_failures") or []),
                *list(semantic_report.get("issues") or []),
            ]
            if str(item).strip()
        ]
        animation = dict(semantic_report.get("animation") or {})
        try:
            animation_score = float(animation.get("score"))
        except (TypeError, ValueError):
            animation_score = None
        repair_action = str(semantic_report.get("repair_action") or "keep")
        reroute_renderer = str(semantic_report.get("reroute_renderer") or "")
        if semantic_passed is False:
            issues.append("Semantic QA rejected the explanatory contract.")
    score = 1.0
    score -= min(len(issues) * 0.12, 0.72)
    score += min(mean_contrast / 120.0, 0.18)
    score += min(mean_occupancy, 0.24) * 0.3
    score += min(motion_delta * 2.4, 0.16)
    if vision_score is not None:
        score = (score * 0.72) + (vision_score * 0.28)
    if semantic_score is not None:
        score = min(score, (score * 0.42) + (semantic_score * 0.58))
    score = round(max(0.0, min(score, 1.0)), 3)
    return HyperframesQualityReport(
        passed=(
            score >= min_score
            and semantic_passed is not False
            and not any("overflow" in issue.lower() for issue in issues)
        ),
        score=score,
        issues=issues,
        frame_stats=stats,
        mean_contrast=mean_contrast,
        mean_occupancy=mean_occupancy,
        mean_edge_occupancy=mean_edge_occupancy,
        mean_dead_space=mean_dead_space,
        motion_delta=motion_delta,
        vision_score=vision_score,
        vision_notes=vision_notes,
        semantic_score=semantic_score,
        semantic_passed=semantic_passed,
        semantic_issues=semantic_issues,
        animation_score=animation_score,
        repair_action=repair_action,
        reroute_renderer=reroute_renderer,
    )


def write_quality_report(path: Path, report: HyperframesQualityReport) -> None:
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
