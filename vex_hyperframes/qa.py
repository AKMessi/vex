from __future__ import annotations

import hashlib
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
    capture_plan: list[dict[str, Any]] | None = None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if duration_sec <= 0:
        return []
    captures = [
        dict(item)
        for item in capture_plan or []
        if isinstance(item, dict)
    ]
    if captures:
        fractions = [
            (
                str(item.get("capture_id") or f"capture_{index + 1:02d}"),
                max(0.0, min(float(item.get("fraction") or 0.0), 1.0)),
            )
            for index, item in enumerate(captures)
        ]
    else:
        fractions = [
            (f"capture_{index + 1:02d}", fraction)
            for index, fraction in enumerate(
                [0.18, 0.52, 0.82, 0.94][: max(1, min(frame_count, 4))]
            )
        ]
    frame_paths: list[Path] = []
    for index, (capture_id, fraction) in enumerate(fractions, start=1):
        safe_capture_id = re.sub(
            r"[^a-zA-Z0-9_-]+",
            "_",
            capture_id,
        ).strip("_") or f"capture_{index:02d}"
        target = output_dir / f"qa_frame_{index:02d}_{safe_capture_id}.png"
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
        try:
            result = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
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


def build_rendered_visual_fingerprint(
    frame_paths: list[Path],
    *,
    visual_world_program: dict[str, Any] | None = None,
) -> dict[str, Any]:
    samples: list[np.ndarray] = []
    for path in frame_paths:
        target = Path(path)
        if not target.is_file():
            continue
        image = iio.imread(target).astype(np.float32) / 255.0
        rgb = image[..., :3]
        step_y = max(1, rgb.shape[0] // 90)
        step_x = max(1, rgb.shape[1] // 160)
        samples.append(rgb[::step_y, ::step_x])
    if not samples:
        return {
            "available": False,
            "signature": "",
            "medium_family": str(
                (visual_world_program or {}).get("medium_family") or ""
            ),
            "background_mode": str(
                (visual_world_program or {}).get("background_mode") or ""
            ),
        }
    pixels = np.concatenate(
        [item.reshape(-1, 3) for item in samples],
        axis=0,
    )
    mean_rgb = np.mean(pixels, axis=0)
    maximum = np.max(pixels, axis=1)
    minimum = np.min(pixels, axis=1)
    saturation = np.mean(maximum - minimum)
    luminance = (
        pixels[:, 0] * 0.2126
        + pixels[:, 1] * 0.7152
        + pixels[:, 2] * 0.0722
    )
    edge_values: list[float] = []
    for image in samples:
        luma = (
            image[..., 0] * 0.2126
            + image[..., 1] * 0.7152
            + image[..., 2] * 0.0722
        )
        edge_values.extend(np.abs(np.diff(luma, axis=0)).reshape(-1).tolist())
        edge_values.extend(np.abs(np.diff(luma, axis=1)).reshape(-1).tolist())
    histogram: list[float] = []
    for channel in range(3):
        counts, _ = np.histogram(
            pixels[:, channel],
            bins=4,
            range=(0.0, 1.0),
        )
        normalized = counts.astype(np.float64) / max(float(counts.sum()), 1.0)
        histogram.extend(round(float(value), 4) for value in normalized)
    world = dict(visual_world_program or {})
    payload = {
        "available": True,
        "medium_family": str(world.get("medium_family") or ""),
        "canvas_system": str(world.get("canvas_system") or ""),
        "background_mode": str(world.get("background_mode") or ""),
        "motion_choreography": str(world.get("motion_choreography") or ""),
        "world_signature": str(world.get("world_signature") or ""),
        "mean_rgb": [round(float(value), 4) for value in mean_rgb],
        "mean_luminance": round(float(np.mean(luminance)), 4),
        "luminance_contrast": round(float(np.std(luminance)), 4),
        "mean_saturation": round(float(saturation), 4),
        "edge_density": round(
            float(np.mean(edge_values)) if edge_values else 0.0,
            4,
        ),
        "color_histogram": histogram,
    }
    signature_payload = {
        **payload,
        "mean_rgb": [round(value, 2) for value in payload["mean_rgb"]],
        "mean_luminance": round(payload["mean_luminance"], 2),
        "luminance_contrast": round(payload["luminance_contrast"], 2),
        "mean_saturation": round(payload["mean_saturation"], 2),
        "edge_density": round(payload["edge_density"], 2),
        "color_histogram": [
            round(value, 2) for value in payload["color_histogram"]
        ],
    }
    payload["signature"] = hashlib.sha256(
        json.dumps(
            signature_payload,
            sort_keys=True,
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    return payload


def visual_fingerprint_distance(
    left: dict[str, Any],
    right: dict[str, Any],
) -> float:
    if not left.get("available") or not right.get("available"):
        return 1.0
    left_histogram = np.array(
        left.get("color_histogram") or [],
        dtype=np.float32,
    )
    right_histogram = np.array(
        right.get("color_histogram") or [],
        dtype=np.float32,
    )
    if (
        left_histogram.size == 0
        or left_histogram.shape != right_histogram.shape
    ):
        return 1.0
    histogram_distance = float(
        np.mean(np.abs(left_histogram - right_histogram))
    )
    left_rgb = np.array(left.get("mean_rgb") or [0.0, 0.0, 0.0])
    right_rgb = np.array(right.get("mean_rgb") or [0.0, 0.0, 0.0])
    rgb_distance = float(np.mean(np.abs(left_rgb - right_rgb)))
    scalar_distance = sum(
        abs(float(left.get(key) or 0.0) - float(right.get(key) or 0.0))
        for key in (
            "mean_luminance",
            "luminance_contrast",
            "mean_saturation",
            "edge_density",
        )
    ) / 4.0
    return round(
        min(
            1.0,
            histogram_distance * 1.8
            + rgb_distance * 0.75
            + scalar_distance * 0.9,
        ),
        4,
    )


def _text_overflow_risk(html: str, *, max_words: int = 42) -> list[str]:
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
    if total_words > max_words:
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
    issues = _text_overflow_risk(
        html,
        max_words=64 if semantic_report else 42,
    )
    motion_intensity = str(design_ir.get("motion_intensity") or "medium")
    semantic_animation_passed = bool(
        (
            (semantic_report or {}).get("animation")
            or {}
        ).get("passed")
    )
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
    if (
        motion_intensity in {"medium", "high"}
        and motion_delta < 0.012
        and not semantic_animation_passed
    ):
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
