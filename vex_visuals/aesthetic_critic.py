from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from vex_visuals.creative_direction import validate_creative_direction


AESTHETIC_CRITIC_VERSION = "vex-aesthetic-critic-v1"


@dataclass(frozen=True)
class AestheticCriticReport:
    version: str
    passed: bool
    score: float
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_frame_aesthetics(
    frame_paths: list[Path | str],
    creative_direction: dict[str, Any],
) -> AestheticCriticReport:
    issues: list[str] = []
    warnings: list[str] = []
    validation = validate_creative_direction(creative_direction)
    if not validation.passed:
        issues.extend(validation.errors)
    paths = [Path(item) for item in frame_paths if Path(item).is_file()]
    if len(paths) < 2:
        issues.append("aesthetic_critic_requires_multiple_frames")
        return AestheticCriticReport(
            version=AESTHETIC_CRITIC_VERSION,
            passed=False,
            score=0.0,
            issues=issues,
            metrics={"frame_count": len(paths)},
        )
    frames = [_load(path) for path in paths]
    final = frames[-1]
    saliency = _saliency(final)
    occupied = saliency > max(0.08, float(np.percentile(saliency, 58)))
    occupancy = float(np.mean(occupied))
    centroid = _centroid(saliency)
    composition = dict(creative_direction.get("composition") or {})
    contract = dict(creative_direction.get("quality_contract") or {})
    target = composition.get("focal_point") or [0.5, 0.5]
    balance_distance = float(np.hypot(centroid[0] - float(target[0]), centroid[1] - float(target[1])))
    balance_score = max(0.0, 1.0 - balance_distance / 0.72)
    edge_intrusion = _edge_intrusion(occupied)
    hierarchy_score = _hierarchy(saliency)
    depth_score = _depth(final)
    palette_vitality = _palette_vitality(final)
    color_count = _dominant_color_count(final)
    motion = [_motion(left, right) for left, right in zip(frames, frames[1:])]
    maximum_motion = max(motion or [0.0])
    terminal_motion = motion[-1] if motion else 0.0
    negative_space = 1.0 - occupancy
    target_space = float(contract.get("negative_space_target") or composition.get("negative_space_target") or 0.32)
    tolerance = max(float(contract.get("negative_space_tolerance") or 0.2), 0.05)
    whitespace_score = max(0.0, 1.0 - abs(negative_space - target_space) / tolerance)
    if balance_score < float(contract.get("minimum_balance_score") or 0.48):
        warnings.append("aesthetic_composition_is_visually_unbalanced")
    if hierarchy_score < float(contract.get("minimum_hierarchy_score") or 0.5):
        issues.append("aesthetic_visual_hierarchy_is_too_flat")
    if edge_intrusion > float(contract.get("maximum_edge_intrusion") or 0.22):
        issues.append("aesthetic_content_intrudes_into_frame_edges")
    if depth_score < float(contract.get("minimum_depth_score") or 0.28):
        warnings.append("aesthetic_scene_has_limited_visual_depth")
    if palette_vitality < float(contract.get("minimum_palette_vitality") or 0.12):
        warnings.append("aesthetic_palette_lacks_vitality")
    if maximum_motion > float(contract.get("maximum_global_motion") or 0.42):
        issues.append("aesthetic_motion_changes_too_much_of_the_frame_at_once")
    if terminal_motion > max(0.04, maximum_motion * 0.72):
        warnings.append("aesthetic_final_state_has_not_visually_settled")
    if color_count < 3:
        warnings.append("aesthetic_palette_has_too_few_dominant_color_roles")
    score = (
        balance_score * 0.19
        + hierarchy_score * 0.22
        + max(0.0, 1.0 - edge_intrusion / 0.28) * 0.16
        + depth_score * 0.13
        + min(palette_vitality / 0.32, 1.0) * 0.12
        + whitespace_score * 0.1
        + max(0.0, 1.0 - terminal_motion / max(maximum_motion, 0.02)) * 0.08
    )
    if score < 0.5:
        issues.append("aesthetic_quality_score_below_floor")
    metrics = {
        "frame_count": len(frames),
        "occupancy": round(occupancy, 4),
        "negative_space": round(negative_space, 4),
        "negative_space_target": round(target_space, 4),
        "saliency_centroid": [round(item, 4) for item in centroid],
        "focal_point_target": [round(float(item), 4) for item in target],
        "balance_score": round(balance_score, 4),
        "hierarchy_score": round(hierarchy_score, 4),
        "edge_intrusion": round(edge_intrusion, 4),
        "depth_score": round(depth_score, 4),
        "palette_vitality": round(palette_vitality, 4),
        "dominant_color_count": color_count,
        "motion_by_pair": [round(item, 4) for item in motion],
        "maximum_motion": round(maximum_motion, 4),
        "terminal_motion": round(terminal_motion, 4),
        "whitespace_score": round(whitespace_score, 4),
    }
    return AestheticCriticReport(
        version=AESTHETIC_CRITIC_VERSION,
        passed=not issues,
        score=round(max(0.0, min(score, 1.0)), 4),
        issues=issues,
        warnings=warnings,
        metrics=metrics,
    )


def _load(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB").resize((320, 180)), dtype=np.float32) / 255.0


def _saliency(frame: np.ndarray) -> np.ndarray:
    gray = frame[..., 0] * 0.2126 + frame[..., 1] * 0.7152 + frame[..., 2] * 0.0722
    gradient_x = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gradient_y = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    corners = np.concatenate(
        [frame[:10, :10].reshape(-1, 3), frame[:10, -10:].reshape(-1, 3), frame[-10:, :10].reshape(-1, 3), frame[-10:, -10:].reshape(-1, 3)]
    )
    background = np.median(corners, axis=0)
    color_distance = np.linalg.norm(frame - background, axis=2) / np.sqrt(3.0)
    return np.clip(color_distance * 0.58 + (gradient_x + gradient_y) * 0.42, 0.0, 1.0)


def _centroid(saliency: np.ndarray) -> tuple[float, float]:
    total = float(np.sum(saliency))
    if total <= 1e-6:
        return 0.5, 0.5
    height, width = saliency.shape
    yy, xx = np.mgrid[0:height, 0:width]
    return float(np.sum(xx * saliency) / total / max(width - 1, 1)), float(np.sum(yy * saliency) / total / max(height - 1, 1))


def _edge_intrusion(occupied: np.ndarray) -> float:
    height, width = occupied.shape
    edge_y = max(4, int(height * 0.045))
    edge_x = max(4, int(width * 0.045))
    edge = np.zeros_like(occupied, dtype=bool)
    outer_y = max(1, int(height * 0.015))
    outer_x = max(1, int(width * 0.015))
    edge[outer_y:edge_y, outer_x:-outer_x] = True
    edge[-edge_y:-outer_y, outer_x:-outer_x] = True
    edge[outer_y:-outer_y, outer_x:edge_x] = True
    edge[outer_y:-outer_y, -edge_x:-outer_x] = True
    occupied_count = max(int(np.count_nonzero(occupied)), 1)
    return float(np.count_nonzero(occupied & edge) / occupied_count)


def _hierarchy(saliency: np.ndarray) -> float:
    flat = saliency.reshape(-1)
    if not np.any(flat):
        return 0.0
    high = float(np.mean(flat[flat >= np.percentile(flat, 90)]))
    middle = float(np.mean(flat[(flat >= np.percentile(flat, 45)) & (flat < np.percentile(flat, 75))]))
    return max(0.0, min((high - middle) / max(high, 0.01) * 1.7, 1.0))


def _depth(frame: np.ndarray) -> float:
    gray = frame[..., 0] * 0.2126 + frame[..., 1] * 0.7152 + frame[..., 2] * 0.0722
    spread = float(np.percentile(gray, 95) - np.percentile(gray, 5))
    local = float(np.std(gray))
    return max(0.0, min(spread * 0.72 + local * 1.2, 1.0))


def _palette_vitality(frame: np.ndarray) -> float:
    red, green, blue = frame[..., 0], frame[..., 1], frame[..., 2]
    rg = red - green
    yb = 0.5 * (red + green) - blue
    return max(0.0, min(float(np.sqrt(np.var(rg) + np.var(yb)) + 0.3 * np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)), 1.0))


def _dominant_color_count(frame: np.ndarray) -> int:
    image = Image.fromarray(np.uint8(np.clip(frame * 255.0, 0, 255)), mode="RGB")
    colors = image.quantize(colors=8).getcolors() or []
    threshold = image.width * image.height * 0.025
    return sum(1 for count, _color in colors if count >= threshold)


def _motion(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.abs(left - right)))


__all__ = [
    "AESTHETIC_CRITIC_VERSION",
    "AestheticCriticReport",
    "evaluate_frame_aesthetics",
]
