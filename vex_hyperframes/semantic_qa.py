from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np

from vex_hyperframes.evaluation import visible_text_from_html


@dataclass(frozen=True)
class AnimationInspection:
    passed: bool
    score: float
    transition_deltas: list[float] = field(default_factory=list)
    active_transition_count: int = 0
    final_hold_delta: float = 0.0
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["score"] = round(float(self.score), 4)
        return payload


@dataclass(frozen=True)
class HyperframesSemanticQaReport:
    passed: bool
    score: float
    hard_failures: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    missing_labels: list[str] = field(default_factory=list)
    missing_objects: list[str] = field(default_factory=list)
    label_coverage: float = 0.0
    object_coverage: float = 0.0
    screenshot_test_passed: bool = False
    animation: AnimationInspection | None = None
    vision: dict[str, Any] = field(default_factory=dict)
    repair_action: str = "keep"
    repair_directives: list[str] = field(default_factory=list)
    reroute_renderer: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["score"] = round(float(self.score), 4)
        payload["label_coverage"] = round(float(self.label_coverage), 4)
        payload["object_coverage"] = round(float(self.object_coverage), 4)
        payload["animation"] = self.animation.to_dict() if self.animation else None
        return payload


def inspect_animation_frames(frame_paths: list[Path]) -> AnimationInspection:
    arrays: list[np.ndarray] = []
    for path in frame_paths:
        if not Path(path).is_file():
            continue
        image = iio.imread(path).astype(np.float32) / 255.0
        arrays.append(image[..., :3])
    if len(arrays) < 2:
        return AnimationInspection(
            passed=False,
            score=0.0,
            issues=["animation_inspection_requires_multiple_frames"],
        )
    deltas = [
        round(float(np.mean(np.abs(first - second))), 5)
        for first, second in zip(arrays, arrays[1:])
    ]
    active = sum(1 for delta in deltas if delta >= 0.009)
    final_hold = deltas[-1]
    issues: list[str] = []
    if active < min(2, len(deltas)):
        issues.append("animation_has_too_few_meaningful_state_changes")
    if max(deltas, default=0.0) < 0.012:
        issues.append("animation_is_effectively_static")
    if final_hold > 0.18:
        issues.append("resolved_state_does_not_hold")
    change_score = min(sum(min(delta / 0.035, 1.0) for delta in deltas) / max(len(deltas), 1), 1.0)
    hold_score = 1.0 if final_hold <= 0.12 else max(0.0, 1.0 - (final_hold - 0.12) / 0.2)
    score = round(change_score * 0.78 + hold_score * 0.22, 4)
    return AnimationInspection(
        passed=not issues and score >= 0.58,
        score=score,
        transition_deltas=deltas,
        active_transition_count=active,
        final_hold_delta=final_hold,
        issues=issues,
    )


def analyze_hyperframes_semantics(
    *,
    html: str,
    frame_paths: list[Path],
    production_contract: dict[str, Any],
    visual_explanation_ir: dict[str, Any],
    storyboard: list[dict[str, Any]],
    stage_metadata: dict[str, Any],
    qa_mode: str,
    vision_report: dict[str, Any] | None = None,
) -> HyperframesSemanticQaReport:
    required_labels = [
        str(item).strip()
        for item in production_contract.get("required_labels") or []
        if str(item).strip()
    ]
    visible_text = visible_text_from_html(html)
    stage_labels = [
        str(item).strip()
        for item in stage_metadata.get("visible_labels") or []
        if str(item).strip()
    ]
    visible_corpus = " ".join([visible_text, *stage_labels])
    missing_labels = [
        label for label in required_labels if not _copy_is_covered(label, visible_corpus)
    ]
    label_coverage = 1.0 - len(missing_labels) / max(len(required_labels), 1)
    objects = [
        dict(item)
        for item in visual_explanation_ir.get("objects") or []
        if isinstance(item, dict)
    ]
    missing_objects = [
        str(item.get("object_id") or "")
        for item in objects
        if not _copy_is_covered(str(item.get("label") or ""), visible_corpus)
    ]
    object_coverage = 1.0 - len(missing_objects) / max(len(objects), 1)
    animation = inspect_animation_frames(frame_paths)
    hard_failures: list[str] = []
    issues: list[str] = []
    warnings: list[str] = []
    if missing_labels:
        hard_failures.append("required_labels_missing_from_render")
    if missing_objects:
        hard_failures.append("grounded_objects_missing_from_render")
    if not storyboard or len(storyboard) < 2:
        hard_failures.append("storyboard_missing_or_incomplete")
    if not animation.passed:
        issues.extend(animation.issues)
        if production_contract.get("required_motion") or len(storyboard) >= 2:
            hard_failures.append("semantic_animation_contract_failed")
    screenshot_test_passed = (
        label_coverage >= 0.95
        and object_coverage >= 0.95
        and bool(frame_paths)
        and animation.final_hold_delta <= 0.18
    )
    if not screenshot_test_passed:
        hard_failures.append("resolved_frame_failed_local_screenshot_test")
    vision = dict(vision_report or {})
    vision_available = bool(vision.get("available"))
    vision_passed = vision.get("passed")
    if vision_available and vision_passed is False:
        hard_failures.append("vision_semantic_critique_failed")
        issues.extend(str(item) for item in vision.get("semantic_issues") or [])
        missing_labels.extend(str(item) for item in vision.get("missing_labels") or [])
    elif not vision_available:
        if str(qa_mode or "").lower() == "vision":
            hard_failures.append("strict_vision_qa_unavailable")
        elif str(qa_mode or "").lower() == "hybrid":
            warnings.append("vision_qa_unavailable_local_semantic_qa_used")
    vision_score = _optional_score(vision.get("score"))
    semantic_score = (
        label_coverage * 0.34
        + object_coverage * 0.28
        + animation.score * 0.23
        + (1.0 if screenshot_test_passed else 0.0) * 0.15
    )
    if vision_score is not None:
        semantic_score = semantic_score * 0.72 + vision_score * 0.28
    semantic_score = max(0.0, min(semantic_score, 1.0))
    repair_action, repair_directives, reroute = _repair_decision(
        hard_failures=hard_failures,
        issues=issues,
        scene_type=str(production_contract.get("scene_type") or ""),
        missing_labels=missing_labels,
        vision_directives=[
            str(item) for item in vision.get("repair_directives") or []
        ],
    )
    passed = not hard_failures and semantic_score >= float(
        production_contract.get("quality_floor") or 0.78
    )
    if not passed and semantic_score < float(production_contract.get("quality_floor") or 0.78):
        issues.append("semantic_quality_below_contract_floor")
    return HyperframesSemanticQaReport(
        passed=passed,
        score=semantic_score,
        hard_failures=_unique(hard_failures),
        issues=_unique(issues),
        warnings=_unique(warnings),
        missing_labels=_unique(missing_labels),
        missing_objects=_unique(missing_objects),
        label_coverage=label_coverage,
        object_coverage=object_coverage,
        screenshot_test_passed=screenshot_test_passed,
        animation=animation,
        vision=vision,
        repair_action=repair_action,
        repair_directives=repair_directives,
        reroute_renderer=reroute,
    )


def _repair_decision(
    *,
    hard_failures: list[str],
    issues: list[str],
    scene_type: str,
    missing_labels: list[str],
    vision_directives: list[str],
) -> tuple[str, list[str], str]:
    directives = list(vision_directives)
    reroute = ""
    if "strict_vision_qa_unavailable" in hard_failures:
        return "retry_when_vision_available", [], ""
    if "vision_semantic_critique_failed" in hard_failures:
        reroute = "ffmpeg_asset" if scene_type == "grounded_interface_walkthrough" else "manim"
        return "reroute_renderer", _unique(directives), reroute
    if "required_labels_missing_from_render" in hard_failures and missing_labels:
        directives.append("Place the missing grounded labels without changing their copy: " + "; ".join(missing_labels[:6]))
        return "repair_grounded_copy_placement", _unique(directives), ""
    if "grounded_objects_missing_from_render" in hard_failures:
        directives.append("Restore every required grounded object and keep its identity across beats.")
        return "repair_object_coverage", _unique(directives), ""
    if "resolved_frame_failed_local_screenshot_test" in hard_failures:
        directives.append("Hold the resolved state and keep all required labels visible in the final frame.")
        return "repair_final_hold", _unique(directives), ""
    if any("static" in issue or "state_changes" in issue for issue in issues):
        directives.append("Increase semantic state change without adding decorative motion.")
        return "repair_semantic_motion", _unique(directives), ""
    return ("keep" if not hard_failures and not issues else "reject_no_safe_repair", _unique(directives), reroute)


def _copy_is_covered(label: str, corpus: str) -> bool:
    normalized_label = _normalize(label)
    normalized_corpus = _normalize(corpus)
    if not normalized_label:
        return False
    if normalized_label in normalized_corpus:
        return True
    label_tokens = set(normalized_label.split())
    corpus_tokens = set(normalized_corpus.split())
    return len(label_tokens & corpus_tokens) / max(len(label_tokens), 1) >= 0.8


def _normalize(value: Any) -> str:
    return re.sub(r"[^a-z0-9%+./-]+", " ", str(value or "").lower()).strip()


def _optional_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(score, 1.0))


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = str(value or "").strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result


__all__ = [
    "AnimationInspection",
    "HyperframesSemanticQaReport",
    "analyze_hyperframes_semantics",
    "inspect_animation_frames",
]
