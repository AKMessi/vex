from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from vex_manim.briefs import SceneBrief


TEXT_ROLES = {"title", "text", "label", "footer", "quote", "support", "metric"}
IMMOVABLE_ROLES = {"panel", "background"}
BOTTOM_SAFE_ROLES = {"title", "text", "label", "support", "quote"}
CONNECTOR_ROLES = {"connector", "motion_spine"}


@dataclass
class LayoutBox:
    name: str
    role: str
    class_name: str
    left: float
    right: float
    top: float
    bottom: float
    width: float
    height: float
    center_x: float
    center_y: float
    text_based: bool = False
    panel_like: bool = False
    connector_like: bool = False
    font_size: float | None = None
    text_preview: str = ""
    allow_scale_down: bool = True
    priority: int = 0
    text_left: float | None = None
    text_right: float | None = None
    text_top: float | None = None
    text_bottom: float | None = None
    text_width: float | None = None
    text_height: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def bounds(self, *, prefer_text: bool = False) -> tuple[float, float, float, float]:
        text_left = self.text_left
        text_right = self.text_right
        text_top = self.text_top
        text_bottom = self.text_bottom
        if (
            prefer_text
            and text_left is not None
            and text_right is not None
            and text_top is not None
            and text_bottom is not None
        ):
            return (
                float(text_left),
                float(text_right),
                float(text_top),
                float(text_bottom),
            )
        return (self.left, self.right, self.top, self.bottom)


@dataclass
class LayoutReport:
    passed: bool
    score: float
    issues: list[str] = field(default_factory=list)
    registered_count: int = 0
    action_count: int = 0
    boxes: list[LayoutBox] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "score": self.score,
            "issues": list(self.issues),
            "registered_count": self.registered_count,
            "action_count": self.action_count,
            "boxes": [box.to_dict() for box in self.boxes],
        }


def load_layout_snapshot(path: str | Path) -> dict[str, Any]:
    snapshot_path = Path(path)
    return json.loads(snapshot_path.read_text(encoding="utf-8"))


def _intersects(
    first: LayoutBox,
    second: LayoutBox,
    *,
    first_prefer_text: bool = False,
    second_prefer_text: bool = False,
) -> tuple[float, float]:
    first_left, first_right, first_top, first_bottom = first.bounds(prefer_text=first_prefer_text)
    second_left, second_right, second_top, second_bottom = second.bounds(prefer_text=second_prefer_text)
    overlap_x = max(0.0, min(first_right, second_right) - max(first_left, second_left))
    overlap_y = max(0.0, min(first_top, second_top) - max(first_bottom, second_bottom))
    return overlap_x, overlap_y


def _overlap_ratio(
    first: LayoutBox,
    second: LayoutBox,
    *,
    first_prefer_text: bool = False,
    second_prefer_text: bool = False,
) -> float:
    overlap_x, overlap_y = _intersects(
        first,
        second,
        first_prefer_text=first_prefer_text,
        second_prefer_text=second_prefer_text,
    )
    if overlap_x <= 0.0 or overlap_y <= 0.0:
        return 0.0
    overlap_area = overlap_x * overlap_y
    first_area = (
        max(float(first.text_width or 0.0), 0.0) * max(float(first.text_height or 0.0), 0.0)
        if first_prefer_text and first.text_width is not None and first.text_height is not None
        else first.width * first.height
    )
    second_area = (
        max(float(second.text_width or 0.0), 0.0) * max(float(second.text_height or 0.0), 0.0)
        if second_prefer_text and second.text_width is not None and second.text_height is not None
        else second.width * second.height
    )
    min_area = max(min(first_area, second_area), 1e-6)
    return overlap_area / min_area


def _layout_boxes(snapshot: dict[str, Any]) -> list[LayoutBox]:
    raw_boxes = list(snapshot.get("registered") or []) or list(snapshot.get("top_level") or [])
    boxes: list[LayoutBox] = []
    for raw in raw_boxes:
        try:
            width = float(raw.get("width") or 0.0)
            height = float(raw.get("height") or 0.0)
            if width <= 0.01 or height <= 0.01:
                continue
            boxes.append(
                LayoutBox(
                    name=str(raw.get("name") or raw.get("path") or "object"),
                    role=str(raw.get("role") or "group"),
                    class_name=str(raw.get("class_name") or "Mobject"),
                    left=float(raw.get("left") or 0.0),
                    right=float(raw.get("right") or 0.0),
                    top=float(raw.get("top") or 0.0),
                    bottom=float(raw.get("bottom") or 0.0),
                    width=width,
                    height=height,
                    center_x=float(raw.get("center_x") or 0.0),
                    center_y=float(raw.get("center_y") or 0.0),
                    text_based=bool(raw.get("text_based")),
                    panel_like=bool(raw.get("panel_like")),
                    connector_like=bool(raw.get("connector_like")),
                    font_size=float(raw["font_size"]) if raw.get("font_size") is not None else None,
                    text_preview=str(raw.get("text_preview") or ""),
                    allow_scale_down=bool(raw.get("allow_scale_down", True)),
                    priority=int(raw.get("priority") or 0),
                    text_left=float(raw["text_left"]) if raw.get("text_left") is not None else None,
                    text_right=float(raw["text_right"]) if raw.get("text_right") is not None else None,
                    text_top=float(raw["text_top"]) if raw.get("text_top") is not None else None,
                    text_bottom=float(raw["text_bottom"]) if raw.get("text_bottom") is not None else None,
                    text_width=float(raw["text_width"]) if raw.get("text_width") is not None else None,
                    text_height=float(raw["text_height"]) if raw.get("text_height") is not None else None,
                )
            )
        except (TypeError, ValueError):
            continue
    return boxes


def analyze_layout_snapshot(snapshot: dict[str, Any], brief: SceneBrief) -> LayoutReport:
    frame = dict(snapshot.get("frame") or {})
    safe_bounds = dict(snapshot.get("safe_bounds") or {})
    boxes = _layout_boxes(snapshot)
    issues: list[str] = []
    registered_count = int(snapshot.get("registered_count") or len(snapshot.get("registered") or []))
    guardrail_actions = list(snapshot.get("guardrail_actions") or [])
    action_count = len(guardrail_actions)
    corrective_action_count = sum(
        1
        for item in guardrail_actions
        if str((item or {}).get("kind") or "").strip()
        in {"anchor_title", "anchor_chart", "anchor_hero", "anchor_metric", "anchor_support", "shift_into_bounds", "resolve_overlap", "scale_after_overlap"}
    )

    frame_left = float(frame.get("left") or -7.11)
    frame_right = float(frame.get("right") or 7.11)
    frame_top = float(frame.get("top") or 4.0)
    frame_bottom = float(frame.get("bottom") or -4.0)
    safe_left = float(safe_bounds.get("left") or frame_left)
    safe_right = float(safe_bounds.get("right") or frame_right)
    safe_top = float(safe_bounds.get("top") or frame_top)
    safe_bottom = float(safe_bounds.get("bottom") or frame_bottom)
    frame_width = max(frame_right - frame_left, 1.0)
    frame_height = max(frame_top - frame_bottom, 1.0)

    if registered_count == 0:
        issues.append("The scene did not register principal layout groups, so deterministic layout control is weak.")

    for box in boxes:
        connector_box = box.connector_like or box.role in CONNECTOR_ROLES
        overflow = (
            box.left < safe_left - 0.04
            or box.right > safe_right + 0.04
            or box.top > safe_top + 0.04
            or box.bottom < frame_bottom - 0.04
        )
        if overflow and not connector_box:
            issues.append(f"{box.name} extends outside the safe frame and may clip on screen.")
        if box.text_based and box.role in BOTTOM_SAFE_ROLES and box.bottom < safe_bottom - 0.02:
            issues.append(f"{box.name} falls into the bottom subtitle-safe region.")
        min_font_size = 15.0 if box.role in {"metric", "label", "hero", "support", "diagram", "system"} else 17.0
        if box.text_based and box.font_size is not None and box.font_size < min_font_size:
            issues.append(f"{box.name} is using a very small font size ({box.font_size:.1f}px).")
        readable_width = float(box.text_width) if box.text_width is not None else float(box.width)
        if box.text_based and box.role not in {"diagram", "system"} and readable_width > frame_width * 0.88:
            issues.append(f"{box.name} is too wide for comfortable readability.")
        if box.height > frame_height * 0.82 and box.role not in IMMOVABLE_ROLES:
            issues.append(f"{box.name} dominates too much of the frame and needs rebalancing.")

    for index, first in enumerate(boxes):
        for second in boxes[index + 1 :]:
            if first.panel_like and second.panel_like:
                continue
            if first.connector_like or second.connector_like or first.role in CONNECTOR_ROLES or second.role in CONNECTOR_ROLES:
                continue
            text_role_pair = first.role in TEXT_ROLES and second.role in TEXT_ROLES
            overlap_ratio = _overlap_ratio(
                first,
                second,
                first_prefer_text=first.text_based,
                second_prefer_text=second.text_based,
            )
            if overlap_ratio <= 0.0:
                continue
            if text_role_pair and first.text_based and second.text_based and overlap_ratio > 0.12:
                issues.append(f"{first.name} overlaps {second.name}; text elements are colliding.")
            elif first.text_based and not second.panel_like and overlap_ratio > 0.22:
                issues.append(f"{first.name} is colliding with {second.name}.")
            elif second.text_based and not first.panel_like and overlap_ratio > 0.22:
                issues.append(f"{second.name} is colliding with {first.name}.")

    panel_count = sum(1 for box in boxes if box.role == "panel" or box.panel_like)
    connector_count = sum(
        1
        for box in boxes
        if box.connector_like or box.role in {"connector", "diagram", "chart", "motion_spine"}
    )
    text_count = sum(1 for box in boxes if box.text_based)
    unique_previews = {
        " ".join(box.text_preview.split()).strip()
        for box in boxes
        if box.text_based and str(box.text_preview or "").strip()
    }
    visible_word_count = sum(len(preview.split()) for preview in unique_previews)
    if brief.composition_mode == "replace" and visible_word_count > brief.text_budget_words + 4:
        issues.append(
            f"The visible on-screen copy is still too dense ({visible_word_count} words for a target near {brief.text_budget_words})."
        )
    for box in boxes:
        if not box.text_based or not str(box.text_preview or "").strip():
            continue
        preview_words = len(box.text_preview.split())
        if box.role in {"support", "label", "metric"} and preview_words >= 9:
            issues.append(f"{box.name} is carrying too much copy for a fast-moving support element.")
        elif box.role not in {"title", "quote"} and preview_words >= 13:
            issues.append(f"{box.name} reads like a paragraph instead of a concise motion-graphics cue.")
    if brief.composition_mode == "replace" and brief.scene_family != "interface_focus":
        if panel_count >= 3 and connector_count == 0:
            issues.append("The scene is dominated by panels instead of a clearer motion system or diagram structure.")
        if panel_count >= text_count and connector_count == 0 and registered_count <= 5:
            issues.append("The composition still reads like boxed editorial cards rather than a bespoke animation.")
        if text_count >= 6 and connector_count == 0 and registered_count <= 6:
            issues.append("Too many separate text-led groups are carrying the frame; consolidate the copy into a stronger visual system.")
        if registered_count <= 3 and connector_count == 0 and panel_count >= 1:
            issues.append("Too few principal layout groups were registered for a full-screen replace visual; the structure feels under-authored.")

    if corrective_action_count >= 8:
        issues.append("The runtime had to reposition many layout groups; the composition should be authored through layout slots.")
    elif brief.animation_intensity in {"medium", "high"} and action_count >= 16:
        issues.append("The runtime had to apply many layout guardrails; the composition is probably over-constrained.")

    deduped: list[str] = []
    for issue in issues:
        cleaned = issue.strip()
        if cleaned and cleaned not in deduped:
            deduped.append(cleaned)

    score = 1.0
    score -= min(len(deduped) * 0.14, 0.82)
    score -= min(max(action_count - 3, 0), 5) * 0.03
    if registered_count >= 3:
        score += 0.08
    score = round(max(0.0, min(score, 1.0)), 3)
    return LayoutReport(
        passed=not deduped,
        score=score,
        issues=deduped,
        registered_count=registered_count,
        action_count=action_count,
        boxes=boxes,
    )
