from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


CANONICAL_FRAME_HEIGHT = 8.0


@dataclass(frozen=True)
class LayoutCanvas:
    pixel_width: int
    pixel_height: int
    frame_width: float
    frame_height: float = CANONICAL_FRAME_HEIGHT
    safe_margin_x: float = 0.0
    safe_margin_top: float = 0.0
    subtitle_safe_height: float = 0.0

    @classmethod
    def from_dimensions(
        cls,
        width: int,
        height: int,
        *,
        frame_height: float = CANONICAL_FRAME_HEIGHT,
    ) -> "LayoutCanvas":
        resolved_width = max(int(width or 0), 1)
        resolved_height = max(int(height or 0), 1)
        frame_w = float(frame_height) * resolved_width / resolved_height
        return cls(
            pixel_width=resolved_width,
            pixel_height=resolved_height,
            frame_width=frame_w,
            frame_height=float(frame_height),
            safe_margin_x=frame_w * 0.04,
            safe_margin_top=float(frame_height) * 0.05,
            subtitle_safe_height=float(frame_height) * 0.17,
        )

    @classmethod
    def from_frame(
        cls,
        *,
        pixel_width: int,
        pixel_height: int,
        left: float,
        right: float,
        top: float,
        bottom: float,
    ) -> "LayoutCanvas":
        frame_width = max(float(right) - float(left), 1e-6)
        frame_height = max(float(top) - float(bottom), 1e-6)
        return cls(
            pixel_width=max(int(pixel_width or 0), 1),
            pixel_height=max(int(pixel_height or 0), 1),
            frame_width=frame_width,
            frame_height=frame_height,
            safe_margin_x=frame_width * 0.04,
            safe_margin_top=frame_height * 0.05,
            subtitle_safe_height=frame_height * 0.17,
        )

    @property
    def aspect_ratio(self) -> float:
        return self.pixel_width / max(self.pixel_height, 1)

    @property
    def aspect_class(self) -> str:
        ratio = self.aspect_ratio
        if ratio < 0.8:
            return "vertical"
        if ratio > 1.35:
            return "landscape"
        return "square"

    @property
    def left(self) -> float:
        return -self.frame_width / 2.0

    @property
    def right(self) -> float:
        return self.frame_width / 2.0

    @property
    def top(self) -> float:
        return self.frame_height / 2.0

    @property
    def bottom(self) -> float:
        return -self.frame_height / 2.0

    @property
    def safe_left(self) -> float:
        return self.left + self.safe_margin_x

    @property
    def safe_right(self) -> float:
        return self.right - self.safe_margin_x

    @property
    def safe_top(self) -> float:
        return self.top - self.safe_margin_top

    @property
    def safe_bottom(self) -> float:
        return self.bottom + self.subtitle_safe_height

    @property
    def safe_width(self) -> float:
        return max(self.safe_right - self.safe_left, 0.0)

    @property
    def safe_height(self) -> float:
        return max(self.safe_top - self.safe_bottom, 0.0)

    def frame_bounds(self) -> dict[str, float]:
        return {
            "left": self.left,
            "right": self.right,
            "top": self.top,
            "bottom": self.bottom,
            "width": self.frame_width,
            "height": self.frame_height,
        }

    def safe_bounds(self) -> dict[str, float]:
        return {
            "left": self.safe_left,
            "right": self.safe_right,
            "top": self.safe_top,
            "bottom": self.safe_bottom,
            "width": self.safe_width,
            "height": self.safe_height,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "aspect_ratio": round(self.aspect_ratio, 4),
                "aspect_class": self.aspect_class,
                "frame": self.frame_bounds(),
                "safe_bounds": self.safe_bounds(),
            }
        )
        return payload


@dataclass(frozen=True)
class LayoutSlot:
    slot_id: str
    role: str
    left: float
    right: float
    top: float
    bottom: float
    padding: float = 0.12
    layer: str = "content"
    allow_overlap: bool = False
    notes: str = ""

    @property
    def width(self) -> float:
        return max(float(self.right) - float(self.left), 0.0)

    @property
    def height(self) -> float:
        return max(float(self.top) - float(self.bottom), 0.0)

    @property
    def inner_width(self) -> float:
        return max(self.width - self.padding * 2.0, 0.08)

    @property
    def inner_height(self) -> float:
        return max(self.height - self.padding * 2.0, 0.08)

    @property
    def center_x(self) -> float:
        return (float(self.left) + float(self.right)) / 2.0

    @property
    def center_y(self) -> float:
        return (float(self.top) + float(self.bottom)) / 2.0

    @property
    def center(self) -> tuple[float, float, float]:
        return (self.center_x, self.center_y, 0.0)

    def point(self, x: float = 0.5, y: float = 0.5) -> tuple[float, float, float]:
        px = float(self.left) + self.width * max(0.0, min(float(x), 1.0))
        py = float(self.bottom) + self.height * max(0.0, min(float(y), 1.0))
        return (px, py, 0.0)

    def anchor(self, side: str) -> tuple[float, float, float]:
        normalized = str(side or "center").lower()
        if normalized == "left":
            return (self.left, self.center_y, 0.0)
        if normalized == "right":
            return (self.right, self.center_y, 0.0)
        if normalized == "top":
            return (self.center_x, self.top, 0.0)
        if normalized == "bottom":
            return (self.center_x, self.bottom, 0.0)
        if normalized == "upper_left":
            return (self.left, self.top, 0.0)
        if normalized == "upper_right":
            return (self.right, self.top, 0.0)
        if normalized == "lower_left":
            return (self.left, self.bottom, 0.0)
        if normalized == "lower_right":
            return (self.right, self.bottom, 0.0)
        return self.center

    def anchor_toward(self, other: "LayoutSlot") -> tuple[float, float, float]:
        dx = other.center_x - self.center_x
        dy = other.center_y - self.center_y
        if abs(dx) >= abs(dy):
            return self.anchor("right" if dx >= 0 else "left")
        return self.anchor("top" if dy >= 0 else "bottom")

    def intersects(self, other: "LayoutSlot", *, padding: float = 0.0) -> bool:
        if self.allow_overlap or other.allow_overlap:
            return False
        return (
            min(self.right, other.right) - max(self.left, other.left) > padding
            and min(self.top, other.top) - max(self.bottom, other.bottom) > padding
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "width": round(self.width, 4),
                "height": round(self.height, 4),
                "inner_width": round(self.inner_width, 4),
                "inner_height": round(self.inner_height, 4),
                "center_x": round(self.center_x, 4),
                "center_y": round(self.center_y, 4),
            }
        )
        return payload


@dataclass(frozen=True)
class LayoutSpec:
    scene_family: str
    aspect_class: str
    canvas: LayoutCanvas
    slots: dict[str, LayoutSlot] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)
    route_points: dict[str, list[tuple[float, float, float]]] = field(default_factory=dict)

    def slot(self, slot_id: str) -> LayoutSlot:
        normalized = str(slot_id or "").strip()
        target = self.aliases.get(normalized, normalized)
        if target in self.slots:
            return self.slots[target]
        role_match = next((slot for slot in self.slots.values() if slot.role == normalized), None)
        if role_match is not None:
            return role_match
        if "hero" in self.slots:
            return self.slots["hero"]
        return next(iter(self.slots.values()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_family": self.scene_family,
            "aspect_class": self.aspect_class,
            "canvas": self.canvas.to_dict(),
            "slots": {key: slot.to_dict() for key, slot in self.slots.items()},
            "aliases": dict(self.aliases),
            "route_points": {
                key: [list(point) for point in points]
                for key, points in self.route_points.items()
            },
        }


def _field(source: Any, name: str, default: Any = "") -> Any:
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


def _slot(
    canvas: LayoutCanvas,
    slot_id: str,
    role: str,
    left: float,
    right: float,
    top: float,
    bottom: float,
    *,
    padding: float = 0.12,
    layer: str = "content",
    allow_overlap: bool = False,
    notes: str = "",
) -> LayoutSlot:
    safe_left = canvas.safe_left
    safe_right = canvas.safe_right
    safe_top = canvas.safe_top
    safe_bottom = canvas.safe_bottom
    clamped_left = max(min(float(left), safe_right), safe_left)
    clamped_right = max(min(float(right), safe_right), safe_left)
    clamped_top = max(min(float(top), safe_top), safe_bottom)
    clamped_bottom = max(min(float(bottom), safe_top), safe_bottom)
    if clamped_right < clamped_left:
        clamped_left, clamped_right = clamped_right, clamped_left
    if clamped_top < clamped_bottom:
        clamped_bottom, clamped_top = clamped_top, clamped_bottom
    return LayoutSlot(
        slot_id=slot_id,
        role=role,
        left=clamped_left,
        right=clamped_right,
        top=clamped_top,
        bottom=clamped_bottom,
        padding=padding,
        layer=layer,
        allow_overlap=allow_overlap,
        notes=notes,
    )


def _add_aliases(slots: dict[str, LayoutSlot], aliases: dict[str, str]) -> dict[str, str]:
    prepared = dict(aliases)
    for key, slot in slots.items():
        prepared.setdefault(key, key)
        prepared.setdefault(slot.role, key)
    return prepared


def _base_slots(canvas: LayoutCanvas) -> dict[str, LayoutSlot]:
    s = canvas.safe_bounds()
    return {
        "full": _slot(canvas, "full", "group", s["left"], s["right"], s["top"], s["bottom"], allow_overlap=True),
        "title": _slot(
            canvas,
            "title",
            "title",
            s["left"],
            s["right"],
            s["top"],
            s["top"] - min(1.18, s["height"] * 0.2),
            padding=0.08,
        ),
    }


def _metric_slots(canvas: LayoutCanvas) -> tuple[dict[str, LayoutSlot], dict[str, list[tuple[float, float, float]]]]:
    s = canvas.safe_bounds()
    slots = _base_slots(canvas)
    if canvas.aspect_class == "vertical":
        slots.update(
            {
                "metric": _slot(canvas, "metric", "metric", s["left"], s["right"], 2.28, 1.12, padding=0.1),
                "chart": _slot(canvas, "chart", "chart", s["left"], s["right"], 0.82, -1.24, padding=0.16),
                "support": _slot(canvas, "support", "support", s["left"], s["right"], -1.52, s["bottom"], padding=0.08),
                "motion_spine": _slot(canvas, "motion_spine", "motion_spine", s["left"], s["right"], 0.82, -1.24, allow_overlap=True),
            }
        )
    elif canvas.aspect_class == "square":
        mid = (s["left"] + s["right"]) / 2.0
        slots.update(
            {
                "metric": _slot(canvas, "metric", "metric", s["left"], mid - 0.12, 1.82, 0.15, padding=0.1),
                "chart": _slot(canvas, "chart", "chart", mid + 0.12, s["right"], 1.82, -1.15, padding=0.14),
                "support": _slot(canvas, "support", "support", s["left"], s["right"], -1.42, s["bottom"], padding=0.08),
                "motion_spine": _slot(canvas, "motion_spine", "motion_spine", mid + 0.12, s["right"], 1.82, -1.15, allow_overlap=True),
            }
        )
    else:
        split = s["left"] + s["width"] * 0.42
        slots.update(
            {
                "metric": _slot(canvas, "metric", "metric", s["left"], split - 0.2, 1.72, -0.25, padding=0.12),
                "chart": _slot(canvas, "chart", "chart", split + 0.16, s["right"], 2.12, -1.4, padding=0.16),
                "support": _slot(canvas, "support", "support", s["left"], split - 0.2, -0.65, s["bottom"], padding=0.08),
                "motion_spine": _slot(canvas, "motion_spine", "motion_spine", split + 0.16, s["right"], 2.12, -1.4, allow_overlap=True),
            }
        )
    routes = {"main": [slots["metric"].anchor_toward(slots["chart"]), slots["chart"].anchor_toward(slots["metric"])]}
    return slots, routes


def _comparison_slots(canvas: LayoutCanvas) -> tuple[dict[str, LayoutSlot], dict[str, list[tuple[float, float, float]]]]:
    s = canvas.safe_bounds()
    slots = _base_slots(canvas)
    if canvas.aspect_class == "vertical":
        slots.update(
            {
                "before": _slot(canvas, "before", "hero", s["left"], s["right"], 2.18, 0.72, padding=0.12),
                "after": _slot(canvas, "after", "hero", s["left"], s["right"], 0.34, -1.18, padding=0.12),
                "support": _slot(canvas, "support", "support", s["left"], s["right"], -1.56, s["bottom"], padding=0.08),
                "motion_spine": _slot(canvas, "motion_spine", "motion_spine", s["left"], s["right"], 0.7, 0.34, allow_overlap=True),
            }
        )
    else:
        left_mid = s["left"] + s["width"] * 0.46
        right_mid = s["left"] + s["width"] * 0.54
        slots.update(
            {
                "before": _slot(canvas, "before", "hero", s["left"], left_mid - 0.18, 1.72, -1.05, padding=0.14),
                "after": _slot(canvas, "after", "hero", right_mid + 0.18, s["right"], 1.72, -1.05, padding=0.14),
                "support": _slot(canvas, "support", "support", s["left"] + s["width"] * 0.24, s["right"] - s["width"] * 0.24, -1.38, s["bottom"], padding=0.08),
                "motion_spine": _slot(canvas, "motion_spine", "motion_spine", left_mid - 0.05, right_mid + 0.05, 0.62, -0.18, allow_overlap=True),
            }
        )
    routes = {"main": [slots["before"].anchor_toward(slots["after"]), slots["after"].anchor_toward(slots["before"])]}
    return slots, routes


def _timeline_slots(canvas: LayoutCanvas) -> tuple[dict[str, LayoutSlot], dict[str, list[tuple[float, float, float]]]]:
    s = canvas.safe_bounds()
    slots = _base_slots(canvas)
    if canvas.aspect_class == "vertical":
        top_y = 2.0
        gap = 0.88
        for index in range(4):
            y = top_y - index * gap
            slots[f"step_{index + 1}"] = _slot(canvas, f"step_{index + 1}", "label", s["left"], s["right"], y + 0.32, y - 0.32, padding=0.06)
        slots["route"] = _slot(canvas, "route", "diagram", -0.36, 0.36, 2.32, -1.28, allow_overlap=True)
        slots["support"] = _slot(canvas, "support", "footer", s["left"], s["right"], -1.72, s["bottom"], padding=0.08)
        route_points = [(0.0, 2.08, 0.0), (0.0, 1.16, 0.0), (0.0, 0.24, 0.0), (0.0, -0.68, 0.0)]
    else:
        route_top = 1.42
        route_bottom = -1.42
        anchors = [
            (s["left"] + s["width"] * 0.08, -1.1, 0.0),
            (s["left"] + s["width"] * 0.36, -0.3, 0.0),
            (s["left"] + s["width"] * 0.64, -1.0, 0.0),
            (s["left"] + s["width"] * 0.92, -0.18, 0.0),
        ]
        for index, point in enumerate(anchors, start=1):
            max_half_width = min(max(s["width"] * 0.13, 0.72), 1.1)
            slots[f"step_{index}"] = _slot(
                canvas,
                f"step_{index}",
                "label",
                point[0] - max_half_width,
                point[0] + max_half_width,
                point[1] + 0.95,
                point[1] + 0.22,
                padding=0.06,
            )
        slots["route"] = _slot(canvas, "route", "diagram", s["left"], s["right"], route_top, route_bottom, allow_overlap=True)
        slots["support"] = _slot(canvas, "support", "footer", s["left"] + s["width"] * 0.16, s["right"] - s["width"] * 0.16, -1.72, s["bottom"], padding=0.08)
        route_points = anchors
    return slots, {"main": route_points}


def _system_slots(canvas: LayoutCanvas) -> tuple[dict[str, LayoutSlot], dict[str, list[tuple[float, float, float]]]]:
    s = canvas.safe_bounds()
    slots = _base_slots(canvas)
    if canvas.aspect_class == "vertical":
        slots.update(
            {
                "source": _slot(canvas, "source", "hero", s["left"], s["right"], 2.08, 1.05, padding=0.1),
                "hub": _slot(canvas, "hub", "hero", s["left"], s["right"], 0.78, -0.28, padding=0.1),
                "outcome": _slot(canvas, "outcome", "hero", s["left"], s["right"], -0.56, -1.58, padding=0.1),
                "motion_spine": _slot(canvas, "motion_spine", "diagram", -0.48, 0.48, 2.16, -1.62, allow_overlap=True),
                "support": _slot(canvas, "support", "support", s["left"], s["right"], -1.84, s["bottom"], padding=0.08),
            }
        )
    else:
        slots.update(
            {
                "source": _slot(canvas, "source", "hero", s["left"], s["left"] + s["width"] * 0.27, 0.78, -1.46, padding=0.1),
                "hub": _slot(canvas, "hub", "hero", s["left"] + s["width"] * 0.36, s["right"] - s["width"] * 0.36, 1.18, -0.88, padding=0.1),
                "outcome": _slot(canvas, "outcome", "hero", s["right"] - s["width"] * 0.29, s["right"], 1.58, -0.62, padding=0.1),
                "motion_spine": _slot(canvas, "motion_spine", "diagram", s["left"] + s["width"] * 0.2, s["right"] - s["width"] * 0.2, 1.1, -1.0, allow_overlap=True),
                "support": _slot(canvas, "support", "support", s["left"], s["left"] + s["width"] * 0.32, -1.72, s["bottom"], padding=0.08),
            }
        )
    routes = {
        "main": [
            slots["source"].anchor_toward(slots["hub"]),
            slots["hub"].center,
            slots["outcome"].anchor_toward(slots["hub"]),
        ]
    }
    return slots, routes


def _interface_slots(canvas: LayoutCanvas) -> tuple[dict[str, LayoutSlot], dict[str, list[tuple[float, float, float]]]]:
    s = canvas.safe_bounds()
    slots = _base_slots(canvas)
    if canvas.aspect_class == "vertical":
        tops = [2.08, 0.72, -0.64]
        for index, top in enumerate(tops, start=1):
            slots[f"module_{index}"] = _slot(canvas, f"module_{index}", "hero", s["left"], s["right"], top, top - 0.94, padding=0.1)
        slots["motion_spine"] = _slot(canvas, "motion_spine", "diagram", -0.32, 0.32, 1.14, -0.74, allow_overlap=True)
        slots["support"] = _slot(canvas, "support", "support", s["left"], s["right"], -1.86, s["bottom"], padding=0.08)
    else:
        gap = s["width"] * 0.035
        module_w = (s["width"] - gap * 2.0) / 3.0
        for index in range(3):
            left = s["left"] + index * (module_w + gap)
            slots[f"module_{index + 1}"] = _slot(canvas, f"module_{index + 1}", "hero", left, left + module_w, 0.88, -1.2, padding=0.1)
        slots["motion_spine"] = _slot(canvas, "motion_spine", "diagram", s["left"] + module_w * 0.7, s["right"] - module_w * 0.7, 0.3, -0.45, allow_overlap=True)
        slots["support"] = _slot(canvas, "support", "support", s["left"] + s["width"] * 0.25, s["right"] - s["width"] * 0.25, -1.72, s["bottom"], padding=0.08)
    routes = {
        "main": [
            slots["module_1"].anchor_toward(slots["module_2"]),
            slots["module_2"].center,
            slots["module_3"].anchor_toward(slots["module_2"]),
        ]
    }
    return slots, routes


def _kinetic_slots(canvas: LayoutCanvas) -> tuple[dict[str, LayoutSlot], dict[str, list[tuple[float, float, float]]]]:
    s = canvas.safe_bounds()
    slots = _base_slots(canvas)
    slots.update(
        {
            "hero": _slot(canvas, "hero", "hero", s["left"], s["right"], 1.55, -0.25, padding=0.14),
            "motion_spine": _slot(canvas, "motion_spine", "diagram", s["left"], s["right"], -0.42, -1.68, padding=0.08, allow_overlap=True),
            "support": _slot(canvas, "support", "support", s["left"], s["right"], -1.84, s["bottom"], padding=0.08),
        }
    )
    if canvas.aspect_class == "vertical":
        route = [
            (s["left"] + s["width"] * 0.2, -0.92, 0.0),
            (s["left"] + s["width"] * 0.5, -1.28, 0.0),
            (s["left"] + s["width"] * 0.8, -0.92, 0.0),
        ]
    else:
        route = [
            (s["left"] + s["width"] * 0.12, -1.48, 0.0),
            (s["left"] + s["width"] * 0.42, -1.08, 0.0),
            (s["left"] + s["width"] * 0.7, -1.48, 0.0),
            (s["left"] + s["width"] * 0.9, -0.98, 0.0),
        ]
    return slots, {"main": route}


def build_layout_spec(
    brief: Any,
    blueprint: Any | None = None,
    *,
    canvas: LayoutCanvas | None = None,
    width: int | None = None,
    height: int | None = None,
) -> LayoutSpec:
    render_constraints = dict(_field(brief, "render_constraints", {}) or {})
    resolved_width = int(width or render_constraints.get("width") or 1920)
    resolved_height = int(height or render_constraints.get("height") or 1080)
    layout_canvas = canvas or LayoutCanvas.from_dimensions(resolved_width, resolved_height)
    scene_family = str(_field(brief, "scene_family", "") or _field(blueprint, "scene_family", "") or "concept_map")
    if scene_family in {"metric_story", "dashboard_build"}:
        slots, routes = _metric_slots(layout_canvas)
        aliases = {"hero_metric": "metric", "evidence_axis": "chart", "metric_graph": "chart"}
    elif scene_family == "comparison_morph":
        slots, routes = _comparison_slots(layout_canvas)
        aliases = {
            "hero": "after",
            "comparison_before_state": "before",
            "comparison_after_state": "after",
            "comparison_verdict": "support",
        }
    elif scene_family == "timeline_journey":
        slots, routes = _timeline_slots(layout_canvas)
        aliases = {"timeline_route": "route", "route_bundle": "route", "footer": "support", "motion_spine": "route"}
    elif scene_family == "system_map":
        slots, routes = _system_slots(layout_canvas)
        aliases = {"network_nodes": "hub", "network_paths": "motion_spine"}
    elif scene_family == "interface_focus":
        slots, routes = _interface_slots(layout_canvas)
        aliases = {"interface_modules": "module_2", "interface_connectors": "motion_spine", "focus": "module_2"}
    else:
        slots, routes = _kinetic_slots(layout_canvas)
        aliases = {"quote": "hero", "quote_stage": "hero", "kinetic_emphasis": "hero", "kinetic_spine": "motion_spine"}
    aliases.update({"title_group": "title", "title_band": "title", "motion_spine": "motion_spine"})
    return LayoutSpec(
        scene_family=scene_family,
        aspect_class=layout_canvas.aspect_class,
        canvas=layout_canvas,
        slots=slots,
        aliases=_add_aliases(slots, aliases),
        route_points=routes,
    )
