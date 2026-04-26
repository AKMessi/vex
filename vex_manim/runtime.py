from __future__ import annotations

import json
import re
from typing import Any

from manim import (
    Arc,
    ArcBetweenPoints,
    Animation,
    BOLD,
    Circle,
    CurvedArrow,
    DashedLine,
    Dot,
    DOWN,
    FadeIn,
    LEFT,
    Line,
    ManimColor,
    MEDIUM,
    MovingCameraScene,
    NORMAL,
    Rectangle,
    RIGHT,
    RoundedRectangle,
    Text,
    UP,
    VGroup,
)


THEME_DEFAULTS = {
    "background": "#0B1020",
    "panel_fill": "#13203A",
    "panel_stroke": "#60A5FA",
    "accent": "#F59E0B",
    "accent_secondary": "#38BDF8",
    "glow": "#1D4ED8",
    "eyebrow_fill": "#14324D",
    "eyebrow_text": "#E0F2FE",
    "grid": "#214668",
    "text_primary": "#F8FAFC",
    "text_secondary": "#CBD5E1",
}

ROLE_PRIORITIES = {
    "background": 100,
    "panel": 90,
    "title": 85,
    "hero": 80,
    "chart": 75,
    "metric": 72,
    "quote": 70,
    "support": 64,
    "footer": 58,
    "label": 52,
    "connector": 40,
    "group": 30,
    "text": 28,
}

TEXT_CLASS_MARKERS = {
    "Text",
    "MarkupText",
    "Paragraph",
    "Code",
    "Tex",
    "MathTex",
    "DecimalNumber",
    "Integer",
    "Title",
}
PANEL_CLASS_MARKERS = {
    "Rectangle",
    "RoundedRectangle",
    "SurroundingRectangle",
    "BackgroundRectangle",
    "Circle",
    "Square",
}
CONNECTOR_CLASS_MARKERS = {
    "Line",
    "Arrow",
    "CurvedArrow",
    "DashedLine",
}


class VexGeneratedScene(MovingCameraScene):
    SCENE_SPEC: dict[str, Any] = {}
    SCENE_BRIEF: dict[str, Any] = {}

    def setup(self) -> None:
        super().setup()
        self.spec = dict(self.SCENE_SPEC or {})
        self.brief = dict(self.SCENE_BRIEF or {})
        self.theme = dict(THEME_DEFAULTS)
        self.theme.update({key: str(value) for key, value in dict(self.spec.get("theme") or {}).items() if value})
        self.camera.background_color = ManimColor(self.theme_color("background"))
        frame = self.camera.frame
        self._reference_frame_bounds = {
            "left": float(frame.get_left()[0]),
            "right": float(frame.get_right()[0]),
            "top": float(frame.get_top()[1]),
            "bottom": float(frame.get_bottom()[1]),
            "width": float(frame.width),
            "height": float(frame.height),
        }
        self._layout_registry: list[dict[str, Any]] = []
        self._layout_name_counts: dict[str, int] = {}
        self._guardrail_actions: list[dict[str, Any]] = []
        self._layout_dump_path = str(self.spec.get("layout_snapshot_path") or "").strip()
        self._guardrails_enabled = True
        self.stage_background = self.apply_house_background(
            motif=str(self.spec.get("background_motif") or self.brief.get("background_motif") or "constellation"),
            add=True,
        )

    def play(self, *args: Any, **kwargs: Any) -> Any:
        return super().play(*args, **kwargs)

    def wait(self, *args: Any, **kwargs: Any) -> Any:
        self._apply_layout_guardrails(reason="pre_wait")
        return super().wait(*args, **kwargs)

    def tear_down(self) -> None:
        self._apply_layout_guardrails(reason="tear_down")
        self._dump_layout_snapshot()
        super().tear_down()

    def theme_color(self, name: str, fallback: str | None = None) -> str:
        return str(self.theme.get(name) or fallback or THEME_DEFAULTS["text_primary"])

    def fit_text(
        self,
        text: str,
        *,
        max_width: float,
        max_font_size: int,
        min_font_size: int = 16,
        color: str | None = None,
        weight=BOLD,
        slant=NORMAL,
    ) -> Text:
        cleaned = str(text or "").strip() or " "
        for size in range(max_font_size, min_font_size - 1, -4):
            candidate = Text(
                cleaned,
                font_size=size,
                color=ManimColor(color or self.theme_color("text_primary")),
                weight=weight,
                slant=slant,
            )
            if candidate.width <= max_width:
                return candidate
        return Text(
            cleaned,
            font_size=min_font_size,
            color=ManimColor(color or self.theme_color("text_primary")),
            weight=weight,
            slant=slant,
        )

    def make_pill(
        self,
        text: str,
        *,
        fill: str | None = None,
        text_color: str | None = None,
        width: float | None = None,
    ) -> VGroup:
        label = self.fit_text(
            str(text or "").upper(),
            max_width=4.0 if width is None else max(width - 0.36, 1.0),
            max_font_size=24,
            min_font_size=14,
            color=text_color or self.theme_color("eyebrow_text"),
            weight=BOLD,
        )
        shell = RoundedRectangle(
            corner_radius=0.18,
            width=max(label.width + 0.44, width or 1.8),
            height=max(label.height + 0.24, 0.52),
        )
        shell.set_fill(ManimColor(fill or self.theme_color("eyebrow_fill")), opacity=1.0)
        shell.set_stroke(width=0)
        label.move_to(shell.get_center())
        return VGroup(shell, label)

    def make_glass_panel(
        self,
        width: float,
        height: float,
        *,
        stroke: str | None = None,
        fill: str | None = None,
        radius: float = 0.22,
    ) -> VGroup:
        outer = RoundedRectangle(corner_radius=radius, width=width, height=height)
        outer.set_fill(ManimColor(fill or self.theme_color("panel_fill")), opacity=0.95)
        outer.set_stroke(ManimColor(stroke or self.theme_color("panel_stroke")), width=2.4, opacity=0.95)
        inner = outer.copy()
        inner.scale(0.985)
        inner.set_stroke(ManimColor(fill or self.theme_color("panel_fill")), width=1.2, opacity=0.4)
        return VGroup(outer, inner)

    def make_title_block(
        self,
        eyebrow: str | None = None,
        headline: str | None = None,
        deck: str | None = None,
        *,
        max_width: float = 8.6,
    ) -> VGroup:
        header = VGroup()
        eyebrow_value = str(eyebrow or self.spec.get("eyebrow") or "").strip()
        headline_value = str(headline or self.spec.get("headline") or "").strip()
        deck_value = str(deck or self.spec.get("deck") or "").strip()
        if eyebrow_value:
            header.add(self.make_pill(eyebrow_value))
        if headline_value:
            header.add(self.fit_text(headline_value, max_width=max_width, max_font_size=52, min_font_size=28))
        if deck_value:
            header.add(
                self.fit_text(
                    deck_value,
                    max_width=max_width,
                    max_font_size=24,
                    min_font_size=16,
                    color=self.theme_color("text_secondary"),
                    weight=MEDIUM,
                )
            )
        if len(header) > 0:
            header.arrange(DOWN, aligned_edge=LEFT, buff=0.18)
            header.to_edge(LEFT, buff=0.78)
            header.to_edge(UP, buff=0.62)
            marker = Line(
                header.get_corner(DOWN + LEFT) + LEFT * 0.18,
                header.get_corner(UP + LEFT) + LEFT * 0.18,
                color=ManimColor(self.theme_color("accent")),
                stroke_width=5,
                stroke_opacity=0.9,
            )
            return VGroup(marker, header)
        return VGroup()

    def make_signal_node(self, label: str, *, number: int | None = None, radius: float = 0.8) -> VGroup:
        circle = Circle(radius=radius)
        circle.set_fill(ManimColor(self.theme_color("panel_fill")), opacity=1.0)
        circle.set_stroke(ManimColor(self.theme_color("panel_stroke")), width=3)
        halo = circle.copy().scale(1.18).set_stroke(ManimColor(self.theme_color("glow")), width=2, opacity=0.18).set_fill(opacity=0)
        parts = [halo, circle]
        if number is not None:
            badge = Text(str(number), font_size=22, color=ManimColor(self.theme_color("accent")), weight=BOLD)
            badge.next_to(circle.get_top(), DOWN, buff=0.24)
            parts.append(badge)
        text = self.fit_text(label, max_width=radius * 2.0, max_font_size=24, min_font_size=16)
        text.move_to(circle.get_center() + DOWN * 0.08)
        parts.append(text)
        return VGroup(*parts)

    def make_connector(self, left: Any, right: Any, *, curved: bool = True, color: str | None = None):
        if curved:
            return CurvedArrow(
                left.get_right() + RIGHT * 0.12,
                right.get_left() + LEFT * 0.12,
                angle=-0.24,
                color=ManimColor(color or self.theme_color("accent_secondary")),
                stroke_width=5,
            )
        return Line(
            left.get_right() + RIGHT * 0.12,
            right.get_left() + LEFT * 0.12,
            color=ManimColor(color or self.theme_color("accent_secondary")),
            stroke_width=4,
        )

    def make_glow_dot(
        self,
        *,
        radius: float = 0.11,
        color: str | None = None,
        glow_scale: float = 2.4,
        glow_opacity: float = 0.18,
    ) -> VGroup:
        tone = ManimColor(color or self.theme_color("accent"))
        glow = Circle(radius=radius * glow_scale).set_fill(tone, opacity=glow_opacity).set_stroke(width=0)
        core = Dot(radius=radius, color=tone)
        return VGroup(glow, core)

    def make_orbit_ring(
        self,
        radius: float,
        *,
        color: str | None = None,
        stroke_width: float = 2.4,
        opacity: float = 0.34,
        arc_angle: float | None = None,
        start_angle: float = 0.0,
    ):
        tone = ManimColor(color or self.theme_color("accent_secondary"))
        if arc_angle is not None:
            ring = Arc(radius=radius, start_angle=start_angle, angle=arc_angle)
        else:
            ring = Circle(radius=radius)
        ring.set_stroke(tone, width=stroke_width, opacity=opacity)
        ring.set_fill(opacity=0)
        return ring

    def make_route_path(
        self,
        start: Any,
        end: Any,
        *,
        angle: float = -0.35,
        color: str | None = None,
        dashed: bool = False,
        stroke_width: float = 4.0,
    ):
        tone = ManimColor(color or self.theme_color("accent_secondary"))
        if angle:
            path = ArcBetweenPoints(start, end, angle=angle)
            path.set_stroke(tone, width=stroke_width, opacity=0.92)
            return path
        if dashed:
            return DashedLine(start, end, color=tone, stroke_width=stroke_width)
        return Line(start, end, color=tone, stroke_width=stroke_width)

    def make_focus_beam(
        self,
        width: float,
        height: float,
        *,
        color: str | None = None,
        opacity: float = 0.14,
        angle: float = -0.28,
    ) -> Rectangle:
        beam = Rectangle(width=width, height=height)
        beam.set_fill(ManimColor(color or self.theme_color("glow")), opacity=opacity)
        beam.set_stroke(width=0)
        beam.rotate(angle)
        return beam

    def make_metric_badge(
        self,
        text: str,
        *,
        width: float = 2.1,
        fill: str | None = None,
        text_color: str | None = None,
    ) -> VGroup:
        shell = self.make_pill(
            text,
            fill=fill or self.theme_color("accent"),
            text_color=text_color or self.theme_color("background"),
            width=width,
        )
        halo = shell[0].copy().scale(1.18).set_fill(opacity=0).set_stroke(
            ManimColor(fill or self.theme_color("accent")),
            width=2.0,
            opacity=0.16,
        )
        return VGroup(halo, shell)

    def make_ribbon_label(
        self,
        text: str,
        *,
        max_width: float = 4.8,
        accent: str | None = None,
        text_color: str | None = None,
    ) -> VGroup:
        line = Line(LEFT * 1.6, RIGHT * 1.6, color=ManimColor(accent or self.theme_color("accent")), stroke_width=5)
        label = self.fit_text(
            text,
            max_width=max_width,
            max_font_size=34,
            min_font_size=18,
            color=text_color or self.theme_color("text_primary"),
            weight=BOLD,
        )
        label.next_to(line, UP, buff=0.18)
        return VGroup(line, label)

    def camera_focus(self, target: Any, *, scale: float = 0.92, run_time: float = 0.7) -> Animation:
        return self.camera.frame.animate.scale(scale).move_to(target).set_run_time(run_time)

    def stagger_fade_in(self, items: list[Any], *, shift=UP * 0.12, lag_ratio: float = 0.12) -> Animation:
        from manim import LaggedStart

        return LaggedStart(*[FadeIn(item, shift=shift) for item in items], lag_ratio=lag_ratio)

    def register_layout_group(
        self,
        name: str,
        mob: Any,
        *,
        role: str = "group",
        priority: int | None = None,
        allow_scale_down: bool = True,
        avoid_safe_bottom: bool | None = None,
    ) -> Any:
        base = re.sub(r"[^a-zA-Z0-9_]+", "_", str(name or role or "group")).strip("_") or "group"
        count = self._layout_name_counts.get(base, 0) + 1
        self._layout_name_counts[base] = count
        unique_name = base if count == 1 else f"{base}_{count}"
        entry = {
            "name": unique_name,
            "mob": mob,
            "role": str(role or "group"),
            "priority": ROLE_PRIORITIES.get(str(role or "group"), 30) if priority is None else int(priority),
            "allow_scale_down": bool(allow_scale_down),
            "avoid_safe_bottom": (
                str(role or "group") in {"title", "text", "label", "support", "quote"}
                if avoid_safe_bottom is None
                else bool(avoid_safe_bottom)
            ),
        }
        self._layout_registry.append(entry)
        return mob

    def register_text_group(self, name: str, mob: Any, *, role: str = "text", priority: int | None = None) -> Any:
        return self.register_layout_group(name, mob, role=role, priority=priority, allow_scale_down=True)

    def register_panel_group(self, name: str, mob: Any, *, priority: int | None = None) -> Any:
        return self.register_layout_group(name, mob, role="panel", priority=priority, allow_scale_down=False, avoid_safe_bottom=False)

    def apply_house_background(self, *, motif: str = "constellation", add: bool = False) -> VGroup:
        layers = VGroup()
        glow_color = self.theme_color("glow")
        accent_secondary = self.theme_color("accent_secondary")
        grid_color = self.theme_color("grid")
        left_glow = Circle(radius=3.1).set_fill(ManimColor(glow_color), opacity=0.12).set_stroke(width=0).move_to(LEFT * 4.4 + UP * 1.4)
        right_glow = Circle(radius=3.5).set_fill(ManimColor(accent_secondary), opacity=0.11).set_stroke(width=0).move_to(RIGHT * 4.5 + DOWN * 1.6)
        top_wash = RoundedRectangle(corner_radius=0.0, width=14.6, height=2.4).set_fill(ManimColor(glow_color), opacity=0.06).set_stroke(width=0).move_to(UP * 3.0)
        bottom_wash = RoundedRectangle(corner_radius=0.0, width=14.6, height=2.2).set_fill(ManimColor(accent_secondary), opacity=0.05).set_stroke(width=0).move_to(DOWN * 3.0)
        layers.add(left_glow, right_glow, top_wash, bottom_wash)
        if motif == "grid":
            grid = VGroup()
            for x in range(-6, 7):
                grid.add(Line([x * 1.08, -4.2, 0], [x * 1.08, 4.2, 0], stroke_width=1, color=ManimColor(grid_color), stroke_opacity=0.14))
            for y in range(-4, 5):
                grid.add(Line([-7.2, y * 0.94, 0], [7.2, y * 0.94, 0], stroke_width=1, color=ManimColor(grid_color), stroke_opacity=0.14))
            layers.add(grid)
        elif motif == "rings":
            rings = VGroup()
            for radius, opacity in ((3.7, 0.12), (2.7, 0.1), (1.75, 0.08)):
                ring = Circle(radius=radius).set_stroke(ManimColor(glow_color), width=1.6, opacity=opacity).set_fill(opacity=0)
                rings.add(ring)
            rings.move_to(RIGHT * 3.9 + UP * 0.2)
            layers.add(rings)
        elif motif == "bands":
            bands = VGroup()
            for offset, color, opacity in ((-1.8, glow_color, 0.14), (0.0, accent_secondary, 0.12), (1.7, glow_color, 0.09)):
                band = RoundedRectangle(corner_radius=0.0, width=5.4, height=0.22).set_fill(ManimColor(color), opacity=opacity).set_stroke(width=0)
                band.rotate(-0.34)
                band.move_to(RIGHT * 3.3 + UP * offset)
                bands.add(band)
            layers.add(bands)
        else:
            dots = VGroup()
            for x, y, radius, opacity in [(-4.8, 2.1, 0.06, 0.2), (-3.8, 1.35, 0.05, 0.15), (3.7, -1.2, 0.07, 0.18), (4.6, 1.7, 0.04, 0.14), (3.0, 2.3, 0.05, 0.12)]:
                dot = Circle(radius=radius).set_fill(ManimColor(glow_color if x < 0 else accent_secondary), opacity=opacity).set_stroke(width=0)
                dot.move_to(RIGHT * x + UP * y)
                dots.add(dot)
            lines = VGroup(
                Line(LEFT * 4.8 + UP * 2.1, LEFT * 3.8 + UP * 1.35, stroke_width=1.2, color=ManimColor(glow_color), stroke_opacity=0.14),
                Line(RIGHT * 3.7 + DOWN * 1.2, RIGHT * 4.6 + UP * 1.7, stroke_width=1.2, color=ManimColor(accent_secondary), stroke_opacity=0.14),
            )
            layers.add(dots, lines)
        if add:
            self.add(layers)
        return layers

    def _frame_bounds(self) -> dict[str, float]:
        return dict(self._reference_frame_bounds)

    def _safe_bounds(self) -> dict[str, float]:
        frame = self._frame_bounds()
        margin_x = frame["width"] * 0.04
        margin_top = frame["height"] * 0.05
        safe_bottom = frame["bottom"] + frame["height"] * 0.17
        return {
            "left": frame["left"] + margin_x,
            "right": frame["right"] - margin_x,
            "top": frame["top"] - margin_top,
            "bottom": safe_bottom,
        }

    def _is_text_based(self, mob: Any) -> bool:
        class_name = mob.__class__.__name__
        if class_name in TEXT_CLASS_MARKERS:
            return True
        return any(self._is_text_based(child) for child in getattr(mob, "submobjects", []))

    def _is_panel_like(self, mob: Any) -> bool:
        class_name = mob.__class__.__name__
        return class_name in PANEL_CLASS_MARKERS

    def _is_connector_like(self, mob: Any) -> bool:
        class_name = mob.__class__.__name__
        return class_name in CONNECTOR_CLASS_MARKERS

    def _text_preview(self, mob: Any) -> str:
        text = getattr(mob, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()[:120]
        for child in getattr(mob, "submobjects", []):
            preview = self._text_preview(child)
            if preview:
                return preview
        return ""

    def _font_size(self, mob: Any) -> float | None:
        font_size = getattr(mob, "font_size", None)
        if isinstance(font_size, (int, float)):
            return float(font_size)
        sizes = [self._font_size(child) for child in getattr(mob, "submobjects", [])]
        sizes = [value for value in sizes if value is not None]
        return max(sizes) if sizes else None

    def _record_guardrail_action(self, kind: str, target: str, **details: Any) -> None:
        self._guardrail_actions.append(
            {
                "kind": kind,
                "target": target,
                "details": {key: details[key] for key in sorted(details)},
            }
        )

    def _top_level_entries(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for index, mob in enumerate(self.mobjects):
            if mob is self.stage_background or float(getattr(mob, "width", 0.0) or 0.0) <= 0.01 or float(getattr(mob, "height", 0.0) or 0.0) <= 0.01:
                continue
            entries.append(
                {
                    "name": f"top_level_{index:02d}",
                    "mob": mob,
                    "role": "text" if self._is_text_based(mob) else "group",
                    "priority": 24,
                    "allow_scale_down": self._is_text_based(mob),
                    "avoid_safe_bottom": self._is_text_based(mob),
                }
            )
        return entries

    def _active_layout_entries(self) -> list[dict[str, Any]]:
        return list(self._layout_registry or self._top_level_entries())

    def _bbox(self, mob: Any) -> dict[str, float]:
        return {
            "left": float(mob.get_left()[0]),
            "right": float(mob.get_right()[0]),
            "top": float(mob.get_top()[1]),
            "bottom": float(mob.get_bottom()[1]),
            "width": float(mob.width),
            "height": float(mob.height),
            "center_x": float(mob.get_center()[0]),
            "center_y": float(mob.get_center()[1]),
        }

    def _clamp_inside_bounds(self, entry: dict[str, Any], *, safe_bounds: dict[str, float], frame_bounds: dict[str, float]) -> None:
        mob = entry["mob"]
        if float(getattr(mob, "width", 0.0) or 0.0) <= 0.01 or float(getattr(mob, "height", 0.0) or 0.0) <= 0.01:
            return
        box = self._bbox(mob)
        left_bound = safe_bounds["left"]
        right_bound = safe_bounds["right"]
        top_bound = safe_bounds["top"]
        bottom_bound = safe_bounds["bottom"] if entry.get("avoid_safe_bottom") else frame_bounds["bottom"] + frame_bounds["height"] * 0.04
        shift_x = 0.0
        shift_y = 0.0
        if box["left"] < left_bound:
            shift_x = left_bound - box["left"]
        elif box["right"] > right_bound:
            shift_x = right_bound - box["right"]
        if box["top"] > top_bound:
            shift_y = top_bound - box["top"]
        elif box["bottom"] < bottom_bound:
            shift_y = bottom_bound - box["bottom"]
        if abs(shift_x) > 1e-4 or abs(shift_y) > 1e-4:
            mob.shift(RIGHT * shift_x + UP * shift_y)
            self._record_guardrail_action("shift_into_bounds", entry["name"], dx=round(shift_x, 4), dy=round(shift_y, 4))

    def _move_entry_to(self, entry: dict[str, Any], *, center_x: float, center_y: float, kind: str) -> None:
        mob = entry["mob"]
        current = mob.get_center()
        dx = float(center_x - current[0])
        dy = float(center_y - current[1])
        if abs(dx) <= 1e-4 and abs(dy) <= 1e-4:
            return
        mob.shift(RIGHT * dx + UP * dy)
        self._record_guardrail_action(kind, entry["name"], dx=round(dx, 4), dy=round(dy, 4))

    def _anchor_layout_roles(self, entries: list[dict[str, Any]], *, safe_bounds: dict[str, float], frame_bounds: dict[str, float]) -> None:
        titles = [entry for entry in entries if str(entry.get("role") or "") == "title"]
        charts = [entry for entry in entries if str(entry.get("role") or "") == "chart"]
        metrics = [entry for entry in entries if str(entry.get("role") or "") == "metric"]
        supports = [entry for entry in entries if str(entry.get("role") or "") in {"support", "footer", "quote"}]

        title_box = None
        if titles:
            title = sorted(titles, key=lambda item: int(item.get("priority") or 0), reverse=True)[0]
            box = self._bbox(title["mob"])
            target_x = safe_bounds["left"] + box["width"] / 2
            target_y = safe_bounds["top"] - box["height"] / 2
            if abs(box["left"] - safe_bounds["left"]) > 0.18 or box["top"] > safe_bounds["top"] + 0.08:
                self._move_entry_to(title, center_x=target_x, center_y=target_y, kind="anchor_title")
            title_box = self._bbox(title["mob"])

        chart_ceiling = safe_bounds["top"] - frame_bounds["height"] * 0.04
        if title_box is not None:
            chart_ceiling = min(chart_ceiling, title_box["bottom"] - 0.2)
        for chart in charts:
            box = self._bbox(chart["mob"])
            if box["top"] > chart_ceiling:
                target_y = chart_ceiling - box["height"] / 2
                self._move_entry_to(chart, center_x=box["center_x"], center_y=target_y, kind="anchor_chart")

        for index, metric in enumerate(sorted(metrics, key=lambda item: int(item.get("priority") or 0), reverse=True)):
            box = self._bbox(metric["mob"])
            target_x = safe_bounds["right"] - box["width"] / 2
            if title_box is not None:
                target_y = title_box["bottom"] - 0.18 - box["height"] / 2 - index * (box["height"] + 0.12)
            else:
                target_y = safe_bounds["top"] - box["height"] / 2 - index * (box["height"] + 0.12)
            overlaps_title_band = title_box is not None and box["top"] > title_box["bottom"] - 0.08 and box["left"] < title_box["right"] + 0.18
            if overlaps_title_band or box["right"] > safe_bounds["right"] + 0.08:
                self._move_entry_to(metric, center_x=target_x, center_y=target_y, kind="anchor_metric")

        chart_centers = [self._bbox(entry["mob"])["center_x"] for entry in charts]
        support_to_left = not chart_centers or sum(chart_centers) / len(chart_centers) >= 0.0
        for index, support in enumerate(sorted(supports, key=lambda item: int(item.get("priority") or 0), reverse=True)):
            box = self._bbox(support["mob"])
            target_x = (
                safe_bounds["left"] + box["width"] / 2
                if support_to_left
                else safe_bounds["right"] - box["width"] / 2
            )
            target_y = safe_bounds["bottom"] + box["height"] / 2 + index * (box["height"] + 0.12)
            if charts or box["bottom"] < safe_bounds["bottom"] - 0.04 or any(
                max(0.0, min(box["right"], self._bbox(chart["mob"])["right"]) - max(box["left"], self._bbox(chart["mob"])["left"])) > 0.18
                and max(0.0, min(box["top"], self._bbox(chart["mob"])["top"]) - max(box["bottom"], self._bbox(chart["mob"])["bottom"])) > 0.18
                for chart in charts
            ):
                self._move_entry_to(support, center_x=target_x, center_y=target_y, kind="anchor_support")

    def _scale_text_group(self, entry: dict[str, Any], frame_bounds: dict[str, float]) -> None:
        if not entry.get("allow_scale_down") or not self._is_text_based(entry["mob"]):
            return
        mob = entry["mob"]
        role = str(entry.get("role") or "text")
        width_factor = {
            "title": 0.7,
            "quote": 0.76,
            "footer": 0.72,
            "support": 0.44,
            "label": 0.26,
            "metric": 0.46,
            "chart": 0.5,
        }.get(role, 0.62)
        max_width = frame_bounds["width"] * width_factor
        if float(mob.width) > max_width > 0.0:
            scale_factor = max(max_width / max(float(mob.width), 1e-6), 0.76)
            if scale_factor < 0.995:
                mob.scale(scale_factor)
                self._record_guardrail_action("scale_text_group", entry["name"], factor=round(scale_factor, 4))

    def _text_inside_panel(self, text_entry: dict[str, Any], panel_entry: dict[str, Any]) -> bool:
        text_box = self._bbox(text_entry["mob"])
        panel_box = self._bbox(panel_entry["mob"])
        return (
            text_box["center_x"] >= panel_box["left"]
            and text_box["center_x"] <= panel_box["right"]
            and text_box["center_y"] >= panel_box["bottom"]
            and text_box["center_y"] <= panel_box["top"]
        )

    def _fit_text_inside_panels(self, entries: list[dict[str, Any]]) -> None:
        panels = [entry for entry in entries if entry.get("role") == "panel"]
        texts = [entry for entry in entries if self._is_text_based(entry["mob"])]
        for text_entry in texts:
            if not text_entry.get("allow_scale_down"):
                continue
            for panel_entry in panels:
                if not self._text_inside_panel(text_entry, panel_entry):
                    continue
                text_box = self._bbox(text_entry["mob"])
                panel_box = self._bbox(panel_entry["mob"])
                max_width = max(panel_box["width"] - 0.42, 0.4)
                max_height = max(panel_box["height"] - 0.36, 0.28)
                scale_width = max_width / max(text_box["width"], 1e-6)
                scale_height = max_height / max(text_box["height"], 1e-6)
                scale_factor = min(scale_width, scale_height, 1.0)
                if scale_factor < 0.97:
                    text_entry["mob"].scale(max(scale_factor, 0.78))
                    text_entry["mob"].move_to(panel_entry["mob"].get_center())
                    self._record_guardrail_action(
                        "fit_text_inside_panel",
                        text_entry["name"],
                        panel=panel_entry["name"],
                        factor=round(max(scale_factor, 0.78), 4),
                    )

    def _resolve_overlap(self, first: dict[str, Any], second: dict[str, Any], *, safe_bounds: dict[str, float], frame_bounds: dict[str, float]) -> bool:
        first_mob = first["mob"]
        second_mob = second["mob"]
        first_box = self._bbox(first_mob)
        second_box = self._bbox(second_mob)
        overlap_x = max(0.0, min(first_box["right"], second_box["right"]) - max(first_box["left"], second_box["left"]))
        overlap_y = max(0.0, min(first_box["top"], second_box["top"]) - max(first_box["bottom"], second_box["bottom"]))
        if overlap_x <= 0.0 or overlap_y <= 0.0:
            return False
        if self._is_panel_like(first_mob) and self._is_text_based(second_mob) and self._text_inside_panel(second, first):
            return False
        if self._is_panel_like(second_mob) and self._is_text_based(first_mob) and self._text_inside_panel(first, second):
            return False
        first_priority = int(first.get("priority") or 0)
        second_priority = int(second.get("priority") or 0)
        mover = second if first_priority >= second_priority else first
        anchor = first if mover is second else second
        if mover.get("role") == "panel" and anchor.get("role") != "panel":
            mover, anchor = anchor, mover
        mover_box = self._bbox(mover["mob"])
        anchor_box = self._bbox(anchor["mob"])
        dx = 0.0
        dy = 0.0
        padding = 0.18
        move_horizontal = overlap_x < overlap_y
        if move_horizontal:
            direction = 1.0 if mover_box["center_x"] >= anchor_box["center_x"] else -1.0
            dx = direction * (overlap_x + padding)
        else:
            direction = 1.0 if mover_box["center_y"] >= anchor_box["center_y"] else -1.0
            dy = direction * (overlap_y + padding)
        mover["mob"].shift(RIGHT * dx + UP * dy)
        if mover.get("allow_scale_down") and (abs(dx) > 0.6 or abs(dy) > 0.6):
            mover["mob"].scale(0.96)
            self._record_guardrail_action("scale_after_overlap", mover["name"], factor=0.96)
        self._clamp_inside_bounds(mover, safe_bounds=safe_bounds, frame_bounds=frame_bounds)
        self._record_guardrail_action("resolve_overlap", mover["name"], against=anchor["name"], dx=round(dx, 4), dy=round(dy, 4))
        return True

    def _apply_layout_guardrails(self, *, reason: str) -> None:
        if not self._guardrails_enabled:
            return
        entries = self._active_layout_entries()
        if not entries:
            return
        frame_bounds = self._frame_bounds()
        safe_bounds = self._safe_bounds()
        for entry in entries:
            self._scale_text_group(entry, frame_bounds)
        self._anchor_layout_roles(entries, safe_bounds=safe_bounds, frame_bounds=frame_bounds)
        self._fit_text_inside_panels(entries)
        for entry in entries:
            self._clamp_inside_bounds(entry, safe_bounds=safe_bounds, frame_bounds=frame_bounds)
        for _ in range(3):
            moved = False
            for index, first in enumerate(entries):
                for second in entries[index + 1 :]:
                    if self._is_connector_like(first["mob"]) or self._is_connector_like(second["mob"]):
                        continue
                    if self._resolve_overlap(first, second, safe_bounds=safe_bounds, frame_bounds=frame_bounds):
                        moved = True
            if not moved:
                break

    def _entry_payload(self, entry: dict[str, Any]) -> dict[str, Any]:
        mob = entry["mob"]
        box = self._bbox(mob)
        payload = {
            "name": str(entry.get("name") or mob.__class__.__name__),
            "role": str(entry.get("role") or "group"),
            "class_name": mob.__class__.__name__,
            "left": round(box["left"], 4),
            "right": round(box["right"], 4),
            "top": round(box["top"], 4),
            "bottom": round(box["bottom"], 4),
            "width": round(box["width"], 4),
            "height": round(box["height"], 4),
            "center_x": round(box["center_x"], 4),
            "center_y": round(box["center_y"], 4),
            "text_based": self._is_text_based(mob),
            "panel_like": self._is_panel_like(mob),
            "connector_like": self._is_connector_like(mob),
            "font_size": self._font_size(mob),
            "text_preview": self._text_preview(mob),
            "allow_scale_down": bool(entry.get("allow_scale_down", True)),
            "priority": int(entry.get("priority") or 0),
        }
        return payload

    def _dump_layout_snapshot(self) -> None:
        if not self._layout_dump_path:
            return
        try:
            payload = {
                "frame": self._frame_bounds(),
                "safe_bounds": self._safe_bounds(),
                "registered_count": len(self._layout_registry),
                "registered": [self._entry_payload(entry) for entry in self._layout_registry],
                "top_level": [self._entry_payload(entry) for entry in self._top_level_entries()[:24]],
                "guardrail_actions": list(self._guardrail_actions[-80:]),
            }
            with open(self._layout_dump_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        except OSError:
            return
