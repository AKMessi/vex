from __future__ import annotations

import json
import math
import re
from typing import Any

import manim
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
    ORIGIN,
    Rectangle,
    RIGHT,
    RoundedRectangle,
    Text,
    UP,
    VGroup,
    VMobject,
)

from vex_manim.layout import LayoutCanvas, LayoutSpec, build_layout_spec


CENTER = ORIGIN
utils = manim.rate_functions
rate_functions = manim.rate_functions
ease_in_sine = manim.rate_functions.ease_in_sine
ease_out_sine = manim.rate_functions.ease_out_sine
ease_in_out_sine = manim.rate_functions.ease_in_out_sine
sine_in = ease_in_sine
sine_out = ease_out_sine
sine_in_out = ease_in_out_sine


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


def _format_numeric_text(
    value: Any,
    *,
    num_decimal_places: int = 0,
    include_sign: bool = False,
    group_with_commas: bool = False,
) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if num_decimal_places <= 0:
        rendered = f"{int(round(numeric))}"
    else:
        rendered = f"{numeric:.{int(num_decimal_places)}f}"
    if group_with_commas:
        if "." in rendered:
            whole, frac = rendered.split(".", 1)
            whole = f"{int(whole):,}"
            rendered = f"{whole}.{frac}"
        else:
            rendered = f"{int(rendered):,}"
    if include_sign and numeric > 0 and not rendered.startswith("+"):
        rendered = f"+{rendered}"
    return rendered


def DecimalNumber(
    number: Any = 0,
    *,
    num_decimal_places: int = 0,
    include_sign: bool = False,
    group_with_commas: bool = False,
    font_size: float = 36,
    color: Any | None = None,
    weight=BOLD,
    slant=NORMAL,
    **_: Any,
) -> Text:
    return Text(
        _format_numeric_text(
            number,
            num_decimal_places=num_decimal_places,
            include_sign=include_sign,
            group_with_commas=group_with_commas,
        ),
        font_size=font_size,
        color=color,
        weight=weight,
        slant=slant,
    )


def Integer(
    number: Any = 0,
    *,
    group_with_commas: bool = False,
    font_size: float = 36,
    color: Any | None = None,
    weight=BOLD,
    slant=NORMAL,
    **kwargs: Any,
) -> Text:
    return DecimalNumber(
        number,
        num_decimal_places=0,
        group_with_commas=group_with_commas,
        font_size=font_size,
        color=color,
        weight=weight,
        slant=slant,
        **kwargs,
    )

ROLE_PRIORITIES = {
    "background": 100,
    "panel": 90,
    "title": 85,
    "hero": 80,
    "chart": 75,
    "structure": 75,
    "system": 75,
    "diagram": 75,
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
        self._layout_spec_dump_path = str(self.spec.get("layout_spec_path") or "").strip()
        self._guardrails_enabled = True
        self._layout_spec = self.build_layout()
        self.stage_background = self.apply_house_background(
            motif=str(self.spec.get("background_motif") or self.brief.get("background_motif") or "constellation"),
            add=True,
        )

    def play(self, *args: Any, **kwargs: Any) -> Any:
        self._apply_layout_guardrails(reason="pre_play")
        result = super().play(*args, **kwargs)
        self._apply_layout_guardrails(reason="post_play")
        return result

    def wait(self, *args: Any, **kwargs: Any) -> Any:
        self._apply_layout_guardrails(reason="pre_wait")
        return super().wait(*args, **kwargs)

    def tear_down(self) -> None:
        self._apply_layout_guardrails(reason="tear_down")
        self._dump_layout_snapshot()
        super().tear_down()

    def theme_color(self, name: str, fallback: str | None = None) -> str:
        return str(self.theme.get(name) or fallback or THEME_DEFAULTS["text_primary"])

    @property
    def camera_frame(self):
        return self.camera.frame

    def build_layout(self, *, scene_family: str | None = None, blueprint: Any | None = None) -> LayoutSpec:
        frame = dict(getattr(self, "_reference_frame_bounds", {}) or {})
        render_constraints = dict(self.brief.get("render_constraints") or {})
        pixel_width = int(render_constraints.get("width") or self.spec.get("width") or 1920)
        pixel_height = int(render_constraints.get("height") or self.spec.get("height") or 1080)
        if frame:
            canvas = LayoutCanvas.from_frame(
                pixel_width=pixel_width,
                pixel_height=pixel_height,
                left=float(frame["left"]),
                right=float(frame["right"]),
                top=float(frame["top"]),
                bottom=float(frame["bottom"]),
            )
        else:
            canvas = LayoutCanvas.from_dimensions(pixel_width, pixel_height)
        brief_payload = dict(self.brief)
        if scene_family:
            brief_payload["scene_family"] = scene_family
        brief_payload["render_constraints"] = {
            **render_constraints,
            "width": pixel_width,
            "height": pixel_height,
            "aspect_ratio": round(pixel_width / max(pixel_height, 1), 3),
        }
        return build_layout_spec(brief_payload, blueprint, canvas=canvas)

    @property
    def layout_spec(self) -> LayoutSpec:
        return self._layout_spec

    def layout_slot(self, slot_id: str):
        return self._layout_spec.slot(slot_id)

    def _layout_point(self, point: Any):
        try:
            return ORIGIN + RIGHT * float(point[0]) + UP * float(point[1])
        except (TypeError, ValueError, IndexError, KeyError):
            return ORIGIN

    def slot_point(self, slot_id: str, *, x: float = 0.5, y: float = 0.5):
        return self._layout_point(self.layout_slot(slot_id).point(x=x, y=y))

    def slot_anchor(self, slot_id: str, side: str = "center"):
        return self._layout_point(self.layout_slot(slot_id).anchor(side))

    def layout_route_points(self, route_id: str = "main") -> list[Any]:
        raw_points = list(self._layout_spec.route_points.get(route_id) or [])
        return [self._layout_point(point) for point in raw_points]

    def place_in_slot(
        self,
        name: str,
        mob: Any,
        slot_id: str,
        *,
        role: str | None = None,
        fit: bool = True,
        scale_up: bool = False,
        x_align: str = "center",
        y_align: str = "center",
        priority: int | None = None,
        allow_scale_down: bool = True,
        avoid_safe_bottom: bool | None = None,
        register: bool = True,
    ) -> Any:
        slot = self.layout_slot(slot_id)
        if fit and float(getattr(mob, "width", 0.0) or 0.0) > 0.01 and float(getattr(mob, "height", 0.0) or 0.0) > 0.01:
            scale_factor = min(
                slot.inner_width / max(float(mob.width), 1e-6),
                slot.inner_height / max(float(mob.height), 1e-6),
            )
            if scale_up:
                scale_factor = min(scale_factor, 1.16)
            else:
                scale_factor = min(scale_factor, 1.0)
            if scale_factor < 0.995 or (scale_up and scale_factor > 1.005):
                mob.scale(max(scale_factor, 0.08))

        mob_width = float(getattr(mob, "width", 0.0) or 0.0)
        mob_height = float(getattr(mob, "height", 0.0) or 0.0)
        normalized_x = str(x_align or "center").lower()
        normalized_y = str(y_align or "center").lower()
        if normalized_x == "left":
            target_x = slot.left + slot.padding + mob_width / 2.0
        elif normalized_x == "right":
            target_x = slot.right - slot.padding - mob_width / 2.0
        else:
            target_x = slot.center_x
        if normalized_y == "top":
            target_y = slot.top - slot.padding - mob_height / 2.0
        elif normalized_y == "bottom":
            target_y = slot.bottom + slot.padding + mob_height / 2.0
        else:
            target_y = slot.center_y
        mob.move_to(ORIGIN + RIGHT * target_x + UP * target_y)
        if register:
            return self.register_layout_group(
                name,
                mob,
                role=role or slot.role,
                priority=priority,
                allow_scale_down=allow_scale_down,
                avoid_safe_bottom=avoid_safe_bottom,
                slot_id=slot.slot_id,
            )
        return mob

    def fit_text_for_slot(
        self,
        text: str,
        slot_id: str,
        *,
        max_font_size: int = 34,
        min_font_size: int = 14,
        max_lines: int = 3,
        color: str | None = None,
        weight=BOLD,
        slant=NORMAL,
        role: str | None = None,
        name: str | None = None,
    ) -> Text:
        slot = self.layout_slot(slot_id)
        text_mob = self.fit_text(
            text,
            max_width=slot.inner_width,
            max_font_size=max_font_size,
            min_font_size=min_font_size,
            max_lines=max_lines,
            color=color,
            weight=weight,
            slant=slant,
        )
        if float(text_mob.height) > slot.inner_height:
            text_mob.scale(max(slot.inner_height / max(float(text_mob.height), 1e-6), 0.72))
        if name:
            self.place_in_slot(name, text_mob, slot_id, role=role or slot.role, fit=False)
        return text_mob

    def route_between_slots(
        self,
        source_slot: str,
        target_slot: str,
        *,
        source_side: str | None = None,
        target_side: str | None = None,
        bend: float | None = None,
        color: str | None = None,
        stroke_width: float = 4.0,
        dashed: bool = False,
        opacity: float = 0.92,
    ):
        source = self.layout_slot(source_slot)
        target = self.layout_slot(target_slot)
        start = source.anchor(source_side) if source_side else source.anchor_toward(target)
        end = target.anchor(target_side) if target_side else target.anchor_toward(source)
        return self.make_route_path(
            self._layout_point(start),
            self._layout_point(end),
            bend=0.0 if bend is None and self.layout_spec.aspect_class == "vertical" else bend,
            color=color,
            stroke_width=stroke_width,
            dashed=dashed,
            opacity=opacity,
        )

    def fit_text(
        self,
        text: str,
        *,
        max_width: float | None = None,
        max_font_size: int | None = None,
        min_font_size: int = 16,
        max_lines: int = 4,
        color: str | None = None,
        weight=BOLD,
        slant=NORMAL,
        font_weight=None,
        font_style=None,
    ) -> Text:
        if font_weight is not None:
            weight = font_weight
        if font_style is not None:
            slant = font_style
        cleaned = re.sub(r"\s+", " ", str(text or "")).strip() or " "
        resolved_max_width = float(max_width if max_width is not None else min(max(len(cleaned) * 0.28, 3.6), 10.0))
        resolved_max_font_size = int(max_font_size if max_font_size is not None else 34)

        def render_candidate(content: str, size: int) -> Text:
            return Text(
                content,
                font_size=size,
                color=ManimColor(color or self.theme_color("text_primary")),
                weight=weight,
                slant=slant,
            )

        def wrap_variant(content: str, size: int) -> str:
            words = [word for word in content.split(" ") if word]
            if len(words) <= 1:
                return content
            best_variant = content
            best_overflow = max(render_candidate(content, size).width - resolved_max_width, 0.0)
            for line_limit in range(2, max(max_lines, 2) + 1):
                lines: list[str] = []
                current: list[str] = []
                success = True
                for word in words:
                    tentative = " ".join([*current, word]).strip()
                    if current and render_candidate(tentative, size).width > resolved_max_width:
                        lines.append(" ".join(current).strip())
                        current = [word]
                        if len(lines) >= line_limit:
                            success = False
                            break
                    else:
                        current.append(word)
                if not success or not current:
                    continue
                lines.append(" ".join(current).strip())
                if len(lines) > line_limit:
                    continue
                candidate_variant = "\n".join(lines)
                candidate = render_candidate(candidate_variant, size)
                overflow = max(candidate.width - resolved_max_width, 0.0)
                if overflow <= 0.01:
                    return candidate_variant
                if overflow < best_overflow:
                    best_variant = candidate_variant
                    best_overflow = overflow
            return best_variant

        for size in range(resolved_max_font_size, min_font_size - 1, -4):
            candidate = render_candidate(cleaned, size)
            if candidate.width <= resolved_max_width:
                return candidate
            wrapped_variant = wrap_variant(cleaned, size)
            wrapped_candidate = render_candidate(wrapped_variant, size)
            if wrapped_candidate.width <= resolved_max_width:
                return wrapped_candidate
        final_variant = wrap_variant(cleaned, min_font_size)
        return Text(
            final_variant,
            font_size=min_font_size,
            color=ManimColor(color or self.theme_color("text_primary")),
            weight=weight,
            slant=slant,
        )

    def make_pill(
        self,
        text: str = "",
        *args: Any,
        fill: str | None = None,
        text_color: str | None = None,
        width: float | None = None,
        height: float | None = None,
        fill_opacity: float = 1.0,
        stroke_color: str | None = None,
        stroke_width: float = 0.0,
        stroke_opacity: float = 0.0,
        color: str | None = None,
        accent: str | None = None,
        **_: Any,
    ) -> VGroup:
        if args:
            if len(args) >= 1 and fill is None:
                fill = str(args[0])
            if len(args) >= 2 and text_color is None:
                text_color = str(args[1])
            if len(args) >= 3 and width is None:
                try:
                    width = float(args[2])
                except (TypeError, ValueError):
                    pass
            if len(args) >= 4 and height is None:
                try:
                    height = float(args[3])
                except (TypeError, ValueError):
                    pass
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
            height=max(height or 0.0, label.height + 0.24, 0.52),
        )
        resolved_fill = fill or accent or color or self.theme_color("eyebrow_fill")
        shell.set_fill(ManimColor(resolved_fill), opacity=fill_opacity)
        shell.set_stroke(
            ManimColor(stroke_color or resolved_fill),
            width=stroke_width,
            opacity=stroke_opacity,
        )
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
        stroke_color: str | None = None,
        fill_color: str | None = None,
        fill_opacity: float = 0.95,
        stroke_opacity: float = 0.95,
        stroke_width: float = 2.4,
        color: str | None = None,
        **_: Any,
    ) -> VGroup:
        outer = RoundedRectangle(corner_radius=radius, width=width, height=height)
        resolved_fill = fill_color or fill or self.theme_color("panel_fill")
        resolved_stroke = stroke_color or stroke or color or self.theme_color("panel_stroke")
        outer.set_fill(ManimColor(resolved_fill), opacity=fill_opacity)
        outer.set_stroke(ManimColor(resolved_stroke), width=stroke_width, opacity=stroke_opacity)
        inner = outer.copy()
        inner.scale(0.985)
        inner.set_stroke(ManimColor(resolved_fill), width=1.2, opacity=0.4)
        return VGroup(outer, inner)

    def make_title_block(
        self,
        eyebrow: str | None = None,
        headline: str | None = None,
        deck: str | None = None,
        *,
        max_width: float = 8.6,
        color: str | None = None,
        deck_color: str | None = None,
        eyebrow_fill: str | None = None,
        eyebrow_text_color: str | None = None,
        accent_color: str | None = None,
        title: str | None = None,
        subtitle: str | None = None,
        **_: Any,
    ) -> VGroup:
        header = VGroup()
        eyebrow_value = str(eyebrow or self.spec.get("eyebrow") or "").strip()
        headline_value = str(headline or title or self.spec.get("headline") or "").strip()
        deck_value = str(deck or subtitle or self.spec.get("deck") or "").strip()
        if eyebrow_value:
            header.add(
                self.make_pill(
                    eyebrow_value,
                    fill=eyebrow_fill or self.theme_color("eyebrow_fill"),
                    text_color=eyebrow_text_color or self.theme_color("eyebrow_text"),
                )
            )
        if headline_value:
            header.add(
                self.fit_text(
                    headline_value,
                    max_width=max_width,
                    max_font_size=52,
                    min_font_size=28,
                    color=color or self.theme_color("text_primary"),
                )
            )
        if deck_value:
            header.add(
                self.fit_text(
                    deck_value,
                    max_width=max_width,
                    max_font_size=24,
                    min_font_size=16,
                    color=deck_color or self.theme_color("text_secondary"),
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
                color=ManimColor(accent_color or self.theme_color("accent")),
                stroke_width=5,
                stroke_opacity=0.9,
            )
            return VGroup(marker, header)
        return VGroup()

    def make_signal_node(
        self,
        label: str = "",
        *,
        number: int | None = None,
        radius: float = 0.8,
        color: str | None = None,
        fill_color: str | None = None,
        stroke_color: str | None = None,
        stroke_width: float = 3.0,
        title: str | None = None,
        **_: Any,
    ) -> VGroup:
        label = str(label or title or "").strip()
        circle = Circle(radius=radius)
        circle.set_fill(ManimColor(fill_color or self.theme_color("panel_fill")), opacity=1.0)
        circle.set_stroke(ManimColor(stroke_color or color or self.theme_color("panel_stroke")), width=stroke_width)
        halo = circle.copy().scale(1.18).set_stroke(ManimColor(self.theme_color("glow")), width=2, opacity=0.18).set_fill(opacity=0)
        parts = [halo, circle]
        if number is not None:
            badge = Text(str(number), font_size=22, color=ManimColor(self.theme_color("accent")), weight=BOLD)
            badge.next_to(circle.get_top(), DOWN, buff=0.24)
            parts.append(badge)
        text = self.fit_text(
            label,
            max_width=max(radius * 1.56, 0.92),
            max_font_size=max(18, int(18 + radius * 8)),
            min_font_size=11,
            max_lines=3,
        )
        max_text_width = max(radius * 1.54, 0.88)
        max_text_height = max(radius * 1.12, 0.72)
        if float(text.width) > max_text_width or float(text.height) > max_text_height:
            scale_factor = min(
                max_text_width / max(float(text.width), 1e-6),
                max_text_height / max(float(text.height), 1e-6),
                1.0,
            )
            if scale_factor < 0.995:
                text.scale(max(scale_factor, 0.62))
        text.move_to(circle.get_center())
        parts.append(text)
        return VGroup(*parts)

    def make_connector(
        self,
        left: Any = None,
        right: Any = None,
        *,
        curved: bool = True,
        color: str | None = None,
        start: Any = None,
        end: Any = None,
        source: Any = None,
        target: Any = None,
        **_: Any,
    ):
        left = left if left is not None else start if start is not None else source
        right = right if right is not None else end if end is not None else target
        if left is None or right is None:
            raise ValueError("make_connector requires two endpoints.")
        def anchor_point(value: Any, *, side: str) -> Any:
            if hasattr(value, "get_right") and hasattr(value, "get_left"):
                return value.get_right() if side == "right" else value.get_left()
            return value
        if curved:
            return CurvedArrow(
                anchor_point(left, side="right") + RIGHT * 0.12,
                anchor_point(right, side="left") + LEFT * 0.12,
                angle=-0.24,
                color=ManimColor(color or self.theme_color("accent_secondary")),
                stroke_width=5,
            )
        return Line(
            anchor_point(left, side="right") + RIGHT * 0.12,
            anchor_point(right, side="left") + LEFT * 0.12,
            color=ManimColor(color or self.theme_color("accent_secondary")),
            stroke_width=4,
        )

    def make_glow_dot(
        self,
        *,
        radius: float = 0.11,
        color: str | None = None,
        glow_color: str | None = None,
        glow_scale: float = 2.4,
        glow_opacity: float = 0.18,
        opacity: float | None = None,
        **_: Any,
    ) -> VGroup:
        tone = ManimColor(color or self.theme_color("accent"))
        glow_tone = ManimColor(glow_color or color or self.theme_color("glow"))
        overall_opacity = 1.0 if opacity is None else max(0.0, min(float(opacity), 1.0))
        glow = Circle(radius=radius * glow_scale).set_fill(
            glow_tone,
            opacity=max(0.0, min(glow_opacity * overall_opacity, 1.0)),
        ).set_stroke(width=0)
        core = Dot(radius=radius, color=tone)
        core.set_opacity(overall_opacity)
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
        **_: Any,
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
        start: Any = None,
        end: Any = None,
        *,
        angle: float = -0.35,
        color: str | None = None,
        dashed: bool = False,
        stroke_width: float = 4.0,
        bend: float | None = None,
        curvature: float | None = None,
        curve_height: float | None = None,
        opacity: float = 0.92,
        path_points: Any = None,
        points: Any = None,
        start_point: Any = None,
        end_point: Any = None,
        from_point: Any = None,
        to_point: Any = None,
        **_: Any,
    ):
        if path_points is not None:
            start = path_points
        elif points is not None:
            start = points
        start = start if start is not None else start_point if start_point is not None else from_point
        end = end if end is not None else end_point if end_point is not None else to_point
        if start is None or end is None:
            if isinstance(start, (list, tuple)) and len(start) >= 2:
                path = VMobject()
                path.set_points_smoothly(list(start))
                path.set_stroke(ManimColor(color or self.theme_color("accent_secondary")), width=stroke_width, opacity=opacity)
                path.set_fill(opacity=0)
                return path
            raise ValueError("make_route_path requires start and end points.")
        tone = ManimColor(color or self.theme_color("accent_secondary"))
        if bend is not None:
            angle = bend
        if curvature is not None:
            angle = curvature
        if curve_height is not None and curvature is None and bend is None:
            try:
                curve_value = float(curve_height)
                if abs(curve_value) > 1e-6:
                    angle = max(min(curve_value / 5.5, 1.15), -1.15)
            except (TypeError, ValueError):
                pass
        if angle:
            path = ArcBetweenPoints(start, end, angle=angle)
            path.set_stroke(tone, width=stroke_width, opacity=opacity)
            return path
        if dashed:
            line = DashedLine(start, end, color=tone, stroke_width=stroke_width)
            line.set_stroke(opacity=opacity)
            return line
        line = Line(start, end, color=tone, stroke_width=stroke_width)
        line.set_stroke(opacity=opacity)
        return line

    def make_focus_beam(
        self,
        width: float | None = None,
        height: float | None = None,
        *,
        length: float | None = None,
        color: str | None = None,
        opacity: float = 0.14,
        angle: float = -0.28,
        center: Any | None = None,
        **_: Any,
    ) -> Rectangle:
        resolved_width = float(width if width is not None else length if length is not None else 4.8)
        resolved_height = float(height if height is not None else 0.42)
        beam = Rectangle(width=resolved_width, height=resolved_height)
        beam.set_fill(ManimColor(color or self.theme_color("glow")), opacity=opacity)
        beam.set_stroke(width=0)
        beam.rotate(angle)
        if center is not None:
            beam.move_to(center)
        return beam

    def make_metric_badge(
        self,
        text: str | None = None,
        subtext: str | None = None,
        *,
        value: str | None = None,
        unit: str | None = None,
        label: str | None = None,
        width: float = 2.1,
        fill: str | None = None,
        text_color: str | None = None,
        color: str | None = None,
        **kwargs: Any,
    ) -> VGroup:
        resolved_fill = fill or color or self.theme_color("accent")
        resolved_text_color = text_color or self.theme_color("background")
        value_text = str(text if text not in {None, ""} else value or "").strip()
        unit_text = str(subtext if subtext not in {None, ""} else unit or "").strip()
        label_text = str(label or "").strip()
        if not unit_text and not label_text:
            shell = self.make_pill(
                value_text,
                fill=resolved_fill,
                text_color=resolved_text_color,
                width=width,
                **kwargs,
            )
            halo = shell[0].copy().scale(1.18).set_fill(opacity=0).set_stroke(
                ManimColor(resolved_fill),
                width=2.0,
                opacity=0.16,
            )
            return VGroup(halo, shell)

        stack = VGroup()
        if label_text:
            label_mob = self.fit_text(
                label_text.upper(),
                max_width=max(width * 1.38, 2.3),
                max_font_size=14,
                min_font_size=9,
                max_lines=2,
                color=self.theme_color("eyebrow_text"),
                weight=BOLD,
            )
            stack.add(label_mob)
        value_line = self.fit_text(
            " ".join(part for part in [value_text, unit_text] if part).strip(),
            max_width=max(width * 1.48, 2.4),
            max_font_size=26,
            min_font_size=13,
            max_lines=2,
            color=resolved_text_color,
            weight=BOLD,
        )
        stack.add(value_line)
        stack.arrange(DOWN, buff=0.14, aligned_edge=LEFT)

        shell_width = max(float(width), stack.width + 0.52)
        shell_height = max(0.72, stack.height + 0.38)
        shell = RoundedRectangle(corner_radius=0.2, width=shell_width, height=shell_height)
        shell.set_fill(ManimColor(resolved_fill), opacity=1.0)
        shell.set_stroke(width=0)
        stack.move_to(shell.get_center())
        halo = shell.copy().scale(1.16).set_fill(opacity=0).set_stroke(
            ManimColor(resolved_fill),
            width=2.0,
            opacity=0.16,
        )
        return VGroup(halo, shell, stack)

    def make_ribbon_label(
        self,
        text: str,
        *,
        max_width: float = 4.8,
        accent: str | None = None,
        text_color: str | None = None,
        color: str | None = None,
        **_: Any,
    ) -> VGroup:
        resolved_accent = accent or color or self.theme_color("accent")
        label = self.fit_text(
            text,
            max_width=max_width,
            max_font_size=28,
            min_font_size=15,
            max_lines=2,
            color=text_color or self.theme_color("text_primary"),
            weight=BOLD,
        )
        half_width = max(1.2, min(max_width / 2.0, max(float(label.width) * 0.58, 1.4)))
        line = Line(LEFT * half_width, RIGHT * half_width, color=ManimColor(resolved_accent), stroke_width=5)
        label.next_to(line, UP, buff=0.18)
        return VGroup(line, label)

    def _attach_drift(
        self,
        mob: Any,
        *,
        dx: float = 0.12,
        dy: float = 0.08,
        speed: float = 0.42,
        phase: float = 0.0,
    ) -> Any:
        anchor = mob.get_center().copy()
        setattr(mob, "_vex_drift_t", 0.0)

        def updater(target: Any, dt: float) -> None:
            target._vex_drift_t = getattr(target, "_vex_drift_t", 0.0) + dt * speed
            t = target._vex_drift_t + phase
            target.move_to(anchor + RIGHT * (math.sin(t) * dx) + UP * (math.cos(t * 0.83) * dy))

        mob.add_updater(updater)
        return mob

    def _attach_pulse(
        self,
        mob: Any,
        *,
        fill_amp: float = 0.018,
        stroke_amp: float = 0.02,
        speed: float = 0.76,
        phase: float = 0.0,
    ) -> Any:
        base_fill = float(mob.get_fill_opacity()) if hasattr(mob, "get_fill_opacity") else 0.0
        base_stroke = float(mob.get_stroke_opacity()) if hasattr(mob, "get_stroke_opacity") else 0.0
        setattr(mob, "_vex_pulse_t", 0.0)

        def updater(target: Any, dt: float) -> None:
            target._vex_pulse_t = getattr(target, "_vex_pulse_t", 0.0) + dt * speed
            wave = math.sin(target._vex_pulse_t + phase)
            if base_fill > 0.0:
                target.set_fill(opacity=max(0.0, min(1.0, base_fill + wave * fill_amp)))
            if base_stroke > 0.0:
                target.set_stroke(opacity=max(0.0, min(1.0, base_stroke + wave * stroke_amp)))

        mob.add_updater(updater)
        return mob

    def _attach_rotation_drift(self, mob: Any, *, speed: float = 0.08) -> Any:
        anchor = mob.get_center().copy()

        def updater(target: Any, dt: float) -> None:
            target.rotate(speed * dt, about_point=anchor)

        mob.add_updater(updater)
        return mob

    def camera_focus(
        self,
        target: Any,
        *,
        scale: float = 0.92,
        run_time: float = 0.7,
        zoom: float | None = None,
        duration: float | None = None,
        center: Any | None = None,
        **_: Any,
    ) -> Animation:
        if zoom is not None:
            try:
                zoom_value = float(zoom)
                if zoom_value > 0.01:
                    scale = 1.0 / zoom_value
            except (TypeError, ValueError):
                pass
        if duration is not None:
            try:
                run_time = float(duration)
            except (TypeError, ValueError):
                pass
        focus_target = center if center is not None else target
        desired_x, desired_y = self._focus_point(focus_target)
        guarded_x, guarded_y, guarded_scale = self._guarded_camera_focus(
            desired_x=desired_x,
            desired_y=desired_y,
            scale=scale,
        )
        if (
            abs(guarded_x - desired_x) > 1e-4
            or abs(guarded_y - desired_y) > 1e-4
            or abs(guarded_scale - scale) > 1e-4
        ):
            self._record_guardrail_action(
                "camera_focus_guardrail",
                "camera",
                requested_x=round(desired_x, 4),
                requested_y=round(desired_y, 4),
                applied_x=round(guarded_x, 4),
                applied_y=round(guarded_y, 4),
                requested_scale=round(scale, 4),
                applied_scale=round(guarded_scale, 4),
            )
        return self.camera.frame.animate.scale(guarded_scale).move_to(
            [guarded_x, guarded_y, 0.0]
        ).set_run_time(run_time)

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
        slot_id: str | None = None,
    ) -> Any:
        base = re.sub(r"[^a-zA-Z0-9_]+", "_", str(name or role or "group")).strip("_") or "group"
        count = self._layout_name_counts.get(base, 0) + 1
        self._layout_name_counts[base] = count
        unique_name = base if count == 1 else f"{base}_{count}"
        normalized_role = str(role or "group")
        lowered_name = unique_name.lower()
        if any(token in lowered_name for token in ("title", "headline", "header", "eyebrow")):
            normalized_role = "title"
        elif normalized_role == "metric":
            if any(token in lowered_name for token in ("title", "headline", "header")):
                normalized_role = "title"
            elif any(token in lowered_name for token in ("verdict", "takeaway", "lesson", "thesis")):
                normalized_role = "hero"
        entry = {
            "name": unique_name,
            "mob": mob,
            "role": normalized_role,
            "priority": ROLE_PRIORITIES.get(normalized_role, 30) if priority is None else int(priority),
            "allow_scale_down": bool(allow_scale_down),
            "slot_id": str(slot_id or ""),
            "avoid_safe_bottom": (
                normalized_role in {"title", "text", "label", "support", "quote"}
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
        self._attach_drift(left_glow, dx=0.18, dy=0.12, speed=0.34, phase=0.2)
        self._attach_pulse(left_glow, fill_amp=0.018, speed=0.72, phase=0.1)
        self._attach_drift(right_glow, dx=0.16, dy=0.1, speed=0.29, phase=1.4)
        self._attach_pulse(right_glow, fill_amp=0.02, speed=0.68, phase=1.0)
        self._attach_drift(top_wash, dx=0.1, dy=0.05, speed=0.22, phase=0.7)
        self._attach_pulse(top_wash, fill_amp=0.01, speed=0.55, phase=0.5)
        self._attach_drift(bottom_wash, dx=0.08, dy=0.06, speed=0.19, phase=1.7)
        self._attach_pulse(bottom_wash, fill_amp=0.012, speed=0.49, phase=1.3)
        layers.add(left_glow, right_glow, top_wash, bottom_wash)
        if motif == "grid":
            grid = VGroup()
            for x in range(-6, 7):
                grid.add(Line([x * 1.08, -4.2, 0], [x * 1.08, 4.2, 0], stroke_width=1, color=ManimColor(grid_color), stroke_opacity=0.14))
            for y in range(-4, 5):
                grid.add(Line([-7.2, y * 0.94, 0], [7.2, y * 0.94, 0], stroke_width=1, color=ManimColor(grid_color), stroke_opacity=0.14))
            self._attach_drift(grid, dx=0.06, dy=0.04, speed=0.18, phase=0.9)
            layers.add(grid)
        elif motif == "rings":
            rings = VGroup()
            for radius, opacity in ((3.7, 0.12), (2.7, 0.1), (1.75, 0.08)):
                ring = Circle(radius=radius).set_stroke(ManimColor(glow_color), width=1.6, opacity=opacity).set_fill(opacity=0)
                self._attach_pulse(ring, fill_amp=0.0, stroke_amp=0.018, speed=0.63 + radius * 0.02, phase=radius)
                rings.add(ring)
            rings.move_to(RIGHT * 3.9 + UP * 0.2)
            self._attach_rotation_drift(rings, speed=0.07)
            layers.add(rings)
        elif motif == "bands":
            bands = VGroup()
            for offset, color, opacity in ((-1.8, glow_color, 0.14), (0.0, accent_secondary, 0.12), (1.7, glow_color, 0.09)):
                band = RoundedRectangle(corner_radius=0.0, width=5.4, height=0.22).set_fill(ManimColor(color), opacity=opacity).set_stroke(width=0)
                band.rotate(-0.34)
                band.move_to(RIGHT * 3.3 + UP * offset)
                self._attach_drift(band, dx=0.1, dy=0.05, speed=0.26, phase=offset)
                self._attach_pulse(band, fill_amp=0.016, speed=0.58, phase=offset)
                bands.add(band)
            layers.add(bands)
        else:
            dots = VGroup()
            for index, (x, y, radius, opacity) in enumerate([(-4.8, 2.1, 0.06, 0.2), (-3.8, 1.35, 0.05, 0.15), (3.7, -1.2, 0.07, 0.18), (4.6, 1.7, 0.04, 0.14), (3.0, 2.3, 0.05, 0.12)]):
                dot = Circle(radius=radius).set_fill(ManimColor(glow_color if x < 0 else accent_secondary), opacity=opacity).set_stroke(width=0)
                dot.move_to(RIGHT * x + UP * y)
                self._attach_pulse(dot, fill_amp=0.03, speed=0.9 + index * 0.08, phase=index * 0.5)
                self._attach_drift(dot, dx=0.035, dy=0.028, speed=0.42 + index * 0.03, phase=index * 0.35)
                dots.add(dot)
            lines = VGroup(
                Line(LEFT * 4.8 + UP * 2.1, LEFT * 3.8 + UP * 1.35, stroke_width=1.2, color=ManimColor(glow_color), stroke_opacity=0.14),
                Line(RIGHT * 3.7 + DOWN * 1.2, RIGHT * 4.6 + UP * 1.7, stroke_width=1.2, color=ManimColor(accent_secondary), stroke_opacity=0.14),
            )
            for index, line in enumerate(lines):
                self._attach_pulse(line, fill_amp=0.0, stroke_amp=0.03, speed=0.62 + index * 0.08, phase=index * 0.9)
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

    def _entry_is_connector_like(self, entry: dict[str, Any]) -> bool:
        role = str(entry.get("role") or "")
        return role in {"connector", "motion_spine"} or self._is_connector_like(entry["mob"])

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
        try:
            font_size = getattr(mob, "font_size", None)
        except Exception:
            font_size = None
        if isinstance(font_size, (int, float)):
            try:
                value = float(font_size)
            except (TypeError, ValueError):
                value = None
            else:
                if math.isfinite(value) and value > 0.0:
                    return value
        sizes = [self._font_size(child) for child in getattr(mob, "submobjects", [])]
        sizes = [value for value in sizes if value is not None]
        return max(sizes) if sizes else None

    def _text_leaves(self, mob: Any) -> list[Any]:
        children = [child for child in getattr(mob, "submobjects", []) if child is not None]
        if mob.__class__.__name__ in TEXT_CLASS_MARKERS and float(getattr(mob, "width", 0.0) or 0.0) > 0.01 and float(getattr(mob, "height", 0.0) or 0.0) > 0.01:
            return [mob]
        leaves: list[Any] = []
        for child in children:
            leaves.extend(self._text_leaves(child))
        return leaves

    def _combined_bbox(self, mobs: list[Any]) -> dict[str, float] | None:
        if not mobs:
            return None
        left = min(float(mob.get_left()[0]) for mob in mobs)
        right = max(float(mob.get_right()[0]) for mob in mobs)
        top = max(float(mob.get_top()[1]) for mob in mobs)
        bottom = min(float(mob.get_bottom()[1]) for mob in mobs)
        width = max(right - left, 0.0)
        height = max(top - bottom, 0.0)
        return {
            "left": left,
            "right": right,
            "top": top,
            "bottom": bottom,
            "width": width,
            "height": height,
            "center_x": (left + right) / 2,
            "center_y": (top + bottom) / 2,
        }

    def _merged_box(self, boxes: list[dict[str, float]]) -> dict[str, float] | None:
        if not boxes:
            return None
        left = min(float(box["left"]) for box in boxes)
        right = max(float(box["right"]) for box in boxes)
        top = max(float(box["top"]) for box in boxes)
        bottom = min(float(box["bottom"]) for box in boxes)
        width = max(right - left, 0.0)
        height = max(top - bottom, 0.0)
        return {
            "left": left,
            "right": right,
            "top": top,
            "bottom": bottom,
            "width": width,
            "height": height,
            "center_x": (left + right) / 2,
            "center_y": (top + bottom) / 2,
        }

    def _text_bbox(self, mob: Any) -> dict[str, float] | None:
        return self._combined_bbox(self._text_leaves(mob))

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

    def _focus_point(self, target: Any) -> tuple[float, float]:
        if hasattr(target, "get_center"):
            center = target.get_center()
            return float(center[0]), float(center[1])
        try:
            return float(target[0]), float(target[1])
        except (TypeError, ValueError, IndexError, KeyError):
            return 0.0, 0.0

    def _camera_preserve_entries(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for entry in self._active_layout_entries():
            mob = entry["mob"]
            if float(getattr(mob, "width", 0.0) or 0.0) <= 0.01 or float(getattr(mob, "height", 0.0) or 0.0) <= 0.01:
                continue
            if self._is_connector_like(mob):
                continue
            role = str(entry.get("role") or "")
            if self._is_text_based(mob) or role in {"hero", "title", "before", "after", "metric", "support", "quote", "label", "footer"}:
                entries.append(entry)
        return entries

    def _camera_preserve_box(self, entry: dict[str, Any]) -> dict[str, float] | None:
        mob = entry["mob"]
        return self._text_bbox(mob) or self._bbox(mob)

    def _guarded_camera_focus(
        self,
        *,
        desired_x: float,
        desired_y: float,
        scale: float,
    ) -> tuple[float, float, float]:
        entries = self._camera_preserve_entries()
        if not entries:
            return desired_x, desired_y, float(scale)
        preserve_boxes = [box for box in (self._camera_preserve_box(entry) for entry in entries) if box is not None]
        union_box = self._merged_box(preserve_boxes)
        if union_box is None:
            return desired_x, desired_y, float(scale)

        frame_bounds = self._frame_bounds()
        base_width = float(frame_bounds["width"])
        base_height = float(frame_bounds["height"])
        guarded_scale = float(scale)
        required_width = union_box["width"] + base_width * 0.08
        required_height = union_box["height"] + base_height * 0.22
        guarded_scale = max(
            guarded_scale,
            required_width / max(base_width, 1e-6),
            required_height / max(base_height, 1e-6),
        )
        guarded_scale = min(max(guarded_scale, 0.82), 1.14)

        focus_width = base_width * guarded_scale
        focus_height = base_height * guarded_scale
        margin_x = max(focus_width * 0.04, 0.18)
        margin_top = max(focus_height * 0.05, 0.16)
        margin_bottom = max(focus_height * 0.17, 0.24)
        min_center_x = union_box["right"] + margin_x - focus_width / 2
        max_center_x = union_box["left"] - margin_x + focus_width / 2
        min_center_y = union_box["top"] + margin_top - focus_height / 2
        max_center_y = union_box["bottom"] - margin_bottom + focus_height / 2

        if min_center_x > max_center_x:
            guarded_x = union_box["center_x"]
        else:
            guarded_x = min(max(desired_x, min_center_x), max_center_x)
        if min_center_y > max_center_y:
            guarded_y = union_box["center_y"]
        else:
            guarded_y = min(max(desired_y, min_center_y), max_center_y)
        return guarded_x, guarded_y, guarded_scale

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
        if self._entry_is_connector_like(entry):
            return
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
        entries = [entry for entry in entries if not str(entry.get("slot_id") or "").strip()]
        if not entries:
            return
        titles = [entry for entry in entries if str(entry.get("role") or "") == "title"]
        charts = [entry for entry in entries if str(entry.get("role") or "") in {"chart", "structure", "system", "diagram"}]
        heroes = [entry for entry in entries if str(entry.get("role") or "") == "hero"]
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

        chart_boxes = [self._bbox(entry["mob"]) for entry in charts]
        if chart_boxes:
            chart_left = min(box["left"] for box in chart_boxes)
            chart_right = max(box["right"] for box in chart_boxes)
            chart_bottom = min(box["bottom"] for box in chart_boxes)
            chart_center_x = sum(box["center_x"] for box in chart_boxes) / len(chart_boxes)
        else:
            chart_left = chart_right = chart_bottom = chart_center_x = 0.0

        for hero in sorted(heroes, key=lambda item: int(item.get("priority") or 0), reverse=True):
            box = self._bbox(hero["mob"])
            target_y = max(frame_bounds["bottom"] + box["height"] / 2 + 0.34, chart_bottom - 0.28 - box["height"] / 2)
            target_x = 0.0
            if chart_boxes:
                chart_span = chart_right - chart_left
                if chart_span < frame_bounds["width"] * 0.48:
                    target_x = (
                        safe_bounds["right"] - box["width"] / 2
                        if chart_center_x <= 0.0
                        else safe_bounds["left"] + box["width"] / 2
                    )
            overlaps_chart_band = chart_boxes and box["top"] > chart_bottom - 0.08
            if overlaps_chart_band or abs(box["center_x"] - target_x) > 0.16 or abs(box["center_y"] - target_y) > 0.16:
                self._move_entry_to(hero, center_x=target_x, center_y=target_y, kind="anchor_hero")

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
        slot_id = str(entry.get("slot_id") or "").strip()
        if slot_id and role in {"diagram", "system", "connector", "motion_spine"}:
            return
        if slot_id:
            max_width = self.layout_slot(slot_id).inner_width
        else:
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
        panels = [entry for entry in entries if entry.get("role") == "panel" or self._is_panel_like(entry["mob"])]
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
                    text_entry["mob"].scale(max(scale_factor, 0.68))
                    text_entry["mob"].move_to(panel_entry["mob"].get_center())
                    self._record_guardrail_action(
                        "fit_text_inside_panel",
                        text_entry["name"],
                        panel=panel_entry["name"],
                        factor=round(max(scale_factor, 0.68), 4),
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
        padding = 0.18
        original_center = mover["mob"].get_center().copy()

        def apply_shift(horizontal: bool) -> tuple[float, float]:
            local_dx = 0.0
            local_dy = 0.0
            if horizontal:
                direction = 1.0 if mover_box["center_x"] >= anchor_box["center_x"] else -1.0
                local_dx = direction * (overlap_x + padding)
            else:
                direction = 1.0 if mover_box["center_y"] >= anchor_box["center_y"] else -1.0
                local_dy = direction * (overlap_y + padding)
            mover["mob"].shift(RIGHT * local_dx + UP * local_dy)
            if mover.get("allow_scale_down") and (abs(local_dx) > 0.6 or abs(local_dy) > 0.6):
                mover["mob"].scale(0.96)
                self._record_guardrail_action("scale_after_overlap", mover["name"], factor=0.96)
            self._clamp_inside_bounds(mover, safe_bounds=safe_bounds, frame_bounds=frame_bounds)
            return local_dx, local_dy

        def still_overlapping() -> bool:
            new_first_box = self._bbox(first["mob"])
            new_second_box = self._bbox(second["mob"])
            return (
                max(0.0, min(new_first_box["right"], new_second_box["right"]) - max(new_first_box["left"], new_second_box["left"])) > 0.0
                and max(0.0, min(new_first_box["top"], new_second_box["top"]) - max(new_first_box["bottom"], new_second_box["bottom"])) > 0.0
            )

        preferred_horizontal = overlap_x < overlap_y
        dx, dy = apply_shift(preferred_horizontal)
        if still_overlapping():
            mover["mob"].move_to(original_center)
            dx, dy = apply_shift(not preferred_horizontal)
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
                    if self._entry_is_connector_like(first) or self._entry_is_connector_like(second):
                        continue
                    if str(first.get("slot_id") or "").strip() or str(second.get("slot_id") or "").strip():
                        continue
                    if self._resolve_overlap(first, second, safe_bounds=safe_bounds, frame_bounds=frame_bounds):
                        moved = True
            if not moved:
                break

    def _entry_payload(self, entry: dict[str, Any]) -> dict[str, Any]:
        mob = entry["mob"]
        box = self._bbox(mob)
        text_box = self._text_bbox(mob)
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
            "slot_id": str(entry.get("slot_id") or ""),
        }
        if text_box is not None:
            payload.update(
                {
                    "text_left": round(text_box["left"], 4),
                    "text_right": round(text_box["right"], 4),
                    "text_top": round(text_box["top"], 4),
                    "text_bottom": round(text_box["bottom"], 4),
                    "text_width": round(text_box["width"], 4),
                    "text_height": round(text_box["height"], 4),
                }
            )
        return payload

    def _dump_layout_snapshot(self) -> None:
        layout_payload = self._layout_spec.to_dict() if getattr(self, "_layout_spec", None) is not None else {}
        if self._layout_spec_dump_path:
            try:
                with open(self._layout_spec_dump_path, "w", encoding="utf-8") as handle:
                    json.dump(layout_payload, handle, indent=2)
            except OSError:
                pass
        if not self._layout_dump_path:
            return
        try:
            payload = {
                "frame": self._frame_bounds(),
                "safe_bounds": self._safe_bounds(),
                "layout_spec": layout_payload,
                "registered_count": len(self._layout_registry),
                "registered": [self._entry_payload(entry) for entry in self._layout_registry],
                "top_level": [self._entry_payload(entry) for entry in self._top_level_entries()[:24]],
                "guardrail_actions": list(self._guardrail_actions[-80:]),
            }
            with open(self._layout_dump_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        except OSError:
            return
