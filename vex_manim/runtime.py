from __future__ import annotations

from typing import Any

from manim import (
    Animation,
    BOLD,
    Circle,
    CurvedArrow,
    DOWN,
    FadeIn,
    LEFT,
    Line,
    ManimColor,
    MEDIUM,
    MovingCameraScene,
    NORMAL,
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
        self.stage_background = self.apply_house_background(
            motif=str(self.spec.get("background_motif") or self.brief.get("background_motif") or "constellation"),
            add=True,
        )

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

    def camera_focus(self, target: Any, *, scale: float = 0.92, run_time: float = 0.7) -> Animation:
        return self.camera.frame.animate.scale(scale).move_to(target).set_run_time(run_time)

    def stagger_fade_in(self, items: list[Any], *, shift=UP * 0.12, lag_ratio: float = 0.12) -> Animation:
        from manim import LaggedStart

        return LaggedStart(*[FadeIn(item, shift=shift) for item in items], lag_ratio=lag_ratio)

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
