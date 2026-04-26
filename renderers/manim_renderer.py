from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from engine import probe_video
from renderers.base import RenderedAsset, RendererStatus, VisualRenderer, VisualRendererError


def _safe_scene_name(spec_id: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", spec_id).strip("_") or "auto_visual"
    return f"AutoVisual_{cleaned}"


def _theme_defaults(spec: dict[str, Any]) -> dict[str, str]:
    theme = dict(spec.get("theme") or {})
    defaults = {
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
    defaults.update({key: str(value) for key, value in theme.items() if value})
    return defaults


def _scene_script(scene_name: str, spec: dict[str, Any]) -> str:
    payload = dict(spec)
    payload["theme"] = _theme_defaults(spec)
    spec_json = json.dumps(payload, ensure_ascii=True)
    return f"""from __future__ import annotations

import json

from manim import *

SPEC = json.loads(r'''{spec_json}''')


def theme(name: str, fallback: str) -> str:
    return str(SPEC.get("theme", {{}}).get(name) or fallback)


def clamp_text(content: str, max_width: float, max_font_size: int, min_font_size: int, color: str, weight=BOLD, slant=NORMAL):
    cleaned = str(content or "").strip() or " "
    for size in range(max_font_size, min_font_size - 1, -4):
        text = Text(cleaned, font_size=size, color=ManimColor(color), weight=weight, slant=slant)
        if text.width <= max_width:
            return text
    return Text(cleaned, font_size=min_font_size, color=ManimColor(color), weight=weight, slant=slant)


def line_stack(lines, max_width: float, max_font_size: int, min_font_size: int, color: str, weight=MEDIUM, aligned_edge=LEFT):
    cleaned = [str(line).strip() for line in (lines or []) if str(line).strip()]
    if not cleaned:
        cleaned = [" "]
    group = VGroup(
        *[
            clamp_text(
                line,
                max_width=max_width,
                max_font_size=max_font_size,
                min_font_size=min_font_size,
                color=color,
                weight=weight,
            )
            for line in cleaned
        ]
    )
    group.arrange(DOWN, buff=0.22, aligned_edge=aligned_edge)
    return group


def pill(text: str, *, fill: str, text_color: str, width: float | None = None):
    label = clamp_text(text.upper(), max_width=4.2 if width is None else max(width - 0.36, 1.0), max_font_size=24, min_font_size=14, color=text_color, weight=BOLD)
    shell = RoundedRectangle(corner_radius=0.18, width=max(label.width + 0.44, width or 1.8), height=max(label.height + 0.24, 0.52))
    shell.set_fill(ManimColor(fill), opacity=1.0)
    shell.set_stroke(width=0)
    label.move_to(shell.get_center())
    return VGroup(shell, label)


def glass_card(width: float, height: float, *, fill: str, stroke: str, radius: float = 0.22):
    outer = RoundedRectangle(corner_radius=radius, width=width, height=height)
    outer.set_fill(ManimColor(fill), opacity=0.95)
    outer.set_stroke(ManimColor(stroke), width=2.4, opacity=0.95)
    inner = outer.copy()
    inner.scale(0.985)
    inner.set_stroke(ManimColor(fill), width=1.2, opacity=0.4)
    return VGroup(outer, inner)


def stage_layers(motif: str, glow_color: str, accent_secondary: str, grid_color: str):
    layers = VGroup()
    left_glow = Circle(radius=3.1).set_fill(ManimColor(glow_color), opacity=0.12).set_stroke(width=0).move_to(LEFT * 4.4 + UP * 1.4)
    right_glow = Circle(radius=3.5).set_fill(ManimColor(accent_secondary), opacity=0.11).set_stroke(width=0).move_to(RIGHT * 4.5 + DOWN * 1.6)
    top_wash = Rectangle(width=14.6, height=2.4).set_fill(ManimColor(glow_color), opacity=0.06).set_stroke(width=0).move_to(UP * 3.0)
    bottom_wash = Rectangle(width=14.6, height=2.2).set_fill(ManimColor(accent_secondary), opacity=0.05).set_stroke(width=0).move_to(DOWN * 3.0)
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
            band = Rectangle(width=5.4, height=0.22).set_fill(ManimColor(color), opacity=opacity).set_stroke(width=0)
            band.rotate(-0.34)
            band.move_to(RIGHT * 3.3 + UP * offset)
            bands.add(band)
        layers.add(bands)
    elif motif == "beams":
        beams = VGroup()
        for offset, color, opacity in ((-2.2, accent_secondary, 0.11), (-0.7, glow_color, 0.1), (1.1, accent_secondary, 0.08)):
            beam = Rectangle(width=6.2, height=0.24).set_fill(ManimColor(color), opacity=opacity).set_stroke(width=0)
            beam.rotate(-0.55)
            beam.move_to(LEFT * 2.8 + UP * offset)
            beams.add(beam)
        layers.add(beams)
    else:
        dots = VGroup()
        dot_specs = [(-4.8, 2.1, 0.06, 0.2), (-3.8, 1.35, 0.05, 0.15), (3.7, -1.2, 0.07, 0.18), (4.6, 1.7, 0.04, 0.14), (3.0, 2.3, 0.05, 0.12)]
        for x, y, radius, opacity in dot_specs:
            dot = Circle(radius=radius).set_fill(ManimColor(glow_color if x < 0 else accent_secondary), opacity=opacity).set_stroke(width=0)
            dot.move_to(RIGHT * x + UP * y)
            dots.add(dot)
        lines = VGroup(
            Line(LEFT * 4.8 + UP * 2.1, LEFT * 3.8 + UP * 1.35, stroke_width=1.2, color=ManimColor(glow_color), stroke_opacity=0.14),
            Line(RIGHT * 3.7 + DOWN * 1.2, RIGHT * 4.6 + UP * 1.7, stroke_width=1.2, color=ManimColor(accent_secondary), stroke_opacity=0.14),
        )
        layers.add(dots, lines)
    return layers


def support_card(text: str, *, width: float, fill: str, stroke: str, color: str):
    shell = glass_card(width, 1.24, fill=fill, stroke=stroke, radius=0.2)
    label = clamp_text(text, max_width=width - 0.58, max_font_size=22, min_font_size=16, color=color, weight=MEDIUM)
    label.move_to(shell[0].get_center())
    return VGroup(shell, label)


class {scene_name}(Scene):
    def construct(self):
        self.camera.background_color = ManimColor(theme("background", "#0B1020"))
        duration = max(float(SPEC.get("duration") or 2.0), 1.0)
        template = str(SPEC.get("template") or "quote_focus")
        intro = min(max(duration * 0.22, 0.35), 0.9)
        accent = min(max(duration * 0.24, 0.35), 1.0)
        reveal = min(max(duration * 0.22, 0.35), 0.9)
        settle = max(duration - intro - accent - reveal, 0.12)

        primary = theme("text_primary", "#F8FAFC")
        secondary = theme("text_secondary", "#CBD5E1")
        panel_fill = theme("panel_fill", "#13203A")
        panel_stroke = theme("panel_stroke", "#60A5FA")
        accent_color = theme("accent", "#F59E0B")
        accent_secondary = theme("accent_secondary", "#38BDF8")
        glow_color = theme("glow", accent_secondary)
        eyebrow_fill = theme("eyebrow_fill", panel_fill)
        eyebrow_text = theme("eyebrow_text", "#E0F2FE")
        grid_color = theme("grid", panel_stroke)
        background_motif = str(SPEC.get("background_motif") or "constellation")

        stage = stage_layers(background_motif, glow_color, accent_secondary, grid_color)
        self.add(stage)

        eyebrow_value = str(SPEC.get("eyebrow") or "").strip()
        headline_value = str(SPEC.get("headline") or "").strip()
        deck_value = str(SPEC.get("deck") or "").strip()
        header_headline_width = 8.1 if template in {"metric_callout", "stat_grid", "comparison_split"} else 9.2
        header_deck_width = 6.8 if template in {"metric_callout", "stat_grid", "comparison_split"} else 8.6
        header_headline_size = 48 if template in {"metric_callout", "stat_grid", "comparison_split"} else 52
        header = VGroup()
        if eyebrow_value:
            header.add(pill(eyebrow_value, fill=eyebrow_fill, text_color=eyebrow_text))
        if headline_value:
            header.add(clamp_text(headline_value, max_width=header_headline_width, max_font_size=header_headline_size, min_font_size=28, color=primary))
        if deck_value:
            header.add(clamp_text(deck_value, max_width=header_deck_width, max_font_size=23, min_font_size=16, color=secondary, weight=MEDIUM))
        if len(header) > 0:
            header.arrange(DOWN, aligned_edge=LEFT, buff=0.18)
            header.to_edge(LEFT, buff=0.78)
            header.to_edge(UP, buff=0.62)
        header_marker = Line(header.get_corner(DL) + LEFT * 0.18, header.get_corner(UL) + LEFT * 0.18, color=ManimColor(accent_color), stroke_width=5, stroke_opacity=0.9) if len(header) > 0 else VGroup()

        if template == "metric_callout":
            hero = glass_card(5.55, 4.15, fill=panel_fill, stroke=panel_stroke, radius=0.28)
            hero.to_edge(LEFT, buff=1.0)
            hero.shift(DOWN * 1.32 + RIGHT * 0.08)
            value = clamp_text(str(SPEC.get("emphasis_text") or SPEC.get("headline") or "Key Point"), max_width=4.3, max_font_size=74, min_font_size=38, color=primary)
            value.move_to(hero[0].get_center() + LEFT * 0.16 + UP * 0.18)
            kicker = clamp_text("MEASURABLE CHANGE", max_width=3.1, max_font_size=17, min_font_size=12, color=accent_secondary, weight=BOLD)
            kicker.move_to(hero[0].get_top() + DOWN * 0.64 + LEFT * 0.54)
            support_terms = [str(item).strip() for item in (SPEC.get("supporting_lines") or []) if str(item).strip()]
            support_cards = VGroup(*[
                support_card(item, width=3.72, fill=panel_fill, stroke=accent_secondary, color=secondary)
                for item in support_terms[:3]
            ])
            if len(support_cards) == 0:
                support_cards.add(support_card(str(SPEC.get("deck") or "Sharper context"), width=3.72, fill=panel_fill, stroke=accent_secondary, color=secondary))
            support_cards.arrange(DOWN, buff=0.28)
            support_label = pill("WHY IT LANDS", fill=eyebrow_fill, text_color=eyebrow_text, width=2.54)
            support_stack = VGroup(support_label, support_cards).arrange(DOWN, buff=0.26, aligned_edge=LEFT)
            support_stack.move_to(RIGHT * 3.85 + DOWN * 1.02)
            support_rail = Line(support_stack.get_corner(UL) + LEFT * 0.18, support_stack.get_corner(DL) + LEFT * 0.18, color=ManimColor(accent_secondary), stroke_width=3, stroke_opacity=0.55)
            accent_rule = Line(hero[0].get_bottom() + LEFT * 1.85 + UP * 0.48, hero[0].get_bottom() + RIGHT * 1.85 + UP * 0.48, color=ManimColor(accent_color), stroke_width=5, stroke_opacity=0.86)
            dots = VGroup(*[
                Dot(radius=0.05 + idx * 0.008, color=ManimColor(accent_secondary)).move_to(hero[0].get_bottom() + UP * 0.82 + RIGHT * (-1.42 + idx * 0.48))
                for idx in range(5)
            ])
            self.play(FadeIn(stage, scale=1.02), run_time=0.26)
            if len(header) > 0:
                self.play(FadeIn(header_marker), FadeIn(header, shift=RIGHT * 0.16), run_time=intro)
            self.play(DrawBorderThenFill(hero[0]), FadeIn(hero[1], scale=0.98), run_time=accent)
            self.play(FadeIn(kicker, shift=UP * 0.08), Write(value), Create(accent_rule), LaggedStart(*[FadeIn(dot, scale=0.85) for dot in dots], lag_ratio=0.08), run_time=reveal)
            self.play(FadeIn(support_rail), FadeIn(support_label, shift=LEFT * 0.12), LaggedStart(*[FadeIn(card, shift=LEFT * 0.15) for card in support_cards], lag_ratio=0.12), run_time=0.58)
            self.wait(settle)
            return

        if template == "keyword_stack":
            keywords = [str(item).strip() for item in (SPEC.get("keywords") or []) if str(item).strip()][:4]
            cards = VGroup()
            widths = [6.1, 7.3, 6.7, 7.7]
            offsets = [0.0, 0.42, -0.3, 0.24]
            for index, keyword in enumerate(keywords or [str(SPEC.get("emphasis_text") or "Key idea")], start=1):
                card_shell = glass_card(widths[(index - 1) % len(widths)], 1.18, fill=panel_fill, stroke=panel_stroke, radius=0.22)
                label = clamp_text(keyword, max_width=card_shell[0].width - 0.7, max_font_size=32, min_font_size=20, color=primary)
                label.move_to(card_shell[0].get_center())
                chip = pill(str(index).zfill(2), fill=accent_color, text_color=theme("background", "#08101E"), width=0.9)
                chip.move_to(card_shell[0].get_left() + RIGHT * 0.56 + UP * 0.26)
                card = VGroup(card_shell, label, chip)
                card.shift(RIGHT * offsets[(index - 1) % len(offsets)])
                cards.add(card)
            cards.arrange(DOWN, buff=0.26, aligned_edge=LEFT)
            cards.shift(DOWN * 1.24 + RIGHT * 0.22)
            footer = clamp_text(str(SPEC.get("footer_text") or "").strip(), max_width=9.6, max_font_size=24, min_font_size=16, color=secondary, weight=MEDIUM)
            footer.next_to(cards, DOWN, buff=0.42, aligned_edge=LEFT)
            self.play(FadeIn(stage, scale=1.02), run_time=0.26)
            if len(header) > 0:
                self.play(FadeIn(header_marker), FadeIn(header, shift=RIGHT * 0.16), run_time=intro)
            self.play(LaggedStart(*[FadeIn(card, shift=UP * 0.18) for card in cards], lag_ratio=0.12), run_time=accent + reveal)
            if str(SPEC.get("footer_text") or "").strip():
                self.play(FadeIn(footer, shift=UP * 0.08), run_time=0.3)
            self.wait(settle)
            return

        if template == "timeline_steps":
            steps = [str(item).strip() for item in (SPEC.get("steps") or []) if str(item).strip()][:4]
            steps = steps or [str(SPEC.get("emphasis_text") or "Step one"), "Step two", "Step three"]
            baseline = Line(LEFT * 5.2 + DOWN * 1.8, RIGHT * 5.2 + DOWN * 1.8, color=ManimColor(accent_secondary), stroke_width=4, stroke_opacity=0.46)
            positions = [LEFT * 4.2, LEFT * 1.4, RIGHT * 1.4, RIGHT * 4.2]
            cards = VGroup()
            dots = VGroup()
            for index, step in enumerate(steps, start=1):
                anchor = positions[index - 1] + DOWN * 1.8
                dot = Dot(anchor, radius=0.1, color=ManimColor(accent_color))
                dots.add(dot)
                stem = Line(anchor, anchor + UP * 1.0, color=ManimColor(accent_secondary), stroke_width=2.5, stroke_opacity=0.82)
                shell = glass_card(2.42, 1.72, fill=panel_fill, stroke=panel_stroke, radius=0.22)
                shell.move_to(anchor + UP * 1.96)
                number = pill(str(index), fill=accent_color, text_color=theme("background", "#08101E"), width=0.84)
                number.move_to(shell[0].get_top() + DOWN * 0.32 + LEFT * 0.62)
                label = clamp_text(step, max_width=1.78, max_font_size=23, min_font_size=16, color=primary)
                label.move_to(shell[0].get_center() + DOWN * 0.05)
                cards.add(VGroup(stem, shell, number, label))
            self.play(FadeIn(stage, scale=1.02), run_time=0.26)
            if len(header) > 0:
                self.play(FadeIn(header_marker), FadeIn(header, shift=RIGHT * 0.16), run_time=intro)
            self.play(Create(baseline), LaggedStart(*[FadeIn(dot, scale=0.82) for dot in dots], lag_ratio=0.07), run_time=accent * 0.82)
            self.play(LaggedStart(*[FadeIn(card, shift=UP * 0.14) for card in cards], lag_ratio=0.1), run_time=max(reveal, 0.48))
            self.wait(settle)
            return

        if template == "comparison_split":
            left_box = glass_card(4.58, 4.38, fill=panel_fill, stroke=panel_stroke, radius=0.24)
            right_box = glass_card(4.58, 4.38, fill=panel_fill, stroke=accent_secondary, radius=0.24)
            left_group = VGroup(
                pill(str(SPEC.get("left_label") or "Before"), fill=eyebrow_fill, text_color=eyebrow_text),
                clamp_text(str(SPEC.get("left_detail") or ""), max_width=3.35, max_font_size=28, min_font_size=18, color=primary, weight=MEDIUM),
            ).arrange(DOWN, buff=0.32)
            left_group.move_to(left_box[0].get_center())
            right_group = VGroup(
                pill(str(SPEC.get("right_label") or "After"), fill=accent_color, text_color=theme("background", "#08101E")),
                clamp_text(str(SPEC.get("right_detail") or ""), max_width=3.35, max_font_size=28, min_font_size=18, color=primary, weight=MEDIUM),
            ).arrange(DOWN, buff=0.32)
            right_group.move_to(right_box[0].get_center())
            left_panel = VGroup(left_box, left_group).shift(LEFT * 2.8 + DOWN * 0.05 + UP * 0.22)
            right_panel = VGroup(right_box, right_group).shift(RIGHT * 2.8 + DOWN * 0.05 + DOWN * 0.22)
            bridge = CurvedArrow(left_panel.get_right() + RIGHT * 0.08, right_panel.get_left() + LEFT * 0.08, angle=-0.18, color=ManimColor(accent_color), stroke_width=5)
            versus = pill("SHIFT", fill=accent_secondary, text_color=theme("background", "#08101E"), width=1.4)
            versus.move_to((left_panel.get_right() + right_panel.get_left()) / 2 + UP * 0.28)
            self.play(FadeIn(stage, scale=1.02), run_time=0.26)
            if len(header) > 0:
                self.play(FadeIn(header_marker), FadeIn(header, shift=RIGHT * 0.16), run_time=intro)
            self.play(FadeIn(left_panel, shift=RIGHT * 0.16), FadeIn(right_panel, shift=LEFT * 0.16), run_time=accent)
            self.play(Create(bridge), FadeIn(versus, scale=0.92), run_time=reveal)
            self.wait(settle)
            return

        if template == "system_flow":
            steps = [str(item).strip() for item in (SPEC.get("steps") or []) if str(item).strip()][:4]
            steps = steps or [str(SPEC.get("headline") or "Input"), str(SPEC.get("emphasis_text") or "Process"), "Output"]
            nodes = VGroup()
            for index, step in enumerate(steps, start=1):
                circle = Circle(radius=0.8)
                circle.set_fill(ManimColor(panel_fill), opacity=1.0)
                circle.set_stroke(ManimColor(panel_stroke), width=3)
                halo = circle.copy().scale(1.18).set_stroke(ManimColor(glow_color), width=2, opacity=0.18).set_fill(opacity=0)
                number = Text(str(index), font_size=22, color=ManimColor(accent_color), weight=BOLD)
                number.next_to(circle.get_top(), DOWN, buff=0.24)
                label = clamp_text(step, max_width=1.72, max_font_size=24, min_font_size=16, color=primary)
                label.move_to(circle.get_center() + DOWN * 0.08)
                nodes.add(VGroup(halo, circle, number, label))
            nodes.arrange(RIGHT, buff=0.9)
            connectors = VGroup()
            for index in range(len(nodes) - 1):
                arrow = CurvedArrow(nodes[index].get_right() + RIGHT * 0.12, nodes[index + 1].get_left() + LEFT * 0.12, angle=-0.24, color=ManimColor(accent_secondary), stroke_width=5)
                connectors.add(arrow)
            layout = VGroup(nodes, connectors)
            layout.move_to(ORIGIN + DOWN * 0.1)
            footer = clamp_text(str(SPEC.get("footer_text") or "").strip(), max_width=9.2, max_font_size=22, min_font_size=16, color=secondary, weight=MEDIUM)
            footer.next_to(layout, DOWN, buff=0.52)
            self.play(FadeIn(stage, scale=1.02), run_time=0.26)
            if len(header) > 0:
                self.play(FadeIn(header_marker), FadeIn(header, shift=RIGHT * 0.16), run_time=intro)
            self.play(LaggedStart(*[GrowFromCenter(node) for node in nodes], lag_ratio=0.12), run_time=accent)
            if len(connectors) > 0:
                self.play(LaggedStart(*[Create(connector) for connector in connectors], lag_ratio=0.1), run_time=reveal)
            if str(SPEC.get("footer_text") or "").strip():
                self.play(FadeIn(footer, shift=UP * 0.08), run_time=0.3)
            self.wait(settle)
            return

        if template == "stat_grid":
            metrics = [str(SPEC.get("emphasis_text") or "Key stat").strip()]
            metrics.extend([str(item).strip() for item in (SPEC.get("supporting_lines") or []) if str(item).strip()][:3])
            keywords = [str(item).strip() for item in (SPEC.get("keywords") or []) if str(item).strip()]
            while len(metrics) < 4:
                metrics.append(keywords[len(metrics) - 1] if len(keywords) >= len(metrics) else "Insight")
            hero = glass_card(4.6, 3.7, fill=panel_fill, stroke=accent_color, radius=0.24)
            hero.to_edge(LEFT, buff=0.92)
            hero.shift(DOWN * 0.28)
            hero_value = clamp_text(metrics[0], max_width=3.68, max_font_size=40, min_font_size=22, color=primary)
            hero_value.move_to(hero[0].get_center())
            side_cards = VGroup()
            for metric in metrics[1:4]:
                shell = glass_card(3.55, 1.08, fill=panel_fill, stroke=panel_stroke, radius=0.2)
                label = clamp_text(metric, max_width=2.9, max_font_size=22, min_font_size=16, color=secondary, weight=MEDIUM)
                label.move_to(shell[0].get_center())
                side_cards.add(VGroup(shell, label))
            side_cards.arrange(DOWN, buff=0.22)
            side_cards.to_edge(RIGHT, buff=0.98)
            side_cards.shift(UP * 0.1)
            bars = VGroup(*[
                Rectangle(width=0.58, height=0.18 + idx * 0.14).set_fill(ManimColor(accent_secondary if idx % 2 else accent_color), opacity=0.9).set_stroke(width=0)
                for idx in range(6)
            ])
            bars.arrange(RIGHT, buff=0.12)
            bars.next_to(hero, DOWN, buff=0.34, aligned_edge=LEFT)
            self.play(FadeIn(stage, scale=1.02), run_time=0.26)
            if len(header) > 0:
                self.play(FadeIn(header_marker), FadeIn(header, shift=RIGHT * 0.16), run_time=intro)
            self.play(DrawBorderThenFill(hero[0]), FadeIn(hero[1], scale=0.98), Write(hero_value), run_time=accent)
            self.play(LaggedStart(*[FadeIn(card, shift=LEFT * 0.14) for card in side_cards], lag_ratio=0.1), LaggedStart(*[GrowFromEdge(bar, DOWN) for bar in bars], lag_ratio=0.06), run_time=reveal)
            self.wait(settle)
            return

        quote = clamp_text(str(SPEC.get("quote_text") or SPEC.get("emphasis_text") or SPEC.get("headline") or "Key quote"), max_width=8.8, max_font_size=58, min_font_size=28, color=primary)
        quote.to_edge(LEFT, buff=1.2)
        quote.shift(DOWN * 0.12)
        left_mark = Text('"', font_size=116, color=ManimColor(accent_color), weight=BOLD)
        left_mark.next_to(quote, LEFT, buff=0.22)
        accent_column = Rectangle(width=0.16, height=3.65).set_fill(ManimColor(accent_secondary), opacity=1.0).set_stroke(width=0)
        accent_column.move_to(LEFT * 5.92 + DOWN * 0.02)
        footer = clamp_text(str(SPEC.get("footer_text") or "").strip(), max_width=8.8, max_font_size=24, min_font_size=16, color=secondary, weight=MEDIUM)
        footer.next_to(quote, DOWN, buff=0.38, aligned_edge=LEFT)
        self.play(FadeIn(stage, scale=1.02), run_time=0.26)
        if len(header) > 0:
            self.play(FadeIn(header_marker), FadeIn(header, shift=RIGHT * 0.16), run_time=intro)
        self.play(FadeIn(accent_column, scale=0.95), FadeIn(left_mark, shift=RIGHT * 0.12), run_time=accent * 0.65)
        self.play(Write(quote), run_time=reveal)
        if str(SPEC.get("footer_text") or "").strip():
            self.play(FadeIn(footer, shift=UP * 0.08), run_time=accent * 0.4)
        self.wait(settle)
"""


class ManimRenderer(VisualRenderer):
    name = "manim"
    supported_templates = {
        "metric_callout",
        "keyword_stack",
        "timeline_steps",
        "comparison_split",
        "quote_focus",
        "system_flow",
        "stat_grid",
    }

    def availability(self) -> RendererStatus:
        if importlib.util.find_spec("manim") is None:
            return RendererStatus(False, "Manim is not installed in the current Python environment.")
        return RendererStatus(True, "")

    def score_spec(self, spec: dict[str, Any]) -> float:
        if not self.supports(spec):
            return -1.0
        template = str(spec.get("template") or "")
        visual_hint = str(spec.get("visual_type_hint") or "")
        composition = str(spec.get("composition_mode") or "")
        importance = float(spec.get("importance") or 0.5)
        score = 0.86
        if template in {"timeline_steps", "system_flow", "comparison_split", "stat_grid"}:
            score += 0.12
        if visual_hint in {"data_graphic", "process"}:
            score += 0.1
        if visual_hint == "product_ui":
            score -= 0.08
        if composition == "replace":
            score += 0.08
        score += importance * 0.04
        return round(score, 3)

    def render(
        self,
        spec: dict[str, Any],
        render_root: Path,
        width: int,
        height: int,
        fps: float,
    ) -> RenderedAsset:
        status = self.availability()
        if not status.available:
            raise VisualRendererError(status.reason)
        spec_id = str(spec.get("visual_id") or spec.get("id") or "visual")
        job_dir = render_root / spec_id
        job_dir.mkdir(parents=True, exist_ok=True)
        scene_name = _safe_scene_name(spec_id)
        script_path = job_dir / "scene.py"
        config_path = job_dir / "manim.cfg"
        media_dir = job_dir / "media"
        script_path.write_text(_scene_script(scene_name, spec), encoding="utf-8")
        config_path.write_text(
            "\n".join(
                [
                    "[CLI]",
                    f"media_dir = {media_dir.as_posix()}",
                    f"output_file = {scene_name}",
                    f"pixel_width = {width}",
                    f"pixel_height = {height}",
                    f"frame_rate = {max(15, int(round(fps)))}",
                    "verbosity = WARNING",
                    "progress_bar = none",
                    "disable_caching = True",
                    "write_to_movie = True",
                ]
            ),
            encoding="utf-8",
        )
        command = [
            sys.executable,
            "-m",
            "manim",
            "render",
            "--config_file",
            str(config_path),
            str(script_path),
            scene_name,
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            stderr = (result.stderr or result.stdout or "").strip()
            if "No module named manim" in stderr:
                raise VisualRendererError(
                    "Manim is not installed in the current Python environment. Install it with `pip install manim`."
                )
            raise VisualRendererError(f"Manim render failed for {spec_id}: {stderr}")
        candidates = [
            path
            for path in media_dir.rglob(f"{scene_name}.mp4")
            if "partial_movie_files" not in path.parts
        ]
        if not candidates:
            raise VisualRendererError(f"Manim render completed but no output video was found for {spec_id}.")
        asset_path = max(candidates, key=lambda path: path.stat().st_mtime)
        metadata = probe_video(str(asset_path))
        return RenderedAsset(
            asset_path=str(asset_path),
            width=int(metadata.get("width") or width),
            height=int(metadata.get("height") or height),
            duration_sec=float(metadata.get("duration_sec") or 0.0),
            renderer=self.name,
            job_dir=str(job_dir),
            script_path=str(script_path),
        )
