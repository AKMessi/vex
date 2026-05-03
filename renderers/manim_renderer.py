from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import config
from engine import probe_video
from renderers.base import RenderedAsset, RendererStatus, VisualRenderer, VisualRendererError
from vex_manim.blueprint import build_scene_blueprints
from vex_manim.briefs import build_scene_brief
from vex_manim.director import (
    build_deterministic_execution_plan,
    request_scene_candidate,
    request_scene_execution_plan,
    write_generation_report,
)
from vex_manim.layout_qa import analyze_layout_snapshot, load_layout_snapshot
from vex_manim.premium_fallback import run_premium_blueprint_scene
from vex_manim.qa import analyze_preview, evaluate_generated_scene_quality, extract_preview_frames
from vex_manim.scene_library import retrieve_scene_examples
from vex_manim.validator import CodeProfile, ValidationReport, profile_scene_code, validate_generated_scene_code
from vex_manim.visual_ir import (
    StoryboardCritique,
    StoryboardFrame,
    VisualExplanationIR,
    build_storyboard_frames,
    build_visual_explanation_ir,
    critique_storyboard,
    storyboard_prompt_block,
)


MAX_GENERATION_ATTEMPTS = 2
_LATEX_RUNTIME_READY_CACHE: bool | None = None
PREMIUM_GENERATED_TEMPLATES = {
    "data_journey",
    "signal_network",
    "kinetic_route",
    "spotlight_compare",
    "interface_cascade",
    "ribbon_quote",
}
FAST_TEMPLATE_TEMPLATES = {
    "metric_callout",
    "keyword_stack",
    "timeline_steps",
    "comparison_split",
    "quote_focus",
    "system_flow",
    "stat_grid",
}
LEGACY_TEMPLATE_ALIASES = {
    "data_journey": "metric_callout",
    "signal_network": "system_flow",
    "kinetic_route": "timeline_steps",
    "spotlight_compare": "comparison_split",
    "interface_cascade": "comparison_split",
    "ribbon_quote": "quote_focus",
}


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


def _legacy_scene_script(scene_name: str, spec: dict[str, Any]) -> str:
    payload = dict(spec)
    payload["theme"] = _theme_defaults(spec)
    spec_json = json.dumps(payload, ensure_ascii=True)
    return f"""from __future__ import annotations

import json
import re

from manim import *

SPEC = json.loads(r'''{spec_json}''')


def theme(name: str, fallback: str) -> str:
    return str(SPEC.get("theme", {{}}).get(name) or fallback)


def render_text_candidate(content: str, size: int, color: str, weight=BOLD, slant=NORMAL):
    return Text(content, font_size=size, color=ManimColor(color), weight=weight, slant=slant)


def wrap_variant(content: str, max_width: float, size: int, color: str, weight=BOLD, slant=NORMAL, max_lines: int = 4):
    words = [word for word in re.sub(r"\\s+", " ", str(content or "")).strip().split(" ") if word]
    if len(words) <= 1:
        return " ".join(words) or " "
    original = " ".join(words)
    best_variant = original
    best_overflow = max(render_text_candidate(original, size, color, weight=weight, slant=slant).width - max_width, 0.0)
    for line_limit in range(2, max_lines + 1):
        lines = []
        current = []
        success = True
        for word in words:
            tentative = " ".join([*current, word]).strip()
            if current and render_text_candidate(tentative, size, color, weight=weight, slant=slant).width > max_width:
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
        candidate_variant = "\\n".join(lines)
        candidate = render_text_candidate(candidate_variant, size, color, weight=weight, slant=slant)
        overflow = max(candidate.width - max_width, 0.0)
        if overflow <= 0.01:
            return candidate_variant
        if overflow < best_overflow:
            best_variant = candidate_variant
            best_overflow = overflow
    return best_variant


def clamp_text(content: str, max_width: float, max_font_size: int, min_font_size: int, color: str, weight=BOLD, slant=NORMAL):
    cleaned = re.sub(r"\\s+", " ", str(content or "")).strip() or " "
    for size in range(max_font_size, min_font_size - 1, -4):
        text = render_text_candidate(cleaned, size, color, weight=weight, slant=slant)
        if text.width <= max_width:
            return text
        wrapped = wrap_variant(cleaned, max_width, size, color, weight=weight, slant=slant)
        wrapped_text = render_text_candidate(wrapped, size, color, weight=weight, slant=slant)
        if wrapped_text.width <= max_width:
            return wrapped_text
    final_variant = wrap_variant(cleaned, max_width, min_font_size, color, weight=weight, slant=slant)
    return render_text_candidate(final_variant, min_font_size, color, weight=weight, slant=slant)


def compact_metric_value(emphasis: str, headline: str):
    emphasis_value = str(emphasis or "").strip()
    if emphasis_value and len(emphasis_value.split()) <= 3 and len(emphasis_value) <= 18:
        return emphasis_value
    numeric_phrase = re.search(r"(\\b\\d+(?:\\.\\d+)?\\s*(?:x|%|h|hr|hrs|hour|hours|min|mins|minutes|sec|seconds|pg|pages?)\\b)", str(headline or ""), re.IGNORECASE)
    if numeric_phrase:
        return numeric_phrase.group(1).strip()
    numeric_value = re.search(r"\\b\\d+(?:\\.\\d+)?\\b", str(headline or ""))
    if numeric_value:
        return numeric_value.group(0).strip()
    words = [word for word in re.sub(r"\\s+", " ", str(headline or "")).strip().split(" ") if word]
    if not words:
        return "Key Point"
    return " ".join(words[: min(len(words), 3)])


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
        template = {{"data_journey": "metric_callout", "signal_network": "system_flow", "kinetic_route": "timeline_steps", "spotlight_compare": "comparison_split", "interface_cascade": "comparison_split", "ribbon_quote": "quote_focus"}}.get(template, template)
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
            value_source = compact_metric_value(str(SPEC.get("emphasis_text") or ""), str(SPEC.get("headline") or ""))
            value = clamp_text(value_source, max_width=4.3, max_font_size=74, min_font_size=38, color=primary)
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
            self.play(FadeIn(kicker, shift=UP * 0.08), FadeIn(value, shift=UP * 0.12, scale=0.96), Create(accent_rule), LaggedStart(*[FadeIn(dot, scale=0.85) for dot in dots], lag_ratio=0.08), run_time=reveal)
            self.play(FadeIn(support_rail), FadeIn(support_label, shift=LEFT * 0.12), LaggedStart(*[FadeIn(card, shift=LEFT * 0.15) for card in support_cards], lag_ratio=0.12), run_time=0.58)
            self.wait(settle)
            return

        if template == "keyword_stack":
            keywords = [str(item).strip() for item in (SPEC.get("keywords") or []) if str(item).strip()][:4]
            keywords = keywords or [str(SPEC.get("emphasis_text") or "Key idea")]
            node_positions = [
                LEFT * 4.8 + DOWN * 1.9,
                LEFT * 1.9 + DOWN * 1.0,
                RIGHT * 1.15 + DOWN * 1.62,
                RIGHT * 4.35 + DOWN * 0.72,
            ]
            path_points = node_positions[: max(len(keywords), 2)]
            spine = VMobject()
            spine.set_points_smoothly(path_points)
            spine.set_stroke(ManimColor(accent_secondary), width=4, opacity=0.55)
            tracer = Dot(path_points[0], radius=0.11, color=ManimColor(accent_color))
            tracer_glow = Circle(radius=0.34).set_fill(ManimColor(accent_color), opacity=0.14).set_stroke(width=0).move_to(tracer)
            nodes = VGroup()
            callouts = VGroup()
            for index, keyword in enumerate(keywords, start=1):
                anchor = path_points[index - 1]
                ring = Circle(radius=0.34).set_stroke(ManimColor(panel_stroke), width=2.2, opacity=0.88).set_fill(ManimColor(panel_fill), opacity=0.92)
                ring.move_to(anchor)
                pulse = ring.copy().scale(1.45).set_stroke(ManimColor(glow_color), width=2, opacity=0.12).set_fill(opacity=0)
                node = Dot(anchor, radius=0.085, color=ManimColor(accent_color))
                badge = Text(str(index), font_size=18, color=ManimColor(primary), weight=BOLD).move_to(anchor)
                nodes.add(VGroup(pulse, ring, node, badge))

                vertical = UP if index % 2 else DOWN
                stub = Line(anchor, anchor + vertical * 0.78, color=ManimColor(panel_stroke), stroke_width=2.6, stroke_opacity=0.82)
                label = pill(
                    keyword,
                    fill=panel_fill,
                    text_color=primary,
                    width=min(max(len(keyword) * 0.24, 2.2), 4.5),
                )
                label.next_to(stub.get_end(), vertical, buff=0.16)
                callouts.add(VGroup(stub, label))
            footer = clamp_text(str(SPEC.get("footer_text") or "").strip(), max_width=9.6, max_font_size=24, min_font_size=16, color=secondary, weight=MEDIUM)
            footer.next_to(spine, DOWN, buff=1.08, aligned_edge=LEFT)
            self.play(FadeIn(stage, scale=1.02), run_time=0.26)
            if len(header) > 0:
                self.play(FadeIn(header_marker), FadeIn(header, shift=RIGHT * 0.16), run_time=intro)
            self.play(Create(spine), FadeIn(tracer_glow, scale=0.92), FadeIn(tracer, scale=0.92), LaggedStart(*[FadeIn(node, scale=0.88) for node in nodes], lag_ratio=0.08), run_time=accent)
            self.play(
                MoveAlongPath(tracer, spine),
                tracer_glow.animate.move_to(path_points[-1]),
                LaggedStart(
                    *[
                        FadeIn(callout, shift=(UP * 0.12 if (idx + 1) % 2 else DOWN * 0.12))
                        for idx, callout in enumerate(callouts)
                    ],
                    lag_ratio=0.1,
                ),
                run_time=max(reveal + 0.14, 0.52),
            )
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

        quote = clamp_text(
            str(SPEC.get("quote_text") or SPEC.get("emphasis_text") or SPEC.get("headline") or "Key quote"),
            max_width=10.2,
            max_font_size=68,
            min_font_size=32,
            color=primary,
        )
        quote.move_to(UP * 0.18)
        quote_eyebrow = pill(eyebrow_value or "INSIGHT", fill=eyebrow_fill, text_color=eyebrow_text, width=1.9)
        quote_eyebrow.next_to(quote, UP, buff=0.42, aligned_edge=LEFT)
        sweep = Line(LEFT * 2.8, RIGHT * 2.8, color=ManimColor(accent_color), stroke_width=5, stroke_opacity=0.96)
        sweep.next_to(quote, DOWN, buff=0.36)
        left_pillar = Rectangle(width=0.12, height=2.6).set_fill(ManimColor(accent_color), opacity=1.0).set_stroke(width=0)
        left_pillar.move_to(LEFT * 4.55 + DOWN * 0.22)
        right_pillar = left_pillar.copy().set_fill(ManimColor(accent_secondary), opacity=1.0).move_to(RIGHT * 4.55 + DOWN * 0.22)
        beam = Rectangle(width=5.9, height=0.24).set_fill(ManimColor(glow_color), opacity=0.16).set_stroke(width=0)
        beam.rotate(-0.14)
        beam.move_to(quote.get_center() + DOWN * 0.1)
        footer = clamp_text(str(SPEC.get("footer_text") or "").strip(), max_width=8.8, max_font_size=24, min_font_size=16, color=secondary, weight=MEDIUM)
        footer.next_to(sweep, DOWN, buff=0.34)
        self.play(FadeIn(stage, scale=1.02), run_time=0.26)
        self.play(FadeIn(left_pillar, scale=0.96), FadeIn(right_pillar, scale=0.96), FadeIn(beam), FadeIn(quote_eyebrow, shift=UP * 0.08), run_time=accent * 0.7)
        self.play(FadeIn(quote, shift=UP * 0.08), GrowFromEdge(sweep, LEFT), run_time=max(reveal, 0.44))
        if str(SPEC.get("footer_text") or "").strip():
            self.play(FadeIn(footer, shift=UP * 0.08), run_time=max(accent * 0.34, 0.22))
        self.wait(settle)
"""


def _preview_dimensions(width: int, height: int, *, compact: bool = False) -> tuple[int, int]:
    preview_width = min(width, 480 if compact else 640)
    preview_height = max(240, int(round(preview_width * (height / max(width, 1)))))
    if preview_height % 2 != 0:
        preview_height += 1
    return preview_width, preview_height


def _emit_render_progress(message: str) -> None:
    print(f"[manim] {message}", flush=True)


def _latex_runtime_ready(probe_root: Path) -> bool:
    global _LATEX_RUNTIME_READY_CACHE
    if _LATEX_RUNTIME_READY_CACHE is not None:
        return _LATEX_RUNTIME_READY_CACHE
    latex_binary = shutil.which("latex")
    if not latex_binary:
        _LATEX_RUNTIME_READY_CACHE = False
        return False
    probe_dir = probe_root / "_runtime_checks" / "latex_probe"
    try:
        probe_dir.mkdir(parents=True, exist_ok=True)
        tex_path = probe_dir / "probe.tex"
        tex_path.write_text(
            "\n".join(
                [
                    r"\documentclass{article}",
                    r"\usepackage{amsmath}",
                    r"\pagestyle{empty}",
                    r"\begin{document}",
                    r"$x$",
                    r"\end{document}",
                ]
            ),
            encoding="utf-8",
        )
    except OSError:
        _LATEX_RUNTIME_READY_CACHE = False
        return False
    command = [
        latex_binary,
        "-interaction=nonstopmode",
        "-halt-on-error",
        f"-output-directory={probe_dir}",
        str(tex_path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        _LATEX_RUNTIME_READY_CACHE = False
        return False
    if result.returncode != 0:
        _LATEX_RUNTIME_READY_CACHE = False
        return False
    dvi_path = probe_dir / "probe.dvi"
    dvisvgm_binary = shutil.which("dvisvgm")
    if not dvi_path.is_file() or not dvisvgm_binary:
        _LATEX_RUNTIME_READY_CACHE = False
        return False
    svg_path = probe_dir / "probe.svg"
    svg_command = [
        dvisvgm_binary,
        str(dvi_path),
        "-n",
        "-o",
        str(svg_path),
    ]
    try:
        svg_result = subprocess.run(svg_command, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        _LATEX_RUNTIME_READY_CACHE = False
        return False
    _LATEX_RUNTIME_READY_CACHE = svg_result.returncode == 0 and svg_path.is_file()
    return _LATEX_RUNTIME_READY_CACHE


def _scene_wrapper(scene_code: str, spec: dict[str, Any], brief_payload: dict[str, Any]) -> str:
    payload = dict(spec)
    payload["theme"] = _theme_defaults(spec)
    spec_json = json.dumps(payload, ensure_ascii=True)
    brief_json = json.dumps(brief_payload, ensure_ascii=True)
    return (
        "from __future__ import annotations\n\n"
        "import json\n\n"
        "import manim\n"
        "from manim import *\n"
        "from manim.utils.rate_functions import *\n"
        "from vex_manim.runtime import *\n\n"
        f"SCENE_SPEC = json.loads(r'''{spec_json}''')\n"
        f"SCENE_BRIEF = json.loads(r'''{brief_json}''')\n\n"
        f"{scene_code.strip()}\n\n"
        "GeneratedScene.SCENE_SPEC = SCENE_SPEC\n"
        "GeneratedScene.SCENE_BRIEF = SCENE_BRIEF\n"
    )


def _premium_blueprint_wrapper(
    spec: dict[str, Any],
    brief_payload: dict[str, Any],
    blueprint_payload: dict[str, Any],
) -> str:
    payload = dict(spec)
    payload["theme"] = _theme_defaults(spec)
    spec_json = json.dumps(payload, ensure_ascii=True)
    brief_json = json.dumps(brief_payload, ensure_ascii=True)
    blueprint_json = json.dumps(blueprint_payload, ensure_ascii=True)
    return (
        "from __future__ import annotations\n\n"
        "import json\n\n"
        "import manim\n"
        "from manim import *\n"
        "from manim.utils.rate_functions import *\n"
        "from vex_manim.runtime import *\n"
        "from vex_manim.premium_fallback import run_premium_blueprint_scene\n\n"
        f"SCENE_SPEC = json.loads(r'''{spec_json}''')\n"
        f"SCENE_BRIEF = json.loads(r'''{brief_json}''')\n"
        f"SCENE_BLUEPRINT = json.loads(r'''{blueprint_json}''')\n\n"
        "class GeneratedScene(VexGeneratedScene):\n"
        "    def construct(self):\n"
        "        run_premium_blueprint_scene(self, SCENE_SPEC, SCENE_BRIEF, SCENE_BLUEPRINT)\n\n"
        "GeneratedScene.SCENE_SPEC = SCENE_SPEC\n"
        "GeneratedScene.SCENE_BRIEF = SCENE_BRIEF\n"
    )


def _render_script(
    script_path: Path,
    *,
    scene_name: str,
    media_dir: Path,
    output_file: str,
    width: int,
    height: int,
    fps: float,
    timeout_sec: int,
    stage_label: str,
) -> Path:
    config_path = script_path.with_name(f"{output_file}.cfg")
    config_path.write_text(
        "\n".join(
            [
                "[CLI]",
                f"media_dir = {media_dir.as_posix()}",
                f"output_file = {output_file}",
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
    _emit_render_progress(f"{stage_label}: rendering {script_path.stem} at {width}x{height} {max(15, int(round(fps)))}fps")
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout_sec)
    except subprocess.TimeoutExpired as exc:
        raise VisualRendererError(
            f"Manim render timed out during {stage_label} after {timeout_sec}s for {script_path.stem}."
        ) from exc
    if result.returncode != 0:
        stderr = (result.stderr or result.stdout or "").strip()
        if "No module named manim" in stderr:
            raise VisualRendererError(
                "Manim is not installed in the current Python environment. Install it with `pip install manim`."
            )
        raise VisualRendererError(f"Manim render failed for {script_path.stem}: {stderr}")
    candidates = [
        path
        for path in media_dir.rglob("*.mp4")
        if "partial_movie_files" not in path.parts
        and path.name in {f"{output_file}.mp4", f"{scene_name}.mp4"}
    ]
    if not candidates:
        candidates = [
            path
            for path in media_dir.rglob("*.mp4")
            if "partial_movie_files" not in path.parts
        ]
    if not candidates:
        raise VisualRendererError(f"Manim render completed but no output video was found for {script_path.stem}.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _feedback_lines(validation_report: dict[str, Any], quality_report: dict[str, Any] | None) -> list[str]:
    lines: list[str] = []
    lines.extend(list(validation_report.get("errors") or []))
    lines.extend(list(validation_report.get("warnings") or []))
    if quality_report is not None:
        lines.extend(list(quality_report.get("issues") or []))
    deduped: list[str] = []
    for line in lines:
        cleaned = str(line).strip()
        if cleaned and cleaned not in deduped:
            deduped.append(cleaned)
    return deduped[:14]


def _is_duration_issue(issue: str) -> bool:
    return str(issue or "").startswith("Preview duration drifted from the target")


def _is_static_issue(issue: str) -> bool:
    return str(issue or "").startswith("The scene is too static for the requested intensity")


def _has_strong_motion_grammar(brief, validation) -> bool:
    profile = validation.profile
    return (
        profile.dynamic_device_count >= max(int(brief.minimum_dynamic_devices) + 2, 4)
        and (len(profile.advanced_features) >= 2 or profile.premium_helper_calls >= 4)
        and (profile.camera_move_mentions > 0 or "always_redraw" in profile.advanced_features or "ValueTracker" in profile.advanced_features)
        and (profile.premium_helper_calls >= 2 or profile.play_calls >= 5)
    )


def _is_minor_layout_overlap_issue(issue: str) -> bool:
    cleaned = str(issue or "").strip().lower()
    return "overlaps" in cleaned or "colliding with" in cleaned


def _is_guardrail_only_issue(issue: str) -> bool:
    cleaned = str(issue or "").strip().lower()
    return cleaned == "the runtime had to apply many layout guardrails; the composition is probably over-constrained."


def _is_minor_font_size_issue(issue: str) -> bool:
    cleaned = str(issue or "").strip().lower()
    match = re.search(r"very small font size \((\d+(?:\.\d+)?)px\)", cleaned)
    if not match:
        return False
    return float(match.group(1)) >= 14.0


def _should_rotate_blueprint(feedback_lines: list[str] | None) -> bool:
    if not feedback_lines:
        return False
    cleaned = " ".join(str(item or "").strip().lower() for item in feedback_lines)
    structural_markers = (
        "extends outside the safe frame",
        "dominates too much of the frame",
        "over-constrained",
        "very small font size",
        "colliding with",
        "overlaps",
        "lacks a real path",
        "lacks a real route",
        "lacks a real signal-flow",
        "stacked boxes and text",
        "editorial title-card pattern",
        "too much visible copy",
        "too static for the requested intensity",
    )
    return any(marker in cleaned for marker in structural_markers)


def _compiler_validation_report(brief, blueprint, scene_source: str) -> ValidationReport:
    copy_bank = dict(getattr(brief, "copy_bank", {}) or {})
    visible_copy = " ".join(
        str(item).strip()
        for item in [
            getattr(brief, "headline", ""),
            getattr(brief, "deck", ""),
            *(copy_bank.get("supporting_lines") or [])[:2],
            *(copy_bank.get("steps") or [])[:3],
            copy_bank.get("left_detail"),
            copy_bank.get("right_detail"),
        ]
        if str(item or "").strip()
    )
    visible_word_count = len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'./%-]*", visible_copy))
    panel_like_kinds = {"panel", "card", "ui_modules", "strip_modules"}
    premium_motion_tokens = ("route", "ring", "beam", "badge", "signal", "connector", "pulse", "glow", "focus")
    dynamic_devices = list(getattr(blueprint, "dynamic_devices", []) or [])
    suggested_features = list(getattr(blueprint, "suggested_features", []) or [])
    profile = CodeProfile(
        advanced_features=suggested_features,
        primitive_features=["VGroup"],
        imports=["manim", "vex_manim.premium_fallback"],
        play_calls=max(len(getattr(blueprint, "motion_beats", []) or []), 3),
        wait_calls=1,
        layout_registration_calls=max(len(getattr(blueprint, "elements", []) or []) // 2, 3),
        panel_helper_calls=sum(
            1 for element in (getattr(blueprint, "elements", []) or []) if getattr(element, "kind", "") in panel_like_kinds
        ),
        premium_helper_calls=max(
            sum(1 for item in dynamic_devices if any(token in str(item).lower() for token in premium_motion_tokens)),
            min(len(dynamic_devices), 4),
        ),
        title_helper_calls=1,
        visible_text_word_count=min(visible_word_count, int(getattr(brief, "text_budget_words", visible_word_count) or visible_word_count)),
        long_visible_text_literals=0,
        dynamic_device_count=max(
            len(set(dynamic_devices)),
            min(max(int(getattr(brief, "minimum_dynamic_devices", 2)) + 1, 2), 5),
        ),
        camera_move_mentions=1 if "camera" in str(getattr(blueprint, "camera_plan", "")).lower() else 0,
        class_names=["GeneratedScene"],
        line_count=len(scene_source.splitlines()),
    )
    return ValidationReport(valid=True, errors=[], warnings=[], profile=profile)


def _can_soft_accept_quality(brief, validation, quality) -> bool:
    if quality.passed:
        return True
    if quality.layout is not None and not quality.layout.passed:
        issues = list(quality.issues)
        if (
            quality.score >= 0.86
            and issues
            and all(_is_guardrail_only_issue(issue) for issue in issues)
            and _has_strong_motion_grammar(brief, validation)
        ):
            return True
        if (
            quality.score >= 0.9
            and issues
            and all(_is_guardrail_only_issue(issue) or _is_minor_font_size_issue(issue) for issue in issues)
            and _has_strong_motion_grammar(brief, validation)
        ):
            return True
        if (
            quality.score >= 0.82
            and issues
            and all(_is_minor_layout_overlap_issue(issue) or _is_guardrail_only_issue(issue) for issue in issues)
            and _has_strong_motion_grammar(brief, validation)
        ):
            return True
        return False
    if quality.score < 0.88:
        return False
    issues = list(quality.issues)
    if not issues:
        return False
    has_static_issue = any(_is_static_issue(issue) for issue in issues)
    if has_static_issue and not _has_strong_motion_grammar(brief, validation):
        return False
    return all(_is_duration_issue(issue) or _is_static_issue(issue) for issue in issues)


def _minimum_blueprint_compiler_quality(brief) -> float:
    family = str(getattr(brief, "scene_family", "") or "")
    if family == "timeline_journey":
        return 0.55
    if family == "system_map":
        return 0.58
    if family in {"metric_story", "dashboard_build"}:
        return 0.64
    if family == "comparison_morph":
        return 0.66
    if family == "interface_focus":
        return 0.62
    return 0.6


def _retime_rendered_video(
    input_path: Path,
    output_path: Path,
    *,
    target_duration_sec: float,
    actual_duration_sec: float,
) -> Path:
    if target_duration_sec <= 0 or actual_duration_sec <= 0:
        return input_path
    setpts_factor = target_duration_sec / actual_duration_sec
    command = [
        config.FFMPEG_PATH,
        "-i",
        str(input_path),
        "-vf",
        f"setpts={setpts_factor:.8f}*PTS",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-y",
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = (result.stderr or result.stdout or "").strip()
        raise VisualRendererError(f"Failed to retime generated Manim clip: {stderr}")
    return output_path


def _history_roots(spec: dict[str, Any]) -> list[Path]:
    roots: list[Path] = []
    for value in spec.get("scene_library_roots") or []:
        path = Path(str(value))
        if path not in roots:
            roots.append(path)
    return roots


def _example_limit_for_brief(brief) -> int:
    if float(getattr(brief, "duration_sec", 0.0) or 0.0) <= 3.8:
        if brief.scene_family in {"kinetic_quote", "kinetic_stack"}:
            return 1
        return 2
    if brief.animation_intensity == "low":
        return 1
    if brief.scene_family in {"kinetic_quote", "kinetic_stack", "dashboard_build"}:
        return 1
    if brief.scene_family in {"system_map", "comparison_morph", "timeline_journey", "interface_focus"}:
        return 3
    return 2 if brief.animation_intensity == "low" else 3


def _attempt_budget_for_brief(brief, spec: dict[str, Any]) -> int:
    model_name = str(spec.get("generation_model") or "").strip().lower()
    generation_tier = str(spec.get("generation_tier") or "").strip().lower()
    composition_mode = str(spec.get("composition_mode") or "").strip().lower()
    if model_name.startswith("gemma"):
        if (
            composition_mode == "replace"
            and (
                float(spec.get("importance") or 0.0) >= 0.62
                or float(spec.get("duration") or 0.0) <= 3.8
                or brief.scene_family in {"metric_story", "comparison_morph", "kinetic_quote", "system_map"}
            )
        ):
            return 2
        if generation_tier == "premium" and composition_mode == "picture_in_picture":
            return 2
        return 1
    importance = float(spec.get("importance") or 0.0)
    template = str(spec.get("template") or "")
    if composition_mode == "replace" and template in {"data_journey", "signal_network", "kinetic_route", "spotlight_compare", "interface_cascade", "ribbon_quote"}:
        return MAX_GENERATION_ATTEMPTS
    if generation_tier == "premium" and composition_mode == "picture_in_picture":
        return MAX_GENERATION_ATTEMPTS
    if brief.scene_family in {"system_map", "comparison_morph", "timeline_journey", "interface_focus"}:
        return MAX_GENERATION_ATTEMPTS
    if importance >= 0.72 or brief.animation_intensity == "high":
        return MAX_GENERATION_ATTEMPTS
    return 1


def _use_compact_preview(brief, spec: dict[str, Any]) -> bool:
    if str(spec.get("composition_mode") or "").strip().lower() == "picture_in_picture":
        return True
    if brief.animation_intensity == "low":
        return True
    return brief.scene_family in {"kinetic_quote", "kinetic_stack"} and float(spec.get("duration") or 0.0) <= 2.6


def _preview_render_budget(brief, fps: float, *, compact: bool) -> tuple[float, int]:
    if compact:
        return min(fps, 12.0), 1
    if brief.animation_intensity == "low" and brief.camera_style == "composed":
        return min(fps, 14.0), 1
    if brief.scene_family in {"kinetic_quote", "kinetic_stack"}:
        return min(fps, 14.0), 1
    return min(fps, 16.0), 2


def _is_request_timeout_error(message: str) -> bool:
    cleaned = str(message or "").upper()
    return "DEADLINE_EXCEEDED" in cleaned or "TIMEOUT" in cleaned or "TIMED OUT" in cleaned


def _should_use_compact_retry(
    brief,
    *,
    attempt_index: int,
    previous_code: str | None,
    feedback_lines: list[str] | None,
) -> bool:
    if attempt_index <= 1:
        return False
    if not previous_code:
        return True
    if previous_code:
        return float(getattr(brief, "duration_sec", 0.0) or 0.0) <= 3.8
    if not feedback_lines:
        return False
    return any(_is_request_timeout_error(item) for item in feedback_lines)


def _should_try_generated_scene(spec: dict[str, Any], brief) -> bool:
    template = str(spec.get("template") or "").strip().lower()
    composition_mode = str(spec.get("composition_mode") or "").strip().lower()
    importance = float(spec.get("importance") or 0.0)
    duration = float(spec.get("duration") or 0.0)
    generation_tier = str(spec.get("generation_tier") or "").strip().lower()
    require_generated_scene = bool(spec.get("require_generated_scene"))
    if composition_mode == "picture_in_picture":
        if require_generated_scene or generation_tier == "premium":
            if template in PREMIUM_GENERATED_TEMPLATES:
                return duration >= 2.2 and importance >= 0.48
            if template in FAST_TEMPLATE_TEMPLATES:
                return duration >= 2.2 and (importance >= 0.58 or brief.animation_intensity in {"medium", "high"})
        return False
    if template in PREMIUM_GENERATED_TEMPLATES:
        if importance >= 0.56:
            return True
        return duration >= 2.6 and brief.animation_intensity in {"medium", "high"}
    if template in FAST_TEMPLATE_TEMPLATES:
        if brief.animation_intensity in {"medium", "high"} and composition_mode == "replace":
            return importance >= 0.66 and duration >= 2.4
        return importance >= 0.9 and brief.animation_intensity == "high" and duration >= 3.2
    return importance >= 0.86 and composition_mode == "replace" and brief.animation_intensity != "low"


def _build_storyboard_contracts(
    spec: dict[str, Any],
    brief,
    blueprint_candidates: list[Any],
) -> list[dict[str, Any]]:
    contracts: list[dict[str, Any]] = []
    for blueprint in blueprint_candidates:
        ir = build_visual_explanation_ir(spec, brief, blueprint)
        frames = build_storyboard_frames(ir, brief, blueprint)
        critique = critique_storyboard(ir, frames, brief, blueprint)
        contracts.append(
            {
                "blueprint": blueprint,
                "ir": ir,
                "frames": frames,
                "critique": critique,
                "prompt": storyboard_prompt_block(ir, frames, critique),
            }
        )
    contracts.sort(
        key=lambda item: (
            1 if item["critique"].passed else 0,
            float(item["critique"].score),
            float(getattr(item["blueprint"], "score", 0.0) or 0.0),
        ),
        reverse=True,
    )
    return contracts


def _storyboard_contract_for_blueprint(
    contracts: list[dict[str, Any]],
    blueprint_id: str,
) -> dict[str, Any] | None:
    for contract in contracts:
        blueprint = contract.get("blueprint")
        if str(getattr(blueprint, "blueprint_id", "") or "") == blueprint_id:
            return contract
    return contracts[0] if contracts else None


def _storyboard_contract_to_dict(contract: dict[str, Any]) -> dict[str, Any]:
    blueprint = contract.get("blueprint")
    ir = contract.get("ir")
    frames = contract.get("frames") or []
    critique = contract.get("critique")
    return {
        "blueprint_id": str(getattr(blueprint, "blueprint_id", "") or ""),
        "blueprint_archetype": str(getattr(blueprint, "archetype", "") or ""),
        "visual_explanation_ir": ir.to_dict() if isinstance(ir, VisualExplanationIR) else {},
        "storyboard_frames": [
            frame.to_dict() if isinstance(frame, StoryboardFrame) else dict(frame)
            for frame in frames
        ],
        "storyboard_critique": critique.to_dict() if isinstance(critique, StoryboardCritique) else {},
    }


def _storyboard_prompt_for_contract(contract: dict[str, Any] | None) -> str:
    if not contract:
        return ""
    prompt = str(contract.get("prompt") or "").strip()
    if prompt:
        return prompt
    ir = contract.get("ir")
    frames = contract.get("frames") or []
    critique = contract.get("critique")
    if isinstance(ir, VisualExplanationIR):
        typed_frames = [frame for frame in frames if isinstance(frame, StoryboardFrame)]
        typed_critique = critique if isinstance(critique, StoryboardCritique) else None
        return storyboard_prompt_block(ir, typed_frames, typed_critique)
    return ""


class ManimRenderer(VisualRenderer):
    name = "manim"
    supported_templates = {
        "data_journey",
        "signal_network",
        "kinetic_route",
        "spotlight_compare",
        "interface_cascade",
        "ribbon_quote",
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
        if template in {"data_journey", "signal_network", "kinetic_route", "spotlight_compare", "interface_cascade", "ribbon_quote"}:
            score += 0.18
        if template in {"timeline_steps", "system_flow", "comparison_split", "stat_grid"}:
            score += 0.12
        if visual_hint in {"data_graphic", "process"}:
            score += 0.1
        if visual_hint in {"abstract_motion"}:
            score += 0.07
        if visual_hint == "product_ui":
            score -= 0.08
        if composition == "replace":
            score += 0.08
        score += importance * 0.04
        return round(score, 3)

    def _attempt_generated_scene(
        self,
        spec: dict[str, Any],
        *,
        job_dir: Path,
        width: int,
        height: int,
        fps: float,
        latex_available: bool,
    ) -> tuple[Path, dict[str, Any], dict[str, str]]:
        provider_name = str(spec.get("generation_provider") or "").strip().lower()
        model_name = str(spec.get("generation_model") or "").strip()
        if provider_name not in {"gemini", "claude"} or not model_name:
            raise VisualRendererError("Generated Manim scenes require a configured reasoning model.")

        brief = build_scene_brief(spec, width=width, height=height, fps=fps, latex_available=latex_available)
        brief.render_constraints["latex_available"] = latex_available
        if not latex_available:
            brief.must_avoid.append("LaTeX-dependent Manim objects or chart labels")
        blueprint_candidates = build_scene_blueprints(brief, limit=3)
        if not blueprint_candidates:
            raise VisualRendererError("No scene blueprints could be constructed for this visual.")
        storyboard_contracts = _build_storyboard_contracts(spec, brief, blueprint_candidates)
        if storyboard_contracts:
            blueprint_candidates = [contract["blueprint"] for contract in storyboard_contracts]
        selected_contract = storyboard_contracts[0] if storyboard_contracts else None
        selected_blueprint = selected_contract["blueprint"] if selected_contract else blueprint_candidates[0]
        selected_storyboard_prompt = _storyboard_prompt_for_contract(selected_contract)
        if selected_contract:
            selected_critique = selected_contract["critique"]
            _emit_render_progress(
                f"{spec.get('visual_id', 'visual')}: storyboard contract {selected_blueprint.archetype} scored {selected_critique.score:.3f}"
            )
        storyboard_candidates_path = job_dir / "storyboard_candidates.json"
        storyboard_candidates_path.write_text(
            json.dumps([_storyboard_contract_to_dict(contract) for contract in storyboard_contracts], indent=2),
            encoding="utf-8",
        )
        blueprints_path = job_dir / "scene_blueprints.json"
        blueprints_path.write_text(
            json.dumps([item.to_dict() for item in blueprint_candidates], indent=2),
            encoding="utf-8",
        )
        examples = retrieve_scene_examples(
            brief,
            history_roots=_history_roots(spec),
            limit=_example_limit_for_brief(brief),
            forbidden_features={"BarChart", "MathTex", "Tex", "Matrix", "Variable"} if not latex_available else None,
            preferred_tags=selected_blueprint.prompt_terms(),
            preferred_features=selected_blueprint.suggested_features,
        )
        full_examples = list(examples)
        _emit_render_progress(
            f"{spec.get('visual_id', 'visual')}: planning scene execution"
        )
        selected_execution_plan = request_scene_execution_plan(
            provider_name,
            model_name,
            brief,
            selected_blueprint,
            alternative_blueprints=[item for item in blueprint_candidates if item.blueprint_id != selected_blueprint.blueprint_id][:2],
            storyboard_context=selected_storyboard_prompt,
        )
        execution_plan_path = job_dir / "scene_execution_plan.json"
        execution_plan_path.write_text(
            json.dumps(selected_execution_plan.to_dict(), indent=2),
            encoding="utf-8",
        )
        attempt_budget = _attempt_budget_for_brief(brief, spec)
        compact_preview = _use_compact_preview(brief, spec)
        preview_fps, preview_frame_count = _preview_render_budget(brief, fps, compact=compact_preview)
        brief_path = job_dir / "scene_brief.json"
        brief_path.write_text(json.dumps(brief.to_dict(), indent=2), encoding="utf-8")
        attempts_root = job_dir / "attempts"
        attempts_root.mkdir(parents=True, exist_ok=True)
        attempts: list[dict[str, Any]] = []
        previous_code: str | None = None
        feedback_lines: list[str] | None = None
        last_request_error: str | None = None
        chosen_scene_source: str | None = None
        chosen_candidate = None
        chosen_quality: dict[str, Any] | None = None
        chosen_blueprint = selected_blueprint
        chosen_execution_plan = selected_execution_plan
        chosen_storyboard_contract = selected_contract

        for attempt_index in range(1, attempt_budget + 1):
            attempt_dir = attempts_root / f"attempt_{attempt_index:02d}"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            compact_retry = _should_use_compact_retry(
                brief,
                attempt_index=attempt_index,
                previous_code=previous_code,
                feedback_lines=feedback_lines,
            )
            rotate_blueprint = (
                attempt_index > 1
                and len(blueprint_candidates) > 1
                and (
                    compact_retry
                    or bool(previous_code)
                    or _should_rotate_blueprint(feedback_lines)
                    or bool(last_request_error)
                )
            )
            active_blueprint = (
                blueprint_candidates[min(attempt_index - 1, len(blueprint_candidates) - 1)]
                if rotate_blueprint
                else (
                    blueprint_candidates[min(attempt_index - 1, len(blueprint_candidates) - 1)]
                    if compact_retry and not previous_code and len(blueprint_candidates) > 1
                    else selected_blueprint
                )
            )
            active_examples = (
                list(full_examples[:1])
                if compact_retry and full_examples
                else list(full_examples)
            )
            active_previous_code = previous_code
            active_feedback_lines = list(feedback_lines or [])
            if rotate_blueprint and active_blueprint.blueprint_id != selected_blueprint.blueprint_id:
                active_previous_code = None
                active_feedback_lines.append(
                    f"Previous blueprint {selected_blueprint.archetype} over-constrained the layout. Use a different composition strategy."
                )
            active_storyboard_contract = _storyboard_contract_for_blueprint(
                storyboard_contracts,
                active_blueprint.blueprint_id,
            )
            active_storyboard_prompt = _storyboard_prompt_for_contract(active_storyboard_contract)
            active_execution_plan = (
                selected_execution_plan
                if active_blueprint.blueprint_id == selected_blueprint.blueprint_id
                else build_deterministic_execution_plan(brief, active_blueprint)
            )
            _emit_render_progress(
                f"{spec.get('visual_id', 'visual')}: generation attempt {attempt_index}/{attempt_budget}"
            )
            try:
                retry_suffix = " (compact retry)" if compact_retry else ""
                _emit_render_progress(
                    f"{spec.get('visual_id', 'visual')}: requesting scene code from {provider_name}/{model_name}{retry_suffix}"
                )
                candidate = request_scene_candidate(
                    provider_name,
                    model_name,
                    brief,
                    active_examples,
                    active_blueprint,
                    active_execution_plan,
                    alternative_blueprints=[item for item in blueprint_candidates if item.blueprint_id != active_blueprint.blueprint_id][:2],
                    previous_code=active_previous_code,
                    feedback_lines=active_feedback_lines,
                    storyboard_context=active_storyboard_prompt,
                )
                last_request_error = None
            except Exception as exc:
                last_request_error = str(exc)
                attempts.append(
                    {
                        "attempt": attempt_index,
                        "blueprint_id": active_blueprint.blueprint_id,
                        "blueprint_archetype": active_blueprint.archetype,
                        "compact_retry": compact_retry,
                        "request_error": str(exc),
                    }
                )
                previous_code = previous_code or ""
                feedback_lines = [f"Model call failed: {exc}"]
                continue

            validation = validate_generated_scene_code(candidate.scene_code, latex_available=latex_available, brief=brief)
            attempt_record: dict[str, Any] = {
                "attempt": attempt_index,
                "blueprint_id": active_blueprint.blueprint_id,
                "blueprint_archetype": active_blueprint.archetype,
                "compact_retry": compact_retry,
                "summary": candidate.summary,
                "features": list(candidate.features),
                "validation": validation.to_dict(),
            }
            if not validation.valid:
                _emit_render_progress(
                    f"{spec.get('visual_id', 'visual')}: validation failed on attempt {attempt_index}"
                )
                attempts.append(attempt_record)
                previous_code = candidate.scene_code
                feedback_lines = _feedback_lines(validation.to_dict(), None)
                continue

            attempt_spec = dict(spec)
            attempt_spec["layout_snapshot_path"] = str(attempt_dir / "layout_snapshot.json")
            if active_storyboard_contract:
                attempt_spec["visual_explanation_ir"] = active_storyboard_contract["ir"].to_dict()
                attempt_spec["storyboard_frames"] = [
                    frame.to_dict() for frame in active_storyboard_contract["frames"]
                ]
            scene_source = _scene_wrapper(candidate.scene_code, attempt_spec, brief.to_dict())
            script_path = attempt_dir / "generated_scene.py"
            script_path.write_text(scene_source, encoding="utf-8")
            preview_width, preview_height = _preview_dimensions(width, height, compact=compact_preview)
            preview_media_dir = attempt_dir / "preview_media"
            try:
                preview_video_path = _render_script(
                    script_path,
                    scene_name="GeneratedScene",
                    media_dir=preview_media_dir,
                    output_file="GeneratedScenePreview",
                    width=preview_width,
                    height=preview_height,
                    fps=preview_fps,
                    timeout_sec=config.MANIM_PREVIEW_TIMEOUT_SEC,
                    stage_label=f"preview attempt {attempt_index}",
                )
                preview_metadata = probe_video(str(preview_video_path))
                preview_frames = extract_preview_frames(
                    str(preview_video_path),
                    attempt_dir / "preview_frames",
                    duration_sec=float(preview_metadata.get("duration_sec") or 0.0),
                    frame_count=preview_frame_count,
                )
                preview_report = analyze_preview(
                    str(preview_video_path),
                    float(preview_metadata.get("duration_sec") or 0.0),
                    preview_frames,
                    theme=_theme_defaults(spec),
                )
                layout_report = None
                layout_snapshot_path = attempt_dir / "layout_snapshot.json"
                if layout_snapshot_path.is_file():
                    layout_report = analyze_layout_snapshot(load_layout_snapshot(layout_snapshot_path), brief)
                    attempt_record["layout"] = layout_report.to_dict()
                quality = evaluate_generated_scene_quality(brief, validation, preview_report, layout=layout_report)
                attempt_record["preview"] = preview_report.to_dict()
                attempt_record["quality"] = quality.to_dict()
            except Exception as exc:
                attempt_record["preview_error"] = str(exc)
                _emit_render_progress(
                    f"{spec.get('visual_id', 'visual')}: preview failed on attempt {attempt_index} - {exc}"
                )
                attempts.append(attempt_record)
                previous_code = candidate.scene_code
                feedback_lines = [f"Preview render failed: {exc}"]
                continue

            attempts.append(attempt_record)
            soft_accept = _can_soft_accept_quality(brief, validation, quality)
            attempt_record["quality_soft_accept"] = soft_accept
            if quality.passed or soft_accept:
                _emit_render_progress(
                    f"{spec.get('visual_id', 'visual')}: accepted attempt {attempt_index} with quality {quality.score:.3f}"
                )
                chosen_scene_source = scene_source
                chosen_candidate = candidate
                chosen_quality = {**quality.to_dict(), "soft_accept": soft_accept}
                chosen_blueprint = active_blueprint
                chosen_execution_plan = active_execution_plan
                chosen_storyboard_contract = active_storyboard_contract
                break

            previous_code = candidate.scene_code
            feedback_lines = _feedback_lines(validation.to_dict(), quality.to_dict())
            _emit_render_progress(
                f"{spec.get('visual_id', 'visual')}: retrying after quality issues on attempt {attempt_index}"
            )

        report_path = job_dir / "generation_report.json"
        used_blueprint_compiler = False
        blueprint_compiler_rejection: str | None = None
        if chosen_scene_source is None:
            used_blueprint_compiler = True
            compiler_rejections: list[str] = []
            compiler_candidates = [selected_blueprint] + [
                item for item in blueprint_candidates if item.blueprint_id != selected_blueprint.blueprint_id
            ]
            for compiler_index, compiler_blueprint in enumerate(compiler_candidates, start=1):
                compiler_attempt_dir = attempts_root / (
                    "blueprint_compiler" if compiler_index == 1 else f"blueprint_compiler_{compiler_index:02d}"
                )
                compiler_attempt_dir.mkdir(parents=True, exist_ok=True)
                compiler_spec = dict(spec)
                compiler_spec["layout_snapshot_path"] = str(compiler_attempt_dir / "layout_snapshot.json")
                compiler_storyboard_contract = _storyboard_contract_for_blueprint(
                    storyboard_contracts,
                    compiler_blueprint.blueprint_id,
                )
                if compiler_storyboard_contract:
                    compiler_spec["visual_explanation_ir"] = compiler_storyboard_contract["ir"].to_dict()
                    compiler_spec["storyboard_frames"] = [
                        frame.to_dict() for frame in compiler_storyboard_contract["frames"]
                    ]
                compiler_execution_plan = (
                    selected_execution_plan
                    if compiler_blueprint.blueprint_id == selected_blueprint.blueprint_id
                    else build_deterministic_execution_plan(brief, compiler_blueprint)
                )
                compiler_scene_source = _premium_blueprint_wrapper(
                    compiler_spec,
                    brief.to_dict(),
                    compiler_blueprint.to_dict(),
                )
                compiler_script_path = compiler_attempt_dir / "generated_scene.py"
                compiler_script_path.write_text(compiler_scene_source, encoding="utf-8")
                compiler_attempt: dict[str, Any] = {
                    "attempt": "blueprint_compiler" if compiler_index == 1 else f"blueprint_compiler_{compiler_index:02d}",
                    "blueprint_id": compiler_blueprint.blueprint_id,
                    "blueprint_archetype": compiler_blueprint.archetype,
                }
                try:
                    preview_video_path = _render_script(
                        compiler_script_path,
                        scene_name="GeneratedScene",
                        media_dir=compiler_attempt_dir / "preview_media",
                        output_file="GeneratedScenePreview",
                        width=preview_width,
                        height=preview_height,
                        fps=preview_fps,
                        timeout_sec=config.MANIM_PREVIEW_TIMEOUT_SEC,
                        stage_label="preview blueprint compiler",
                    )
                    preview_metadata = probe_video(str(preview_video_path))
                    preview_frames = extract_preview_frames(
                        str(preview_video_path),
                        compiler_attempt_dir / "preview_frames",
                        duration_sec=float(preview_metadata.get("duration_sec") or 0.0),
                        frame_count=preview_frame_count,
                    )
                    preview_report = analyze_preview(
                        str(preview_video_path),
                        float(preview_metadata.get("duration_sec") or 0.0),
                        preview_frames,
                        theme=_theme_defaults(spec),
                    )
                    layout_report = None
                    compiler_layout_snapshot = compiler_attempt_dir / "layout_snapshot.json"
                    if compiler_layout_snapshot.is_file():
                        layout_report = analyze_layout_snapshot(load_layout_snapshot(compiler_layout_snapshot), brief)
                        compiler_attempt["layout"] = layout_report.to_dict()
                    compiler_validation = _compiler_validation_report(
                        brief,
                        compiler_blueprint,
                        compiler_scene_source,
                    )
                    quality = evaluate_generated_scene_quality(
                        brief,
                        compiler_validation,
                        preview_report,
                        layout=layout_report,
                    )
                    soft_accept = _can_soft_accept_quality(brief, compiler_validation, quality)
                    compiler_attempt["preview"] = preview_report.to_dict()
                    compiler_attempt["quality"] = quality.to_dict()
                    compiler_attempt["quality_soft_accept"] = soft_accept
                    min_compiler_quality = _minimum_blueprint_compiler_quality(brief)
                    if float(quality.score) >= min_compiler_quality:
                        chosen_scene_source = compiler_scene_source
                        chosen_quality = {**quality.to_dict(), "soft_accept": soft_accept}
                        chosen_blueprint = compiler_blueprint
                        chosen_execution_plan = compiler_execution_plan
                        chosen_storyboard_contract = compiler_storyboard_contract
                        attempts.append(compiler_attempt)
                        blueprint_compiler_rejection = None
                        break
                    rejection = (
                        f"Deterministic premium fallback quality {quality.score:.3f} was below the required "
                        f"{min_compiler_quality:.2f} for {brief.scene_family}."
                    )
                    compiler_attempt["rejected"] = rejection
                    compiler_rejections.append(rejection)
                except Exception as exc:
                    compiler_attempt["preview_error"] = str(exc)
                    compiler_rejections.append(f"Deterministic premium fallback preview failed: {exc}")
                attempts.append(compiler_attempt)
            if chosen_scene_source is None:
                chosen_quality = {
                    "score": 0.0,
                    "issues": compiler_rejections[-3:],
                    "soft_accept": False,
                }
                blueprint_compiler_rejection = compiler_rejections[-1] if compiler_rejections else "Deterministic premium fallback failed."
        fallback_used = bool(used_blueprint_compiler)
        execution_plan_path.write_text(
            json.dumps(chosen_execution_plan.to_dict(), indent=2),
            encoding="utf-8",
        )
        final_storyboard_contract = chosen_storyboard_contract or selected_contract
        final_storyboard_payload = (
            _storyboard_contract_to_dict(final_storyboard_contract)
            if final_storyboard_contract
            else {}
        )
        visual_ir_path = job_dir / "visual_explanation_ir.json"
        storyboard_path = job_dir / "storyboard_contract.json"
        visual_ir_path.write_text(
            json.dumps(final_storyboard_payload.get("visual_explanation_ir", {}), indent=2),
            encoding="utf-8",
        )
        storyboard_path.write_text(
            json.dumps(final_storyboard_payload, indent=2),
            encoding="utf-8",
        )
        write_generation_report(
            report_path,
            brief=brief,
            blueprint_candidates=blueprint_candidates,
            selected_blueprint=chosen_blueprint,
            selected_execution_plan=chosen_execution_plan,
            selected_examples=full_examples,
            attempts=attempts,
            final_candidate=chosen_candidate,
            final_scene_code=chosen_candidate.scene_code if chosen_candidate else chosen_scene_source,
            quality_score=(chosen_quality or {}).get("score"),
            fallback_used=fallback_used,
            visual_explanation_ir=final_storyboard_payload.get("visual_explanation_ir", {}),
            storyboard_frames=final_storyboard_payload.get("storyboard_frames", []),
            storyboard_critique=final_storyboard_payload.get("storyboard_critique", {}),
            storyboard_candidates=[
                _storyboard_contract_to_dict(contract) for contract in storyboard_contracts
            ],
        )
        if blueprint_compiler_rejection:
            raise VisualRendererError(blueprint_compiler_rejection)
        artifact_paths = {
            "generation_report_path": str(report_path),
            "scene_brief_path": str(brief_path),
            "scene_blueprints_path": str(blueprints_path),
            "visual_explanation_ir_path": str(visual_ir_path),
            "storyboard_contract_path": str(storyboard_path),
            "storyboard_candidates_path": str(storyboard_candidates_path),
            "scene_execution_plan_path": str(execution_plan_path),
        }
        layout_snapshot_path = job_dir / "layout_snapshot.json"
        artifact_paths["layout_snapshot_path"] = str(layout_snapshot_path)
        metadata = {
            "scene_generation_mode": "blueprint_compiler" if used_blueprint_compiler else "llm_codegen",
            "scene_family": brief.scene_family,
            "blueprint_id": chosen_blueprint.blueprint_id,
            "blueprint_archetype": chosen_blueprint.archetype,
            "execution_plan_source": chosen_execution_plan.source,
            "camera_style": brief.camera_style,
            "animation_intensity": brief.animation_intensity,
            "visual_ir_scene_type": (final_storyboard_payload.get("visual_explanation_ir") or {}).get("scene_type"),
            "storyboard_score": (final_storyboard_payload.get("storyboard_critique") or {}).get("score"),
            "storyboard_passed": bool((final_storyboard_payload.get("storyboard_critique") or {}).get("passed")),
            "selected_examples": [example.example_id for example in examples],
            "quality_score": (chosen_quality or {}).get("score"),
            "quality_soft_accept": bool((chosen_quality or {}).get("soft_accept")),
            "fallback_used": fallback_used,
            "attempt_budget": attempt_budget,
        }
        final_script_path = job_dir / "scene.py"
        final_spec = dict(spec)
        final_spec["layout_snapshot_path"] = str(layout_snapshot_path)
        if final_storyboard_contract:
            final_spec["visual_explanation_ir"] = final_storyboard_contract["ir"].to_dict()
            final_spec["storyboard_frames"] = [
                frame.to_dict() for frame in final_storyboard_contract["frames"]
            ]
        if used_blueprint_compiler:
            _emit_render_progress(
                f"{spec.get('visual_id', 'visual')}: switching to deterministic premium blueprint compiler"
            )
            final_script_path.write_text(
                _premium_blueprint_wrapper(final_spec, brief.to_dict(), chosen_blueprint.to_dict()),
                encoding="utf-8",
            )
        else:
            final_script_path.write_text(
                _scene_wrapper(chosen_candidate.scene_code, final_spec, brief.to_dict()),
                encoding="utf-8",
            )
        return final_script_path, metadata, artifact_paths

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
        artifact_paths: dict[str, str] = {}
        scene_metadata: dict[str, Any] = {}
        scene_name = "GeneratedScene"
        latex_available = _latex_runtime_ready(job_dir)
        decision_brief = build_scene_brief(spec, width=width, height=height, fps=fps, latex_available=latex_available)
        require_generated_scene = bool(spec.get("require_generated_scene"))
        try:
            if _should_try_generated_scene(spec, decision_brief):
                _emit_render_progress(f"{spec_id}: preparing generated Manim scene")
                script_path, scene_metadata, artifact_paths = self._attempt_generated_scene(
                    spec,
                    job_dir=job_dir,
                    width=width,
                    height=height,
                    fps=fps,
                    latex_available=latex_available,
                )
            else:
                raise VisualRendererError("Fast-path deterministic template selected for this lightweight visual.")
        except Exception as exc:
            if require_generated_scene:
                raise VisualRendererError(
                    f"Premium generated Manim scene was required for {spec_id}, but the generated path failed: {exc}"
                ) from exc
            scene_name = _safe_scene_name(spec_id)
            script_path = job_dir / "scene.py"
            script_path.write_text(_legacy_scene_script(scene_name, spec), encoding="utf-8")
            scene_metadata = {
                "scene_generation_mode": "legacy_template",
                "template": str(spec.get("template") or ""),
                "generation_failure": str(exc),
            }
            if str(exc) == "Fast-path deterministic template selected for this lightweight visual.":
                scene_metadata["generation_skipped"] = "fast_path_lightweight_visual"
                _emit_render_progress(f"{spec_id}: using fast deterministic Manim template")
            else:
                _emit_render_progress(f"{spec_id}: using legacy template fallback - {exc}")

        media_dir = job_dir / "media"
        output_stem = scene_name if scene_name != "GeneratedScene" else f"GeneratedScene_{_safe_scene_name(spec_id)}"
        asset_path = _render_script(
            script_path,
            scene_name=scene_name,
            media_dir=media_dir,
            output_file=output_stem,
            width=width,
            height=height,
            fps=fps,
            timeout_sec=config.MANIM_FINAL_TIMEOUT_SEC,
            stage_label="final render",
        )
        video_metadata = probe_video(str(asset_path))
        target_duration_sec = float(spec.get("duration") or 0.0)
        actual_duration_sec = float(video_metadata.get("duration_sec") or 0.0)
        if (
            scene_metadata.get("scene_generation_mode") == "llm_codegen"
            and not scene_metadata.get("fallback_used")
            and not bool(video_metadata.get("has_audio"))
            and target_duration_sec > 0.0
            and actual_duration_sec > 0.0
            and abs(actual_duration_sec - target_duration_sec) > 0.12
        ):
            retimed_path = job_dir / f"{Path(asset_path).stem}_retimed.mp4"
            asset_path = _retime_rendered_video(
                asset_path,
                retimed_path,
                target_duration_sec=target_duration_sec,
                actual_duration_sec=actual_duration_sec,
            )
            video_metadata = probe_video(str(asset_path))
            scene_metadata = {
                **scene_metadata,
                "retime_source_duration_sec": round(actual_duration_sec, 3),
                "retimed_to_duration_sec": round(target_duration_sec, 3),
            }
        final_asset = RenderedAsset(
            asset_path=str(asset_path),
            width=int(video_metadata.get("width") or width),
            height=int(video_metadata.get("height") or height),
            duration_sec=float(video_metadata.get("duration_sec") or 0.0),
            renderer=self.name,
            job_dir=str(job_dir),
            script_path=str(script_path),
            artifact_paths=artifact_paths,
            metadata={**scene_metadata, **video_metadata},
        )
        return final_asset
