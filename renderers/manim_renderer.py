from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from engine import probe_video
from renderers.base import RenderedAsset, VisualRenderer, VisualRendererError


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


def clamp_text(content: str, max_width: float, max_font_size: int, min_font_size: int, color: str, weight=BOLD):
    cleaned = str(content or "").strip() or " "
    for size in range(max_font_size, min_font_size - 1, -4):
        text = Text(cleaned, font_size=size, color=ManimColor(color), weight=weight)
        if text.width <= max_width:
            return text
    return Text(cleaned, font_size=min_font_size, color=ManimColor(color), weight=weight)


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
    group.arrange(DOWN, buff=0.24, aligned_edge=aligned_edge)
    return group


class {scene_name}(Scene):
    def construct(self):
        self.camera.background_color = ManimColor(theme("background", "#0B1020"))
        duration = max(float(SPEC.get("duration") or 2.0), 1.0)
        template = str(SPEC.get("template") or "quote_focus")
        headline_text = str(SPEC.get("headline") or "").strip()
        footer_text_value = str(SPEC.get("footer_text") or "").strip()
        intro = min(max(duration * 0.22, 0.35), 0.9)
        accent = min(max(duration * 0.24, 0.35), 1.0)
        reveal = min(max(duration * 0.22, 0.35), 0.9)
        settle = max(duration - intro - accent - reveal, 0.12)

        primary = theme("text_primary", "#F8FAFC")
        secondary = theme("text_secondary", "#CBD5E1")
        panel_fill = theme("panel_fill", "#13203A")
        panel_stroke = theme("panel_stroke", "#60A5FA")
        accent_color = theme("accent", "#F59E0B")

        headline = clamp_text(headline_text, max_width=11.8, max_font_size=46, min_font_size=26, color=secondary)
        headline.to_edge(UP, buff=0.55)

        if template == "metric_callout":
            panel = RoundedRectangle(corner_radius=0.28, width=11.8, height=5.8)
            panel.set_fill(ManimColor(panel_fill), opacity=1.0)
            panel.set_stroke(ManimColor(panel_stroke), width=3)
            value = clamp_text(str(SPEC.get("emphasis_text") or SPEC.get("headline") or "Key Point"), max_width=9.8, max_font_size=76, min_font_size=36, color=primary)
            support = line_stack(SPEC.get("supporting_lines") or [], max_width=9.6, max_font_size=28, min_font_size=18, color=secondary, weight=MEDIUM, aligned_edge=LEFT)
            content = VGroup(value, support).arrange(DOWN, buff=0.48)
            content.move_to(panel.get_center())
            accent_bar = Rectangle(width=0.26, height=4.8)
            accent_bar.set_fill(ManimColor(accent_color), opacity=1.0)
            accent_bar.set_stroke(width=0)
            accent_bar.next_to(panel, LEFT, buff=0.0)
            if headline_text:
                self.play(FadeIn(headline, shift=UP * 0.2), run_time=intro)
            self.play(DrawBorderThenFill(panel), FadeIn(accent_bar, shift=RIGHT * 0.1), run_time=accent)
            self.play(Write(value), LaggedStart(*[FadeIn(item, shift=UP * 0.12) for item in support], lag_ratio=0.12), run_time=reveal)
            self.wait(settle)
            return

        if template == "keyword_stack":
            keywords = [str(item).strip() for item in (SPEC.get("keywords") or []) if str(item).strip()][:4]
            cards = VGroup()
            for keyword in keywords or [str(SPEC.get("emphasis_text") or "Key idea")]:
                box = RoundedRectangle(corner_radius=0.22, width=8.8, height=1.15)
                box.set_fill(ManimColor(panel_fill), opacity=1.0)
                box.set_stroke(ManimColor(panel_stroke), width=2.5)
                label = clamp_text(keyword, max_width=7.6, max_font_size=34, min_font_size=20, color=primary)
                label.move_to(box.get_center())
                cards.add(VGroup(box, label))
            cards.arrange(DOWN, buff=0.32)
            if headline_text:
                self.play(FadeIn(headline, shift=UP * 0.2), run_time=intro)
            self.play(LaggedStart(*[FadeIn(card, shift=UP * 0.18) for card in cards], lag_ratio=0.14), run_time=accent + reveal)
            footer = clamp_text(footer_text_value, max_width=10.8, max_font_size=26, min_font_size=18, color=secondary, weight=MEDIUM)
            footer.next_to(cards, DOWN, buff=0.42)
            if footer_text_value:
                self.play(FadeIn(footer, shift=UP * 0.12), run_time=0.28)
            self.wait(settle)
            return

        if template == "timeline_steps":
            steps = [str(item).strip() for item in (SPEC.get("steps") or []) if str(item).strip()][:4]
            steps = steps or [str(SPEC.get("emphasis_text") or "Step one"), "Step two", "Step three"]
            cards = VGroup()
            for index, step in enumerate(steps, start=1):
                box = RoundedRectangle(corner_radius=0.22, width=2.8, height=2.0)
                box.set_fill(ManimColor(panel_fill), opacity=1.0)
                box.set_stroke(ManimColor(panel_stroke), width=2.5)
                number = Text(str(index), font_size=24, color=ManimColor(accent_color), weight=BOLD)
                number.next_to(box.get_top(), DOWN, buff=0.26)
                label = clamp_text(step, max_width=2.2, max_font_size=28, min_font_size=18, color=primary)
                label.move_to(box.get_center() + DOWN * 0.1)
                cards.add(VGroup(box, number, label))
            cards.arrange(RIGHT, buff=0.55)
            arrows = VGroup()
            for index in range(len(cards) - 1):
                arrow = Arrow(cards[index].get_right(), cards[index + 1].get_left(), buff=0.14, stroke_width=5, color=ManimColor(accent_color), max_tip_length_to_length_ratio=0.12)
                arrows.add(arrow)
            scene_group = VGroup(cards, arrows)
            scene_group.move_to(ORIGIN + DOWN * 0.2)
            if headline_text:
                self.play(FadeIn(headline, shift=UP * 0.2), run_time=intro)
            self.play(LaggedStart(*[FadeIn(card, shift=UP * 0.12) for card in cards], lag_ratio=0.1), run_time=accent)
            if len(arrows) > 0:
                self.play(LaggedStart(*[GrowArrow(arrow) for arrow in arrows], lag_ratio=0.1), run_time=reveal)
            self.wait(settle)
            return

        if template == "comparison_split":
            left_box = RoundedRectangle(corner_radius=0.22, width=5.0, height=4.8)
            left_box.set_fill(ManimColor(panel_fill), opacity=1.0)
            left_box.set_stroke(ManimColor(panel_stroke), width=2.5)
            right_box = left_box.copy()
            left_group = VGroup(
                clamp_text(str(SPEC.get("left_label") or "Before"), max_width=4.0, max_font_size=34, min_font_size=22, color=primary),
                line_stack([str(SPEC.get("left_detail") or "")], max_width=3.8, max_font_size=24, min_font_size=16, color=secondary, weight=MEDIUM),
            ).arrange(DOWN, buff=0.34)
            left_group.move_to(left_box.get_center())
            right_group = VGroup(
                clamp_text(str(SPEC.get("right_label") or "After"), max_width=4.0, max_font_size=34, min_font_size=22, color=primary),
                line_stack([str(SPEC.get("right_detail") or "")], max_width=3.8, max_font_size=24, min_font_size=16, color=secondary, weight=MEDIUM),
            ).arrange(DOWN, buff=0.34)
            right_group.move_to(right_box.get_center())
            left_panel = VGroup(left_box, left_group)
            right_panel = VGroup(right_box, right_group)
            layout = VGroup(left_panel, right_panel).arrange(RIGHT, buff=1.0)
            versus = Text("VS", font_size=28, color=ManimColor(accent_color), weight=BOLD)
            versus.move_to((left_panel.get_right() + right_panel.get_left()) / 2)
            if headline_text:
                self.play(FadeIn(headline, shift=UP * 0.2), run_time=intro)
            self.play(FadeIn(left_panel, shift=RIGHT * 0.18), FadeIn(right_panel, shift=LEFT * 0.18), run_time=accent)
            self.play(FadeIn(versus, scale=0.9), run_time=reveal)
            self.wait(settle)
            return

        quote = clamp_text(str(SPEC.get("quote_text") or SPEC.get("emphasis_text") or SPEC.get("headline") or "Key quote"), max_width=10.8, max_font_size=54, min_font_size=26, color=primary)
        quote.move_to(ORIGIN)
        bars = VGroup(
            Rectangle(width=0.22, height=3.2, fill_color=ManimColor(accent_color), fill_opacity=1.0, stroke_width=0),
            Rectangle(width=0.22, height=3.2, fill_color=ManimColor(accent_color), fill_opacity=1.0, stroke_width=0),
        )
        bars.arrange(RIGHT, buff=7.6)
        footer = clamp_text(footer_text_value, max_width=10.8, max_font_size=24, min_font_size=16, color=secondary, weight=MEDIUM)
        footer.next_to(quote, DOWN, buff=0.45)
        if headline_text:
            self.play(FadeIn(headline, shift=UP * 0.2), run_time=intro)
        self.play(FadeIn(bars, scale=0.92), run_time=accent * 0.6)
        self.play(Write(quote), run_time=reveal)
        if footer_text_value:
            self.play(FadeIn(footer, shift=UP * 0.08), run_time=accent * 0.4)
        self.wait(settle)
"""


class ManimRenderer(VisualRenderer):
    name = "manim"

    def render(
        self,
        spec: dict[str, Any],
        render_root: Path,
        width: int,
        height: int,
        fps: float,
    ) -> RenderedAsset:
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
