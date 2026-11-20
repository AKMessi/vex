"""Manim adapter entry point, invoked with a validated JSON shot next to this file."""
from __future__ import annotations

import json
from pathlib import Path

from manim import (
    AnimationGroup, Dot, DOWN, FadeIn, FadeOut, ManimColor, ORIGIN, RIGHT,
    Scene, Text, Transform, UP, VGroup, tempconfig,
)


def render() -> None:
    root = Path(__file__).resolve().parent
    shot = json.loads((root / "shot.json").read_text(encoding="utf-8"))
    colors = shot["palette"]
    duration = shot["duration"]
    aspect = shot["width"] / shot["height"]

    class GeometryShot(Scene):
        def construct(self):
            span = min(aspect * 3.2, 4.6)
            field = VGroup(*(
                Dot(RIGHT * (x - 1.5) * span / 2.4 + UP * (y - 1) * 0.6,
                    radius=min(0.11, span * 0.055),
                    color=ManimColor(colors["accent"] if (x + y) % 2 == 0 else colors["text"]))
                for y in range(3) for x in range(4)
            )).move_to(ORIGIN + UP * 0.38)
            target = field.copy()
            action = shot["action"]
            if shot.get("marks"):
                def points(state):
                    return VGroup(*(
                        Dot(RIGHT * mark[state][0] * aspect * 2.6 + UP * (mark[state][1] * 1.4 + .38),
                            radius=min(.12, aspect * .13),
                            color=ManimColor(colors["accent"] if index % 2 == 0 else colors["text"]))
                        for index, mark in enumerate(shot["marks"])
                    ))

                field, target = points("before"), points("after")
            elif action == "rotate":
                target.rotate(1.18)
            elif action == "compress":
                target.scale(0.46)
            else:
                field.scale(0.46)

            def label(value):
                line = Text(value, color=ManimColor(colors["text"]), font_size=44)
                if line.width > aspect * 6.6:
                    line.scale_to_fit_width(aspect * 6.6)
                return line.move_to(DOWN * 1.58)

            before, after = map(label, shot["labels"])
            self.play(FadeIn(field, shift=UP * 0.16), FadeIn(before), run_time=duration * .24)
            self.play(AnimationGroup(Transform(field, target), FadeOut(before), FadeIn(after)),
                      run_time=duration * .51)
            self.wait(duration * .25)

    with tempconfig({
        "media_dir": str(root / "media"), "output_file": str(root / "native.mp4"),
        "pixel_width": shot["width"], "pixel_height": shot["height"],
        "frame_width": 8 * aspect, "frame_height": 8,
        "frame_rate": shot["fps"], "format": "mp4", "write_to_movie": True,
        "disable_caching": True, "background_color": colors["background"],
    }):
        GeometryShot().render()


if __name__ == "__main__":
    render()
