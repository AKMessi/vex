"""Manim adapter entry point, invoked with a validated JSON shot next to this file."""
from __future__ import annotations

import json
from pathlib import Path

from manim import (
    AnimationGroup, Create, Dot, DOWN, Ellipse, FadeIn, FadeOut, Line, ManimColor, ORIGIN, RIGHT,
    Scene, Succession, Text, Transform, UP, VGroup, VMobject, tempconfig,
)


def render() -> None:
    root = Path(__file__).resolve().parent
    shot = json.loads((root / "shot.json").read_text(encoding="utf-8"))
    colors = shot["palette"]
    duration = shot["duration"]
    aspect = shot["width"] / shot["height"]

    class GeometryShot(Scene):
        def construct(self):
            def label(value):
                line = Text(value, color=ManimColor(colors["text"]), font_size=44)
                if line.width > aspect * 6.6:
                    line.scale_to_fit_width(aspect * 6.6)
                return line.move_to(DOWN * 1.58)

            reveal_labels, change_labels = [], []
            if shot["labels"]:
                before, after = map(label, shot["labels"])
                reveal_labels = [FadeIn(before)]
                change_labels = [FadeOut(before), FadeIn(after)]
            if shot["action"] == "curve_morph":
                x_span, y_span = min(2.9 * aspect, 4.8), 1.52
                origin = UP * .72
                ink = ManimColor(colors["text"])
                accent = ManimColor(colors["accent"])
                axes = VGroup(
                    Line(origin - RIGHT * x_span, origin + RIGHT * x_span, color=ink, stroke_opacity=.32),
                    Line(origin - UP * y_span, origin + UP * y_span, color=ink, stroke_opacity=.32),
                )

                def curve(state):
                    points = [origin + RIGHT * mark[state][0] * x_span + UP * mark[state][1] * y_span
                              for mark in shot["marks"]]
                    # Interpolate the original samples without overshooting their authored bounds.
                    line = VMobject(stroke_color=accent, stroke_width=5).set_points_as_corners(points)
                    return VGroup(line, *(Dot(point, radius=.055, color=accent) for point in points))

                plot = curve("before")
                self.play(FadeIn(axes), FadeIn(plot), *reveal_labels, run_time=duration * .23)
                self.play(Transform(plot, curve("after")), *change_labels, run_time=duration * .55)
                self.wait(duration * .22)
                return
            if shot["action"] == "focus_rays":
                lens_center = UP * .55
                x_span = min(2.65 * aspect, 3.5)
                focal_point = lens_center + RIGHT * x_span * .84
                levels = [mark["before"][1] for mark in shot.get("marks", [])] or [-.8, -.4, 0, .4, .8]
                ray_color = ManimColor(colors["accent"])
                lens = Ellipse(width=min(.48, x_span * .25), height=2.75,
                               color=ManimColor(colors["text"]), fill_opacity=.08).move_to(lens_center)
                rays = []
                for level in levels:
                    pivot = lens_center + UP * level * 1.28
                    rays.append(Succession(
                        Create(Line(pivot - RIGHT * x_span, pivot, color=ray_color, stroke_width=3.8)),
                        Create(Line(pivot, focal_point, color=ray_color, stroke_width=3.8)),
                    ))
                self.play(FadeIn(lens), *reveal_labels, run_time=duration * .20)
                self.play(AnimationGroup(*rays, lag_ratio=.04), run_time=duration * .49)
                self.play(*change_labels, FadeIn(Dot(focal_point, radius=.10, color=ray_color)),
                          run_time=duration * .13)
                self.wait(duration * .18)
                return

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
            else:
                clustered = target if action == "compress" else field
                center = clustered.get_center()
                for mark in clustered:
                    mark.move_to(center + (mark.get_center() - center) * .46)

            self.play(FadeIn(field, shift=UP * 0.16), *reveal_labels, run_time=duration * .24)
            self.play(AnimationGroup(Transform(field, target), *change_labels),
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
