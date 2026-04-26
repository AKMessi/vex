from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from vex_manim.briefs import SceneBrief


@dataclass
class SceneExample:
    example_id: str
    scene_family: str
    tags: list[str]
    summary: str
    manim_features: list[str]
    why_it_works: str
    code_excerpt: str
    source: str = "builtin"

    def to_prompt_block(self) -> str:
        return "\n".join(
            [
                f"Example: {self.example_id} ({self.scene_family})",
                f"Tags: {', '.join(self.tags)}",
                f"Features: {', '.join(self.manim_features)}",
                f"Why it works: {self.why_it_works}",
                "Code excerpt:",
                self.code_excerpt.strip(),
            ]
        )


BUILTIN_SCENE_EXAMPLES: list[SceneExample] = [
    SceneExample(
        example_id="metric_story_tracker",
        scene_family="metric_story",
        tags=["metric_story", "data_graphic", "value_tracker", "camera"],
        summary="A metric reveal that feels alive because the number, bars, and camera all land together.",
        manim_features=["ValueTracker", "DecimalNumber", "BarChart", "LaggedStart", "MovingCameraScene"],
        why_it_works="It animates the claim instead of just typesetting it, and it uses a tracker to give the scene a premium feel.",
        code_excerpt="""
tracker = ValueTracker(1.0)
hero = always_redraw(
    lambda: DecimalNumber(
        tracker.get_value(),
        num_decimal_places=1,
        font_size=82,
        color=self.theme_color("text_primary"),
    ).set_value(tracker.get_value()).move_to(LEFT * 3.2 + DOWN * 0.1)
)
bars = BarChart(
    values=[1.0, 2.2, 3.0],
    bar_names=["Before", "Search", "Vex"],
    y_range=[0, 3.2, 1],
    y_length=3.2,
    x_length=4.2,
    bar_colors=[self.theme_color("panel_stroke"), self.theme_color("accent_secondary"), self.theme_color("accent")],
)
bars.move_to(RIGHT * 2.8 + DOWN * 0.25)
self.play(LaggedStart(FadeIn(hero), DrawBorderThenFill(bars), lag_ratio=0.18), run_time=0.9)
self.play(tracker.animate.set_value(3.0), self.camera_frame.animate.scale(0.92).move_to(bars), run_time=0.8)
""",
    ),
    SceneExample(
        example_id="metric_story_shapes",
        scene_family="metric_story",
        tags=["metric_story", "data_graphic", "value_tracker", "latex_free"],
        summary="A metric reveal that stays premium without TeX by building the comparison from live text, axes, and manually animated bars.",
        manim_features=["ValueTracker", "Axes", "always_redraw", "LaggedStart", "MovingCameraScene"],
        why_it_works="It keeps the scene dynamic and data-driven while staying safe on runtimes without a LaTeX toolchain.",
        code_excerpt="""
tracker = ValueTracker(1.0)
axis = Axes(
    x_range=[0, 4, 1],
    y_range=[0, 3.5, 1],
    x_length=4.4,
    y_length=3.2,
    axis_config={"include_ticks": False, "include_numbers": False, "color": self.theme_color("grid")},
).move_to(RIGHT * 2.8 + DOWN * 0.2)
labels = VGroup(*[
    self.fit_text(label, max_width=1.1, max_font_size=18)
    for label in ["Before", "Search", "Vex"]
])
for index, label in enumerate(labels, start=1):
    label.next_to(axis.c2p(index, 0), DOWN, buff=0.24)
bars = VGroup(*[
    always_redraw(
        lambda index=index, color=color: Rectangle(
            width=0.7,
            height=max(0.18, tracker.get_value() if index == 3 else [1.0, 2.0][index - 1]),
            fill_color=color,
            fill_opacity=0.92,
            stroke_width=0,
        ).move_to(axis.c2p(index, 0), DOWN).shift(UP * max(0.09, (tracker.get_value() if index == 3 else [1.0, 2.0][index - 1]) / 2))
    )
    for index, color in [(1, self.theme_color("panel_stroke")), (2, self.theme_color("accent_secondary")), (3, self.theme_color("accent"))]
])
hero = always_redraw(lambda: self.fit_text(f"{tracker.get_value():.1f}x faster", max_width=4.6, max_font_size=78).move_to(LEFT * 3.0))
self.play(FadeIn(hero), Create(axis), FadeIn(labels), LaggedStart(*[FadeIn(bar) for bar in bars], lag_ratio=0.08), run_time=0.9)
self.play(tracker.animate.set_value(3.0), self.camera.frame.animate.scale(0.92).move_to(axis), run_time=0.8)
""",
    ),
    SceneExample(
        example_id="system_map_pan",
        scene_family="system_map",
        tags=["system_map", "process", "camera", "connectors"],
        summary="A connected process map with guided camera motion and staged arrows.",
        manim_features=["MovingCameraScene", "CurvedArrow", "LaggedStart", "TracedPath", "always_redraw"],
        why_it_works="The camera guides attention across the chain so the viewer reads the process in the right order.",
        code_excerpt="""
nodes = VGroup(*[
    self.make_signal_node(label, number=index + 1)
    for index, label in enumerate(["Capture", "Score", "Generate", "Composite"])
]).arrange(RIGHT, buff=1.1)
connectors = VGroup(*[
    self.make_connector(nodes[i], nodes[i + 1], curved=True)
    for i in range(len(nodes) - 1)
])
path_glow = TracedPath(nodes[1].get_center, stroke_color=self.theme_color("accent_secondary"), stroke_width=4)
self.add(path_glow)
self.play(LaggedStart(*[GrowFromCenter(node) for node in nodes], lag_ratio=0.14), run_time=0.9)
self.play(LaggedStart(*[Create(connector) for connector in connectors], lag_ratio=0.1), run_time=0.8)
self.play(self.camera_frame.animate.scale(0.88).move_to(nodes[2]), run_time=0.7)
""",
    ),
    SceneExample(
        example_id="comparison_morph_shift",
        scene_family="comparison_morph",
        tags=["comparison_morph", "comparison_split", "transform"],
        summary="A before/after scene that morphs one layout into another instead of cutting between two static cards.",
        manim_features=["TransformMatchingShapes", "ReplacementTransform", "MovingCameraScene", "LaggedStart"],
        why_it_works="Morphing shapes makes the change itself the visual story.",
        code_excerpt="""
before = VGroup(
    self.make_pill("Manual"),
    self.fit_text("Search the timeline by hand", max_width=3.4, max_font_size=28),
).arrange(DOWN, buff=0.28)
after = VGroup(
    self.make_pill("Guided", fill=self.theme_color("accent")),
    self.fit_text("Score transcript beats first", max_width=3.4, max_font_size=28),
).arrange(DOWN, buff=0.28)
panel = self.make_glass_panel(4.2, 2.7)
before.move_to(panel.get_center())
self.play(FadeIn(panel), FadeIn(before), run_time=0.7)
self.play(TransformMatchingShapes(before, after), self.camera_frame.animate.scale(0.94), run_time=0.8)
""",
    ),
    SceneExample(
        example_id="kinetic_quote_focus",
        scene_family="kinetic_quote",
        tags=["kinetic_quote", "quote_focus", "text"],
        summary="A quote scene that feels editorial because the typography and underline choreography are doing real work.",
        manim_features=["LaggedStart", "FadeTransform", "Underline", "MovingCameraScene"],
        why_it_works="It treats the line like a statement worth staging rather than a generic title card.",
        code_excerpt="""
quote = self.fit_text("Specific beats beat generic motion", max_width=9.2, max_font_size=60)
underline = Underline(quote, color=self.theme_color("accent"), stroke_width=6).shift(DOWN * 0.08)
kicker = self.make_pill("INSIGHT")
self.play(FadeIn(kicker, shift=UP * 0.1), Write(quote), run_time=0.8)
self.play(Create(underline), self.camera_frame.animate.scale(0.95).move_to(quote), run_time=0.6)
""",
    ),
    SceneExample(
        example_id="timeline_journey_path",
        scene_family="timeline_journey",
        tags=["timeline_journey", "timeline_steps", "path"],
        summary="A timeline that feels like movement across a route instead of a row of boxes.",
        manim_features=["NumberLine", "MoveAlongPath", "LaggedStart", "Succession"],
        why_it_works="The marker traveling the path gives the process a clear sense of progression.",
        code_excerpt="""
line = NumberLine(x_range=[0, 3, 1], length=8.6, include_numbers=False, color=self.theme_color("accent_secondary"))
labels = VGroup(*[
    self.fit_text(label, max_width=1.5, max_font_size=22)
    for label in ["Capture", "Score", "Render", "Composite"]
]).arrange(RIGHT, buff=1.1).next_to(line, UP, buff=0.5)
marker = Dot(line.n2p(0), radius=0.12, color=self.theme_color("accent"))
self.play(Create(line), FadeIn(labels), run_time=0.7)
self.play(MoveAlongPath(marker, line), LaggedStart(*[Indicate(label) for label in labels], lag_ratio=0.18), run_time=1.0)
""",
    ),
    SceneExample(
        example_id="interface_focus_zoom",
        scene_family="interface_focus",
        tags=["interface_focus", "product_ui", "camera"],
        summary="A product-style focus scene with framed modules and camera punch-ins.",
        manim_features=["MovingCameraScene", "SurroundingRectangle", "LaggedStart", "FadeTransform"],
        why_it_works="The camera and focus ring make the composition feel like a premium walkthrough instead of a flat dashboard mock.",
        code_excerpt="""
panels = VGroup(*[
    self.make_glass_panel(2.4, 1.6) for _ in range(3)
]).arrange(RIGHT, buff=0.36)
labels = VGroup(*[
    self.fit_text(label, max_width=1.7, max_font_size=22)
    for label in ["Transcript", "Planner", "Renderer"]
])
for panel, label in zip(panels, labels):
    label.move_to(panel.get_center())
focus = always_redraw(lambda: SurroundingRectangle(labels[1], buff=0.22, color=self.theme_color("accent"), stroke_width=4))
self.play(LaggedStart(*[FadeIn(panel) for panel in panels], *[FadeIn(label) for label in labels], lag_ratio=0.08), run_time=0.8)
self.add(focus)
self.play(self.camera_frame.animate.scale(0.82).move_to(panels[1]), run_time=0.7)
""",
    ),
]


def _score_example(brief: SceneBrief, example: SceneExample) -> float:
    score = 0.0
    if example.scene_family == brief.scene_family:
        score += 4.0
    score += len(set(example.tags) & set(brief.example_tags)) * 1.2
    score += len(set(example.manim_features) & set(brief.preferred_manim_features)) * 0.35
    if brief.visual_type_hint in example.tags:
        score += 1.5
    return score


def _history_examples(history_roots: Iterable[Path]) -> list[SceneExample]:
    examples: list[SceneExample] = []
    for root in history_roots:
        if not root.exists():
            continue
        for report_path in root.rglob("generation_report.json"):
            try:
                payload = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            brief = payload.get("scene_brief") or {}
            scene_code = str(payload.get("final_scene_code") or "").strip()
            if not scene_code:
                continue
            examples.append(
                SceneExample(
                    example_id=f"history::{report_path.parent.name}",
                    scene_family=str(brief.get("scene_family") or "metric_story"),
                    tags=list(brief.get("example_tags") or []),
                    summary=str(payload.get("summary") or "Successful prior generated scene."),
                    manim_features=list(payload.get("final_features") or []),
                    why_it_works="Previously generated scene that passed validation and render QA.",
                    code_excerpt=scene_code[:2200],
                    source=str(report_path),
                )
            )
    return examples


def retrieve_scene_examples(
    brief: SceneBrief,
    *,
    history_roots: Iterable[Path] | None = None,
    limit: int = 3,
    forbidden_features: Iterable[str] | None = None,
) -> list[SceneExample]:
    candidates = list(BUILTIN_SCENE_EXAMPLES)
    if history_roots:
        candidates.extend(_history_examples(history_roots))
    blocked = {str(feature).strip() for feature in (forbidden_features or []) if str(feature).strip()}
    if blocked:
        candidates = [
            example
            for example in candidates
            if not blocked.intersection(example.manim_features)
        ]
    ranked = sorted(candidates, key=lambda item: (_score_example(brief, item), item.source == "builtin"), reverse=True)
    picked: list[SceneExample] = []
    seen_sources: set[str] = set()
    for example in ranked:
        if example.source in seen_sources:
            continue
        picked.append(example)
        seen_sources.add(example.source)
        if len(picked) >= limit:
            break
    return picked
