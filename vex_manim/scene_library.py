from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
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
    boxy: bool = False

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
self.register_layout_group("hero_metric", hero, role="metric")
self.register_layout_group("bars", bars, role="chart")
self.play(LaggedStart(FadeIn(hero), DrawBorderThenFill(bars), lag_ratio=0.18), run_time=0.9)
self.play(tracker.animate.set_value(3.0), self.camera.frame.animate.scale(0.92).move_to(bars), run_time=0.8)
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
self.register_layout_group("hero_metric", hero, role="metric")
self.register_layout_group("axis_bundle", VGroup(axis, labels, bars), role="chart")
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
self.register_layout_group("flow_nodes", nodes, role="chart")
self.add(path_glow)
self.play(LaggedStart(*[GrowFromCenter(node) for node in nodes], lag_ratio=0.14), run_time=0.9)
self.play(LaggedStart(*[Create(connector) for connector in connectors], lag_ratio=0.1), run_time=0.8)
self.play(self.camera.frame.animate.scale(0.88).move_to(nodes[2]), run_time=0.7)
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
self.register_layout_group("comparison_panel", panel, role="panel")
self.register_layout_group("comparison_copy", before, role="hero")
self.play(FadeIn(panel), FadeIn(before), run_time=0.7)
self.play(TransformMatchingShapes(before, after), self.camera.frame.animate.scale(0.94), run_time=0.8)
""",
        boxy=True,
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
self.register_layout_group("quote_block", VGroup(kicker, quote, underline), role="quote")
self.play(FadeIn(kicker, shift=UP * 0.1), Write(quote), run_time=0.8)
self.play(Create(underline), self.camera.frame.animate.scale(0.95).move_to(quote), run_time=0.6)
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
self.register_layout_group("timeline_bundle", VGroup(line, labels, marker), role="chart")
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
self.register_layout_group("ui_panels", panels, role="panel")
self.register_layout_group("ui_focus", VGroup(labels, focus), role="chart")
self.play(LaggedStart(*[FadeIn(panel) for panel in panels], *[FadeIn(label) for label in labels], lag_ratio=0.08), run_time=0.8)
self.add(focus)
self.play(self.camera.frame.animate.scale(0.82).move_to(panels[1]), run_time=0.7)
""",
        boxy=True,
    ),
    SceneExample(
        example_id="data_journey_sweep",
        scene_family="metric_story",
        tags=["metric_story", "data_graphic", "data_journey", "route", "camera"],
        summary="A premium metric reveal that uses a tracked journey across data instead of stacking cards.",
        manim_features=["ValueTracker", "Axes", "MoveAlongPath", "always_redraw", "MovingCameraScene"],
        why_it_works="The metric is tied to a moving signal on a path, so the change feels discovered rather than merely announced.",
        code_excerpt="""
axis = Axes(
    x_range=[0, 4, 1],
    y_range=[0, 3.4, 1],
    x_length=5.2,
    y_length=3.1,
    axis_config={"include_ticks": False, "include_numbers": False, "color": self.theme_color("grid")},
).move_to(RIGHT * 1.7 + DOWN * 0.3)
path = axis.plot(lambda value: 0.45 + 0.78 * value, x_range=[0.5, 3.2], color=self.theme_color("accent_secondary"), stroke_width=5)
tracker = ValueTracker(0.5)
pulse = always_redraw(lambda: self.make_glow_dot(color=self.theme_color("accent")).move_to(axis.c2p(tracker.get_value(), 0.45 + 0.78 * tracker.get_value())))
hero = always_redraw(lambda: self.make_metric_badge(f"{tracker.get_value():.1f}x", width=2.2).move_to(LEFT * 4.0 + UP * 1.8))
self.register_layout_group("data_path", VGroup(axis, path, pulse), role="chart")
self.register_layout_group("hero_metric", hero, role="metric")
self.play(Create(axis), Create(path), FadeIn(hero), run_time=0.75)
self.add(pulse)
self.play(tracker.animate.set_value(3.0), self.camera.frame.animate.scale(0.9).move_to(path), run_time=0.9)
""",
    ),
    SceneExample(
        example_id="signal_network_orbit",
        scene_family="system_map",
        tags=["system_map", "process", "signal_network", "orbit", "camera"],
        summary="A premium system map built around orbital flow, guided paths, and a travelling pulse.",
        manim_features=["MovingCameraScene", "CurvedArrow", "TracedPath", "LaggedStart", "always_redraw"],
        why_it_works="The flow reads directionally and cinematically because the pulse, rings, and camera all reinforce the same path.",
        code_excerpt="""
hub = self.make_signal_node("Planner", number=2).move_to(ORIGIN + UP * 0.15)
left = self.make_signal_node("Transcript", number=1).move_to(LEFT * 3.0 + DOWN * 0.8)
right = self.make_signal_node("Render", number=3).move_to(RIGHT * 3.2 + UP * 0.9)
orbit = self.make_orbit_ring(2.8, color=self.theme_color("accent_secondary"), arc_angle=4.5, start_angle=-1.5).move_to(hub)
links = VGroup(
    self.make_route_path(left.get_right(), hub.get_left(), angle=0.25),
    self.make_route_path(hub.get_right(), right.get_left(), angle=-0.22),
)
pulse = self.make_glow_dot(color=self.theme_color("accent"))
trail = TracedPath(pulse.get_center, stroke_color=self.theme_color("accent"), stroke_width=4)
self.register_layout_group("network_nodes", VGroup(left, hub, right, orbit, links), role="chart")
self.add(trail)
self.play(LaggedStart(FadeIn(left), FadeIn(hub), FadeIn(right), Create(orbit), lag_ratio=0.12), run_time=0.85)
self.play(LaggedStart(*[Create(link) for link in links], lag_ratio=0.1), run_time=0.65)
self.add(pulse)
self.play(MoveAlongPath(pulse, links[0]), MoveAlongPath(pulse.copy(), links[1]), self.camera.frame.animate.scale(0.88).move_to(hub), run_time=0.9)
""",
    ),
    SceneExample(
        example_id="kinetic_route_curve",
        scene_family="timeline_journey",
        tags=["timeline_journey", "kinetic_route", "path", "camera"],
        summary="A route-based sequence that feels guided and premium rather than a row of repeated cards.",
        manim_features=["MoveAlongPath", "LaggedStart", "Succession", "MovingCameraScene", "TracedPath"],
        why_it_works="The route geometry turns a process into a visible journey with pace, hierarchy, and direction.",
        code_excerpt="""
route = self.make_route_path(LEFT * 5.0 + DOWN * 1.3, RIGHT * 4.6 + UP * 1.0, angle=0.36, stroke_width=5)
steps = VGroup(*[
    self.make_ribbon_label(label, max_width=2.1)
    for label in ["Capture", "Score", "Generate", "Composite"]
])
anchors = [route.point_from_proportion(value) for value in (0.08, 0.34, 0.63, 0.9)]
for label, anchor in zip(steps, anchors):
    label.move_to(anchor + UP * 0.68)
marker = self.make_glow_dot(color=self.theme_color("accent"))
marker.move_to(anchors[0])
trail = TracedPath(marker.get_center, stroke_color=self.theme_color("accent_secondary"), stroke_width=4)
self.register_layout_group("route_bundle", VGroup(route, steps, marker), role="chart")
self.add(trail)
self.play(Create(route), LaggedStart(*[FadeIn(label, shift=UP * 0.08) for label in steps], lag_ratio=0.1), run_time=0.8)
self.play(MoveAlongPath(marker, route), self.camera.frame.animate.scale(0.9).move_to(anchors[2]), run_time=1.0)
""",
    ),
    SceneExample(
        example_id="spotlight_compare_beam",
        scene_family="comparison_morph",
        tags=["comparison_morph", "spotlight_compare", "transform", "camera"],
        summary="A contrast scene that stages the change with ribbons, a focus beam, and matched-shape morphing instead of box cards.",
        manim_features=["TransformMatchingShapes", "FadeTransform", "LaggedStart", "MovingCameraScene"],
        why_it_works="The viewer tracks the actual shift in wording and emphasis, not a decorative panel swap.",
        code_excerpt="""
before = VGroup(
    self.make_ribbon_label("Manual Search", max_width=3.0),
    self.fit_text("hunt through the timeline", max_width=3.4, max_font_size=28),
).arrange(DOWN, buff=0.22).move_to(LEFT * 3.1 + DOWN * 0.25)
after = VGroup(
    self.make_ribbon_label("Beat Scoring", max_width=3.0, accent=self.theme_color("accent_secondary")),
    self.fit_text("rank the best visual moment", max_width=3.5, max_font_size=28),
).arrange(DOWN, buff=0.22).move_to(RIGHT * 2.7 + UP * 0.2)
beam = self.make_focus_beam(4.8, 0.5, color=self.theme_color("accent"), opacity=0.16)
beam.move_to(ORIGIN + DOWN * 0.15)
bridge = self.make_route_path(before.get_right(), after.get_left(), angle=-0.18)
self.register_layout_group("compare_words", VGroup(before, after, beam, bridge), role="chart")
self.play(FadeIn(beam), FadeIn(before, shift=RIGHT * 0.1), run_time=0.55)
self.play(Create(bridge), FadeIn(after, shift=LEFT * 0.1), run_time=0.55)
self.play(TransformMatchingShapes(before.copy(), after), self.camera.frame.animate.scale(0.92).move_to(after), run_time=0.85)
""",
    ),
    SceneExample(
        example_id="ribbon_quote_sweep",
        scene_family="kinetic_quote",
        tags=["kinetic_quote", "ribbon_quote", "text", "motion"],
        summary="A premium quote treatment that relies on kinetic type, directional ribbons, and moving emphasis rather than a static panel.",
        manim_features=["LaggedStart", "FadeTransform", "TransformMatchingShapes", "MovingCameraScene"],
        why_it_works="The statement feels authored because the emphasis moves across the phrase instead of just appearing as text on a card.",
        code_excerpt="""
headline = self.fit_text("Specific beats beat generic motion", max_width=9.0, max_font_size=62).move_to(UP * 0.18)
ribbon = self.make_ribbon_label("INSIGHT", max_width=2.0, accent=self.theme_color("accent"))
ribbon.move_to(LEFT * 4.1 + UP * 1.55)
sweep = self.make_focus_beam(6.4, 0.34, color=self.theme_color("accent_secondary"), opacity=0.14, angle=0.0)
sweep.move_to(DOWN * 0.62)
marker = self.make_glow_dot(color=self.theme_color("accent")).move_to(headline.get_left() + RIGHT * 0.1 + DOWN * 0.42)
self.register_layout_group("quote_stage", VGroup(ribbon, headline, sweep, marker), role="quote")
self.play(FadeIn(ribbon, shift=UP * 0.12), Write(headline), run_time=0.75)
self.play(FadeIn(sweep), marker.animate.shift(RIGHT * 4.8), self.camera.frame.animate.scale(0.95).move_to(headline), run_time=0.7)
""",
    ),
]


def _score_example(
    brief: SceneBrief,
    example: SceneExample,
    *,
    preferred_tags: set[str],
    preferred_features: set[str],
) -> float:
    score = 0.0
    if example.scene_family == brief.scene_family:
        score += 4.0
    score += len(set(example.tags) & set(brief.example_tags)) * 1.2
    score += len(set(example.manim_features) & set(brief.preferred_manim_features)) * 0.35
    score += len(set(example.tags) & preferred_tags) * 0.75
    score += len(set(example.manim_features) & preferred_features) * 0.45
    if brief.visual_type_hint in example.tags:
        score += 1.5
    if example.boxy and brief.scene_family != "interface_focus" and brief.composition_mode == "replace":
        score -= 1.6
    return score


def _normalized_history_roots(history_roots: Iterable[Path]) -> tuple[str, ...]:
    normalized = {
        str(Path(root).resolve())
        for root in history_roots
        if str(root).strip()
    }
    return tuple(sorted(normalized))


def _history_scene_is_reusable(payload: dict[str, Any]) -> bool:
    if bool(payload.get("fallback_used")):
        return False
    quality_score = payload.get("quality_score")
    try:
        quality_value = float(quality_score)
    except (TypeError, ValueError):
        return False
    if quality_value < 0.9:
        return False
    scene_code = str(payload.get("final_scene_code") or "").strip()
    if not scene_code:
        return False
    panel_count = scene_code.count("make_glass_panel") + scene_code.count("RoundedRectangle(")
    if panel_count >= 2:
        return False
    dynamic_markers = [
        "ValueTracker(",
        "always_redraw(",
        "TransformMatchingShapes(",
        "MoveAlongPath(",
        "TracedPath(",
        "MovingCameraScene",
    ]
    if not any(marker in scene_code for marker in dynamic_markers):
        return False
    return True


@lru_cache(maxsize=12)
def _history_examples_cached(history_roots: tuple[str, ...]) -> tuple[SceneExample, ...]:
    examples: list[SceneExample] = []
    for root_value in history_roots:
        root = Path(root_value)
        if not root.exists():
            continue
        report_paths = [
            path
            for path in root.rglob("generation_report.json")
            if "_manim_cache" not in path.parts and "__pycache__" not in path.parts
        ]
        report_paths.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        for report_path in report_paths[:48]:
            try:
                payload = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict) or not _history_scene_is_reusable(payload):
                continue
            brief = payload.get("scene_brief") or {}
            scene_code = str(payload.get("final_scene_code") or "").strip()
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
                    boxy=scene_code.count("make_glass_panel") + scene_code.count("RoundedRectangle(") >= 3,
                )
            )
    return tuple(examples)


def _history_examples(history_roots: Iterable[Path]) -> list[SceneExample]:
    return list(_history_examples_cached(_normalized_history_roots(history_roots)))


def retrieve_scene_examples(
    brief: SceneBrief,
    *,
    history_roots: Iterable[Path] | None = None,
    limit: int = 3,
    forbidden_features: Iterable[str] | None = None,
    preferred_tags: Iterable[str] | None = None,
    preferred_features: Iterable[str] | None = None,
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
    if brief.composition_mode == "replace" and brief.scene_family != "interface_focus":
        premium_candidates = [example for example in candidates if not example.boxy]
        if premium_candidates:
            candidates = premium_candidates
    preferred_tag_set = {str(tag).strip().lower() for tag in (preferred_tags or []) if str(tag).strip()}
    preferred_feature_set = {str(feature).strip() for feature in (preferred_features or []) if str(feature).strip()}
    ranked = sorted(
        candidates,
        key=lambda item: (
            _score_example(
                brief,
                item,
                preferred_tags=preferred_tag_set,
                preferred_features=preferred_feature_set,
            ),
            item.source == "builtin",
        ),
        reverse=True,
    )
    picked: list[SceneExample] = []
    seen_examples: set[tuple[str, str]] = set()
    for example in ranked:
        dedupe_key = (example.source, example.example_id)
        if dedupe_key in seen_examples:
            continue
        picked.append(example)
        seen_examples.add(dedupe_key)
        if len(picked) >= limit:
            break
    return picked
