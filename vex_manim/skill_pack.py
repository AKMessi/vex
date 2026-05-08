from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from vex_manim.briefs import SceneBrief


@dataclass
class SkillSlice:
    skill_id: str
    title: str
    scene_families: tuple[str, ...]
    visual_types: tuple[str, ...]
    camera_styles: tuple[str, ...]
    animation_levels: tuple[str, ...]
    manim_features: tuple[str, ...]
    guidance: tuple[str, ...]
    anti_patterns: tuple[str, ...]
    mandatory: bool = False

    def to_prompt_block(self) -> str:
        lines = [
            f"Skill: {self.skill_id} - {self.title}",
            "Guidance:",
        ]
        lines.extend(f"- {item}" for item in self.guidance)
        if self.anti_patterns:
            lines.append("Avoid:")
            lines.extend(f"- {item}" for item in self.anti_patterns)
        return "\n".join(lines)


BUILTIN_SKILL_SLICES: tuple[SkillSlice, ...] = (
    SkillSlice(
        skill_id="manim-ce-production-contract",
        title="ManimCE Production Contract",
        scene_families=(),
        visual_types=(),
        camera_styles=(),
        animation_levels=(),
        manim_features=("Scene", "MovingCameraScene", "VexGeneratedScene"),
        guidance=(
            "Write for Manim Community Edition only: use `from manim import *` semantics supplied by the wrapper and never ManimGL/manimlib APIs.",
            "Define exactly one `GeneratedScene(VexGeneratedScene)` with a correctly spelled `construct(self)` method; put all animation orchestration behind that method.",
            "Use modern ManimCE object construction and keyword arguments. Set custom attributes directly; do not use legacy `CONFIG` dictionaries.",
            "Call Vex runtime helpers through `self.` and use layout slots for scene-level placement so the same code survives landscape, vertical, and square renders.",
            "Prefer `python -m manim` compatible code paths conceptually; do not rely on notebook magics, CLI globals, filesystem reads, or external mutable state.",
        ),
        anti_patterns=(
            "`from manimlib import *`, `from big_ol_pile_of_manim_imports import *`, `GraphScene`, `CONFIG = {...}`, or 3B1B-only helper names.",
            "Misspelling `construct`, defining helper-only classes with no rendered scene, or producing code that can render a black frame.",
            "Bare calls such as `make_title_block(...)` instead of `self.make_title_block(...)`.",
        ),
        mandatory=True,
    ),
    SkillSlice(
        skill_id="visual-logic-fidelity",
        title="Visual Logic Fidelity",
        scene_families=(),
        visual_types=(),
        camera_styles=(),
        animation_levels=(),
        manim_features=("LaggedStart", "Succession", "ReplacementTransform", "TransformMatchingShapes"),
        guidance=(
            "Before writing code, map each planned beat to a visible event: what appears, what changes, why it changes, and when it happens.",
            "Keep object identity stable across beats. If a concept transforms, animate the transformation instead of deleting the source and fading in an unrelated target.",
            "Use `Succession`, `LaggedStart`, or explicit waits to preserve temporal order; valid code is not enough if the causal sequence is wrong.",
            "Make required semantic events observable: before state, intervention, after state, metric change, or route progression must each have a visual counterpart.",
            "Register principal groups so the runtime can inspect and correct layout; unregistered hero objects are easy to clip or overlap silently.",
        ),
        anti_patterns=(
            "A syntactically valid scene that omits the decisive visual event from the storyboard.",
            "Decorative motion that competes with or contradicts the intended causal relationship.",
            "Timing all animations at once when the idea depends on a sequence.",
        ),
        mandatory=True,
    ),
    SkillSlice(
        skill_id="runtime-safety-and-api-footguns",
        title="Runtime Safety And Manim API Footguns",
        scene_families=(),
        visual_types=(),
        camera_styles=(),
        animation_levels=(),
        manim_features=("Text", "Tex", "MathTex", "ValueTracker", "always_redraw", "Axes", "Transform"),
        guidance=(
            "Use `Text` for normal labels and reserve `Tex`/`MathTex` for mathematical typesetting; when LaTeX is unavailable, avoid TeX-backed mobjects completely.",
            "When using TeX, use raw strings for backslashes and add packages through `TexTemplate` only when a command actually requires them.",
            "Use `ValueTracker` with `always_redraw` or `add_updater` for live geometry; do not freeze tracker values inside `mobject.always` chains.",
            "Understand `.animate`: it interpolates between the current and final point states. For path travel, rotations, or continuously changing geometry, use trackers, updaters, `MoveAlongPath`, `Rotating`, or `Rotate`.",
            "For axes and plotted data, convert data coordinates through `ax.c2p(...)`/`ax.coords_to_point(...)`; do not mix raw scene coordinates with axis coordinates.",
            "Transform mutates the source mobject into the target. Use copies, `ReplacementTransform`, `FadeTransform`, or `TransformMatchingShapes` deliberately when object continuity matters.",
        ),
        anti_patterns=(
            "Using `MathTex` for ordinary prose or when the runtime says LaTeX is unavailable.",
            "Calling `tracker.get_value()` once while expecting a mobject to keep updating automatically.",
            "Placing chart dots at `(x, y, 0)` when they should be positioned with `ax.c2p(x, y)`.",
            "Assuming `Transform(source, target)` leaves both original objects independently visible afterward.",
        ),
        mandatory=True,
    ),
    SkillSlice(
        skill_id="scene-architecture",
        title="Premium Scene Architecture",
        scene_families=(),
        visual_types=(),
        camera_styles=(),
        animation_levels=("medium", "high"),
        manim_features=("MovingCameraScene", "LaggedStart", "FadeTransform", "always_redraw"),
        guidance=(
            "Compose premium scenes in layers: atmosphere in the back, structure in the middle, annotation in the front.",
            "Give the scene one visual spine such as a path, chart, orbit, or morph so the frame has a clear organizing idea.",
            "Use asymmetry and depth to create intention; the scene should feel staged, not tiled.",
        ),
        anti_patterns=(
            "Building the entire frame from repeated panels with copy inside each one.",
            "Treating the headline as the whole scene instead of one layer inside it.",
        ),
    ),
    SkillSlice(
        skill_id="atmospheric-depth",
        title="Atmospheric Depth And Accent Motion",
        scene_families=("metric_story", "system_map", "timeline_journey", "comparison_morph", "kinetic_quote"),
        visual_types=("data_graphic", "process", "abstract_motion"),
        camera_styles=(),
        animation_levels=("medium", "high"),
        manim_features=("always_redraw", "MoveAlongPath", "TracedPath", "FadeTransform"),
        guidance=(
            "Add one restrained atmosphere layer such as a focus beam, orbital ring, route arc, or travelling glow so the scene has depth.",
            "Use accent motion to support the hero action, not to decorate empty space.",
            "Let the background and foreground rhyme with the main geometry so the whole shot feels authored.",
        ),
        anti_patterns=(
            "Leaving the background completely inert while the scene tries to feel cinematic.",
            "Throwing decorative glows everywhere with no compositional purpose.",
        ),
    ),
    SkillSlice(
        skill_id="text-economy",
        title="Text Economy For Motion Design",
        scene_families=(),
        visual_types=(),
        camera_styles=(),
        animation_levels=(),
        manim_features=("Text", "FadeTransform", "TransformMatchingShapes"),
        guidance=(
            "Compress copy aggressively: labels, chips, numerals, and short phrases read better in motion than full sentences.",
            "Keep the visible words low enough that the viewer can read them instantly while tracking the animation.",
            "If a phrase is too long, split it into a hero phrase plus one supporting line instead of one wide paragraph.",
        ),
        anti_patterns=(
            "Using transcript-length sentences as labels.",
            "Stacking multiple long text blocks in one frame.",
        ),
    ),
    SkillSlice(
        skill_id="layout-discipline",
        title="Layout Discipline And Safe Framing",
        scene_families=(),
        visual_types=(),
        camera_styles=(),
        animation_levels=(),
        manim_features=(),
        guidance=(
            "Keep the title treatment in the top editorial band unless the scene uses a stronger anchored framing.",
            "Register a title or hero group plus one or two supporting groups so runtime guardrails can rebalance the frame.",
            "Use asymmetry and negative space instead of filling the frame with evenly sized cards.",
            "Keep important text out of the bottom subtitle-safe zone and keep copy blocks comfortably narrower than the full frame.",
        ),
        anti_patterns=(
            "Stacking four identical centered rectangles.",
            "Letting labels touch bars, nodes, or arrows without padding.",
            "Using long transcript sentences as the headline.",
        ),
    ),
    SkillSlice(
        skill_id="metric-story",
        title="Quantitative Storytelling",
        scene_families=("metric_story", "dashboard_build"),
        visual_types=("data_graphic",),
        camera_styles=(),
        animation_levels=("medium", "high"),
        manim_features=("ValueTracker", "Axes", "always_redraw", "LaggedStart", "MovingCameraScene"),
        guidance=(
            "Link the hero number to a changing visual so the metric feels earned rather than merely typeset.",
            "Use trackers, axes, manual bars, or comparative geometry to express change over time.",
            "Stage the metric and the evidence in separate zones, then use camera punch-ins or transforms to connect them.",
            "Keep labels short and literal: the chart should clarify the narration, not restate it.",
        ),
        anti_patterns=(
            "A giant number floating alone with no supporting structure.",
            "Fake dashboard clutter with too many tiny stats.",
        ),
    ),
    SkillSlice(
        skill_id="process-choreography",
        title="Process Maps And Guided Camera Motion",
        scene_families=("system_map", "timeline_journey"),
        visual_types=("process",),
        camera_styles=("guided", "punch_in"),
        animation_levels=("medium", "high"),
        manim_features=("MovingCameraScene", "MoveAlongPath", "CurvedArrow", "TracedPath", "LaggedStart"),
        guidance=(
            "Make the sequence directional: nodes, connectors, or a traveling marker should show where the viewer goes next.",
            "Use guided camera reframing to reveal the process in the intended reading order.",
            "Prefer a few strong stages with clear spacing over many tiny steps.",
            "Animate the flow itself with path motion, tracer glow, or staged arrows so the system feels alive.",
        ),
        anti_patterns=(
            "Disconnected cards that claim to be a workflow.",
            "Centering every node at equal weight when one stage should lead attention.",
        ),
    ),
    SkillSlice(
        skill_id="comparison-morph",
        title="Comparison Through Morphing",
        scene_families=("comparison_morph",),
        visual_types=("product_ui", "process"),
        camera_styles=("guided", "punch_in"),
        animation_levels=("medium", "high"),
        manim_features=("TransformMatchingShapes", "ReplacementTransform", "LaggedStart", "MovingCameraScene"),
        guidance=(
            "Use morphs or replacements so the transition itself explains the difference between the two states.",
            "Anchor the before and after layouts to a common structure so the viewer can track what changed.",
            "Let the winning state arrive cleaner, brighter, or more focused rather than simply appearing on the other side.",
        ),
        anti_patterns=(
            "Two static cards with a literal VS divider and no visual evolution.",
            "Overloading both sides with equal paragraph-sized copy.",
        ),
    ),
    SkillSlice(
        skill_id="interface-focus",
        title="Premium Interface Focus",
        scene_families=("interface_focus",),
        visual_types=("product_ui",),
        camera_styles=("guided", "punch_in"),
        animation_levels=("medium", "high"),
        manim_features=("MovingCameraScene", "SurroundingRectangle", "FadeTransform", "LaggedStart"),
        guidance=(
            "Build interface modules in depth and use focus rings or subtle camera punch-ins to guide attention.",
            "Keep UI labels concise and use grouped panels rather than a flat wall of elements.",
            "Let one focused module become the hero so the scene feels like a walkthrough, not a dashboard screenshot.",
        ),
        anti_patterns=(
            "Static boxes pretending to be UI.",
            "No focus state or camera emphasis on the important module.",
        ),
    ),
    SkillSlice(
        skill_id="motion-spine",
        title="Motion Spine And Non-Box Composition",
        scene_families=("metric_story", "system_map", "timeline_journey", "comparison_morph", "kinetic_stack", "kinetic_quote"),
        visual_types=(),
        camera_styles=(),
        animation_levels=("medium", "high"),
        manim_features=("MoveAlongPath", "TracedPath", "TransformMatchingShapes", "always_redraw", "MovingCameraScene"),
        guidance=(
            "Build the shot around one motion spine: a route, orbit, bridge, ladder, or sweep that tells the eye where to go next.",
            "Let that spine control supporting objects so the scene reads like choreography, not like scattered widgets.",
            "When possible, make the primary copy ride along the spine as badges, ribbons, checkpoints, or morph targets instead of sitting inside panels.",
        ),
        anti_patterns=(
            "Replacing the motion spine with a wall of boxes once the scene gets complicated.",
            "Putting every idea into its own isolated card.",
        ),
    ),
    SkillSlice(
        skill_id="camera-authoring",
        title="Camera As A Design Tool",
        scene_families=("metric_story", "system_map", "timeline_journey", "comparison_morph", "interface_focus", "kinetic_quote"),
        visual_types=(),
        camera_styles=("guided", "punch_in"),
        animation_levels=("medium", "high"),
        manim_features=("MovingCameraScene", "LaggedStart", "FadeTransform"),
        guidance=(
            "Use the camera to reveal sequence and hierarchy: wide to orient, punch in to prove, settle to land the idea.",
            "A camera move should follow a meaningful object or state change, not wander independently.",
            "Micro-reframes are often better than giant zooms; they keep the scene premium and controlled.",
        ),
        anti_patterns=(
            "Leaving the camera static while the scene tries to feel cinematic.",
            "Huge zooms that create layout chaos instead of focus.",
        ),
    ),
    SkillSlice(
        skill_id="anti-panel-premium",
        title="Anti-Panel Premium Bias",
        scene_families=(),
        visual_types=(),
        camera_styles=(),
        animation_levels=("medium", "high"),
        manim_features=("TransformMatchingShapes", "MoveAlongPath", "CurvedArrow", "Axes"),
        guidance=(
            "Before using a panel, ask whether the idea could be shown as geometry, a route, a beam, a morph, or a tracked object instead.",
            "Reserve panels for true interface modules or when a bounded surface is semantically meaningful.",
            "If a panel exists, make it secondary to the actual motion system of the scene.",
        ),
        anti_patterns=(
            "Turning every beat into three rounded rectangles with text.",
            "Using cards as a substitute for hierarchy, pacing, or composition.",
        ),
    ),
    SkillSlice(
        skill_id="kinetic-type",
        title="Kinetic Typography With Restraint",
        scene_families=("kinetic_quote", "kinetic_stack"),
        visual_types=("abstract_motion",),
        camera_styles=(),
        animation_levels=("low", "medium"),
        manim_features=("LaggedStart", "FadeTransform", "TransformMatchingShapes", "Underline"),
        guidance=(
            "Treat typography as choreography: reveals, underlines, morphs, and staggered emphasis should support a memorable phrase.",
            "Keep copy extremely distilled and stage one statement at a time.",
            "Use one strong accent device such as an underline, motion trail, or morph instead of many decorative effects.",
        ),
        anti_patterns=(
            "Paragraphs of text floating on top of glass cards.",
            "Using quote scenes for vague filler beats that do not deserve emphasis.",
        ),
    ),
    SkillSlice(
        skill_id="latex-free",
        title="LaTeX-Free Runtime Patterns",
        scene_families=(),
        visual_types=(),
        camera_styles=(),
        animation_levels=(),
        manim_features=("Axes", "always_redraw", "Text", "Rectangle"),
        guidance=(
            "When LaTeX is unavailable, build numeric and chart storytelling from Text, Axes, and manually animated geometry.",
            "Prefer Text plus simple formatting for labels instead of TeX-backed number mobjects.",
            "Use custom rectangles, lines, and trackers to keep the scene premium without relying on TeX-dependent helpers.",
        ),
        anti_patterns=(
            "MathTex, Tex, DecimalNumber, Integer, or BarChart when LaTeX is unavailable.",
        ),
    ),
)


def _score_slice(brief: SceneBrief, skill: SkillSlice, *, preferred_features: set[str]) -> float:
    score = 0.0
    if not skill.scene_families or brief.scene_family in skill.scene_families:
        score += 2.5 if skill.scene_families else 0.8
    if not skill.visual_types or brief.visual_type_hint in skill.visual_types:
        score += 1.6 if skill.visual_types else 0.5
    if not skill.camera_styles or brief.camera_style in skill.camera_styles:
        score += 1.0 if skill.camera_styles else 0.35
    if not skill.animation_levels or brief.animation_intensity in skill.animation_levels:
        score += 0.9 if skill.animation_levels else 0.2
    score += len(set(skill.manim_features) & preferred_features) * 0.3
    return score


def retrieve_skill_slices(
    brief: SceneBrief,
    *,
    limit: int = 3,
    preferred_features: Iterable[str] | None = None,
) -> list[SkillSlice]:
    feature_set = set(brief.preferred_manim_features)
    feature_set.update(str(feature).strip() for feature in (preferred_features or []) if str(feature).strip())
    mandatory = [skill for skill in BUILTIN_SKILL_SLICES if skill.mandatory]
    ranked = sorted(
        [skill for skill in BUILTIN_SKILL_SLICES if not skill.mandatory],
        key=lambda item: _score_slice(brief, item, preferred_features=feature_set),
        reverse=True,
    )
    selected: list[SkillSlice] = []
    seen_ids: set[str] = set()
    for skill in mandatory:
        if skill.skill_id in seen_ids:
            continue
        selected.append(skill)
        seen_ids.add(skill.skill_id)
    domain_limit = max(0, int(limit))
    domain_count = 0
    for skill in ranked:
        if skill.skill_id in seen_ids:
            continue
        selected.append(skill)
        seen_ids.add(skill.skill_id)
        domain_count += 1
        if domain_count >= domain_limit:
            break
    return selected
