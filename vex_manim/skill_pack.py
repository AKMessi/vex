from __future__ import annotations

from dataclasses import dataclass

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


def _score_slice(brief: SceneBrief, skill: SkillSlice) -> float:
    score = 0.0
    if not skill.scene_families or brief.scene_family in skill.scene_families:
        score += 2.5 if skill.scene_families else 0.8
    if not skill.visual_types or brief.visual_type_hint in skill.visual_types:
        score += 1.6 if skill.visual_types else 0.5
    if not skill.camera_styles or brief.camera_style in skill.camera_styles:
        score += 1.0 if skill.camera_styles else 0.35
    if not skill.animation_levels or brief.animation_intensity in skill.animation_levels:
        score += 0.9 if skill.animation_levels else 0.2
    score += len(set(skill.manim_features) & set(brief.preferred_manim_features)) * 0.3
    return score


def retrieve_skill_slices(brief: SceneBrief, *, limit: int = 3) -> list[SkillSlice]:
    ranked = sorted(BUILTIN_SKILL_SLICES, key=lambda item: _score_slice(brief, item), reverse=True)
    selected: list[SkillSlice] = []
    seen_ids: set[str] = set()
    for skill in ranked:
        if skill.skill_id in seen_ids:
            continue
        selected.append(skill)
        seen_ids.add(skill.skill_id)
        if len(selected) >= limit:
            break
    return selected
