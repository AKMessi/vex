from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


SCENE_FAMILY_BY_TEMPLATE = {
    "data_journey": "metric_story",
    "signal_network": "system_map",
    "kinetic_route": "timeline_journey",
    "spotlight_compare": "comparison_morph",
    "interface_cascade": "interface_focus",
    "ribbon_quote": "kinetic_quote",
    "metric_callout": "metric_story",
    "stat_grid": "dashboard_build",
    "timeline_steps": "timeline_journey",
    "system_flow": "system_map",
    "comparison_split": "comparison_morph",
    "keyword_stack": "kinetic_stack",
    "quote_focus": "kinetic_quote",
}


PREFERRED_FEATURES = {
    "metric_story": ["ValueTracker", "DecimalNumber", "Axes", "BarChart", "LaggedStart", "TransformMatchingShapes"],
    "dashboard_build": ["ValueTracker", "BarChart", "Axes", "LaggedStart", "FadeTransform"],
    "timeline_journey": ["MovingCameraScene", "MoveAlongPath", "LaggedStart", "Succession", "NumberLine"],
    "system_map": ["MovingCameraScene", "CurvedArrow", "TracedPath", "LaggedStart", "always_redraw"],
    "comparison_morph": ["TransformMatchingShapes", "ReplacementTransform", "MovingCameraScene", "LaggedStart"],
    "kinetic_stack": ["LaggedStart", "Succession", "TransformMatchingShapes", "ValueTracker"],
    "kinetic_quote": ["LaggedStart", "FadeTransform", "TransformMatchingShapes", "Underline"],
    "interface_focus": ["MovingCameraScene", "LaggedStart", "SurroundingRectangle", "FadeTransform"],
}


LATEX_FREE_PREFERRED_FEATURES = {
    "metric_story": ["ValueTracker", "Axes", "LaggedStart", "TransformMatchingShapes", "MovingCameraScene", "always_redraw"],
    "dashboard_build": ["ValueTracker", "Axes", "LaggedStart", "FadeTransform", "SurroundingRectangle"],
    "timeline_journey": ["MovingCameraScene", "MoveAlongPath", "LaggedStart", "Succession", "NumberLine"],
    "system_map": ["MovingCameraScene", "CurvedArrow", "TracedPath", "LaggedStart", "always_redraw"],
    "comparison_morph": ["TransformMatchingShapes", "ReplacementTransform", "MovingCameraScene", "LaggedStart"],
    "kinetic_stack": ["LaggedStart", "Succession", "TransformMatchingShapes", "ValueTracker"],
    "kinetic_quote": ["LaggedStart", "FadeTransform", "TransformMatchingShapes", "Underline"],
    "interface_focus": ["MovingCameraScene", "LaggedStart", "SurroundingRectangle", "FadeTransform"],
}


def _scene_family(spec: dict[str, Any]) -> str:
    template = str(spec.get("template") or "").strip().lower()
    visual_type = str(spec.get("visual_type_hint") or "").strip().lower()
    if visual_type == "product_ui":
        return "interface_focus"
    return SCENE_FAMILY_BY_TEMPLATE.get(template, "metric_story")


def _camera_style(scene_family: str, spec: dict[str, Any]) -> str:
    if scene_family in {"timeline_journey", "system_map", "interface_focus", "comparison_morph"}:
        return "guided"
    if float(spec.get("importance") or 0.0) >= 0.75:
        return "punch_in"
    return "composed"


def _animation_intensity(spec: dict[str, Any], scene_family: str) -> str:
    importance = float(spec.get("importance") or 0.5)
    if scene_family in {"system_map", "timeline_journey", "comparison_morph"}:
        return "high" if importance >= 0.72 else "medium"
    if scene_family in {"metric_story", "dashboard_build", "kinetic_stack"}:
        return "medium" if importance >= 0.5 else "low"
    return "medium"


def _collect_must_show_terms(spec: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for item in [
        spec.get("headline"),
        spec.get("deck"),
        spec.get("emphasis_text"),
        *(spec.get("supporting_lines") or []),
        *(spec.get("steps") or []),
        *(spec.get("keywords") or []),
        spec.get("left_detail"),
        spec.get("right_detail"),
    ]:
        text = str(item or "").strip()
        if not text or text.lower() in {value.lower() for value in values}:
            continue
        values.append(text)
        if len(values) >= 10:
            break
    return values


def _must_avoid_terms(spec: dict[str, Any]) -> list[str]:
    avoid = [
        "generic motivational typography",
        "plain centered talking-head replacement",
        "static box-only layout",
        "verbatim transcript repetition",
    ]
    visual_type = str(spec.get("visual_type_hint") or "")
    if visual_type == "data_graphic":
        avoid.append("fake data without numeric grounding")
    if visual_type == "process":
        avoid.append("unconnected cards with no flow")
    return avoid


def _objective(spec: dict[str, Any], scene_family: str) -> str:
    headline = str(spec.get("headline") or spec.get("emphasis_text") or "Key point").strip()
    if scene_family == "comparison_morph":
        return f"Make the change from one state to another instantly legible: {headline}"
    if scene_family in {"system_map", "timeline_journey"}:
        return f"Show the process as a visual journey that matches the narration: {headline}"
    if scene_family in {"metric_story", "dashboard_build"}:
        return f"Turn the spoken claim into a concrete quantitative visual: {headline}"
    if scene_family == "interface_focus":
        return f"Stage the idea like a premium product walkthrough: {headline}"
    return f"Create a high-taste animated visual that clarifies the spoken beat: {headline}"


@dataclass
class SceneBrief:
    visual_id: str
    scene_family: str
    objective: str
    spoken_anchor: str
    context: str
    headline: str
    deck: str
    duration_sec: float
    start_sec: float
    end_sec: float
    composition_mode: str
    visual_type_hint: str
    style_pack: str
    theme: dict[str, str]
    background_motif: str
    layout_variant: str
    camera_style: str
    animation_intensity: str
    must_show_terms: list[str] = field(default_factory=list)
    must_avoid: list[str] = field(default_factory=list)
    preferred_manim_features: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    render_constraints: dict[str, Any] = field(default_factory=dict)
    example_tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_scene_brief(
    spec: dict[str, Any],
    *,
    width: int,
    height: int,
    fps: float,
    latex_available: bool | None = None,
) -> SceneBrief:
    scene_family = _scene_family(spec)
    visual_type = str(spec.get("visual_type_hint") or "general")
    camera_style = _camera_style(scene_family, spec)
    animation_intensity = _animation_intensity(spec, scene_family)
    feature_source = (
        LATEX_FREE_PREFERRED_FEATURES
        if latex_available is False
        else PREFERRED_FEATURES
    )
    preferred_features = list(feature_source.get(scene_family, feature_source["metric_story"]))
    tags = [
        scene_family,
        visual_type,
        str(spec.get("template") or "").strip().lower(),
        str(spec.get("background_motif") or "").strip().lower(),
        camera_style,
        animation_intensity,
    ]
    return SceneBrief(
        visual_id=str(spec.get("visual_id") or spec.get("id") or "visual"),
        scene_family=scene_family,
        objective=_objective(spec, scene_family),
        spoken_anchor=str(spec.get("sentence_text") or spec.get("headline") or "").strip(),
        context=str(spec.get("context_text") or spec.get("deck") or "").strip(),
        headline=str(spec.get("headline") or spec.get("emphasis_text") or "").strip(),
        deck=str(spec.get("deck") or spec.get("footer_text") or "").strip(),
        duration_sec=round(float(spec.get("duration") or max(float(spec.get("end") or 0.0) - float(spec.get("start") or 0.0), 1.0)), 2),
        start_sec=round(float(spec.get("start") or 0.0), 2),
        end_sec=round(float(spec.get("end") or 0.0), 2),
        composition_mode=str(spec.get("composition_mode") or "replace"),
        visual_type_hint=visual_type,
        style_pack=str(spec.get("style_pack") or "editorial_clean"),
        theme={str(key): str(value) for key, value in dict(spec.get("theme") or {}).items()},
        background_motif=str(spec.get("background_motif") or "constellation"),
        layout_variant=str(spec.get("layout_variant") or "hero_split"),
        camera_style=camera_style,
        animation_intensity=animation_intensity,
        must_show_terms=_collect_must_show_terms(spec),
        must_avoid=_must_avoid_terms(spec),
        preferred_manim_features=preferred_features,
        evidence=dict(spec.get("evidence") or {}),
        render_constraints={
            "width": width,
            "height": height,
            "fps": fps,
            "aspect_ratio": round(width / max(height, 1), 3),
            "target_duration_sec": round(float(spec.get("duration") or 0.0), 2),
        },
        example_tags=[tag for tag in tags if tag],
    )
