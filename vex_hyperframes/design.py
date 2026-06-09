from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any


PREMIUM_ARCHETYPES = {
    "semantic_architecture": "semantic_architecture",
    "semantic_causal": "semantic_causal",
    "semantic_decision": "semantic_decision",
    "semantic_interface": "semantic_interface",
    "semantic_metric": "semantic_metric",
    "semantic_narrative": "semantic_narrative",
    "semantic_quote": "semantic_quote",
    "semantic_route": "semantic_route",
    "semantic_transform": "semantic_transform",
    "data_journey": "metric_proof",
    "metric_callout": "metric_proof",
    "stat_grid": "metric_proof",
    "signal_network": "signal_map",
    "system_flow": "signal_map",
    "kinetic_route": "route_journey",
    "timeline_steps": "route_journey",
    "spotlight_compare": "contrast_shift",
    "comparison_split": "contrast_shift",
    "interface_cascade": "interface_cascade",
    "ribbon_quote": "kinetic_quote",
    "keyword_stack": "kinetic_quote",
    "quote_focus": "kinetic_quote",
    "causal_chain": "causal_chain",
    "flywheel_loop": "flywheel_loop",
    "decision_matrix": "decision_matrix",
    "anatomy_cutaway": "anatomy_cutaway",
    "stack_ranking": "stack_ranking",
    "contrast_ladder": "contrast_shift",
    "proof_sequence": "metric_proof",
    "narrative_arc": "story_arc",
    "concept_map": "signal_map",
    "problem_solution": "contrast_shift",
    "myth_buster": "contrast_shift",
    "checklist_reveal": "stack_ranking",
    "risk_radar": "metric_proof",
    "opportunity_map": "signal_map",
    "scorecard": "decision_matrix",
    "pipeline_xray": "anatomy_cutaway",
    "decision_tree": "route_journey",
    "momentum_wave": "metric_proof",
    "focus_ring": "kinetic_quote",
    "timeline_filmstrip": "story_arc",
    "quote_breakdown": "anatomy_cutaway",
    "market_map": "signal_map",
    "mechanism_blueprint": "anatomy_cutaway",
    "data_pulse": "metric_proof",
}


@dataclass(frozen=True)
class ArtDirection:
    direction_id: str
    label: str
    theme: dict[str, str]
    motif: str
    typography: str
    depth: str
    motion_profile: str
    texture: str
    contrast_target: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DesignIR:
    design_id: str
    template: str
    archetype: str
    visual_type_hint: str
    claim: str
    support: list[str]
    keywords: list[str]
    duration_sec: float
    width: int
    height: int
    fps: float
    composition_mode: str
    content_density: str
    motion_intensity: str
    safe_margin_px: int
    subtitle_safe_px: int
    art_direction: ArtDirection
    visual_beats: list[dict[str, Any]] = field(default_factory=list)
    program_context: dict[str, Any] = field(default_factory=dict)
    episode_context: dict[str, Any] = field(default_factory=dict)
    continuity_group: str = ""
    concept_ids: list[str] = field(default_factory=list)
    transition_in: dict[str, Any] = field(default_factory=dict)
    transition_out: dict[str, Any] = field(default_factory=dict)
    variant_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["art_direction"] = self.art_direction.to_dict()
        return payload


def _clean(value: Any, *, max_chars: int = 120) -> str:
    cleaned = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip(" ,.;:-")
    return cleaned


def _word_count(*values: Any) -> int:
    return len(re.findall(r"[A-Za-z0-9%+.-]+", " ".join(str(value or "") for value in values)))


def _string_list(value: Any, *, limit: int, max_chars: int) -> list[str]:
    raw_items = value if isinstance(value, list) else []
    result: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        cleaned = _clean(item, max_chars=max_chars)
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
        if len(result) >= limit:
            break
    return result


def _dict_list(value: Any, *, limit: int = 6) -> list[dict[str, Any]]:
    raw_items = value if isinstance(value, list) else []
    result: list[dict[str, Any]] = []
    for item in raw_items:
        if isinstance(item, dict):
            result.append(dict(item))
        if len(result) >= limit:
            break
    return result


def _theme_defaults(spec: dict[str, Any]) -> dict[str, str]:
    theme = dict(spec.get("theme") or {})
    defaults = {
        "background": "#08111F",
        "panel_fill": "#101E33",
        "panel_stroke": "#5BC0EB",
        "accent": "#F59E0B",
        "accent_secondary": "#38BDF8",
        "glow": "#1D4ED8",
        "eyebrow_fill": "#14324D",
        "eyebrow_text": "#E0F2FE",
        "grid": "#244760",
        "text_primary": "#F8FAFC",
        "text_secondary": "#D6E3F3",
    }
    defaults.update({key: str(value) for key, value in theme.items() if value})
    return defaults


def _merge_theme(base: dict[str, str], overlay: dict[str, str]) -> dict[str, str]:
    merged = dict(base)
    merged.update({key: value for key, value in overlay.items() if value})
    return merged


_ART_DIRECTIONS: dict[str, dict[str, Any]] = {
    "premium_explainer": {
        "label": "Premium Explainer",
        "motif": "precision_grid",
        "typography": "bold_editorial",
        "depth": "layered_glass",
        "motion_profile": "crisp_reveal",
        "texture": "fine_grid",
        "contrast_target": 0.82,
        "theme": {
            "background": "#06131E",
            "panel_fill": "#10243A",
            "panel_stroke": "#5BC0EB",
            "accent": "#F59E0B",
            "accent_secondary": "#2DD4BF",
            "glow": "#0EA5E9",
            "grid": "#1E5267",
            "text_primary": "#F8FAFC",
            "text_secondary": "#CDE7F3",
        },
    },
    "cinematic_editorial": {
        "label": "Cinematic Editorial",
        "motif": "light_sweep",
        "typography": "magazine_display",
        "depth": "cinematic_planes",
        "motion_profile": "slow_focus",
        "texture": "film_grain",
        "contrast_target": 0.86,
        "theme": {
            "background": "#100B11",
            "panel_fill": "#231520",
            "panel_stroke": "#FB7185",
            "accent": "#FBBF24",
            "accent_secondary": "#F472B6",
            "glow": "#BE185D",
            "grid": "#432032",
            "text_primary": "#FFF7ED",
            "text_secondary": "#FED7AA",
        },
    },
    "product_ui": {
        "label": "Product UI",
        "motif": "interface_depth",
        "typography": "product_system",
        "depth": "stacked_surfaces",
        "motion_profile": "cascade_focus",
        "texture": "subtle_mesh",
        "contrast_target": 0.8,
        "theme": {
            "background": "#07111F",
            "panel_fill": "#10223E",
            "panel_stroke": "#818CF8",
            "accent": "#22C55E",
            "accent_secondary": "#60A5FA",
            "glow": "#4F46E5",
            "grid": "#22365E",
            "text_primary": "#F8FAFC",
            "text_secondary": "#C7D2FE",
        },
    },
    "data_proof": {
        "label": "Data Proof",
        "motif": "dashboard_signal",
        "typography": "numeric_display",
        "depth": "illuminated_panels",
        "motion_profile": "measured_build",
        "texture": "analytic_grid",
        "contrast_target": 0.84,
        "theme": {
            "background": "#07131A",
            "panel_fill": "#0C2730",
            "panel_stroke": "#38BDF8",
            "accent": "#FACC15",
            "accent_secondary": "#22D3EE",
            "glow": "#0891B2",
            "grid": "#17485A",
            "text_primary": "#F0FDFA",
            "text_secondary": "#BAE6FD",
        },
    },
    "system_flow": {
        "label": "System Flow",
        "motif": "signal_routes",
        "typography": "technical_label",
        "depth": "network_layers",
        "motion_profile": "guided_trace",
        "texture": "signal_noise",
        "contrast_target": 0.81,
        "theme": {
            "background": "#071714",
            "panel_fill": "#0F2722",
            "panel_stroke": "#34D399",
            "accent": "#F97316",
            "accent_secondary": "#2DD4BF",
            "glow": "#0F766E",
            "grid": "#1C4A41",
            "text_primary": "#F0FDFA",
            "text_secondary": "#A7F3D0",
        },
    },
}


def _choose_direction_id(spec: dict[str, Any], template: str, visual_type_hint: str, variant_index: int) -> str:
    explicit = str(spec.get("art_direction") or spec.get("hyperframes_art_direction") or "").strip().lower()
    if explicit in _ART_DIRECTIONS:
        return explicit
    archetype = PREMIUM_ARCHETYPES.get(template, "kinetic_quote")
    if visual_type_hint == "product_ui" or template in {"interface_cascade", "semantic_interface"}:
        base = "product_ui"
    elif archetype in {"metric_proof", "semantic_metric"}:
        base = "data_proof"
    elif archetype in {
        "signal_map",
        "route_journey",
        "causal_chain",
        "flywheel_loop",
        "story_arc",
        "semantic_architecture",
        "semantic_causal",
        "semantic_narrative",
        "semantic_route",
    }:
        base = "system_flow"
    elif archetype in {
        "decision_matrix",
        "anatomy_cutaway",
        "stack_ranking",
        "semantic_decision",
        "semantic_transform",
    }:
        base = "premium_explainer"
    elif visual_type_hint == "abstract_motion" or archetype in {"kinetic_quote", "semantic_quote"}:
        base = "cinematic_editorial"
    else:
        base = "premium_explainer"
    alternates = {
        "premium_explainer": ("premium_explainer", "system_flow", "cinematic_editorial"),
        "cinematic_editorial": ("cinematic_editorial", "premium_explainer", "system_flow"),
        "product_ui": ("product_ui", "premium_explainer", "data_proof"),
        "data_proof": ("data_proof", "premium_explainer", "product_ui"),
        "system_flow": ("system_flow", "premium_explainer", "data_proof"),
    }
    choices = alternates.get(base, (base,))
    return choices[max(variant_index, 0) % len(choices)]


def build_art_direction(spec: dict[str, Any], *, template: str, visual_type_hint: str, variant_index: int = 0) -> ArtDirection:
    direction_id = _choose_direction_id(spec, template, visual_type_hint, variant_index)
    definition = _ART_DIRECTIONS[direction_id]
    return ArtDirection(
        direction_id=direction_id,
        label=str(definition["label"]),
        theme=_merge_theme(_theme_defaults(spec), dict(definition["theme"])),
        motif=str(definition["motif"]),
        typography=str(definition["typography"]),
        depth=str(definition["depth"]),
        motion_profile=str(definition["motion_profile"]),
        texture=str(definition["texture"]),
        contrast_target=float(definition["contrast_target"]),
    )


def build_design_ir(
    spec: dict[str, Any],
    *,
    width: int,
    height: int,
    fps: float,
    variant_index: int = 0,
) -> DesignIR:
    template = str(spec.get("template") or "ribbon_quote").strip().lower()
    archetype = PREMIUM_ARCHETYPES.get(template, "kinetic_quote")
    visual_type_hint = str(spec.get("visual_type_hint") or "general").strip().lower()
    support = _string_list(spec.get("supporting_lines"), limit=4, max_chars=54)
    keywords = _string_list(spec.get("keywords"), limit=6, max_chars=28)
    claim = _clean(spec.get("headline") or spec.get("quote_text") or spec.get("sentence_text") or "Key idea", max_chars=72)
    density_words = _word_count(claim, spec.get("deck"), spec.get("footer_text"), " ".join(support), " ".join(keywords))
    if density_words <= 13:
        density = "minimal"
    elif density_words <= 26:
        density = "balanced"
    else:
        density = "dense"
    importance = float(spec.get("importance") or 0.5)
    duration = max(float(spec.get("duration") or 2.8), 1.0)
    if importance >= 0.78 and duration >= 2.4:
        motion_intensity = "high"
    elif importance >= 0.52 or duration >= 2.0:
        motion_intensity = "medium"
    else:
        motion_intensity = "low"
    shortest_edge = max(min(width, height), 1)
    safe_margin = max(48, int(round(shortest_edge * 0.07)))
    subtitle_safe = max(56, int(round(height * 0.1)))
    art_direction = build_art_direction(
        spec,
        template=template,
        visual_type_hint=visual_type_hint,
        variant_index=variant_index,
    )
    return DesignIR(
        design_id=f"{template}:{art_direction.direction_id}:v{variant_index}",
        template=template,
        archetype=archetype,
        visual_type_hint=visual_type_hint,
        claim=claim,
        support=support,
        keywords=keywords,
        duration_sec=duration,
        width=width,
        height=height,
        fps=fps,
        composition_mode=str(spec.get("composition_mode") or "replace").strip().lower(),
        content_density=density,
        motion_intensity=motion_intensity,
        safe_margin_px=safe_margin,
        subtitle_safe_px=subtitle_safe,
        art_direction=art_direction,
        visual_beats=_dict_list(spec.get("visual_beats"), limit=6),
        program_context=dict(spec.get("program_context") or {}),
        episode_context=dict(spec.get("episode_context") or {}),
        continuity_group=_clean(spec.get("continuity_group"), max_chars=80),
        concept_ids=_string_list(spec.get("concept_ids"), limit=6, max_chars=40),
        transition_in=dict(spec.get("transition_in") or {}),
        transition_out=dict(spec.get("transition_out") or {}),
        variant_index=variant_index,
    )


def root_class_names(ir: DesignIR) -> str:
    return " ".join(
        [
            f"ad-{ir.art_direction.direction_id}",
            f"motif-{ir.art_direction.motif}",
            f"density-{ir.content_density}",
            f"motion-{ir.motion_intensity}",
            f"archetype-{ir.archetype}",
        ]
    )
