from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any


CREATIVE_DIRECTION_VERSION = "vex-creative-direction-v1"

MEDIUM_FAMILIES = (
    "data_sculpture",
    "diagrammatic_system",
    "editorial_collage",
    "kinetic_typography",
    "product_interface",
    "source_media_composite",
    "spatial_metaphor",
)

_SCENE_MEDIUMS = {
    "metric": ("data_sculpture", "editorial_collage", "diagrammatic_system"),
    "mechanism": ("spatial_metaphor", "diagrammatic_system", "data_sculpture"),
    "contrast": ("editorial_collage", "spatial_metaphor", "kinetic_typography"),
    "timeline": ("spatial_metaphor", "editorial_collage", "diagrammatic_system"),
    "interface": ("product_interface", "source_media_composite", "diagrammatic_system"),
    "emphasis": ("kinetic_typography", "editorial_collage", "spatial_metaphor"),
}

_MEDIUM_TOKENS = {
    "data_sculpture": {
        "canvas": "data_field",
        "shape": "particles_orbits_mass",
        "material": "luminous_matter",
        "typography": "numeric_monument",
        "motif": "orbital_evidence",
        "depth": "orthographic_depth",
        "motion": "measure_build_resolve",
        "panel_ratio": 0.08,
    },
    "diagrammatic_system": {
        "canvas": "technical_canvas",
        "shape": "routes_boundaries_nodes",
        "material": "wire_signal_solid",
        "typography": "technical_label",
        "motif": "guided_trace",
        "depth": "orthographic",
        "motion": "trace_handoff_resolve",
        "panel_ratio": 0.18,
    },
    "editorial_collage": {
        "canvas": "editorial_canvas",
        "shape": "asymmetric_blocks_rules",
        "material": "paper_ink_color",
        "typography": "editorial_display",
        "motif": "registration_marks",
        "depth": "layered_parallax",
        "motion": "assemble_mask_resolve",
        "panel_ratio": 0.16,
    },
    "kinetic_typography": {
        "canvas": "ink_stage",
        "shape": "type_rules_fragments",
        "material": "ink_light",
        "typography": "oversized_display",
        "motif": "type_lockup",
        "depth": "flat_macro",
        "motion": "type_lockup_resolve",
        "panel_ratio": 0.0,
    },
    "product_interface": {
        "canvas": "clean_workspace",
        "shape": "windows_controls_focus",
        "material": "product_surfaces",
        "typography": "product_system",
        "motif": "focus_cascade",
        "depth": "stacked_surfaces",
        "motion": "shell_focus_cascade",
        "panel_ratio": 0.52,
    },
    "source_media_composite": {
        "canvas": "source_media",
        "shape": "anchored_annotations",
        "material": "photographic_overlay",
        "typography": "documentary_caption",
        "motif": "tracked_focus",
        "depth": "source_locked",
        "motion": "track_reveal_resolve",
        "panel_ratio": 0.12,
    },
    "spatial_metaphor": {
        "canvas": "spatial_stage",
        "shape": "tracks_gates_chambers",
        "material": "soft_solid_light",
        "typography": "architectural_label",
        "motif": "journey_track",
        "depth": "perspective_stage",
        "motion": "travel_transform_resolve",
        "panel_ratio": 0.1,
    },
}

_DEFAULT_PALETTES = {
    "light": {
        "background": "#F5F1E8",
        "panel_fill": "#FFFDF8",
        "panel_stroke": "#171717",
        "text_primary": "#111111",
        "text_secondary": "#4B4740",
        "accent": "#E11D48",
        "accent_secondary": "#0759FF",
        "glow": "#F59E0B",
    },
    "dark": {
        "background": "#090C10",
        "panel_fill": "#17202A",
        "panel_stroke": "#5BC0EB",
        "text_primary": "#F8FAFC",
        "text_secondary": "#B6C2CF",
        "accent": "#F43F5E",
        "accent_secondary": "#14B8A6",
        "glow": "#2563EB",
    },
}


@dataclass(frozen=True)
class CreativeDirectionProgram:
    version: str
    direction_id: str
    signature: str
    visual_id: str
    scene_type: str
    scene_family: str
    medium_family: str
    orientation: str
    focal_object_ids: list[str]
    supporting_object_ids: list[str]
    relation_ids: list[str]
    composition: dict[str, Any]
    art_direction: dict[str, Any]
    choreography: dict[str, Any]
    quality_contract: dict[str, Any]
    candidate_scores: list[dict[str, Any]] = field(default_factory=list)
    rationale: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CreativeDirectionValidation:
    passed: bool
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compile_creative_direction(
    spec: dict[str, Any],
    *,
    scene_type: str,
    scene_family: str,
    objects: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    width: int,
    height: int,
    variant_index: int = 0,
    visual_world: dict[str, Any] | None = None,
) -> CreativeDirectionProgram:
    normalized = dict(spec or {})
    world = dict(visual_world or {})
    orientation = _orientation(width, height)
    scores = _rank_mediums(
        normalized,
        scene_type=scene_type,
        scene_family=scene_family,
        orientation=orientation,
        variant_index=variant_index,
        visual_world=world,
    )
    medium = str(scores[0]["medium"])
    tokens = dict(_MEDIUM_TOKENS[medium])
    palette = _palette(normalized, world, variant_index=variant_index)
    ordered_objects = sorted(
        [dict(item) for item in objects if isinstance(item, dict)],
        key=lambda item: (
            -_emphasis(item),
            0 if str(item.get("role") or "") in {"metric", "result", "focus"} else 1,
        ),
    )
    focal_count = 1 if scene_family not in {"contrast", "interface"} else min(2, len(ordered_objects))
    focal_ids = [_object_id(item) for item in ordered_objects[:focal_count] if _object_id(item)]
    support_ids = [
        _object_id(item)
        for item in ordered_objects[focal_count:]
        if _object_id(item)
    ]
    relation_ids = [
        str(item.get("relation_id") or item.get("edge_id") or "").strip()
        for item in relations
        if isinstance(item, dict)
        and str(item.get("relation_id") or item.get("edge_id") or "").strip()
    ]
    composition = _composition(
        scene_family,
        medium,
        orientation,
        object_count=len(ordered_objects),
        panel_ratio=float(tokens["panel_ratio"]),
    )
    art_direction = {
        "palette": palette,
        "canvas_system": tokens["canvas"],
        "shape_language": tokens["shape"],
        "material_system": tokens["material"],
        "typography_system": tokens["typography"],
        "motif": tokens["motif"],
        "depth_model": tokens["depth"],
        "texture_strength": 0.16 if medium in {"editorial_collage", "spatial_metaphor"} else 0.1,
        "corner_policy": "sharp" if medium in {"editorial_collage", "kinetic_typography"} else "restrained",
    }
    choreography = _choreography(
        tokens["motion"],
        object_count=len(ordered_objects),
        duration=float(normalized.get("duration") or 3.0),
    )
    quality_contract = {
        "minimum_text_contrast": 4.5,
        "minimum_graphic_contrast": 3.0,
        "negative_space_target": composition["negative_space_target"],
        "negative_space_tolerance": 0.2,
        "minimum_hierarchy_score": 0.5,
        "minimum_balance_score": 0.48,
        "minimum_depth_score": 0.28,
        "minimum_palette_vitality": 0.1,
        "maximum_edge_intrusion": (
            0.2
            if medium == "editorial_collage"
            else 0.07
            if medium == "kinetic_typography"
            else 0.14
        ),
        "maximum_global_motion": 0.42,
        "final_hold_start": 0.8,
        "forbid_color_only_encoding": True,
        "forbid_simultaneous_full_scene_motion": True,
        "required_focal_object_ids": focal_ids,
    }
    rationale = [
        f"{scene_family or scene_type} semantics favor {medium.replace('_', ' ')}",
        f"{orientation} composition uses {composition['layout_grammar'].replace('_', ' ')}",
        "stable context precedes evidence and the final state holds for reading",
    ]
    unsigned = {
        "version": CREATIVE_DIRECTION_VERSION,
        "visual_id": str(normalized.get("visual_id") or normalized.get("id") or "visual"),
        "scene_type": str(scene_type or "none"),
        "scene_family": str(scene_family or "emphasis"),
        "medium_family": medium,
        "orientation": orientation,
        "focal_object_ids": focal_ids,
        "supporting_object_ids": support_ids,
        "relation_ids": relation_ids,
        "composition": composition,
        "art_direction": art_direction,
        "choreography": choreography,
        "quality_contract": quality_contract,
        "candidate_scores": scores,
        "rationale": rationale,
    }
    signature = _signature(unsigned)
    return CreativeDirectionProgram(
        **unsigned,
        direction_id=f"direction-{signature[:16]}",
        signature=signature,
    )


def validate_creative_direction(
    program: CreativeDirectionProgram | dict[str, Any],
) -> CreativeDirectionValidation:
    payload = program.to_dict() if isinstance(program, CreativeDirectionProgram) else dict(program or {})
    errors: list[str] = []
    if payload.get("version") != CREATIVE_DIRECTION_VERSION:
        errors.append("unsupported_creative_direction_version")
    if payload.get("medium_family") not in MEDIUM_FAMILIES:
        errors.append("unsupported_creative_direction_medium")
    if payload.get("orientation") not in {"landscape", "portrait", "square"}:
        errors.append("unsupported_creative_direction_orientation")
    palette = dict((payload.get("art_direction") or {}).get("palette") or {})
    required_colors = {"background", "panel_fill", "text_primary", "accent", "accent_secondary"}
    if not required_colors.issubset(palette):
        errors.append("creative_direction_palette_incomplete")
    elif _contrast(palette["text_primary"], palette["background"]) < 4.5:
        errors.append("creative_direction_text_contrast_below_floor")
    unsigned = {
        key: value
        for key, value in payload.items()
        if key not in {"direction_id", "signature"}
    }
    if payload.get("signature") != _signature(unsigned):
        errors.append("creative_direction_signature_mismatch")
    return CreativeDirectionValidation(passed=not errors, errors=errors)


def _rank_mediums(
    spec: dict[str, Any],
    *,
    scene_type: str,
    scene_family: str,
    orientation: str,
    variant_index: int,
    visual_world: dict[str, Any],
) -> list[dict[str, Any]]:
    forced = str(visual_world.get("medium_family") or "")
    candidates = list(_SCENE_MEDIUMS.get(scene_family, _SCENE_MEDIUMS["emphasis"]))
    if forced in MEDIUM_FAMILIES:
        candidates = [forced, *[item for item in candidates if item != forced]]
    source_available = bool(
        str((spec.get("source_asset_grounding") or {}).get("asset_path") or "").strip()
    )
    history = [
        str(item.get("medium_family") or "")
        for item in [
            *(spec.get("creative_direction_history") or []),
            *(spec.get("visual_world_history") or []),
        ]
        if isinstance(item, dict)
    ]
    result: list[dict[str, Any]] = []
    for index, medium in enumerate(dict.fromkeys(candidates)):
        hard: list[str] = []
        reasons = ["semantic_medium_match"]
        score = 0.9 - index * 0.09
        if medium == forced:
            score += 0.5
            reasons.append("renderer_world_alignment")
        if medium == "source_media_composite" and not source_available:
            hard.append("source_asset_unavailable")
        if medium in history[-2:]:
            score -= 0.24
            reasons.append("recent_medium_repetition_penalty")
        if orientation == "portrait" and medium in {"editorial_collage", "product_interface"}:
            score += 0.05
            reasons.append("portrait_composition_fit")
        if scene_type == "architecture_flow" and medium == "diagrammatic_system":
            score += 0.12
            reasons.append("architecture_geometry_fit")
        result.append(
            {
                "medium": medium,
                "score": round(score, 4),
                "hard_violations": hard,
                "reasons": reasons,
            }
        )
    viable = [item for item in result if not item["hard_violations"]]
    viable.sort(key=lambda item: (-float(item["score"]), str(item["medium"])))
    if viable and not forced:
        offset = max(0, variant_index) % len(viable)
        viable = viable[offset:] + viable[:offset]
    return viable or result


def _composition(
    family: str,
    medium: str,
    orientation: str,
    *,
    object_count: int,
    panel_ratio: float,
) -> dict[str, Any]:
    grammars = {
        "metric": "asymmetric_monument",
        "mechanism": "guided_route",
        "contrast": "editorial_diptych",
        "timeline": "chapter_journey",
        "interface": "focus_workspace",
        "emphasis": "typographic_lockup",
    }
    focal_points = {
        "landscape": [0.68 if family == "metric" else 0.5, 0.54],
        "portrait": [0.5, 0.58],
        "square": [0.5, 0.53],
    }
    density = "minimal" if object_count <= 2 else "balanced" if object_count <= 4 else "dense"
    negative_space = (
        0.82
        if medium == "kinetic_typography"
        else 0.56
        if medium == "product_interface"
        else 0.7
        if medium == "data_sculpture"
        else 0.74
        if density == "minimal"
        else 0.68
    )
    return {
        "layout_grammar": grammars.get(family, "asymmetric_focus"),
        "focal_point": focal_points[orientation],
        "reading_path": "top_left_to_focal_to_resolution",
        "density": density,
        "negative_space_target": negative_space,
        "panel_ratio_max": max(panel_ratio, 0.04),
        "spatial_layers": 3 if medium in {"spatial_metaphor", "editorial_collage", "product_interface"} else 2,
        "max_simultaneous_elements": 3,
        "safe_margin_ratio": 0.055 if orientation == "landscape" else 0.065,
    }


def _choreography(motion: str, *, object_count: int, duration: float) -> dict[str, Any]:
    stagger = min(0.07, 0.22 / max(object_count, 1))
    macro_span = min(0.18, 0.42 / max(duration, 1.0))
    return {
        "motion_language": motion,
        "easing": "purposeful_spring_no_overshoot",
        "stagger_fraction": round(stagger, 4),
        "macro_span_fraction": round(macro_span, 4),
        "max_simultaneous_motion": 3,
        "phases": [
            {"phase": "establish", "start": 0.0, "end": 0.16, "job": "stable_context"},
            {"phase": "reveal", "start": 0.1, "end": 0.48, "job": "ordered_evidence"},
            {"phase": "relate", "start": 0.32, "end": 0.68, "job": "explain_connection"},
            {"phase": "resolve", "start": 0.62, "end": 0.8, "job": "focus_outcome"},
            {"phase": "hold", "start": 0.8, "end": 1.0, "job": "readable_final_state"},
        ],
        "reduced_motion_policy": "preserve_semantic_state_without_translation",
    }


def _palette(spec: dict[str, Any], world: dict[str, Any], *, variant_index: int) -> dict[str, str]:
    bible = dict(spec.get("video_design_bible") or {})
    sequence = [dict(item) for item in bible.get("palette_sequence") or [] if isinstance(item, dict)]
    source = dict(world.get("palette") or {})
    if not source and sequence:
        ordinal = _integer(spec.get("visual_world_ordinal"), 0)
        source = sequence[(ordinal + variant_index) % len(sequence)]
    if not source:
        source = dict(spec.get("theme") or {})
    base = dict(_DEFAULT_PALETTES["light" if _luminance(source.get("background", "#090C10")) > 0.45 else "dark"])
    base.update({str(key): str(value) for key, value in source.items() if _is_hex(value)})
    background = base["background"]
    if _contrast(base["text_primary"], background) < 4.5:
        base["text_primary"] = "#111111" if _luminance(background) > 0.45 else "#FFFFFF"
    if _contrast(base["text_secondary"], background) < 3.0:
        base["text_secondary"] = base["text_primary"]
    if _contrast(base["accent"], background) < 3.0:
        base["accent"] = "#B91C1C" if _luminance(background) > 0.45 else "#FBBF24"
    base["ink"] = "#111111" if _luminance(base["panel_fill"]) > 0.45 else "#FFFFFF"
    return base


def _object_id(item: dict[str, Any]) -> str:
    return str(item.get("object_id") or item.get("node_id") or item.get("element_id") or "").strip()


def _emphasis(item: dict[str, Any]) -> float:
    try:
        return max(0.0, min(float(item.get("emphasis") or 0.5), 1.0))
    except (TypeError, ValueError):
        return 0.5


def _orientation(width: int, height: int) -> str:
    aspect = max(int(width), 1) / max(int(height), 1)
    return "portrait" if aspect < 0.82 else "landscape" if aspect > 1.22 else "square"


def _signature(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _is_hex(value: Any) -> bool:
    return bool(re.fullmatch(r"#[0-9A-Fa-f]{6}", str(value or "")))


def _rgb(value: Any) -> tuple[float, float, float]:
    text = str(value or "#000000")
    if not _is_hex(text):
        text = "#000000"
    channels = [int(text[index : index + 2], 16) / 255.0 for index in (1, 3, 5)]
    return tuple(channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4 for channel in channels)


def _luminance(value: Any) -> float:
    red, green, blue = _rgb(value)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _contrast(left: Any, right: Any) -> float:
    light, dark = sorted((_luminance(left), _luminance(right)), reverse=True)
    return (light + 0.05) / (dark + 0.05)


def _integer(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


__all__ = [
    "CREATIVE_DIRECTION_VERSION",
    "CreativeDirectionProgram",
    "CreativeDirectionValidation",
    "compile_creative_direction",
    "validate_creative_direction",
]
