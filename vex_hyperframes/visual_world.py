from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


VISUAL_WORLD_VERSION = "hyperframes-visual-world-v1"
VIDEO_DESIGN_BIBLE_VERSION = "hyperframes-video-design-bible-v1"

MEDIUM_FAMILIES = frozenset(
    {
        "data_sculpture",
        "diagrammatic_system",
        "editorial_collage",
        "kinetic_typography",
        "product_interface",
        "source_media_composite",
        "spatial_metaphor",
    }
)
CANVAS_SYSTEMS = frozenset(
    {
        "chromatic_field",
        "clean_workspace",
        "data_field",
        "ink_stage",
        "paper_canvas",
        "source_media",
        "spatial_stage",
        "technical_canvas",
    }
)
CARD_POLICIES = frozenset({"allowed", "forbidden", "source_only"})


@dataclass(frozen=True)
class VisualFingerprint:
    medium_family: str
    canvas_system: str
    shape_language: str
    material_system: str
    typography_system: str
    motion_choreography: str
    camera_depth: str
    background_mode: str
    panel_ratio_target: float
    signature: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["panel_ratio_target"] = round(float(self.panel_ratio_target), 3)
        return payload


@dataclass(frozen=True)
class VideoDesignBible:
    version: str
    design_id: str
    palette_sequence: list[dict[str, str]]
    typography_anchor: str
    continuity_motif: str
    repetition_window: int
    max_card_ratio: float
    forbidden_repetition: list[str]
    signature: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["max_card_ratio"] = round(float(self.max_card_ratio), 3)
        return payload


@dataclass(frozen=True)
class VisualWorldProgram:
    version: str
    world_id: str
    visual_id: str
    proof_program_id: str
    scene_type: str
    medium_family: str
    canvas_system: str
    shape_language: str
    material_system: str
    typography_system: str
    camera_depth: str
    motion_choreography: str
    background_mode: str
    header_mode: str
    card_policy: str
    palette: dict[str, str]
    semantic_bindings: dict[str, Any]
    fingerprint: VisualFingerprint
    rationale: str
    world_signature: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["fingerprint"] = self.fingerprint.to_dict()
        return payload


@dataclass(frozen=True)
class VisualWorldValidation:
    passed: bool
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_PALETTE_SEQUENCES: tuple[tuple[dict[str, str], ...], ...] = (
    (
        {
            "background": "#F4F0E8",
            "panel_fill": "#FFFDF8",
            "panel_stroke": "#151515",
            "accent": "#F04438",
            "accent_secondary": "#1E5EFF",
            "glow": "#FFB800",
            "grid": "#C8C0B4",
            "text_primary": "#111111",
            "text_secondary": "#4B4740",
            "eyebrow_fill": "#111111",
            "eyebrow_text": "#FFFDF8",
        },
        {
            "background": "#090A0C",
            "panel_fill": "#15171C",
            "panel_stroke": "#D8FF3E",
            "accent": "#D8FF3E",
            "accent_secondary": "#58E6FF",
            "glow": "#8B5CF6",
            "grid": "#343A40",
            "text_primary": "#F8F8F2",
            "text_secondary": "#B9BCC4",
            "eyebrow_fill": "#D8FF3E",
            "eyebrow_text": "#090A0C",
        },
        {
            "background": "#EAF5FF",
            "panel_fill": "#FFFFFF",
            "panel_stroke": "#073B4C",
            "accent": "#FF5A5F",
            "accent_secondary": "#0077FF",
            "glow": "#FFD166",
            "grid": "#A9C8DC",
            "text_primary": "#072A3D",
            "text_secondary": "#35586A",
            "eyebrow_fill": "#073B4C",
            "eyebrow_text": "#FFFFFF",
        },
    ),
    (
        {
            "background": "#FFF8E7",
            "panel_fill": "#FFFFFF",
            "panel_stroke": "#161616",
            "accent": "#FF3D81",
            "accent_secondary": "#0759FF",
            "glow": "#F9C80E",
            "grid": "#D9D0BC",
            "text_primary": "#141414",
            "text_secondary": "#514B41",
            "eyebrow_fill": "#0759FF",
            "eyebrow_text": "#FFFFFF",
        },
        {
            "background": "#081C15",
            "panel_fill": "#102A20",
            "panel_stroke": "#D8F3DC",
            "accent": "#FFB703",
            "accent_secondary": "#80ED99",
            "glow": "#52B788",
            "grid": "#2D6A4F",
            "text_primary": "#F5FFF7",
            "text_secondary": "#B7DCC0",
            "eyebrow_fill": "#FFB703",
            "eyebrow_text": "#081C15",
        },
        {
            "background": "#F2ECFF",
            "panel_fill": "#FFFFFF",
            "panel_stroke": "#25213C",
            "accent": "#6D28D9",
            "accent_secondary": "#F43F5E",
            "glow": "#22D3EE",
            "grid": "#C7BCE3",
            "text_primary": "#211D35",
            "text_secondary": "#5D5672",
            "eyebrow_fill": "#25213C",
            "eyebrow_text": "#FFFFFF",
        },
    ),
)

_MEDIUM_PROFILES: dict[str, dict[str, Any]] = {
    "kinetic_typography": {
        "canvas_system": "ink_stage",
        "shape_language": "typographic_field",
        "material_system": "ink_and_light",
        "typography_system": "oversized_condensed",
        "camera_depth": "flat_macro",
        "motion_choreography": "type_lockup",
        "background_mode": "editorial_field",
        "header_mode": "integrated",
        "card_policy": "forbidden",
        "panel_ratio_target": 0.0,
    },
    "editorial_collage": {
        "canvas_system": "paper_canvas",
        "shape_language": "cut_paper",
        "material_system": "paper_ink_photo",
        "typography_system": "magazine_display",
        "camera_depth": "layered_parallax",
        "motion_choreography": "assemble_and_mask",
        "background_mode": "paper_registration",
        "header_mode": "integrated",
        "card_policy": "forbidden",
        "panel_ratio_target": 0.04,
    },
    "data_sculpture": {
        "canvas_system": "data_field",
        "shape_language": "particles_and_mass",
        "material_system": "luminous_matter",
        "typography_system": "numeric_display",
        "camera_depth": "orthographic_depth",
        "motion_choreography": "swarm_and_compress",
        "background_mode": "radial_data_field",
        "header_mode": "minimal",
        "card_policy": "forbidden",
        "panel_ratio_target": 0.0,
    },
    "spatial_metaphor": {
        "canvas_system": "spatial_stage",
        "shape_language": "gates_tracks_chambers",
        "material_system": "soft_solid",
        "typography_system": "architectural_label",
        "camera_depth": "perspective_stage",
        "motion_choreography": "travel_transform_resolve",
        "background_mode": "spatial_horizon",
        "header_mode": "minimal",
        "card_policy": "forbidden",
        "panel_ratio_target": 0.02,
    },
    "diagrammatic_system": {
        "canvas_system": "technical_canvas",
        "shape_language": "routes_and_boundaries",
        "material_system": "wire_and_signal",
        "typography_system": "technical_label",
        "camera_depth": "orthographic",
        "motion_choreography": "trace_and_handoff",
        "background_mode": "technical_field",
        "header_mode": "minimal",
        "card_policy": "source_only",
        "panel_ratio_target": 0.14,
    },
    "product_interface": {
        "canvas_system": "clean_workspace",
        "shape_language": "windows_and_controls",
        "material_system": "product_surfaces",
        "typography_system": "product_system",
        "camera_depth": "stacked_surfaces",
        "motion_choreography": "focus_and_cascade",
        "background_mode": "workspace",
        "header_mode": "minimal",
        "card_policy": "allowed",
        "panel_ratio_target": 0.46,
    },
    "source_media_composite": {
        "canvas_system": "source_media",
        "shape_language": "anchored_annotations",
        "material_system": "photographic",
        "typography_system": "documentary_caption",
        "camera_depth": "source_locked",
        "motion_choreography": "track_and_reveal",
        "background_mode": "source_full_bleed",
        "header_mode": "minimal",
        "card_policy": "source_only",
        "panel_ratio_target": 0.08,
    },
}

_SCENE_MEDIUMS: dict[str, tuple[str, ...]] = {
    "architecture_flow": (
        "diagrammatic_system",
        "data_sculpture",
        "spatial_metaphor",
        "kinetic_typography",
    ),
    "causal_intervention": (
        "spatial_metaphor",
        "data_sculpture",
        "kinetic_typography",
        "editorial_collage",
    ),
    "decision_branch": (
        "spatial_metaphor",
        "kinetic_typography",
        "editorial_collage",
        "diagrammatic_system",
    ),
    "evidence_backed_quote": (
        "kinetic_typography",
        "editorial_collage",
        "spatial_metaphor",
        "data_sculpture",
    ),
    "grounded_interface_walkthrough": (
        "source_media_composite",
        "product_interface",
        "editorial_collage",
        "diagrammatic_system",
    ),
    "guided_process": (
        "spatial_metaphor",
        "diagrammatic_system",
        "kinetic_typography",
        "editorial_collage",
    ),
    "matched_state_transform": (
        "editorial_collage",
        "data_sculpture",
        "kinetic_typography",
        "spatial_metaphor",
    ),
    "metric_delta": (
        "data_sculpture",
        "kinetic_typography",
        "editorial_collage",
        "spatial_metaphor",
    ),
    "metric_intervention": (
        "data_sculpture",
        "spatial_metaphor",
        "kinetic_typography",
        "editorial_collage",
    ),
    "metric_proof": (
        "data_sculpture",
        "editorial_collage",
        "kinetic_typography",
        "diagrammatic_system",
    ),
    "narrative_progression": (
        "editorial_collage",
        "spatial_metaphor",
        "kinetic_typography",
        "data_sculpture",
    ),
    "set_partition": (
        "data_sculpture",
        "spatial_metaphor",
        "kinetic_typography",
        "editorial_collage",
    ),
}


def build_video_design_bible(specs: list[dict[str, Any]]) -> VideoDesignBible:
    stable_payload = [
        {
            "visual_id": str(item.get("visual_id") or ""),
            "episode_id": str(item.get("episode_id") or ""),
            "continuity_group": str(item.get("continuity_group") or ""),
            "concept_ids": sorted(str(value) for value in item.get("concept_ids") or []),
        }
        for item in specs
    ]
    seed = int(_signature({"specs": stable_payload})[:8], 16)
    palette_sequence = [
        dict(item)
        for item in _PALETTE_SEQUENCES[seed % len(_PALETTE_SEQUENCES)]
    ]
    base_payload = {
        "version": VIDEO_DESIGN_BIBLE_VERSION,
        "design_id": f"video-world-{_signature(stable_payload)[:12]}",
        "palette_sequence": palette_sequence,
        "typography_anchor": "display_plus_system",
        "continuity_motif": (
            "precision_marks" if seed % 2 == 0 else "chapter_registration"
        ),
        "repetition_window": 3,
        "max_card_ratio": 0.18,
        "forbidden_repetition": [
            "same_medium_in_previous_two_visuals",
            "same_background_in_previous_three_visuals",
            "card_surfaces_for_non_interface_scenes",
            "cosmetic_only_candidate_variation",
        ],
    }
    return VideoDesignBible(
        **base_payload,
        signature=_signature(base_payload),
    )


def build_visual_world_program(
    ir: Any,
    scene_program: Any,
    *,
    proof_program_id: str,
    proof_encoding: str,
    variant_index: int,
    spec: dict[str, Any],
) -> VisualWorldProgram:
    ir_payload = _payload(ir)
    scene_payload = _payload(scene_program)
    scene_type = str(ir_payload.get("scene_type") or "")
    design_bible = _design_bible(spec)
    history = [
        dict(item)
        for item in spec.get("visual_world_history") or []
        if isinstance(item, dict)
    ]
    source_available = bool(
        str((spec.get("source_asset_grounding") or {}).get("asset_path") or "").strip()
    )
    medium = _choose_medium(
        scene_type,
        proof_encoding=proof_encoding,
        variant_index=variant_index,
        history=history,
        source_available=source_available,
    )
    profile = dict(_MEDIUM_PROFILES[medium])
    visual_ordinal = _integer(spec.get("visual_world_ordinal"), 0)
    palettes = list(design_bible.palette_sequence)
    palette = dict(palettes[(visual_ordinal + variant_index) % max(len(palettes), 1)])
    if medium == "kinetic_typography" and variant_index % 2:
        profile["canvas_system"] = "chromatic_field"
        profile["background_mode"] = "chromatic_blocks"
    semantic_bindings = _semantic_bindings(scene_payload)
    fingerprint_base = {
        "medium_family": medium,
        "canvas_system": profile["canvas_system"],
        "shape_language": profile["shape_language"],
        "material_system": profile["material_system"],
        "typography_system": profile["typography_system"],
        "motion_choreography": profile["motion_choreography"],
        "camera_depth": profile["camera_depth"],
        "background_mode": profile["background_mode"],
        "panel_ratio_target": profile["panel_ratio_target"],
    }
    fingerprint = VisualFingerprint(
        **fingerprint_base,
        signature=_signature(fingerprint_base),
    )
    base_payload = {
        "version": VISUAL_WORLD_VERSION,
        "world_id": f"{_safe_id(proof_program_id)}-{medium}",
        "visual_id": str(ir_payload.get("visual_id") or ""),
        "proof_program_id": str(proof_program_id or ""),
        "scene_type": scene_type,
        "medium_family": medium,
        "canvas_system": profile["canvas_system"],
        "shape_language": profile["shape_language"],
        "material_system": profile["material_system"],
        "typography_system": profile["typography_system"],
        "camera_depth": profile["camera_depth"],
        "motion_choreography": profile["motion_choreography"],
        "background_mode": profile["background_mode"],
        "header_mode": profile["header_mode"],
        "card_policy": profile["card_policy"],
        "palette": palette,
        "semantic_bindings": semantic_bindings,
        "fingerprint": fingerprint.to_dict(),
        "rationale": _rationale(scene_type, medium, proof_encoding),
    }
    return VisualWorldProgram(
        version=VISUAL_WORLD_VERSION,
        world_id=str(base_payload["world_id"]),
        visual_id=str(base_payload["visual_id"]),
        proof_program_id=str(base_payload["proof_program_id"]),
        scene_type=scene_type,
        medium_family=medium,
        canvas_system=str(base_payload["canvas_system"]),
        shape_language=str(base_payload["shape_language"]),
        material_system=str(base_payload["material_system"]),
        typography_system=str(base_payload["typography_system"]),
        camera_depth=str(base_payload["camera_depth"]),
        motion_choreography=str(base_payload["motion_choreography"]),
        background_mode=str(base_payload["background_mode"]),
        header_mode=str(base_payload["header_mode"]),
        card_policy=str(base_payload["card_policy"]),
        palette=palette,
        semantic_bindings=semantic_bindings,
        fingerprint=fingerprint,
        rationale=str(base_payload["rationale"]),
        world_signature=_signature(base_payload),
    )


def validate_visual_world_program(
    program: VisualWorldProgram | dict[str, Any],
    *,
    scene_program: Any,
) -> VisualWorldValidation:
    payload = program.to_dict() if isinstance(program, VisualWorldProgram) else dict(program or {})
    scene_payload = _payload(scene_program)
    errors: list[str] = []
    if payload.get("version") != VISUAL_WORLD_VERSION:
        errors.append("unsupported_visual_world_version")
    if payload.get("medium_family") not in MEDIUM_FAMILIES:
        errors.append("unsupported_visual_world_medium")
    if payload.get("canvas_system") not in CANVAS_SYSTEMS:
        errors.append("unsupported_visual_world_canvas")
    if payload.get("card_policy") not in CARD_POLICIES:
        errors.append("unsupported_visual_world_card_policy")
    if not payload.get("palette") or not isinstance(payload.get("palette"), dict):
        errors.append("visual_world_palette_missing")
    bindings = dict(payload.get("semantic_bindings") or {})
    expected_objects = {
        str(item.get("object_id") or "")
        for item in scene_payload.get("elements") or []
        if isinstance(item, dict) and str(item.get("object_id") or "")
    }
    bound_objects = set((bindings.get("objects") or {}).keys())
    if expected_objects != bound_objects:
        errors.append("visual_world_object_bindings_incomplete")
    expected_relations = {
        str(item.get("relation_id") or "")
        for item in scene_payload.get("relations") or []
        if isinstance(item, dict) and str(item.get("relation_id") or "")
    }
    bound_relations = set((bindings.get("relations") or {}).keys())
    if expected_relations != bound_relations:
        errors.append("visual_world_relation_bindings_incomplete")
    fingerprint = dict(payload.get("fingerprint") or {})
    fingerprint_base = {
        key: fingerprint.get(key)
        for key in (
            "medium_family",
            "canvas_system",
            "shape_language",
            "material_system",
            "typography_system",
            "motion_choreography",
            "camera_depth",
            "background_mode",
            "panel_ratio_target",
        )
    }
    if fingerprint.get("signature") != _signature(fingerprint_base):
        errors.append("visual_world_fingerprint_signature_mismatch")
    world_base = {key: value for key, value in payload.items() if key != "world_signature"}
    if payload.get("world_signature") != _signature(world_base):
        errors.append("visual_world_signature_mismatch")
    if (
        payload.get("card_policy") == "forbidden"
        and float(fingerprint.get("panel_ratio_target") or 0.0) > 0.08
    ):
        errors.append("visual_world_forbidden_card_ratio_exceeded")
    return VisualWorldValidation(passed=not errors, errors=_unique(errors))


def _choose_medium(
    scene_type: str,
    *,
    proof_encoding: str,
    variant_index: int,
    history: list[dict[str, Any]],
    source_available: bool,
) -> str:
    candidates = list(
        _SCENE_MEDIUMS.get(
            scene_type,
            (
                "editorial_collage",
                "kinetic_typography",
                "spatial_metaphor",
                "data_sculpture",
            ),
        )
    )
    if not source_available:
        candidates = [item for item in candidates if item != "source_media_composite"]
    encoding_offset = {
        "focal_gate": 0,
        "layered_flow": 1,
        "linear_trace": 2,
        "radial_evidence": 3,
        "split_register": 1,
    }.get(str(proof_encoding or ""), 0)
    offset = (max(variant_index, 0) + encoding_offset) % max(len(candidates), 1)
    ordered = candidates[offset:] + candidates[:offset]
    recent_mediums = [
        str(item.get("medium_family") or "")
        for item in history[-2:]
    ]
    recent_backgrounds = {
        str(item.get("background_mode") or "")
        for item in history[-3:]
    }
    for medium in ordered:
        profile = _MEDIUM_PROFILES[medium]
        if medium in recent_mediums:
            continue
        if str(profile["background_mode"]) in recent_backgrounds:
            continue
        return medium
    for medium in ordered:
        if medium not in recent_mediums:
            return medium
    return ordered[0]


def _semantic_bindings(scene_program: dict[str, Any]) -> dict[str, Any]:
    objects = {
        str(item.get("object_id") or ""): {
            "element_id": str(item.get("element_id") or ""),
            "role": str(item.get("role") or ""),
            "evidence_ids": list(item.get("evidence_ids") or []),
            "fact_ids": list(item.get("fact_ids") or []),
        }
        for item in scene_program.get("elements") or []
        if isinstance(item, dict) and str(item.get("object_id") or "")
    }
    relations = {
        str(item.get("relation_id") or ""): {
            "source_element_id": str(item.get("source_element_id") or ""),
            "target_element_id": str(item.get("target_element_id") or ""),
            "relation_type": str(item.get("relation_type") or ""),
            "evidence_ids": list(item.get("evidence_ids") or []),
        }
        for item in scene_program.get("relations") or []
        if isinstance(item, dict) and str(item.get("relation_id") or "")
    }
    return {"objects": objects, "relations": relations}


def _design_bible(spec: dict[str, Any]) -> VideoDesignBible:
    payload = dict(spec.get("video_design_bible") or {})
    if payload:
        signature_payload = {key: value for key, value in payload.items() if key != "signature"}
        if payload.get("signature") == _signature(signature_payload):
            return VideoDesignBible(
                version=str(payload.get("version") or VIDEO_DESIGN_BIBLE_VERSION),
                design_id=str(payload.get("design_id") or ""),
                palette_sequence=[
                    dict(item)
                    for item in payload.get("palette_sequence") or []
                    if isinstance(item, dict)
                ],
                typography_anchor=str(payload.get("typography_anchor") or ""),
                continuity_motif=str(payload.get("continuity_motif") or ""),
                repetition_window=_integer(payload.get("repetition_window"), 3),
                max_card_ratio=float(payload.get("max_card_ratio") or 0.18),
                forbidden_repetition=[
                    str(item) for item in payload.get("forbidden_repetition") or []
                ],
                signature=str(payload.get("signature") or ""),
            )
    return build_video_design_bible([spec])


def _rationale(scene_type: str, medium: str, proof_encoding: str) -> str:
    return (
        f"Render {scene_type or 'grounded explanation'} as {medium.replace('_', ' ')} "
        f"while preserving the {proof_encoding or 'signed'} proof structure."
    )


def _payload(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        return dict(value.to_dict())
    return dict(value or {})


def _signature(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _safe_id(value: Any) -> str:
    return "".join(
        char if char.isalnum() or char in {"-", "_"} else "-"
        for char in str(value or "world").lower()
    ).strip("-_") or "world"


def _integer(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


__all__ = [
    "CANVAS_SYSTEMS",
    "CARD_POLICIES",
    "MEDIUM_FAMILIES",
    "VIDEO_DESIGN_BIBLE_VERSION",
    "VISUAL_WORLD_VERSION",
    "VideoDesignBible",
    "VisualFingerprint",
    "VisualWorldProgram",
    "VisualWorldValidation",
    "build_video_design_bible",
    "build_visual_world_program",
    "validate_visual_world_program",
]
