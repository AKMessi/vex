from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from importlib.resources import files
from typing import Any, Iterable

from jsonschema import Draft202012Validator


SCENE_GRAPH_VERSION = "vex-scene-graph-v2"

MAX_SCENE_NODES = 64
MAX_SCENE_RELATIONS = 48
MAX_MOTION_TRACKS = 144
MAX_MOTION_KEYFRAMES = 12
MAX_CAPABILITY_COST = 240.0

ALLOWED_BACKENDS = {"dom", "svg", "skia", "three", "rive", "lottie"}
ALLOWED_MOTION_PROPERTIES = {
    "blur",
    "clip_progress",
    "color",
    "emphasis",
    "gradient_progress",
    "light_intensity",
    "mask_progress",
    "opacity",
    "path_progress",
    "perspective",
    "progress",
    "rotation",
    "rotation_x",
    "rotation_y",
    "scale",
    "skew_x",
    "skew_y",
    "stroke_progress",
    "translate_x",
    "translate_y",
    "translate_z",
}
ALLOWED_EASINGS = {
    "ease_in",
    "ease_in_out",
    "ease_out",
    "linear",
    "spring_gentle",
    "spring_snappy",
}
ALLOWED_CONSTRAINTS = {
    "align",
    "avoid_overlap",
    "contain",
    "distribute",
    "keep_inside_safe_area",
    "minimum_gap",
}
ALLOWED_RELATION_ROUTING = {"cubic", "direct", "orthogonal"}


@dataclass(frozen=True)
class RenderCapability:
    primitive: str
    backend: str
    fallback_backend: str
    motion_properties: tuple[str, ...]
    cost: float
    semantic: bool = True

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["motion_properties"] = list(self.motion_properties)
        return payload


_COMMON_MOTION = (
    "blur",
    "emphasis",
    "opacity",
    "rotation",
    "scale",
    "translate_x",
    "translate_y",
)
_DEPTH_MOTION = (
    *_COMMON_MOTION,
    "perspective",
    "rotation_x",
    "rotation_y",
    "translate_z",
)
_VECTOR_MOTION = (
    *_COMMON_MOTION,
    "color",
    "gradient_progress",
    "mask_progress",
    "path_progress",
    "stroke_progress",
)


CAPABILITY_REGISTRY: dict[str, RenderCapability] = {
    "data_chart": RenderCapability(
        "data_chart",
        "svg",
        "dom",
        (*_VECTOR_MOTION, "progress"),
        7.0,
    ),
    "graph_node": RenderCapability("graph_node", "svg", "dom", _VECTOR_MOTION, 3.0),
    "group": RenderCapability("group", "dom", "dom", _COMMON_MOTION, 1.0),
    "kinetic_text_run": RenderCapability(
        "kinetic_text_run",
        "dom",
        "dom",
        (*_COMMON_MOTION, "clip_progress", "color", "skew_x", "skew_y"),
        2.5,
    ),
    "mask_group": RenderCapability(
        "mask_group",
        "svg",
        "dom",
        (*_VECTOR_MOTION, "clip_progress"),
        4.0,
    ),
    "masked_media": RenderCapability(
        "masked_media",
        "dom",
        "dom",
        (*_DEPTH_MOTION, "clip_progress", "mask_progress"),
        6.0,
    ),
    "metric_mark": RenderCapability(
        "metric_mark",
        "dom",
        "dom",
        (*_COMMON_MOTION, "color", "progress"),
        3.0,
    ),
    "particle_field": RenderCapability(
        "particle_field",
        "skia",
        "dom",
        (*_DEPTH_MOTION, "color", "progress"),
        10.0,
        semantic=False,
    ),
    "semantic_token": RenderCapability(
        "semantic_token",
        "dom",
        "dom",
        (*_COMMON_MOTION, "color"),
        1.5,
    ),
    "text_block": RenderCapability(
        "text_block",
        "dom",
        "dom",
        (*_COMMON_MOTION, "clip_progress", "color"),
        2.0,
    ),
    "vector_icon": RenderCapability(
        "vector_icon",
        "svg",
        "dom",
        _VECTOR_MOTION,
        2.0,
    ),
    "vector_path": RenderCapability(
        "vector_path",
        "svg",
        "dom",
        (*_VECTOR_MOTION, "progress"),
        2.5,
    ),
    "vector_shape": RenderCapability(
        "vector_shape",
        "svg",
        "dom",
        _VECTOR_MOTION,
        2.5,
        semantic=False,
    ),
}

_TYPE_TO_PRIMITIVE = {
    "chart": "data_chart",
    "connector": "vector_path",
    "group": "group",
    "icon": "vector_icon",
    "image": "masked_media",
    "mask": "mask_group",
    "metric": "metric_mark",
    "particle": "particle_field",
    "path": "vector_path",
    "shape": "graph_node",
    "text": "text_block",
    "token": "semantic_token",
}
_REMOTE_URI_RE = re.compile(r"^[a-z][a-z0-9+.-]*://", flags=re.IGNORECASE)


@dataclass(frozen=True)
class SceneGraphValidation:
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    capability_cost: float = 0.0
    backend_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["capability_cost"] = round(float(self.capability_cost), 4)
        return payload


def canonical_scene_graph_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def scene_graph_signature(scene_graph: dict[str, Any]) -> str:
    unsigned = copy.deepcopy(dict(scene_graph or {}))
    unsigned.pop("signature", None)
    return hashlib.sha256(canonical_scene_graph_json(unsigned).encode("utf-8")).hexdigest()


def sign_scene_graph(scene_graph: dict[str, Any]) -> dict[str, Any]:
    signed = copy.deepcopy(dict(scene_graph or {}))
    signed["signature"] = scene_graph_signature(signed)
    return signed


def compile_scene_graph(
    open_visual_program: dict[str, Any],
    *,
    creative_direction: dict[str, Any],
) -> dict[str, Any]:
    """Compile a validated Open Visual Program into the bounded runtime graph.

    The Open Visual Program remains the authored semantic contract. SceneGraph v2
    is a deterministic capability plan: it chooses a renderer lane, normalizes
    motion and layout contracts, and declares the telemetry needed to prove the
    rendered result.
    """

    source = copy.deepcopy(dict(open_visual_program or {}))
    direction = copy.deepcopy(dict(creative_direction or {}))
    canvas = copy.deepcopy(dict(source.get("canvas") or {}))
    palette = copy.deepcopy(dict(source.get("palette") or {}))
    art_direction = dict(direction.get("art_direction") or {})
    choreography = dict(direction.get("choreography") or {})

    nodes = [_compile_node(item, index=index) for index, item in enumerate(source.get("elements") or [])]
    node_ids = {str(item["node_id"]) for item in nodes}
    relations = [
        _compile_relation(item, nodes=nodes, index=index)
        for index, item in enumerate(source.get("relations") or [])
        if str(item.get("source_id") or "") in node_ids
        and str(item.get("target_id") or "") in node_ids
    ]
    relation_ids = {str(item["relation_id"]) for item in relations}
    tracks = [
        _compile_track(
            item,
            index=index,
            phases=list(choreography.get("phases") or []),
        )
        for index, item in enumerate(source.get("tracks") or [])
        if str(item.get("target_id") or "") in node_ids | relation_ids
    ]
    phases = _compile_phases(
        choreography.get("phases") or [],
        nodes=nodes,
        tracks=tracks,
    )
    dependencies = [
        {
            "dependency_id": f"phase_dependency_{index + 1:02d}",
            "source_phase_id": str(source_phase["phase_id"]),
            "target_phase_id": str(target_phase["phase_id"]),
            "type": "precedes",
        }
        for index, (source_phase, target_phase) in enumerate(zip(phases, phases[1:]))
    ]
    constraints = [
        _compile_constraint(item, index=index)
        for index, item in enumerate(source.get("constraints") or [])
    ]
    used_primitives = sorted({str(item["primitive"]) for item in nodes})
    capability_manifest = [
        CAPABILITY_REGISTRY[primitive].to_dict() for primitive in used_primitives
    ]
    final_hold = _bounded_float(
        (source.get("quality_contract") or {}).get(
            "final_hold_start",
            (direction.get("quality_contract") or {}).get("final_hold_start", 0.8),
        ),
        0.8,
        minimum=0.5,
        maximum=0.95,
    )
    sample_times = sorted(
        {
            0.0,
            1.0,
            final_hold,
            *[
                round(_bounded_float(phase.get("start"), 0.0), 4)
                for phase in phases
            ],
            *[
                round(_bounded_float(phase.get("end"), 1.0), 4)
                for phase in phases
            ],
        }
    )
    unsigned = {
        "version": SCENE_GRAPH_VERSION,
        "source_program_id": str(source.get("program_id") or ""),
        "source_program_signature": str(source.get("signature") or ""),
        "evidence_signature": str(source.get("evidence_signature") or ""),
        "canvas": canvas,
        "design_system": {
            "palette": palette,
            "typography": _typography_system(art_direction),
            "material": str(art_direction.get("material_system") or "soft_solid_light"),
            "depth_model": str(art_direction.get("depth_model") or "orthographic"),
            "motif": str(art_direction.get("motif") or "guided_trace"),
            "texture_strength": _bounded_float(
                art_direction.get("texture_strength"),
                0.1,
                minimum=0.0,
                maximum=0.35,
            ),
        },
        "capability_manifest": capability_manifest,
        "assets": _compile_assets(nodes),
        "nodes": nodes,
        "relations": relations,
        "motion_graph": {
            "phases": phases,
            "dependencies": dependencies,
            "tracks": tracks,
            "reduced_motion_policy": str(
                choreography.get("reduced_motion_policy")
                or "preserve_semantic_state_without_translation"
            ),
            "maximum_simultaneous_motion": max(
                1,
                min(int(choreography.get("max_simultaneous_motion") or 3), 8),
            ),
        },
        "constraints": constraints,
        "quality_contract": {
            **copy.deepcopy(dict(source.get("quality_contract") or {})),
            "minimum_text_contrast": _bounded_float(
                (direction.get("quality_contract") or {}).get("minimum_text_contrast"),
                4.5,
                minimum=3.0,
                maximum=21.0,
            ),
            "minimum_graphic_contrast": _bounded_float(
                (direction.get("quality_contract") or {}).get("minimum_graphic_contrast"),
                3.0,
                minimum=1.0,
                maximum=21.0,
            ),
            "final_hold_start": final_hold,
        },
        "telemetry_contract": {
            "sample_times": sample_times,
            "required_probes": [
                "geometry",
                "motion_state",
                "relation_paths",
                "semantic_bindings",
                "text_bounds",
            ],
            "fail_on": [
                "clipped_text",
                "constraint_violation",
                "missing_semantic_binding",
                "relation_endpoint_detached",
                "unsafe_area_intrusion",
            ],
        },
    }
    unsigned_signature = hashlib.sha256(
        canonical_scene_graph_json(unsigned).encode("utf-8")
    ).hexdigest()
    graph = sign_scene_graph(
        {
            **unsigned,
            "scene_graph_id": f"scene-graph-{unsigned_signature[:16]}",
        }
    )
    validation = validate_scene_graph(graph)
    if not validation.passed:
        raise ValueError(
            "compiled_scene_graph_invalid:" + ",".join(validation.errors[:8])
        )
    return graph


def validate_scene_graph(scene_graph: dict[str, Any]) -> SceneGraphValidation:
    payload = dict(scene_graph or {})
    errors = _schema_errors(payload)
    warnings: list[str] = []

    if str(payload.get("version") or "") != SCENE_GRAPH_VERSION:
        errors.append("unsupported_scene_graph_version")
    signature = str(payload.get("signature") or "")
    if not signature:
        errors.append("scene_graph_unsigned")
    elif signature != scene_graph_signature(payload):
        errors.append("scene_graph_signature_mismatch")

    nodes = [item for item in payload.get("nodes") or [] if isinstance(item, dict)]
    relations = [
        item for item in payload.get("relations") or [] if isinstance(item, dict)
    ]
    tracks = [
        item
        for item in ((payload.get("motion_graph") or {}).get("tracks") or [])
        if isinstance(item, dict)
    ]
    if not nodes:
        errors.append("scene_graph_has_no_nodes")
    if len(nodes) > MAX_SCENE_NODES:
        errors.append("scene_graph_node_budget_exceeded")
    if len(relations) > MAX_SCENE_RELATIONS:
        errors.append("scene_graph_relation_budget_exceeded")
    if len(tracks) > MAX_MOTION_TRACKS:
        errors.append("scene_graph_motion_budget_exceeded")

    node_ids: set[str] = set()
    backend_counts: dict[str, int] = {}
    capability_cost = 0.0
    parent_by_node: dict[str, str] = {}
    for index, node in enumerate(nodes):
        node_id = str(node.get("node_id") or "")
        if not node_id or node_id in node_ids:
            errors.append(f"invalid_or_duplicate_scene_node:{node_id or index}")
        node_ids.add(node_id)
        primitive = str(node.get("primitive") or "")
        backend = str(node.get("backend") or "")
        capability = CAPABILITY_REGISTRY.get(primitive)
        if capability is None:
            errors.append(f"unsupported_scene_primitive:{node_id}:{primitive}")
        elif backend not in {capability.backend, capability.fallback_backend}:
            errors.append(f"unsupported_scene_backend:{node_id}:{backend}")
        if backend not in ALLOWED_BACKENDS:
            errors.append(f"unknown_scene_backend:{node_id}:{backend}")
        backend_counts[backend] = backend_counts.get(backend, 0) + 1
        repeat = max(1, min(int(_number(node.get("repeat"), 1.0)), 24))
        capability_cost += (capability.cost if capability else 20.0) * repeat
        layout = dict(node.get("layout") or {})
        if not _valid_layout(layout):
            errors.append(f"invalid_scene_layout:{node_id}")
        parent_id = str(node.get("parent_id") or "")
        if parent_id:
            parent_by_node[node_id] = parent_id
        asset = dict((node.get("content") or {}).get("asset") or {})
        uri = str(asset.get("uri") or "")
        if uri and _REMOTE_URI_RE.match(uri):
            errors.append(f"remote_scene_asset_forbidden:{node_id}")

    for node_id, parent_id in parent_by_node.items():
        if parent_id not in node_ids:
            errors.append(f"scene_node_unknown_parent:{node_id}:{parent_id}")
    errors.extend(_parent_cycle_errors(parent_by_node))

    relation_ids: set[str] = set()
    for index, relation in enumerate(relations):
        relation_id = str(relation.get("relation_id") or "")
        if not relation_id or relation_id in relation_ids:
            errors.append(
                f"invalid_or_duplicate_scene_relation:{relation_id or index}"
            )
        relation_ids.add(relation_id)
        if str(relation.get("source_id") or "") not in node_ids:
            errors.append(f"scene_relation_unknown_source:{relation_id}")
        if str(relation.get("target_id") or "") not in node_ids:
            errors.append(f"scene_relation_unknown_target:{relation_id}")
        if str(relation.get("routing") or "") not in ALLOWED_RELATION_ROUTING:
            errors.append(f"unsupported_scene_relation_routing:{relation_id}")
        capability_cost += 2.5

    target_ids = node_ids | relation_ids
    for index, track in enumerate(tracks):
        track_id = str(track.get("track_id") or f"track_{index}")
        target_id = str(track.get("target_id") or "")
        property_name = str(track.get("property") or "")
        if target_id not in target_ids:
            errors.append(f"scene_motion_unknown_target:{track_id}")
        if property_name not in ALLOWED_MOTION_PROPERTIES:
            errors.append(f"unsupported_scene_motion_property:{track_id}")
        if target_id in node_ids:
            target_node = next(
                (item for item in nodes if str(item.get("node_id") or "") == target_id),
                {},
            )
            capability = CAPABILITY_REGISTRY.get(str(target_node.get("primitive") or ""))
            if capability and property_name not in capability.motion_properties:
                errors.append(
                    f"scene_capability_rejects_motion:{track_id}:{property_name}"
                )
        keyframes = [
            item for item in track.get("keyframes") or [] if isinstance(item, dict)
        ]
        if not 2 <= len(keyframes) <= MAX_MOTION_KEYFRAMES:
            errors.append(f"invalid_scene_motion_keyframes:{track_id}")
        previous_t = -1.0
        for keyframe in keyframes:
            timestamp = _number(keyframe.get("t"), -1.0)
            if timestamp < previous_t or not 0.0 <= timestamp <= 1.0:
                errors.append(f"invalid_scene_motion_timeline:{track_id}")
                break
            previous_t = timestamp
            if str(keyframe.get("easing") or "") not in ALLOWED_EASINGS:
                errors.append(f"unsupported_scene_motion_easing:{track_id}")

    phases = [
        item
        for item in ((payload.get("motion_graph") or {}).get("phases") or [])
        if isinstance(item, dict)
    ]
    previous_phase_start = -1.0
    phase_ids: set[str] = set()
    for index, phase in enumerate(phases):
        phase_id = str(phase.get("phase_id") or "")
        if not phase_id or phase_id in phase_ids:
            errors.append(f"invalid_or_duplicate_motion_phase:{phase_id or index}")
        phase_ids.add(phase_id)
        start = _number(phase.get("start"), -1.0)
        end = _number(phase.get("end"), -1.0)
        if start < previous_phase_start or not 0.0 <= start < end <= 1.0:
            errors.append(f"invalid_motion_phase_timeline:{phase_id or index}")
        previous_phase_start = start

    for index, constraint in enumerate(payload.get("constraints") or []):
        if not isinstance(constraint, dict):
            continue
        constraint_id = str(constraint.get("constraint_id") or index)
        if str(constraint.get("type") or "") not in ALLOWED_CONSTRAINTS:
            errors.append(f"unsupported_scene_constraint:{constraint_id}")
        targets = [str(value) for value in constraint.get("targets") or []]
        if not targets or any(target not in node_ids for target in targets):
            errors.append(f"scene_constraint_unknown_target:{constraint_id}")
        if len(targets) != len(set(targets)):
            errors.append(f"scene_constraint_duplicate_target:{constraint_id}")
        if (
            str(constraint.get("type") or "") == "contain"
            and len(targets) < 2
        ):
            errors.append(f"scene_containment_requires_child:{constraint_id}")

    manifest = {
        str(item.get("primitive") or ""): dict(item)
        for item in payload.get("capability_manifest") or []
        if isinstance(item, dict)
    }
    used_primitives = {str(item.get("primitive") or "") for item in nodes}
    if used_primitives - set(manifest):
        errors.append("scene_graph_capability_manifest_incomplete")
    for primitive, item in manifest.items():
        capability = CAPABILITY_REGISTRY.get(primitive)
        if capability is None or item != capability.to_dict():
            errors.append(f"scene_graph_capability_manifest_tampered:{primitive}")

    if capability_cost > MAX_CAPABILITY_COST:
        errors.append("scene_graph_capability_budget_exceeded")
    if backend_counts.get("dom", 0) == len(nodes) and len(nodes) >= 4:
        warnings.append("scene_graph_uses_dom_only_visual_language")

    return SceneGraphValidation(
        passed=not errors,
        errors=_unique(errors),
        warnings=_unique(warnings),
        capability_cost=capability_cost,
        backend_counts=backend_counts,
    )


def _compile_node(item: dict[str, Any], *, index: int) -> dict[str, Any]:
    source = copy.deepcopy(dict(item or {}))
    element_type = str(source.get("type") or "shape")
    primitive = _primitive_for(source)
    capability = CAPABILITY_REGISTRY[primitive]
    layout = dict(source.get("layout") or {})
    return {
        "node_id": str(source.get("element_id") or f"node_{index + 1:02d}"),
        "parent_id": str(source.get("parent_id") or ""),
        "primitive": primitive,
        "backend": capability.backend,
        "fallback_backend": capability.fallback_backend,
        "role": str(source.get("role") or "evidence"),
        "binding": copy.deepcopy(dict(source.get("binding") or {})),
        "decorative": bool(source.get("decorative")),
        "layout": {
            "x": _bounded_float(layout.get("x"), 0.0),
            "y": _bounded_float(layout.get("y"), 0.0),
            "width": _bounded_float(layout.get("width"), 0.1, minimum=0.001),
            "height": _bounded_float(layout.get("height"), 0.1, minimum=0.001),
            "anchor": str(layout.get("anchor") or "top_left"),
            "z_index": _z_index(element_type, bool(source.get("decorative"))),
        },
        "content": {
            "text": str(source.get("text") or ""),
            "asset": copy.deepcopy(dict(source.get("asset") or {})),
            "data": copy.deepcopy(source.get("data") or []),
        },
        "style": copy.deepcopy(dict(source.get("style") or {})),
        "repeat": max(1, min(int(_number(source.get("repeat"), 1.0)), 24)),
        "telemetry": {
            "measure_geometry": True,
            "measure_text": bool(str(source.get("text") or "")),
            "semantic_probe": not bool(source.get("decorative")),
        },
    }


def _compile_relation(
    item: dict[str, Any],
    *,
    nodes: list[dict[str, Any]],
    index: int,
) -> dict[str, Any]:
    source = copy.deepcopy(dict(item or {}))
    by_id = {str(node["node_id"]): node for node in nodes}
    source_id = str(source.get("source_id") or "")
    target_id = str(source.get("target_id") or "")
    return {
        "relation_id": str(source.get("relation_id") or f"relation_{index + 1:02d}"),
        "source_id": source_id,
        "target_id": target_id,
        "type": str(source.get("type") or "relates"),
        "binding": copy.deepcopy(dict(source.get("binding") or {})),
        "primitive": "vector_path",
        "backend": "svg",
        "routing": _routing_for(by_id.get(source_id, {}), by_id.get(target_id, {})),
        "style": copy.deepcopy(dict(source.get("style") or {})),
        "telemetry": {
            "measure_path": True,
            "verify_endpoints": True,
        },
    }


def _compile_track(
    item: dict[str, Any],
    *,
    index: int,
    phases: list[dict[str, Any]],
) -> dict[str, Any]:
    source = copy.deepcopy(dict(item or {}))
    keyframes = [
        {
            "t": _bounded_float(frame.get("t"), 0.0),
            "value": copy.deepcopy(frame.get("value")),
            "easing": str(frame.get("easing") or "linear"),
        }
        for frame in source.get("keyframes") or []
        if isinstance(frame, dict)
    ]
    active_t = sum(float(frame["t"]) for frame in keyframes) / max(len(keyframes), 1)
    return {
        "track_id": str(source.get("track_id") or f"track_{index + 1:02d}"),
        "target_id": str(source.get("target_id") or ""),
        "property": _normalized_motion_property(source.get("property")),
        "semantic_intent": str(source.get("semantic_intent") or "semantic reveal"),
        "phase_id": _phase_for_t(phases, active_t),
        "keyframes": keyframes,
    }


def _compile_phases(
    phases: Iterable[dict[str, Any]],
    *,
    nodes: list[dict[str, Any]],
    tracks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    source_phases = [dict(item) for item in phases if isinstance(item, dict)]
    if not source_phases:
        source_phases = [
            {"phase": "establish", "start": 0.0, "end": 0.16, "job": "stable_context"},
            {"phase": "reveal", "start": 0.1, "end": 0.5, "job": "ordered_evidence"},
            {"phase": "relate", "start": 0.34, "end": 0.7, "job": "explain_connection"},
            {"phase": "resolve", "start": 0.64, "end": 0.82, "job": "focus_outcome"},
            {"phase": "hold", "start": 0.8, "end": 1.0, "job": "readable_final_state"},
        ]
    node_ids = {str(item["node_id"]) for item in nodes}
    result: list[dict[str, Any]] = []
    for index, phase in enumerate(source_phases):
        start = _bounded_float(phase.get("start"), 0.0)
        end = _bounded_float(phase.get("end"), 1.0)
        if end <= start:
            end = min(1.0, start + 0.01)
        phase_name = str(phase.get("phase") or f"phase_{index + 1:02d}")
        phase_id = f"phase_{index + 1:02d}_{_identifier(phase_name)}"
        target_ids = sorted(
            {
                str(track.get("target_id") or "")
                for track in tracks
                if str(track.get("phase_id") or "") in {phase_id, phase_name}
                and str(track.get("target_id") or "") in node_ids
            }
        )
        result.append(
            {
                "phase_id": phase_id,
                "role": phase_name,
                "start": start,
                "end": end,
                "job": str(phase.get("job") or "semantic_choreography"),
                "target_ids": target_ids,
            }
        )
    phase_by_name = {
        str(source.get("phase") or f"phase_{index + 1:02d}"): str(compiled["phase_id"])
        for index, (source, compiled) in enumerate(zip(source_phases, result))
    }
    for track in tracks:
        raw_phase = str(track.get("phase_id") or "")
        if raw_phase in phase_by_name:
            track["phase_id"] = phase_by_name[raw_phase]
    return result


def _compile_constraint(item: dict[str, Any], *, index: int) -> dict[str, Any]:
    source = dict(item or {})
    return {
        "constraint_id": str(
            source.get("constraint_id") or f"constraint_{index + 1:02d}"
        ),
        "type": str(source.get("type") or "keep_inside_safe_area"),
        "targets": [str(value) for value in source.get("targets") or []],
        "axis": str(source.get("axis") or "both"),
        "priority": max(1, min(int(_number(source.get("priority"), 100.0)), 1000)),
        "gap": _bounded_float(source.get("gap"), 0.02, minimum=0.0, maximum=0.5),
        "padding": _bounded_float(
            source.get("padding"),
            0.0,
            minimum=0.0,
            maximum=0.25,
        ),
    }


def _compile_assets(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for node in nodes:
        asset = dict((node.get("content") or {}).get("asset") or {})
        uri = str(asset.get("uri") or "")
        data_uri = str(asset.get("data_uri") or "")
        if not uri and not data_uri:
            continue
        digest_source = data_uri if data_uri else uri
        result.append(
            {
                "asset_id": f"asset-{hashlib.sha256(digest_source.encode('utf-8')).hexdigest()[:16]}",
                "node_id": str(node.get("node_id") or ""),
                "uri": uri,
                "embedded": bool(data_uri),
                "content_hash": str(asset.get("content_hash") or ""),
                "provenance": str(asset.get("provenance") or "program_supplied"),
                "semantic_role": str(node.get("role") or "evidence"),
            }
        )
    return result


def _primitive_for(element: dict[str, Any]) -> str:
    element_type = str(element.get("type") or "shape")
    role = str(element.get("role") or "").lower()
    decorative = bool(element.get("decorative"))
    if element_type == "text" and role in {"headline", "title", "takeaway"}:
        return "kinetic_text_run"
    if element_type == "shape" and decorative:
        return "vector_shape"
    return _TYPE_TO_PRIMITIVE.get(element_type, "vector_shape")


def _routing_for(source: dict[str, Any], target: dict[str, Any]) -> str:
    source_layout = dict(source.get("layout") or {})
    target_layout = dict(target.get("layout") or {})
    source_x = _number(source_layout.get("x"), 0.0) + _number(
        source_layout.get("width"), 0.0
    ) / 2.0
    source_y = _number(source_layout.get("y"), 0.0) + _number(
        source_layout.get("height"), 0.0
    ) / 2.0
    target_x = _number(target_layout.get("x"), 0.0) + _number(
        target_layout.get("width"), 0.0
    ) / 2.0
    target_y = _number(target_layout.get("y"), 0.0) + _number(
        target_layout.get("height"), 0.0
    ) / 2.0
    if abs(source_x - target_x) < 0.06 or abs(source_y - target_y) < 0.06:
        return "direct"
    if abs(source_x - target_x) > 0.18 and abs(source_y - target_y) > 0.18:
        return "orthogonal"
    return "cubic"


def _phase_for_t(phases: Iterable[dict[str, Any]], timestamp: float) -> str:
    normalized = [dict(item) for item in phases if isinstance(item, dict)]
    containing = [
        item
        for item in normalized
        if _number(item.get("start"), 0.0) <= timestamp <= _number(item.get("end"), 1.0)
    ]
    if containing:
        containing.sort(
            key=lambda item: _number(item.get("end"), 1.0)
            - _number(item.get("start"), 0.0)
        )
        return str(containing[0].get("phase") or "")
    return str(normalized[-1].get("phase") or "") if normalized else ""


def _typography_system(art_direction: dict[str, Any]) -> dict[str, Any]:
    system = str(art_direction.get("typography_system") or "architectural_label")
    family_by_system = {
        "architectural_label": "Inter, Arial, sans-serif",
        "documentary_caption": "Inter, Arial, sans-serif",
        "editorial_display": "Georgia, Times New Roman, serif",
        "numeric_monument": "Arial Black, Inter, sans-serif",
        "oversized_display": "Arial Black, Inter, sans-serif",
        "product_system": "Inter, Arial, sans-serif",
        "technical_label": "IBM Plex Sans, Inter, Arial, sans-serif",
    }
    return {
        "system": system,
        "font_family": family_by_system.get(system, "Inter, Arial, sans-serif"),
        "fallback_family": "Arial, sans-serif",
        "title_weight": 850,
        "body_weight": 600,
        "minimum_font_px": 18,
        "line_height": 1.08 if system in {"oversized_display", "numeric_monument"} else 1.16,
        "letter_spacing_em": -0.025 if system != "technical_label" else -0.01,
    }


def _normalized_motion_property(value: Any) -> str:
    property_name = str(value or "")
    if property_name == "progress":
        return "progress"
    return property_name


def _z_index(element_type: str, decorative: bool) -> int:
    if decorative:
        return 0
    if element_type in {"connector", "path"}:
        return 2
    if element_type in {"text", "metric", "icon"}:
        return 6
    return 4


def _valid_layout(layout: dict[str, Any]) -> bool:
    x = _number(layout.get("x"), -1.0)
    y = _number(layout.get("y"), -1.0)
    width = _number(layout.get("width"), -1.0)
    height = _number(layout.get("height"), -1.0)
    anchor = str(layout.get("anchor") or "")
    z_index = int(_number(layout.get("z_index"), -1.0))
    return (
        0.0 <= x <= 1.0
        and 0.0 <= y <= 1.0
        and 0.0 < width <= 1.0
        and 0.0 < height <= 1.0
        and x + width <= 1.0001
        and y + height <= 1.0001
        and anchor in {"bottom_left", "bottom_right", "center", "top_left", "top_right"}
        and 0 <= z_index <= 100
    )


def _parent_cycle_errors(parent_by_node: dict[str, str]) -> list[str]:
    errors: list[str] = []
    for node_id in parent_by_node:
        seen = {node_id}
        current = parent_by_node.get(node_id, "")
        while current:
            if current in seen:
                errors.append(f"scene_node_parent_cycle:{node_id}")
                break
            seen.add(current)
            current = parent_by_node.get(current, "")
    return errors


@lru_cache(maxsize=1)
def _schema_validator() -> Draft202012Validator:
    schema_path = files("vex_visuals").joinpath("scene_graph_v2.schema.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


def _schema_errors(payload: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for error in _schema_validator().iter_errors(payload):
        path = ".".join(str(item) for item in error.absolute_path) or "root"
        result.append(f"scene_graph_schema:{path}:{error.message}")
    return result


def _number(value: Any, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    if number != number or number in {float("inf"), float("-inf")}:
        return fallback
    return number


def _bounded_float(
    value: Any,
    fallback: float,
    *,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    return max(minimum, min(_number(value, fallback), maximum))


def _identifier(value: Any) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return cleaned[:48] or "phase"


def _unique(values: Iterable[str], *, limit: int = 50) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
        if len(result) >= limit:
            break
    return result


__all__ = [
    "ALLOWED_BACKENDS",
    "ALLOWED_CONSTRAINTS",
    "ALLOWED_MOTION_PROPERTIES",
    "CAPABILITY_REGISTRY",
    "MAX_CAPABILITY_COST",
    "RenderCapability",
    "SCENE_GRAPH_VERSION",
    "SceneGraphValidation",
    "compile_scene_graph",
    "scene_graph_signature",
    "sign_scene_graph",
    "validate_scene_graph",
]
