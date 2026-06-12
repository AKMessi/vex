from __future__ import annotations

import hashlib
import html
import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from vex_hyperframes.safety import validate_authored_html_safety


SCENE_PROGRAM_VERSION = "hyperframes-scene-program-v2"
ALLOWED_ELEMENT_KINDS = {
    "decision",
    "evidence",
    "interface_state",
    "metric",
    "quote",
    "state",
    "step",
}
ALLOWED_LAYOUT_FAMILIES = {
    "focal_gate",
    "layered_flow",
    "linear_trace",
    "radial_evidence",
    "split_register",
}
ALLOWED_MOTION_KINDS = {
    "activate",
    "focus",
    "hold",
    "morph",
    "reveal",
    "trace",
    "travel",
}


@dataclass(frozen=True)
class SceneElement:
    element_id: str
    object_id: str
    kind: str
    role: str
    text: str
    evidence_ids: list[str]
    fact_ids: list[str]
    beat_ids: list[str]
    dom_selector: str
    x: float
    y: float
    width: float
    height: float
    z_index: int
    visible_start: float
    visible_end: float
    emphasis: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SceneRelation:
    relation_id: str
    source_element_id: str
    target_element_id: str
    relation_type: str
    evidence_ids: list[str]
    beat_id: str
    dom_selector: str
    reveal_fraction: float
    required: bool = True
    strength: float = 0.72

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SceneMotion:
    motion_id: str
    target_id: str
    target_type: str
    kind: str
    beat_id: str
    start_fraction: float
    end_fraction: float
    path: list[tuple[float, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SceneProgram:
    version: str
    program_id: str
    visual_id: str
    blueprint_id: str
    proof_program_id: str
    scene_type: str
    layout_family: str
    graph_signature: str
    semantic_signature: str
    elements: list[SceneElement]
    relations: list[SceneRelation]
    motions: list[SceneMotion]
    final_hold_start: float
    program_signature: str

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "elements": [item.to_dict() for item in self.elements],
            "relations": [item.to_dict() for item in self.relations],
            "motions": [item.to_dict() for item in self.motions],
        }


@dataclass(frozen=True)
class SceneProgramValidation:
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    object_coverage: float = 0.0
    relation_coverage: float = 0.0
    grounded_copy_ratio: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "object_coverage": round(float(self.object_coverage), 4),
            "relation_coverage": round(float(self.relation_coverage), 4),
            "grounded_copy_ratio": round(float(self.grounded_copy_ratio), 4),
        }


@dataclass(frozen=True)
class CompiledSceneStage:
    html: str
    metadata: dict[str, Any]


def build_scene_program(
    ir: Any,
    claim_graph: Any,
    storyboard: list[Any],
    *,
    blueprint_id: str,
    proof_program_id: str,
    proof_encoding: str,
    semantic_signature: str,
) -> SceneProgram:
    ir_payload = _payload(ir)
    graph_payload = _payload(claim_graph)
    panels = [_payload(item) for item in storyboard]
    nodes = [
        dict(item)
        for item in graph_payload.get("nodes") or []
        if isinstance(item, dict)
    ][:8]
    if not nodes:
        raise ValueError("Cannot compile Scene Program V2 without claim-graph nodes.")
    layout_family = _layout_family(proof_encoding)
    positions = _layout_positions(layout_family, nodes, graph_payload)
    ir_objects = {
        str(item.get("object_id") or ""): dict(item)
        for item in ir_payload.get("objects") or []
        if isinstance(item, dict)
    }
    element_by_object: dict[str, str] = {}
    elements: list[SceneElement] = []
    for index, node in enumerate(nodes):
        object_id = str(node.get("node_id") or "")
        element_id = f"element_{_safe_id(object_id)}"
        element_by_object[object_id] = element_id
        beat_ids = _beat_ids_for_object(object_id, panels)
        visible_start = _visible_start_for_object(object_id, panels)
        source_object = ir_objects.get(object_id, {})
        x, y, width, height = positions[index]
        elements.append(
            SceneElement(
                element_id=element_id,
                object_id=object_id,
                kind=_element_kind(str(node.get("role") or "")),
                role=str(node.get("role") or ""),
                text=str(node.get("label") or ""),
                evidence_ids=_strings(node.get("evidence_ids")),
                fact_ids=_strings(node.get("fact_ids")),
                beat_ids=beat_ids,
                dom_selector=f'[data-element-id="{element_id}"]',
                x=x,
                y=y,
                width=width,
                height=height,
                z_index=10 + index,
                visible_start=visible_start,
                visible_end=1.0,
                emphasis=_bounded(source_object.get("emphasis"), 0.5),
            )
        )
    relations: list[SceneRelation] = []
    for item in graph_payload.get("relations") or []:
        if not isinstance(item, dict):
            continue
        relation_id = str(item.get("relation_id") or "")
        source_id = element_by_object.get(str(item.get("source_id") or ""))
        target_id = element_by_object.get(str(item.get("target_id") or ""))
        if not relation_id or not source_id or not target_id:
            continue
        beat_id, reveal_fraction = _relation_beat(
            str(item.get("source_id") or ""),
            str(item.get("target_id") or ""),
            panels,
        )
        relations.append(
            SceneRelation(
                relation_id=relation_id,
                source_element_id=source_id,
                target_element_id=target_id,
                relation_type=str(item.get("relation_type") or ""),
                evidence_ids=_strings(item.get("evidence_ids")),
                beat_id=beat_id,
                dom_selector=f'[data-relation-id="{relation_id}"]',
                reveal_fraction=reveal_fraction,
                required=bool(item.get("required", True)),
                strength=0.86 if bool(item.get("required", True)) else 0.64,
            )
        )
    motions = _build_motions(elements, relations, panels)
    base_payload = {
        "version": SCENE_PROGRAM_VERSION,
        "program_id": f"{_safe_id(proof_program_id)}-scene-v2",
        "visual_id": str(ir_payload.get("visual_id") or ""),
        "blueprint_id": str(blueprint_id or ""),
        "proof_program_id": str(proof_program_id or ""),
        "scene_type": str(ir_payload.get("scene_type") or ""),
        "layout_family": layout_family,
        "graph_signature": str(graph_payload.get("graph_signature") or ""),
        "semantic_signature": str(semantic_signature or ""),
        "elements": [item.to_dict() for item in elements],
        "relations": [item.to_dict() for item in relations],
        "motions": [item.to_dict() for item in motions],
        "final_hold_start": _final_hold_start(panels),
    }
    return SceneProgram(
        version=SCENE_PROGRAM_VERSION,
        program_id=str(base_payload["program_id"]),
        visual_id=str(base_payload["visual_id"]),
        blueprint_id=str(base_payload["blueprint_id"]),
        proof_program_id=str(base_payload["proof_program_id"]),
        scene_type=str(base_payload["scene_type"]),
        layout_family=layout_family,
        graph_signature=str(base_payload["graph_signature"]),
        semantic_signature=str(base_payload["semantic_signature"]),
        elements=elements,
        relations=relations,
        motions=motions,
        final_hold_start=float(base_payload["final_hold_start"]),
        program_signature=_signature(base_payload),
    )


def validate_scene_program(
    program: SceneProgram | dict[str, Any],
    *,
    ir: Any,
    claim_graph: Any,
) -> SceneProgramValidation:
    payload = program.to_dict() if isinstance(program, SceneProgram) else dict(program or {})
    ir_payload = _payload(ir)
    graph_payload = _payload(claim_graph)
    errors: list[str] = []
    warnings: list[str] = []
    if payload.get("version") != SCENE_PROGRAM_VERSION:
        errors.append("unsupported_scene_program_version")
    if payload.get("layout_family") not in ALLOWED_LAYOUT_FAMILIES:
        errors.append("unsupported_scene_layout_family")
    expected_signature = _signature(
        {key: value for key, value in payload.items() if key != "program_signature"}
    )
    if not payload.get("program_signature"):
        errors.append("scene_program_missing_signature")
    elif payload.get("program_signature") != expected_signature:
        errors.append("scene_program_signature_mismatch")
    if payload.get("graph_signature") != graph_payload.get("graph_signature"):
        errors.append("scene_program_graph_signature_mismatch")

    elements = [
        dict(item)
        for item in payload.get("elements") or []
        if isinstance(item, dict)
    ]
    relations = [
        dict(item)
        for item in payload.get("relations") or []
        if isinstance(item, dict)
    ]
    motions = [
        dict(item)
        for item in payload.get("motions") or []
        if isinstance(item, dict)
    ]
    graph_nodes = {
        str(item.get("node_id") or ""): dict(item)
        for item in graph_payload.get("nodes") or []
        if isinstance(item, dict) and str(item.get("node_id") or "")
    }
    graph_relations = {
        str(item.get("relation_id") or ""): dict(item)
        for item in graph_payload.get("relations") or []
        if isinstance(item, dict) and bool(item.get("required", True))
    }
    ir_labels = {
        str(item.get("object_id") or ""): str(item.get("label") or "")
        for item in ir_payload.get("objects") or []
        if isinstance(item, dict)
    }
    element_ids: set[str] = set()
    represented_objects: set[str] = set()
    grounded_copy = 0
    for item in elements:
        element_id = str(item.get("element_id") or "")
        object_id = str(item.get("object_id") or "")
        if not element_id or element_id in element_ids:
            errors.append(f"invalid_or_duplicate_element_id:{element_id or 'missing'}")
        element_ids.add(element_id)
        if object_id not in graph_nodes:
            errors.append(f"element_references_unknown_object:{element_id}")
        else:
            represented_objects.add(object_id)
        if item.get("kind") not in ALLOWED_ELEMENT_KINDS:
            errors.append(f"unsupported_element_kind:{element_id}")
        if _normalize(item.get("text")) == _normalize(ir_labels.get(object_id)):
            grounded_copy += 1
        else:
            errors.append(f"element_copy_is_not_grounded:{element_id}")
        if item.get("dom_selector") != f'[data-element-id="{element_id}"]':
            errors.append(f"element_selector_mismatch:{element_id}")
        _validate_box(item, element_id, errors)
        start = _bounded(item.get("visible_start"), -1.0)
        end = _bounded(item.get("visible_end"), -1.0)
        if start < 0.0 or end > 1.0 or end <= start:
            errors.append(f"element_visibility_invalid:{element_id}")
        evidence_ids = set(_strings(item.get("evidence_ids")))
        expected_evidence = set(_strings(graph_nodes.get(object_id, {}).get("evidence_ids")))
        if not expected_evidence.issubset(evidence_ids):
            errors.append(f"element_evidence_incomplete:{element_id}")

    represented_relations: set[str] = set()
    for item in relations:
        relation_id = str(item.get("relation_id") or "")
        if relation_id not in graph_relations:
            errors.append(f"scene_relation_unknown:{relation_id or 'missing'}")
            continue
        represented_relations.add(relation_id)
        if item.get("source_element_id") not in element_ids:
            errors.append(f"scene_relation_unknown_source:{relation_id}")
        if item.get("target_element_id") not in element_ids:
            errors.append(f"scene_relation_unknown_target:{relation_id}")
        if item.get("dom_selector") != f'[data-relation-id="{relation_id}"]':
            errors.append(f"relation_selector_mismatch:{relation_id}")
        if not 0.0 <= _bounded(item.get("reveal_fraction"), -1.0) <= 1.0:
            errors.append(f"scene_relation_timing_invalid:{relation_id}")
        if not 0.2 <= _bounded(item.get("strength"), -1.0) <= 1.0:
            errors.append(f"scene_relation_strength_invalid:{relation_id}")
        expected = graph_relations[relation_id]
        if item.get("relation_type") != expected.get("relation_type"):
            errors.append(f"scene_relation_type_mismatch:{relation_id}")
        if not set(_strings(expected.get("evidence_ids"))).issubset(
            set(_strings(item.get("evidence_ids")))
        ):
            errors.append(f"scene_relation_evidence_incomplete:{relation_id}")

    motion_targets = {str(item.get("target_id") or "") for item in motions}
    for item in motions:
        motion_id = str(item.get("motion_id") or "missing")
        if item.get("kind") not in ALLOWED_MOTION_KINDS:
            errors.append(f"unsupported_motion_kind:{motion_id}")
        if item.get("target_type") not in {"element", "relation"}:
            errors.append(f"unsupported_motion_target_type:{motion_id}")
        start = _bounded(item.get("start_fraction"), -1.0)
        end = _bounded(item.get("end_fraction"), -1.0)
        if start < 0.0 or end > 1.0 or end <= start:
            errors.append(f"motion_timing_invalid:{motion_id}")
    for element_id in element_ids - motion_targets:
        warnings.append(f"element_has_no_motion:{element_id}")

    overlaps = _overlap_pairs(elements)
    if overlaps:
        errors.append("scene_program_elements_overlap:" + ",".join(overlaps[:6]))
    object_coverage = len(represented_objects) / max(len(graph_nodes), 1)
    relation_coverage = len(represented_relations) / max(len(graph_relations), 1)
    copy_ratio = grounded_copy / max(len(elements), 1)
    if object_coverage < 1.0:
        errors.append("scene_program_object_coverage_below_100_percent")
    if relation_coverage < 1.0:
        errors.append("scene_program_relation_coverage_below_100_percent")
    return SceneProgramValidation(
        passed=not errors and copy_ratio == 1.0,
        errors=_unique(errors),
        warnings=_unique(warnings),
        object_coverage=object_coverage,
        relation_coverage=relation_coverage,
        grounded_copy_ratio=copy_ratio,
    )


def compile_scene_stage(
    program: SceneProgram | dict[str, Any],
    *,
    ir: Any,
    claim_graph: Any,
) -> CompiledSceneStage:
    payload = program.to_dict() if isinstance(program, SceneProgram) else dict(program or {})
    validation = validate_scene_program(payload, ir=ir, claim_graph=claim_graph)
    if not validation.passed:
        raise ValueError("Unsafe Scene Program V2: " + "; ".join(validation.errors))
    elements = [dict(item) for item in payload.get("elements") or []]
    element_by_id = {
        str(item.get("element_id") or ""): item
        for item in elements
    }
    motions = {
        str(item.get("target_id") or ""): dict(item)
        for item in payload.get("motions") or []
        if isinstance(item, dict) and item.get("target_type") == "element"
    }
    element_html = "\n".join(
        _element_html(item, motions.get(str(item.get("element_id") or ""), {}))
        for item in elements
    )
    relation_html = "\n".join(
        _relation_html(dict(item), element_by_id)
        for item in payload.get("relations") or []
        if isinstance(item, dict)
    )
    scoped_id = _safe_id(payload.get("program_id") or "scene-program-v2")
    fragment = f"""
      <style>
        #{scoped_id} {{ position:absolute; inset:0; overflow:hidden; }}
        #{scoped_id} .scene-v2-relations {{ position:absolute; inset:0; width:100%; height:100%; overflow:visible; }}
        #{scoped_id} .scene-v2-relation {{ stroke:var(--accent-2); stroke-width:.5; vector-effect:non-scaling-stroke; opacity:.78; transform-origin:center; transform:scaleX(var(--line-progress, 0)); }}
        #{scoped_id} .scene-v2-relation.relation-branches_to_high, #{scoped_id} .scene-v2-relation.relation-produces {{ stroke:var(--accent); }}
        #{scoped_id} .scene-v2-element {{ position:absolute; display:grid; align-content:center; min-width:0; padding:16px 18px; border:1px solid color-mix(in srgb, var(--stroke) 64%, transparent); border-left:4px solid var(--accent-2); background:color-mix(in srgb, var(--panel) 90%, transparent); box-shadow:0 20px 48px color-mix(in srgb, black 24%, transparent); overflow:hidden; }}
        #{scoped_id} .scene-v2-element.role-result, #{scoped_id} .scene-v2-element.role-branch_high, #{scoped_id} .scene-v2-element.role-metric {{ border-left-color:var(--accent); background:color-mix(in srgb, var(--panel) 76%, var(--accent) 8%); }}
        #{scoped_id} .scene-v2-element span {{ color:var(--accent-2); font-size:12px; font-weight:850; text-transform:uppercase; }}
        #{scoped_id} .scene-v2-element b {{ margin-top:7px; color:var(--text); font-size:clamp(17px, 2.05vw, 25px); line-height:1.08; overflow-wrap:anywhere; }}
        #{scoped_id} .kind-metric b {{ font-size:clamp(22px, 2.8vw, 36px); }}
        #{scoped_id} .kind-quote {{ border:0; border-left:5px solid var(--accent); background:color-mix(in srgb, var(--panel) 72%, transparent); }}
      </style>
      <section id="{scoped_id}" class="scene-program-v2 layout-{html.escape(str(payload.get("layout_family") or ""), quote=True)}"
        data-scene-program-version="{SCENE_PROGRAM_VERSION}"
        data-program-signature="{html.escape(str(payload.get("program_signature") or ""), quote=True)}">
        <svg class="scene-v2-relations" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">{relation_html}</svg>
        {element_html}
      </section>
    """
    safety = validate_authored_html_safety(fragment)
    if not safety.safe:
        raise ValueError("Compiled Scene Program V2 failed safety validation: " + "; ".join(safety.errors))
    return CompiledSceneStage(
        html=fragment,
        metadata={
            "program_id": str(payload.get("program_id") or ""),
            "version": SCENE_PROGRAM_VERSION,
            "layout_family": str(payload.get("layout_family") or ""),
            "program_signature": str(payload.get("program_signature") or ""),
            "element_count": len(elements),
            "relation_count": len(payload.get("relations") or []),
            "motion_count": len(payload.get("motions") or []),
            "object_coverage": validation.object_coverage,
            "relation_coverage": validation.relation_coverage,
            "grounded_copy_ratio": validation.grounded_copy_ratio,
            "safety": safety.to_dict(),
        },
    )


def _element_html(item: dict[str, Any], motion: dict[str, Any]) -> str:
    element_id = str(item.get("element_id") or "")
    x = _bounded(item.get("x"), 0.5)
    y = _bounded(item.get("y"), 0.5)
    width = _bounded(item.get("width"), 0.25)
    height = _bounded(item.get("height"), 0.18)
    start = _bounded(motion.get("start_fraction"), _bounded(item.get("visible_start"), 0.1))
    end = _bounded(motion.get("end_fraction"), min(start + 0.34, 0.9))
    anim = {
        "activate": "pop",
        "focus": "scale",
        "hold": "scale",
        "morph": "slide-right",
        "reveal": "rise",
        "trace": "slide-right",
        "travel": "slide-right",
    }.get(str(motion.get("kind") or ""), "rise")
    return (
        f'<article id="{html.escape(element_id, quote=True)}" '
        f'class="scene-v2-element kind-{html.escape(str(item.get("kind") or ""), quote=True)} '
        f'role-{html.escape(str(item.get("role") or ""), quote=True)}" '
        f'data-element-id="{html.escape(element_id, quote=True)}" '
        f'data-object-id="{html.escape(str(item.get("object_id") or ""), quote=True)}" '
        f'data-beat-ids="{html.escape(",".join(_strings(item.get("beat_ids"))), quote=True)}" '
        f'data-evidence-ids="{html.escape(",".join(_strings(item.get("evidence_ids"))), quote=True)}" '
        f'data-anim="{anim}" data-delay="{start:.3f}" data-span="{max(end - start, 0.12):.3f}" '
        f'data-y="22" data-scale="0.940" '
        f'style="left:{(x - width / 2) * 100:.3f}%;top:{(y - height / 2) * 100:.3f}%;'
        f'width:{width * 100:.3f}%;height:{height * 100:.3f}%;z-index:{int(item.get("z_index") or 10)};">'
        f'<span>{html.escape(str(item.get("role") or "").replace("_", " "), quote=True)}</span>'
        f'<b>{html.escape(str(item.get("text") or ""), quote=True)}</b>'
        "</article>"
    )


def _relation_html(
    item: dict[str, Any],
    element_by_id: dict[str, dict[str, Any]],
) -> str:
    source = element_by_id[str(item.get("source_element_id") or "")]
    target = element_by_id[str(item.get("target_element_id") or "")]
    relation_id = str(item.get("relation_id") or "")
    relation_type = str(item.get("relation_type") or "")
    strength = _bounded(item.get("strength"), 0.72)
    return (
        f'<line class="scene-v2-relation relation-{html.escape(relation_type, quote=True)}" data-line '
        f'data-relation-id="{html.escape(relation_id, quote=True)}" '
        f'data-beat-id="{html.escape(str(item.get("beat_id") or ""), quote=True)}" '
        f'data-evidence-ids="{html.escape(",".join(_strings(item.get("evidence_ids"))), quote=True)}" '
        f'data-relation-strength="{strength:.3f}" '
        f'data-delay="{_bounded(item.get("reveal_fraction"), 0.2):.3f}" '
        f'x1="{_bounded(source.get("x"), 0.5) * 100:.3f}" '
        f'y1="{_bounded(source.get("y"), 0.5) * 100:.3f}" '
        f'x2="{_bounded(target.get("x"), 0.5) * 100:.3f}" '
        f'y2="{_bounded(target.get("y"), 0.5) * 100:.3f}" '
        f'style="stroke-width:{0.32 + strength * 0.42:.3f};opacity:{0.46 + strength * 0.5:.3f}" />'
    )


def _layout_positions(
    family: str,
    nodes: list[dict[str, Any]],
    graph: dict[str, Any],
) -> list[tuple[float, float, float, float]]:
    count = len(nodes)
    if family == "split_register":
        return _split_positions(count)
    if family == "focal_gate":
        return _focal_positions(nodes)
    if family == "layered_flow":
        return _layered_positions(count)
    if family == "radial_evidence":
        return _radial_positions(nodes, graph)
    return _linear_positions(count)


def _linear_positions(count: int) -> list[tuple[float, float, float, float]]:
    if count <= 4:
        width = min(0.22, 0.82 / max(count, 1))
        return [
            (0.10 + width / 2 + index * ((0.80 - width) / max(count - 1, 1)), 0.5, width, 0.24)
            for index in range(count)
        ]
    columns = 3
    return [
        (
            0.2 + (index % columns) * 0.3,
            0.29 + (index // columns) * 0.34,
            0.24,
            0.22,
        )
        for index in range(count)
    ]


def _split_positions(count: int) -> list[tuple[float, float, float, float]]:
    if count == 1:
        return [(0.5, 0.5, 0.42, 0.3)]
    positions: list[tuple[float, float, float, float]] = []
    left_count = (count + 1) // 2
    right_count = count - left_count
    for index in range(left_count):
        positions.append((0.27, (index + 1) / (left_count + 1), 0.36, min(0.24, 0.68 / left_count)))
    for index in range(right_count):
        positions.append((0.73, (index + 1) / (right_count + 1), 0.36, min(0.24, 0.68 / max(right_count, 1))))
    return positions


def _focal_positions(nodes: list[dict[str, Any]]) -> list[tuple[float, float, float, float]]:
    count = len(nodes)
    focus_index = max(
        range(count),
        key=lambda index: (
            1 if str(nodes[index].get("role") or "") in {"decision", "intervention", "metric", "result"} else 0,
            len(_strings(nodes[index].get("evidence_ids"))),
            index,
        ),
    )
    satellites = [index for index in range(count) if index != focus_index]
    positions: list[tuple[float, float, float, float] | None] = [None] * count
    positions[focus_index] = (0.5, 0.5, 0.34, 0.28)
    anchors = [(0.19, 0.24), (0.81, 0.24), (0.19, 0.76), (0.81, 0.76), (0.5, 0.16), (0.5, 0.84), (0.14, 0.5)]
    for index, node_index in enumerate(satellites):
        x, y = anchors[index]
        positions[node_index] = (x, y, 0.25, 0.18)
    return [item or (0.5, 0.5, 0.25, 0.18) for item in positions]


def _layered_positions(count: int) -> list[tuple[float, float, float, float]]:
    columns = 2 if count > 3 else 1
    rows = math.ceil(count / columns)
    positions: list[tuple[float, float, float, float]] = []
    for index in range(count):
        row = index // columns
        column = index % columns
        x = 0.5 if columns == 1 or (row == rows - 1 and count % 2 == 1) else 0.29 + column * 0.42
        y = 0.18 + row * (0.64 / max(rows - 1, 1))
        positions.append((x, y, 0.34 if columns == 2 else 0.46, min(0.2, 0.62 / rows)))
    return positions


def _radial_positions(
    nodes: list[dict[str, Any]],
    graph: dict[str, Any],
) -> list[tuple[float, float, float, float]]:
    incoming: dict[str, int] = {}
    for relation in graph.get("relations") or []:
        if isinstance(relation, dict):
            target = str(relation.get("target_id") or "")
            incoming[target] = incoming.get(target, 0) + 1
    center_index = max(
        range(len(nodes)),
        key=lambda index: (incoming.get(str(nodes[index].get("node_id") or ""), 0), index),
    )
    positions: list[tuple[float, float, float, float] | None] = [None] * len(nodes)
    positions[center_index] = (0.5, 0.5, 0.32, 0.25)
    satellites = [index for index in range(len(nodes)) if index != center_index]
    for order, node_index in enumerate(satellites):
        angle = (-math.pi / 2) + order * (2 * math.pi / max(len(satellites), 1))
        positions[node_index] = (
            0.5 + math.cos(angle) * 0.34,
            0.5 + math.sin(angle) * 0.32,
            0.23,
            0.17,
        )
    return [item or (0.5, 0.5, 0.23, 0.17) for item in positions]


def _build_motions(
    elements: list[SceneElement],
    relations: list[SceneRelation],
    panels: list[dict[str, Any]],
) -> list[SceneMotion]:
    motions: list[SceneMotion] = []
    for index, element in enumerate(elements):
        panel = next(
            (
                item
                for item in panels
                if str(item.get("focus_object_id") or "") == element.object_id
            ),
            {},
        )
        start = _bounded(panel.get("start_fraction"), element.visible_start)
        end = max(start + 0.12, _bounded(panel.get("end_fraction"), min(start + 0.34, 0.92)))
        motions.append(
            SceneMotion(
                motion_id=f"motion_element_{index + 1:02d}",
                target_id=element.element_id,
                target_type="element",
                kind=_motion_kind(element.role, str(panel.get("phase") or "")),
                beat_id=str(panel.get("panel_id") or (element.beat_ids[0] if element.beat_ids else "")),
                start_fraction=round(start, 3),
                end_fraction=round(min(end, 0.96), 3),
            )
        )
    for index, relation in enumerate(relations):
        motions.append(
            SceneMotion(
                motion_id=f"motion_relation_{index + 1:02d}",
                target_id=relation.relation_id,
                target_type="relation",
                kind="trace",
                beat_id=relation.beat_id,
                start_fraction=relation.reveal_fraction,
                end_fraction=round(min(relation.reveal_fraction + 0.22, 0.96), 3),
            )
        )
    return motions


def _relation_beat(
    source_object_id: str,
    target_object_id: str,
    panels: list[dict[str, Any]],
) -> tuple[str, float]:
    for panel in panels:
        visible = set(_strings(panel.get("visible_object_ids")))
        if source_object_id in visible and target_object_id in visible:
            return (
                str(panel.get("panel_id") or ""),
                round(_bounded(panel.get("end_fraction"), 0.5), 3),
            )
    return ("", 0.5)


def _beat_ids_for_object(object_id: str, panels: list[dict[str, Any]]) -> list[str]:
    focused = [
        str(item.get("panel_id") or "")
        for item in panels
        if str(item.get("focus_object_id") or "") == object_id
    ]
    if focused:
        return _unique(focused)
    return _unique(
        [
            str(item.get("panel_id") or "")
            for item in panels
            if object_id in _strings(item.get("visible_object_ids"))
        ]
    )


def _visible_start_for_object(object_id: str, panels: list[dict[str, Any]]) -> float:
    starts = [
        _bounded(item.get("start_fraction"), 0.0)
        for item in panels
        if object_id in _strings(item.get("visible_object_ids"))
    ]
    return round(min(starts) if starts else 0.08, 3)


def _final_hold_start(panels: list[dict[str, Any]]) -> float:
    if not panels:
        return 0.82
    return round(min(0.9, max(0.72, _bounded(panels[-1].get("start_fraction"), 0.78))), 3)


def _layout_family(value: Any) -> str:
    cleaned = str(value or "linear_trace").strip().lower().replace("-", "_")
    return cleaned if cleaned in ALLOWED_LAYOUT_FAMILIES else "linear_trace"


def _element_kind(role: str) -> str:
    return {
        "branch_high": "decision",
        "branch_low": "decision",
        "decision": "decision",
        "interface": "interface_state",
        "metric": "metric",
        "quote": "quote",
        "required": "evidence",
        "setup": "state",
    }.get(role, "step" if role in {"intervention", "mechanism"} else "state")


def _motion_kind(role: str, phase: str) -> str:
    if role in {"decision", "intervention"}:
        return "activate"
    if role in {"interface", "metric"}:
        return "focus"
    if role in {"result", "branch_high"} or phase == "resolve":
        return "hold"
    if role in {"problem", "setup"}:
        return "reveal"
    return "travel"


def _validate_box(item: dict[str, Any], element_id: str, errors: list[str]) -> None:
    x = _bounded(item.get("x"), -1.0)
    y = _bounded(item.get("y"), -1.0)
    width = _bounded(item.get("width"), -1.0)
    height = _bounded(item.get("height"), -1.0)
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        errors.append(f"element_position_out_of_bounds:{element_id}")
    if not (0.1 <= width <= 0.55 and 0.08 <= height <= 0.4):
        errors.append(f"element_size_out_of_bounds:{element_id}")
    if x - width / 2 < 0.0 or x + width / 2 > 1.0:
        errors.append(f"element_exceeds_horizontal_bounds:{element_id}")
    if y - height / 2 < 0.0 or y + height / 2 > 1.0:
        errors.append(f"element_exceeds_vertical_bounds:{element_id}")


def _overlap_pairs(elements: list[dict[str, Any]]) -> list[str]:
    pairs: list[str] = []
    for index, first in enumerate(elements):
        for second in elements[index + 1 :]:
            if _intersection_over_min_area(first, second) > 0.08:
                pairs.append(f"{first.get('element_id')}+{second.get('element_id')}")
    return pairs


def _intersection_over_min_area(first: dict[str, Any], second: dict[str, Any]) -> float:
    ax1 = _bounded(first.get("x"), 0.0) - _bounded(first.get("width"), 0.0) / 2
    ay1 = _bounded(first.get("y"), 0.0) - _bounded(first.get("height"), 0.0) / 2
    ax2 = ax1 + _bounded(first.get("width"), 0.0)
    ay2 = ay1 + _bounded(first.get("height"), 0.0)
    bx1 = _bounded(second.get("x"), 0.0) - _bounded(second.get("width"), 0.0) / 2
    by1 = _bounded(second.get("y"), 0.0) - _bounded(second.get("height"), 0.0) / 2
    bx2 = bx1 + _bounded(second.get("width"), 0.0)
    by2 = by1 + _bounded(second.get("height"), 0.0)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0,
        min(ay2, by2) - max(ay1, by1),
    )
    minimum_area = max(
        min(
            _bounded(first.get("width"), 0.0) * _bounded(first.get("height"), 0.0),
            _bounded(second.get("width"), 0.0) * _bounded(second.get("height"), 0.0),
        ),
        1e-9,
    )
    return intersection / minimum_area


def _payload(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        return dict(value.to_dict())
    return dict(value or {})


def _safe_id(value: Any) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", str(value or "scene")).strip("-_").lower() or "scene"


def _normalize(value: Any) -> str:
    return re.sub(r"[^a-z0-9%+./-]+", " ", str(value or "").lower()).strip()


def _strings(value: Any) -> list[str]:
    return _unique([str(item) for item in value or []])


def _bounded(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def _signature(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = str(value or "").strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result


__all__ = [
    "ALLOWED_ELEMENT_KINDS",
    "ALLOWED_LAYOUT_FAMILIES",
    "ALLOWED_MOTION_KINDS",
    "CompiledSceneStage",
    "SCENE_PROGRAM_VERSION",
    "SceneElement",
    "SceneMotion",
    "SceneProgram",
    "SceneProgramValidation",
    "SceneRelation",
    "build_scene_program",
    "compile_scene_stage",
    "validate_scene_program",
]
