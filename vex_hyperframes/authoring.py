from __future__ import annotations

import html
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from visual_copy_contract import copy_allowed_for_binding
from vex_hyperframes.safety import validate_authored_html_safety


AUTHORING_VERSION = "hyperframes-scene-program-v1"
ALLOWED_PRIMITIVES = {
    "axis",
    "decision",
    "highlight",
    "interface_state",
    "label",
    "metric",
    "node",
    "quote",
    "state",
    "token",
}
ALLOWED_MOTIONS = {
    "activate",
    "focus",
    "lock",
    "morph",
    "pop",
    "reveal",
    "rise",
    "trace",
    "travel",
}


@dataclass(frozen=True)
class BespokePrimitive:
    primitive_id: str
    kind: str
    object_id: str
    role: str
    text: str
    x: float
    y: float
    width: float
    height: float
    emphasis: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BespokeConnection:
    connection_id: str
    source_id: str
    target_id: str
    kind: str = "directed"
    start_fraction: float = 0.2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BespokeMotion:
    motion_id: str
    primitive_id: str
    kind: str
    start_fraction: float
    end_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BespokeSceneProgram:
    version: str
    program_id: str
    blueprint_id: str
    scene_type: str
    primitives: list[BespokePrimitive]
    connections: list[BespokeConnection]
    motions: list[BespokeMotion]
    background_mode: str = "precision_grid"

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "primitives": [item.to_dict() for item in self.primitives],
            "connections": [item.to_dict() for item in self.connections],
            "motions": [item.to_dict() for item in self.motions],
        }


@dataclass(frozen=True)
class BespokeProgramValidation:
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    grounded_copy_ratio: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["grounded_copy_ratio"] = round(float(self.grounded_copy_ratio), 4)
        return payload


@dataclass(frozen=True)
class CompiledBespokeStage:
    html: str
    metadata: dict[str, Any]


def build_bespoke_program(
    ir: Any,
    *,
    blueprint_id: str,
    variant_index: int = 0,
) -> BespokeSceneProgram:
    payload = _payload(ir)
    objects = [dict(item) for item in payload.get("objects") or [] if isinstance(item, dict)][:8]
    if not objects:
        raise ValueError("Cannot author a bespoke scene without grounded IR objects.")
    scene_type = str(payload.get("scene_type") or "")
    positions = _positions(scene_type, len(objects), variant_index=variant_index)
    primitives = [
        BespokePrimitive(
            primitive_id=f"primitive_{index + 1:02d}",
            kind=_primitive_kind(str(item.get("role") or "")),
            object_id=str(item.get("object_id") or ""),
            role=str(item.get("role") or ""),
            text=str(item.get("label") or ""),
            x=positions[index][0],
            y=positions[index][1],
            width=positions[index][2],
            height=positions[index][3],
            emphasis=float(item.get("emphasis") or 0.5),
        )
        for index, item in enumerate(objects)
    ]
    connections = [
        BespokeConnection(
            connection_id=f"connection_{index + 1:02d}",
            source_id=primitives[index].primitive_id,
            target_id=primitives[index + 1].primitive_id,
            kind="directed",
            start_fraction=round(0.18 + index * 0.1, 3),
        )
        for index in range(max(len(primitives) - 1, 0))
    ]
    motions = [
        BespokeMotion(
            motion_id=f"motion_{index + 1:02d}",
            primitive_id=primitive.primitive_id,
            kind=_motion_kind(scene_type, primitive.role, index),
            start_fraction=round(0.08 + index * (0.68 / max(len(primitives), 1)), 3),
            end_fraction=round(
                min(0.96, 0.08 + index * (0.68 / max(len(primitives), 1)) + 0.34),
                3,
            ),
        )
        for index, primitive in enumerate(primitives)
    ]
    return BespokeSceneProgram(
        version=AUTHORING_VERSION,
        program_id=f"{_safe_id(payload.get('visual_id') or 'visual')}-bespoke-{variant_index + 1:02d}",
        blueprint_id=_safe_id(blueprint_id),
        scene_type=scene_type,
        primitives=primitives,
        connections=connections,
        motions=motions,
        background_mode="precision_grid",
    )


def validate_bespoke_program(
    program: BespokeSceneProgram | dict[str, Any],
    ir: Any,
) -> BespokeProgramValidation:
    payload = program.to_dict() if isinstance(program, BespokeSceneProgram) else dict(program or {})
    ir_payload = _payload(ir)
    errors: list[str] = []
    warnings: list[str] = []
    if str(payload.get("version") or "") != AUTHORING_VERSION:
        errors.append("unsupported_scene_program_version")
    primitives = [item for item in payload.get("primitives") or [] if isinstance(item, dict)]
    connections = [item for item in payload.get("connections") or [] if isinstance(item, dict)]
    motions = [item for item in payload.get("motions") or [] if isinstance(item, dict)]
    if not primitives:
        errors.append("scene_program_has_no_primitives")
    if len(primitives) > 12:
        errors.append("scene_program_has_too_many_primitives")
    if len(connections) > 16:
        errors.append("scene_program_has_too_many_connections")
    if len(motions) > 20:
        errors.append("scene_program_has_too_many_motions")
    ir_objects = {
        str(item.get("object_id") or ""): dict(item)
        for item in ir_payload.get("objects") or []
        if isinstance(item, dict) and str(item.get("object_id") or "")
    }
    required_object_ids = set(ir_objects)
    represented_object_ids: set[str] = set()
    primitive_ids: set[str] = set()
    grounded_copy = 0
    for item in primitives:
        primitive_id = str(item.get("primitive_id") or "")
        if not primitive_id or primitive_id in primitive_ids:
            errors.append(f"invalid_or_duplicate_primitive_id:{primitive_id or 'missing'}")
        primitive_ids.add(primitive_id)
        kind = str(item.get("kind") or "")
        if kind not in ALLOWED_PRIMITIVES:
            errors.append(f"unsupported_primitive_kind:{kind or 'missing'}")
        object_id = str(item.get("object_id") or "")
        if object_id not in ir_objects:
            errors.append(f"primitive_references_unknown_object:{primitive_id}")
            continue
        represented_object_ids.add(object_id)
        text = str(item.get("text") or "").strip()
        if _copy_is_grounded(text, ir_objects[object_id], ir_payload):
            grounded_copy += 1
        else:
            errors.append(f"primitive_copy_is_not_grounded:{primitive_id}")
        x = _float(item.get("x"), -1.0)
        y = _float(item.get("y"), -1.0)
        width = _float(item.get("width"), -1.0)
        height = _float(item.get("height"), -1.0)
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            errors.append(f"primitive_position_out_of_bounds:{primitive_id}")
        if not (0.08 <= width <= 0.62 and 0.06 <= height <= 0.5):
            errors.append(f"primitive_size_out_of_bounds:{primitive_id}")
        if x - width / 2 < 0.0 or x + width / 2 > 1.0:
            errors.append(f"primitive_exceeds_horizontal_bounds:{primitive_id}")
        if y - height / 2 < 0.0 or y + height / 2 > 1.0:
            errors.append(f"primitive_exceeds_vertical_bounds:{primitive_id}")
    missing_objects = sorted(required_object_ids - represented_object_ids)
    if missing_objects:
        errors.append("scene_program_omits_required_objects:" + ",".join(missing_objects))
    for item in connections:
        connection_id = str(item.get("connection_id") or "missing")
        if str(item.get("source_id") or "") not in primitive_ids:
            errors.append(f"connection_unknown_source:{connection_id}")
        if str(item.get("target_id") or "") not in primitive_ids:
            errors.append(f"connection_unknown_target:{connection_id}")
        start = _float(item.get("start_fraction"), -1.0)
        if not 0.0 <= start <= 1.0:
            errors.append(f"connection_invalid_timing:{connection_id}")
    motion_primitives: set[str] = set()
    for item in motions:
        motion_id = str(item.get("motion_id") or "missing")
        primitive_id = str(item.get("primitive_id") or "")
        if primitive_id not in primitive_ids:
            errors.append(f"motion_unknown_primitive:{motion_id}")
        else:
            motion_primitives.add(primitive_id)
        kind = str(item.get("kind") or "")
        if kind not in ALLOWED_MOTIONS:
            errors.append(f"unsupported_motion_kind:{kind or 'missing'}")
        start = _float(item.get("start_fraction"), -1.0)
        end = _float(item.get("end_fraction"), -1.0)
        if start < 0.0 or end > 1.0 or end <= start:
            errors.append(f"motion_invalid_timing:{motion_id}")
    missing_motion = sorted(primitive_ids - motion_primitives)
    if missing_motion:
        errors.append("scene_program_has_static_required_objects:" + ",".join(missing_motion))
    overlap_pairs = _high_overlap_pairs(primitives)
    if overlap_pairs:
        warnings.append("scene_program_has_high_overlap:" + ",".join(overlap_pairs[:4]))
    ratio = grounded_copy / max(len(primitives), 1)
    return BespokeProgramValidation(
        passed=not errors and ratio >= 0.95,
        errors=_unique(errors),
        warnings=_unique(warnings),
        grounded_copy_ratio=ratio,
    )


def compile_bespoke_stage(
    program: BespokeSceneProgram | dict[str, Any],
    ir: Any,
) -> CompiledBespokeStage:
    payload = program.to_dict() if isinstance(program, BespokeSceneProgram) else dict(program or {})
    validation = validate_bespoke_program(payload, ir)
    if not validation.passed:
        raise ValueError("Unsafe bespoke scene program: " + "; ".join(validation.errors))
    primitives = [dict(item) for item in payload.get("primitives") or []]
    primitive_by_id = {
        str(item.get("primitive_id") or ""): item
        for item in primitives
    }
    motions = {
        str(item.get("primitive_id") or ""): dict(item)
        for item in payload.get("motions") or []
        if isinstance(item, dict)
    }
    connections = [dict(item) for item in payload.get("connections") or []]
    primitive_html = "\n".join(
        _primitive_html(item, motions.get(str(item.get("primitive_id") or ""), {}))
        for item in primitives
    )
    connection_html = "\n".join(
        _connection_html(item, primitive_by_id)
        for item in connections
    )
    scoped_id = _safe_id(payload.get("program_id") or "bespoke")
    fragment = f"""
      <style>
        #{scoped_id} {{ position: absolute; inset: 0; overflow: hidden; }}
        #{scoped_id} .bespoke-connections {{ position: absolute; inset: 0; width: 100%; height: 100%; overflow: visible; }}
        #{scoped_id} .bespoke-connection {{ stroke: var(--accent-2); stroke-width: 0.45; vector-effect: non-scaling-stroke; opacity: .72; transform-origin: center; transform: scaleX(var(--line-progress, 0)); }}
        #{scoped_id} .bespoke-primitive {{ position: absolute; display: grid; align-content: center; padding: 18px; border: 1px solid color-mix(in srgb, var(--stroke) 56%, transparent); border-top: 4px solid var(--accent-2); background: color-mix(in srgb, var(--panel) 86%, transparent); box-shadow: 0 24px 58px color-mix(in srgb, black 28%, transparent); }}
        #{scoped_id} .bespoke-primitive.role-result, #{scoped_id} .bespoke-primitive.role-branch_high {{ border-top-color: var(--accent); background: color-mix(in srgb, var(--panel) 74%, var(--accent) 9%); }}
        #{scoped_id} .bespoke-primitive.role-intervention {{ border-top-color: var(--accent); }}
        #{scoped_id} .bespoke-primitive span {{ color: var(--accent-2); font-size: 14px; font-weight: 850; text-transform: uppercase; }}
        #{scoped_id} .bespoke-primitive b {{ margin-top: 9px; color: var(--text); font-size: 23px; line-height: 1.08; overflow-wrap: anywhere; }}
        #{scoped_id} .kind-metric b {{ font-size: 36px; }}
        #{scoped_id} .kind-quote {{ border: 0; border-left: 5px solid var(--accent); background: color-mix(in srgb, var(--panel) 68%, transparent); }}
      </style>
      <section id="{scoped_id}" class="bespoke-canvas" data-scene-type="{html.escape(str(payload.get("scene_type") or ""), quote=True)}">
        <svg class="bespoke-connections" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">{connection_html}</svg>
        {primitive_html}
      </section>
    """
    safety = validate_authored_html_safety(fragment)
    if not safety.safe:
        raise ValueError("Compiled bespoke HTML failed safety validation: " + "; ".join(safety.errors))
    return CompiledBespokeStage(
        html=fragment,
        metadata={
            "program_id": str(payload.get("program_id") or ""),
            "blueprint_id": str(payload.get("blueprint_id") or ""),
            "scene_type": str(payload.get("scene_type") or ""),
            "primitive_count": len(primitives),
            "connection_count": len(connections),
            "motion_count": len(motions),
            "grounded_copy_ratio": validation.grounded_copy_ratio,
            "safety": safety.to_dict(),
        },
    )


def _primitive_html(item: dict[str, Any], motion: dict[str, Any]) -> str:
    x = _float(item.get("x"), 0.5)
    y = _float(item.get("y"), 0.5)
    width = _float(item.get("width"), 0.2)
    height = _float(item.get("height"), 0.16)
    motion_kind = str(motion.get("kind") or "rise")
    anim = {
        "activate": "pop",
        "focus": "scale",
        "lock": "scale",
        "morph": "slide-right",
        "pop": "pop",
        "reveal": "rise",
        "rise": "rise",
        "trace": "slide-right",
        "travel": "slide-right",
    }.get(motion_kind, "rise")
    start = _float(motion.get("start_fraction"), 0.12)
    span = max(_float(motion.get("end_fraction"), 0.55) - start, 0.12)
    return (
        f'<article id="{html.escape(str(item.get("primitive_id") or ""), quote=True)}" '
        f'class="bespoke-primitive kind-{html.escape(str(item.get("kind") or ""), quote=True)} '
        f'role-{html.escape(str(item.get("role") or ""), quote=True)}" '
        f'data-object-id="{html.escape(str(item.get("object_id") or ""), quote=True)}" '
        f'data-anim="{anim}" data-delay="{start:.3f}" data-span="{span:.3f}" data-y="24" data-scale="0.930" '
        f'style="left:{(x - width / 2) * 100:.3f}%;top:{(y - height / 2) * 100:.3f}%;'
        f'width:{width * 100:.3f}%;height:{height * 100:.3f}%;">'
        f'<span>{html.escape(str(item.get("role") or "").replace("_", " "), quote=True)}</span>'
        f'<b>{html.escape(str(item.get("text") or ""), quote=True)}</b>'
        "</article>"
    )


def _connection_html(
    item: dict[str, Any],
    primitive_by_id: dict[str, dict[str, Any]],
) -> str:
    source = primitive_by_id[str(item.get("source_id") or "")]
    target = primitive_by_id[str(item.get("target_id") or "")]
    return (
        f'<line class="bespoke-connection" data-line '
        f'data-delay="{_float(item.get("start_fraction"), 0.2):.3f}" '
        f'x1="{_float(source.get("x"), 0.5) * 100:.3f}" '
        f'y1="{_float(source.get("y"), 0.5) * 100:.3f}" '
        f'x2="{_float(target.get("x"), 0.5) * 100:.3f}" '
        f'y2="{_float(target.get("y"), 0.5) * 100:.3f}" />'
    )


def _positions(
    scene_type: str,
    count: int,
    *,
    variant_index: int,
) -> list[tuple[float, float, float, float]]:
    if scene_type == "decision_branch" and count >= 3:
        base = [(0.5, 0.2, 0.26, 0.18), (0.27, 0.68, 0.32, 0.22), (0.73, 0.68, 0.32, 0.22)]
    elif scene_type == "matched_state_transform" and count >= 2:
        base = [(0.25, 0.48, 0.34, 0.32), (0.75, 0.48, 0.34, 0.32)]
    elif scene_type == "grounded_interface_walkthrough":
        base = [
            (0.5, 0.18 + index * (0.66 / max(count - 1, 1)), 0.62, 0.12)
            for index in range(count)
        ]
    elif scene_type == "evidence_backed_quote":
        base = [(0.5, 0.48, 0.72, 0.34)]
    else:
        base = [
            (
                0.12 + index * (0.76 / max(count - 1, 1)),
                0.5 + (0.12 if (index + variant_index) % 2 else -0.1),
                min(0.22, 0.78 / max(count, 1)),
                0.22,
            )
            for index in range(count)
        ]
    if len(base) < count:
        base.extend(
            (
                0.5,
                0.82,
                0.28,
                0.12,
            )
            for _ in range(count - len(base))
        )
    return base[:count]


def _primitive_kind(role: str) -> str:
    return {
        "branch_high": "decision",
        "branch_low": "decision",
        "constraint": "highlight",
        "decision": "decision",
        "interface": "interface_state",
        "intervention": "token",
        "metric": "metric",
        "problem": "state",
        "quote": "quote",
        "result": "state",
    }.get(role, "node")


def _motion_kind(scene_type: str, role: str, index: int) -> str:
    if role == "intervention":
        return "activate"
    if role in {"result", "branch_high"}:
        return "lock"
    if scene_type in {"architecture_flow", "guided_process"}:
        return "travel" if index else "reveal"
    if scene_type == "matched_state_transform" and index:
        return "morph"
    if scene_type == "grounded_interface_walkthrough":
        return "focus"
    return "rise" if index else "reveal"


def _copy_is_grounded(text: str, obj: dict[str, Any], ir: dict[str, Any]) -> bool:
    normalized = _normalize(text)
    if not normalized:
        return False
    copy_contract = dict((ir.get("metadata") or {}).get("visual_copy_contract") or {})
    if copy_contract:
        return copy_allowed_for_binding(
            text,
            copy_contract,
            binding_kind="object",
            binding_id=str(obj.get("object_id") or ""),
        )
    allowed = {_normalize(obj.get("label")), _normalize(obj.get("meaning"))}
    return normalized in allowed


def _high_overlap_pairs(primitives: list[dict[str, Any]]) -> list[str]:
    pairs: list[str] = []
    for index, first in enumerate(primitives):
        for second in primitives[index + 1 :]:
            if _intersection_over_union(first, second) > 0.45:
                pairs.append(
                    f"{first.get('primitive_id')}+{second.get('primitive_id')}"
                )
    return pairs


def _intersection_over_union(first: dict[str, Any], second: dict[str, Any]) -> float:
    ax1 = _float(first.get("x"), 0.0) - _float(first.get("width"), 0.0) / 2
    ay1 = _float(first.get("y"), 0.0) - _float(first.get("height"), 0.0) / 2
    ax2 = ax1 + _float(first.get("width"), 0.0)
    ay2 = ay1 + _float(first.get("height"), 0.0)
    bx1 = _float(second.get("x"), 0.0) - _float(second.get("width"), 0.0) / 2
    by1 = _float(second.get("y"), 0.0) - _float(second.get("height"), 0.0) / 2
    bx2 = bx1 + _float(second.get("width"), 0.0)
    by2 = by1 + _float(second.get("height"), 0.0)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0,
        min(ay2, by2) - max(ay1, by1),
    )
    union = max((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection, 1e-9)
    return intersection / union


def _payload(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        return dict(value.to_dict())
    return dict(value or {})


def _safe_id(value: Any) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", str(value or "scene")).strip("-_").lower() or "scene"


def _normalize(value: Any) -> str:
    return re.sub(r"[^a-z0-9%+./-]+", " ", str(value or "").lower()).strip()


def _float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


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
    "ALLOWED_MOTIONS",
    "ALLOWED_PRIMITIVES",
    "AUTHORING_VERSION",
    "BespokeConnection",
    "BespokeMotion",
    "BespokePrimitive",
    "BespokeProgramValidation",
    "BespokeSceneProgram",
    "CompiledBespokeStage",
    "build_bespoke_program",
    "compile_bespoke_stage",
    "validate_bespoke_program",
]
