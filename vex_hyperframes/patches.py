from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

from vex_hyperframes.counterexamples import (
    ALLOWED_REPAIR_OPERATIONS,
    VisualCounterexample,
)
from vex_hyperframes.scene_program import (
    ALLOWED_LAYOUT_FAMILIES,
    SceneProgramValidation,
    validate_scene_program,
)


PATCH_VERSION = "hyperframes-visual-patch-v1"


@dataclass(frozen=True)
class VisualPatchOperation:
    operation_id: str
    operation: str
    target_ids: list[str]
    parameters: dict[str, Any]
    counterexample_ids: list[str]
    evidence_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VisualPatchSet:
    version: str
    patch_id: str
    base_program_id: str
    base_program_signature: str
    operations: list[VisualPatchOperation]
    patch_signature: str

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "operations": [item.to_dict() for item in self.operations],
        }


@dataclass(frozen=True)
class VisualPatchValidation:
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PatchApplication:
    passed: bool
    scene_program: dict[str, Any]
    validation: SceneProgramValidation
    patch_validation: VisualPatchValidation
    applied_operation_ids: list[str]
    spec_updates: dict[str, Any] = field(default_factory=dict)
    disposition: str = "rerender"

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "scene_program": dict(self.scene_program),
            "validation": self.validation.to_dict(),
            "patch_validation": self.patch_validation.to_dict(),
            "applied_operation_ids": list(self.applied_operation_ids),
            "spec_updates": dict(self.spec_updates),
            "disposition": self.disposition,
        }


def plan_visual_patches(
    counterexamples: list[VisualCounterexample],
    *,
    scene_program: dict[str, Any],
    round_index: int,
    max_operations: int = 8,
) -> VisualPatchSet:
    operations: list[VisualPatchOperation] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for counterexample in counterexamples:
        operation, targets, parameters = _operation_for_counterexample(
            counterexample,
            scene_program=scene_program,
            round_index=round_index,
        )
        if not operation:
            continue
        key = (operation, tuple(targets))
        if key in seen:
            continue
        seen.add(key)
        operations.append(
            VisualPatchOperation(
                operation_id=f"patch_op_{len(operations) + 1:02d}",
                operation=operation,
                target_ids=targets,
                parameters=parameters,
                counterexample_ids=[counterexample.counterexample_id],
                evidence_ids=list(counterexample.evidence_ids),
            )
        )
        if len(operations) >= max(1, min(int(max_operations), 12)):
            break
    base_payload = {
        "version": PATCH_VERSION,
        "patch_id": (
            f"{scene_program.get('program_id')}-repair-{round_index:02d}"
        ),
        "base_program_id": str(scene_program.get("program_id") or ""),
        "base_program_signature": str(
            scene_program.get("program_signature") or ""
        ),
        "operations": [item.to_dict() for item in operations],
    }
    return VisualPatchSet(
        version=PATCH_VERSION,
        patch_id=str(base_payload["patch_id"]),
        base_program_id=str(base_payload["base_program_id"]),
        base_program_signature=str(base_payload["base_program_signature"]),
        operations=operations,
        patch_signature=_signature(base_payload),
    )


def validate_patch_set(
    patch_set: VisualPatchSet | dict[str, Any],
    *,
    scene_program: dict[str, Any],
) -> VisualPatchValidation:
    payload = (
        patch_set.to_dict()
        if isinstance(patch_set, VisualPatchSet)
        else dict(patch_set or {})
    )
    errors: list[str] = []
    warnings: list[str] = []
    if payload.get("version") != PATCH_VERSION:
        errors.append("unsupported_visual_patch_version")
    if payload.get("base_program_id") != scene_program.get("program_id"):
        errors.append("visual_patch_program_id_mismatch")
    if scene_program.get("program_signature") != _program_signature(
        scene_program
    ):
        errors.append("scene_program_signature_invalid_before_patch")
    if (
        payload.get("base_program_signature")
        != scene_program.get("program_signature")
    ):
        errors.append("visual_patch_base_signature_mismatch")
    expected_signature = _signature(
        {key: value for key, value in payload.items() if key != "patch_signature"}
    )
    if payload.get("patch_signature") != expected_signature:
        errors.append("visual_patch_signature_mismatch")
    elements = {
        str(item.get("element_id") or "")
        for item in scene_program.get("elements") or []
        if isinstance(item, dict)
    }
    relations = {
        str(item.get("relation_id") or "")
        for item in scene_program.get("relations") or []
        if isinstance(item, dict)
    }
    operations = [
        dict(item)
        for item in payload.get("operations") or []
        if isinstance(item, dict)
    ]
    if not operations:
        warnings.append("visual_patch_set_has_no_operations")
    operation_ids: set[str] = set()
    for item in operations:
        operation_id = str(item.get("operation_id") or "")
        operation = str(item.get("operation") or "")
        targets = [
            str(value) for value in item.get("target_ids") or []
        ]
        if not operation_id or operation_id in operation_ids:
            errors.append(
                f"invalid_or_duplicate_patch_operation_id:{operation_id or 'missing'}"
            )
        operation_ids.add(operation_id)
        if operation not in ALLOWED_REPAIR_OPERATIONS:
            errors.append(f"unsupported_patch_operation:{operation_id}")
            continue
        _validate_operation_targets(
            operation,
            targets,
            elements=elements,
            relations=relations,
            operation_id=operation_id,
            errors=errors,
        )
        _validate_operation_parameters(
            operation,
            dict(item.get("parameters") or {}),
            operation_id=operation_id,
            errors=errors,
        )
    return VisualPatchValidation(
        passed=not errors,
        errors=_unique(errors),
        warnings=_unique(warnings),
    )


def apply_visual_patch_set(
    patch_set: VisualPatchSet | dict[str, Any],
    *,
    scene_program: dict[str, Any],
    ir: dict[str, Any],
    claim_graph: dict[str, Any],
) -> PatchApplication:
    payload = (
        patch_set.to_dict()
        if isinstance(patch_set, VisualPatchSet)
        else dict(patch_set or {})
    )
    patch_validation = validate_patch_set(
        payload,
        scene_program=scene_program,
    )
    if not patch_validation.passed:
        validation = validate_scene_program(
            scene_program,
            ir=ir,
            claim_graph=claim_graph,
        )
        return PatchApplication(
            passed=False,
            scene_program=dict(scene_program),
            validation=validation,
            patch_validation=patch_validation,
            applied_operation_ids=[],
            disposition="reject_patch",
        )
    working = json.loads(json.dumps(scene_program))
    applied: list[str] = []
    spec_updates: dict[str, Any] = {}
    disposition = "rerender"
    for item in payload.get("operations") or []:
        operation = str(item.get("operation") or "")
        targets = [str(value) for value in item.get("target_ids") or []]
        parameters = dict(item.get("parameters") or {})
        _apply_operation(
            working,
            operation=operation,
            targets=targets,
            parameters=parameters,
            spec_updates=spec_updates,
        )
        if operation == "reroute_renderer":
            disposition = "reroute"
        applied.append(str(item.get("operation_id") or ""))
    working["program_signature"] = _program_signature(working)
    validation = validate_scene_program(
        working,
        ir=ir,
        claim_graph=claim_graph,
    )
    evidence_errors = _evidence_preservation_errors(
        before=scene_program,
        after=working,
        claim_graph=claim_graph,
    )
    if evidence_errors:
        patch_validation = VisualPatchValidation(
            passed=False,
            errors=_unique([*patch_validation.errors, *evidence_errors]),
            warnings=patch_validation.warnings,
        )
    passed = validation.passed and patch_validation.passed
    return PatchApplication(
        passed=passed,
        scene_program=working if passed else dict(scene_program),
        validation=validation,
        patch_validation=patch_validation,
        applied_operation_ids=applied if passed else [],
        spec_updates=spec_updates if passed else {},
        disposition=disposition if passed else "reject_patch",
    )


def _operation_for_counterexample(
    counterexample: VisualCounterexample,
    *,
    scene_program: dict[str, Any],
    round_index: int,
) -> tuple[str, list[str], dict[str, Any]]:
    allowed = set(counterexample.allowed_repairs)
    elements = [
        dict(item)
        for item in scene_program.get("elements") or []
        if isinstance(item, dict)
    ]
    relations = [
        dict(item)
        for item in scene_program.get("relations") or []
        if isinstance(item, dict)
    ]
    if counterexample.issue_type in {"missing_relation", "weak_relation_encoding"}:
        targets = counterexample.relation_ids or [
            str(item.get("relation_id") or "") for item in relations
        ][:2]
        if "strengthen_relation" in allowed and targets:
            return (
                "strengthen_relation",
                targets,
                {"strength": 1.0, "reveal_shift": -0.08},
            )
    if counterexample.issue_type == "missing_object":
        targets = counterexample.element_ids
        if "persist_element" in allowed and targets:
            return (
                "persist_element",
                targets,
                {"visible_start": 0.04, "visible_end": 1.0},
            )
    if counterexample.issue_type in {"motion", "pacing"}:
        targets = counterexample.element_ids or [
            str(item.get("element_id") or "") for item in elements
        ]
        if "retime_reveal" in allowed and targets:
            return (
                "retime_reveal",
                targets,
                {"shift": -0.045 * max(1, round_index)},
            )
    if counterexample.issue_type == "source_asset_required":
        if "reroute_renderer" in allowed:
            return (
                "reroute_renderer",
                [],
                {"renderer": "ffmpeg_asset"},
            )
    if counterexample.issue_type == "unsupported_content":
        if "remove_unsupported_content" in allowed:
            return (
                "remove_unsupported_content",
                counterexample.element_ids,
                {},
            )
    if counterexample.issue_type in {"density", "overlap", "overflow"}:
        if "change_layout_family" in allowed:
            return (
                "change_layout_family",
                [],
                {"layout_family": _next_layout(scene_program)},
            )
        if "resize_element" in allowed and counterexample.element_ids:
            return (
                "resize_element",
                counterexample.element_ids,
                {"scale": 0.88},
            )
    if counterexample.issue_type in {"hierarchy", "ambiguous_thesis"}:
        target = _focus_element(elements)
        if "strengthen_hierarchy" in allowed and target:
            return (
                "strengthen_hierarchy",
                [target],
                {"emphasis": 1.0, "scale": 1.12},
            )
        if "swap_proof_encoding" in allowed:
            return (
                "swap_proof_encoding",
                [],
                {"layout_family": _next_layout(scene_program)},
            )
    for fallback in (
        "change_layout_family",
        "swap_proof_encoding",
        "reroute_renderer",
    ):
        if fallback in allowed:
            parameters = (
                {"renderer": "ffmpeg_asset"}
                if fallback == "reroute_renderer"
                else {"layout_family": _next_layout(scene_program)}
            )
            return fallback, [], parameters
    return "", [], {}


def _apply_operation(
    program: dict[str, Any],
    *,
    operation: str,
    targets: list[str],
    parameters: dict[str, Any],
    spec_updates: dict[str, Any],
) -> None:
    elements = [
        item
        for item in program.get("elements") or []
        if isinstance(item, dict)
    ]
    relations = [
        item
        for item in program.get("relations") or []
        if isinstance(item, dict)
    ]
    element_by_id = {
        str(item.get("element_id") or ""): item for item in elements
    }
    relation_by_id = {
        str(item.get("relation_id") or ""): item for item in relations
    }
    if operation == "move_element":
        for target in targets:
            item = element_by_id[target]
            item["x"] = _clamp(
                float(item.get("x") or 0.5)
                + float(parameters.get("dx") or 0.0),
                float(item.get("width") or 0.2) / 2,
                1.0 - float(item.get("width") or 0.2) / 2,
            )
            item["y"] = _clamp(
                float(item.get("y") or 0.5)
                + float(parameters.get("dy") or 0.0),
                float(item.get("height") or 0.16) / 2,
                1.0 - float(item.get("height") or 0.16) / 2,
            )
    elif operation == "resize_element":
        scale = float(parameters.get("scale") or 1.0)
        for target in targets:
            item = element_by_id[target]
            item["width"] = _clamp(
                float(item.get("width") or 0.2) * scale,
                0.1,
                0.55,
            )
            item["height"] = _clamp(
                float(item.get("height") or 0.16) * scale,
                0.08,
                0.4,
            )
    elif operation in {"change_layout_family", "swap_proof_encoding", "reduce_density"}:
        family = str(parameters.get("layout_family") or _next_layout(program))
        _apply_layout(program, family)
        if operation == "swap_proof_encoding":
            spec_updates["proof_encoding"] = family
    elif operation == "strengthen_relation":
        strength = _clamp(float(parameters.get("strength") or 1.0), 0.2, 1.0)
        shift = float(parameters.get("reveal_shift") or 0.0)
        for target in targets:
            item = relation_by_id[target]
            item["strength"] = strength
            item["reveal_fraction"] = _clamp(
                float(item.get("reveal_fraction") or 0.5) + shift,
                0.04,
                0.94,
            )
    elif operation == "persist_element":
        for target in targets:
            item = element_by_id[target]
            item["visible_start"] = _clamp(
                float(parameters.get("visible_start") or 0.04),
                0.0,
                0.9,
            )
            item["visible_end"] = _clamp(
                float(parameters.get("visible_end") or 1.0),
                item["visible_start"] + 0.05,
                1.0,
            )
    elif operation == "retime_reveal":
        shift = float(parameters.get("shift") or 0.0)
        for motion in program.get("motions") or []:
            if (
                isinstance(motion, dict)
                and str(motion.get("target_id") or "") in targets
            ):
                span = max(
                    0.12,
                    float(motion.get("end_fraction") or 0.5)
                    - float(motion.get("start_fraction") or 0.1),
                )
                start = _clamp(
                    float(motion.get("start_fraction") or 0.1) + shift,
                    0.02,
                    0.82,
                )
                motion["start_fraction"] = start
                motion["end_fraction"] = min(0.96, start + span)
    elif operation == "strengthen_hierarchy":
        scale = float(parameters.get("scale") or 1.1)
        emphasis = _clamp(
            float(parameters.get("emphasis") or 1.0),
            0.0,
            1.0,
        )
        for target in targets:
            item = element_by_id[target]
            item["emphasis"] = emphasis
            item["width"] = _clamp(
                float(item.get("width") or 0.2) * scale,
                0.1,
                0.55,
            )
            item["height"] = _clamp(
                float(item.get("height") or 0.16) * scale,
                0.08,
                0.4,
            )
    elif operation == "remove_unsupported_content":
        target_set = set(targets)
        program["elements"] = [
            item
            for item in elements
            if str(item.get("element_id") or "") not in target_set
        ]
        program["motions"] = [
            item
            for item in program.get("motions") or []
            if not isinstance(item, dict)
            or str(item.get("target_id") or "") not in target_set
        ]
    elif operation == "bind_source_asset":
        spec_updates["source_asset_grounding"] = {
            "asset_path": str(parameters.get("asset_path") or ""),
            "validated": True,
        }
    elif operation == "reroute_renderer":
        spec_updates["reroute_renderer"] = str(
            parameters.get("renderer") or "ffmpeg_asset"
        )


def _apply_layout(program: dict[str, Any], family: str) -> None:
    if family not in ALLOWED_LAYOUT_FAMILIES:
        return
    elements = [
        item
        for item in program.get("elements") or []
        if isinstance(item, dict)
    ]
    count = len(elements)
    positions: list[tuple[float, float, float, float]]
    if family == "linear_trace":
        positions = _linear_positions(count)
    elif family == "split_register":
        positions = _split_positions(count)
    elif family == "layered_flow":
        positions = _layered_positions(count)
    elif family == "focal_gate":
        focus = _focus_element(elements)
        positions = _focal_positions(elements, focus)
    else:
        focus = _focus_element(elements)
        positions = _radial_positions(elements, focus)
    for item, position in zip(elements, positions):
        item["x"], item["y"], item["width"], item["height"] = position
    program["layout_family"] = family


def _validate_operation_targets(
    operation: str,
    targets: list[str],
    *,
    elements: set[str],
    relations: set[str],
    operation_id: str,
    errors: list[str],
) -> None:
    element_operations = {
        "move_element",
        "persist_element",
        "remove_unsupported_content",
        "resize_element",
        "retime_reveal",
        "strengthen_hierarchy",
    }
    relation_operations = {"strengthen_relation"}
    no_target_operations = {
        "bind_source_asset",
        "change_layout_family",
        "reduce_density",
        "reroute_renderer",
        "swap_proof_encoding",
    }
    if operation in element_operations:
        if not targets:
            errors.append(f"patch_operation_requires_element_target:{operation_id}")
        if any(target not in elements for target in targets):
            errors.append(f"patch_operation_unknown_element_target:{operation_id}")
    elif operation in relation_operations:
        if not targets:
            errors.append(f"patch_operation_requires_relation_target:{operation_id}")
        if any(target not in relations for target in targets):
            errors.append(f"patch_operation_unknown_relation_target:{operation_id}")
    elif operation in no_target_operations and targets:
        errors.append(f"patch_operation_must_not_have_targets:{operation_id}")


def _validate_operation_parameters(
    operation: str,
    parameters: dict[str, Any],
    *,
    operation_id: str,
    errors: list[str],
) -> None:
    if operation in {"change_layout_family", "reduce_density", "swap_proof_encoding"}:
        if parameters.get("layout_family") not in ALLOWED_LAYOUT_FAMILIES:
            errors.append(f"patch_operation_invalid_layout:{operation_id}")
    if operation in {"resize_element", "strengthen_hierarchy"}:
        scale = _number(parameters.get("scale"), 1.0)
        if not 0.72 <= scale <= 1.25:
            errors.append(f"patch_operation_invalid_scale:{operation_id}")
    if operation == "move_element":
        for key in ("dx", "dy"):
            value = _number(parameters.get(key), 0.0)
            if abs(value) > 0.25:
                errors.append(f"patch_operation_move_too_large:{operation_id}")
    if operation == "bind_source_asset":
        if not str(parameters.get("asset_path") or "").strip():
            errors.append(f"patch_operation_missing_asset_path:{operation_id}")
    if operation == "reroute_renderer":
        if parameters.get("renderer") not in {
            "ffmpeg_asset",
            "manim",
            "blender",
            "none",
        }:
            errors.append(f"patch_operation_invalid_renderer:{operation_id}")


def _evidence_preservation_errors(
    *,
    before: dict[str, Any],
    after: dict[str, Any],
    claim_graph: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if before.get("graph_signature") != after.get("graph_signature"):
        errors.append("patch_changed_graph_signature")
    if before.get("semantic_signature") != after.get("semantic_signature"):
        errors.append("patch_changed_semantic_signature")
    before_elements = {
        str(item.get("element_id") or ""): dict(item)
        for item in before.get("elements") or []
        if isinstance(item, dict)
    }
    after_elements = {
        str(item.get("element_id") or ""): dict(item)
        for item in after.get("elements") or []
        if isinstance(item, dict)
    }
    required_objects = {
        str(item.get("node_id") or "")
        for item in claim_graph.get("nodes") or []
        if isinstance(item, dict)
    }
    for element_id, item in before_elements.items():
        if str(item.get("object_id") or "") not in required_objects:
            continue
        candidate = after_elements.get(element_id)
        if candidate is None:
            errors.append(f"patch_removed_required_element:{element_id}")
            continue
        if candidate.get("text") != item.get("text"):
            errors.append(f"patch_changed_grounded_copy:{element_id}")
        if not set(item.get("evidence_ids") or []).issubset(
            set(candidate.get("evidence_ids") or [])
        ):
            errors.append(f"patch_removed_element_evidence:{element_id}")
    before_relations = {
        str(item.get("relation_id") or ""): dict(item)
        for item in before.get("relations") or []
        if isinstance(item, dict)
    }
    after_relations = {
        str(item.get("relation_id") or ""): dict(item)
        for item in after.get("relations") or []
        if isinstance(item, dict)
    }
    for relation_id, item in before_relations.items():
        candidate = after_relations.get(relation_id)
        if candidate is None:
            errors.append(f"patch_removed_required_relation:{relation_id}")
            continue
        for key in (
            "source_element_id",
            "target_element_id",
            "relation_type",
        ):
            if candidate.get(key) != item.get(key):
                errors.append(f"patch_changed_relation_contract:{relation_id}")
                break
        if not set(item.get("evidence_ids") or []).issubset(
            set(candidate.get("evidence_ids") or [])
        ):
            errors.append(f"patch_removed_relation_evidence:{relation_id}")
    return _unique(errors)


def _next_layout(program: dict[str, Any]) -> str:
    order = [
        "linear_trace",
        "focal_gate",
        "split_register",
        "layered_flow",
        "radial_evidence",
    ]
    current = str(program.get("layout_family") or "")
    try:
        index = order.index(current)
    except ValueError:
        index = 0
    return order[(index + 1) % len(order)]


def _focus_element(elements: list[dict[str, Any]]) -> str:
    if not elements:
        return ""
    item = max(
        elements,
        key=lambda value: (
            1
            if str(value.get("role") or "")
            in {"decision", "intervention", "metric", "result", "branch_high"}
            else 0,
            float(value.get("emphasis") or 0.5),
            int(value.get("z_index") or 0),
        ),
    )
    return str(item.get("element_id") or "")


def _linear_positions(count: int) -> list[tuple[float, float, float, float]]:
    if count <= 4:
        width = min(0.22, 0.82 / max(count, 1))
        return [
            (
                0.10
                + width / 2
                + index * ((0.80 - width) / max(count - 1, 1)),
                0.5,
                width,
                0.24,
            )
            for index in range(count)
        ]
    return [
        (
            0.2 + (index % 3) * 0.3,
            0.29 + (index // 3) * 0.34,
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
        positions.append(
            (
                0.27,
                (index + 1) / (left_count + 1),
                0.36,
                min(0.24, 0.68 / left_count),
            )
        )
    for index in range(right_count):
        positions.append(
            (
                0.73,
                (index + 1) / (right_count + 1),
                0.36,
                min(0.24, 0.68 / max(right_count, 1)),
            )
        )
    return positions


def _layered_positions(count: int) -> list[tuple[float, float, float, float]]:
    columns = 2 if count > 3 else 1
    rows = math.ceil(count / columns)
    positions: list[tuple[float, float, float, float]] = []
    for index in range(count):
        row = index // columns
        column = index % columns
        x = (
            0.5
            if columns == 1 or (row == rows - 1 and count % 2 == 1)
            else 0.29 + column * 0.42
        )
        y = 0.18 + row * (0.64 / max(rows - 1, 1))
        positions.append(
            (
                x,
                y,
                0.34 if columns == 2 else 0.46,
                min(0.2, 0.62 / rows),
            )
        )
    return positions


def _focal_positions(
    elements: list[dict[str, Any]],
    focus_id: str,
) -> list[tuple[float, float, float, float]]:
    anchors = [
        (0.19, 0.24),
        (0.81, 0.24),
        (0.19, 0.76),
        (0.81, 0.76),
        (0.5, 0.16),
        (0.5, 0.84),
        (0.14, 0.5),
    ]
    positions: list[tuple[float, float, float, float]] = []
    satellite_index = 0
    for item in elements:
        if str(item.get("element_id") or "") == focus_id:
            positions.append((0.5, 0.5, 0.34, 0.28))
        else:
            x, y = anchors[satellite_index]
            satellite_index += 1
            positions.append((x, y, 0.25, 0.18))
    return positions


def _radial_positions(
    elements: list[dict[str, Any]],
    focus_id: str,
) -> list[tuple[float, float, float, float]]:
    satellites = max(len(elements) - 1, 1)
    positions: list[tuple[float, float, float, float]] = []
    satellite_index = 0
    for item in elements:
        if str(item.get("element_id") or "") == focus_id:
            positions.append((0.5, 0.5, 0.32, 0.25))
        else:
            angle = (-math.pi / 2) + satellite_index * (
                2 * math.pi / satellites
            )
            satellite_index += 1
            positions.append(
                (
                    0.5 + math.cos(angle) * 0.34,
                    0.5 + math.sin(angle) * 0.32,
                    0.23,
                    0.17,
                )
            )
    return positions


def _program_signature(program: dict[str, Any]) -> str:
    return _signature(
        {
            key: value
            for key, value in program.items()
            if key != "program_signature"
        }
    )


def _number(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


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
    "PATCH_VERSION",
    "PatchApplication",
    "VisualPatchOperation",
    "VisualPatchSet",
    "VisualPatchValidation",
    "apply_visual_patch_set",
    "plan_visual_patches",
    "validate_patch_set",
]
