from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from importlib.resources import files
from typing import Any, Iterable

from jsonschema import Draft202012Validator


OPEN_VISUAL_PROGRAM_VERSION = "vex-open-visual-program-v1"
OPEN_VISUAL_TOURNAMENT_VERSION = "vex-open-visual-tournament-v1"
OPEN_VISUAL_PATCH_VERSION = "vex-open-visual-patch-v1"

ALLOWED_ELEMENT_TYPES = {
    "chart",
    "connector",
    "group",
    "icon",
    "image",
    "mask",
    "metric",
    "particle",
    "path",
    "shape",
    "text",
    "token",
}
ALLOWED_BINDING_KINDS = {"fact", "object", "relation"}
ALLOWED_MOTION_PROPERTIES = {
    "blur",
    "emphasis",
    "opacity",
    "progress",
    "rotation",
    "scale",
    "translate_x",
    "translate_y",
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
}
ALLOWED_PATCH_OPERATIONS = {
    "move",
    "remove_decorative",
    "replace_text",
    "resize",
    "set_concept",
    "set_motion",
    "set_style",
}

MAX_ELEMENTS = 48
MAX_RELATIONS = 32
MAX_TRACKS = 96
MAX_KEYFRAMES_PER_TRACK = 8
MAX_CONSTRAINTS = 32
MAX_REPEAT_COUNT = 24
MAX_TOTAL_TEXT_CHARS = 1400
MAX_RESOURCE_COST = 180.0

_WORD_RE = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)?", re.IGNORECASE)
_NUMBER_WORD_TOKENS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
}


@dataclass(frozen=True)
class OpenVisualProgramValidation:
    passed: bool
    score: float
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    object_coverage: float = 0.0
    relation_coverage: float = 0.0
    grounded_text_ratio: float = 0.0
    motion_coverage: float = 0.0
    semantic_fitness: float = 0.0
    novelty_score: float = 1.0
    resource_cost: float = 0.0
    fingerprint: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "score",
            "object_coverage",
            "relation_coverage",
            "grounded_text_ratio",
            "motion_coverage",
            "semantic_fitness",
            "novelty_score",
            "resource_cost",
        ):
            payload[key] = round(float(payload[key]), 4)
        return payload


@dataclass(frozen=True)
class OpenVisualTournament:
    version: str
    tournament_id: str
    tournament_signature: str
    selected_program_id: str
    candidates: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OpenVisualPatchResult:
    passed: bool
    program: dict[str, Any]
    applied_operations: list[dict[str, Any]] = field(default_factory=list)
    rejected_operations: list[dict[str, Any]] = field(default_factory=list)
    validation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def open_visual_program_signature(program: dict[str, Any]) -> str:
    payload = copy.deepcopy(dict(program or {}))
    payload.pop("signature", None)
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def sign_open_visual_program(program: dict[str, Any]) -> dict[str, Any]:
    signed = copy.deepcopy(dict(program or {}))
    signed["signature"] = open_visual_program_signature(signed)
    return signed


def open_visual_program_fingerprint(program: dict[str, Any]) -> dict[str, Any]:
    elements = [item for item in program.get("elements") or [] if isinstance(item, dict)]
    tracks = [item for item in program.get("tracks") or [] if isinstance(item, dict)]
    concept = dict(program.get("concept") or {})
    type_counts: dict[str, int] = {}
    for item in elements:
        key = str(item.get("type") or "unknown")
        type_counts[key] = type_counts.get(key, 0) + 1
    layout_cells = sorted(
        (
            round(_number((item.get("layout") or {}).get("x"), 0.0), 1),
            round(_number((item.get("layout") or {}).get("y"), 0.0), 1),
        )
        for item in elements
        if not bool(item.get("decorative"))
    )
    payload = {
        "medium": str(concept.get("medium") or ""),
        "metaphor": str(concept.get("metaphor") or ""),
        "composition": str(concept.get("composition") or ""),
        "type_counts": type_counts,
        "layout_cells": layout_cells,
        "motion_properties": sorted(
            {str(item.get("property") or "") for item in tracks}
        ),
    }
    payload["signature"] = hashlib.sha256(
        canonical_json(payload).encode("utf-8")
    ).hexdigest()
    return payload


def validate_open_visual_program(
    program: dict[str, Any],
    *,
    ir: dict[str, Any],
    history: Iterable[dict[str, Any]] | None = None,
    require_signature: bool = True,
) -> OpenVisualProgramValidation:
    payload = dict(program or {})
    evidence = dict(ir or {})
    errors: list[str] = []
    warnings: list[str] = []

    errors.extend(_schema_errors(payload))

    if str(payload.get("version") or "") != OPEN_VISUAL_PROGRAM_VERSION:
        errors.append("unsupported_open_visual_program_version")
    if not str(payload.get("program_id") or "").strip():
        errors.append("open_visual_program_has_no_id")
    expected_evidence_signature = _ir_signature(evidence)
    if str(payload.get("evidence_signature") or "") != expected_evidence_signature:
        errors.append("open_visual_program_evidence_signature_mismatch")
    signature = str(payload.get("signature") or "")
    if require_signature and not signature:
        errors.append("open_visual_program_unsigned")
    elif signature and signature != open_visual_program_signature(payload):
        errors.append("open_visual_program_signature_mismatch")

    canvas = dict(payload.get("canvas") or {})
    safe_area = dict(canvas.get("safe_area") or {})
    for key in ("top", "right", "bottom", "left"):
        value = _number(safe_area.get(key), 0.04)
        if not 0.0 <= value <= 0.2:
            errors.append(f"invalid_safe_area:{key}")

    elements = [item for item in payload.get("elements") or [] if isinstance(item, dict)]
    relations = [item for item in payload.get("relations") or [] if isinstance(item, dict)]
    tracks = [item for item in payload.get("tracks") or [] if isinstance(item, dict)]
    constraints = [item for item in payload.get("constraints") or [] if isinstance(item, dict)]
    if not elements:
        errors.append("open_visual_program_has_no_elements")
    if len(elements) > MAX_ELEMENTS:
        errors.append("open_visual_program_element_budget_exceeded")
    if len(relations) > MAX_RELATIONS:
        errors.append("open_visual_program_relation_budget_exceeded")
    if len(tracks) > MAX_TRACKS:
        errors.append("open_visual_program_motion_budget_exceeded")
    if len(constraints) > MAX_CONSTRAINTS:
        errors.append("open_visual_program_constraint_budget_exceeded")

    known = {
        "fact": _ids(evidence.get("facts"), "fact_id"),
        "object": _ids(evidence.get("objects"), "object_id"),
        "relation": _ids(evidence.get("relations"), "relation_id"),
    }
    element_ids: set[str] = set()
    represented = {"fact": set(), "object": set(), "relation": set()}
    grounded_text_count = 0
    text_count = 0
    text_chars = 0
    source_text = _source_text(evidence)
    required_labels = [str(item) for item in evidence.get("required_labels") or [] if str(item)]
    for index, item in enumerate(elements):
        element_id = str(item.get("element_id") or "")
        if not element_id or element_id in element_ids:
            errors.append(f"invalid_or_duplicate_element_id:{element_id or index}")
        element_ids.add(element_id)
        element_type = str(item.get("type") or "")
        if element_type not in ALLOWED_ELEMENT_TYPES:
            errors.append(f"unsupported_element_type:{element_id}:{element_type}")
        repeat = int(_number(item.get("repeat"), 1.0))
        if repeat < 1 or repeat > MAX_REPEAT_COUNT:
            errors.append(f"invalid_element_repeat:{element_id}")
        layout = dict(item.get("layout") or {})
        if not _valid_layout(layout):
            errors.append(f"invalid_element_layout:{element_id}")
        style = dict(item.get("style") or {})
        if _number(style.get("blur"), 0.0) > 32.0:
            errors.append(f"element_blur_budget_exceeded:{element_id}")
        binding = dict(item.get("binding") or {})
        decorative = bool(item.get("decorative"))
        if not decorative:
            kind = str(binding.get("kind") or "")
            reference = str(binding.get("id") or "")
            if kind not in ALLOWED_BINDING_KINDS or reference not in known.get(kind, set()):
                errors.append(f"invalid_semantic_binding:{element_id}")
            else:
                represented[kind].add(reference)
        text = _clean_text(item.get("text"), 240)
        if text:
            text_count += 1
            text_chars += len(text)
            if _copy_is_grounded(text, source_text, required_labels):
                grounded_text_count += 1
            elif not decorative:
                errors.append(f"ungrounded_element_copy:{element_id}")
        asset = dict(item.get("asset") or {})
        if asset:
            uri = str(asset.get("uri") or "")
            if re.match(r"^[a-z]+://", uri, flags=re.IGNORECASE):
                errors.append(f"remote_asset_forbidden:{element_id}")
    if text_chars > MAX_TOTAL_TEXT_CHARS:
        errors.append("open_visual_program_text_budget_exceeded")

    relation_bindings: set[str] = set()
    for index, item in enumerate(relations):
        relation_id = str(item.get("relation_id") or f"relation_{index}")
        if str(item.get("source_id") or "") not in element_ids:
            errors.append(f"visual_relation_unknown_source:{relation_id}")
        if str(item.get("target_id") or "") not in element_ids:
            errors.append(f"visual_relation_unknown_target:{relation_id}")
        binding = dict(item.get("binding") or {})
        reference = str(binding.get("id") or "")
        if str(binding.get("kind") or "") != "relation" or reference not in known["relation"]:
            errors.append(f"invalid_visual_relation_binding:{relation_id}")
        else:
            relation_bindings.add(reference)
            represented["relation"].add(reference)

    moved_element_ids: set[str] = set()
    for index, item in enumerate(tracks):
        track_id = str(item.get("track_id") or f"track_{index}")
        target_id = str(item.get("target_id") or "")
        if target_id not in element_ids:
            errors.append(f"motion_track_unknown_target:{track_id}")
        property_name = str(item.get("property") or "")
        if property_name not in ALLOWED_MOTION_PROPERTIES:
            errors.append(f"unsupported_motion_property:{track_id}:{property_name}")
        intent = str(item.get("semantic_intent") or "").strip()
        if not intent:
            errors.append(f"motion_track_has_no_semantic_intent:{track_id}")
        keyframes = [frame for frame in item.get("keyframes") or [] if isinstance(frame, dict)]
        if not 2 <= len(keyframes) <= MAX_KEYFRAMES_PER_TRACK:
            errors.append(f"invalid_motion_keyframes:{track_id}")
        previous_t = -1.0
        values: list[float] = []
        for frame in keyframes:
            timestamp = _number(frame.get("t"), -1.0)
            if timestamp < previous_t or not 0.0 <= timestamp <= 1.0:
                errors.append(f"invalid_motion_timeline:{track_id}")
                break
            previous_t = timestamp
            values.append(_number(frame.get("value"), 0.0))
            easing = str(frame.get("easing") or "linear")
            if easing not in ALLOWED_EASINGS:
                errors.append(f"unsupported_motion_easing:{track_id}:{easing}")
        if len(values) >= 2 and max(values) - min(values) > 1e-4:
            moved_element_ids.add(target_id)

    for index, item in enumerate(constraints):
        constraint_id = str(item.get("constraint_id") or index)
        if str(item.get("type") or "") not in ALLOWED_CONSTRAINTS:
            errors.append(f"unsupported_layout_constraint:{constraint_id}")
        targets = [str(value) for value in item.get("targets") or []]
        if any(target not in element_ids for target in targets):
            errors.append(f"layout_constraint_unknown_target:{constraint_id}")

    required_objects = known["object"]
    required_relations = known["relation"]
    object_coverage = len(represented["object"] & required_objects) / max(len(required_objects), 1)
    relation_coverage = len(relation_bindings & required_relations) / max(len(required_relations), 1)
    bound_element_ids = {
        str(item.get("element_id") or "")
        for item in elements
        if not bool(item.get("decorative"))
    }
    motion_coverage = len(moved_element_ids & bound_element_ids) / max(len(bound_element_ids), 1)
    grounded_text_ratio = grounded_text_count / max(text_count, 1)
    if required_objects and object_coverage < 1.0:
        errors.append("open_visual_program_omits_required_objects")
    if required_relations and relation_coverage < 1.0:
        errors.append("open_visual_program_omits_required_relations")
    if bound_element_ids and motion_coverage < 0.45:
        errors.append("open_visual_program_has_insufficient_semantic_motion")
    elif motion_coverage < 0.7:
        warnings.append("open_visual_program_motion_coverage_is_low")

    overlap_pairs = _high_overlap_pairs(elements)
    if overlap_pairs:
        warnings.append("open_visual_program_has_high_overlap:" + ",".join(overlap_pairs[:4]))

    resource_cost = _resource_cost(elements, relations, tracks)
    if resource_cost > MAX_RESOURCE_COST:
        errors.append("open_visual_program_resource_budget_exceeded")
    fingerprint = open_visual_program_fingerprint(payload)
    novelty_score = _novelty_score(fingerprint, history or [])
    semantic_fitness = _semantic_fitness(payload, evidence)
    if novelty_score < 0.24:
        warnings.append("open_visual_program_repeats_recent_visual_language")

    score = (
        object_coverage * 0.20
        + relation_coverage * 0.14
        + grounded_text_ratio * 0.16
        + motion_coverage * 0.16
        + semantic_fitness * 0.16
        + novelty_score * 0.10
        + max(0.0, 1.0 - resource_cost / MAX_RESOURCE_COST) * 0.08
    )
    if errors:
        score *= 0.35
    return OpenVisualProgramValidation(
        passed=not errors,
        score=max(0.0, min(score, 1.0)),
        errors=_unique(errors),
        warnings=_unique(warnings),
        object_coverage=object_coverage,
        relation_coverage=relation_coverage,
        grounded_text_ratio=grounded_text_ratio,
        motion_coverage=motion_coverage,
        semantic_fitness=semantic_fitness,
        novelty_score=novelty_score,
        resource_cost=resource_cost,
        fingerprint=fingerprint,
    )


def build_open_visual_program_candidates(
    ir: dict[str, Any],
    *,
    visual_id: str,
    width: int,
    height: int,
    duration_sec: float,
    fps: float,
    theme: dict[str, Any] | None = None,
    history: Iterable[dict[str, Any]] | None = None,
    candidate_count: int = 3,
) -> list[dict[str, Any]]:
    evidence = dict(ir or {})
    count = max(1, min(int(candidate_count), 4))
    source_text = _source_text(evidence).lower()
    candidates: list[dict[str, Any]] = []
    builders = (
        _compression_program if _is_compression_explanation(source_text) else _mechanism_program,
        _spatial_program,
        _editorial_program,
        _mechanism_program,
    )
    for index in range(count):
        candidate = builders[index](
            evidence,
            visual_id=visual_id,
            width=width,
            height=height,
            duration_sec=duration_sec,
            fps=fps,
            theme=theme or {},
            variant_index=index,
        )
        validation = validate_open_visual_program(
            candidate,
            ir=evidence,
            history=history,
        )
        if validation.passed:
            candidates.append(candidate)
    return candidates


def select_open_visual_program(
    candidates: Iterable[dict[str, Any]],
    *,
    ir: dict[str, Any],
    history: Iterable[dict[str, Any]] | None = None,
) -> OpenVisualTournament:
    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        validation = validate_open_visual_program(
            dict(candidate),
            ir=dict(ir or {}),
            history=history,
        )
        scored.append(
            {
                "program_id": str(candidate.get("program_id") or ""),
                "program": dict(candidate),
                "validation": validation.to_dict(),
                "eligible": validation.passed,
                "score": validation.score,
            }
        )
    scored.sort(
        key=lambda item: (
            bool(item["eligible"]),
            float(item["score"]),
            str(item["program_id"]),
        ),
        reverse=True,
    )
    selected = next((item for item in scored if item["eligible"]), None)
    selected_program_id = str(selected.get("program_id") or "") if selected else ""
    summary = [
        {
            "program_id": item["program_id"],
            "eligible": item["eligible"],
            "score": item["score"],
            "validation": item["validation"],
        }
        for item in scored
    ]
    signature_payload = {
        "version": OPEN_VISUAL_TOURNAMENT_VERSION,
        "selected_program_id": selected_program_id,
        "candidates": summary,
    }
    signature = hashlib.sha256(
        canonical_json(signature_payload).encode("utf-8")
    ).hexdigest()
    return OpenVisualTournament(
        version=OPEN_VISUAL_TOURNAMENT_VERSION,
        tournament_id=f"open-visual-tournament-{signature[:16]}",
        tournament_signature=signature,
        selected_program_id=selected_program_id,
        candidates=summary,
    )


def apply_open_visual_patch(
    program: dict[str, Any],
    operations: Iterable[dict[str, Any]],
    *,
    ir: dict[str, Any],
) -> OpenVisualPatchResult:
    patched = copy.deepcopy(dict(program or {}))
    applied: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    elements = {
        str(item.get("element_id") or ""): item
        for item in patched.get("elements") or []
        if isinstance(item, dict)
    }
    tracks = {
        str(item.get("track_id") or ""): item
        for item in patched.get("tracks") or []
        if isinstance(item, dict)
    }
    for raw in operations:
        operation = dict(raw or {})
        kind = str(operation.get("op") or "")
        target_id = str(operation.get("target_id") or "")
        if kind not in ALLOWED_PATCH_OPERATIONS:
            rejected.append({**operation, "reason": "unsupported_patch_operation"})
            continue
        target = elements.get(target_id)
        if kind not in {"set_concept", "remove_decorative"} and target is None and target_id not in tracks:
            rejected.append({**operation, "reason": "unknown_patch_target"})
            continue
        if kind == "move" and target is not None:
            layout = dict(target.get("layout") or {})
            layout["x"] = _number(operation.get("x"), layout.get("x", 0.0))
            layout["y"] = _number(operation.get("y"), layout.get("y", 0.0))
            target["layout"] = layout
        elif kind == "resize" and target is not None:
            layout = dict(target.get("layout") or {})
            layout["width"] = _number(operation.get("width"), layout.get("width", 0.2))
            layout["height"] = _number(operation.get("height"), layout.get("height", 0.2))
            target["layout"] = layout
        elif kind == "replace_text" and target is not None:
            target["text"] = _clean_text(operation.get("text"), 240)
        elif kind == "set_style" and target is not None:
            allowed_style = {
                key: value
                for key, value in dict(operation.get("style") or {}).items()
                if key in {"fill", "stroke", "stroke_width", "radius", "opacity", "font_size", "font_weight", "blur"}
            }
            target["style"] = {**dict(target.get("style") or {}), **allowed_style}
        elif kind == "set_motion":
            track = tracks.get(target_id)
            if track is None:
                rejected.append({**operation, "reason": "unknown_motion_track"})
                continue
            track["keyframes"] = copy.deepcopy(list(operation.get("keyframes") or []))
            if operation.get("semantic_intent"):
                track["semantic_intent"] = str(operation["semantic_intent"])
        elif kind == "remove_decorative":
            remove_ids = {
                str(item.get("element_id") or "")
                for item in patched.get("elements") or []
                if isinstance(item, dict) and bool(item.get("decorative"))
            }
            patched["elements"] = [
                item for item in patched.get("elements") or []
                if str((item or {}).get("element_id") or "") not in remove_ids
            ]
            patched["tracks"] = [
                item for item in patched.get("tracks") or []
                if str((item or {}).get("target_id") or "") not in remove_ids
            ]
        elif kind == "set_concept":
            updates = dict(operation.get("concept") or {})
            patched["concept"] = {
                **dict(patched.get("concept") or {}),
                **{
                    key: _clean_text(value, 160)
                    for key, value in updates.items()
                    if key in {"title", "medium", "metaphor", "composition", "takeaway"}
                },
            }
        applied.append(operation)
    patched = sign_open_visual_program(patched)
    validation = validate_open_visual_program(patched, ir=dict(ir or {}))
    return OpenVisualPatchResult(
        passed=validation.passed,
        program=patched,
        applied_operations=applied,
        rejected_operations=rejected,
        validation=validation.to_dict(),
    )


def open_visual_program_prompt_block(ir: dict[str, Any], *, candidate_count: int = 2) -> str:
    payload = dict(ir or {})
    compact_ir = {
        "visual_id": payload.get("visual_id"),
        "scene_type": payload.get("scene_type"),
        "thesis": payload.get("thesis"),
        "takeaway": payload.get("takeaway"),
        "facts": payload.get("facts") or [],
        "objects": payload.get("objects") or [],
        "relations": payload.get("relations") or [],
        "beats": payload.get("beats") or [],
        "required_labels": payload.get("required_labels") or [],
        "forbidden_content": payload.get("forbidden_content") or [],
        "evidence_signature": _ir_signature(payload),
    }
    return "\n".join(
        [
            "OPEN VISUAL PROGRAM AUTHORING CONTRACT",
            f"Return one JSON object with a 'programs' array containing exactly {max(1, min(candidate_count, 3))} distinct programs.",
            "Do not use a preset scene family as the composition. Author the objects, geometry, relations, and motion.",
            "Every non-decorative element needs binding={kind: fact|object|relation, id: exact IR id}.",
            "Only use text supported by the evidence. Never invent metrics, entities, outcomes, or interface states.",
            "Use normalized layout coordinates x/y/width/height in [0,1]. Keep all final bounds inside the canvas.",
            "Element types: " + ", ".join(sorted(ALLOWED_ELEMENT_TYPES)),
            "Motion properties: " + ", ".join(sorted(ALLOWED_MOTION_PROPERTIES)),
            "Each motion track needs at least two keyframes with t in [0,1], numeric value, and an allowed easing.",
            "Motion must demonstrate the mechanism or relationship, not merely decorate the scene.",
            "Prefer visual transformations, paths, grouping, ranking, comparison, and spatial metaphor over explanatory cards.",
            "Required root keys: version, program_id, evidence_signature, seed, canvas, concept, palette, elements, relations, tracks, constraints, quality_contract.",
            f"Set version to {OPEN_VISUAL_PROGRAM_VERSION!r}. Omit signature; Vex signs valid programs.",
            "Evidence IR:",
            canonical_json(compact_ir),
        ]
    )


def normalize_authored_open_visual_programs(
    payload: dict[str, Any],
    *,
    ir: dict[str, Any],
    visual_id: str,
    width: int,
    height: int,
    duration_sec: float,
    fps: float,
    theme: dict[str, Any] | None = None,
    history: Iterable[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    evidence_signature = _ir_signature(ir)
    for index, raw in enumerate(payload.get("programs") or []):
        if not isinstance(raw, dict):
            rejected.append({"index": index, "errors": ["program_is_not_an_object"]})
            continue
        program = copy.deepcopy(raw)
        program["version"] = OPEN_VISUAL_PROGRAM_VERSION
        program["program_id"] = _safe_id(
            program.get("program_id") or f"{visual_id}-authored-{index + 1:02d}"
        )
        program["evidence_signature"] = evidence_signature
        program["seed"] = int(_number(program.get("seed"), _seed(visual_id, index)))
        program["canvas"] = {
            "width": int(width),
            "height": int(height),
            "duration_sec": round(float(duration_sec), 4),
            "fps": round(float(fps), 4),
            "safe_area": {"top": 0.05, "right": 0.05, "bottom": 0.05, "left": 0.05},
            **dict(program.get("canvas") or {}),
        }
        program["palette"] = {
            **_palette(theme or {}),
            **dict(program.get("palette") or {}),
        }
        program.setdefault("relations", [])
        program.setdefault("tracks", [])
        program.setdefault("constraints", [])
        program.setdefault("quality_contract", _quality_contract(ir))
        program = sign_open_visual_program(program)
        validation = validate_open_visual_program(
            program,
            ir=dict(ir or {}),
            history=history,
        )
        if validation.passed:
            accepted.append(program)
        else:
            rejected.append(
                {
                    "index": index,
                    "program_id": program["program_id"],
                    "errors": validation.errors,
                    "warnings": validation.warnings,
                }
            )
    return accepted, rejected


def _base_program(
    ir: dict[str, Any],
    *,
    visual_id: str,
    width: int,
    height: int,
    duration_sec: float,
    fps: float,
    theme: dict[str, Any],
    variant_index: int,
    medium: str,
    metaphor: str,
    composition: str,
) -> dict[str, Any]:
    return {
        "version": OPEN_VISUAL_PROGRAM_VERSION,
        "program_id": f"{_safe_id(visual_id)}-ovp-{variant_index + 1:02d}",
        "evidence_signature": _ir_signature(ir),
        "seed": _seed(visual_id, variant_index),
        "canvas": {
            "width": int(width),
            "height": int(height),
            "duration_sec": round(float(duration_sec), 4),
            "fps": round(float(fps), 4),
            "safe_area": {"top": 0.05, "right": 0.05, "bottom": 0.05, "left": 0.05},
        },
        "concept": {
            "title": _clean_text((ir.get("metadata") or {}).get("display_title") or ir.get("thesis"), 100),
            "medium": medium,
            "metaphor": metaphor,
            "composition": composition,
            "takeaway": _clean_text(ir.get("takeaway"), 180),
        },
        "palette": _palette(theme),
        "elements": [],
        "relations": [],
        "tracks": [],
        "constraints": [],
        "quality_contract": _quality_contract(ir),
    }


def _compression_program(ir: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    program = _base_program(
        ir,
        **kwargs,
        medium="spatial_metaphor",
        metaphor="many signals physically converge into one compact representation",
        composition="compression funnel with a resolved output",
    )
    objects = [dict(item) for item in ir.get("objects") or [] if isinstance(item, dict)]
    relations = [dict(item) for item in ir.get("relations") or [] if isinstance(item, dict)]
    source = objects[0]
    mechanism = objects[1] if len(objects) >= 2 else objects[0]
    first_fact = _first_id(ir.get("facts"), "fact_id")
    title = _clean_text((ir.get("metadata") or {}).get("display_title") or ir.get("thesis"), 100)
    program["elements"].append(
        _element("title", "text", 0.06, 0.06, 0.86, 0.14, title, "fact", first_fact, role="title", style={"font_size": 62, "font_weight": 900, "fill": "text", "stroke_width": 0})
    )
    program["elements"].append(
        _element("source_label", "text", 0.11, 0.29, 0.265, 0.06, _grounded_short_label("FOUR TOKENS", source.get("label"), ir), "object", source["object_id"], role="source_label", style={"font_size": 28, "font_weight": 850, "color": "muted", "stroke_width": 0})
    )
    token_positions = [(0.11, 0.38), (0.11, 0.53), (0.27, 0.38), (0.27, 0.53)]
    for index, (x, y) in enumerate(token_positions):
        token_id = f"source_token_{index + 1:02d}"
        program["elements"].append(
            _element(token_id, "token", x, y, 0.105, 0.1, "", "object", source["object_id"], role="source_signal", style={"fill": "surface", "stroke": "accent_secondary", "radius": 8})
        )
        program["tracks"].extend(
            [
                _track(f"{token_id}_opacity", token_id, "opacity", [(0.06 + index * 0.03, 0.0), (0.18 + index * 0.03, 1.0), (0.63, 1.0)], "reveal the uncompressed inputs"),
                _track(f"{token_id}_x", token_id, "translate_x", [(0.48, 0.0), (0.69, 0.48 - x)], "converge four inputs into one representation"),
                _track(f"{token_id}_y", token_id, "translate_y", [(0.48, 0.0), (0.69, 0.475 - y)], "converge four inputs into one representation"),
                _track(f"{token_id}_scale", token_id, "scale", [(0.48, 1.0), (0.69, 0.18)], "collapse source detail during compression"),
            ]
        )
    program["elements"].append(
        _element("compression_gate", "shape", 0.43, 0.34, 0.12, 0.38, "", "object", mechanism["object_id"], role="transformation_gate", style={"fill": "accent", "stroke": "ink", "radius": 10})
    )
    program["elements"].append(
        _element("compressed_output", "token", 0.575, 0.385, 0.175, 0.22, _grounded_short_label("1 COMPRESSED KV ENTRY", mechanism.get("label"), ir), "object", mechanism["object_id"], role="compressed_representation", style={"fill": "accent_secondary", "stroke": "ink", "radius": 12, "font_weight": 800, "font_size": 30})
    )
    result = objects[-1] if len(objects) >= 3 else mechanism
    if result is not mechanism:
        program["elements"].append(
            _element("indexer_result", "shape", 0.78, 0.34, 0.17, 0.31, _grounded_short_label("INDEXER PICKS TOP BLOCKS", result.get("label"), ir), "object", result["object_id"], role="selection_result", style={"fill": "surface", "stroke": "accent", "radius": 8, "font_weight": 800, "font_size": 30})
        )
    program["tracks"].extend(
        [
            _track("gate_progress", "compression_gate", "progress", [(0.38, 0.0), (0.7, 1.0)], "show the compression transformation"),
            _track("output_opacity", "compressed_output", "opacity", [(0.57, 0.0), (0.72, 1.0), (1.0, 1.0)], "reveal the compressed result"),
            _track("output_scale", "compressed_output", "scale", [(0.57, 0.72), (0.75, 1.0), (1.0, 1.0)], "resolve the compressed result"),
            _track("title_opacity", "title", "opacity", [(0.0, 0.0), (0.12, 1.0), (1.0, 1.0)], "establish the explained mechanism"),
            _track("source_label_opacity", "source_label", "opacity", [(0.04, 0.0), (0.16, 1.0), (0.7, 1.0)], "label the four grounded source tokens"),
        ]
    )
    if result is not mechanism:
        program["tracks"].extend(
            [
                _track("indexer_opacity", "indexer_result", "opacity", [(0.68, 0.0), (0.82, 1.0), (1.0, 1.0)], "reveal the grounded indexer selection stage"),
                _track("indexer_emphasis", "indexer_result", "emphasis", [(0.68, 0.0), (0.86, 1.0), (1.0, 0.82)], "focus attention on the selected result"),
            ]
        )
    object_to_element = {
        str(source.get("object_id") or ""): "source_token_01",
        str(mechanism.get("object_id") or ""): "compressed_output",
        str(result.get("object_id") or ""): "indexer_result" if result is not mechanism else "compressed_output",
    }
    for index, relation in enumerate(relations):
        source_element = object_to_element.get(str(relation.get("source_id") or ""))
        target_element = object_to_element.get(str(relation.get("target_id") or ""))
        if not source_element or not target_element:
            continue
        program["relations"].append(
            {
                "relation_id": f"visual_relation_{index + 1:02d}",
                "source_id": source_element,
                "target_id": target_element,
                "type": "transforms_into" if index == 0 else "feeds_selection",
                "binding": {"kind": "relation", "id": str(relation.get("relation_id") or "")},
                "style": {"stroke": "accent", "stroke_width": 4},
            }
        )
    program["constraints"] = [
        {"constraint_id": "safe", "type": "keep_inside_safe_area", "targets": [item["element_id"] for item in program["elements"]]},
        {"constraint_id": "tokens", "type": "distribute", "targets": [f"source_token_{index:02d}" for index in range(1, 5)], "axis": "both"},
    ]
    return sign_open_visual_program(program)


def _mechanism_program(ir: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    program = _base_program(
        ir,
        **kwargs,
        medium="diagrammatic_system",
        metaphor="a directed mechanism whose state travels through transformations",
        composition="asymmetric route with a strong resolved destination",
    )
    objects = [dict(item) for item in ir.get("objects") or [] if isinstance(item, dict)]
    relation_items = [dict(item) for item in ir.get("relations") or [] if isinstance(item, dict)]
    count = max(len(objects), 1)
    first_fact = _first_id(ir.get("facts"), "fact_id")
    program["elements"].append(
        _element("title", "text", 0.06, 0.07, 0.78, 0.1, _clean_text((ir.get("metadata") or {}).get("display_title") or ir.get("thesis"), 100), "fact", first_fact, role="title", style={"font_size": 60, "font_weight": 900, "fill": "text"})
    )
    for index, obj in enumerate(objects):
        x = 0.08 + index * (0.78 / max(count, 1))
        y = 0.37 + (0.07 if index % 2 else 0.0)
        element_id = f"mechanism_{index + 1:02d}"
        program["elements"].append(
            _element(element_id, "shape", x, y, min(0.19, 0.7 / count), 0.24, _clean_text(obj.get("label"), 90), "object", str(obj.get("object_id") or ""), role=str(obj.get("role") or "step"), style={"fill": "surface", "stroke": "accent_secondary" if index % 2 else "accent", "radius": 8, "font_weight": 800})
        )
        program["tracks"].extend(
            [
                _track(f"{element_id}_opacity", element_id, "opacity", [(0.08 + index * 0.1, 0.0), (0.24 + index * 0.1, 1.0), (1.0, 1.0)], "reveal the mechanism in explanatory order"),
                _track(f"{element_id}_y", element_id, "translate_y", [(0.08 + index * 0.1, 0.08), (0.28 + index * 0.1, 0.0), (1.0, 0.0)], "settle each mechanism state into its final position"),
            ]
        )
    object_to_element = {
        str(obj.get("object_id") or ""): f"mechanism_{index + 1:02d}"
        for index, obj in enumerate(objects)
    }
    for index, relation in enumerate(relation_items):
        source_id = object_to_element.get(str(relation.get("source_id") or ""))
        target_id = object_to_element.get(str(relation.get("target_id") or ""))
        if not source_id or not target_id:
            continue
        visual_id = f"route_{index + 1:02d}"
        program["relations"].append(
            {"relation_id": visual_id, "source_id": source_id, "target_id": target_id, "type": str(relation.get("relation_type") or "directed"), "binding": {"kind": "relation", "id": str(relation.get("relation_id") or "")}, "style": {"stroke": "accent", "stroke_width": 4}}
        )
    program["tracks"].append(
        _track("title_opacity", "title", "opacity", [(0.0, 0.0), (0.12, 1.0), (1.0, 1.0)], "establish the mechanism")
    )
    program["constraints"] = [
        {"constraint_id": "safe", "type": "keep_inside_safe_area", "targets": [item["element_id"] for item in program["elements"]]},
        {"constraint_id": "route", "type": "distribute", "targets": list(object_to_element.values()), "axis": "x"},
    ]
    return sign_open_visual_program(program)


def _spatial_program(ir: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    program = _mechanism_program(ir, **kwargs)
    program["concept"].update(
        {"medium": "spatial_stage", "metaphor": "evidence orbits a focal transformation and resolves into the outcome", "composition": "radial evidence field with depth"}
    )
    bound = [item for item in program["elements"] if item["element_id"] != "title"]
    for index, item in enumerate(bound):
        angle = -math.pi * 0.82 + index * (math.pi * 1.15 / max(len(bound) - 1, 1))
        layout = dict(item["layout"])
        layout["x"] = round(0.45 + math.cos(angle) * 0.27 - layout["width"] / 2, 4)
        layout["y"] = round(0.52 + math.sin(angle) * 0.23 - layout["height"] / 2, 4)
        item["layout"] = layout
        item["style"] = {**dict(item.get("style") or {}), "depth": index + 1}
    program["signature"] = open_visual_program_signature(program)
    return program


def _editorial_program(ir: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    program = _mechanism_program(ir, **kwargs)
    program["concept"].update(
        {"medium": "editorial_motion", "metaphor": "one dominant statement is assembled from moving evidence fragments", "composition": "large type with a cinematic evidence rail"}
    )
    elements = program["elements"]
    title = next((item for item in elements if item["element_id"] == "title"), None)
    if title:
        title["layout"] = {"x": 0.06, "y": 0.1, "width": 0.86, "height": 0.22, "anchor": "top_left"}
        title["style"] = {**dict(title.get("style") or {}), "font_size": 86}
    supporting = [item for item in elements if item["element_id"] != "title"]
    for index, item in enumerate(supporting):
        item["type"] = "text"
        item["layout"] = {"x": 0.08 + index * (0.82 / max(len(supporting), 1)), "y": 0.55, "width": min(0.24, 0.76 / max(len(supporting), 1)), "height": 0.18, "anchor": "top_left"}
        item["style"] = {"fill": "text", "stroke": "accent" if index == 0 else "accent_secondary", "font_size": 34, "font_weight": 850}
    program["signature"] = open_visual_program_signature(program)
    return program


def _element(
    element_id: str,
    element_type: str,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    binding_kind: str,
    binding_id: str,
    *,
    role: str,
    style: dict[str, Any],
) -> dict[str, Any]:
    return {
        "element_id": element_id,
        "type": element_type,
        "role": role,
        "text": text,
        "binding": {"kind": binding_kind, "id": binding_id},
        "decorative": False,
        "layout": {"x": round(x, 4), "y": round(y, 4), "width": round(width, 4), "height": round(height, 4), "anchor": "top_left"},
        "style": style,
        "repeat": 1,
    }


def _track(
    track_id: str,
    target_id: str,
    property_name: str,
    values: list[tuple[float, float]],
    intent: str,
) -> dict[str, Any]:
    return {
        "track_id": track_id,
        "target_id": target_id,
        "property": property_name,
        "semantic_intent": intent,
        "keyframes": [
            {"t": round(t, 4), "value": round(value, 4), "easing": "ease_in_out"}
            for t, value in values
        ],
    }


def _quality_contract(ir: dict[str, Any]) -> dict[str, Any]:
    return {
        "required_object_ids": sorted(_ids(ir.get("objects"), "object_id")),
        "required_relation_ids": sorted(_ids(ir.get("relations"), "relation_id")),
        "forbidden_content": [str(item) for item in ir.get("forbidden_content") or []],
        "minimum_motion_coverage": 0.45,
        "minimum_grounded_text_ratio": 1.0,
        "final_hold_start": 0.8,
        "maximum_resource_cost": MAX_RESOURCE_COST,
    }


def _palette(theme: dict[str, Any]) -> dict[str, str]:
    return {
        "background": str(theme.get("background") or "#F4F0E8"),
        "surface": str(theme.get("panel_fill") or "#FFFDF8"),
        "ink": str(theme.get("text_primary") or "#111111"),
        "muted": str(theme.get("text_secondary") or "#4B4740"),
        "accent": str(theme.get("accent") or "#F04438"),
        "accent_secondary": str(theme.get("accent_secondary") or "#1E5EFF"),
        "grid": str(theme.get("grid") or "#C8C0B4"),
    }


def _ir_signature(ir: dict[str, Any]) -> str:
    existing = str(ir.get("signature") or ir.get("visual_explanation_ir_signature") or "")
    if existing:
        return existing
    payload = copy.deepcopy(dict(ir or {}))
    payload.pop("signature", None)
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _source_text(ir: dict[str, Any]) -> str:
    values = [
        str(item.get("text") or "")
        for item in ir.get("evidence") or []
        if isinstance(item, dict)
    ]
    values.extend(
        [str(ir.get("thesis") or ""), str(ir.get("takeaway") or "")]
    )
    metadata = dict(ir.get("metadata") or {})
    values.extend(
        [
            str(metadata.get("display_title") or ""),
            str(metadata.get("display_title_evidence") or ""),
        ]
    )
    values.extend(str(item) for item in ir.get("required_labels") or [])
    return " ".join(value for value in values if value).strip()


def _copy_is_grounded(text: str, source_text: str, required_labels: list[str]) -> bool:
    normalized = _normalized_grounding_text(text)
    if not normalized:
        return True
    source = _normalized_grounding_text(source_text)
    if normalized in source:
        return True
    if any(normalized == _normalized_grounding_text(label) for label in required_labels):
        return True
    tokens = set(normalized.split())
    source_tokens = set(source.split())
    return bool(tokens) and tokens.issubset(source_tokens)


def _normalized_grounding_text(value: Any) -> str:
    return " ".join(
        _NUMBER_WORD_TOKENS.get(token, token)
        for token in _WORD_RE.findall(str(value or "").lower())
    )


def _grounded_short_label(preferred: str, fallback: Any, ir: dict[str, Any]) -> str:
    candidate = _clean_text(preferred, 90)
    if _copy_is_grounded(
        candidate,
        _source_text(ir),
        [str(item) for item in ir.get("required_labels") or []],
    ):
        return candidate
    return _clean_text(fallback, 90)


def _valid_layout(layout: dict[str, Any]) -> bool:
    try:
        x = float(layout.get("x"))
        y = float(layout.get("y"))
        width = float(layout.get("width"))
        height = float(layout.get("height"))
    except (TypeError, ValueError):
        return False
    return (
        0.0 <= x <= 1.0
        and 0.0 <= y <= 1.0
        and 0.005 <= width <= 1.0
        and 0.005 <= height <= 1.0
        and x + width <= 1.0001
        and y + height <= 1.0001
    )


def _resource_cost(
    elements: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    tracks: list[dict[str, Any]],
) -> float:
    cost = float(len(elements)) + len(relations) * 1.5 + len(tracks) * 0.8
    for item in elements:
        repeat = max(1, int(_number(item.get("repeat"), 1.0)))
        cost += max(0, repeat - 1) * 0.7
        element_type = str(item.get("type") or "")
        if element_type in {"chart", "image", "mask", "particle", "path"}:
            cost += 3.0
        cost += _number((item.get("style") or {}).get("blur"), 0.0) * 0.08
    return round(cost, 4)


def _high_overlap_pairs(elements: list[dict[str, Any]]) -> list[str]:
    meaningful = [item for item in elements if not bool(item.get("decorative"))]
    pairs: list[str] = []
    for index, first in enumerate(meaningful):
        for second in meaningful[index + 1 :]:
            if _intersection_over_union(first.get("layout") or {}, second.get("layout") or {}) > 0.72:
                pairs.append(f"{first.get('element_id')}+{second.get('element_id')}")
    return pairs


def _intersection_over_union(first: dict[str, Any], second: dict[str, Any]) -> float:
    ax, ay = _number(first.get("x")), _number(first.get("y"))
    aw, ah = _number(first.get("width")), _number(first.get("height"))
    bx, by = _number(second.get("x")), _number(second.get("y"))
    bw, bh = _number(second.get("width")), _number(second.get("height"))
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def _novelty_score(fingerprint: dict[str, Any], history: Iterable[dict[str, Any]]) -> float:
    current_signature = str(fingerprint.get("signature") or "")
    current_medium = str(fingerprint.get("medium") or "")
    current_composition = str(fingerprint.get("composition") or "")
    similarities: list[float] = []
    for item in history:
        prior = dict(item.get("fingerprint") or item)
        if str(prior.get("signature") or "") == current_signature:
            similarities.append(1.0)
            continue
        similarity = 0.0
        similarity += 0.4 if str(prior.get("medium") or "") == current_medium else 0.0
        similarity += 0.3 if str(prior.get("composition") or "") == current_composition else 0.0
        prior_types = set((prior.get("type_counts") or {}).keys())
        current_types = set((fingerprint.get("type_counts") or {}).keys())
        if prior_types or current_types:
            similarity += 0.3 * len(prior_types & current_types) / max(len(prior_types | current_types), 1)
        similarities.append(similarity)
    return max(0.0, 1.0 - max(similarities, default=0.0))


def _is_compression_explanation(text: str) -> bool:
    return "compress" in text and bool(
        re.search(r"\b(?:two|three|four|five|six|\d+)\b.{0,80}\b(?:one|1)\b", text)
        or "compressed" in text
    )


def _semantic_fitness(program: dict[str, Any], ir: dict[str, Any]) -> float:
    source = _source_text(ir).lower()
    concept = dict(program.get("concept") or {})
    medium = str(concept.get("medium") or "")
    tracks = [item for item in program.get("tracks") or [] if isinstance(item, dict)]
    properties = [str(item.get("property") or "") for item in tracks]
    relation_count = len([item for item in program.get("relations") or [] if isinstance(item, dict)])
    if _is_compression_explanation(source):
        score = 0.48
        score += 0.2 if medium == "spatial_metaphor" else 0.0
        score += min(properties.count("translate_x") + properties.count("translate_y"), 4) * 0.05
        score += 0.08 if "scale" in properties else 0.0
        score += 0.04 if "progress" in properties else 0.0
        return min(score, 1.0)
    scene_type = str(ir.get("scene_type") or "")
    score = 0.62
    if scene_type in {"architecture_flow", "causal_intervention", "guided_process", "set_partition"}:
        score += 0.12 if medium in {"diagrammatic_system", "spatial_metaphor", "spatial_stage"} else 0.0
        score += min(relation_count, 2) * 0.06
        score += 0.08 if any(item in properties for item in {"progress", "translate_x", "translate_y"}) else 0.0
    elif scene_type in {"metric_delta", "metric_intervention", "metric_proof"}:
        score += 0.18 if any(str(item.get("type") or "") in {"chart", "metric"} for item in program.get("elements") or []) else 0.0
    elif scene_type in {"matched_state_transform", "decision_branch"}:
        score += 0.12 if relation_count else 0.0
    return min(score, 1.0)


def _ids(value: Any, key: str) -> set[str]:
    return {
        str(item.get(key) or "")
        for item in value or []
        if isinstance(item, dict) and str(item.get(key) or "")
    }


@lru_cache(maxsize=1)
def _schema_validator() -> Draft202012Validator:
    schema_path = files("vex_visuals").joinpath(
        "open_visual_program.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _schema_errors(payload: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    for error in sorted(
        _schema_validator().iter_errors(payload),
        key=lambda item: tuple(str(part) for part in item.absolute_path),
    ):
        path = ".".join(str(part) for part in error.absolute_path) or "root"
        issues.append(f"open_visual_schema:{path}:{error.validator}")
    return issues[:24]


def _first_id(value: Any, key: str) -> str:
    return next(iter(sorted(_ids(value, key))), "")


def _seed(visual_id: str, variant_index: int) -> int:
    digest = hashlib.sha256(f"{visual_id}:{variant_index}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _safe_id(value: Any) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(value or "visual")).strip("-_").lower()
    return cleaned[:96] or "visual"


def _clean_text(value: Any, limit: int) -> str:
    cleaned = re.sub(r"\s+", " ", str(value or "")).strip()
    return cleaned[:limit].rstrip(" ,.;:-")


def _number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value)))


__all__ = [
    "ALLOWED_PATCH_OPERATIONS",
    "OPEN_VISUAL_PATCH_VERSION",
    "OPEN_VISUAL_PROGRAM_VERSION",
    "OPEN_VISUAL_TOURNAMENT_VERSION",
    "OpenVisualPatchResult",
    "OpenVisualProgramValidation",
    "OpenVisualTournament",
    "apply_open_visual_patch",
    "build_open_visual_program_candidates",
    "normalize_authored_open_visual_programs",
    "open_visual_program_fingerprint",
    "open_visual_program_prompt_block",
    "open_visual_program_signature",
    "select_open_visual_program",
    "sign_open_visual_program",
    "validate_open_visual_program",
]
