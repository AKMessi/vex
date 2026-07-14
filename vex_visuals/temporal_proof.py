from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any


TEMPORAL_PROOF_VERSION = "vex-temporal-proof-v1"
DEFAULT_INITIAL_SAMPLE = 0.03
DEFAULT_INITIAL_OPACITY = 0.72
DEFAULT_MIN_FONT_PX = 22.0
DEFAULT_FINAL_HOLD_START = 0.8


@dataclass(frozen=True)
class TemporalProofValidation:
    passed: bool
    score: float
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    initial_readability: float = 0.0
    sequence_legibility: float = 0.0
    final_readability: float = 0.0
    minimum_estimated_font_px: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "score",
            "initial_readability",
            "sequence_legibility",
            "final_readability",
            "minimum_estimated_font_px",
        ):
            payload[key] = round(float(payload[key]), 4)
        return payload


def attach_temporal_proof_contract(program: dict[str, Any]) -> dict[str, Any]:
    """Attach a renderer-independent proof of how the explanation unfolds."""
    elements = [item for item in program.get("elements") or [] if isinstance(item, dict)]
    relations = [item for item in program.get("relations") or [] if isinstance(item, dict)]
    title_ids = [
        str(item.get("element_id") or "")
        for item in elements
        if str(item.get("role") or "").lower() == "title"
    ]
    semantic_ids = [
        str(item.get("element_id") or "")
        for item in elements
        if not bool(item.get("decorative")) and str(item.get("element_id") or "")
    ]
    ordered_ids = _topological_element_order(elements, relations)
    quality = dict(program.get("quality_contract") or {})
    quality["temporal_proof"] = {
        "version": TEMPORAL_PROOF_VERSION,
        "initial_sample": DEFAULT_INITIAL_SAMPLE,
        "initial_context_ids": title_ids[:1] or semantic_ids[:1],
        "ordered_element_ids": ordered_ids,
        "required_relation_ids": [
            str(item.get("relation_id") or "")
            for item in relations
            if str(item.get("relation_id") or "")
        ],
        "final_state_ids": semantic_ids,
        "minimum_initial_opacity": DEFAULT_INITIAL_OPACITY,
        "minimum_semantic_font_px": DEFAULT_MIN_FONT_PX,
        "final_hold_start": float(
            quality.get("final_hold_start") or DEFAULT_FINAL_HOLD_START
        ),
    }
    program["quality_contract"] = quality
    return program


def validate_temporal_proof(program: dict[str, Any]) -> TemporalProofValidation:
    elements = [item for item in program.get("elements") or [] if isinstance(item, dict)]
    relations = [item for item in program.get("relations") or [] if isinstance(item, dict)]
    tracks = [item for item in program.get("tracks") or [] if isinstance(item, dict)]
    by_id = {str(item.get("element_id") or ""): item for item in elements}
    tracks_by_target: dict[str, list[dict[str, Any]]] = {}
    for track in tracks:
        tracks_by_target.setdefault(str(track.get("target_id") or ""), []).append(track)

    proof = dict((program.get("quality_contract") or {}).get("temporal_proof") or {})
    errors: list[str] = []
    warnings: list[str] = []
    if str(proof.get("version") or "") != TEMPORAL_PROOF_VERSION:
        errors.append("temporal_proof_contract_missing_or_unsupported")

    initial_sample = _number(proof.get("initial_sample"), DEFAULT_INITIAL_SAMPLE)
    min_initial_opacity = _number(
        proof.get("minimum_initial_opacity"), DEFAULT_INITIAL_OPACITY
    )
    min_font = _number(proof.get("minimum_semantic_font_px"), DEFAULT_MIN_FONT_PX)
    final_hold = _number(proof.get("final_hold_start"), DEFAULT_FINAL_HOLD_START)
    initial_ids = [str(value) for value in proof.get("initial_context_ids") or []]
    final_ids = [str(value) for value in proof.get("final_state_ids") or []]
    required_relation_ids = {
        str(value) for value in proof.get("required_relation_ids") or [] if str(value)
    }

    readable_initial = 0
    for element_id in initial_ids:
        if element_id not in by_id:
            errors.append(f"temporal_proof_unknown_initial_context:{element_id}")
            continue
        opacity = _property_value(
            tracks_by_target.get(element_id, []),
            "opacity",
            initial_sample,
            _number((by_id[element_id].get("style") or {}).get("opacity"), 1.0),
        )
        if opacity + 1e-6 < min_initial_opacity:
            errors.append(f"temporal_proof_initial_context_hidden:{element_id}")
        else:
            readable_initial += 1
    if not initial_ids:
        errors.append("temporal_proof_has_no_initial_context")
    initial_readability = readable_initial / max(len(initial_ids), 1)

    readable_final = 0
    for element_id in final_ids:
        if element_id not in by_id:
            errors.append(f"temporal_proof_unknown_final_state:{element_id}")
            continue
        opacity = _property_value(
            tracks_by_target.get(element_id, []),
            "opacity",
            final_hold,
            _number((by_id[element_id].get("style") or {}).get("opacity"), 1.0),
        )
        if opacity < 0.9:
            errors.append(f"temporal_proof_final_state_hidden:{element_id}")
        else:
            readable_final += 1
    final_readability = readable_final / max(len(final_ids), 1)

    estimated_fonts: list[float] = []
    for item in elements:
        if bool(item.get("decorative")) or not str(item.get("text") or "").strip():
            continue
        estimate = _estimated_font_size(program, item)
        estimated_fonts.append(estimate)
        if estimate + 1e-6 < min_font:
            errors.append(
                f"temporal_proof_semantic_copy_too_small:{item.get('element_id')}"
            )
    minimum_estimated_font = min(estimated_fonts, default=min_font)

    sequence_checks = 0
    sequence_passes = 0
    relation_types = {
        str(item.get("element_id") or ""): str(item.get("type") or "")
        for item in elements
    }
    for relation in relations:
        relation_id = str(relation.get("relation_id") or "")
        if required_relation_ids and relation_id not in required_relation_ids:
            continue
        sequence_checks += 1
        source_id = str(relation.get("source_id") or "")
        target_id = str(relation.get("target_id") or "")
        progress_track = next(
            (
                item
                for item in tracks_by_target.get(relation_id, [])
                if str(item.get("property") or "") == "progress"
            ),
            None,
        )
        if progress_track is None:
            errors.append(f"temporal_proof_relation_has_no_reveal_track:{relation_id}")
            continue
        source_time = _first_visible_time(tracks_by_target.get(source_id, []))
        relation_time = _first_visible_time([progress_track], property_name="progress")
        target_time = _first_visible_time(tracks_by_target.get(target_id, []))
        if source_time > relation_time + 0.02 or relation_time > target_time + 0.02:
            errors.append(f"temporal_proof_sequence_is_not_causal:{relation_id}")
            continue
        if not _relation_has_separation(by_id.get(source_id), by_id.get(target_id)):
            errors.append(f"temporal_proof_relation_endpoints_overlap:{relation_id}")
            continue
        if relation_types.get(source_id) == "text" and relation_types.get(target_id) == "text":
            errors.append(f"temporal_proof_relation_is_unframed_text:{relation_id}")
            continue
        sequence_passes += 1
    sequence_legibility = sequence_passes / max(sequence_checks, 1)

    if final_hold > 0.84:
        warnings.append("temporal_proof_final_hold_is_short")
    score = (
        initial_readability * 0.28
        + sequence_legibility * 0.42
        + final_readability * 0.22
        + min(max(minimum_estimated_font / max(min_font * 1.35, 1.0), 0.0), 1.0)
        * 0.08
    )
    if errors:
        score *= 0.3
    return TemporalProofValidation(
        passed=not errors,
        score=max(0.0, min(score, 1.0)),
        errors=_unique(errors),
        warnings=_unique(warnings),
        initial_readability=initial_readability,
        sequence_legibility=sequence_legibility,
        final_readability=final_readability,
        minimum_estimated_font_px=minimum_estimated_font,
    )


def _topological_element_order(
    elements: list[dict[str, Any]], relations: list[dict[str, Any]]
) -> list[str]:
    semantic_ids = [
        str(item.get("element_id") or "")
        for item in elements
        if not bool(item.get("decorative"))
        and str(item.get("role") or "").lower() != "title"
        and str(item.get("element_id") or "")
    ]
    if not relations:
        return semantic_ids
    incoming = {element_id: 0 for element_id in semantic_ids}
    outgoing: dict[str, list[str]] = {element_id: [] for element_id in semantic_ids}
    for relation in relations:
        source = str(relation.get("source_id") or "")
        target = str(relation.get("target_id") or "")
        if source in outgoing and target in incoming:
            outgoing[source].append(target)
            incoming[target] += 1
    queue = [item for item in semantic_ids if incoming[item] == 0]
    ordered: list[str] = []
    while queue:
        current = queue.pop(0)
        ordered.append(current)
        for target in outgoing[current]:
            incoming[target] -= 1
            if incoming[target] == 0:
                queue.append(target)
    return ordered + [item for item in semantic_ids if item not in ordered]


def _property_value(
    tracks: list[dict[str, Any]], property_name: str, timestamp: float, fallback: float
) -> float:
    track = next(
        (item for item in tracks if str(item.get("property") or "") == property_name),
        None,
    )
    if track is None:
        return fallback
    keyframes = sorted(
        (
            (_number(item.get("t"), 0.0), _number(item.get("value"), fallback))
            for item in track.get("keyframes") or []
            if isinstance(item, dict)
        ),
        key=lambda item: item[0],
    )
    if not keyframes:
        return fallback
    if timestamp <= keyframes[0][0]:
        return keyframes[0][1]
    if timestamp >= keyframes[-1][0]:
        return keyframes[-1][1]
    for (left_t, left_value), (right_t, right_value) in zip(keyframes, keyframes[1:]):
        if left_t <= timestamp <= right_t:
            progress = (timestamp - left_t) / max(right_t - left_t, 1e-6)
            return left_value + (right_value - left_value) * progress
    return fallback


def _first_visible_time(
    tracks: list[dict[str, Any]], *, property_name: str = "opacity"
) -> float:
    track = next(
        (item for item in tracks if str(item.get("property") or "") == property_name),
        None,
    )
    if track is None:
        return 0.0
    keyframes = sorted(
        [item for item in track.get("keyframes") or [] if isinstance(item, dict)],
        key=lambda item: _number(item.get("t"), 0.0),
    )
    threshold = 0.62
    previous_t = 0.0
    previous_value = _number(keyframes[0].get("value"), 0.0) if keyframes else 1.0
    if previous_value >= threshold:
        return _number(keyframes[0].get("t"), 0.0) if keyframes else 0.0
    for item in keyframes[1:]:
        timestamp = _number(item.get("t"), previous_t)
        value = _number(item.get("value"), previous_value)
        if value >= threshold and value > previous_value:
            ratio = (threshold - previous_value) / max(value - previous_value, 1e-6)
            return previous_t + (timestamp - previous_t) * ratio
        previous_t, previous_value = timestamp, value
    return 1.0


def _estimated_font_size(program: dict[str, Any], element: dict[str, Any]) -> float:
    canvas = dict(program.get("canvas") or {})
    layout = dict(element.get("layout") or {})
    style = dict(element.get("style") or {})
    requested = min(max(_number(style.get("font_size"), 30.0), 12.0), 110.0)
    width_px = max(_number(layout.get("width"), 0.1) * _number(canvas.get("width"), 1920), 1.0)
    height_px = max(_number(layout.get("height"), 0.1) * _number(canvas.get("height"), 1080), 1.0)
    framed = str(element.get("type") or "") in {
        "shape",
        "token",
        "metric",
        "group",
        "chart",
        "image",
    }
    width_px -= 32.0 if framed else 4.0
    height_px -= 24.0 if framed else 2.0
    text = " ".join(str(element.get("text") or "").split())
    for size in range(int(math.floor(requested)), 11, -1):
        chars_per_line = max(int(width_px / max(size * 0.54, 1.0)), 1)
        line_count = max(1, math.ceil(len(text) / chars_per_line))
        if line_count * size * 1.12 <= height_px:
            return float(size)
    return 12.0


def _relation_has_separation(
    source: dict[str, Any] | None, target: dict[str, Any] | None
) -> bool:
    if not source or not target:
        return False
    left = dict(source.get("layout") or {})
    right = dict(target.get("layout") or {})
    left_center = (
        _number(left.get("x"), 0.0) + _number(left.get("width"), 0.0) / 2,
        _number(left.get("y"), 0.0) + _number(left.get("height"), 0.0) / 2,
    )
    right_center = (
        _number(right.get("x"), 0.0) + _number(right.get("width"), 0.0) / 2,
        _number(right.get("y"), 0.0) + _number(right.get("height"), 0.0) / 2,
    )
    return math.dist(left_center, right_center) >= 0.12


def _number(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


__all__ = [
    "TEMPORAL_PROOF_VERSION",
    "TemporalProofValidation",
    "attach_temporal_proof_contract",
    "validate_temporal_proof",
]
