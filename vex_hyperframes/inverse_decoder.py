from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np

from vex_hyperframes.claim_graph import RELATION_TYPES


INVERSE_DECODER_VERSION = "hyperframes-inverse-decoder-v1"


@dataclass(frozen=True)
class DecodedRelation:
    source: str
    relation_type: str
    target: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["confidence"] = round(float(self.confidence), 4)
        return payload


@dataclass(frozen=True)
class BlindFrameDecode:
    thesis: str
    objects: list[str]
    relations: list[DecodedRelation]
    sequence: list[str]
    confidence: float
    ambiguities: list[str] = field(default_factory=list)
    unsupported_visual_claims: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["relations"] = [item.to_dict() for item in self.relations]
        payload["confidence"] = round(float(self.confidence), 4)
        return payload


@dataclass(frozen=True)
class CounterfactualSensitivity:
    enabled: bool
    passed: bool | None
    relation_ablation_delta: float | None
    temporal_scramble_delta: float | None
    score: float | None
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("relation_ablation_delta", "temporal_scramble_delta", "score"):
            if payload[key] is not None:
                payload[key] = round(float(payload[key]), 4)
        return payload


@dataclass(frozen=True)
class InverseDecodeEvaluation:
    passed: bool
    score: float
    thesis_score: float
    object_coverage: float
    relation_coverage: float
    sequence_score: float
    missing_labels: list[str]
    missing_relation_ids: list[str]
    issues: list[str]
    repair_directives: list[str]
    counterfactual: CounterfactualSensitivity

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "score",
            "thesis_score",
            "object_coverage",
            "relation_coverage",
            "sequence_score",
        ):
            payload[key] = round(float(payload[key]), 4)
        payload["counterfactual"] = self.counterfactual.to_dict()
        return payload


def blind_decode_prompt(frame_count: int) -> str:
    relation_vocabulary = ", ".join(sorted(RELATION_TYPES))
    return "\n".join(
        [
            f"You will receive {frame_count} chronological frames from one silent motion graphic.",
            "Infer only what a viewer can understand from the pixels. You are not given the intended script, transcript, storyboard, or answer.",
            "Do not reward polish. Report ambiguity instead of guessing.",
            "Use visible wording exactly when possible.",
            f"Choose relation_type only from: {relation_vocabulary}.",
            "Return one JSON object with exactly these keys:",
            "thesis: one sentence describing the visual's apparent claim;",
            "objects: array of distinct visible entities, states, values, or actions;",
            "relations: array of {source, relation_type, target, confidence};",
            "sequence: array of objects in the order the visual communicates them;",
            "confidence: number from 0 to 1;",
            "ambiguities: array of unclear or contradictory elements;",
            "unsupported_visual_claims: array of claims implied by the visual without visible evidence.",
        ]
    )


def parse_blind_decode(payload: dict[str, Any]) -> BlindFrameDecode:
    relations: list[DecodedRelation] = []
    for item in payload.get("relations") or []:
        if not isinstance(item, dict):
            continue
        source = _clean_text(item.get("source"))
        target = _clean_text(item.get("target"))
        relation_type = _normalize(item.get("relation_type")).replace(" ", "_")
        if not source or not target or relation_type not in RELATION_TYPES:
            continue
        relations.append(
            DecodedRelation(
                source=source,
                relation_type=relation_type,
                target=target,
                confidence=_bounded(item.get("confidence"), 0.5),
            )
        )
    return BlindFrameDecode(
        thesis=_clean_text(payload.get("thesis")),
        objects=_string_list(payload.get("objects"), limit=16),
        relations=relations[:16],
        sequence=_string_list(payload.get("sequence"), limit=16),
        confidence=_bounded(payload.get("confidence"), 0.0),
        ambiguities=_string_list(payload.get("ambiguities"), limit=12),
        unsupported_visual_claims=_string_list(
            payload.get("unsupported_visual_claims"),
            limit=12,
        ),
    )


def evaluate_inverse_decode(
    decoded: BlindFrameDecode,
    *,
    production_contract: dict[str, Any],
    relation_ablation_decode: BlindFrameDecode | None = None,
    temporal_scramble_decode: BlindFrameDecode | None = None,
    min_score: float = 0.68,
    require_counterfactuals: bool = True,
) -> InverseDecodeEvaluation:
    graph = dict(production_contract.get("visual_claim_graph") or {})
    nodes = [dict(item) for item in graph.get("nodes") or [] if isinstance(item, dict)]
    relations = [
        dict(item)
        for item in graph.get("relations") or []
        if isinstance(item, dict) and item.get("required", True)
    ]
    node_labels = {
        str(item.get("node_id") or ""): _clean_text(item.get("label"))
        for item in nodes
        if item.get("node_id") and _clean_text(item.get("label"))
    }
    decoded_corpus = [
        *decoded.objects,
        *decoded.sequence,
        *[item.source for item in decoded.relations],
        *[item.target for item in decoded.relations],
    ]
    missing_labels = [
        label
        for label in node_labels.values()
        if not any(_text_match(label, candidate) >= 0.7 for candidate in decoded_corpus)
    ]
    object_coverage = 1.0 - len(missing_labels) / max(len(node_labels), 1)
    missing_relation_ids: list[str] = []
    relation_scores: list[float] = []
    for relation in relations:
        source = node_labels.get(str(relation.get("source_id") or ""), "")
        target = node_labels.get(str(relation.get("target_id") or ""), "")
        expected_type = str(relation.get("relation_type") or "")
        best = max(
            (
                _decoded_relation_score(
                    candidate,
                    source=source,
                    relation_type=expected_type,
                    target=target,
                )
                for candidate in decoded.relations
            ),
            default=0.0,
        )
        relation_scores.append(best)
        if best < 0.62:
            missing_relation_ids.append(str(relation.get("relation_id") or ""))
    relation_coverage = (
        sum(relation_scores) / len(relation_scores)
        if relation_scores
        else 1.0
    )
    expected_sequence = [
        node_labels.get(str(node_id), "")
        for node_id in graph.get("sequence_node_ids") or []
        if node_labels.get(str(node_id), "")
    ]
    sequence_score = _sequence_coverage(expected_sequence, decoded.sequence)
    thesis_score = max(
        _text_match(decoded.thesis, production_contract.get("thesis")),
        _text_match(decoded.thesis, production_contract.get("takeaway")),
    )
    score = (
        thesis_score * 0.24
        + object_coverage * 0.24
        + relation_coverage * 0.37
        + sequence_score * 0.15
    )
    counterfactual = _counterfactual_sensitivity(
        production_contract=production_contract,
        original=decoded,
        relation_ablation=relation_ablation_decode,
        temporal_scramble=temporal_scramble_decode,
        require_counterfactuals=require_counterfactuals,
    )
    if counterfactual.score is not None:
        score = score * 0.88 + counterfactual.score * 0.12
    issues: list[str] = []
    if thesis_score < 0.36:
        issues.append("blind_decoder_could_not_recover_visual_thesis")
    if object_coverage < 0.68:
        issues.append("blind_decoder_missed_grounded_objects")
    if relations and relation_coverage < 0.62:
        issues.append("blind_decoder_could_not_recover_required_relations")
    if expected_sequence and sequence_score < 0.62:
        issues.append("blind_decoder_could_not_recover_visual_sequence")
    if decoded.unsupported_visual_claims:
        issues.append("blind_decoder_found_unsupported_visual_claims")
    issues.extend(counterfactual.issues)
    score = max(0.0, min(score, 1.0))
    passed = (
        score >= min_score
        and not issues
        and (counterfactual.passed is not False)
    )
    directives: list[str] = []
    if missing_labels:
        directives.append(
            "Make these grounded objects visually identifiable without transcript help: "
            + "; ".join(missing_labels[:6])
        )
    if missing_relation_ids:
        directives.append(
            "Rebuild the visual grammar so these required relations are directly decodable: "
            + "; ".join(missing_relation_ids[:6])
        )
    if counterfactual.passed is False:
        directives.append(
            "Make proof-bearing geometry causally necessary; removing connectors, gates, or ordering must reduce comprehension."
        )
    return InverseDecodeEvaluation(
        passed=passed,
        score=score,
        thesis_score=thesis_score,
        object_coverage=object_coverage,
        relation_coverage=relation_coverage,
        sequence_score=sequence_score,
        missing_labels=_unique(missing_labels),
        missing_relation_ids=_unique(missing_relation_ids),
        issues=_unique(issues),
        repair_directives=_unique(directives),
        counterfactual=counterfactual,
    )


def build_counterfactual_frames(
    frame_paths: list[Path],
    output_dir: Path,
    *,
    encoding_family: str,
) -> tuple[list[Path], list[Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ablated: list[Path] = []
    for index, path in enumerate(frame_paths, start=1):
        if not Path(path).is_file():
            continue
        image = iio.imread(path)
        masked = _ablate_relation_region(image, encoding_family)
        target = output_dir / f"relation_ablation_{index:02d}.png"
        iio.imwrite(target, masked)
        ablated.append(target)
    if len(frame_paths) <= 1:
        scrambled = list(frame_paths)
    else:
        order = [len(frame_paths) - 1, *range(1, len(frame_paths) - 1), 0]
        scrambled = [frame_paths[index] for index in order]
    return ablated, scrambled


def _counterfactual_sensitivity(
    *,
    production_contract: dict[str, Any],
    original: BlindFrameDecode,
    relation_ablation: BlindFrameDecode | None,
    temporal_scramble: BlindFrameDecode | None,
    require_counterfactuals: bool,
) -> CounterfactualSensitivity:
    graph = dict(production_contract.get("visual_claim_graph") or {})
    relations = [
        item
        for item in graph.get("relations") or []
        if isinstance(item, dict) and item.get("required", True)
    ]
    sequence_required = len(graph.get("sequence_node_ids") or []) >= 3
    if not require_counterfactuals:
        return CounterfactualSensitivity(False, None, None, None, None, [])
    if relation_ablation is None or temporal_scramble is None:
        return CounterfactualSensitivity(
            True,
            False,
            None,
            None,
            0.0,
            ["blind_decoder_counterfactuals_unavailable"],
        )
    original_relation_strength = _decode_relation_strength(original)
    ablated_relation_strength = _decode_relation_strength(relation_ablation)
    relation_delta = max(0.0, original_relation_strength - ablated_relation_strength)
    original_sequence_strength = _decode_sequence_strength(original)
    scrambled_sequence_strength = _decode_sequence_strength(temporal_scramble)
    temporal_delta = max(0.0, original_sequence_strength - scrambled_sequence_strength)
    required_scores: list[float] = []
    issues: list[str] = []
    if relations:
        required_scores.append(min(relation_delta / 0.08, 1.0))
        if relation_delta < 0.025:
            issues.append("visual_claim_survives_relation_ablation")
    if sequence_required:
        required_scores.append(min(temporal_delta / 0.08, 1.0))
        if temporal_delta < 0.02:
            issues.append("visual_claim_survives_temporal_scramble")
    score = sum(required_scores) / len(required_scores) if required_scores else 1.0
    return CounterfactualSensitivity(
        enabled=True,
        passed=not issues,
        relation_ablation_delta=relation_delta if relations else None,
        temporal_scramble_delta=temporal_delta if sequence_required else None,
        score=score,
        issues=issues,
    )


def _decoded_relation_score(
    candidate: DecodedRelation,
    *,
    source: str,
    relation_type: str,
    target: str,
) -> float:
    source_score = _text_match(candidate.source, source)
    target_score = _text_match(candidate.target, target)
    if source_score < 0.55 or target_score < 0.55:
        return 0.0
    type_score = 1.0 if candidate.relation_type == relation_type else 0.0
    if not type_score and _relation_family(candidate.relation_type) == _relation_family(relation_type):
        type_score = 0.72
    return (
        source_score * 0.27
        + target_score * 0.27
        + type_score * 0.31
        + candidate.confidence * 0.15
    )


def _relation_family(relation_type: str) -> str:
    if relation_type in {"causes", "changes_after", "enables", "produces"}:
        return "causal"
    if relation_type in {"precedes", "routes_to", "leads_to"}:
        return "sequence"
    if relation_type in {"branches_to_high", "branches_to_low"}:
        return "branch"
    if relation_type in {"transforms_to", "preserves"}:
        return "transform"
    if relation_type in {"activates", "contains_action"}:
        return "interaction"
    return relation_type


def _sequence_coverage(expected: list[str], decoded: list[str]) -> float:
    if not expected:
        return 1.0
    if not decoded:
        return 0.0
    matched_indices: list[int] = []
    start = 0
    for expected_label in expected:
        best_index = -1
        best_score = 0.0
        for index in range(start, len(decoded)):
            score = _text_match(expected_label, decoded[index])
            if score > best_score:
                best_index = index
                best_score = score
        if best_index >= 0 and best_score >= 0.62:
            matched_indices.append(best_index)
            start = best_index + 1
    return len(matched_indices) / len(expected)


def _decode_relation_strength(decoded: BlindFrameDecode) -> float:
    if not decoded.relations:
        return decoded.confidence * 0.15
    relation_confidence = sum(item.confidence for item in decoded.relations) / len(
        decoded.relations
    )
    return min(1.0, relation_confidence * 0.7 + decoded.confidence * 0.3)


def _decode_sequence_strength(decoded: BlindFrameDecode) -> float:
    if len(decoded.sequence) < 2:
        return decoded.confidence * 0.1
    return min(1.0, decoded.confidence * 0.65 + min(len(decoded.sequence) / 5, 1.0) * 0.35)


def _ablate_relation_region(image: np.ndarray, encoding_family: str) -> np.ndarray:
    result = np.array(image, copy=True)
    if result.ndim < 3:
        return result
    height, width = result.shape[:2]
    corner_size = max(2, min(height, width) // 24)
    corners = np.concatenate(
        [
            result[:corner_size, :corner_size, :3].reshape(-1, 3),
            result[:corner_size, -corner_size:, :3].reshape(-1, 3),
            result[-corner_size:, :corner_size, :3].reshape(-1, 3),
            result[-corner_size:, -corner_size:, :3].reshape(-1, 3),
        ],
        axis=0,
    )
    fill = np.median(corners, axis=0).astype(result.dtype)
    encoding = _normalize(encoding_family).replace(" ", "_")
    if encoding == "linear_trace":
        y1, y2, x1, x2 = 0.42, 0.59, 0.07, 0.93
    elif encoding == "split_register":
        y1, y2, x1, x2 = 0.16, 0.9, 0.43, 0.57
    elif encoding == "layered_flow":
        y1, y2, x1, x2 = 0.3, 0.7, 0.34, 0.66
    else:
        y1, y2, x1, x2 = 0.27, 0.73, 0.32, 0.68
    result[
        int(height * y1) : max(int(height * y2), int(height * y1) + 1),
        int(width * x1) : max(int(width * x2), int(width * x1) + 1),
        :3,
    ] = fill
    return result


def _text_match(first: Any, second: Any) -> float:
    first_tokens = set(_normalized_tokens(first))
    second_tokens = set(_normalized_tokens(second))
    if not first_tokens or not second_tokens:
        return 0.0
    intersection = len(first_tokens & second_tokens)
    precision = intersection / len(first_tokens)
    recall = intersection / len(second_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _normalized_tokens(value: Any) -> list[str]:
    return [_stem_token(item) for item in _normalize(value).split()]


def _stem_token(token: str) -> str:
    token = token.strip("./-")
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _normalize(value: Any) -> str:
    return re.sub(r"[^a-z0-9%+./-]+", " ", str(value or "").lower()).strip()


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _string_list(value: Any, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    return _unique([_clean_text(item) for item in value if _clean_text(item)])[:limit]


def _bounded(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(number, 1.0))


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = _normalize(value)
        if key and key not in seen:
            seen.add(key)
            result.append(value)
    return result


__all__ = [
    "INVERSE_DECODER_VERSION",
    "BlindFrameDecode",
    "CounterfactualSensitivity",
    "DecodedRelation",
    "InverseDecodeEvaluation",
    "blind_decode_prompt",
    "build_counterfactual_frames",
    "evaluate_inverse_decode",
    "parse_blind_decode",
]
