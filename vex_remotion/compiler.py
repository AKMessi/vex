from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from visual_explanation import (
    GENERIC_LABELS,
    build_visual_explanation_ir,
    validate_visual_explanation_ir,
    visual_explanation_ir_signature,
)
from vex_visuals.creative_direction import compile_creative_direction


REMOTION_SCENE_PROGRAM_VERSION = "remotion-scene-program-v3"
FAMILY_ORDER = ("metric", "mechanism", "contrast", "timeline", "interface", "emphasis")
SCENE_FAMILY = {
    "metric_delta": "metric",
    "metric_intervention": "metric",
    "metric_proof": "metric",
    "architecture_flow": "mechanism",
    "causal_intervention": "mechanism",
    "guided_process": "mechanism",
    "set_partition": "mechanism",
    "matched_state_transform": "contrast",
    "decision_branch": "contrast",
    "narrative_progression": "timeline",
    "grounded_interface_walkthrough": "interface",
    "evidence_backed_quote": "emphasis",
}


@dataclass(frozen=True)
class RemotionNode:
    node_id: str
    label: str
    detail: str = ""
    value: str = ""
    role: str = "evidence"
    emphasis: float = 0.5
    fact_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RemotionEdge:
    edge_id: str
    source_id: str
    target_id: str
    relation: str
    provenance: str
    required: bool = True


@dataclass(frozen=True)
class RemotionBeat:
    beat_id: str
    phase: str
    action: str
    target_ids: list[str]
    start_fraction: float
    end_fraction: float


@dataclass(frozen=True)
class RemotionLayout:
    orientation: str
    variant: str
    density: str
    safe_margin: int
    title_lines: list[str]
    title_max_size: int
    content_columns: int
    candidate_scores: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class RemotionQualityContract:
    required_labels: list[str]
    required_edges: list[str]
    forbidden_content: list[str]
    final_hold_start: float
    min_occupancy: float
    max_occupancy: float
    min_contrast: float
    min_motion_delta: float
    min_motion_area: float


@dataclass(frozen=True)
class RemotionSceneProgram:
    version: str
    program_id: str
    signature: str
    visual_id: str
    scene_family: str
    scene_type: str
    narrative_strategy: str
    title: str
    takeaway: str
    eyebrow: str
    duration_sec: float
    fps: float
    width: int
    height: int
    style_pack: str
    theme: dict[str, str]
    nodes: list[RemotionNode]
    edges: list[RemotionEdge]
    annotations: list[str]
    beats: list[RemotionBeat]
    layout: RemotionLayout
    quality_contract: RemotionQualityContract
    creative_direction: dict[str, Any]
    evidence: list[dict[str, Any]]
    grounding_mode: str
    semantic_score: float
    compiler_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RemotionCompilationResult:
    passed: bool
    program: RemotionSceneProgram | None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    candidate_scores: list[dict[str, Any]] = field(default_factory=list)
    visual_explanation_validation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "program": self.program.to_dict() if self.program else None,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "candidate_scores": list(self.candidate_scores),
            "visual_explanation_validation": dict(self.visual_explanation_validation),
        }


def compile_remotion_scene_program(
    spec: dict[str, Any],
    *,
    width: int,
    height: int,
    fps: float,
) -> RemotionCompilationResult:
    normalized = dict(spec or {})
    visual_id = _clean(normalized.get("visual_id") or normalized.get("id") or "visual", 96, 12)
    source_grounded = bool(
        _clean(normalized.get("sentence_text"), 500, 80)
        or _clean(normalized.get("context_text"), 700, 110)
        or isinstance(normalized.get("visual_explanation_ir"), dict)
    )
    ir_payload = dict(normalized.get("visual_explanation_ir") or {})
    expected_ir_signature = str(
        (normalized.get("opportunity_contract") or {}).get(
            "visual_explanation_ir_signature"
        )
        or (normalized.get("opportunity_preflight") or {}).get(
            "visual_explanation_ir_signature"
        )
        or ""
    )
    if not ir_payload:
        ir_source = dict(normalized)
        if not source_grounded:
            compatibility_text = " ".join(
                part
                for part in [
                    _clean(normalized.get("headline"), 120, 12),
                    _clean(normalized.get("deck"), 220, 28),
                    " ".join(_string_list(normalized.get("steps"), limit=6)),
                    " ".join(_string_list(normalized.get("supporting_lines"), limit=6)),
                ]
                if part
            )
            ir_source["sentence_text"] = compatibility_text
        ir_payload = build_visual_explanation_ir(ir_source).to_dict()
    validation = validate_visual_explanation_ir(ir_payload)
    validation_payload = validation.to_dict()
    warnings = list(validation.warnings)
    errors: list[str] = []
    if (
        expected_ir_signature
        and visual_explanation_ir_signature(ir_payload) != expected_ir_signature
    ):
        errors.append("visual_explanation_ir_signature_mismatch")
    grounding_mode = "transcript_evidence" if source_grounded else "structured_input"
    if source_grounded:
        if str(ir_payload.get("render_policy") or "").strip().lower() != "render":
            errors.extend(
                str(item)
                for item in ir_payload.get("rejection_reasons")
                or ["visual_explanation_ir_rejected"]
            )
        if validation.invented_numeric_facts:
            errors.append("remotion_numeric_fact_lacks_source_provenance")
        if not validation.passed:
            errors.extend(validation.errors)

    nodes = _nodes_from_ir(ir_payload)
    if not source_grounded:
        structured_nodes = _nodes_from_structured_input(normalized)
        metric_nodes = [item for item in structured_nodes if item.value]
        nodes = _dedupe_nodes([*metric_nodes, *nodes, *structured_nodes])
    facts = [dict(item) for item in ir_payload.get("facts") or [] if isinstance(item, dict)]
    edges = _edges_from_ir(ir_payload, nodes)
    scene_type = str(ir_payload.get("scene_type") or "none").strip().lower()
    candidate_scores = _score_family_candidates(
        normalized,
        scene_type=scene_type,
        nodes=nodes,
        facts=facts,
        edges=edges,
    )
    viable = [item for item in candidate_scores if not item["hard_violations"]]
    if not viable:
        errors.append("no_semantically_supported_remotion_scene_family")
        family = "mechanism"
    else:
        family = str(viable[0]["family"])
    if family in {"mechanism", "timeline"} and len(nodes) >= 3:
        takeaway_key = _normalize(ir_payload.get("takeaway"))
        nodes = [
            item
            for item in nodes
            if not (
                item.role in {"result", "summary", "takeaway"}
                and takeaway_key
                and _normalize(item.detail or item.label) == takeaway_key
            )
        ]
    nodes = _limit_nodes(nodes, family=family, width=width, height=height)
    node_ids = {item.node_id for item in nodes}
    edges = [item for item in edges if item.source_id in node_ids and item.target_id in node_ids]
    if family in {"mechanism", "timeline"} and len(nodes) >= 2 and not edges:
        edges = [
            RemotionEdge(
                edge_id=f"layout_edge_{index + 1:02d}",
                source_id=source.node_id,
                target_id=target.node_id,
                relation="sequence",
                provenance="layout_sequence",
                required=False,
            )
            for index, (source, target) in enumerate(zip(nodes, nodes[1:]))
        ]
        warnings.append("sequence_edges_derived_from_grounded_object_order")

    title = _title_for(normalized, ir_payload, nodes, family=family)
    takeaway = _takeaway_for(normalized, ir_payload, title)
    annotations = _grounded_annotations(
        ir_payload,
        title=title,
        takeaway=takeaway,
        nodes=nodes,
    )
    if not title:
        errors.append("remotion_program_has_no_grounded_title")
    if not nodes:
        errors.append("remotion_program_has_no_grounded_nodes")
    generic_nodes = [item.label for item in nodes if _normalize(item.label) in GENERIC_LABELS]
    if generic_nodes:
        errors.append("remotion_program_contains_generic_node_labels")
    if family == "metric" and not any(item.value for item in nodes):
        errors.append("metric_scene_has_no_grounded_metric_value")
    if family in {"mechanism", "contrast", "timeline", "interface"} and len(nodes) < 2:
        errors.append(f"{family}_scene_has_too_few_grounded_nodes")

    layout = _select_layout(
        family=family,
        title=title,
        node_count=len(nodes),
        width=width,
        height=height,
    )
    beats = _beats_from_ir(ir_payload, nodes)
    if not beats:
        beats = _default_beats(nodes)
    required_labels = _unique([*[item.label for item in nodes], *annotations], limit=12)
    required_labels = [
        item for item in required_labels if _normalize(item) not in GENERIC_LABELS
    ]
    orientation = layout.orientation
    quality_contract = RemotionQualityContract(
        required_labels=required_labels,
        required_edges=[item.edge_id for item in edges if item.required],
        forbidden_content=_unique(
            [
                *[str(item) for item in ir_payload.get("forbidden_content") or []],
                "generic placeholder copy",
                "invented metrics",
            ],
            limit=12,
        ),
        final_hold_start=0.78,
        min_occupancy=0.07 if orientation == "landscape" else 0.055,
        max_occupancy=0.86,
        min_contrast=0.075,
        min_motion_delta=0.0035,
        min_motion_area=0.018,
    )
    creative_direction = compile_creative_direction(
        normalized,
        scene_type=scene_type,
        scene_family=family,
        objects=[asdict(item) for item in nodes],
        relations=[asdict(item) for item in edges],
        width=width,
        height=height,
        variant_index=int(normalized.get("variant_index") or 0),
    ).to_dict()
    semantic_score = _semantic_score(
        validation_payload,
        nodes=nodes,
        edges=edges,
        family=family,
        grounding_mode=grounding_mode,
    )
    if semantic_score < (0.62 if source_grounded else 0.52):
        errors.append("remotion_semantic_program_score_below_floor")
    errors = _unique(errors, limit=30)
    warnings = _unique(warnings, limit=20)
    if errors:
        return RemotionCompilationResult(
            passed=False,
            program=None,
            errors=errors,
            warnings=warnings,
            candidate_scores=candidate_scores,
            visual_explanation_validation=validation_payload,
        )

    duration = _duration(normalized)
    program_without_signature = {
        "version": REMOTION_SCENE_PROGRAM_VERSION,
        "visual_id": visual_id,
        "scene_family": family,
        "scene_type": scene_type,
        "title": title,
        "takeaway": takeaway,
        "nodes": [asdict(item) for item in nodes],
        "edges": [asdict(item) for item in edges],
        "annotations": annotations,
        "beats": [asdict(item) for item in beats],
        "layout": asdict(layout),
        "quality_contract": asdict(quality_contract),
        "creative_direction": creative_direction,
    }
    signature = hashlib.sha256(
        json.dumps(program_without_signature, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    program = RemotionSceneProgram(
        version=REMOTION_SCENE_PROGRAM_VERSION,
        program_id=f"remotion_{signature[:16]}",
        signature=signature,
        visual_id=visual_id,
        scene_family=family,
        scene_type=scene_type,
        narrative_strategy="author_driven_setup_evidence_synthesis",
        title=title,
        takeaway=takeaway,
        eyebrow=_eyebrow(scene_type, family),
        duration_sec=duration,
        fps=max(15.0, min(float(fps or 30.0), 120.0)),
        width=max(320, int(width)),
        height=max(240, int(height)),
        style_pack=str(normalized.get("style_pack") or "editorial_clean").strip().lower(),
        theme={str(key): str(value) for key, value in dict(normalized.get("theme") or {}).items()},
        nodes=nodes,
        edges=edges,
        annotations=annotations,
        beats=beats,
        layout=layout,
        quality_contract=quality_contract,
        creative_direction=creative_direction,
        evidence=[dict(item) for item in ir_payload.get("evidence") or [] if isinstance(item, dict)],
        grounding_mode=grounding_mode,
        semantic_score=semantic_score,
        compiler_warnings=warnings,
    )
    return RemotionCompilationResult(
        passed=True,
        program=program,
        errors=[],
        warnings=warnings,
        candidate_scores=candidate_scores,
        visual_explanation_validation=validation_payload,
    )


def _nodes_from_ir(ir: dict[str, Any]) -> list[RemotionNode]:
    facts = {
        str(item.get("fact_id") or ""): dict(item)
        for item in ir.get("facts") or []
        if isinstance(item, dict) and str(item.get("fact_id") or "")
    }
    result: list[RemotionNode] = []
    for index, item in enumerate(ir.get("objects") or []):
        if not isinstance(item, dict):
            continue
        label = _clean(item.get("label"), 46, 6)
        if not label:
            continue
        fact_ids = [str(value) for value in item.get("fact_ids") or [] if str(value) in facts]
        metric = next(
            (facts[value] for value in fact_ids if str(facts[value].get("fact_type") or "") == "metric"),
            {},
        )
        value = _metric_display_value(metric)
        result.append(
            RemotionNode(
                node_id=_identifier(item.get("object_id") or f"node_{index + 1:02d}"),
                label=label,
                detail=_clean(item.get("meaning"), 100, 14),
                value=value,
                role=_clean(item.get("role") or "evidence", 24, 3).lower(),
                emphasis=max(0.0, min(_as_float(item.get("emphasis"), 0.5), 1.0)),
                fact_ids=fact_ids,
            )
        )
    return _dedupe_nodes(result)


def _nodes_from_structured_input(spec: dict[str, Any]) -> list[RemotionNode]:
    nodes: list[RemotionNode] = []
    for index, item in enumerate(spec.get("metric_facts") or []):
        if not isinstance(item, dict):
            continue
        value = _clean(item.get("value"), 24, 3)
        label = _clean(item.get("label") or value, 46, 6)
        if label:
            nodes.append(RemotionNode(f"metric_{index + 1:02d}", label, value=value, role="metric", emphasis=0.9))
    source = (
        _string_list(spec.get("steps"), limit=6)
        or _string_list(spec.get("supporting_lines"), limit=6)
        or _string_list(spec.get("keywords"), limit=6)
    )
    for index, label in enumerate(source):
        cleaned = _clean(label, 46, 6)
        if cleaned:
            nodes.append(RemotionNode(f"node_{index + 1:02d}", cleaned, role="step", emphasis=0.65))
    return _dedupe_nodes(nodes)


def _edges_from_ir(ir: dict[str, Any], nodes: list[RemotionNode]) -> list[RemotionEdge]:
    node_ids = {item.node_id for item in nodes}
    result: list[RemotionEdge] = []
    for index, item in enumerate(ir.get("relations") or []):
        if not isinstance(item, dict):
            continue
        source = _identifier(item.get("source_id"))
        target = _identifier(item.get("target_id"))
        if source not in node_ids or target not in node_ids or source == target:
            continue
        result.append(
            RemotionEdge(
                edge_id=_identifier(item.get("relation_id") or f"edge_{index + 1:02d}"),
                source_id=source,
                target_id=target,
                relation=_clean(item.get("relation_type") or "related", 28, 3).lower(),
                provenance=_clean(item.get("provenance") or "transcript_relation", 40, 4),
                required=bool(item.get("required", True)),
            )
        )
    return result


def _score_family_candidates(
    spec: dict[str, Any],
    *,
    scene_type: str,
    nodes: list[RemotionNode],
    facts: list[dict[str, Any]],
    edges: list[RemotionEdge],
) -> list[dict[str, Any]]:
    mapped = SCENE_FAMILY.get(scene_type)
    template_value = f"{spec.get('template', '')} {spec.get('visual_intent_type', '')}".lower()
    metric_count = sum(1 for item in facts if str(item.get("fact_type") or "") == "metric")
    explicit_metric_count = sum(
        1 for item in spec.get("metric_facts") or [] if isinstance(item, dict) and item.get("value")
    )
    scores: list[dict[str, Any]] = []
    for ordinal, family in enumerate(FAMILY_ORDER):
        score = 0.25 - ordinal * 0.002
        reasons: list[str] = []
        hard: list[str] = []
        if family == mapped:
            score += 0.58
            reasons.append("visual_explanation_scene_match")
        patterns = {
            "metric": r"data|metric|score|risk|proof|stat",
            "mechanism": r"mechanism|architecture|causal|flow|process|system",
            "contrast": r"compare|contrast|decision|myth|problem|transform",
            "timeline": r"timeline|sequence|narrative|route|journey",
            "interface": r"interface|\bui\b|product",
            "emphasis": r"quote|keyword|focus|emphasis",
        }
        if re.search(patterns[family], template_value):
            score += 0.16
            reasons.append("template_intent_match")
        if family == "metric" and explicit_metric_count:
            score += 0.7
            reasons.append("explicit_structured_metric_evidence")
        if family == "metric" and metric_count == 0 and not any(item.value for item in nodes):
            hard.append("no_grounded_metric")
        if family in {"mechanism", "contrast", "timeline", "interface"} and len(nodes) < 2:
            hard.append("insufficient_nodes")
        if family == "emphasis" and not nodes:
            hard.append("no_emphasis_evidence")
        if family == "mechanism" and edges:
            score += 0.08
            reasons.append("evidence_backed_relations")
        if family in {"timeline", "mechanism"} and len(nodes) >= 3:
            score += 0.05
            reasons.append("multi_stage_structure")
        scores.append(
            {
                "family": family,
                "score": round(score, 4),
                "reasons": reasons,
                "hard_violations": hard,
            }
        )
    return sorted(scores, key=lambda item: (-float(item["score"]), FAMILY_ORDER.index(str(item["family"]))))


def _select_layout(*, family: str, title: str, node_count: int, width: int, height: int) -> RemotionLayout:
    aspect = max(width, 1) / max(height, 1)
    orientation = "portrait" if aspect < 0.82 else "landscape" if aspect > 1.22 else "square"
    variants = {
        "landscape": {
            "metric": ["split_evidence", "centered_focus"],
            "mechanism": ["horizontal_flow", "stacked_flow"],
            "contrast": ["balanced_split", "stacked_states"],
            "timeline": ["horizontal_timeline", "stacked_timeline"],
            "interface": ["narrative_with_surface", "stacked_surface"],
            "emphasis": ["centered_quote", "editorial_quote"],
        },
        "portrait": {
            "metric": ["stacked_metric", "centered_focus"],
            "mechanism": ["vertical_flow", "stacked_flow"],
            "contrast": ["stacked_states", "balanced_split"],
            "timeline": ["vertical_timeline", "stacked_timeline"],
            "interface": ["stacked_surface", "narrative_with_surface"],
            "emphasis": ["editorial_quote", "centered_quote"],
        },
        "square": {
            "metric": ["centered_focus", "split_evidence"],
            "mechanism": ["stacked_flow", "horizontal_flow"],
            "contrast": ["balanced_split", "stacked_states"],
            "timeline": ["stacked_timeline", "horizontal_timeline"],
            "interface": ["stacked_surface", "narrative_with_surface"],
            "emphasis": ["centered_quote", "editorial_quote"],
        },
    }[orientation][family]
    candidates: list[dict[str, Any]] = []
    for index, variant in enumerate(variants):
        score = 0.8 - index * 0.08
        violations: list[str] = []
        if node_count > 4 and "horizontal" in variant:
            score -= 0.18
            violations.append("horizontal_density_penalty")
        if len(title) > 52 and variant in {"split_evidence", "narrative_with_surface"}:
            score -= 0.1
            violations.append("long_title_split_penalty")
        candidates.append({"variant": variant, "score": round(score, 4), "soft_violations": violations})
    candidates.sort(key=lambda item: -float(item["score"]))
    density = "dense" if node_count >= 5 else "balanced" if node_count >= 3 else "focused"
    title_width = 28 if orientation == "portrait" else 42 if orientation == "landscape" else 34
    return RemotionLayout(
        orientation=orientation,
        variant=str(candidates[0]["variant"]),
        density=density,
        safe_margin=48 if orientation == "landscape" else 42,
        title_lines=_balanced_lines(title, max_chars=title_width, max_lines=2),
        title_max_size=78 if orientation == "landscape" else 62 if orientation == "portrait" else 68,
        content_columns=1 if orientation == "portrait" else min(max(node_count, 1), 4),
        candidate_scores=candidates,
    )


def _beats_from_ir(ir: dict[str, Any], nodes: list[RemotionNode]) -> list[RemotionBeat]:
    node_ids = {item.node_id for item in nodes}
    result: list[RemotionBeat] = []
    for index, item in enumerate(ir.get("beats") or []):
        if not isinstance(item, dict):
            continue
        targets = [
            value
            for value in [_identifier(item.get("subject_id")), _identifier(item.get("target_id"))]
            if value in node_ids
        ]
        if not targets:
            continue
        start = max(0.04, min(_as_float(item.get("start_fraction"), 0.1), 0.72))
        end = max(start + 0.08, min(_as_float(item.get("end_fraction"), start + 0.2), 0.78))
        result.append(
            RemotionBeat(
                beat_id=_identifier(item.get("beat_id") or f"beat_{index + 1:02d}"),
                phase=_clean(item.get("phase") or "evidence", 22, 2),
                action=_clean(item.get("action") or "reveal", 28, 3),
                target_ids=_unique(targets, limit=2),
                start_fraction=round(start, 4),
                end_fraction=round(end, 4),
            )
        )
    return result


def _default_beats(nodes: list[RemotionNode]) -> list[RemotionBeat]:
    count = max(len(nodes), 1)
    return [
        RemotionBeat(
            beat_id=f"beat_{index + 1:02d}",
            phase="evidence",
            action="reveal",
            target_ids=[node.node_id],
            start_fraction=round(0.1 + index * min(0.13, 0.5 / count), 4),
            end_fraction=round(min(0.7, 0.28 + index * min(0.13, 0.5 / count)), 4),
        )
        for index, node in enumerate(nodes)
    ]


def _semantic_score(
    validation: dict[str, Any],
    *,
    nodes: list[RemotionNode],
    edges: list[RemotionEdge],
    family: str,
    grounding_mode: str,
) -> float:
    grounded = _as_float(validation.get("grounded_fact_ratio"), 0.55 if grounding_mode == "structured_input" else 0.0)
    structure = min(len(nodes) / (2.0 if family != "emphasis" else 1.0), 1.0)
    relation = 1.0 if edges else 0.65 if family in {"metric", "emphasis", "interface"} else 0.35
    provenance = 1.0 if grounding_mode == "transcript_evidence" else 0.62
    score = grounded * 0.4 + structure * 0.24 + relation * 0.18 + provenance * 0.18
    return round(max(0.0, min(score, 1.0)), 4)


def _limit_nodes(nodes: list[RemotionNode], *, family: str, width: int, height: int) -> list[RemotionNode]:
    orientation = "portrait" if width / max(height, 1) < 0.82 else "landscape"
    limits = {
        "metric": 4,
        "mechanism": 4 if orientation == "landscape" else 5,
        "contrast": 2,
        "timeline": 5,
        "interface": 4,
        "emphasis": 4,
    }
    limit = limits[family]
    if family == "metric":
        selected = sorted(nodes, key=lambda item: -item.emphasis)[:limit]
        selected_ids = {item.node_id for item in selected}
        return [item for item in nodes if item.node_id in selected_ids]
    return nodes[:limit]


def _title_for(
    spec: dict[str, Any],
    ir: dict[str, Any],
    nodes: list[RemotionNode],
    *,
    family: str,
) -> str:
    semantic_frame = dict(spec.get("semantic_frame") or {})
    display_title = _clean((ir.get("metadata") or {}).get("display_title"), 72, 10)
    if display_title:
        return display_title
    if family == "emphasis":
        exact = semantic_frame.get("exact_quote") or spec.get("quote_text") or spec.get("sentence_text")
        if exact:
            return _clean(exact, 96, 16)
    return _clean(
        ir.get("thesis") or spec.get("headline") or spec.get("emphasis_text") or (nodes[0].label if nodes else ""),
        72,
        10,
    )


def _takeaway_for(spec: dict[str, Any], ir: dict[str, Any], title: str) -> str:
    value = _clean(ir.get("takeaway") or spec.get("deck") or spec.get("footer_text"), 128, 18)
    return "" if _normalize(value) == _normalize(title) else value


def _grounded_annotations(
    ir: dict[str, Any],
    *,
    title: str,
    takeaway: str,
    nodes: list[RemotionNode],
) -> list[str]:
    visible_text = _normalize(
        " ".join(
            [
                title,
                takeaway,
                *[
                    " ".join([item.label, item.detail, item.value])
                    for item in nodes
                ],
            ]
        )
    )
    annotations: list[str] = []
    for value in ir.get("required_labels") or []:
        cleaned = _clean(value, 56, 7)
        key = _normalize(cleaned)
        if not key or key in visible_text:
            continue
        annotations.append(cleaned)
    return _unique(annotations, limit=3)


def _eyebrow(scene_type: str, family: str) -> str:
    names = {
        "metric": "Evidence",
        "mechanism": "How it works",
        "contrast": "State change",
        "timeline": "Progression",
        "interface": "Interface",
        "emphasis": "Key point",
    }
    return names.get(family, scene_type.replace("_", " ").title())


def _duration(spec: dict[str, Any]) -> float:
    value = _as_float(spec.get("duration"), 0.0)
    if value <= 0:
        value = _as_float(spec.get("end"), 0.0) - _as_float(spec.get("start"), 0.0)
    return round(max(0.8, min(value or 3.0, 30.0)), 3)


def _balanced_lines(value: str, *, max_chars: int, max_lines: int) -> list[str]:
    words = value.split()
    if not words:
        return []
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        proposal = " ".join([*current, word])
        if current and len(proposal) > max_chars and len(lines) < max_lines - 1:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines[:max_lines]


def _dedupe_nodes(nodes: list[RemotionNode]) -> list[RemotionNode]:
    seen: set[str] = set()
    result: list[RemotionNode] = []
    for node in nodes:
        key = _normalize(node.label)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(node)
    return result


def _string_list(value: object, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_clean(item, 100, 14) for item in value if _clean(item, 100, 14)][:limit]


def _metric_display_value(metric: dict[str, Any]) -> str:
    value = _clean(metric.get("value"), 20, 2)
    unit = _clean(metric.get("unit"), 8, 1)
    if not value or not unit:
        return value or unit
    value_key = value.casefold()
    unit_key = unit.casefold()
    if unit_key == "%" and (value_key.endswith("%") or "percent" in value_key):
        return value
    if _normalize(unit) and _normalize(unit) in _normalize(value).split():
        return value
    return _clean(f"{value} {unit}", 24, 3)


def _clean(value: object, max_chars: int, max_words: int) -> str:
    words = re.sub(r"\s+", " ", str(value or "")).strip().split()
    words = words[:max_words]
    while words and len(" ".join(words)) > max_chars:
        words.pop()
    return " ".join(words)


def _identifier(value: object) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value or "")).strip("_")[:64]
    return cleaned or "item"


def _normalize(value: object) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").lower()))


def _unique(values: list[str], *, limit: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = _clean(value, 120, 18)
        key = _normalize(cleaned)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
        if len(result) >= limit:
            break
    return result


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
