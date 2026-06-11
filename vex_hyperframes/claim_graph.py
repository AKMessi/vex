from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from visual_explanation import VisualExplanationIR


CLAIM_GRAPH_VERSION = "visual-claim-graph-v1"
RELATION_TYPES = {
    "activates",
    "branches_to_high",
    "branches_to_low",
    "causes",
    "changes_after",
    "contains_action",
    "enables",
    "leads_to",
    "precedes",
    "preserves",
    "produces",
    "routes_to",
    "supports",
    "transforms_to",
}
RELATION_REQUIRED_SCENES = {
    "architecture_flow",
    "causal_intervention",
    "decision_branch",
    "grounded_interface_walkthrough",
    "guided_process",
    "matched_state_transform",
    "metric_delta",
    "metric_intervention",
    "narrative_progression",
}


@dataclass(frozen=True)
class VisualClaimNode:
    node_id: str
    label: str
    role: str
    meaning: str
    fact_ids: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    sequence_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VisualClaimRelation:
    relation_id: str
    source_id: str
    relation_type: str
    target_id: str
    evidence_ids: list[str] = field(default_factory=list)
    sequence_index: int | None = None
    required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VisualProofQuestion:
    question_id: str
    prompt: str
    answer_type: str
    expected_answer: str
    expected_node_ids: list[str] = field(default_factory=list)
    expected_relation_ids: list[str] = field(default_factory=list)
    required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VisualClaimGraph:
    version: str
    visual_id: str
    scene_type: str
    nodes: list[VisualClaimNode]
    relations: list[VisualClaimRelation]
    questions: list[VisualProofQuestion]
    sequence_node_ids: list[str]
    takeaway: str
    graph_signature: str

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "nodes": [item.to_dict() for item in self.nodes],
            "relations": [item.to_dict() for item in self.relations],
            "questions": [item.to_dict() for item in self.questions],
        }


@dataclass(frozen=True)
class VisualClaimGraphValidation:
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    node_count: int = 0
    relation_count: int = 0
    required_relation_count: int = 0
    relation_node_coverage: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["relation_node_coverage"] = round(
            float(self.relation_node_coverage),
            4,
        )
        return payload


def build_visual_claim_graph(ir: VisualExplanationIR) -> VisualClaimGraph:
    facts = {item.fact_id: item for item in ir.facts}
    canonical_objects = _canonical_objects(ir)
    nodes = [
        VisualClaimNode(
            node_id=item.object_id,
            label=item.label,
            role=item.role,
            meaning=item.meaning,
            fact_ids=list(item.fact_ids),
            evidence_ids=_unique(
                [
                    evidence_id
                    for fact_id in item.fact_ids
                    if fact_id in facts
                    for evidence_id in facts[fact_id].evidence_ids
                ]
            ),
            sequence_index=index,
        )
        for index, item in enumerate(canonical_objects)
    ]
    relations = _relations_for_scene(ir.scene_type, nodes)
    questions = _questions_for_graph(
        scene_type=ir.scene_type,
        viewer_question=ir.viewer_question,
        takeaway=ir.takeaway,
        nodes=nodes,
        relations=relations,
    )
    sequence = [item.node_id for item in nodes]
    signature_payload = {
        "version": CLAIM_GRAPH_VERSION,
        "visual_id": ir.visual_id,
        "scene_type": ir.scene_type,
        "nodes": [item.to_dict() for item in nodes],
        "relations": [item.to_dict() for item in relations],
        "questions": [item.to_dict() for item in questions],
        "sequence_node_ids": sequence,
        "takeaway": ir.takeaway,
    }
    signature = hashlib.sha256(
        json.dumps(
            signature_payload,
            sort_keys=True,
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    return VisualClaimGraph(
        version=CLAIM_GRAPH_VERSION,
        visual_id=ir.visual_id,
        scene_type=ir.scene_type,
        nodes=nodes,
        relations=relations,
        questions=questions,
        sequence_node_ids=sequence,
        takeaway=ir.takeaway,
        graph_signature=signature,
    )


def validate_visual_claim_graph(
    graph: VisualClaimGraph | dict[str, Any],
) -> VisualClaimGraphValidation:
    payload = graph.to_dict() if isinstance(graph, VisualClaimGraph) else dict(graph)
    errors: list[str] = []
    warnings: list[str] = []
    nodes = [
        dict(item)
        for item in payload.get("nodes") or []
        if isinstance(item, dict)
    ]
    relations = [
        dict(item)
        for item in payload.get("relations") or []
        if isinstance(item, dict)
    ]
    questions = [
        dict(item)
        for item in payload.get("questions") or []
        if isinstance(item, dict)
    ]
    node_ids = {
        str(item.get("node_id") or "")
        for item in nodes
        if str(item.get("node_id") or "")
    }
    relation_ids = {
        str(item.get("relation_id") or "")
        for item in relations
        if str(item.get("relation_id") or "")
    }
    scene_type = str(payload.get("scene_type") or "")
    if str(payload.get("version") or "") != CLAIM_GRAPH_VERSION:
        errors.append("unsupported_visual_claim_graph_version")
    if len(node_ids) != len(nodes):
        errors.append("claim_graph_has_missing_or_duplicate_node_ids")
    if len(relation_ids) != len(relations):
        errors.append("claim_graph_has_missing_or_duplicate_relation_ids")
    for relation in relations:
        relation_id = str(relation.get("relation_id") or "")
        source_id = str(relation.get("source_id") or "")
        target_id = str(relation.get("target_id") or "")
        relation_type = str(relation.get("relation_type") or "")
        if source_id not in node_ids:
            errors.append(f"claim_relation_unknown_source:{relation_id}")
        if target_id not in node_ids:
            errors.append(f"claim_relation_unknown_target:{relation_id}")
        if source_id == target_id:
            errors.append(f"claim_relation_self_loop:{relation_id}")
        if relation_type not in RELATION_TYPES:
            errors.append(f"claim_relation_unsupported_type:{relation_id}")
        if not relation.get("evidence_ids"):
            errors.append(f"claim_relation_without_evidence:{relation_id}")
    for question in questions:
        question_id = str(question.get("question_id") or "")
        if not str(question.get("prompt") or "").strip():
            errors.append(f"proof_question_without_prompt:{question_id}")
        if not str(question.get("expected_answer") or "").strip():
            errors.append(f"proof_question_without_answer:{question_id}")
        expected_nodes = {
            str(item) for item in question.get("expected_node_ids") or []
        }
        expected_relations = {
            str(item) for item in question.get("expected_relation_ids") or []
        }
        if not expected_nodes.issubset(node_ids):
            errors.append(f"proof_question_unknown_node:{question_id}")
        if not expected_relations.issubset(relation_ids):
            errors.append(f"proof_question_unknown_relation:{question_id}")
    required_relations = [
        item for item in relations if bool(item.get("required", True))
    ]
    if scene_type in RELATION_REQUIRED_SCENES and not required_relations:
        errors.append("claim_graph_missing_required_relation")
    if nodes and not questions:
        errors.append("claim_graph_has_no_visual_questions")
    related_node_ids = {
        str(item.get(key) or "")
        for item in required_relations
        for key in ("source_id", "target_id")
        if str(item.get(key) or "")
    }
    coverage = len(related_node_ids) / max(len(node_ids), 1)
    if (
        scene_type in RELATION_REQUIRED_SCENES
        and len(nodes) >= 3
        and coverage < 0.66
    ):
        errors.append("claim_graph_relation_coverage_below_66_percent")
    elif nodes and coverage < 0.5 and required_relations:
        warnings.append("claim_graph_relation_coverage_is_sparse")
    expected_signature = _signature_for_payload(payload)
    if not payload.get("graph_signature"):
        errors.append("claim_graph_missing_signature")
    elif str(payload.get("graph_signature")) != expected_signature:
        errors.append("claim_graph_signature_mismatch")
    return VisualClaimGraphValidation(
        passed=not errors,
        errors=_unique(errors),
        warnings=_unique(warnings),
        node_count=len(nodes),
        relation_count=len(relations),
        required_relation_count=len(required_relations),
        relation_node_coverage=coverage,
    )


def visual_claim_graph_prompt_block(
    graph: VisualClaimGraph | dict[str, Any],
) -> str:
    payload = graph.to_dict() if isinstance(graph, VisualClaimGraph) else dict(graph)
    nodes = [
        f"- {item.get('node_id')}: {item.get('label')} ({item.get('role')})"
        for item in payload.get("nodes") or []
        if isinstance(item, dict)
    ]
    relations = [
        (
            f"- {item.get('relation_id')}: {item.get('source_id')} "
            f"{item.get('relation_type')} {item.get('target_id')}"
        )
        for item in payload.get("relations") or []
        if isinstance(item, dict)
    ]
    questions = [
        f"- {item.get('question_id')}: {item.get('prompt')}"
        for item in payload.get("questions") or []
        if isinstance(item, dict)
    ]
    return "\n".join(
        [
            "Visual Claim Graph:",
            f"- Version: {payload.get('version')}",
            f"- Scene type: {payload.get('scene_type')}",
            "- Nodes:",
            *nodes,
            "- Required relations:",
            *relations,
            "- Blind visual questions:",
            *questions,
            f"- Graph signature: {payload.get('graph_signature')}",
        ]
    )


def _relations_for_scene(
    scene_type: str,
    nodes: list[VisualClaimNode],
) -> list[VisualClaimRelation]:
    by_role: dict[str, list[VisualClaimNode]] = {}
    for node in nodes:
        by_role.setdefault(node.role, []).append(node)
    edges: list[tuple[VisualClaimNode, str, VisualClaimNode]] = []
    if scene_type == "metric_delta":
        metrics = by_role.get("metric", [])
        before = _first(by_role, "problem") or (metrics[0] if metrics else None)
        after = _best_metric_state(
            by_role.get("result", []),
            metrics[-1] if metrics else None,
        ) or (metrics[-1] if metrics else None)
        outcome = next(
            (
                item
                for item in by_role.get("result", [])
                if after is None or item.node_id != after.node_id
            ),
            None,
        )
        _append_edge(edges, before, "transforms_to", after)
        if metrics:
            _append_edge(edges, metrics[-1], "supports", after)
        _append_edge(edges, _first(by_role, "mechanism"), "causes", after)
        _append_edge(edges, after, "enables", outcome)
    elif scene_type == "metric_intervention":
        metrics = by_role.get("metric", [])
        intervention = _first(by_role, "intervention")
        if metrics and intervention:
            _append_edge(edges, metrics[0], "changes_after", intervention)
            if len(metrics) > 1:
                _append_edge(edges, intervention, "produces", metrics[-1])
        if metrics:
            _append_edge(
                edges,
                _first(by_role, "problem"),
                "supports",
                metrics[0],
            )
        _append_edge(edges, intervention, "produces", _first(by_role, "result"))
    elif scene_type == "metric_proof":
        metric = _first(by_role, "metric")
        support = _first(by_role, "mechanism") or _first(by_role, "required")
        _append_edge(edges, support, "supports", metric)
    elif scene_type == "causal_intervention":
        problem = _first(by_role, "problem")
        mechanism = _first(by_role, "mechanism")
        intervention = _first(by_role, "intervention")
        result = _first(by_role, "result")
        if mechanism:
            _append_edge(edges, problem, "causes", mechanism)
            if intervention:
                _append_edge(edges, mechanism, "activates", intervention)
                _append_edge(edges, intervention, "produces", result)
            else:
                _append_edge(edges, mechanism, "produces", result)
        else:
            _append_edge(edges, problem, "causes", result)
    elif scene_type in {"guided_process", "architecture_flow"}:
        relation_type = "routes_to" if scene_type == "architecture_flow" else "precedes"
        for source, target in zip(nodes, nodes[1:]):
            _append_edge(edges, source, relation_type, target)
    elif scene_type == "matched_state_transform":
        before = _first(by_role, "problem")
        after = _first(by_role, "result")
        constraint = _first(by_role, "constraint")
        _append_edge(edges, before, "transforms_to", after)
        _append_edge(edges, before, "preserves", constraint)
        _append_edge(edges, after, "preserves", constraint)
    elif scene_type == "grounded_interface_walkthrough":
        interface = _first(by_role, "interface")
        interventions = by_role.get("intervention", [])
        action = interventions[0] if interventions else None
        result = _first(by_role, "result")
        for item in interventions:
            _append_edge(edges, interface, "contains_action", item)
        _append_edge(edges, action, "produces", result)
    elif scene_type == "decision_branch":
        decision = _first(by_role, "decision")
        _append_edge(
            edges,
            decision,
            "branches_to_low",
            _first(by_role, "branch_low"),
        )
        _append_edge(
            edges,
            decision,
            "branches_to_high",
            _first(by_role, "branch_high"),
        )
    elif scene_type == "narrative_progression":
        setup = _first(by_role, "setup")
        turn = _first(by_role, "intervention")
        result = _first(by_role, "result")
        _append_edge(edges, setup, "leads_to", turn)
        _append_edge(edges, turn, "leads_to", result)
    relations: list[VisualClaimRelation] = []
    seen: set[tuple[str, str, str]] = set()
    for source, relation_type, target in edges:
        key = (source.node_id, relation_type, target.node_id)
        if key in seen or source.node_id == target.node_id:
            continue
        seen.add(key)
        relations.append(
            VisualClaimRelation(
                relation_id=f"relation_{len(relations) + 1:02d}",
                source_id=source.node_id,
                relation_type=relation_type,
                target_id=target.node_id,
                evidence_ids=_unique(
                    [*source.evidence_ids, *target.evidence_ids]
                ),
                sequence_index=len(relations),
            )
        )
    return relations


def _questions_for_graph(
    *,
    scene_type: str,
    viewer_question: str,
    takeaway: str,
    nodes: list[VisualClaimNode],
    relations: list[VisualClaimRelation],
) -> list[VisualProofQuestion]:
    node_by_id = {item.node_id: item for item in nodes}
    questions: list[VisualProofQuestion] = []
    for relation in relations[:5]:
        source = node_by_id[relation.source_id]
        target = node_by_id[relation.target_id]
        questions.append(
            VisualProofQuestion(
                question_id=f"question_{len(questions) + 1:02d}",
                prompt=_relation_question(
                    relation.relation_type,
                    source.label,
                ),
                answer_type="relation_target",
                expected_answer=target.label,
                expected_node_ids=[target.node_id],
                expected_relation_ids=[relation.relation_id],
            )
        )
    if not questions and nodes:
        questions.append(
            VisualProofQuestion(
                question_id="question_01",
                prompt=viewer_question
                or "What concrete claim does this visual communicate?",
                answer_type="takeaway",
                expected_answer=takeaway or nodes[-1].label,
                expected_node_ids=[nodes[-1].node_id],
            )
        )
    if (
        scene_type in {"guided_process", "architecture_flow", "narrative_progression"}
        and len(nodes) >= 3
    ):
        questions.append(
            VisualProofQuestion(
                question_id=f"question_{len(questions) + 1:02d}",
                prompt="What is the order of the visible process from first to last?",
                answer_type="ordered_sequence",
                expected_answer=" -> ".join(item.label for item in nodes),
                expected_node_ids=[item.node_id for item in nodes],
                expected_relation_ids=[item.relation_id for item in relations],
            )
        )
    return questions[:6]


def _relation_question(relation_type: str, source_label: str) -> str:
    prompts = {
        "activates": f"What does {source_label} activate?",
        "branches_to_high": f"What high-confidence outcome follows {source_label}?",
        "branches_to_low": f"What low-confidence outcome follows {source_label}?",
        "causes": f"What does {source_label} cause?",
        "changes_after": f"What intervention changes {source_label}?",
        "contains_action": f"What action is shown inside {source_label}?",
        "enables": f"What does {source_label} enable?",
        "leads_to": f"What happens after {source_label}?",
        "precedes": f"What follows {source_label}?",
        "preserves": f"What does {source_label} preserve?",
        "produces": f"What result does {source_label} produce?",
        "routes_to": f"Where does {source_label} route next?",
        "supports": f"What claim is supported by {source_label}?",
        "transforms_to": f"What does {source_label} transform into?",
    }
    return prompts.get(
        relation_type,
        f"What relationship follows from {source_label}?",
    )


def _signature_for_payload(payload: dict[str, Any]) -> str:
    signature_payload = {
        "version": payload.get("version"),
        "visual_id": payload.get("visual_id"),
        "scene_type": payload.get("scene_type"),
        "nodes": payload.get("nodes") or [],
        "relations": payload.get("relations") or [],
        "questions": payload.get("questions") or [],
        "sequence_node_ids": payload.get("sequence_node_ids") or [],
        "takeaway": payload.get("takeaway") or "",
    }
    return hashlib.sha256(
        json.dumps(
            signature_payload,
            sort_keys=True,
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _first(
    by_role: dict[str, list[VisualClaimNode]],
    role: str,
) -> VisualClaimNode | None:
    values = by_role.get(role) or []
    return values[0] if values else None


def _canonical_objects(ir: VisualExplanationIR) -> list[Any]:
    result: list[Any] = []
    for item in ir.objects:
        if item.role == "required":
            continue
        if any(
            item.role == existing.role
            and _labels_overlap(item.label, existing.label)
            for existing in result
        ):
            continue
        result.append(item)
    for item in ir.objects:
        if item.role != "required":
            continue
        if any(_labels_overlap(item.label, existing.label) for existing in result):
            continue
        result.append(item)
    return result


def _labels_overlap(first: str, second: str) -> bool:
    first_tokens = set(_tokens(first))
    second_tokens = set(_tokens(second))
    if not first_tokens or not second_tokens:
        return False
    overlap = len(first_tokens & second_tokens)
    return (
        overlap / len(first_tokens) >= 0.6
        or overlap / len(second_tokens) >= 0.6
    )


def _tokens(value: str) -> list[str]:
    normalized = str(value or "").lower().replace("%", " percent ")
    return [
        token
        for token in re.findall(r"[a-z0-9+./-]+", normalized)
        if len(token) >= 2
    ]


def _best_metric_state(
    candidates: list[VisualClaimNode],
    metric: VisualClaimNode | None,
) -> VisualClaimNode | None:
    if not candidates:
        return None
    if metric is None:
        return candidates[-1]
    metric_tokens = set(_tokens(metric.label))
    return max(
        candidates,
        key=lambda item: (
            len(metric_tokens & set(_tokens(item.label))),
            -int(item.sequence_index or 0),
        ),
    )


def _append_edge(
    edges: list[tuple[VisualClaimNode, str, VisualClaimNode]],
    source: VisualClaimNode | None,
    relation_type: str,
    target: VisualClaimNode | None,
) -> None:
    if source is not None and target is not None and source.node_id != target.node_id:
        edges.append((source, relation_type, target))


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
    "CLAIM_GRAPH_VERSION",
    "RELATION_TYPES",
    "VisualClaimGraph",
    "VisualClaimGraphValidation",
    "VisualClaimNode",
    "VisualClaimRelation",
    "VisualProofQuestion",
    "build_visual_claim_graph",
    "validate_visual_claim_graph",
    "visual_claim_graph_prompt_block",
]
