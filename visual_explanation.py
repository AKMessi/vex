from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any


VISUAL_EXPLANATION_VERSION = "visual-explanation-v2"
SCENE_TYPES = {
    "architecture_flow",
    "causal_intervention",
    "decision_branch",
    "evidence_backed_quote",
    "grounded_interface_walkthrough",
    "guided_process",
    "matched_state_transform",
    "metric_delta",
    "metric_intervention",
    "metric_proof",
    "narrative_progression",
    "set_partition",
    "none",
}
GENERIC_LABELS = {
    "action",
    "better",
    "clear",
    "context",
    "core idea",
    "core loop",
    "decision",
    "focus",
    "hidden layer",
    "input",
    "mechanism",
    "outcome",
    "output",
    "proof",
    "result",
    "signal",
    "simple",
    "start",
    "system",
    "timing",
    "useful",
    "workflow",
}
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "when",
    "with",
    "you",
}
VERB_FRAGMENT_WORDS = {
    "add",
    "apply",
    "become",
    "break",
    "build",
    "choose",
    "compile",
    "create",
    "drop",
    "generate",
    "help",
    "keep",
    "make",
    "render",
    "route",
    "run",
    "turn",
}
NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9.])"
    r"(?P<number>\d+(?:\.\d+)?)"
    r"\s*(?P<unit>%|percent|x|ms|milliseconds?|s|sec|seconds?|kb|mb|gb|tb|k|m|b|tokens?|parameters?|users?)?"
    r"(?![A-Za-z0-9.])",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class EvidenceSpan:
    evidence_id: str
    source_type: str
    text: str
    start_sec: float | None = None
    end_sec: float | None = None
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GroundedFact:
    fact_id: str
    fact_type: str
    label: str
    subject: str = ""
    predicate: str = ""
    object: str = ""
    value: str = ""
    unit: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    grounding: str = "semantic_derived"
    confidence: float = 0.72

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExplanationObject:
    object_id: str
    role: str
    label: str
    meaning: str
    fact_ids: list[str] = field(default_factory=list)
    emphasis: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExplanationBeat:
    beat_id: str
    phase: str
    action: str
    subject_id: str
    target_id: str = ""
    start_fraction: float = 0.0
    end_fraction: float = 1.0
    fact_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExplanationRelation:
    relation_id: str
    source_id: str
    relation_type: str
    target_id: str
    evidence_ids: list[str] = field(default_factory=list)
    provenance: str = "transcript_relation"
    confidence: float = 0.8
    required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VisualExplanationIR:
    version: str
    visual_id: str
    scene_type: str
    render_policy: str
    viewer_question: str
    thesis: str
    takeaway: str
    duration_sec: float
    composition_mode: str
    evidence: list[EvidenceSpan] = field(default_factory=list)
    facts: list[GroundedFact] = field(default_factory=list)
    objects: list[ExplanationObject] = field(default_factory=list)
    relations: list[ExplanationRelation] = field(default_factory=list)
    beats: list[ExplanationBeat] = field(default_factory=list)
    required_labels: list[str] = field(default_factory=list)
    forbidden_content: list[str] = field(default_factory=list)
    rejection_reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "evidence": [item.to_dict() for item in self.evidence],
            "facts": [item.to_dict() for item in self.facts],
            "objects": [item.to_dict() for item in self.objects],
            "relations": [item.to_dict() for item in self.relations],
            "beats": [item.to_dict() for item in self.beats],
        }


@dataclass(frozen=True)
class VisualExplanationValidation:
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    grounded_fact_ratio: float = 0.0
    required_label_count: int = 0
    generic_label_count: int = 0
    invented_numeric_facts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["grounded_fact_ratio"] = round(float(self.grounded_fact_ratio), 4)
        return payload


def build_visual_explanation_ir(spec: dict[str, Any]) -> VisualExplanationIR:
    visual_id = str(spec.get("visual_id") or spec.get("id") or "visual").strip()
    duration = _as_float(spec.get("duration"), max(_as_float(spec.get("end")) - _as_float(spec.get("start")), 2.8))
    sentence_text = _clean(spec.get("sentence_text"), max_chars=420)
    context_text = _clean(spec.get("context_text"), max_chars=520)
    semantic_frame = dict(spec.get("semantic_frame") or {})
    evidence = _build_evidence(spec, sentence_text=sentence_text, context_text=context_text)
    source_text = " ".join(item.text for item in evidence if item.source_type != "semantic_frame")
    partition_model = _extract_partition_model(evidence, source_text)
    facts = (
        _partition_facts(partition_model)
        if partition_model.get("valid")
        else _build_facts(
            spec,
            semantic_frame=semantic_frame,
            evidence=evidence,
            source_text=source_text,
        )
    )
    scene_type = (
        "set_partition"
        if partition_model.get("valid")
        else _scene_type(spec, semantic_frame=semantic_frame, facts=facts)
    )
    objects = _objects_for_scene(scene_type, facts)
    executable_model = (
        partition_model
        if scene_type == "set_partition"
        else _build_executable_model(
            scene_type,
            facts=facts,
            evidence=evidence,
            source_text=source_text,
        )
    )
    relations = _relations_for_scene(
        scene_type,
        objects=objects,
        evidence=evidence,
        executable_model=executable_model,
    )
    beats = _beats_for_scene(scene_type, objects)
    requested_labels = _unique(
        [str(item) for item in (spec.get("required_labels") or [])],
        limit=12,
    )
    grounded_requested_labels = [
        label
        for label in requested_labels
        if _grounding_for_label(label, source_text)[0] != "unverified"
    ]
    unsupported_requested_labels = [
        label for label in requested_labels if label not in grounded_requested_labels
    ]
    required_labels = _unique(
        [*grounded_requested_labels, *[item.label for item in objects if item.label]],
        limit=12,
    )
    generic_labels = [label for label in required_labels if _normalize(label) in GENERIC_LABELS]
    rejection_reasons: list[str] = []
    if scene_type == "none":
        rejection_reasons.append("no_supported_explanatory_structure")
    if len(objects) < _minimum_object_count(scene_type):
        rejection_reasons.append("insufficient_grounded_objects")
    if len(beats) < _minimum_beat_count(scene_type):
        rejection_reasons.append("insufficient_motion_contract")
    if generic_labels:
        rejection_reasons.append("generic_placeholder_labels")
    if unsupported_requested_labels:
        rejection_reasons.append("required_label_lacks_source_provenance")
    if _contains_invented_numeric_fact(facts, source_text):
        rejection_reasons.append("numeric_fact_lacks_source_provenance")
    if any(_label_is_fragmented(item.label) for item in facts if item.grounding != "unverified"):
        rejection_reasons.append("fragmented_semantic_labels")
    if not bool(executable_model.get("valid")) and scene_type != "none":
        rejection_reasons.extend(
            str(item)
            for item in executable_model.get("issues") or ["no_executable_visual_model"]
        )
    if scene_type in RELATION_REQUIRED_SCENES and not relations:
        rejection_reasons.append("no_evidence_backed_relations")
    if not facts:
        rejection_reasons.append("no_grounded_facts")
    render_policy = "reject" if rejection_reasons else "render"
    if scene_type == "set_partition":
        thesis = str(executable_model.get("headline") or "")
        takeaway = str(executable_model.get("takeaway") or "")
    else:
        takeaway = _first_grounded_semantic_value(
            semantic_frame,
            source_text,
            "viewer_takeaway",
            "result",
            "effect",
            "after_state",
            "payoff",
            "exact_quote",
        ) or (objects[-1].meaning if objects else "")
        thesis = _first_grounded_semantic_value(
            semantic_frame,
            source_text,
            "thesis",
            "mechanism",
        ) or _grounded_display_copy(
            spec.get("headline"),
            source_text,
        ) or takeaway or (objects[0].meaning if objects else "")
    viewer_question = _viewer_question(scene_type, required_labels)
    forbidden_content = _unique(
        [
            *[str(item) for item in (spec.get("forbidden_content") or []) if str(item).strip()],
            "invented metrics",
            "generic placeholder copy",
            "unsupported entities",
        ],
        limit=12,
    )
    return VisualExplanationIR(
        version=VISUAL_EXPLANATION_VERSION,
        visual_id=visual_id,
        scene_type=scene_type,
        render_policy=render_policy,
        viewer_question=viewer_question,
        thesis=thesis,
        takeaway=takeaway,
        duration_sec=max(duration, 0.1),
        composition_mode=str(spec.get("composition_mode") or "replace").strip().lower(),
        evidence=evidence,
        facts=facts,
        objects=objects,
        relations=relations,
        beats=beats,
        required_labels=required_labels,
        forbidden_content=forbidden_content,
        rejection_reasons=_unique(rejection_reasons, limit=8),
        metadata={
            "template_requested": str(spec.get("template") or "").strip().lower(),
            "visual_type_hint": str(spec.get("visual_type_hint") or "").strip().lower(),
            "card_id": str(spec.get("card_id") or "").strip(),
            "start_sec": _as_float(spec.get("start"), 0.0),
            "end_sec": _as_float(spec.get("end"), 0.0),
            "unsupported_required_labels": unsupported_requested_labels,
            "executable_model": executable_model,
        },
    )


def validate_visual_explanation_ir(ir: VisualExplanationIR | dict[str, Any]) -> VisualExplanationValidation:
    payload = ir.to_dict() if isinstance(ir, VisualExplanationIR) else dict(ir or {})
    errors: list[str] = []
    warnings: list[str] = []
    scene_type = str(payload.get("scene_type") or "")
    render_policy = str(payload.get("render_policy") or "")
    facts = [item for item in (payload.get("facts") or []) if isinstance(item, dict)]
    objects = [item for item in (payload.get("objects") or []) if isinstance(item, dict)]
    relations = [item for item in (payload.get("relations") or []) if isinstance(item, dict)]
    beats = [item for item in (payload.get("beats") or []) if isinstance(item, dict)]
    required_labels = [str(item).strip() for item in (payload.get("required_labels") or []) if str(item).strip()]
    evidence_ids = {
        str(item.get("evidence_id") or "")
        for item in (payload.get("evidence") or [])
        if isinstance(item, dict) and str(item.get("evidence_id") or "")
    }
    fact_ids = {str(item.get("fact_id") or "") for item in facts if str(item.get("fact_id") or "")}
    object_ids = {str(item.get("object_id") or "") for item in objects if str(item.get("object_id") or "")}
    grounded_facts = 0
    invented_numbers: list[str] = []
    if str(payload.get("version") or "") != VISUAL_EXPLANATION_VERSION:
        errors.append("unsupported_visual_explanation_version")
    if scene_type not in SCENE_TYPES:
        errors.append("unsupported_scene_type")
    if render_policy not in {"render", "reject"}:
        errors.append("unsupported_render_policy")
    for fact in facts:
        references = [str(item) for item in (fact.get("evidence_ids") or []) if str(item)]
        if references and set(references).issubset(evidence_ids):
            grounded_facts += 1
        else:
            errors.append(f"fact_without_valid_evidence:{fact.get('fact_id')}")
        if str(fact.get("fact_type") or "") == "metric" and str(fact.get("grounding") or "") == "unverified":
            invented_numbers.append(str(fact.get("value") or fact.get("label") or "unknown"))
    for obj in objects:
        references = [str(item) for item in (obj.get("fact_ids") or []) if str(item)]
        if not references or not set(references).issubset(fact_ids):
            errors.append(f"object_without_valid_fact:{obj.get('object_id')}")
    for relation in relations:
        relation_id = str(relation.get("relation_id") or "")
        source_id = str(relation.get("source_id") or "")
        target_id = str(relation.get("target_id") or "")
        references = [str(item) for item in (relation.get("evidence_ids") or []) if str(item)]
        if source_id not in object_ids:
            errors.append(f"relation_unknown_source:{relation_id}")
        if target_id not in object_ids:
            errors.append(f"relation_unknown_target:{relation_id}")
        if source_id == target_id:
            errors.append(f"relation_self_loop:{relation_id}")
        if not references or not set(references).issubset(evidence_ids):
            errors.append(f"relation_without_valid_evidence:{relation_id}")
    for beat in beats:
        subject = str(beat.get("subject_id") or "")
        target = str(beat.get("target_id") or "")
        if subject not in object_ids:
            errors.append(f"beat_unknown_subject:{beat.get('beat_id')}")
        if target and target not in object_ids:
            errors.append(f"beat_unknown_target:{beat.get('beat_id')}")
        start_fraction = _as_float(beat.get("start_fraction"), -1.0)
        end_fraction = _as_float(beat.get("end_fraction"), -1.0)
        if start_fraction < 0.0 or end_fraction > 1.0 or end_fraction <= start_fraction:
            errors.append(f"beat_invalid_timing:{beat.get('beat_id')}")
    generic_count = sum(1 for label in required_labels if _normalize(label) in GENERIC_LABELS)
    if generic_count:
        errors.append("required_labels_contain_generic_placeholders")
    if invented_numbers and render_policy == "render":
        errors.append("metric_facts_lack_source_provenance")
    elif invented_numbers:
        warnings.append("rejected_contract_contains_unverified_metric")
    grounded_ratio = grounded_facts / max(len(facts), 1)
    if render_policy == "render":
        if scene_type == "none":
            errors.append("render_policy_requires_supported_scene_type")
        if len(objects) < _minimum_object_count(scene_type):
            errors.append("render_contract_has_too_few_objects")
        if len(beats) < _minimum_beat_count(scene_type):
            errors.append("render_contract_has_too_few_beats")
        if scene_type in RELATION_REQUIRED_SCENES and not relations:
            errors.append("render_contract_has_no_evidence_backed_relations")
        if grounded_ratio < 0.95:
            errors.append("render_contract_grounding_below_95_percent")
        if not required_labels:
            errors.append("render_contract_has_no_required_labels")
    elif not payload.get("rejection_reasons"):
        warnings.append("rejected_contract_has_no_rejection_reason")
    return VisualExplanationValidation(
        passed=not errors,
        errors=_unique(errors, limit=30),
        warnings=_unique(warnings, limit=12),
        grounded_fact_ratio=grounded_ratio,
        required_label_count=len(required_labels),
        generic_label_count=generic_count,
        invented_numeric_facts=_unique(invented_numbers, limit=8),
    )


def visual_explanation_prompt_block(ir: VisualExplanationIR | dict[str, Any]) -> str:
    payload = ir.to_dict() if isinstance(ir, VisualExplanationIR) else dict(ir or {})
    object_lines = [
        (
            f"- {item.get('object_id')}: role={item.get('role')} "
            f"label={item.get('label')} meaning={item.get('meaning')}"
        )
        for item in (payload.get("objects") or [])
        if isinstance(item, dict)
    ]
    beat_lines = [
        (
            f"- {item.get('phase')} {item.get('start_fraction')}-{item.get('end_fraction')}: "
            f"{item.get('subject_id')} {item.get('action')}"
            + (f" {item.get('target_id')}" if item.get("target_id") else "")
        )
        for item in (payload.get("beats") or [])
        if isinstance(item, dict)
    ]
    relation_lines = [
        (
            f"- {item.get('relation_id')}: {item.get('source_id')} "
            f"{item.get('relation_type')} {item.get('target_id')} "
            f"provenance={item.get('provenance')}"
        )
        for item in (payload.get("relations") or [])
        if isinstance(item, dict)
    ]
    return "\n".join(
        [
            "Visual Explanation IR:",
            f"- Version: {payload.get('version')}",
            f"- Scene type: {payload.get('scene_type')}",
            f"- Render policy: {payload.get('render_policy')}",
            f"- Viewer question: {payload.get('viewer_question')}",
            f"- Thesis: {payload.get('thesis')}",
            f"- Takeaway: {payload.get('takeaway')}",
            "- Required visible objects:",
            *object_lines,
            "- Evidence-backed relations:",
            *relation_lines,
            "- Required motion beats:",
            *beat_lines,
            "- Required labels: " + "; ".join(str(item) for item in (payload.get("required_labels") or [])),
            "- Forbidden: " + "; ".join(str(item) for item in (payload.get("forbidden_content") or [])),
        ]
    )


def _build_evidence(spec: dict[str, Any], *, sentence_text: str, context_text: str) -> list[EvidenceSpan]:
    evidence: list[EvidenceSpan] = []
    start = _optional_float(spec.get("start"))
    end = _optional_float(spec.get("end"))
    if sentence_text:
        evidence.append(EvidenceSpan("evidence_transcript", "transcript", sentence_text, start, end, 1.0))
    if context_text and _normalize(context_text) != _normalize(sentence_text):
        evidence.append(EvidenceSpan("evidence_context", "context", context_text, start, end, 0.96))
    for index, item in enumerate(spec.get("evidence_spans") or [], start=1):
        if not isinstance(item, dict):
            continue
        text = _clean(item.get("text"), max_chars=520)
        if not text:
            continue
        evidence.append(
            EvidenceSpan(
                evidence_id=str(item.get("evidence_id") or f"evidence_imported_{index:02d}"),
                source_type=str(item.get("source_type") or "imported_transcript"),
                text=text,
                start_sec=_optional_float(item.get("start_sec")),
                end_sec=_optional_float(item.get("end_sec")),
                confidence=_bounded(item.get("confidence"), 0.9),
            )
        )
    return evidence


def _build_facts(
    spec: dict[str, Any],
    *,
    semantic_frame: dict[str, Any],
    evidence: list[EvidenceSpan],
    source_text: str,
) -> list[GroundedFact]:
    facts: list[GroundedFact] = []
    source_ids = [item.evidence_id for item in evidence]
    metric_items = [item for item in (spec.get("metric_facts") or []) if isinstance(item, dict)]
    for index, item in enumerate(metric_items, start=1):
        value = _clean(item.get("value"), max_chars=28)
        label = _clean(item.get("label") or value, max_chars=64)
        if not value:
            continue
        grounded = _number_is_grounded(value, source_text)
        facts.append(
            GroundedFact(
                fact_id=f"fact_metric_{index:02d}",
                fact_type="metric",
                label=label,
                subject=label,
                predicate="has_value",
                value=value,
                unit=_metric_unit(value),
                evidence_ids=source_ids,
                grounding="transcript_exact" if grounded else "unverified",
                confidence=0.98 if grounded else 0.0,
            )
        )
    semantic_roles = (
        ("problem", ("problem", "before_state", "cause", "input")),
        ("mechanism", ("mechanism", "stages", "steps")),
        ("interface", ("screen",)),
        ("intervention", ("intervention", "action", "turn", "focus")),
        ("result", ("after_state", "result", "effect", "payoff", "viewer_takeaway")),
        ("decision", ("decision",)),
        ("branch_low", ("low_branch",)),
        ("branch_high", ("high_branch",)),
        ("constraint", ("constraint", "preserved_constraint")),
        ("quote", ("exact_quote",)),
        ("setup", ("setup",)),
    )
    seen_labels: set[str] = {item.label.lower() for item in facts}
    for fact_type, keys in semantic_roles:
        values: list[str] = []
        for key in keys:
            values.extend(_semantic_values(semantic_frame.get(key)))
        for value in values:
            label = _clean(value, max_chars=72)
            if not label or label.lower() in seen_labels:
                continue
            grounding, confidence = _grounding_for_label(label, source_text)
            facts.append(
                GroundedFact(
                    fact_id=f"fact_{fact_type}_{len(facts) + 1:02d}",
                    fact_type=fact_type,
                    label=label,
                    subject=label,
                    evidence_ids=source_ids,
                    grounding=grounding,
                    confidence=confidence,
                )
            )
            seen_labels.add(label.lower())
    for label in _unique([str(item) for item in (spec.get("required_labels") or [])], limit=12):
        if label.lower() in seen_labels:
            continue
        grounding, confidence = _grounding_for_label(label, source_text)
        if grounding == "unverified":
            continue
        facts.append(
            GroundedFact(
                fact_id=f"fact_required_{len(facts) + 1:02d}",
                fact_type="required",
                label=label,
                subject=label,
                evidence_ids=source_ids,
                grounding=grounding,
                confidence=confidence,
            )
        )
        seen_labels.add(label.lower())
    if not facts:
        headline = _clean(spec.get("headline"), max_chars=72)
        if headline and _normalize(headline) not in GENERIC_LABELS:
            grounding, confidence = _grounding_for_label(headline, source_text)
            facts.append(
                GroundedFact(
                    fact_id="fact_claim_01",
                    fact_type="claim",
                    label=headline,
                    subject=headline,
                    evidence_ids=source_ids,
                    grounding=grounding,
                    confidence=confidence,
                )
            )
    return facts


def _scene_type(
    spec: dict[str, Any],
    *,
    semantic_frame: dict[str, Any],
    facts: list[GroundedFact],
) -> str:
    metric_count = sum(1 for item in facts if item.fact_type == "metric" and item.grounding != "unverified")
    before = _first_value(semantic_frame, "before_state")
    after = _first_value(semantic_frame, "after_state")
    intervention = _first_value(semantic_frame, "intervention", "action", "turn")
    problem = _first_value(semantic_frame, "problem", "cause")
    mechanism = _first_value(semantic_frame, "mechanism", "mental_model")
    result = _first_value(semantic_frame, "result", "effect", "payoff", "viewer_takeaway")
    steps = _semantic_values(semantic_frame.get("steps")) or _semantic_values(semantic_frame.get("stages"))
    decision = _first_value(semantic_frame, "decision")
    low_branch = _first_value(semantic_frame, "low_branch")
    high_branch = _first_value(semantic_frame, "high_branch")
    exact_quote = _first_value(semantic_frame, "exact_quote")
    setup = _first_value(semantic_frame, "setup")
    visual_hint = str(spec.get("visual_type_hint") or "").strip().lower()
    source_text = f"{spec.get('sentence_text', '')} {spec.get('context_text', '')}".lower()
    if metric_count >= 1 and before and after and intervention:
        return "metric_intervention"
    if metric_count >= 1 and before and after:
        return "metric_delta"
    if metric_count >= 1:
        return "metric_proof"
    if decision and low_branch and high_branch:
        return "decision_branch"
    if setup and intervention and result:
        return "narrative_progression"
    if exact_quote:
        return "evidence_backed_quote"
    if visual_hint == "product_ui" and intervention and result:
        return "grounded_interface_walkthrough"
    if len(steps) >= 2:
        architecture_terms = set(
            re.findall(
                r"\b(?:api|service|authentication|planner|renderer|database|queue|gateway|worker)\b",
                source_text,
            )
        )
        if len(architecture_terms) >= 2:
            return "architecture_flow"
        return "guided_process"
    if problem and mechanism and (intervention or result):
        return "causal_intervention"
    if before and after and _normalize(before) != _normalize(after):
        return "matched_state_transform"
    return "none"


def _objects_for_scene(scene_type: str, facts: list[GroundedFact]) -> list[ExplanationObject]:
    by_type: dict[str, list[GroundedFact]] = {}
    for fact in facts:
        if fact.grounding == "unverified":
            continue
        by_type.setdefault(fact.fact_type, []).append(fact)
    role_limits = {
        "metric_delta": (("metric", 2), ("problem", 1), ("result", 2)),
        "metric_intervention": (("metric", 2), ("intervention", 1), ("result", 1)),
        "metric_proof": (("metric", 1), ("result", 1), ("mechanism", 1)),
        "causal_intervention": (
            ("problem", 1),
            ("mechanism", 1),
            ("intervention", 1),
            ("result", 1),
            ("constraint", 1),
        ),
        "guided_process": (
            ("problem", 1),
            ("mechanism", 4),
            ("result", 1),
            ("constraint", 1),
        ),
        "architecture_flow": (
            ("problem", 1),
            ("mechanism", 4),
            ("result", 1),
            ("constraint", 1),
        ),
        "matched_state_transform": (
            ("problem", 1),
            ("result", 1),
            ("constraint", 1),
        ),
        "grounded_interface_walkthrough": (
            ("interface", 1),
            ("intervention", 2),
            ("result", 1),
            ("constraint", 1),
        ),
        "decision_branch": (
            ("decision", 1),
            ("branch_low", 1),
            ("branch_high", 1),
            ("constraint", 1),
        ),
        "narrative_progression": (
            ("setup", 1),
            ("intervention", 1),
            ("result", 1),
            ("constraint", 1),
        ),
        "set_partition": (
            ("input", 1),
            ("group_size", 1),
            ("result", 1),
        ),
        "evidence_backed_quote": (("quote", 1),),
    }.get(scene_type, ())
    selected: list[GroundedFact] = []
    for fact_type, limit in role_limits:
        for fact in by_type.get(fact_type, []):
            if any(_labels_overlap(fact.label, existing.label) for existing in selected):
                continue
            selected.append(fact)
            if sum(1 for item in selected if item.fact_type == fact_type) >= limit:
                break
    selected = selected[:6]
    return [
        ExplanationObject(
            object_id=f"object_{index:02d}",
            role=fact.fact_type,
            label=fact.value if fact.fact_type == "metric" and fact.value else fact.label,
            meaning=fact.label,
            fact_ids=[fact.fact_id],
            emphasis=0.85 if fact.fact_type in {"metric", "result", "quote"} else 0.64,
        )
        for index, fact in enumerate(selected, start=1)
    ]


def _beats_for_scene(scene_type: str, objects: list[ExplanationObject]) -> list[ExplanationBeat]:
    if not objects or scene_type == "none":
        return []
    if len(objects) == 1:
        return [
            ExplanationBeat(
                beat_id="beat_01",
                phase="establish",
                action="reveal",
                subject_id=objects[0].object_id,
                start_fraction=0.08,
                end_fraction=0.42,
                fact_ids=list(objects[0].fact_ids),
            ),
            ExplanationBeat(
                beat_id="beat_02",
                phase="resolve",
                action="lock_focus",
                subject_id=objects[0].object_id,
                start_fraction=0.42,
                end_fraction=0.92,
                fact_ids=list(objects[0].fact_ids),
            ),
        ]
    beats: list[ExplanationBeat] = []
    count = len(objects)
    span = 0.72 / max(count, 1)
    for index, obj in enumerate(objects):
        phase = "establish" if index == 0 else ("resolve" if index == count - 1 else "explain")
        action = {
            "metric_delta": "compare_metric",
            "metric_intervention": "trace_metric_change",
            "metric_proof": "reveal_evidence",
            "causal_intervention": "propagate_cause",
            "guided_process": "advance_route",
            "architecture_flow": "route_request",
            "matched_state_transform": "transform_state",
            "grounded_interface_walkthrough": "focus_interface_state",
            "decision_branch": "activate_branch",
            "narrative_progression": "advance_story",
            "set_partition": "partition_set",
        }.get(scene_type, "reveal")
        start = 0.08 + index * span
        end = min(start + span + 0.1, 0.94)
        beats.append(
            ExplanationBeat(
                beat_id=f"beat_{index + 1:02d}",
                phase=phase,
                action=action,
                subject_id=obj.object_id,
                target_id=objects[index + 1].object_id if index + 1 < count else "",
                start_fraction=round(start, 3),
                end_fraction=round(end, 3),
                fact_ids=list(obj.fact_ids),
            )
        )
    return beats


def _viewer_question(scene_type: str, labels: list[str]) -> str:
    subject = labels[0] if labels else "the idea"
    return {
        "metric_delta": f"How does {subject} change?",
        "metric_intervention": f"What intervention changes {subject}?",
        "metric_proof": f"What evidence supports {subject}?",
        "causal_intervention": f"What mechanism turns {subject} into the result?",
        "guided_process": f"What sequence moves {subject} forward?",
        "architecture_flow": f"How does {subject} move through the system?",
        "matched_state_transform": f"What changes between the two states of {subject}?",
        "grounded_interface_walkthrough": f"What interface action produces the result for {subject}?",
        "decision_branch": f"How does the decision about {subject} branch?",
        "narrative_progression": f"What changes after {subject}?",
        "set_partition": f"How is {subject} grouped into fewer blocks?",
        "evidence_backed_quote": f"What phrase should the viewer retain about {subject}?",
    }.get(scene_type, "Is there enough concrete evidence to justify a generated visual?")


def _minimum_object_count(scene_type: str) -> int:
    if scene_type == "none":
        return 0
    if scene_type in {"evidence_backed_quote", "metric_proof"}:
        return 1
    return 2


def _minimum_beat_count(scene_type: str) -> int:
    if scene_type == "none":
        return 0
    return 2


def _contains_invented_numeric_fact(facts: list[GroundedFact], source_text: str) -> bool:
    return any(
        item.fact_type == "metric"
        and (item.grounding == "unverified" or not _number_is_grounded(item.value, source_text))
        for item in facts
    )


def _number_is_grounded(value: str, source_text: str) -> bool:
    target = _normalize_metric(value)
    if not target:
        return False
    return target in {_normalize_metric(match.group(0)) for match in NUMBER_PATTERN.finditer(source_text)}


def _normalize_metric(value: str) -> str:
    normalized = str(value or "").lower().replace("percent", "%")
    normalized = re.sub(r"milliseconds?", "ms", normalized)
    normalized = re.sub(r"seconds?|sec", "s", normalized)
    return re.sub(r"\s+", "", normalized)


def _metric_unit(value: str) -> str:
    match = NUMBER_PATTERN.search(str(value or ""))
    return str(match.group("unit") or "").lower() if match else ""


def _grounding_for_label(label: str, source_text: str) -> tuple[str, float]:
    if NUMBER_PATTERN.fullmatch(str(label or "").strip()) and _number_is_grounded(label, source_text):
        return "transcript_exact", 0.98
    label_tokens = set(_tokens(label))
    source_tokens = set(_tokens(source_text))
    if not label_tokens:
        return "unverified", 0.0
    overlap = len(label_tokens & source_tokens) / len(label_tokens)
    normalized_label = _normalize(label)
    normalized_source = _normalize(source_text)
    if normalized_label and normalized_label in normalized_source:
        return "transcript_exact", 0.98
    if overlap >= 0.5:
        return "transcript_paraphrase", min(0.92, 0.62 + overlap * 0.3)
    if overlap >= 0.2:
        return "semantic_derived", 0.68
    return "unverified", 0.0


def _semantic_values(value: Any) -> list[str]:
    if isinstance(value, list):
        return [_clean(item, max_chars=72) for item in value if _clean(item, max_chars=72)]
    if isinstance(value, str) and ";" in value:
        return [_clean(item, max_chars=72) for item in value.split(";") if _clean(item, max_chars=72)]
    cleaned = _clean(value, max_chars=72)
    return [cleaned] if cleaned else []


def _first_value(payload: dict[str, Any], *keys: str) -> str:
    for key in keys:
        values = _semantic_values(payload.get(key))
        if values:
            return values[0]
    return ""


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
    "set_partition",
}


def _extract_partition_model(
    evidence: list[EvidenceSpan],
    source_text: str,
) -> dict[str, Any]:
    source = str(source_text or "")
    ratio_match = re.search(
        r"\b(?P<size>\d+)\s*(?:is\s*)?to\s*1\b",
        source,
        flags=re.IGNORECASE,
    )
    block_match = re.search(
        r"\b(?P<count>\d+)\s+compressed\s+blocks?\b",
        source,
        flags=re.IGNORECASE,
    )
    if not ratio_match or not block_match:
        return {
            "version": "executable-visual-model-v1",
            "model_type": "set_partition",
            "valid": False,
            "operators": ["partition"],
            "issues": ["partition_model_missing_ratio_or_block_count"],
        }
    group_size = int(ratio_match.group("size"))
    group_count = int(block_match.group("count"))
    input_count = group_size * group_count
    source_numbers = {
        int(float(match.group("number")))
        for match in NUMBER_PATTERN.finditer(source)
        if float(match.group("number")).is_integer()
    }
    valid = (
        2 <= group_size <= 16
        and 2 <= group_count <= 32
        and input_count <= 128
        and input_count in source_numbers
    )
    evidence_ids = _unique(
        [
            item.evidence_id
            for item in evidence
            if re.search(r"\b\d+\s*(?:is\s*)?to\s*1\b", item.text, flags=re.IGNORECASE)
            or re.search(r"\b\d+\s+compressed\s+blocks?\b", item.text, flags=re.IGNORECASE)
        ],
        limit=12,
    )
    return {
        "version": "executable-visual-model-v1",
        "model_type": "set_partition",
        "valid": valid,
        "operators": ["partition", "compress"],
        "input_count": input_count,
        "group_size": group_size,
        "group_count": group_count,
        "headline": f"{input_count} tokens become {group_count} blocks",
        "takeaway": f"{group_size} original tokens per compressed block",
        "evidence_ids": evidence_ids,
        "checks": {
            "arithmetic_consistent": input_count == group_size * group_count,
            "input_count_appears_in_source": input_count in source_numbers,
        },
        "issues": [] if valid else ["partition_model_failed_arithmetic_grounding"],
    }


def _partition_facts(model: dict[str, Any]) -> list[GroundedFact]:
    evidence_ids = [str(item) for item in model.get("evidence_ids") or [] if str(item)]
    input_count = int(model.get("input_count") or 0)
    group_size = int(model.get("group_size") or 0)
    group_count = int(model.get("group_count") or 0)
    return [
        GroundedFact(
            fact_id="fact_partition_input",
            fact_type="input",
            label=f"{input_count} original tokens",
            subject="original tokens",
            predicate="has_count",
            value=str(input_count),
            unit="tokens",
            evidence_ids=evidence_ids,
            grounding="deterministic_derived",
            confidence=0.99,
        ),
        GroundedFact(
            fact_id="fact_partition_group_size",
            fact_type="group_size",
            label=f"{group_size} tokens per block",
            subject="compressed block",
            predicate="summarizes",
            object="original tokens",
            value=str(group_size),
            unit="tokens",
            evidence_ids=evidence_ids,
            grounding="deterministic_derived",
            confidence=0.99,
        ),
        GroundedFact(
            fact_id="fact_partition_result",
            fact_type="result",
            label=f"{group_count} compressed blocks",
            subject="compressed blocks",
            predicate="has_count",
            value=str(group_count),
            evidence_ids=evidence_ids,
            grounding="transcript_exact",
            confidence=0.99,
        ),
    ]


def _build_executable_model(
    scene_type: str,
    *,
    facts: list[GroundedFact],
    evidence: list[EvidenceSpan],
    source_text: str,
) -> dict[str, Any]:
    source = str(source_text or "").lower()
    metric_facts = [
        item
        for item in facts
        if item.fact_type == "metric" and item.grounding != "unverified"
    ]
    if scene_type in {"metric_delta", "metric_intervention", "metric_proof"}:
        return _metric_executable_model(
            scene_type,
            metric_facts=metric_facts,
            evidence=evidence,
            source=source,
        )
    patterns = {
        "architecture_flow": (
            r"\b(?:calls?|connects?|enters?|invokes?|links?|loads?|passes?|queries?|"
            r"reads?|reaches?|receives?|routes?|returns?|sends?|stores?|through|"
            r"validates?|writes?)\b",
        ),
        "causal_intervention": (
            r"\b(?:because|therefore|causes?|forces?|makes?|exposes?|leads?\s+to|results?\s+in)\b",
        ),
        "decision_branch": (
            r"\bif\b.+\b(?:otherwise|else)\b",
        ),
        "grounded_interface_walkthrough": (
            r"\b(?:highlights?|opens?|clicks?|selects?|lets?|retries?|shows?)\b",
        ),
        "guided_process": (
            r"\b(?:adds?|applies?|before|after|builds?|checks?|chooses?|classifies?|"
            r"connects?|creates?|filters?|finally|generates?|groups?|hands?|links?|"
            r"maps?|next|reads?|renders?|routes?|runs?|scores?|selects?|then|turns?|"
            r"updates?|uses?|validates?|writes?)\b",
        ),
        "matched_state_transform": (
            r"\b(?:from\b.+\bto|before|after|old|new|while|instead|replaces?|keeps?|preserves?)\b",
        ),
        "narrative_progression": (
            r"\b(?:first|then|next|second|after|finally|traced?|handled?)\b",
        ),
        "evidence_backed_quote": (r".+",),
    }.get(scene_type, ())
    evidence_ids = _evidence_ids_matching(evidence, patterns)
    valid = scene_type == "none" or bool(evidence_ids)
    return {
        "version": "executable-visual-model-v1",
        "model_type": scene_type,
        "valid": valid,
        "operators": _operators_for_scene(scene_type),
        "evidence_ids": evidence_ids,
        "issues": [] if valid else ["no_executable_visual_model"],
    }


def _metric_executable_model(
    scene_type: str,
    *,
    metric_facts: list[GroundedFact],
    evidence: list[EvidenceSpan],
    source: str,
) -> dict[str, Any]:
    values = [item.value for item in metric_facts]
    units = [_metric_unit(item.value) for item in metric_facts]
    explicit_transition = bool(
        re.search(
            r"\bfrom\s+\d+(?:\.\d+)?(?:\s*[a-z%]+)?\s+to\s+\d+(?:\.\d+)?",
            source,
        )
        or re.search(
            r"\b(?:falls?|drops?|declines?|rises?|grows?|increases?|decreases?|reduces?)\b"
            r"[^.]{0,80}\bto\s+\d+(?:\.\d+)?",
            source,
        )
    )
    relative_baseline = bool(
        re.search(
            r"\b(?:only|just)\s+\d+(?:\.\d+)?\s*(?:%|percent|x)\b[^.]{0,80}\b(?:before|previous|prior|used)\b",
            source,
        )
        or re.search(
            r"\b\d+(?:\.\d+)?\s*(?:%|percent|x)\b[^.]{0,60}\b(?:less|more|of\s+the)\b",
            source,
        )
    )
    compatible_units = (
        len(units) >= 2
        and bool(units[0])
        and all(unit == units[0] for unit in units[:2])
    )
    labels_are_complete = all(
        item.label and not _label_is_fragmented(item.label)
        for item in metric_facts
    )
    if scene_type == "metric_proof":
        valid = bool(metric_facts) and labels_are_complete
        model_type = "measured_claim"
    elif len(metric_facts) >= 2:
        valid = labels_are_complete and (explicit_transition or compatible_units)
        model_type = "measured_transition"
    else:
        valid = (
            len(metric_facts) == 1
            and labels_are_complete
            and relative_baseline
        )
        model_type = "relative_baseline"
    issues: list[str] = []
    if not metric_facts:
        issues.append("metric_model_has_no_grounded_values")
    if not labels_are_complete:
        issues.append("fragmented_semantic_labels")
    if metric_facts and not valid:
        issues.append("ambiguous_metric_model")
    patterns = (
        r"\bfrom\b.+\bto\b",
        r"\b(?:falls?|drops?|rises?|increases?|decreases?|reduces?|only|before|previous|prior)\b",
    )
    return {
        "version": "executable-visual-model-v1",
        "model_type": model_type,
        "valid": valid,
        "operators": _operators_for_scene(scene_type),
        "values": values,
        "units": units,
        "evidence_ids": _evidence_ids_matching(evidence, patterns),
        "checks": {
            "explicit_transition": explicit_transition,
            "relative_baseline": relative_baseline,
            "compatible_units": compatible_units,
            "labels_are_complete": labels_are_complete,
        },
        "issues": _unique(issues, limit=6),
    }


def _relations_for_scene(
    scene_type: str,
    *,
    objects: list[ExplanationObject],
    evidence: list[EvidenceSpan],
    executable_model: dict[str, Any],
) -> list[ExplanationRelation]:
    if not bool(executable_model.get("valid")):
        return []
    evidence_ids = [
        str(item)
        for item in executable_model.get("evidence_ids") or []
        if str(item)
    ]
    if not evidence_ids:
        evidence_ids = [item.evidence_id for item in evidence]
    by_role: dict[str, list[ExplanationObject]] = {}
    for obj in objects:
        by_role.setdefault(obj.role, []).append(obj)
    edges: list[tuple[ExplanationObject | None, str, ExplanationObject | None]] = []
    if scene_type == "metric_delta":
        metrics = by_role.get("metric", [])
        before = _first_object(by_role, "problem") or (metrics[0] if metrics else None)
        results = by_role.get("result", [])
        after = results[0] if results else (metrics[-1] if metrics else None)
        outcome = results[1] if len(results) > 1 else None
        _append_relation_edge(edges, before, "transforms_to", after)
        if metrics:
            _append_relation_edge(edges, metrics[-1], "supports", after)
        _append_relation_edge(edges, after, "enables", outcome)
    elif scene_type == "metric_intervention":
        metrics = by_role.get("metric", [])
        intervention = _first_object(by_role, "intervention")
        if metrics:
            _append_relation_edge(edges, metrics[0], "changes_after", intervention)
        if len(metrics) > 1:
            _append_relation_edge(edges, intervention, "produces", metrics[-1])
    elif scene_type == "metric_proof":
        metric = _first_object(by_role, "metric")
        support = _first_object(by_role, "result") or _first_object(by_role, "mechanism")
        _append_relation_edge(edges, support, "supports", metric)
    elif scene_type == "causal_intervention":
        chain = [
            _first_object(by_role, role)
            for role in ("problem", "mechanism", "intervention", "result")
        ]
        chain = [item for item in chain if item is not None]
        for source, target in zip(chain, chain[1:]):
            relation_type = "produces" if target.role == "result" else "causes"
            _append_relation_edge(edges, source, relation_type, target)
    elif scene_type in {"guided_process", "architecture_flow", "narrative_progression"}:
        relation_type = "routes_to" if scene_type == "architecture_flow" else "precedes"
        if scene_type == "narrative_progression":
            relation_type = "leads_to"
        for source, target in zip(objects, objects[1:]):
            _append_relation_edge(edges, source, relation_type, target)
    elif scene_type == "matched_state_transform":
        before = _first_object(by_role, "problem")
        after = _first_object(by_role, "result")
        constraint = _first_object(by_role, "constraint")
        _append_relation_edge(edges, before, "transforms_to", after)
        _append_relation_edge(edges, before, "preserves", constraint)
        _append_relation_edge(edges, after, "preserves", constraint)
    elif scene_type == "grounded_interface_walkthrough":
        interface = _first_object(by_role, "interface")
        actions = by_role.get("intervention", [])
        result = _first_object(by_role, "result")
        for action in actions:
            _append_relation_edge(edges, interface, "contains_action", action)
        _append_relation_edge(edges, actions[-1] if actions else None, "produces", result)
    elif scene_type == "decision_branch":
        decision = _first_object(by_role, "decision")
        _append_relation_edge(edges, decision, "branches_to_low", _first_object(by_role, "branch_low"))
        _append_relation_edge(edges, decision, "branches_to_high", _first_object(by_role, "branch_high"))
    elif scene_type == "set_partition":
        source = _first_object(by_role, "input")
        group_size = _first_object(by_role, "group_size")
        result = _first_object(by_role, "result")
        _append_relation_edge(edges, source, "transforms_to", result)
        _append_relation_edge(edges, group_size, "supports", result)
    relations: list[ExplanationRelation] = []
    seen: set[tuple[str, str, str]] = set()
    for source, relation_type, target in edges:
        if source is None or target is None or source.object_id == target.object_id:
            continue
        key = (source.object_id, relation_type, target.object_id)
        if key in seen:
            continue
        seen.add(key)
        relations.append(
            ExplanationRelation(
                relation_id=f"relation_{len(relations) + 1:02d}",
                source_id=source.object_id,
                relation_type=relation_type,
                target_id=target.object_id,
                evidence_ids=evidence_ids,
                provenance="executable_visual_model",
                confidence=0.9,
            )
        )
    return relations


def _operators_for_scene(scene_type: str) -> list[str]:
    return {
        "architecture_flow": ["route"],
        "causal_intervention": ["cause", "intervene", "resolve"],
        "decision_branch": ["branch"],
        "evidence_backed_quote": ["focus"],
        "grounded_interface_walkthrough": ["locate", "act", "observe"],
        "guided_process": ["sequence"],
        "matched_state_transform": ["transform", "preserve"],
        "metric_delta": ["compare", "transform"],
        "metric_intervention": ["compare", "intervene"],
        "metric_proof": ["measure", "support"],
        "narrative_progression": ["sequence", "resolve"],
        "set_partition": ["partition", "compress"],
    }.get(scene_type, [])


def _evidence_ids_matching(
    evidence: list[EvidenceSpan],
    patterns: tuple[str, ...],
) -> list[str]:
    return _unique(
        [
            item.evidence_id
            for item in evidence
            if any(re.search(pattern, item.text, flags=re.IGNORECASE | re.DOTALL) for pattern in patterns)
        ],
        limit=12,
    )


def _first_object(
    by_role: dict[str, list[ExplanationObject]],
    role: str,
) -> ExplanationObject | None:
    values = by_role.get(role) or []
    return values[0] if values else None


def _append_relation_edge(
    edges: list[tuple[ExplanationObject | None, str, ExplanationObject | None]],
    source: ExplanationObject | None,
    relation_type: str,
    target: ExplanationObject | None,
) -> None:
    if source is not None and target is not None:
        edges.append((source, relation_type, target))


def _label_is_fragmented(value: str) -> bool:
    cleaned = _clean(value, max_chars=160)
    if not cleaned:
        return True
    lowered = cleaned.lower()
    if re.search(r"[.!?]", cleaned):
        return True
    if re.search(r"\b(?:that|which|because|and|or|of|for|to|with|is|are|was|were)\s*$", lowered):
        return True
    if re.search(r"\b(?:square|squared)\s+that\b", lowered):
        return True
    words = _tokens(cleaned)
    if len(words) == 2 and all(word in VERB_FRAGMENT_WORDS for word in words):
        return True
    return _looks_like_internal_instruction(cleaned)


def _looks_like_internal_instruction(value: str) -> bool:
    lowered = str(value or "").lower()
    return bool(
        re.match(r"^(?:ground|show|make|use|keep|tie|give)\b", lowered)
        and re.search(r"\b(?:viewer|visual|spoken|claim|evidence|track|legible)\b", lowered)
    )


def _first_grounded_semantic_value(
    payload: dict[str, Any],
    source_text: str,
    *keys: str,
) -> str:
    for key in keys:
        for value in _semantic_values(payload.get(key)):
            if _label_is_fragmented(value):
                continue
            if _grounding_for_label(value, source_text)[0] != "unverified":
                return value
    return ""


def _grounded_display_copy(value: Any, source_text: str) -> str:
    cleaned = _clean(value, max_chars=96)
    if not cleaned or _label_is_fragmented(cleaned):
        return ""
    return (
        cleaned
        if _grounding_for_label(cleaned, source_text)[0] != "unverified"
        else ""
    )


def _labels_overlap(first: str, second: str) -> bool:
    first_tokens = set(_tokens(first))
    second_tokens = set(_tokens(second))
    if not first_tokens or not second_tokens:
        return _normalize(first) == _normalize(second)
    overlap = len(first_tokens & second_tokens)
    return (
        overlap / len(first_tokens) >= 0.72
        or overlap / len(second_tokens) >= 0.72
    )


def _unique(values: list[str], *, limit: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = str(value or "").strip()
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
        if len(result) >= limit:
            break
    return result


def _tokens(value: str) -> list[str]:
    return [
        _stem(token)
        for token in re.findall(r"[a-z0-9%+./-]+", str(value or "").lower())
        if len(token) >= 2 and token not in STOPWORDS and not token.isdigit()
    ]


def _stem(token: str) -> str:
    replacements = {
        "retrieval": "retrieve",
        "rendering": "render",
        "validated": "validate",
        "validation": "validate",
        "copying": "copy",
        "caching": "cache",
    }
    if token in replacements:
        return replacements[token]
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def _normalize(value: Any) -> str:
    return re.sub(r"[^a-z0-9%+./-]+", " ", str(value or "").lower()).strip()


def _clean(value: Any, *, max_chars: int) -> str:
    cleaned = re.sub(r"\s+", " ", str(value or "").replace("\ufeff", "")).strip(" ,.;:-")
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip(" ,.;:-")
    return cleaned


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bounded(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(number, 1.0))


__all__ = [
    "EvidenceSpan",
    "ExplanationBeat",
    "ExplanationObject",
    "ExplanationRelation",
    "GroundedFact",
    "SCENE_TYPES",
    "VISUAL_EXPLANATION_VERSION",
    "VisualExplanationIR",
    "VisualExplanationValidation",
    "build_visual_explanation_ir",
    "validate_visual_explanation_ir",
    "visual_explanation_prompt_block",
]
