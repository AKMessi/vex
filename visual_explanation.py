from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any


VISUAL_EXPLANATION_VERSION = "visual-explanation-v1"
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
    "none",
}
GENERIC_LABELS = {
    "action",
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
    "start",
    "system",
    "timing",
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
    facts = _build_facts(
        spec,
        semantic_frame=semantic_frame,
        evidence=evidence,
        source_text=source_text,
    )
    scene_type = _scene_type(spec, semantic_frame=semantic_frame, facts=facts)
    objects = _objects_for_scene(scene_type, facts)
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
    if not facts:
        rejection_reasons.append("no_grounded_facts")
    render_policy = "reject" if rejection_reasons else "render"
    takeaway = _first_value(
        semantic_frame,
        "viewer_takeaway",
        "result",
        "effect",
        "after_state",
        "payoff",
        "exact_quote",
    ) or (required_labels[-1] if required_labels else "")
    thesis = _first_value(semantic_frame, "mental_model", "thesis", "mechanism") or _clean(
        spec.get("headline") or sentence_text,
        max_chars=150,
    )
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
        ("mechanism", ("mechanism", "mental_model", "stages", "steps")),
        ("interface", ("screen",)),
        ("intervention", ("intervention", "action", "turn", "focus")),
        ("result", ("result", "effect", "after_state", "payoff", "viewer_takeaway")),
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
    ordered_types = {
        "metric_delta": ("metric", "problem", "result", "mechanism", "required"),
        "metric_intervention": ("metric", "problem", "intervention", "result", "mechanism", "required"),
        "metric_proof": ("metric", "result", "mechanism", "required"),
        "causal_intervention": ("problem", "mechanism", "intervention", "result", "constraint", "required"),
        "guided_process": ("problem", "mechanism", "intervention", "result", "constraint", "required"),
        "architecture_flow": ("problem", "mechanism", "intervention", "result", "constraint", "required"),
        "matched_state_transform": ("problem", "result", "constraint", "mechanism", "required"),
        "grounded_interface_walkthrough": (
            "interface",
            "problem",
            "intervention",
            "result",
            "constraint",
            "required",
        ),
        "decision_branch": (
            "decision",
            "branch_low",
            "branch_high",
            "constraint",
            "result",
            "required",
        ),
        "narrative_progression": ("setup", "intervention", "result", "constraint", "required"),
        "evidence_backed_quote": ("quote", "required"),
    }.get(scene_type, ())
    selected: list[GroundedFact] = []
    for fact_type in ordered_types:
        selected.extend(by_type.get(fact_type, []))
    if scene_type in {"guided_process", "architecture_flow"} and len(selected) < 2:
        selected.extend(by_type.get("mechanism", []))
    selected = _dedupe_facts(selected)[:6]
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
    if NUMBER_PATTERN.search(label) and _number_is_grounded(label, source_text):
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


def _dedupe_facts(facts: list[GroundedFact]) -> list[GroundedFact]:
    result: list[GroundedFact] = []
    seen: set[str] = set()
    for fact in facts:
        key = _normalize(fact.value or fact.label)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(fact)
    return result


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
    cleaned = re.sub(r"\s+", " ", str(value or "")).strip(" ,.;:-")
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
    "GroundedFact",
    "SCENE_TYPES",
    "VISUAL_EXPLANATION_VERSION",
    "VisualExplanationIR",
    "VisualExplanationValidation",
    "build_visual_explanation_ir",
    "validate_visual_explanation_ir",
    "visual_explanation_prompt_block",
]
