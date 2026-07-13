from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import re
from typing import Any, Iterable


COMMUNICATION_CONTRACT_VERSION = "vex-communication-contract-v1"
COMMUNICATION_EVALUATION_VERSION = "vex-communication-evaluation-v1"

_WORD_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_NUMBER_RE = re.compile(r"(?<![a-z0-9.])\d+(?:\.\d+)?(?:\s*(?:%|x|ms|s|kb|mb|gb|tb|k|m|b|tokens?))?", re.IGNORECASE)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "with",
}
_SYNONYM_GROUPS = (
    {"compress", "compressed", "compression", "compact", "condense", "condensed"},
    {"choose", "chooses", "pick", "picks", "select", "selects", "selected"},
    {"score", "scores", "rank", "ranks", "rate", "rates"},
    {"become", "becomes", "convert", "converts", "transform", "transforms", "turn", "turns"},
    {"cause", "causes", "create", "creates", "lead", "leads", "produce", "produces"},
    {"before", "first", "precede", "precedes"},
    {"after", "follow", "follows", "next", "then"},
    {"block", "blocks", "chunk", "chunks", "group", "groups"},
    {"entry", "entries", "representation", "representations", "summary", "summaries"},
    {"enable", "enables", "allow", "allows"},
    {"token", "tokens"},
    {"best", "strong", "strongest", "top"},
)
_SYNONYM_CANONICAL = {
    token: sorted(group)[0]
    for group in _SYNONYM_GROUPS
    for token in group
}
_NUMBER_WORDS = {
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
_NUMBER_UNITS = {"token": "tokens", "tokens": "tokens", "percent": "%", "percentage": "%"}


@dataclass(frozen=True)
class AtomicProposition:
    proposition_id: str
    proposition: str
    expected_answers: list[str]
    proposition_type: str
    evidence_ids: list[str]
    dependency_ids: list[str] = field(default_factory=list)
    required: bool = True
    weight: float = 1.0
    exact_numbers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["weight"] = round(float(self.weight), 4)
        return payload


@dataclass(frozen=True)
class ViewerQuestion:
    question_id: str
    proposition_id: str
    question: str
    expected_answers: list[str]
    dependency_question_ids: list[str] = field(default_factory=list)
    answer_type: str = "short_text"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CommunicationContract:
    version: str
    contract_id: str
    visual_id: str
    thesis: str
    takeaway: str
    propositions: list[AtomicProposition]
    questions: list[ViewerQuestion]
    required_terms: list[str]
    forbidden_claims: list[str]
    temporal_sequence: list[str]
    source_ir_signature: str
    minimum_semantic_score: float = 0.72
    signature: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "propositions": [item.to_dict() for item in self.propositions],
            "questions": [item.to_dict() for item in self.questions],
            "minimum_semantic_score": round(float(self.minimum_semantic_score), 4),
        }


@dataclass(frozen=True)
class PropositionResult:
    proposition_id: str
    score: float
    passed: bool
    matched_answer: str
    viewer_answer: str
    dependency_passed: bool
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["score"] = round(float(self.score), 4)
        return payload


@dataclass(frozen=True)
class CommunicationEvaluation:
    version: str
    passed: bool
    score: float
    proposition_coverage: float
    temporal_score: float
    unsupported_claims: list[str]
    missing_proposition_ids: list[str]
    results: list[PropositionResult]
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("score", "proposition_coverage", "temporal_score"):
            payload[key] = round(float(payload[key]), 4)
        payload["results"] = [item.to_dict() for item in self.results]
        return payload


def build_communication_contract(ir: dict[str, Any]) -> CommunicationContract:
    payload = dict(ir or {})
    visual_id = _clean(payload.get("visual_id"), limit=100) or "visual"
    objects = {
        str(item.get("object_id") or ""): dict(item)
        for item in payload.get("objects") or []
        if isinstance(item, dict) and item.get("object_id")
    }
    facts = {
        str(item.get("fact_id") or ""): dict(item)
        for item in payload.get("facts") or []
        if isinstance(item, dict) and item.get("fact_id")
    }
    propositions: list[AtomicProposition] = []
    object_to_proposition: dict[str, str] = {}
    for index, obj in enumerate(objects.values()):
        fact_ids = _strings(obj.get("fact_ids"), limit=8)
        fact = next((facts[item] for item in fact_ids if item in facts), {})
        label = _clean(
            fact.get("label") or obj.get("meaning") or obj.get("label"),
            limit=260,
        )
        if not label:
            continue
        proposition_id = f"proposition_{index + 1:02d}"
        aliases = _unique(
            [
                label,
                _clean(obj.get("meaning"), limit=260),
                _clean(obj.get("label"), limit=180),
                _fact_statement(fact),
            ],
            limit=5,
        )
        evidence_ids = _unique(
            [
                *_strings(fact.get("evidence_ids"), limit=8),
                *_evidence_ids_for_facts(fact_ids, facts),
            ],
            limit=8,
        )
        propositions.append(
            AtomicProposition(
                proposition_id=proposition_id,
                proposition=label,
                expected_answers=aliases,
                proposition_type=_clean(fact.get("fact_type") or obj.get("role"), limit=40) or "fact",
                evidence_ids=evidence_ids,
                weight=_proposition_weight(obj, fact),
                exact_numbers=_numbers(" ".join(aliases)),
            )
        )
        object_to_proposition[str(obj.get("object_id") or "")] = proposition_id

    relation_propositions: list[AtomicProposition] = []
    for index, relation in enumerate(payload.get("relations") or []):
        if not isinstance(relation, dict) or not bool(relation.get("required", True)):
            continue
        source = objects.get(str(relation.get("source_id") or ""), {})
        target = objects.get(str(relation.get("target_id") or ""), {})
        source_label = _clean(source.get("label") or source.get("meaning"), limit=120)
        target_label = _clean(target.get("label") or target.get("meaning"), limit=120)
        relation_type = _clean(relation.get("relation_type"), limit=40).replace("_", " ")
        if not source_label or not target_label or not relation_type:
            continue
        proposition_id = f"relation_{index + 1:02d}"
        statement = f"{source_label} {relation_type} {target_label}"
        dependencies = _unique(
            [
                object_to_proposition.get(str(relation.get("source_id") or ""), ""),
                object_to_proposition.get(str(relation.get("target_id") or ""), ""),
            ],
            limit=4,
        )
        relation_propositions.append(
            AtomicProposition(
                proposition_id=proposition_id,
                proposition=statement,
                expected_answers=_unique(
                    [statement, f"{source_label} leads to {target_label}", f"{target_label} follows {source_label}"],
                    limit=4,
                ),
                proposition_type="relation",
                evidence_ids=_strings(relation.get("evidence_ids"), limit=8),
                dependency_ids=dependencies,
                weight=1.15,
                # Object propositions already prove exact quantities. Relation
                # questions test the transformation and should allow concise answers.
                exact_numbers=[],
            )
        )
    propositions.extend(relation_propositions)
    propositions = _with_valid_dependencies(propositions)

    questions = [
        ViewerQuestion(
            question_id=f"question_{index + 1:02d}",
            proposition_id=item.proposition_id,
            question=_question_for(item),
            expected_answers=list(item.expected_answers),
            dependency_question_ids=[
                f"question_{next_index + 1:02d}"
                for next_index, candidate in enumerate(propositions)
                if candidate.proposition_id in item.dependency_ids
            ],
            answer_type="relation" if item.proposition_type == "relation" else "short_text",
        )
        for index, item in enumerate(propositions)
    ]
    temporal_sequence = _temporal_sequence(payload, objects)
    source_signature = _signature(payload)
    unsigned = CommunicationContract(
        version=COMMUNICATION_CONTRACT_VERSION,
        contract_id=f"{visual_id}-communication",
        visual_id=visual_id,
        thesis=_clean(payload.get("thesis"), limit=240),
        takeaway=_clean(payload.get("takeaway"), limit=280),
        propositions=propositions,
        questions=questions,
        required_terms=_required_terms(payload, propositions),
        forbidden_claims=_unique(
            [*_strings(payload.get("forbidden_content"), limit=12), "unsupported numeric claims"],
            limit=16,
        ),
        temporal_sequence=temporal_sequence,
        source_ir_signature=source_signature,
        minimum_semantic_score=0.72,
    )
    return replace(unsigned, signature=communication_contract_signature(unsigned))


def communication_contract_signature(contract: CommunicationContract | dict[str, Any]) -> str:
    payload = contract.to_dict() if isinstance(contract, CommunicationContract) else dict(contract or {})
    payload.pop("signature", None)
    return _signature(payload)


def validate_communication_contract(
    contract: CommunicationContract | dict[str, Any],
    *,
    source_ir: dict[str, Any] | None = None,
) -> list[str]:
    payload = contract.to_dict() if isinstance(contract, CommunicationContract) else dict(contract or {})
    errors: list[str] = []
    if payload.get("version") != COMMUNICATION_CONTRACT_VERSION:
        errors.append("unsupported_communication_contract_version")
    if not payload.get("propositions"):
        errors.append("communication_contract_has_no_propositions")
    proposition_ids = {
        str(item.get("proposition_id") or "")
        for item in payload.get("propositions") or []
        if isinstance(item, dict)
    }
    for item in payload.get("propositions") or []:
        if not isinstance(item, dict):
            errors.append("communication_contract_has_invalid_proposition")
            continue
        proposition_id = str(item.get("proposition_id") or "")
        if not proposition_id or not item.get("expected_answers"):
            errors.append(f"communication_contract_invalid_proposition:{proposition_id or 'unknown'}")
        unknown = set(_strings(item.get("dependency_ids"), limit=20)) - proposition_ids
        if unknown:
            errors.append(f"communication_contract_unknown_dependency:{proposition_id}")
    expected_signature = communication_contract_signature(payload)
    if str(payload.get("signature") or "") != expected_signature:
        errors.append("communication_contract_signature_mismatch")
    if source_ir is not None and str(payload.get("source_ir_signature") or "") != _signature(source_ir):
        errors.append("communication_contract_source_signature_mismatch")
    return _unique(errors, limit=30)


def evaluate_viewer_answers(
    contract: CommunicationContract | dict[str, Any],
    answers: dict[str, Any],
    *,
    decoded_thesis: str = "",
    decoded_sequence: Iterable[str] = (),
    unsupported_claims: Iterable[str] = (),
) -> CommunicationEvaluation:
    payload = contract.to_dict() if isinstance(contract, CommunicationContract) else dict(contract or {})
    propositions = [dict(item) for item in payload.get("propositions") or [] if isinstance(item, dict)]
    question_by_proposition = {
        str(item.get("proposition_id") or ""): str(item.get("question_id") or "")
        for item in payload.get("questions") or []
        if isinstance(item, dict)
    }
    raw_scores: dict[str, float] = {}
    selected_answers: dict[str, tuple[str, str]] = {}
    number_issues: dict[str, list[str]] = {}
    for proposition in propositions:
        proposition_id = str(proposition.get("proposition_id") or "")
        question_id = question_by_proposition.get(proposition_id, proposition_id)
        viewer_answer = _clean(answers.get(question_id, answers.get(proposition_id, "")), limit=500)
        expected_answers = _strings(proposition.get("expected_answers"), limit=10)
        if proposition.get("proposition_type") in {"thesis", "takeaway"} and decoded_thesis:
            viewer_answer = viewer_answer or _clean(decoded_thesis, limit=500)
        best_expected, score = _best_semantic_match(viewer_answer, expected_answers)
        exact_numbers = _strings(proposition.get("exact_numbers"), limit=12)
        issues: list[str] = []
        if exact_numbers and viewer_answer:
            actual_numbers = _numbers(viewer_answer)
            if not set(exact_numbers).issubset(set(actual_numbers)):
                score = min(score, 0.35)
                issues.append("viewer_answer_numeric_mismatch")
        raw_scores[proposition_id] = score
        selected_answers[proposition_id] = (best_expected, viewer_answer)
        number_issues[proposition_id] = issues

    results: list[PropositionResult] = []
    for proposition in propositions:
        proposition_id = str(proposition.get("proposition_id") or "")
        dependencies = _strings(proposition.get("dependency_ids"), limit=12)
        dependency_passed = all(raw_scores.get(item, 0.0) >= 0.56 for item in dependencies)
        score = raw_scores.get(proposition_id, 0.0)
        if not dependency_passed:
            score *= 0.55
        expected, viewer_answer = selected_answers.get(proposition_id, ("", ""))
        issues = list(number_issues.get(proposition_id, []))
        if not viewer_answer:
            issues.append("viewer_answer_missing")
        if not dependency_passed:
            issues.append("viewer_answer_dependency_failed")
        results.append(
            PropositionResult(
                proposition_id=proposition_id,
                score=score,
                passed=score >= 0.56 and not issues,
                matched_answer=expected,
                viewer_answer=viewer_answer,
                dependency_passed=dependency_passed,
                issues=issues,
            )
        )

    total_weight = sum(max(float(item.get("weight") or 1.0), 0.1) for item in propositions)
    weighted_score = sum(
        result.score * max(float(proposition.get("weight") or 1.0), 0.1)
        for result, proposition in zip(results, propositions)
    ) / max(total_weight, 0.1)
    required_results = [
        result
        for result, proposition in zip(results, propositions)
        if bool(proposition.get("required", True))
    ]
    coverage = sum(1 for item in required_results if item.passed) / max(len(required_results), 1)
    temporal_score = semantic_sequence_score(
        _strings(payload.get("temporal_sequence"), limit=20),
        [_clean(item, limit=180) for item in decoded_sequence],
    )
    claims = _unique([_clean(item, limit=280) for item in unsupported_claims], limit=16)
    missing = [item.proposition_id for item in required_results if not item.passed]
    issues: list[str] = []
    if missing:
        issues.append("viewer_could_not_recover_required_propositions")
    if claims:
        issues.append("viewer_found_unsupported_claims")
    if payload.get("temporal_sequence") and temporal_score < 0.52:
        issues.append("viewer_could_not_recover_temporal_sequence")
    minimum = _bounded(payload.get("minimum_semantic_score"), 0.72)
    final_score = weighted_score * 0.82 + temporal_score * 0.18
    passed = final_score >= minimum and coverage >= 0.72 and not claims and not issues
    return CommunicationEvaluation(
        version=COMMUNICATION_EVALUATION_VERSION,
        passed=passed,
        score=final_score,
        proposition_coverage=coverage,
        temporal_score=temporal_score,
        unsupported_claims=claims,
        missing_proposition_ids=missing,
        results=results,
        issues=issues,
    )


def semantic_text_score(first: Any, second: Any) -> float:
    left = _semantic_tokens(first)
    right = _semantic_tokens(second)
    if not left or not right:
        return 0.0
    left_set = set(left)
    right_set = set(right)
    shared = left_set & right_set
    precision = len(shared) / len(left_set)
    recall = len(shared) / len(right_set)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
    left_text = " ".join(left)
    right_text = " ".join(right)
    containment = min(len(left_set), len(right_set)) / max(len(left_set), len(right_set)) if left_text in right_text or right_text in left_text else 0.0
    number_score = _number_alignment(first, second)
    return _bounded(max(f1, containment) * 0.84 + number_score * 0.16, 0.0)


def semantic_sequence_score(expected: list[str], observed: list[str]) -> float:
    if not expected:
        return 1.0
    if not observed:
        return 0.0
    cursor = 0
    matches = 0.0
    for expected_item in expected:
        best_index = -1
        best_score = 0.0
        for index in range(cursor, len(observed)):
            score = semantic_text_score(expected_item, observed[index])
            if score > best_score:
                best_score = score
                best_index = index
        if best_index >= 0 and best_score >= 0.34:
            matches += best_score
            cursor = best_index + 1
    return _bounded(matches / max(len(expected), 1), 0.0)


def _question_for(proposition: AtomicProposition) -> str:
    if proposition.proposition_type == "relation":
        return "What relationship or transformation connects the visible ideas?"
    if proposition.proposition_type in {"metric", "quantity"}:
        return "What exact quantity or comparison is communicated?"
    if proposition.proposition_type in {"result", "outcome", "takeaway"}:
        return "What result or outcome does the visual communicate?"
    if proposition.proposition_type in {"mechanism", "process"}:
        return "What mechanism or process is shown?"
    return "What concrete idea, state, or entity is visible?"


def _fact_statement(fact: dict[str, Any]) -> str:
    subject = _clean(fact.get("subject"), limit=100)
    predicate = _clean(fact.get("predicate"), limit=80)
    obj = _clean(fact.get("object"), limit=100)
    value = _clean(fact.get("value"), limit=60)
    unit = _clean(fact.get("unit"), limit=30)
    return " ".join(item for item in (subject, predicate, obj, value, unit) if item)


def _proposition_weight(obj: dict[str, Any], fact: dict[str, Any]) -> float:
    emphasis = _bounded(obj.get("emphasis"), 0.5)
    confidence = _bounded(fact.get("confidence"), 0.72)
    return max(0.45, min(1.4, 0.52 + emphasis * 0.48 + confidence * 0.28))


def _with_valid_dependencies(propositions: list[AtomicProposition]) -> list[AtomicProposition]:
    ids = {item.proposition_id for item in propositions}
    return [
        AtomicProposition(
            **{
                **item.to_dict(),
                "dependency_ids": [value for value in item.dependency_ids if value in ids and value != item.proposition_id],
            }
        )
        for item in propositions
    ]


def _temporal_sequence(payload: dict[str, Any], objects: dict[str, dict[str, Any]]) -> list[str]:
    ordered = sorted(
        [item for item in payload.get("beats") or [] if isinstance(item, dict)],
        key=lambda item: (_number(item.get("start_fraction"), 0.0), str(item.get("beat_id") or "")),
    )
    values: list[str] = []
    for beat in ordered:
        subject = objects.get(str(beat.get("subject_id") or ""), {})
        target = objects.get(str(beat.get("target_id") or ""), {})
        subject_label = _clean(subject.get("label") or subject.get("meaning"), limit=120)
        target_label = _clean(target.get("label") or target.get("meaning"), limit=120)
        action = _clean(beat.get("action"), limit=80).replace("_", " ")
        value = " ".join(item for item in (subject_label, action, target_label) if item)
        if value:
            values.append(value)
    return _unique(values, limit=12)


def _required_terms(payload: dict[str, Any], propositions: list[AtomicProposition]) -> list[str]:
    explicit = _strings(payload.get("required_labels"), limit=12)
    numeric = [number for item in propositions for number in item.exact_numbers]
    concise = [
        value
        for value in explicit
        if len(_semantic_tokens(value)) <= 8 or _numbers(value)
    ]
    return _unique([*concise, *numeric], limit=16)


def _best_semantic_match(value: str, candidates: list[str]) -> tuple[str, float]:
    best = ""
    score = 0.0
    for candidate in candidates:
        current = semantic_text_score(value, candidate)
        if current > score:
            best = candidate
            score = current
    return best, score


def _semantic_tokens(value: Any) -> list[str]:
    tokens: list[str] = []
    for token in _WORD_RE.findall(str(value or "").lower()):
        if token in _STOPWORDS:
            continue
        canonical = _SYNONYM_CANONICAL.get(token, _stem(token))
        if canonical and canonical not in _STOPWORDS:
            tokens.append(canonical)
    return tokens


def _stem(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 3:
            return token[: -len(suffix)]
    return token


def _number_alignment(first: Any, second: Any) -> float:
    left = set(_numbers(first))
    right = set(_numbers(second))
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _numbers(value: Any) -> list[str]:
    text = str(value or "").lower()
    values = [re.sub(r"\s+", "", item.lower()) for item in _NUMBER_RE.findall(text)]
    words = _WORD_RE.findall(text)
    for index, token in enumerate(words):
        number = _NUMBER_WORDS.get(token)
        if number is None:
            continue
        unit = _NUMBER_UNITS.get(words[index + 1], "") if index + 1 < len(words) else ""
        values.append(number + unit)
    return _unique(values, limit=20)


def _evidence_ids_for_facts(fact_ids: list[str], facts: dict[str, dict[str, Any]]) -> list[str]:
    return [
        evidence_id
        for fact_id in fact_ids
        for evidence_id in _strings(facts.get(fact_id, {}).get("evidence_ids"), limit=8)
    ]


def _signature(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _strings(value: Any, *, limit: int) -> list[str]:
    if isinstance(value, str):
        values = [value]
    else:
        values = list(value or [])
    return _unique([_clean(item, limit=500) for item in values], limit=limit)


def _unique(values: Iterable[str], *, limit: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = str(value or "").strip()
        normalized = " ".join(cleaned.lower().split())
        if not cleaned or normalized in seen:
            continue
        seen.add(normalized)
        result.append(cleaned)
        if len(result) >= limit:
            break
    return result


def _clean(value: Any, *, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit].strip()


def _number(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _bounded(value: Any, default: float) -> float:
    return max(0.0, min(_number(value, default), 1.0))


__all__ = [
    "COMMUNICATION_CONTRACT_VERSION",
    "COMMUNICATION_EVALUATION_VERSION",
    "AtomicProposition",
    "CommunicationContract",
    "CommunicationEvaluation",
    "PropositionResult",
    "ViewerQuestion",
    "build_communication_contract",
    "communication_contract_signature",
    "evaluate_viewer_answers",
    "semantic_sequence_score",
    "semantic_text_score",
    "validate_communication_contract",
]
