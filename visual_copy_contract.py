from __future__ import annotations

import hashlib
import hmac
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


VISUAL_COPY_CONTRACT_VERSION = "visual-copy-contract-v1"

_WORD_RE = re.compile(r"[a-z0-9%+./-]+", re.IGNORECASE)
_NUMBER_RE = re.compile(
    r"(?<![a-z0-9.])(?P<number>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>%|percent|x|ms|milliseconds?|s|sec|seconds?|kb|mb|gb|tb|"
    r"tokens?|parameters?|users?|fps|hz)?(?![a-z0-9.])",
    re.IGNORECASE,
)
_VERSION_SEQUENCE_RE = re.compile(
    r"\b(?:v(?:ersion)?\s*)?\d+\.\d+"
    r"(?:\s*(?:and|or|/|,)\s*(?:v(?:ersion)?\s*)?\d+\.\d+)*"
    r"\s+(?:models?|releases?|versions?|series|checkpoints?)\b",
    re.IGNORECASE,
)
_METRIC_CONTEXT_RE = re.compile(
    r"\b(?:accuracy|average|baseline|blocks?|budget|cache|capacity|cost|count|"
    r"decrease[ds]?|difference|duration|faster|fps|gap|increase[ds]?|latency|"
    r"memory|milliseconds?|parameters?|percent|percentage|ratio|reduction|"
    r"score|seconds?|slower|speed|throughput|tokens?|total|users?)\b",
    re.IGNORECASE,
)
_COMPARISON_RE = re.compile(
    r"\b(?:at\s+the\s+cost\s+of|compared\s+to|decreases?|drops?|falls?|"
    r"from|increases?|less|more|only|ratio|rises?|to|versus|vs\.?)\b",
    re.IGNORECASE,
)
_LEADING_DISCOURSE_RE = re.compile(
    r"^(?:(?:okay|well|basically|actually|literally|obviously|honestly|"
    r"so|but|and|now|next|then|finally|after\s+that)\b[\s,:-]*)+",
    re.IGNORECASE,
)
_TRAILING_FRAGMENT_RE = re.compile(
    r"\b(?:a|an|the|and|or|but|as|at|by|for|from|in|into|of|on|per|than|"
    r"to|with|is|are|was|were|be|been|being|has|have|had|can|could|may|"
    r"might|must|should|would|will|takes?|replacing|controlling|this|that|"
    r"these|those|that['’]s|it['’]s)\s*$",
    re.IGNORECASE,
)
_UNRESOLVED_OPENING_RE = re.compile(
    r"^(?:it|its|this|that|these|those|they|them|their|both|he|him|his|she|"
    r"her|hers|we|our|you|your|i|my|where|why|who|what|how\s+much)\b",
    re.IGNORECASE,
)
_LEADING_PROCEDURAL_PRONOUN_RE = re.compile(
    r"^(?:it|they)\s+(?=(?:applies?|builds?|checks?|chooses?|classifies?|"
    r"compares?|compresses?|connects?|converts?|creates?|filters?|generates?|"
    r"groups?|maps?|passes?|picks?|reads?|renders?|routes?|scores?|selects?|"
    r"sends?|stores?|transforms?|updates?|validates?|writes?)\b)",
    re.IGNORECASE,
)
_LOW_SIGNAL_RE = re.compile(
    r"\b(?:about\s+the\s+hype|absolutely\s+not|caught\s+me\s+off\s+guard|"
    r"going\s+to\s+explain|interesting\s+thing|marketing\s+view|"
    r"let'?s\s+(?:inspect|look|talk)|"
    r"specific\s+thing|thought\s+of\s+sharing|who\s+knows)\b",
    re.IGNORECASE,
)
_PLACEHOLDER_NOUN_RE = re.compile(r"\b(?:thing|things|stuff)\b", re.IGNORECASE)
_ASR_DISCOURSE_SPLICE_RE = re.compile(
    r"[a-z0-9][,;]?\s+(?:And|But|No|So|Then|This|That)\b"
)
_GENERIC_COPY = {
    "action",
    "core idea",
    "core loop",
    "decision",
    "focus",
    "input",
    "mechanism",
    "output",
    "result",
    "signal",
    "system",
    "the thing",
    "this thing",
    "workflow",
}
_STOPWORDS = {
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
    "how",
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
    "which",
    "with",
    "you",
}


@dataclass(frozen=True)
class VisualCopyItem:
    copy_id: str
    role: str
    text: str
    binding_kind: str
    binding_id: str
    evidence_ids: list[str] = field(default_factory=list)
    grounding: str = "semantic_paraphrase"
    confidence: float = 0.72
    required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VisualCopyContract:
    version: str
    passed: bool
    title: str
    takeaway: str
    items: list[VisualCopyItem] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    signature: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "passed": self.passed,
            "title": self.title,
            "takeaway": self.takeaway,
            "items": [item.to_dict() for item in self.items],
            "issues": list(self.issues),
            "signature": self.signature,
        }


def build_visual_copy_contract(
    *,
    source_text: str,
    evidence: Iterable[Any],
    facts: Iterable[Any],
    objects: Iterable[Any],
    required_labels: Iterable[str],
    title_candidates: Iterable[str],
    takeaway_candidates: Iterable[str],
) -> VisualCopyContract:
    evidence_items = list(evidence)
    fact_items = list(facts)
    object_items = list(objects)
    required_values = list(required_labels)
    evidence_ids = [
        str(_field(item, "evidence_id"))
        for item in evidence_items
        if str(_field(item, "evidence_id"))
    ]
    facts_by_id = {
        str(_field(item, "fact_id")): item
        for item in fact_items
        if str(_field(item, "fact_id"))
    }
    items: list[VisualCopyItem] = []
    issues: list[str] = []
    canonical_required = {
        _normalize(label): normalize_display_copy(label)
        for label in required_values
        if normalize_display_copy(label)
    }

    for obj in object_items:
        object_id = str(_field(obj, "object_id"))
        linked_facts = [
            facts_by_id[fact_id]
            for fact_id in _string_list(_field(obj, "fact_ids"))
            if fact_id in facts_by_id
        ]
        candidate_values = [
            str(_field(obj, "label")),
            *[
                str(_field(fact, "value") or _field(fact, "label"))
                for fact in linked_facts
            ],
            str(_field(obj, "meaning")),
        ]
        selected = _select_copy(
            candidate_values,
            role="label",
            source_text=source_text,
        )
        if selected is None:
            selected = _select_derived_fact_copy(linked_facts, role="label")
        if selected is None:
            issues.append(f"object_has_no_publishable_copy:{object_id}")
            continue
        text, grounding, confidence = selected
        text = canonical_required.get(_normalize(text), text)
        fact_evidence = _unique(
            entry
            for fact in linked_facts
            for entry in _string_list(_field(fact, "evidence_ids"))
        )
        items.append(
            VisualCopyItem(
                copy_id=f"copy_object_{len(items) + 1:02d}",
                role="object_label",
                text=text,
                binding_kind="object",
                binding_id=object_id,
                evidence_ids=fact_evidence or evidence_ids,
                grounding=grounding,
                confidence=confidence,
            )
        )

    for label in required_values:
        selected = _select_copy([label], role="label", source_text=source_text)
        if selected is None:
            selected = _select_matching_derived_copy(
                label,
                fact_items=fact_items,
                object_items=object_items,
                role="label",
            )
        if selected is None:
            issues.append(f"required_label_is_not_publishable:{_issue_slug(label)}")
            continue
        text, grounding, confidence = selected
        if any(_normalize(item.text) == _normalize(text) for item in items):
            continue
        binding_kind, binding_id, binding_evidence = _best_binding(
            text,
            fact_items=fact_items,
            object_items=object_items,
            evidence_ids=evidence_ids,
        )
        if not binding_id:
            issues.append(f"required_label_has_no_claim_binding:{_issue_slug(text)}")
            continue
        items.append(
            VisualCopyItem(
                copy_id=f"copy_required_{len(items) + 1:02d}",
                role="required_label",
                text=text,
                binding_kind=binding_kind,
                binding_id=binding_id,
                evidence_ids=binding_evidence,
                grounding=grounding,
                confidence=confidence,
            )
        )

    title = ""
    for candidate in title_candidates:
        selected = _select_copy([candidate], role="title", source_text=source_text)
        if selected is None:
            selected = _select_matching_derived_copy(
                candidate,
                fact_items=fact_items,
                object_items=object_items,
                role="title",
            )
        if selected is None:
            continue
        text, grounding, confidence = selected
        if _NUMBER_RE.fullmatch(text.strip()):
            continue
        binding_kind, binding_id, binding_evidence = _best_binding(
            text,
            fact_items=fact_items,
            object_items=object_items,
            evidence_ids=evidence_ids,
        )
        if binding_kind != "fact" or not binding_id:
            continue
        title = text
        items.append(
            VisualCopyItem(
                copy_id="copy_title",
                role="title",
                text=text,
                binding_kind="fact",
                binding_id=binding_id,
                evidence_ids=binding_evidence,
                grounding=grounding,
                confidence=confidence,
            )
        )
        break
    if not title:
        issues.append("visual_copy_has_no_publishable_title")

    takeaway = ""
    for candidate in takeaway_candidates:
        selected = _select_copy([candidate], role="takeaway", source_text=source_text)
        if selected is None:
            continue
        text, grounding, confidence = selected
        binding_kind, binding_id, binding_evidence = _best_binding(
            text,
            fact_items=fact_items,
            object_items=object_items,
            evidence_ids=evidence_ids,
        )
        if not binding_id:
            continue
        takeaway = text
        if _normalize(text) != _normalize(title):
            items.append(
                VisualCopyItem(
                    copy_id="copy_takeaway",
                    role="takeaway",
                    text=text,
                    binding_kind=binding_kind,
                    binding_id=binding_id,
                    evidence_ids=binding_evidence,
                    grounding=grounding,
                    confidence=confidence,
                    required=False,
                )
            )
        break

    object_ids = {
        str(_field(item, "object_id"))
        for item in object_items
        if str(_field(item, "object_id"))
    }
    covered_objects = {
        item.binding_id
        for item in items
        if item.binding_kind == "object" and item.required
    }
    if object_ids - covered_objects:
        issues.append("visual_copy_omits_required_objects")

    issues = _unique(issues)
    payload = {
        "version": VISUAL_COPY_CONTRACT_VERSION,
        "passed": not issues,
        "title": title,
        "takeaway": takeaway,
        "items": [item.to_dict() for item in items],
        "issues": issues,
    }
    signature = _signature(payload)
    return VisualCopyContract(
        version=VISUAL_COPY_CONTRACT_VERSION,
        passed=not issues,
        title=title,
        takeaway=takeaway,
        items=items,
        issues=issues,
        signature=signature,
    )


def display_copy_issues(value: Any, *, role: str = "label") -> list[str]:
    text = _clean(value)
    if not text:
        return ["empty"]
    issues: list[str] = []
    words = _WORD_RE.findall(text)
    limits = {"title": (2, 10), "takeaway": (2, 16), "label": (1, 12)}
    minimum, maximum = limits.get(role, limits["label"])
    if not minimum <= len(words) <= maximum:
        issues.append("word_count")
    if "..." in text or text.endswith("\u2026"):
        issues.append("ellipsis")
    if _TRAILING_FRAGMENT_RE.search(text):
        issues.append("trailing_fragment")
    if _UNRESOLVED_OPENING_RE.search(text):
        issues.append("unresolved_opening")
    if _LOW_SIGNAL_RE.search(text):
        issues.append("low_signal_filler")
    if _PLACEHOLDER_NOUN_RE.search(text):
        issues.append("placeholder_noun")
    if _ASR_DISCOURSE_SPLICE_RE.search(text):
        issues.append("asr_discourse_splice")
    if _normalize(text) in _GENERIC_COPY:
        issues.append("generic_copy")
    normalized_words = [_normalize_word(word) for word in words]
    if any(
        normalized_words[index] == normalized_words[index - 1]
        for index in range(1, len(normalized_words))
    ):
        issues.append("repeated_word")
    content = [word for word in normalized_words if word not in _STOPWORDS and not word.isdigit()]
    if not content:
        issues.append("no_semantic_content")
    if re.search(r"\b(?:and|but|so)\s+(?:and|but|so)\b", text, re.IGNORECASE):
        issues.append("discourse_splice")
    return _unique(issues)


def normalize_display_copy(value: Any) -> str:
    text = _clean(value).strip(" \t\r\n,.;:-")
    text = _LEADING_DISCOURSE_RE.sub("", text).strip(" \t\r\n,.;:-")
    text = _LEADING_PROCEDURAL_PRONOUN_RE.sub("", text).strip(" \t\r\n,.;:-")
    return text


def metric_value_is_visual_measure(value: Any, label: Any, source_text: Any) -> bool:
    metric = _clean(value)
    source = _clean(source_text)
    if not metric or not source:
        return False
    normalized_metric = re.sub(r"\s+", "", metric.lower().replace("percent", "%"))
    number_match = re.match(r"\d+(?:\.\d+)?", metric)
    if number_match is None:
        return False
    target_number = number_match.group(0)
    matches = [
        match
        for match in _NUMBER_RE.finditer(source)
        if (
            re.sub(
                r"\s+",
                "",
                match.group(0).lower().replace("percent", "%"),
            )
            == normalized_metric
            or match.group("number") == target_number
        )
    ]
    if not matches:
        return False
    version_spans = [sequence.span() for sequence in _VERSION_SEQUENCE_RE.finditer(source)]
    matches = [
        match
        for match in matches
        if not any(start <= match.start() < end for start, end in version_spans)
    ]
    if not matches:
        return False
    label_text = _clean(label)
    if re.search(r"\b(?:model|release|version|checkpoint)\b", label_text, re.IGNORECASE):
        return False
    for match in matches:
        unit = str(match.group("unit") or "").lower()
        if unit:
            return True
        window = source[max(0, match.start() - 64) : match.end() + 64]
        if _METRIC_CONTEXT_RE.search(window) and (
            _COMPARISON_RE.search(window) or "." not in match.group("number")
        ):
            return True
    return False


def copy_allowed_for_binding(
    text: Any,
    contract: dict[str, Any],
    *,
    binding_kind: str,
    binding_id: str,
) -> bool:
    normalized = _normalize(text)
    if not normalized:
        return True
    if validate_visual_copy_contract(contract):
        return False
    return any(
        _normalize(item.get("text")) == normalized
        and str(item.get("binding_kind") or "") == binding_kind
        and str(item.get("binding_id") or "") == binding_id
        for item in contract.get("items") or []
        if isinstance(item, dict)
    )


def contract_copy(
    contract: dict[str, Any],
    *,
    role: str,
    binding_kind: str = "",
    binding_id: str = "",
) -> dict[str, Any]:
    if validate_visual_copy_contract(contract):
        return {}
    for item in contract.get("items") or []:
        if not isinstance(item, dict) or str(item.get("role") or "") != role:
            continue
        if binding_kind and str(item.get("binding_kind") or "") != binding_kind:
            continue
        if binding_id and str(item.get("binding_id") or "") != binding_id:
            continue
        return dict(item)
    return {}


def validate_visual_copy_contract(contract: dict[str, Any]) -> list[str]:
    if not isinstance(contract, dict) or not contract:
        return ["visual_copy_contract_missing"]
    issues: list[str] = []
    if str(contract.get("version") or "") != VISUAL_COPY_CONTRACT_VERSION:
        issues.append("visual_copy_contract_version_mismatch")
    if not bool(contract.get("passed")):
        issues.append("visual_copy_contract_not_passed")
    items = contract.get("items")
    if not isinstance(items, list):
        issues.append("visual_copy_contract_items_invalid")
        items = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            issues.append(f"visual_copy_item_invalid:{index}")
            continue
        role = str(item.get("role") or "")
        quality_role = role if role in {"title", "takeaway"} else "label"
        if display_copy_issues(item.get("text"), role=quality_role):
            issues.append(f"visual_copy_item_text_invalid:{index}")
        if not str(item.get("binding_kind") or "") or not str(item.get("binding_id") or ""):
            issues.append(f"visual_copy_item_binding_missing:{index}")
        if not _string_list(item.get("evidence_ids")):
            issues.append(f"visual_copy_item_evidence_missing:{index}")
    signed_payload = {
        "version": contract.get("version"),
        "passed": bool(contract.get("passed")),
        "title": str(contract.get("title") or ""),
        "takeaway": str(contract.get("takeaway") or ""),
        "items": items,
        "issues": list(contract.get("issues") or []),
    }
    if not str(contract.get("signature") or "") or not hmac.compare_digest(
        str(contract.get("signature") or ""),
        _signature(signed_payload),
    ):
        issues.append("visual_copy_contract_signature_invalid")
    return _unique(issues)


def _select_copy(
    candidates: Iterable[Any],
    *,
    role: str,
    source_text: str,
) -> tuple[str, str, float] | None:
    for candidate in candidates:
        text = normalize_display_copy(candidate)
        if display_copy_issues(text, role=role):
            continue
        grounding, confidence = _grounding(text, source_text)
        if grounding != "transcript_exact":
            continue
        return text, grounding, confidence
    return None


def _select_derived_fact_copy(
    facts: Iterable[Any],
    *,
    role: str,
) -> tuple[str, str, float] | None:
    for fact in facts:
        grounding = str(_field(fact, "grounding") or "")
        confidence = float(_field(fact, "confidence") or 0.0)
        if not _fact_copy_is_authorized(grounding, confidence):
            continue
        for value in (_field(fact, "label"), _field(fact, "value")):
            text = normalize_display_copy(value)
            if display_copy_issues(text, role=role):
                continue
            return text, grounding, confidence
    return None


def _select_matching_derived_copy(
    value: Any,
    *,
    fact_items: Iterable[Any],
    object_items: Iterable[Any],
    role: str,
) -> tuple[str, str, float] | None:
    text = normalize_display_copy(value)
    if display_copy_issues(text, role=role):
        return None
    normalized = _normalize(text)
    for fact in fact_items:
        grounding = str(_field(fact, "grounding") or "")
        confidence = float(_field(fact, "confidence") or 0.0)
        if not _fact_copy_is_authorized(grounding, confidence):
            continue
        if normalized in {
            _normalize(_field(fact, "label")),
            _normalize(_field(fact, "value")),
        }:
            return text, grounding, confidence
    for obj in object_items:
        if normalized not in {
            _normalize(_field(obj, "label")),
            _normalize(_field(obj, "meaning")),
        }:
            continue
        linked = set(_string_list(_field(obj, "fact_ids")))
        fact = next(
            (
                item
                for item in fact_items
                if str(_field(item, "fact_id")) in linked
                and _fact_copy_is_authorized(
                    str(_field(item, "grounding") or ""),
                    float(_field(item, "confidence") or 0.0),
                )
            ),
            None,
        )
        if fact is not None:
            grounding = str(_field(fact, "grounding") or "semantic_derived")
            return text, grounding, float(_field(fact, "confidence") or 0.68)
    return None


def _best_binding(
    text: str,
    *,
    fact_items: list[Any],
    object_items: list[Any],
    evidence_ids: list[str],
) -> tuple[str, str, list[str]]:
    text_tokens = set(_tokens(text))
    ranked: list[tuple[float, str, Any]] = []
    for fact in fact_items:
        fact_text = " ".join(
            str(_field(fact, key) or "")
            for key in ("label", "subject", "predicate", "object", "value")
        )
        score = _coverage(text_tokens, set(_tokens(fact_text)))
        if _normalize(text) == _normalize(_field(fact, "label")):
            score += 1.0
        ranked.append((score, "fact", fact))
    for obj in object_items:
        object_text = f"{_field(obj, 'label')} {_field(obj, 'meaning')}"
        score = _coverage(text_tokens, set(_tokens(object_text)))
        if _normalize(text) == _normalize(_field(obj, "label")):
            score += 0.9
        ranked.append((score, "object", obj))
    if not ranked:
        return "", "", []
    score, kind, selected = max(ranked, key=lambda item: item[0])
    if score < 0.5:
        return "", "", []
    if kind == "fact":
        return (
            kind,
            str(_field(selected, "fact_id")),
            _string_list(_field(selected, "evidence_ids")) or evidence_ids,
        )
    linked = _string_list(_field(selected, "fact_ids"))
    selected_evidence = _unique(
        evidence_id
        for fact in fact_items
        if str(_field(fact, "fact_id")) in linked
        for evidence_id in _string_list(_field(fact, "evidence_ids"))
    )
    return kind, str(_field(selected, "object_id")), selected_evidence or evidence_ids


def _grounding(text: str, source_text: str) -> tuple[str, float]:
    normalized = _normalize(text)
    source = _normalize(source_text)
    if normalized and normalized in source:
        return "transcript_exact", 0.98
    compact_metric = _compact_metric(text)
    if compact_metric and compact_metric in _compact_metric(source_text):
        return "transcript_exact", 0.98
    return "unverified", 0.0


def _fact_copy_is_authorized(grounding: str, confidence: float) -> bool:
    if grounding == "transcript_exact" or grounding.startswith("deterministic_"):
        return True
    if grounding == "transcript_paraphrase":
        return confidence >= 0.75
    return grounding == "semantic_derived" and confidence >= 0.68


def _coverage(expected: set[str], available: set[str]) -> float:
    if not expected:
        return 0.0
    return len(expected & available) / len(expected)


def _tokens(value: Any) -> list[str]:
    return [
        _stem(token.lower())
        for token in _WORD_RE.findall(str(value or ""))
        if len(token) >= 2 and token.lower() not in _STOPWORDS and not token.isdigit()
    ]


def _stem(token: str) -> str:
    replacements = {
        "caching": "cache",
        "copying": "copy",
        "nothing": "no",
        "rendering": "render",
        "retrieval": "retrieve",
        "validated": "validate",
        "validation": "validate",
    }
    if token in replacements:
        return replacements[token]
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def _field(item: Any, key: str) -> Any:
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return [str(value)] if value else []


def _normalize(value: Any) -> str:
    normalized = str(value or "").lower()
    normalized = re.sub(r"\bpercent\b", "%", normalized)
    normalized = re.sub(r"\bmilliseconds?\b", "ms", normalized)
    normalized = re.sub(r"\bseconds?|\bsec\b", "s", normalized)
    return " ".join(_WORD_RE.findall(normalized))


def _normalize_word(value: str) -> str:
    return str(value or "").lower().strip()


def _compact_metric(value: Any) -> str:
    normalized = str(value or "").lower()
    normalized = re.sub(r"\bpercent\b", "%", normalized)
    normalized = re.sub(r"\bmilliseconds?\b", "ms", normalized)
    normalized = re.sub(r"\bseconds?|\bsec\b", "s", normalized)
    matches = _NUMBER_RE.findall(normalized)
    return " ".join(f"{number}{unit}" for number, unit in matches)


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _issue_slug(value: Any) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")
    return slug[:48] or "empty"


def _signature(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = [
    "VISUAL_COPY_CONTRACT_VERSION",
    "VisualCopyContract",
    "VisualCopyItem",
    "build_visual_copy_contract",
    "contract_copy",
    "copy_allowed_for_binding",
    "display_copy_issues",
    "metric_value_is_visual_measure",
    "normalize_display_copy",
    "validate_visual_copy_contract",
]
