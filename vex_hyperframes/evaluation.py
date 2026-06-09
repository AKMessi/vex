from __future__ import annotations

import html as html_module
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


STRUCTURAL_TERMS = {
    "after",
    "before",
    "cause",
    "effect",
    "evidence",
    "from",
    "how",
    "mechanism",
    "problem",
    "proof",
    "result",
    "solution",
    "step",
    "to",
    "why",
}
GENERIC_PLACEHOLDER_LABELS = {
    "action",
    "context",
    "core idea",
    "core loop",
    "decision layer",
    "focus",
    "hidden layer",
    "input",
    "input captured",
    "model scores context",
    "outcome",
    "output",
    "primary signal",
    "result",
    "signal",
    "signal becomes visible",
    "start",
    "surface signal",
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
    r"\s*(?P<unit>%|x|ms|s|sec|seconds?|kb|mb|gb|tb|k|m|b|tokens?|parameters?|users?|steps?)?"
    r"(?![A-Za-z0-9.])",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class SemanticFixture:
    case_id: str
    transcript: str
    context: str
    expected_action: str
    expected_scene_type: str
    semantic_frame: dict[str, Any] = field(default_factory=dict)
    metric_facts: list[dict[str, str]] = field(default_factory=list)
    required_labels: list[str] = field(default_factory=list)
    forbidden_labels: list[str] = field(default_factory=list)
    allowed_copy: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SemanticFixture":
        return cls(
            case_id=str(payload.get("case_id") or "").strip(),
            transcript=str(payload.get("transcript") or "").strip(),
            context=str(payload.get("context") or "").strip(),
            expected_action=str(payload.get("expected_action") or "render").strip().lower(),
            expected_scene_type=str(payload.get("expected_scene_type") or "").strip().lower(),
            semantic_frame=dict(payload.get("semantic_frame") or {}),
            metric_facts=[
                {"value": str(item.get("value") or "").strip(), "label": str(item.get("label") or "").strip()}
                for item in (payload.get("metric_facts") or [])
                if isinstance(item, dict)
            ],
            required_labels=[str(item).strip() for item in (payload.get("required_labels") or []) if str(item).strip()],
            forbidden_labels=[str(item).strip() for item in (payload.get("forbidden_labels") or []) if str(item).strip()],
            allowed_copy=[str(item).strip() for item in (payload.get("allowed_copy") or []) if str(item).strip()],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SemanticEvaluation:
    case_id: str
    passed: bool
    score: float
    action_matched: bool
    required_coverage: float
    provenance_coverage: float
    invented_numeric_facts: list[str] = field(default_factory=list)
    missing_required_labels: list[str] = field(default_factory=list)
    forbidden_labels_found: list[str] = field(default_factory=list)
    generic_labels_found: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["score"] = round(float(self.score), 4)
        payload["required_coverage"] = round(float(self.required_coverage), 4)
        payload["provenance_coverage"] = round(float(self.provenance_coverage), 4)
        return payload


def load_semantic_fixtures(path: str | Path) -> list[SemanticFixture]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("HyperFrames semantic fixture corpus must be a JSON array.")
    fixtures = [SemanticFixture.from_dict(item) for item in payload if isinstance(item, dict)]
    case_ids = [fixture.case_id for fixture in fixtures]
    if not fixtures:
        raise ValueError("HyperFrames semantic fixture corpus is empty.")
    if any(not case_id for case_id in case_ids):
        raise ValueError("Every HyperFrames semantic fixture must define case_id.")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("HyperFrames semantic fixture case_id values must be unique.")
    if any(fixture.expected_action not in {"render", "reject"} for fixture in fixtures):
        raise ValueError("HyperFrames semantic fixtures support expected_action render or reject.")
    return fixtures


def visible_text_from_html(html: str) -> str:
    visible = re.sub(r"<head\b[^>]*>.*?</head>", " ", str(html or ""), flags=re.DOTALL | re.IGNORECASE)
    visible = re.sub(r"<(?:script|style)\b[^>]*>.*?</(?:script|style)>", " ", visible, flags=re.DOTALL | re.IGNORECASE)
    visible = re.sub(r"<[^>]+>", " ", visible)
    return re.sub(r"\s+", " ", html_module.unescape(visible)).strip()


def evaluate_semantic_output(
    fixture: SemanticFixture,
    *,
    html: str = "",
    selected: bool = True,
) -> SemanticEvaluation:
    expected_selected = fixture.expected_action == "render"
    action_matched = bool(selected) == expected_selected
    if not selected:
        issues = [] if action_matched else ["renderer_rejected_a_required_visual"]
        return SemanticEvaluation(
            case_id=fixture.case_id,
            passed=action_matched,
            score=1.0 if action_matched else 0.0,
            action_matched=action_matched,
            required_coverage=1.0 if action_matched else 0.0,
            provenance_coverage=1.0 if action_matched else 0.0,
            issues=issues,
        )

    visible_text = visible_text_from_html(html)
    normalized_visible = _normalize_phrase(visible_text)
    allowed_text = _allowed_source_text(fixture)
    allowed_tokens = set(_content_tokens(allowed_text))
    visible_tokens = _content_tokens(visible_text)
    grounded_tokens = [token for token in visible_tokens if token in allowed_tokens or token in STRUCTURAL_TERMS]
    provenance_coverage = len(grounded_tokens) / max(len(visible_tokens), 1)

    missing_required = [
        label for label in fixture.required_labels if not _phrase_is_present(label, normalized_visible)
    ]
    required_coverage = (
        1.0 - (len(missing_required) / len(fixture.required_labels))
        if fixture.required_labels
        else 1.0
    )
    forbidden_found = [
        label for label in fixture.forbidden_labels if _phrase_is_present(label, normalized_visible)
    ]
    allowed_normalized = _normalize_phrase(allowed_text)
    generic_found = [
        label
        for label in sorted(GENERIC_PLACEHOLDER_LABELS)
        if _phrase_is_present(label, normalized_visible) and not _phrase_is_present(label, allowed_normalized)
    ]
    allowed_numbers = set(_numeric_facts(allowed_text))
    invented_numbers = [
        value
        for value in _numeric_facts(visible_text)
        if value not in allowed_numbers and not _is_structural_number(value)
    ]

    issues: list[str] = []
    if not action_matched:
        issues.append("renderer_selected_a_fixture_that_should_be_rejected")
    if missing_required:
        issues.append("required_semantic_labels_missing")
    if provenance_coverage < 0.85:
        issues.append("visible_copy_is_not_sufficiently_grounded")
    if invented_numbers:
        issues.append("render_contains_invented_numeric_facts")
    if forbidden_found:
        issues.append("render_contains_forbidden_copy")
    if generic_found:
        issues.append("render_contains_generic_placeholder_copy")
    if not visible_text:
        issues.append("render_has_no_visible_semantic_copy")

    numeric_score = 1.0 if not invented_numbers else 0.0
    forbidden_score = 1.0 if not forbidden_found and not generic_found else 0.0
    score = (
        required_coverage * 0.42
        + provenance_coverage * 0.36
        + numeric_score * 0.14
        + forbidden_score * 0.08
    )
    passed = (
        action_matched
        and required_coverage >= 0.8
        and provenance_coverage >= 0.85
        and not invented_numbers
        and not forbidden_found
        and not generic_found
        and bool(visible_text)
    )
    return SemanticEvaluation(
        case_id=fixture.case_id,
        passed=passed,
        score=score,
        action_matched=action_matched,
        required_coverage=required_coverage,
        provenance_coverage=provenance_coverage,
        invented_numeric_facts=invented_numbers,
        missing_required_labels=missing_required,
        forbidden_labels_found=forbidden_found,
        generic_labels_found=generic_found,
        issues=issues,
    )


def _allowed_source_text(fixture: SemanticFixture) -> str:
    semantic_values = [
        str(value)
        for value in fixture.semantic_frame.values()
        if isinstance(value, (str, int, float)) and str(value).strip()
    ]
    metric_values = [
        str(value)
        for item in fixture.metric_facts
        for value in (item.get("value"), item.get("label"))
        if str(value or "").strip()
    ]
    return " ".join(
        [
            fixture.transcript,
            fixture.context,
            *semantic_values,
            *metric_values,
            *fixture.required_labels,
            *fixture.allowed_copy,
        ]
    )


def _normalize_phrase(value: str) -> str:
    return re.sub(r"[^a-z0-9%+./-]+", " ", str(value or "").lower()).strip()


def _phrase_is_present(phrase: str, normalized_haystack: str) -> bool:
    normalized_phrase = _normalize_phrase(phrase)
    if not normalized_phrase:
        return False
    if normalized_phrase in normalized_haystack:
        return True
    phrase_tokens = set(_content_tokens(normalized_phrase))
    haystack_tokens = set(_content_tokens(normalized_haystack))
    return bool(phrase_tokens) and len(phrase_tokens & haystack_tokens) / len(phrase_tokens) >= 0.8


def _content_tokens(value: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9%+./-]+", str(value or "").lower())
        if len(token) >= 2 and token not in STOPWORDS and not token.isdigit()
    ]


def _numeric_facts(value: str) -> list[str]:
    result: list[str] = []
    for match in NUMBER_PATTERN.finditer(str(value or "")):
        number = match.group("number")
        unit = (match.group("unit") or "").lower()
        normalized = f"{number}{unit}"
        if normalized not in result:
            result.append(normalized)
    return result


def _is_structural_number(value: str) -> bool:
    match = re.fullmatch(r"(\d+)(?:steps?)?", value)
    return bool(match and int(match.group(1)) <= 6)


__all__ = [
    "SemanticEvaluation",
    "SemanticFixture",
    "evaluate_semantic_output",
    "load_semantic_fixtures",
    "visible_text_from_html",
]
