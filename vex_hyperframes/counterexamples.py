from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


COUNTEREXAMPLE_VERSION = "hyperframes-visual-counterexample-v1"
ISSUE_TYPES = {
    "ambiguous_thesis",
    "density",
    "hierarchy",
    "missing_object",
    "missing_relation",
    "motion",
    "overflow",
    "overlap",
    "pacing",
    "source_asset_required",
    "unsupported_content",
    "weak_grounding",
    "weak_relation_encoding",
}
SEVERITIES = {"info", "warning", "error", "hard_failure"}
ALLOWED_REPAIR_OPERATIONS = {
    "bind_source_asset",
    "change_layout_family",
    "move_element",
    "persist_element",
    "reduce_density",
    "remove_unsupported_content",
    "resize_element",
    "reroute_renderer",
    "retime_reveal",
    "strengthen_hierarchy",
    "strengthen_relation",
    "swap_proof_encoding",
}


@dataclass(frozen=True)
class VisualRegion:
    x: float
    y: float
    width: float
    height: float

    def to_dict(self) -> dict[str, float]:
        return {
            "x": round(float(self.x), 4),
            "y": round(float(self.y), 4),
            "width": round(float(self.width), 4),
            "height": round(float(self.height), 4),
        }


@dataclass(frozen=True)
class VisualCounterexample:
    counterexample_id: str
    critic: str
    issue_type: str
    severity: str
    summary: str
    expected: str
    observed: str
    confidence: float
    frame_id: str = ""
    timestamp_sec: float | None = None
    element_ids: list[str] = field(default_factory=list)
    relation_ids: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    regions: list[VisualRegion] = field(default_factory=list)
    allowed_repairs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["confidence"] = round(float(self.confidence), 4)
        if self.timestamp_sec is not None:
            payload["timestamp_sec"] = round(float(self.timestamp_sec), 3)
        payload["regions"] = [item.to_dict() for item in self.regions]
        return payload


@dataclass(frozen=True)
class CriticReport:
    critic: str
    available: bool
    passed: bool | None
    score: float | None
    counterexamples: list[VisualCounterexample] = field(default_factory=list)
    notes: str = ""
    model: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.score is not None:
            payload["score"] = round(float(self.score), 4)
        payload["counterexamples"] = [
            item.to_dict() for item in self.counterexamples
        ]
        return payload


@dataclass(frozen=True)
class VisualCriticBundle:
    version: str
    passed: bool
    score: float
    blind: CriticReport
    grounded: CriticReport
    design: CriticReport
    counterexamples: list[VisualCounterexample]
    hard_failure_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "passed": self.passed,
            "score": round(float(self.score), 4),
            "blind": self.blind.to_dict(),
            "grounded": self.grounded.to_dict(),
            "design": self.design.to_dict(),
            "counterexamples": [
                item.to_dict() for item in self.counterexamples
            ],
            "hard_failure_count": self.hard_failure_count,
        }


def parse_counterexamples(
    payload: Any,
    *,
    critic: str,
    scene_program: dict[str, Any],
    render_trace: dict[str, Any],
    limit: int = 16,
) -> list[VisualCounterexample]:
    raw_items = payload if isinstance(payload, list) else []
    valid_elements = {
        str(item.get("element_id") or ""): dict(item)
        for item in scene_program.get("elements") or []
        if isinstance(item, dict) and str(item.get("element_id") or "")
    }
    valid_relations = {
        str(item.get("relation_id") or ""): dict(item)
        for item in scene_program.get("relations") or []
        if isinstance(item, dict) and str(item.get("relation_id") or "")
    }
    capture_by_id = {
        str(item.get("capture_id") or ""): dict(item)
        for item in render_trace.get("captures") or []
        if isinstance(item, dict)
    }
    result: list[VisualCounterexample] = []
    for index, item in enumerate(raw_items, start=1):
        if not isinstance(item, dict):
            continue
        issue_type = str(item.get("issue_type") or "").strip().lower()
        severity = str(item.get("severity") or "").strip().lower()
        if issue_type not in ISSUE_TYPES or severity not in SEVERITIES:
            continue
        element_ids = [
            value
            for value in _strings(item.get("element_ids"))
            if value in valid_elements
        ]
        relation_ids = [
            value
            for value in _strings(item.get("relation_ids"))
            if value in valid_relations
        ]
        allowed_repairs = [
            value
            for value in _strings(item.get("allowed_repairs"))
            if value in ALLOWED_REPAIR_OPERATIONS
        ]
        if not allowed_repairs:
            allowed_repairs = _default_repairs(issue_type)
        frame_id = str(item.get("frame_id") or "").strip()
        capture = capture_by_id.get(frame_id, {})
        timestamp = _optional_float(item.get("timestamp_sec"))
        if timestamp is None:
            timestamp = _optional_float(capture.get("time_sec"))
        evidence_ids = _unique(
            [
                *_strings(item.get("evidence_ids")),
                *[
                    evidence_id
                    for element_id in element_ids
                    for evidence_id in _strings(
                        valid_elements[element_id].get("evidence_ids")
                    )
                ],
                *[
                    evidence_id
                    for relation_id in relation_ids
                    for evidence_id in _strings(
                        valid_relations[relation_id].get("evidence_ids")
                    )
                ],
            ]
        )
        result.append(
            VisualCounterexample(
                counterexample_id=str(
                    item.get("counterexample_id")
                    or f"{critic}_{index:02d}_{issue_type}"
                ),
                critic=critic,
                issue_type=issue_type,
                severity=severity,
                summary=_text(item.get("summary"), max_chars=180)
                or issue_type.replace("_", " "),
                expected=_text(item.get("expected"), max_chars=300),
                observed=_text(item.get("observed"), max_chars=300),
                confidence=_bounded(item.get("confidence"), 0.7),
                frame_id=frame_id if frame_id in capture_by_id else "",
                timestamp_sec=timestamp,
                element_ids=element_ids,
                relation_ids=relation_ids,
                evidence_ids=evidence_ids,
                regions=_parse_regions(item.get("regions")),
                allowed_repairs=allowed_repairs,
            )
        )
        if len(result) >= max(1, min(int(limit), 32)):
            break
    return result


def _parse_regions(value: Any) -> list[VisualRegion]:
    result: list[VisualRegion] = []
    for item in value or []:
        if not isinstance(item, dict):
            continue
        x = _bounded(item.get("x"), -1.0)
        y = _bounded(item.get("y"), -1.0)
        width = _bounded(item.get("width"), -1.0)
        height = _bounded(item.get("height"), -1.0)
        if (
            0.0 <= x <= 1.0
            and 0.0 <= y <= 1.0
            and 0.0 < width <= 1.0
            and 0.0 < height <= 1.0
        ):
            result.append(VisualRegion(x, y, width, height))
    return result[:6]


def _default_repairs(issue_type: str) -> list[str]:
    return {
        "ambiguous_thesis": ["strengthen_hierarchy", "swap_proof_encoding"],
        "density": ["reduce_density", "change_layout_family"],
        "hierarchy": ["strengthen_hierarchy", "resize_element"],
        "missing_object": ["persist_element", "change_layout_family"],
        "missing_relation": ["strengthen_relation", "swap_proof_encoding"],
        "motion": ["retime_reveal"],
        "overflow": ["resize_element", "reduce_density"],
        "overlap": ["move_element", "change_layout_family"],
        "pacing": ["retime_reveal"],
        "source_asset_required": ["bind_source_asset", "reroute_renderer"],
        "unsupported_content": ["remove_unsupported_content"],
        "weak_grounding": ["bind_source_asset", "remove_unsupported_content"],
        "weak_relation_encoding": [
            "strengthen_relation",
            "swap_proof_encoding",
        ],
    }.get(issue_type, [])


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return _unique([str(item) for item in value])


def _text(value: Any, *, max_chars: int) -> str:
    return " ".join(str(value or "").split()).strip()[:max_chars]


def _bounded(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(number, 1.0))


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
    "ALLOWED_REPAIR_OPERATIONS",
    "COUNTEREXAMPLE_VERSION",
    "CriticReport",
    "ISSUE_TYPES",
    "SEVERITIES",
    "VisualCounterexample",
    "VisualCriticBundle",
    "VisualRegion",
    "parse_counterexamples",
]
