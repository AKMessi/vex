from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


REPAIR_LOOP_VERSION = "hyperframes-visual-cegis-v1"


@dataclass(frozen=True)
class CandidateSnapshot:
    variant_id: str
    composite_score: float
    quality_score: float
    semantic_score: float
    critic_score: float
    hard_failure_count: int
    issue_keys: list[str]
    object_coverage: float
    relation_coverage: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "composite_score",
            "quality_score",
            "semantic_score",
            "critic_score",
            "object_coverage",
            "relation_coverage",
        ):
            payload[key] = round(float(payload[key]), 4)
        return payload


@dataclass(frozen=True)
class MonotonicRepairDecision:
    accepted: bool
    reason: str
    score_delta: float
    hard_failure_delta: int
    new_issue_keys: list[str] = field(default_factory=list)
    resolved_issue_keys: list[str] = field(default_factory=list)
    before: CandidateSnapshot | None = None
    after: CandidateSnapshot | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "score_delta": round(float(self.score_delta), 4),
            "hard_failure_delta": self.hard_failure_delta,
            "new_issue_keys": list(self.new_issue_keys),
            "resolved_issue_keys": list(self.resolved_issue_keys),
            "before": self.before.to_dict() if self.before else None,
            "after": self.after.to_dict() if self.after else None,
        }


def snapshot_candidate(record: dict[str, Any]) -> CandidateSnapshot:
    qa = dict(record.get("qa") or {})
    metadata = dict(record.get("metadata") or {})
    semantic = dict(metadata.get("semantic_qa") or {})
    critics = dict(metadata.get("visual_critics") or {})
    vision = dict(metadata.get("vision_qa") or {})
    quality_score = _bounded(qa.get("score"))
    semantic_score = _bounded(semantic.get("score"))
    critic_score = _bounded(
        critics.get("score"),
        default=_bounded(vision.get("score"), default=quality_score),
    )
    hard_failures = [
        str(item)
        for item in semantic.get("hard_failures") or []
        if str(item).strip()
    ]
    counterexamples = [
        dict(item)
        for item in critics.get("counterexamples") or []
        if isinstance(item, dict)
    ]
    hard_counterexamples = [
        item
        for item in counterexamples
        if str(item.get("severity") or "") == "hard_failure"
    ]
    issue_keys = _unique(
        [
            *[f"semantic:{item}" for item in hard_failures],
            *[
                _counterexample_key(item)
                for item in counterexamples
                if str(item.get("severity") or "")
                in {"error", "hard_failure"}
            ],
        ]
    )
    stage = dict(metadata.get("stage") or {})
    object_coverage = _bounded(
        stage.get("object_coverage"),
        default=_bounded(semantic.get("object_coverage"), default=0.0),
    )
    relation_coverage = _bounded(
        stage.get("relation_coverage"),
        default=_bounded(vision.get("relation_coverage"), default=0.0),
    )
    composite_score = (
        quality_score * 0.4
        + semantic_score * 0.28
        + critic_score * 0.32
    )
    return CandidateSnapshot(
        variant_id=str(record.get("variant_id") or ""),
        composite_score=composite_score,
        quality_score=quality_score,
        semantic_score=semantic_score,
        critic_score=critic_score,
        hard_failure_count=len(hard_failures) + len(hard_counterexamples),
        issue_keys=issue_keys,
        object_coverage=object_coverage,
        relation_coverage=relation_coverage,
    )


def assess_monotonic_improvement(
    before_record: dict[str, Any],
    after_record: dict[str, Any],
    *,
    min_score_delta: float,
) -> MonotonicRepairDecision:
    before = snapshot_candidate(before_record)
    after = snapshot_candidate(after_record)
    score_delta = after.composite_score - before.composite_score
    hard_failure_delta = (
        before.hard_failure_count - after.hard_failure_count
    )
    new_issues = sorted(set(after.issue_keys) - set(before.issue_keys))
    resolved_issues = sorted(set(before.issue_keys) - set(after.issue_keys))
    coverage_regressed = (
        after.object_coverage + 0.001 < before.object_coverage
        or after.relation_coverage + 0.001 < before.relation_coverage
    )
    quality_regressed = after.quality_score + 0.025 < before.quality_score
    hard_failure_regressed = (
        after.hard_failure_count > before.hard_failure_count
    )
    if hard_failure_regressed:
        accepted = False
        reason = "hard_failure_count_increased"
    elif coverage_regressed:
        accepted = False
        reason = "semantic_coverage_regressed"
    elif quality_regressed:
        accepted = False
        reason = "render_quality_regressed"
    elif new_issues and hard_failure_delta <= 0:
        accepted = False
        reason = "repair_introduced_new_failures"
    elif hard_failure_delta > 0:
        accepted = True
        reason = "hard_failures_reduced"
    elif score_delta >= float(min_score_delta):
        accepted = True
        reason = "composite_score_improved"
    else:
        accepted = False
        reason = "repair_did_not_meet_minimum_improvement"
    return MonotonicRepairDecision(
        accepted=accepted,
        reason=reason,
        score_delta=score_delta,
        hard_failure_delta=hard_failure_delta,
        new_issue_keys=new_issues,
        resolved_issue_keys=resolved_issues,
        before=before,
        after=after,
    )


def _counterexample_key(item: dict[str, Any]) -> str:
    return "|".join(
        [
            "critic",
            str(item.get("issue_type") or ""),
            ",".join(str(value) for value in item.get("element_ids") or []),
            ",".join(str(value) for value in item.get("relation_ids") or []),
        ]
    )


def _bounded(value: Any, *, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(number, 1.0))


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
    "CandidateSnapshot",
    "MonotonicRepairDecision",
    "REPAIR_LOOP_VERSION",
    "assess_monotonic_improvement",
    "snapshot_candidate",
]
