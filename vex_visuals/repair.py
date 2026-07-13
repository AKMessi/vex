from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from enum import StrEnum
import hashlib
import json
import re
from typing import Any, Iterable

from vex_visuals.open_visual_program import sign_open_visual_program, validate_open_visual_program
from vex_visuals.verifier import VisualQualityState, VisualVerifierReport


VISUAL_REPAIR_VERSION = "vex-typed-visual-repair-v1"
VISUAL_REPAIR_ASSESSMENT_VERSION = "vex-visual-repair-assessment-v1"

_FILLER_COPY_RE = re.compile(
    r"^(?:basically|okay|ok|so|now|the next thing|very(?:\s+very)* interesting|"
    r"i can (?:just )?show you|as you can see|you know|right)[\s,.:;!?-]*$",
    re.IGNORECASE,
)


class RepairLevel(StrEnum):
    SEMANTIC_ENCODING = "semantic_encoding"
    COMPOSITION = "composition"
    MOTION_CAUSALITY = "motion_causality"
    COPY_FIDELITY = "copy_fidelity"
    EXECUTION = "execution"
    VERIFICATION = "verification"


@dataclass(frozen=True)
class TypedRepairOperation:
    operation_id: str
    level: RepairLevel
    operation: str
    reason: str
    target_id: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.8

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["level"] = self.level.value
        payload["confidence"] = round(float(self.confidence), 4)
        return payload


@dataclass(frozen=True)
class VisualRepairPlan:
    version: str
    repair_id: str
    round_index: int
    source_state: VisualQualityState
    requires_concept_regeneration: bool
    operations: list[TypedRepairOperation]
    signature: str

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "source_state": self.source_state.value,
            "operations": [item.to_dict() for item in self.operations],
        }


@dataclass(frozen=True)
class VisualRepairApplication:
    passed: bool
    changed: bool
    spec: dict[str, Any]
    applied_operation_ids: list[str]
    rejected_operations: list[dict[str, str]]
    validation: dict[str, Any]
    promoted_program_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RepairImprovementAssessment:
    version: str
    accepted: bool
    score_delta: float
    semantic_delta: float
    design_delta: float
    temporal_delta: float
    technical_delta: float
    state_improved: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "score_delta",
            "semantic_delta",
            "design_delta",
            "temporal_delta",
            "technical_delta",
        ):
            payload[key] = round(float(payload[key]), 4)
        return payload


def plan_visual_repair(
    report: VisualVerifierReport | dict[str, Any],
    spec: dict[str, Any],
    *,
    round_index: int,
    explore_alternate: bool = False,
) -> VisualRepairPlan:
    payload = report.to_dict() if isinstance(report, VisualVerifierReport) else dict(report or {})
    state = VisualQualityState(str(payload.get("state") or VisualQualityState.UNVERIFIED.value))
    operations: list[TypedRepairOperation] = []
    semantic = dict(payload.get("semantic") or {})
    design = dict(payload.get("design") or {})
    temporal = dict(payload.get("temporal") or {})
    technical = dict(payload.get("technical") or {})
    communication = dict(payload.get("communication") or {})
    if explore_alternate:
        operations.append(
            _operation(
                round_index,
                RepairLevel.SEMANTIC_ENCODING,
                "promote_alternate_concept",
                "Bounded test-time search requested an independently rendered concept.",
                confidence=0.86,
            )
        )
    if payload.get("available") is False:
        operations.append(
            _operation(
                round_index,
                RepairLevel.VERIFICATION,
                "retry_independent_verifier",
                "Independent rendered-visual verification was unavailable.",
                confidence=1.0,
            )
        )
    if not bool(semantic.get("passed", False)):
        operations.append(
            _operation(
                round_index,
                RepairLevel.SEMANTIC_ENCODING,
                "promote_alternate_concept",
                _issues_reason(semantic, "The visual thesis was not recoverable."),
                confidence=0.92,
            )
        )
    numeric_mismatch = any(
        "numeric_mismatch" in str(issue)
        for result in communication.get("results") or []
        if isinstance(result, dict)
        for issue in result.get("issues") or []
    )
    if numeric_mismatch or _has_filler_title(spec):
        operations.append(
            _operation(
                round_index,
                RepairLevel.COPY_FIDELITY,
                "replace_weak_copy",
                "Visible copy is conversational, ambiguous, or numerically unreliable.",
                confidence=0.95,
            )
        )
    if not bool(design.get("passed", False)):
        operations.extend(
            [
                _operation(
                    round_index,
                    RepairLevel.COMPOSITION,
                    "focus_hero_composition",
                    _issues_reason(design, "The composition lacks a dominant proof-bearing focal point."),
                    confidence=0.84,
                ),
                _operation(
                    round_index,
                    RepairLevel.COMPOSITION,
                    "reduce_explanatory_copy",
                    "Supporting copy should annotate the visual rather than become the visual.",
                    confidence=0.78,
                ),
            ]
        )
    if not bool(temporal.get("passed", False)):
        operations.extend(
            [
                _operation(
                    round_index,
                    RepairLevel.MOTION_CAUSALITY,
                    "strengthen_causal_motion",
                    _issues_reason(temporal, "Motion does not reveal the mechanism clearly enough."),
                    confidence=0.9,
                ),
                _operation(
                    round_index,
                    RepairLevel.MOTION_CAUSALITY,
                    "extend_final_hold",
                    "The resolved explanation needs a stable silent-reading hold.",
                    confidence=0.88,
                ),
            ]
        )
    if not bool(technical.get("passed", False)):
        operations.append(
            _operation(
                round_index,
                RepairLevel.EXECUTION,
                "repair_layout_bounds",
                _issues_reason(technical, "The render contains technical integrity defects."),
                confidence=0.96,
            )
        )
    operations = _dedupe_operations(operations)
    unsigned = {
        "version": VISUAL_REPAIR_VERSION,
        "repair_id": f"{_safe_id(spec.get('visual_id'))}-repair-{round_index:02d}",
        "round_index": int(round_index),
        "source_state": state.value,
        "requires_concept_regeneration": any(
            item.level == RepairLevel.SEMANTIC_ENCODING for item in operations
        ),
        "operations": [item.to_dict() for item in operations],
    }
    return VisualRepairPlan(
        version=VISUAL_REPAIR_VERSION,
        repair_id=unsigned["repair_id"],
        round_index=int(round_index),
        source_state=state,
        requires_concept_regeneration=bool(unsigned["requires_concept_regeneration"]),
        operations=operations,
        signature=_signature(unsigned),
    )


def apply_visual_repair(
    spec: dict[str, Any],
    plan: VisualRepairPlan,
    *,
    ir: dict[str, Any],
) -> VisualRepairApplication:
    working = copy.deepcopy(dict(spec or {}))
    original_program = copy.deepcopy(dict(working.get("open_visual_program") or {}))
    program = copy.deepcopy(original_program)
    applied: list[str] = []
    rejected: list[dict[str, str]] = []
    promoted_program_id = ""
    for operation in plan.operations:
        changed = False
        if operation.operation == "retry_independent_verifier":
            rejected.append(
                {
                    "operation_id": operation.operation_id,
                    "reason": "verification_retry_requires_render_or_provider_orchestrator",
                }
            )
            continue
        if operation.operation == "promote_alternate_concept":
            alternate = _alternate_program(working, program)
            if alternate is not None:
                program = alternate
                promoted_program_id = str(program.get("program_id") or "")
                changed = True
        elif operation.operation == "replace_weak_copy":
            changed = _replace_weak_copy(program, ir)
        elif operation.operation == "focus_hero_composition":
            changed = _focus_hero(program)
        elif operation.operation == "reduce_explanatory_copy":
            changed = _reduce_copy(program)
        elif operation.operation == "strengthen_causal_motion":
            changed = _strengthen_motion(program)
        elif operation.operation == "extend_final_hold":
            changed = _extend_final_hold(program)
        elif operation.operation == "repair_layout_bounds":
            changed = _repair_bounds(program)
        else:
            rejected.append(
                {
                    "operation_id": operation.operation_id,
                    "reason": "unsupported_visual_repair_operation",
                }
            )
            continue
        if changed:
            applied.append(operation.operation_id)
        else:
            rejected.append(
                {
                    "operation_id": operation.operation_id,
                    "reason": "visual_repair_operation_made_no_change",
                }
            )
    if not program:
        return VisualRepairApplication(
            passed=False,
            changed=False,
            spec=dict(spec),
            applied_operation_ids=[],
            rejected_operations=rejected,
            validation={"passed": False, "errors": ["visual_repair_has_no_open_visual_program"]},
        )
    program = sign_open_visual_program(program)
    validation = validate_open_visual_program(program, ir=dict(ir or {}))
    if not validation.passed:
        return VisualRepairApplication(
            passed=False,
            changed=False,
            spec=dict(spec),
            applied_operation_ids=[],
            rejected_operations=[
                *rejected,
                {
                    "operation_id": "validation",
                    "reason": ";".join(validation.errors[:8]),
                },
            ],
            validation=validation.to_dict(),
        )
    changed = program != original_program
    working["open_visual_program"] = program
    if promoted_program_id:
        tournament = dict(working.get("open_visual_tournament") or {})
        tournament["selected_program_id"] = promoted_program_id
        tournament["selection_mode"] = "typed_cegis_repair"
        working["open_visual_tournament"] = tournament
    repair_history = list(working.get("visual_repair_history") or [])
    repair_history.append(
        {
            "plan": plan.to_dict(),
            "applied_operation_ids": applied,
            "rejected_operations": rejected,
            "program_id": str(program.get("program_id") or ""),
        }
    )
    working["visual_repair_history"] = repair_history
    return VisualRepairApplication(
        passed=changed and bool(applied),
        changed=changed,
        spec=working,
        applied_operation_ids=applied,
        rejected_operations=rejected,
        validation=validation.to_dict(),
        promoted_program_id=promoted_program_id,
    )


def assess_repair_improvement(
    before: VisualVerifierReport,
    after: VisualVerifierReport,
    *,
    minimum_delta: float = 0.025,
) -> RepairImprovementAssessment:
    deltas = {
        "score": after.score - before.score,
        "semantic": after.semantic.score - before.semantic.score,
        "design": after.design.score - before.design.score,
        "temporal": after.temporal.score - before.temporal.score,
        "technical": after.technical.score - before.technical.score,
    }
    state_improved = _state_rank(after.state) > _state_rank(before.state)
    semantic_regressed = deltas["semantic"] < -0.02
    technical_regressed = deltas["technical"] < -0.02
    accepted = bool(
        after.publishable
        and not semantic_regressed
        and not technical_regressed
        and (state_improved or deltas["score"] >= float(minimum_delta))
    )
    reasons: list[str] = []
    if state_improved:
        reasons.append("quality_state_improved")
    if deltas["score"] >= float(minimum_delta):
        reasons.append("quality_score_improved")
    if semantic_regressed:
        reasons.append("semantic_quality_regressed")
    if technical_regressed:
        reasons.append("technical_quality_regressed")
    if not after.publishable:
        reasons.append("repaired_candidate_not_publishable")
    return RepairImprovementAssessment(
        version=VISUAL_REPAIR_ASSESSMENT_VERSION,
        accepted=accepted,
        score_delta=deltas["score"],
        semantic_delta=deltas["semantic"],
        design_delta=deltas["design"],
        temporal_delta=deltas["temporal"],
        technical_delta=deltas["technical"],
        state_improved=state_improved,
        reasons=_unique(reasons, limit=12),
    )


def _alternate_program(spec: dict[str, Any], current: dict[str, Any]) -> dict[str, Any] | None:
    current_id = str(current.get("program_id") or "")
    current_concept = str((current.get("quality_contract") or {}).get("visual_concept_id") or "")
    tried_program_ids = {
        str(item.get("program_id") or "")
        for item in spec.get("visual_repair_history") or []
        if isinstance(item, dict)
    }
    tried_program_ids.add(
        str((spec.get("open_visual_tournament") or {}).get("selected_program_id") or "")
    )
    tried_program_ids.add(current_id)
    candidates = [
        copy.deepcopy(dict(item))
        for item in spec.get("open_visual_program_candidates") or []
        if isinstance(item, dict)
        and str(item.get("program_id") or "") not in tried_program_ids
    ]
    candidates.sort(
        key=lambda item: (
            str((item.get("quality_contract") or {}).get("visual_concept_id") or "") != current_concept,
            str(item.get("program_id") or ""),
        ),
        reverse=True,
    )
    return candidates[0] if candidates else None


def _replace_weak_copy(program: dict[str, Any], ir: dict[str, Any]) -> bool:
    changed = False
    grounded_facts = [
        dict(item)
        for item in ir.get("facts") or []
        if isinstance(item, dict) and str(item.get("label") or "").strip()
    ]
    fallback = min(
        (str(item.get("label") or "").strip() for item in grounded_facts),
        key=len,
        default=str(ir.get("takeaway") or "").strip(),
    )[:120]
    for element in program.get("elements") or []:
        if not isinstance(element, dict):
            continue
        text = str(element.get("text") or "").strip()
        role = str(element.get("role") or "").lower()
        if text and (role == "title" or "headline" in role) and _is_filler_copy(text) and fallback:
            element["text"] = fallback
            changed = True
    return changed


def _focus_hero(program: dict[str, Any]) -> bool:
    elements = [
        item
        for item in program.get("elements") or []
        if isinstance(item, dict) and not bool(item.get("decorative")) and str(item.get("role") or "") != "title"
    ]
    if not elements:
        return False
    hero = max(
        elements,
        key=lambda item: (
            "result" in str(item.get("role") or "").lower(),
            _area(dict(item.get("layout") or {})),
        ),
    )
    layout = dict(hero.get("layout") or {})
    old = dict(layout)
    width = min(0.46, max(_number(layout.get("width"), 0.2) * 1.18, 0.2))
    height = min(0.46, max(_number(layout.get("height"), 0.2) * 1.15, 0.18))
    layout.update(
        {
            "x": min(max(_number(layout.get("x"), 0.5) - (width - _number(old.get("width"), width)) / 2, 0.05), 0.95 - width),
            "y": min(max(_number(layout.get("y"), 0.45) - (height - _number(old.get("height"), height)) / 2, 0.08), 0.92 - height),
            "width": width,
            "height": height,
        }
    )
    hero["layout"] = layout
    hero["style"] = {**dict(hero.get("style") or {}), "emphasis": 1.0}
    return layout != old


def _reduce_copy(program: dict[str, Any]) -> bool:
    changed = False
    for element in program.get("elements") or []:
        if not isinstance(element, dict) or str(element.get("type") or "") != "text":
            continue
        role = str(element.get("role") or "").lower()
        text = str(element.get("text") or "")
        if role != "title" and len(text) > 72:
            shortened = _shorten_grounded_copy(text, 72)
            if shortened != text:
                element["text"] = shortened
                changed = True
    return changed


def _strengthen_motion(program: dict[str, Any]) -> bool:
    tracks = [item for item in program.get("tracks") or [] if isinstance(item, dict)]
    elements = [item for item in program.get("elements") or [] if isinstance(item, dict)]
    result = next(
        (
            item
            for item in reversed(elements)
            if "result" in str(item.get("role") or "").lower()
            or "output" in str(item.get("role") or "").lower()
        ),
        elements[-1] if elements else None,
    )
    if result is None:
        return False
    target_id = str(result.get("element_id") or "")
    existing = next(
        (
            item
            for item in tracks
            if str(item.get("target_id") or "") == target_id
            and str(item.get("property") or "") == "emphasis"
        ),
        None,
    )
    if existing is not None:
        before = copy.deepcopy(existing.get("keyframes") or [])
        existing["semantic_intent"] = "resolve the causal transformation into the grounded outcome"
        existing["keyframes"] = [
            {"t": 0.58, "value": 0.0, "easing": "ease_in_out"},
            {"t": 0.78, "value": 1.0, "easing": "spring_gentle"},
            {"t": 1.0, "value": 0.82, "easing": "ease_out"},
        ]
        return existing["keyframes"] != before
    program.setdefault("tracks", []).append(
        {
            "track_id": f"{target_id}_causal_emphasis",
            "target_id": target_id,
            "property": "emphasis",
            "semantic_intent": "resolve the causal transformation into the grounded outcome",
            "keyframes": [
                {"t": 0.58, "value": 0.0, "easing": "ease_in_out"},
                {"t": 0.78, "value": 1.0, "easing": "spring_gentle"},
                {"t": 1.0, "value": 0.82, "easing": "ease_out"},
            ],
        }
    )
    return True


def _extend_final_hold(program: dict[str, Any]) -> bool:
    changed = False
    for track in program.get("tracks") or []:
        if not isinstance(track, dict):
            continue
        keyframes = [dict(item) for item in track.get("keyframes") or [] if isinstance(item, dict)]
        if not keyframes:
            continue
        keyframes.sort(key=lambda item: _number(item.get("t"), 0.0))
        final = keyframes[-1]
        if _number(final.get("t"), 0.0) < 1.0:
            keyframes.append({**final, "t": 1.0, "easing": "ease_out"})
            changed = True
        if len(keyframes) >= 2 and _number(keyframes[-2].get("t"), 0.0) > 0.84:
            keyframes[-2]["t"] = 0.82
            changed = True
        track["keyframes"] = keyframes
    return changed


def _repair_bounds(program: dict[str, Any]) -> bool:
    changed = False
    for element in program.get("elements") or []:
        if not isinstance(element, dict):
            continue
        layout = dict(element.get("layout") or {})
        old = dict(layout)
        width = min(max(_number(layout.get("width"), 0.2), 0.04), 0.9)
        height = min(max(_number(layout.get("height"), 0.2), 0.04), 0.86)
        layout.update(
            {
                "width": width,
                "height": height,
                "x": min(max(_number(layout.get("x"), 0.05), 0.05), 0.95 - width),
                "y": min(max(_number(layout.get("y"), 0.05), 0.05), 0.95 - height),
            }
        )
        element["layout"] = layout
        style = dict(element.get("style") or {})
        if "font_size" in style:
            style["font_size"] = max(22, min(_number(style.get("font_size"), 32), 110))
            element["style"] = style
        changed = changed or layout != old
    return changed


def _has_filler_title(spec: dict[str, Any]) -> bool:
    program = dict(spec.get("open_visual_program") or {})
    return any(
        _is_filler_copy(str(item.get("text") or ""))
        for item in program.get("elements") or []
        if isinstance(item, dict) and str(item.get("role") or "").lower() == "title"
    )


def _is_filler_copy(value: str) -> bool:
    cleaned = " ".join(str(value or "").split()).strip()
    if not cleaned:
        return False
    if _FILLER_COPY_RE.fullmatch(cleaned):
        return True
    normalized = re.sub(r"[^a-z0-9\s]", " ", cleaned.casefold())
    tokens = normalized.split()
    discourse_fillers = {"basically", "okay", "ok", "so", "now", "right", "well"}
    return bool(tokens and len(tokens) <= 5 and all(token in discourse_fillers for token in tokens))


def _shorten_grounded_copy(value: str, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    clause = re.split(r"(?<=[,;:])\s+|\s+(?:and then|because|which means)\s+", text, maxsplit=1)[0]
    words = clause.split()
    while words and len(" ".join(words)) > limit:
        words.pop()
    return " ".join(words).rstrip(" ,;:")


def _operation(
    round_index: int,
    level: RepairLevel,
    operation: str,
    reason: str,
    *,
    confidence: float,
) -> TypedRepairOperation:
    return TypedRepairOperation(
        operation_id=f"repair_{round_index:02d}_{level.value}_{operation}",
        level=level,
        operation=operation,
        reason=reason,
        confidence=confidence,
    )


def _issues_reason(dimension: dict[str, Any], fallback: str) -> str:
    issues = _unique([str(item) for item in dimension.get("issues") or []], limit=3)
    return "; ".join(issues) if issues else fallback


def _dedupe_operations(operations: Iterable[TypedRepairOperation]) -> list[TypedRepairOperation]:
    result: list[TypedRepairOperation] = []
    seen: set[str] = set()
    for item in operations:
        key = f"{item.level.value}:{item.operation}:{item.target_id}"
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _state_rank(state: VisualQualityState) -> int:
    return {
        VisualQualityState.REJECTED: 0,
        VisualQualityState.UNVERIFIED: 1,
        VisualQualityState.DEGRADED: 2,
        VisualQualityState.VERIFIED: 3,
    }[state]


def _area(layout: dict[str, Any]) -> float:
    return _number(layout.get("width"), 0.0) * _number(layout.get("height"), 0.0)


def _safe_id(value: Any) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(value or "visual")).strip("-")
    return cleaned[:80] or "visual"


def _signature(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _number(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _unique(values: Iterable[str], *, limit: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = str(value or "").strip()
        normalized = cleaned.lower()
        if not cleaned or normalized in seen:
            continue
        seen.add(normalized)
        result.append(cleaned)
        if len(result) >= limit:
            break
    return result


__all__ = [
    "VISUAL_REPAIR_ASSESSMENT_VERSION",
    "VISUAL_REPAIR_VERSION",
    "RepairImprovementAssessment",
    "RepairLevel",
    "TypedRepairOperation",
    "VisualRepairApplication",
    "VisualRepairPlan",
    "apply_visual_repair",
    "assess_repair_improvement",
    "plan_visual_repair",
]
