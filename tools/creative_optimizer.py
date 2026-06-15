from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Any

from tools.automation import normalize_coverage_policy


CREATIVE_SET_OPTIMIZER_VERSION = "creative-set-optimizer-v1"


@dataclass(frozen=True)
class CandidateEvidence:
    candidate_id: str
    start: float
    end: float
    base_score: float
    intent_type: str
    renderer: str
    template: str
    episode_id: str
    beat_ids: tuple[str, ...]
    concept_ids: tuple[str, ...]
    medium_family: str
    canvas_system: str
    background_mode: str
    motion_choreography: str
    fingerprint_signature: str
    panel_ratio_target: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["beat_ids"] = list(self.beat_ids)
        payload["concept_ids"] = list(self.concept_ids)
        return payload


def optimize_creative_set(
    candidates: list[dict[str, Any]],
    *,
    budget: int,
    min_gap_sec: float = 0.2,
    phase: str = "plan",
    coverage_policy: str = "quality_only",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select a coherent visual portfolio instead of truncating candidates by time."""
    normalized_phase = "rendered" if str(phase).strip().lower() == "rendered" else "plan"
    policy = normalize_coverage_policy(coverage_policy)
    safe_budget = max(0, int(budget))
    safe_gap = max(0.0, _finite_float(min_gap_sec, 0.0))
    evidence: list[CandidateEvidence] = []
    candidate_ids: set[str] = set()
    for index, candidate in enumerate(candidates):
        item = _candidate_evidence(candidate, index=index, phase=normalized_phase)
        candidate_id = _unique_candidate_id(item.candidate_id, candidate_ids)
        candidate_ids.add(candidate_id)
        evidence.append(replace(item, candidate_id=candidate_id))
    evidence_by_id = {item.candidate_id: item for item in evidence}
    candidate_by_id = {
        item.candidate_id: dict(candidate)
        for item, candidate in zip(evidence, candidates, strict=True)
    }

    selected_ids: list[str] = []
    current_score = 0.0
    while len(selected_ids) < safe_budget:
        best_id = ""
        best_score = current_score
        for item in evidence:
            if item.candidate_id in selected_ids:
                continue
            if _conflicts_with_selection(
                item,
                selected_ids,
                evidence_by_id,
                min_gap_sec=safe_gap,
            ):
                continue
            proposed = [*selected_ids, item.candidate_id]
            proposed_score = _set_objective(
                proposed,
                evidence_by_id,
                coverage_policy=policy,
            )
            if _is_better_choice(
                proposed_score,
                item,
                best_score=best_score,
                best_id=best_id,
                evidence_by_id=evidence_by_id,
            ):
                best_id = item.candidate_id
                best_score = proposed_score
        if not best_id:
            break
        selected_ids.append(best_id)
        current_score = best_score

    selected_ids = _improve_selection(
        selected_ids,
        evidence,
        evidence_by_id,
        budget=safe_budget,
        min_gap_sec=safe_gap,
        coverage_policy=policy,
    )
    selected_ids.sort(key=lambda candidate_id: _chronological_key(evidence_by_id[candidate_id]))
    selected_set = set(selected_ids)
    objective_score = _set_objective(
        selected_ids,
        evidence_by_id,
        coverage_policy=policy,
    )

    selected: list[dict[str, Any]] = []
    selected_details: list[dict[str, Any]] = []
    for candidate_id in selected_ids:
        item = evidence_by_id[candidate_id]
        without = [other_id for other_id in selected_ids if other_id != candidate_id]
        contribution = max(
            0.0,
            objective_score
            - _set_objective(without, evidence_by_id, coverage_policy=policy),
        )
        candidate = candidate_by_id[candidate_id]
        candidate["creative_set_selection"] = {
            "version": CREATIVE_SET_OPTIMIZER_VERSION,
            "phase": normalized_phase,
            "base_score": round(item.base_score, 4),
            "portfolio_contribution": round(contribution, 4),
            "coverage_policy": policy,
        }
        selected.append(candidate)
        selected_details.append(
            {
                **item.to_dict(),
                "portfolio_contribution": round(contribution, 4),
            }
        )

    rejected: list[dict[str, Any]] = []
    for item in evidence:
        if item.candidate_id in selected_set:
            continue
        conflicts = [
            selected_id
            for selected_id in selected_ids
            if _timing_conflict(item, evidence_by_id[selected_id], min_gap_sec=safe_gap)
        ]
        if conflicts:
            reason = "timing_conflict_with_stronger_set"
        elif len(selected_ids) >= safe_budget:
            reason = "set_budget_outcompeted"
        else:
            reason = "lower_marginal_portfolio_value"
        rejected.append(
            {
                **item.to_dict(),
                "reason": reason,
                "conflicting_visual_ids": conflicts,
            }
        )

    metrics = _set_metrics(selected_ids, evidence, evidence_by_id, min_gap_sec=safe_gap)
    report = {
        "version": CREATIVE_SET_OPTIMIZER_VERSION,
        "phase": normalized_phase,
        "coverage_policy": policy,
        "input_count": len(candidates),
        "budget": safe_budget,
        "selected_count": len(selected),
        "rejected_count": len(rejected),
        "objective_score": round(objective_score, 4),
        "selected": selected_details,
        "rejected": rejected,
        "metrics": metrics,
    }
    return selected, report


def _candidate_evidence(
    candidate: dict[str, Any],
    *,
    index: int,
    phase: str,
) -> CandidateEvidence:
    director = _mapping(candidate.get("auto_visuals_director"))
    graph = _mapping(candidate.get("creative_graph_signals"))
    rendered_qa = _mapping(candidate.get("rendered_visual_qa"))
    candidate_id = str(
        candidate.get("visual_id")
        or candidate.get("candidate_id")
        or candidate.get("card_id")
        or f"candidate_{index + 1:03d}"
    ).strip()
    director_score = _bounded(_finite_float(director.get("director_score"), 55.0) / 100.0)
    copy_alignment = _bounded(director.get("copy_alignment"), 0.4)
    confidence = _bounded(candidate.get("confidence"), 0.55)
    graph_opportunity = _bounded(graph.get("graph_visual_opportunity"), 0.45)
    graph_retention = _bounded(graph.get("graph_retention_score"), 0.45)
    graph_topic = _bounded(graph.get("graph_topic_alignment"), 0.45)
    rendered_score = _bounded(rendered_qa.get("score"), 0.55)
    policy_prior = _mapping(candidate.get("creative_policy_prior"))
    renderer_metadata = _mapping(candidate.get("renderer_metadata"))
    visual_world = _mapping(
        candidate.get("visual_world_program")
        or renderer_metadata.get("visual_world_program")
    )
    fingerprint = _mapping(
        candidate.get("rendered_visual_fingerprint")
        or renderer_metadata.get("rendered_visual_fingerprint")
        or visual_world.get("fingerprint")
    )
    policy_adjustment = max(
        -0.04,
        min(_finite_float(policy_prior.get("selection_adjustment"), 0.0), 0.04),
    )
    if phase == "rendered":
        base_score = (
            rendered_score * 0.5
            + director_score * 0.2
            + copy_alignment * 0.08
            + graph_opportunity * 0.08
            + graph_retention * 0.08
            + graph_topic * 0.06
            + policy_adjustment
        )
    else:
        base_score = (
            director_score * 0.5
            + copy_alignment * 0.1
            + confidence * 0.1
            + graph_opportunity * 0.12
            + graph_retention * 0.1
            + graph_topic * 0.08
            + policy_adjustment
        )
    renderer = str(
        candidate.get("renderer")
        or candidate.get("renderer_hint")
        or director.get("renderer_policy")
        or "unknown"
    ).strip().lower()
    return CandidateEvidence(
        candidate_id=candidate_id,
        start=_finite_float(
            candidate.get("start")
            if candidate.get("start") is not None
            else candidate.get("start_sec"),
            0.0,
        ),
        end=_finite_float(
            candidate.get("end")
            if candidate.get("end") is not None
            else candidate.get("end_sec"),
            0.0,
        ),
        base_score=round(_bounded(base_score), 6),
        intent_type=str(candidate.get("visual_intent_type") or "unknown").strip().lower(),
        renderer=renderer,
        template=str(candidate.get("template") or "unknown").strip().lower(),
        episode_id=str(candidate.get("episode_id") or "").strip(),
        beat_ids=_string_tuple(
            graph.get("graph_beat_ids")
            or candidate.get("graph_beat_ids")
            or candidate.get("beat_ids")
        ),
        concept_ids=_string_tuple(candidate.get("concept_ids")),
        medium_family=str(
            visual_world.get("medium_family") or "unknown"
        ).strip().lower(),
        canvas_system=str(
            visual_world.get("canvas_system") or "unknown"
        ).strip().lower(),
        background_mode=str(
            visual_world.get("background_mode") or "unknown"
        ).strip().lower(),
        motion_choreography=str(
            visual_world.get("motion_choreography") or "unknown"
        ).strip().lower(),
        fingerprint_signature=str(
            fingerprint.get("signature") or ""
        ).strip().lower(),
        panel_ratio_target=_bounded(
            fingerprint.get("panel_ratio_target"),
            0.0,
        ),
    )


def _set_objective(
    selected_ids: list[str],
    evidence_by_id: dict[str, CandidateEvidence],
    *,
    coverage_policy: str,
) -> float:
    if not selected_ids:
        return 0.0
    selected = [evidence_by_id[candidate_id] for candidate_id in selected_ids]
    quality_weight = 1.12 if coverage_policy == "quality_only" else 1.0
    objective = sum(item.base_score for item in selected) * quality_weight
    objective += _novelty_reward(selected, "beat_ids", unit=0.11, cap=0.55)
    objective += _novelty_reward(selected, "concept_ids", unit=0.08, cap=0.4)
    objective += _novelty_reward(selected, "episode_id", unit=0.07, cap=0.28)
    objective += _novelty_reward(selected, "intent_type", unit=0.035, cap=0.18)
    objective += _novelty_reward(selected, "renderer", unit=0.02, cap=0.08)
    objective += _novelty_reward(
        selected,
        "medium_family",
        unit=0.085,
        cap=0.42,
    )
    objective += _novelty_reward(
        selected,
        "canvas_system",
        unit=0.055,
        cap=0.28,
    )
    objective += _novelty_reward(
        selected,
        "background_mode",
        unit=0.045,
        cap=0.24,
    )
    objective += _novelty_reward(
        selected,
        "motion_choreography",
        unit=0.035,
        cap=0.18,
    )
    objective -= _redundancy_penalty(selected)
    return round(objective, 8)


def _novelty_reward(
    selected: list[CandidateEvidence],
    field_name: str,
    *,
    unit: float,
    cap: float,
) -> float:
    values: set[str] = set()
    for item in selected:
        raw = getattr(item, field_name)
        if isinstance(raw, tuple):
            values.update(value for value in raw if value)
        elif raw and raw != "unknown":
            values.add(str(raw))
    return min(len(values) * unit, cap)


def _redundancy_penalty(selected: list[CandidateEvidence]) -> float:
    signatures: dict[tuple[str, str], int] = {}
    world_signatures: dict[tuple[str, str], int] = {}
    rendered_signatures: dict[str, int] = {}
    for item in selected:
        signature = (item.intent_type, item.template)
        signatures[signature] = signatures.get(signature, 0) + 1
        world_signature = (item.medium_family, item.background_mode)
        world_signatures[world_signature] = (
            world_signatures.get(world_signature, 0) + 1
        )
        if item.fingerprint_signature:
            rendered_signatures[item.fingerprint_signature] = (
                rendered_signatures.get(item.fingerprint_signature, 0) + 1
            )
    return (
        sum(max(0, count - 1) * 0.055 for count in signatures.values())
        + sum(
            max(0, count - 1) * 0.13
            for signature, count in world_signatures.items()
            if "unknown" not in signature
        )
        + sum(
            max(0, count - 1) * 0.2
            for count in rendered_signatures.values()
        )
    )


def _improve_selection(
    selected_ids: list[str],
    evidence: list[CandidateEvidence],
    evidence_by_id: dict[str, CandidateEvidence],
    *,
    budget: int,
    min_gap_sec: float,
    coverage_policy: str,
) -> list[str]:
    selected = list(selected_ids)
    for _ in range(4):
        current_score = _set_objective(
            selected,
            evidence_by_id,
            coverage_policy=coverage_policy,
        )
        best_proposal: list[str] | None = None
        best_score = current_score
        selected_set = set(selected)
        for candidate in evidence:
            if candidate.candidate_id in selected_set:
                continue
            conflicts = [
                selected_id
                for selected_id in selected
                if _timing_conflict(
                    candidate,
                    evidence_by_id[selected_id],
                    min_gap_sec=min_gap_sec,
                )
            ]
            removal_options: list[list[str]] = []
            if conflicts:
                removal_options.append(conflicts)
            elif len(selected) >= budget:
                removal_options.extend([[selected_id] for selected_id in selected])
            else:
                removal_options.append([])
            for removals in removal_options:
                proposed = [
                    selected_id for selected_id in selected if selected_id not in removals
                ]
                proposed.append(candidate.candidate_id)
                if len(proposed) > budget or not _selection_is_compatible(
                    proposed,
                    evidence_by_id,
                    min_gap_sec=min_gap_sec,
                ):
                    continue
                proposed_score = _set_objective(
                    proposed,
                    evidence_by_id,
                    coverage_policy=coverage_policy,
                )
                if proposed_score > best_score + 1e-9:
                    best_score = proposed_score
                    best_proposal = proposed
                elif (
                    abs(proposed_score - best_score) <= 1e-9
                    and best_proposal is not None
                    and _selection_tie_key(proposed, evidence_by_id)
                    < _selection_tie_key(best_proposal, evidence_by_id)
                ):
                    best_proposal = proposed
        if best_proposal is None:
            break
        selected = best_proposal
    return selected


def _set_metrics(
    selected_ids: list[str],
    all_evidence: list[CandidateEvidence],
    evidence_by_id: dict[str, CandidateEvidence],
    *,
    min_gap_sec: float,
) -> dict[str, Any]:
    selected = [evidence_by_id[candidate_id] for candidate_id in selected_ids]
    all_beats = {value for item in all_evidence for value in item.beat_ids}
    selected_beats = {value for item in selected for value in item.beat_ids}
    all_concepts = {value for item in all_evidence for value in item.concept_ids}
    selected_concepts = {value for item in selected for value in item.concept_ids}
    ordered = sorted(selected, key=_chronological_key)
    gaps = [
        max(0.0, ordered[index + 1].start - ordered[index].end)
        for index in range(len(ordered) - 1)
    ]
    return {
        "average_base_score": round(
            sum(item.base_score for item in selected) / max(len(selected), 1),
            4,
        ),
        "beat_coverage": round(
            len(selected_beats) / max(len(all_beats), 1),
            4,
        ),
        "concept_coverage": round(
            len(selected_concepts) / max(len(all_concepts), 1),
            4,
        ),
        "unique_intents": len(
            {item.intent_type for item in selected if item.intent_type != "unknown"}
        ),
        "unique_renderers": len(
            {item.renderer for item in selected if item.renderer != "unknown"}
        ),
        "unique_media": len(
            {
                item.medium_family
                for item in selected
                if item.medium_family != "unknown"
            }
        ),
        "unique_canvases": len(
            {
                item.canvas_system
                for item in selected
                if item.canvas_system != "unknown"
            }
        ),
        "unique_backgrounds": len(
            {
                item.background_mode
                for item in selected
                if item.background_mode != "unknown"
            }
        ),
        "average_panel_ratio_target": round(
            sum(item.panel_ratio_target for item in selected)
            / max(len(selected), 1),
            4,
        ),
        "minimum_gap_sec": round(min(gaps), 4) if gaps else None,
        "required_gap_sec": round(min_gap_sec, 4),
    }


def _is_better_choice(
    proposed_score: float,
    item: CandidateEvidence,
    *,
    best_score: float,
    best_id: str,
    evidence_by_id: dict[str, CandidateEvidence],
) -> bool:
    if proposed_score > best_score + 1e-9:
        return True
    if abs(proposed_score - best_score) > 1e-9:
        return False
    if not best_id:
        return True
    return _chronological_key(item) < _chronological_key(evidence_by_id[best_id])


def _conflicts_with_selection(
    item: CandidateEvidence,
    selected_ids: list[str],
    evidence_by_id: dict[str, CandidateEvidence],
    *,
    min_gap_sec: float,
) -> bool:
    return any(
        _timing_conflict(item, evidence_by_id[selected_id], min_gap_sec=min_gap_sec)
        for selected_id in selected_ids
    )


def _selection_is_compatible(
    selected_ids: list[str],
    evidence_by_id: dict[str, CandidateEvidence],
    *,
    min_gap_sec: float,
) -> bool:
    ordered = sorted(
        (evidence_by_id[candidate_id] for candidate_id in selected_ids),
        key=_chronological_key,
    )
    return all(
        not _timing_conflict(left, right, min_gap_sec=min_gap_sec)
        for index, left in enumerate(ordered)
        for right in ordered[index + 1 :]
    )


def _timing_conflict(
    left: CandidateEvidence,
    right: CandidateEvidence,
    *,
    min_gap_sec: float,
) -> bool:
    return not (
        left.end + min_gap_sec <= right.start
        or right.end + min_gap_sec <= left.start
    )


def _selection_tie_key(
    selected_ids: list[str],
    evidence_by_id: dict[str, CandidateEvidence],
) -> tuple[tuple[float, float, str], ...]:
    return tuple(
        sorted(
            (_chronological_key(evidence_by_id[candidate_id]) for candidate_id in selected_ids)
        )
    )


def _chronological_key(item: CandidateEvidence) -> tuple[float, float, str]:
    return (item.start, item.end, item.candidate_id)


def _unique_candidate_id(candidate_id: str, existing: set[str]) -> str:
    base = candidate_id or "candidate"
    if base not in existing:
        return base
    suffix = 2
    while f"{base}#{suffix}" in existing:
        suffix += 1
    return f"{base}#{suffix}"


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _string_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        values = []
    return tuple(
        sorted(
            {
                str(item).strip()
                for item in values
                if str(item).strip()
            }
        )
    )


def _finite_float(value: object, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _bounded(value: object, default: float = 0.0) -> float:
    return max(0.0, min(_finite_float(value, default), 1.0))


__all__ = [
    "CREATIVE_SET_OPTIMIZER_VERSION",
    "CandidateEvidence",
    "optimize_creative_set",
]
