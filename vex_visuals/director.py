from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from vex_visuals.repair import (
    RepairImprovementAssessment,
    apply_visual_repair,
    assess_repair_improvement,
    plan_visual_repair,
)
from vex_visuals.verifier import (
    PairwiseVisualTournament,
    VisualCandidateEvidence,
    VisualQualityState,
    VisualVerifierReport,
    VisionRequest,
    run_pairwise_visual_tournament,
    run_visual_verifier,
)


VISUAL_DIRECTOR_RUNTIME_VERSION = "vex-visual-director-runtime-v1"


class LocalQualityResult(Protocol):
    passed: bool
    score: float
    issues: list[str]

    def to_dict(self) -> dict[str, Any]: ...


RenderCandidate = Callable[[dict[str, Any], int], tuple[Any, str]]
EvaluateLocalQuality = Callable[[dict[str, Any], Any], LocalQualityResult]
ExtractCandidateFrames = Callable[[dict[str, Any], Any, str], Iterable[Path]]


@dataclass(frozen=True)
class DirectedCandidate:
    candidate_id: str
    round_index: int
    spec: dict[str, Any]
    asset: Any
    selection_reason: str
    local_quality: LocalQualityResult
    frame_paths: list[Path]
    verification: VisualVerifierReport
    publication_ready: bool
    hard_local_issues: list[str] = field(default_factory=list)
    repair_assessment: RepairImprovementAssessment | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "round_index": self.round_index,
            "program_id": str(
                (self.spec.get("open_visual_program") or {}).get("program_id")
                or ""
            ),
            "renderer": str(getattr(self.asset, "renderer", "") or ""),
            "asset_path": str(getattr(self.asset, "asset_path", "") or ""),
            "selection_reason": self.selection_reason,
            "local_quality": self.local_quality.to_dict(),
            "frame_paths": [str(path) for path in self.frame_paths],
            "verification": self.verification.to_dict(),
            "publication_ready": self.publication_ready,
            "hard_local_issues": list(self.hard_local_issues),
            "repair_assessment": (
                self.repair_assessment.to_dict()
                if self.repair_assessment is not None
                else None
            ),
        }


@dataclass(frozen=True)
class VisualDirectionOutcome:
    version: str
    passed: bool
    selected: DirectedCandidate
    candidates: list[DirectedCandidate]
    tournament: PairwiseVisualTournament | None
    repair_history: list[dict[str, Any]] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "passed": self.passed,
            "selected_candidate_id": self.selected.candidate_id,
            "selected_quality_state": self.selected.verification.state.value,
            "selected_verifier_score": round(
                float(self.selected.verification.score), 4
            ),
            "selected_local_score": round(float(self.selected.local_quality.score), 4),
            "candidates": [candidate.summary() for candidate in self.candidates],
            "tournament": self.tournament.to_dict() if self.tournament else None,
            "repair_history": list(self.repair_history),
            "issues": list(self.issues),
            "warnings": list(self.warnings),
        }


def direct_rendered_visual(
    spec: dict[str, Any],
    initial_asset: Any,
    initial_selection_reason: str,
    *,
    ir: dict[str, Any],
    contract: dict[str, Any],
    render_candidate: RenderCandidate,
    evaluate_local_quality: EvaluateLocalQuality,
    extract_candidate_frames: ExtractCandidateFrames,
    strict: bool,
    max_repair_rounds: int = 2,
    minimum_repair_delta: float = 0.025,
    provider_models: Iterable[tuple[str, str]] | None = None,
    vision_request: VisionRequest | None = None,
    cache_dir: Path | None = None,
    pairwise_top_k: int = 3,
    target_publishable_candidates: int = 1,
) -> VisualDirectionOutcome:
    """Verify a render, repair counterexamples, and select only grounded candidates.

    The orchestrator intentionally knows nothing about Remotion or HyperFrames. Renderer
    adapters own rendering, local structural QA, and frame extraction; this boundary owns
    semantic publication policy and monotonic counterexample-guided search.
    """

    candidates: list[DirectedCandidate] = []
    repair_history: list[dict[str, Any]] = []
    current = _evaluate_candidate(
        spec,
        initial_asset,
        initial_selection_reason,
        round_index=0,
        contract=contract,
        evaluate_local_quality=evaluate_local_quality,
        extract_candidate_frames=extract_candidate_frames,
        strict=strict,
        provider_models=provider_models,
        vision_request=vision_request,
        cache_dir=cache_dir,
    )
    candidates.append(current)

    search_target = max(1, min(int(target_publishable_candidates), 3))
    for round_index in range(1, max(0, min(int(max_repair_rounds), 4)) + 1):
        publishable_count = sum(item.publication_ready for item in candidates)
        exploring = bool(
            current.publication_ready
            and current.verification.available
            and current.verification.verified
            and publishable_count < search_target
        )
        if current.publication_ready and not exploring:
            break
        plan = plan_visual_repair(
            current.verification,
            current.spec,
            round_index=round_index,
            explore_alternate=exploring,
        )
        application = apply_visual_repair(current.spec, plan, ir=ir)
        round_record: dict[str, Any] = {
            "round_index": round_index,
            "mode": "alternate_concept_search" if exploring else "counterexample_repair",
            "plan": plan.to_dict(),
            "application": {
                "passed": application.passed,
                "changed": application.changed,
                "applied_operation_ids": list(application.applied_operation_ids),
                "rejected_operations": list(application.rejected_operations),
                "validation": dict(application.validation),
                "promoted_program_id": application.promoted_program_id,
            },
        }
        if not application.passed:
            round_record["accepted_for_next_round"] = False
            round_record["stop_reason"] = "repair_program_was_not_renderable"
            repair_history.append(round_record)
            break
        try:
            repaired_asset, repaired_reason = render_candidate(
                application.spec,
                round_index,
            )
        except Exception as exc:  # noqa: BLE001
            round_record["accepted_for_next_round"] = False
            round_record["stop_reason"] = "repair_render_failed"
            round_record["render_error"] = f"{type(exc).__name__}: {exc}"
            repair_history.append(round_record)
            break
        repaired = _evaluate_candidate(
            application.spec,
            repaired_asset,
            repaired_reason,
            round_index=round_index,
            contract=contract,
            evaluate_local_quality=evaluate_local_quality,
            extract_candidate_frames=extract_candidate_frames,
            strict=strict,
            provider_models=provider_models,
            vision_request=vision_request,
            cache_dir=cache_dir,
        )
        assessment = assess_repair_improvement(
            current.verification,
            repaired.verification,
            minimum_delta=minimum_repair_delta,
        )
        repaired = DirectedCandidate(
            **{
                **repaired.__dict__,
                "repair_assessment": assessment,
            }
        )
        candidates.append(repaired)
        progressed = (
            repaired.publication_ready
            if exploring
            else _monotonic_progress(
                current,
                repaired,
                assessment,
                minimum_delta=minimum_repair_delta,
            )
        )
        round_record["assessment"] = assessment.to_dict()
        round_record["candidate_id"] = repaired.candidate_id
        round_record["accepted_for_next_round"] = progressed
        repair_history.append(round_record)
        if not progressed:
            break
        current = repaired

    publishable = [candidate for candidate in candidates if candidate.publication_ready]
    tournament: PairwiseVisualTournament | None = None
    if publishable:
        tournament = run_pairwise_visual_tournament(
            [
                VisualCandidateEvidence(
                    candidate_id=item.candidate_id,
                    frame_paths=item.frame_paths,
                    verifier_report=item.verification,
                    local_score=float(item.local_quality.score),
                )
                for item in publishable
            ],
            contract,
            provider_models=provider_models,
            request=vision_request,
            cache_dir=cache_dir,
            top_k=pairwise_top_k,
        )
        selected = next(
            (
                candidate
                for candidate in publishable
                if candidate.candidate_id == tournament.selected_candidate_id
            ),
            max(publishable, key=_candidate_rank_key),
        )
    else:
        selected = max(candidates, key=_candidate_rank_key)

    issues: list[str] = []
    warnings: list[str] = []
    if not publishable:
        issues.extend(selected.verification.issues)
        issues.extend(selected.hard_local_issues)
        issues.append("visual_director_has_no_publishable_candidate")
    if selected.verification.state == VisualQualityState.DEGRADED:
        warnings.append("visual_published_with_explicit_degraded_verification")
    warnings.extend(selected.verification.warnings)
    return VisualDirectionOutcome(
        version=VISUAL_DIRECTOR_RUNTIME_VERSION,
        passed=bool(publishable),
        selected=selected,
        candidates=candidates,
        tournament=tournament,
        repair_history=repair_history,
        issues=_unique(issues),
        warnings=_unique(warnings),
    )


def _evaluate_candidate(
    spec: dict[str, Any],
    asset: Any,
    selection_reason: str,
    *,
    round_index: int,
    contract: dict[str, Any],
    evaluate_local_quality: EvaluateLocalQuality,
    extract_candidate_frames: ExtractCandidateFrames,
    strict: bool,
    provider_models: Iterable[tuple[str, str]] | None,
    vision_request: VisionRequest | None,
    cache_dir: Path | None,
) -> DirectedCandidate:
    local_quality = evaluate_local_quality(spec, asset)
    candidate_id = _candidate_id(spec, asset, round_index)
    frames = [
        Path(path)
        for path in extract_candidate_frames(spec, asset, candidate_id)
        if Path(path).is_file()
    ]
    hard_local_issues = _hard_local_issues(local_quality.issues)
    local_degraded_gate = bool(
        local_quality.passed
        or (not hard_local_issues and float(local_quality.score) >= 0.56)
    )
    verification = run_visual_verifier(
        frames,
        contract,
        provider_models=provider_models,
        request=vision_request,
        strict=strict,
        local_gate_passed=local_degraded_gate,
        local_score=float(local_quality.score),
        cache_dir=cache_dir,
    )
    publication_ready = bool(
        verification.publishable
        and (
            local_quality.passed
            or (verification.verified and not hard_local_issues)
        )
    )
    return DirectedCandidate(
        candidate_id=candidate_id,
        round_index=round_index,
        spec=dict(spec),
        asset=asset,
        selection_reason=selection_reason,
        local_quality=local_quality,
        frame_paths=frames,
        verification=verification,
        publication_ready=publication_ready,
        hard_local_issues=hard_local_issues,
    )


def _monotonic_progress(
    before: DirectedCandidate,
    after: DirectedCandidate,
    assessment: RepairImprovementAssessment,
    *,
    minimum_delta: float,
) -> bool:
    if assessment.accepted:
        return True
    semantic_delta = after.verification.semantic.score - before.verification.semantic.score
    technical_delta = after.verification.technical.score - before.verification.technical.score
    score_delta = after.verification.score - before.verification.score
    state_delta = _quality_state_rank(after.verification.state) - _quality_state_rank(
        before.verification.state
    )
    return bool(
        semantic_delta >= -0.02
        and technical_delta >= -0.02
        and (state_delta > 0 or score_delta >= float(minimum_delta))
    )


def _hard_local_issues(issues: Iterable[str]) -> list[str]:
    hard_markers = (
        "blank",
        "clipped",
        "collision",
        "corrupt",
        "illegible",
        "missing_final",
        "no_frames",
        "semantic_program_failed",
        "semantic_qa_failed",
        "unsupported_claim",
    )
    return _unique(
        issue
        for issue in issues
        if any(marker in str(issue).casefold() for marker in hard_markers)
    )


def _candidate_id(spec: dict[str, Any], asset: Any, round_index: int) -> str:
    visual_id = str(spec.get("visual_id") or "visual")
    program_id = str(
        (spec.get("open_visual_program") or {}).get("program_id") or "program"
    )
    renderer = str(getattr(asset, "renderer", "renderer") or "renderer")
    return f"{visual_id}:{renderer}:{program_id}:r{round_index}"


def _candidate_rank_key(candidate: DirectedCandidate) -> tuple[int, float, float, int]:
    return (
        _quality_state_rank(candidate.verification.state),
        float(candidate.verification.score),
        float(candidate.local_quality.score),
        -candidate.round_index,
    )


def _quality_state_rank(state: VisualQualityState) -> int:
    return {
        VisualQualityState.VERIFIED: 4,
        VisualQualityState.DEGRADED: 3,
        VisualQualityState.UNVERIFIED: 2,
        VisualQualityState.REJECTED: 1,
    }.get(state, 0)


def _unique(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = str(value or "").strip()
        key = cleaned.casefold()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        output.append(cleaned)
    return output


__all__ = [
    "VISUAL_DIRECTOR_RUNTIME_VERSION",
    "DirectedCandidate",
    "VisualDirectionOutcome",
    "direct_rendered_visual",
]
