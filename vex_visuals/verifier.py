from __future__ import annotations

import base64
from dataclasses import asdict, dataclass, field
from enum import StrEnum
import hashlib
import json
from pathlib import Path
import threading
import time
from typing import Any, Callable, Iterable

import config
from vex_visuals.communication_contract import (
    CommunicationContract,
    CommunicationEvaluation,
    evaluate_viewer_answers,
)


VISUAL_VERIFIER_VERSION = "vex-multimodal-visual-verifier-v1"
PAIRWISE_TOURNAMENT_VERSION = "vex-pairwise-visual-tournament-v1"

VisionRequest = Callable[[str, str, str, list[Path]], dict[str, Any]]

_HARD_TECHNICAL_DEFECTS = {
    "blank frame",
    "clipped required text",
    "element collision",
    "illegible required text",
    "missing final state",
    "severe flicker",
    "unsupported claim",
}


class VisualQualityState(StrEnum):
    VERIFIED = "verified"
    DEGRADED = "degraded"
    UNVERIFIED = "unverified"
    REJECTED = "rejected"


@dataclass(frozen=True)
class VerifierDimension:
    name: str
    score: float
    passed: bool
    issues: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["score"] = round(float(self.score), 4)
        return payload


@dataclass(frozen=True)
class VisualVerifierReport:
    version: str
    available: bool
    publishable: bool
    state: VisualQualityState
    score: float
    semantic: VerifierDimension
    design: VerifierDimension
    temporal: VerifierDimension
    technical: VerifierDimension
    communication: CommunicationEvaluation | None
    provider: str = ""
    model: str = ""
    attempts: int = 0
    cache_hit: bool = False
    decoded: dict[str, Any] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    repair_directives: list[dict[str, str]] = field(default_factory=list)

    @property
    def verified(self) -> bool:
        return self.state == VisualQualityState.VERIFIED

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["state"] = self.state.value
        payload["score"] = round(float(self.score), 4)
        payload["semantic"] = self.semantic.to_dict()
        payload["design"] = self.design.to_dict()
        payload["temporal"] = self.temporal.to_dict()
        payload["technical"] = self.technical.to_dict()
        payload["communication"] = self.communication.to_dict() if self.communication else None
        return payload


@dataclass(frozen=True)
class PairwisePreference:
    candidate_a: str
    candidate_b: str
    available: bool
    winner: str
    confidence: float
    order_consistent: bool
    reasons: list[str] = field(default_factory=list)
    provider: str = ""
    model: str = ""
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["confidence"] = round(float(self.confidence), 4)
        return payload


@dataclass(frozen=True)
class VisualCandidateEvidence:
    candidate_id: str
    frame_paths: list[Path]
    verifier_report: VisualVerifierReport
    local_score: float = 0.0


@dataclass(frozen=True)
class PairwiseVisualTournament:
    version: str
    selected_candidate_id: str
    selection_mode: str
    comparisons: list[PairwisePreference]
    ranked_candidate_ids: list[str]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "comparisons": [item.to_dict() for item in self.comparisons],
        }


@dataclass
class _CircuitState:
    failures: int = 0
    open_until: float = 0.0


_CIRCUITS: dict[str, _CircuitState] = {}
_CIRCUIT_LOCK = threading.Lock()


def run_visual_verifier(
    frame_paths: Iterable[Path],
    contract: CommunicationContract | dict[str, Any],
    *,
    provider_models: Iterable[tuple[str, str]] | None = None,
    request: VisionRequest | None = None,
    strict: bool = True,
    local_gate_passed: bool = False,
    local_score: float = 0.0,
    cache_dir: Path | None = None,
) -> VisualVerifierReport:
    frames = [Path(path) for path in frame_paths if Path(path).is_file()]
    contract_payload = contract.to_dict() if isinstance(contract, CommunicationContract) else dict(contract or {})
    if not frames:
        return _unavailable_report(
            "visual_verifier_has_no_frames",
            strict=strict,
            local_gate_passed=local_gate_passed,
            local_score=local_score,
        )
    endpoints = list(provider_models or _configured_provider_models())
    if request is not None and not endpoints:
        endpoints = [("test", "test-vision-model")]
    if not endpoints:
        return _unavailable_report(
            "visual_verifier_has_no_configured_provider",
            strict=strict,
            local_gate_passed=local_gate_passed,
            local_score=local_score,
        )
    vision_request = request or _default_vision_request
    prompt = blind_visual_verifier_prompt(contract_payload, frame_count=len(frames))
    attempts = 0
    errors: list[str] = []
    for provider, model in endpoints:
        circuit_key = f"{provider}:{model}"
        if _circuit_is_open(circuit_key):
            errors.append(f"{circuit_key}:circuit_open")
            continue
        cache_path = _cache_path(cache_dir, frames, contract_payload, provider, model, "blind")
        if cache_path is not None and cache_path.is_file():
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                report = evaluate_verifier_payload(
                    payload,
                    contract_payload,
                    provider=provider,
                    model=model,
                    attempts=attempts,
                    cache_hit=True,
                )
                _circuit_success(circuit_key)
                return report
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                cache_path.unlink(missing_ok=True)
        retries = max(1, min(int(getattr(config, "VISUAL_DIRECTOR_VERIFIER_RETRIES", 2)), 3))
        for retry in range(retries):
            attempts += 1
            try:
                payload = vision_request(provider, model, prompt, frames)
                report = evaluate_verifier_payload(
                    payload,
                    contract_payload,
                    provider=provider,
                    model=model,
                    attempts=attempts,
                )
                _circuit_success(circuit_key)
                if cache_path is not None:
                    _write_cache(cache_path, payload)
                return report
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{circuit_key}:{type(exc).__name__}:{_clean(exc, 180)}")
                _circuit_failure(circuit_key)
                if retry + 1 < retries:
                    delay = min(4.0, float(getattr(config, "LLM_RETRY_BASE_DELAY_SEC", 1.0)) * (2**retry))
                    time.sleep(max(0.0, delay))
    report = _unavailable_report(
        "visual_verifier_all_providers_failed",
        strict=strict,
        local_gate_passed=local_gate_passed,
        local_score=local_score,
    )
    return VisualVerifierReport(
        **{
            **report.__dict__,
            "attempts": attempts,
            "warnings": [*report.warnings, *errors[:8]],
        }
    )


def evaluate_verifier_payload(
    payload: dict[str, Any],
    contract: CommunicationContract | dict[str, Any],
    *,
    provider: str = "",
    model: str = "",
    attempts: int = 1,
    cache_hit: bool = False,
) -> VisualVerifierReport:
    value = dict(payload or {})
    contract_payload = contract.to_dict() if isinstance(contract, CommunicationContract) else dict(contract or {})
    answers = {
        str(key): _clean(answer, 500)
        for key, answer in dict(value.get("answers") or {}).items()
    }
    sequence = _strings(value.get("sequence"), limit=20)
    unsupported = _strings(value.get("unsupported_claims"), limit=16)
    communication = evaluate_viewer_answers(
        contract_payload,
        answers,
        decoded_thesis=_clean(value.get("thesis"), 500),
        decoded_sequence=sequence,
        unsupported_claims=unsupported,
    )
    semantic_issues = list(communication.issues)
    semantic = VerifierDimension(
        name="semantic_communication",
        score=communication.score,
        passed=communication.passed,
        issues=semantic_issues,
        evidence={
            "proposition_coverage": communication.proposition_coverage,
            "temporal_sequence_score": communication.temporal_score,
            "missing_proposition_ids": communication.missing_proposition_ids,
        },
    )
    design_scores = _dimension_scores(
        value.get("design"),
        keys=("hierarchy", "composition", "typography", "density", "polish", "originality"),
    )
    design_score = _mean(design_scores.values(), default=0.0)
    design_issues = _strings((value.get("design") or {}).get("issues"), limit=12)
    if design_score < 0.68:
        design_issues.append("visual_design_below_production_floor")
    design = VerifierDimension(
        name="design_quality",
        score=design_score,
        passed=design_score >= 0.68 and not design_issues,
        issues=_unique(design_issues, limit=16),
        evidence=design_scores,
    )
    temporal_scores = _dimension_scores(
        value.get("temporal"),
        keys=("causal_readability", "meaningful_motion", "smoothness", "sequence_legibility", "settling"),
    )
    temporal_score = _mean(temporal_scores.values(), default=0.0)
    temporal_issues = _strings((value.get("temporal") or {}).get("issues"), limit=12)
    if temporal_score < 0.64:
        temporal_issues.append("visual_motion_below_production_floor")
    temporal = VerifierDimension(
        name="temporal_quality",
        score=temporal_score,
        passed=temporal_score >= 0.64 and not temporal_issues,
        issues=_unique(temporal_issues, limit=16),
        evidence=temporal_scores,
    )
    defects = _strings(value.get("technical_defects"), limit=20)
    hard_defects = [item for item in defects if _is_hard_defect(item)]
    technical_score = max(0.0, 1.0 - len(defects) * 0.12 - len(hard_defects) * 0.28)
    technical = VerifierDimension(
        name="technical_integrity",
        score=technical_score,
        passed=not hard_defects and technical_score >= 0.76,
        issues=defects,
        evidence={"hard_defects": hard_defects},
    )
    score = semantic.score * 0.4 + design.score * 0.25 + temporal.score * 0.25 + technical.score * 0.1
    hard_semantic = bool(unsupported) or communication.proposition_coverage < 0.45
    all_passed = semantic.passed and design.passed and temporal.passed and technical.passed
    if all_passed and score >= 0.72:
        state = VisualQualityState.VERIFIED
    elif hard_semantic or hard_defects or score < 0.54:
        state = VisualQualityState.REJECTED
    else:
        state = VisualQualityState.DEGRADED
    issues = _unique(
        [*semantic.issues, *design.issues, *temporal.issues, *technical.issues],
        limit=30,
    )
    return VisualVerifierReport(
        version=VISUAL_VERIFIER_VERSION,
        available=True,
        publishable=state == VisualQualityState.VERIFIED,
        state=state,
        score=score,
        semantic=semantic,
        design=design,
        temporal=temporal,
        technical=technical,
        communication=communication,
        provider=provider,
        model=model,
        attempts=attempts,
        cache_hit=cache_hit,
        decoded=value,
        issues=issues,
        repair_directives=_repair_directives(semantic, design, temporal, technical),
    )


def run_pairwise_visual_tournament(
    candidates: Iterable[VisualCandidateEvidence],
    contract: CommunicationContract | dict[str, Any],
    *,
    provider_models: Iterable[tuple[str, str]] | None = None,
    request: VisionRequest | None = None,
    cache_dir: Path | None = None,
    top_k: int = 3,
) -> PairwiseVisualTournament:
    values = list(candidates)
    ranked = sorted(
        values,
        key=lambda item: (
            _state_rank(item.verifier_report.state),
            item.verifier_report.score,
            item.local_score,
            item.candidate_id,
        ),
        reverse=True,
    )
    if not ranked:
        return PairwiseVisualTournament(
            version=PAIRWISE_TOURNAMENT_VERSION,
            selected_candidate_id="",
            selection_mode="no_candidates",
            comparisons=[],
            ranked_candidate_ids=[],
            warnings=["pairwise_tournament_has_no_candidates"],
        )
    contenders = ranked[: max(1, min(int(top_k), 4))]
    endpoints = list(provider_models or _configured_provider_models())
    if request is not None and not endpoints:
        endpoints = [("test", "test-vision-model")]
    if len(contenders) == 1 or not endpoints:
        return PairwiseVisualTournament(
            version=PAIRWISE_TOURNAMENT_VERSION,
            selected_candidate_id=contenders[0].candidate_id,
            selection_mode="verified_score_fallback",
            comparisons=[],
            ranked_candidate_ids=[item.candidate_id for item in ranked],
            warnings=[] if endpoints else ["pairwise_judge_unavailable"],
        )
    winner = contenders[0]
    comparisons: list[PairwisePreference] = []
    for challenger in contenders[1:]:
        preference = compare_visual_candidates(
            winner,
            challenger,
            contract,
            provider_models=endpoints,
            request=request,
            cache_dir=cache_dir,
        )
        comparisons.append(preference)
        if preference.available and preference.order_consistent:
            if preference.winner == challenger.candidate_id:
                winner = challenger
        elif _candidate_rank_key(challenger) > _candidate_rank_key(winner):
            winner = challenger
    available = [item for item in comparisons if item.available and item.order_consistent]
    return PairwiseVisualTournament(
        version=PAIRWISE_TOURNAMENT_VERSION,
        selected_candidate_id=winner.candidate_id,
        selection_mode="bidirectional_pairwise" if available else "verified_score_fallback",
        comparisons=comparisons,
        ranked_candidate_ids=[item.candidate_id for item in ranked],
        warnings=[] if available else ["pairwise_judge_did_not_produce_consistent_preference"],
    )


def compare_visual_candidates(
    first: VisualCandidateEvidence,
    second: VisualCandidateEvidence,
    contract: CommunicationContract | dict[str, Any],
    *,
    provider_models: Iterable[tuple[str, str]],
    request: VisionRequest | None = None,
    cache_dir: Path | None = None,
) -> PairwisePreference:
    del cache_dir
    contract_payload = contract.to_dict() if isinstance(contract, CommunicationContract) else dict(contract or {})
    endpoints = list(provider_models)
    vision_request = request or _default_vision_request
    errors: list[str] = []
    for provider, model in endpoints:
        circuit_key = f"{provider}:{model}"
        if _circuit_is_open(circuit_key):
            continue
        try:
            first_payload = vision_request(
                provider,
                model,
                pairwise_visual_prompt(
                    contract_payload,
                    first_frame_count=len(first.frame_paths),
                    second_frame_count=len(second.frame_paths),
                ),
                [*first.frame_paths, *second.frame_paths],
            )
            second_payload = vision_request(
                provider,
                model,
                pairwise_visual_prompt(
                    contract_payload,
                    first_frame_count=len(second.frame_paths),
                    second_frame_count=len(first.frame_paths),
                ),
                [*second.frame_paths, *first.frame_paths],
            )
            first_winner = _pairwise_winner(first_payload, first.candidate_id, second.candidate_id)
            second_winner = _pairwise_winner(second_payload, second.candidate_id, first.candidate_id)
            consistent = first_winner == second_winner and first_winner in {first.candidate_id, second.candidate_id, "tie"}
            confidence = min(
                _bounded(first_payload.get("confidence"), 0.0),
                _bounded(second_payload.get("confidence"), 0.0),
            )
            _circuit_success(circuit_key)
            return PairwisePreference(
                candidate_a=first.candidate_id,
                candidate_b=second.candidate_id,
                available=True,
                winner=first_winner if consistent else "tie",
                confidence=confidence,
                order_consistent=consistent,
                reasons=_unique(
                    [
                        *_strings(first_payload.get("reasons"), limit=8),
                        *_strings(second_payload.get("reasons"), limit=8),
                    ],
                    limit=12,
                ),
                provider=provider,
                model=model,
                errors=[] if consistent else ["pairwise_judge_order_inconsistent"],
            )
        except Exception as exc:  # noqa: BLE001
            _circuit_failure(circuit_key)
            errors.append(f"{circuit_key}:{type(exc).__name__}:{_clean(exc, 180)}")
    return PairwisePreference(
        candidate_a=first.candidate_id,
        candidate_b=second.candidate_id,
        available=False,
        winner="tie",
        confidence=0.0,
        order_consistent=False,
        errors=errors or ["pairwise_judge_unavailable"],
    )


def blind_visual_verifier_prompt(contract: dict[str, Any], *, frame_count: int) -> str:
    questions = [
        {
            "question_id": item.get("question_id"),
            "question": item.get("question"),
        }
        for item in contract.get("questions") or []
        if isinstance(item, dict)
    ]
    return "\n".join(
        [
            f"You receive {frame_count} chronological frames from one silent motion graphic.",
            "Judge only visible pixels. You are not given the intended answers, transcript, program, renderer, or template.",
            "Do not reward complexity or polish when meaning is unclear. Do not infer claims from the questions.",
            "Answer each viewer question using only what the frames visibly communicate.",
            "Return strict JSON with keys: thesis, answers, sequence, unsupported_claims, design, temporal, technical_defects.",
            "answers is an object keyed by question_id with short answers.",
            "sequence is an array of visibly communicated states in chronological order.",
            "design contains hierarchy, composition, typography, density, polish, originality scores in [0,1], plus issues array.",
            "temporal contains causal_readability, meaningful_motion, smoothness, sequence_legibility, settling scores in [0,1], plus issues array.",
            "technical_defects is an array. Use concrete defects such as blank frame, clipped required text, element collision, illegible required text, missing final state, or severe flicker.",
            "Viewer questions:",
            json.dumps(questions, ensure_ascii=True, sort_keys=True),
        ]
    )


def pairwise_visual_prompt(
    contract: dict[str, Any],
    *,
    first_frame_count: int,
    second_frame_count: int,
) -> str:
    intent = {
        "thesis": contract.get("thesis"),
        "takeaway": contract.get("takeaway"),
        "propositions": [
            item.get("proposition")
            for item in contract.get("propositions") or []
            if isinstance(item, dict) and bool(item.get("required", True))
        ],
        "forbidden_claims": contract.get("forbidden_claims") or [],
    }
    return "\n".join(
        [
            f"The first {first_frame_count} frames are Candidate A in time order.",
            f"The next {second_frame_count} frames are Candidate B in time order.",
            "Choose which candidate communicates the supplied intent more accurately and beautifully as a silent production motion graphic.",
            "Prioritize: no unsupported claims; immediate semantic comprehension; proof-bearing motion; hierarchy and typography; temporal coherence; originality without clutter.",
            "Do not prefer a candidate because it is first, brighter, denser, or contains more text.",
            "Return strict JSON: winner is A, B, or tie; confidence in [0,1]; reasons array with specific comparative evidence.",
            "Communication intent:",
            json.dumps(intent, ensure_ascii=True, sort_keys=True),
        ]
    )


def reset_verifier_circuits() -> None:
    with _CIRCUIT_LOCK:
        _CIRCUITS.clear()


def _configured_provider_models() -> list[tuple[str, str]]:
    configured: list[tuple[str, str]] = []
    gemini_model = str(
        getattr(config, "VISUAL_DIRECTOR_VISION_MODEL", "")
        or getattr(config, "HYPERFRAMES_VISION_MODEL", "")
        or config.GEMINI_MODEL
        or ""
    ).strip()
    claude_model = str(
        getattr(config, "VISUAL_DIRECTOR_CLAUDE_VISION_MODEL", "")
        or config.CLAUDE_MODEL
        or ""
    ).strip()
    available = {
        "gemini": (gemini_model, bool(config.GEMINI_API_KEY)),
        "claude": (claude_model, bool(config.ANTHROPIC_API_KEY)),
    }
    preferred = str(config.PROVIDER or "").strip().lower()
    order = [preferred, "gemini", "claude"]
    for provider in order:
        if provider not in available:
            continue
        model, enabled = available[provider]
        endpoint = (provider, model)
        if enabled and model and endpoint not in configured:
            configured.append(endpoint)
    return configured


def _default_vision_request(
    provider: str,
    model: str,
    prompt: str,
    frame_paths: list[Path],
) -> dict[str, Any]:
    if provider == "gemini":
        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=config.GEMINI_API_KEY,
            http_options=config.google_genai_http_options(),
        )
        contents: list[Any] = [types.Part.from_text(text=prompt)]
        contents.extend(
            types.Part.from_bytes(data=path.read_bytes(), mime_type="image/png")
            for path in frame_paths
        )
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config.build_gemini_generation_config(
                "You are an independent visual communication evaluator. Return strict JSON only.",
                model_name=model,
            ),
        )
        return json.loads(_extract_json_object(getattr(response, "text", "") or ""))
    if provider == "claude":
        from anthropic import Anthropic

        client = Anthropic(api_key=config.ANTHROPIC_API_KEY, timeout=config.ANTHROPIC_TIMEOUT_SEC)
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for path in frame_paths:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.b64encode(path.read_bytes()).decode("ascii"),
                    },
                }
            )
        response = client.messages.create(
            model=model,
            max_tokens=3000,
            system="You are an independent visual communication evaluator. Return strict JSON only.",
            messages=[{"role": "user", "content": content}],
        )
        text = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
        return json.loads(_extract_json_object(text))
    raise ValueError(f"Unsupported visual verifier provider: {provider}")


def _unavailable_report(
    issue: str,
    *,
    strict: bool,
    local_gate_passed: bool,
    local_score: float,
) -> VisualVerifierReport:
    degraded = bool(local_gate_passed and not strict)
    state = VisualQualityState.DEGRADED if degraded else VisualQualityState.UNVERIFIED
    dimension = VerifierDimension(
        name="unavailable",
        score=0.0,
        passed=False,
        issues=[issue],
    )
    warnings = ["local_gate_used_for_explicit_degraded_publish"] if degraded else []
    return VisualVerifierReport(
        version=VISUAL_VERIFIER_VERSION,
        available=False,
        publishable=degraded,
        state=state,
        score=_bounded(local_score, 0.0) * 0.72 if degraded else 0.0,
        semantic=dimension,
        design=dimension,
        temporal=dimension,
        technical=dimension,
        communication=None,
        issues=[issue],
        warnings=warnings,
        repair_directives=[
            {
                "level": "verification",
                "operation": "retry_with_independent_vision_provider",
                "reason": issue,
            }
        ],
    )


def _repair_directives(
    semantic: VerifierDimension,
    design: VerifierDimension,
    temporal: VerifierDimension,
    technical: VerifierDimension,
) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    if not semantic.passed:
        result.append(
            {
                "level": "semantic_encoding",
                "operation": "regenerate_concept_or_encoding",
                "reason": "; ".join(semantic.issues[:3]),
            }
        )
    if not design.passed:
        result.append(
            {
                "level": "composition",
                "operation": "rebuild_hierarchy_and_visual_target",
                "reason": "; ".join(design.issues[:3]),
            }
        )
    if not temporal.passed:
        result.append(
            {
                "level": "motion_causality",
                "operation": "rebuild_proof_bearing_reveal",
                "reason": "; ".join(temporal.issues[:3]),
            }
        )
    if not technical.passed:
        result.append(
            {
                "level": "execution",
                "operation": "patch_render_defects",
                "reason": "; ".join(technical.issues[:3]),
            }
        )
    return result


def _dimension_scores(value: Any, *, keys: tuple[str, ...]) -> dict[str, float]:
    payload = dict(value or {})
    return {key: _bounded(payload.get(key), 0.0) for key in keys}


def _mean(values: Iterable[float], *, default: float) -> float:
    items = list(values)
    return sum(items) / len(items) if items else default


def _pairwise_winner(payload: dict[str, Any], candidate_a: str, candidate_b: str) -> str:
    winner = str(payload.get("winner") or "tie").strip().upper()
    if winner == "A":
        return candidate_a
    if winner == "B":
        return candidate_b
    return "tie"


def _candidate_rank_key(candidate: VisualCandidateEvidence) -> tuple[int, float, float, str]:
    return (
        _state_rank(candidate.verifier_report.state),
        candidate.verifier_report.score,
        candidate.local_score,
        candidate.candidate_id,
    )


def _state_rank(state: VisualQualityState) -> int:
    return {
        VisualQualityState.VERIFIED: 4,
        VisualQualityState.DEGRADED: 3,
        VisualQualityState.UNVERIFIED: 2,
        VisualQualityState.REJECTED: 1,
    }.get(state, 0)


def _cache_path(
    cache_dir: Path | None,
    frames: list[Path],
    contract: dict[str, Any],
    provider: str,
    model: str,
    operation: str,
) -> Path | None:
    if cache_dir is None:
        return None
    digest = hashlib.sha256()
    digest.update(VISUAL_VERIFIER_VERSION.encode("utf-8"))
    digest.update(str(contract.get("signature") or "").encode("utf-8"))
    digest.update(f"{provider}:{model}:{operation}".encode("utf-8"))
    for path in frames:
        digest.update(path.read_bytes())
    return Path(cache_dir) / f"{digest.hexdigest()}.json"


def _write_cache(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _circuit_is_open(key: str) -> bool:
    with _CIRCUIT_LOCK:
        state = _CIRCUITS.get(key)
        return bool(state and state.open_until > time.monotonic())


def _circuit_failure(key: str) -> None:
    threshold = max(1, int(getattr(config, "VISUAL_DIRECTOR_CIRCUIT_FAILURES", 2)))
    cooldown = max(1.0, float(getattr(config, "VISUAL_DIRECTOR_CIRCUIT_COOLDOWN_SEC", 45.0)))
    with _CIRCUIT_LOCK:
        state = _CIRCUITS.setdefault(key, _CircuitState())
        state.failures += 1
        if state.failures >= threshold:
            state.open_until = time.monotonic() + cooldown


def _circuit_success(key: str) -> None:
    with _CIRCUIT_LOCK:
        _CIRCUITS[key] = _CircuitState()


def _is_hard_defect(value: str) -> bool:
    normalized = " ".join(str(value or "").lower().replace("_", " ").split())
    return any(item in normalized for item in _HARD_TECHNICAL_DEFECTS)


def _extract_json_object(raw: str) -> str:
    text = str(raw or "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Visual verifier did not return a JSON object.")
    return text[start : end + 1]


def _strings(value: Any, *, limit: int) -> list[str]:
    values = [value] if isinstance(value, str) else list(value or [])
    return _unique([_clean(item, 500) for item in values], limit=limit)


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


def _clean(value: Any, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit].strip()


def _bounded(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(number, 1.0))


__all__ = [
    "PAIRWISE_TOURNAMENT_VERSION",
    "VISUAL_VERIFIER_VERSION",
    "PairwisePreference",
    "PairwiseVisualTournament",
    "VerifierDimension",
    "VisualCandidateEvidence",
    "VisualQualityState",
    "VisualVerifierReport",
    "blind_visual_verifier_prompt",
    "compare_visual_candidates",
    "evaluate_verifier_payload",
    "pairwise_visual_prompt",
    "reset_verifier_circuits",
    "run_pairwise_visual_tournament",
    "run_visual_verifier",
]
