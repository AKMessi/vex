from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from broll_intelligence import extract_json_object
from vex_visuals.open_visual_program import (
    OpenVisualTournament,
    build_open_visual_program_candidates,
    normalize_authored_open_visual_programs,
    open_visual_program_fingerprint,
    open_visual_program_prompt_block,
    select_open_visual_program,
)


GENERATIVE_AUTHORING_VERSION = "vex-generative-visual-authoring-v1"


@dataclass(frozen=True)
class GenerativeAuthoringResult:
    passed: bool
    selected_program: dict[str, Any] | None
    programs: list[dict[str, Any]]
    tournament: OpenVisualTournament
    authoring_mode: str
    model_attempts: int = 0
    model_program_count: int = 0
    deterministic_program_count: int = 0
    rejected_model_programs: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["selected_program"] = (
            dict(self.selected_program) if self.selected_program else None
        )
        payload["programs"] = [dict(item) for item in self.programs]
        payload["tournament"] = self.tournament.to_dict()
        return payload


def author_open_visual_programs(
    spec: dict[str, Any],
    *,
    ir: dict[str, Any],
    width: int,
    height: int,
    fps: float,
    reasoning_call: Callable[[str, str, str, str], str] | None = None,
    enable_model_authoring: bool = True,
    candidate_count: int = 3,
    max_model_attempts: int = 2,
) -> GenerativeAuthoringResult:
    normalized = dict(spec or {})
    evidence = dict(ir or {})
    visual_id = str(normalized.get("visual_id") or normalized.get("id") or "visual")
    duration_sec = _duration(normalized)
    theme = dict(normalized.get("theme") or {})
    history = _history(normalized)
    count = max(1, min(int(candidate_count), 4))
    deterministic = build_open_visual_program_candidates(
        evidence,
        visual_id=visual_id,
        width=width,
        height=height,
        duration_sec=duration_sec,
        fps=fps,
        theme=theme,
        history=history,
        candidate_count=count,
    )

    provider_name = str(normalized.get("generation_provider") or "").strip().lower()
    model_name = str(normalized.get("generation_model") or "").strip()
    authored: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    warnings: list[str] = []
    attempts = 0
    can_call_model = bool(
        enable_model_authoring
        and reasoning_call is not None
        and provider_name in {"claude", "gemini"}
        and model_name
    )
    if can_call_model:
        prompt = _authoring_prompt(
            normalized,
            evidence,
            candidate_count=min(count, 2),
        )
        previous_errors: list[dict[str, Any]] = []
        for attempt in range(max(1, min(int(max_model_attempts), 2))):
            attempts += 1
            attempt_prompt = prompt
            if previous_errors:
                attempt_prompt += (
                    "\n\nYour previous output failed validation. Repair it without weakening "
                    "grounding or removing required objects/relations. Validation errors:\n"
                    + json.dumps(previous_errors[:6], ensure_ascii=True)
                )
            try:
                raw = reasoning_call(
                    provider_name,
                    model_name,
                    _system_prompt(),
                    attempt_prompt,
                )
                parsed = json.loads(extract_json_object(raw))
                accepted, attempt_rejected = normalize_authored_open_visual_programs(
                    parsed,
                    ir=evidence,
                    visual_id=visual_id,
                    width=width,
                    height=height,
                    duration_sec=duration_sec,
                    fps=fps,
                    theme=theme,
                    history=history,
                )
                authored.extend(accepted)
                rejected.extend(attempt_rejected)
                previous_errors = attempt_rejected
                if accepted:
                    break
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                previous_errors = [{"errors": [f"invalid_model_response:{exc}"]}]
                rejected.extend(previous_errors)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"model_authoring_unavailable:{type(exc).__name__}")
                break
    elif enable_model_authoring:
        warnings.append("model_authoring_not_configured")

    programs = _dedupe_programs([*authored, *deterministic])
    tournament = select_open_visual_program(
        programs,
        ir=evidence,
        history=history,
    )
    selected = next(
        (
            item
            for item in programs
            if str(item.get("program_id") or "") == tournament.selected_program_id
        ),
        None,
    )
    mode = (
        "llm_authored"
        if selected is not None and selected in authored
        else "deterministic_open_program"
        if selected is not None
        else "failed"
    )
    return GenerativeAuthoringResult(
        passed=selected is not None,
        selected_program=selected,
        programs=programs,
        tournament=tournament,
        authoring_mode=mode,
        model_attempts=attempts,
        model_program_count=len(authored),
        deterministic_program_count=len(deterministic),
        rejected_model_programs=rejected,
        warnings=list(dict.fromkeys(warnings)),
    )


def compile_open_visual_program_for_spec(
    spec: dict[str, Any],
    *,
    ir: dict[str, Any],
    width: int,
    height: int,
    fps: float,
    reasoning_call: Callable[[str, str, str, str], str] | None = None,
    enable_model_authoring: bool = True,
    candidate_count: int = 3,
    max_model_attempts: int = 2,
) -> tuple[dict[str, Any], GenerativeAuthoringResult]:
    result = author_open_visual_programs(
        spec,
        ir=ir,
        width=width,
        height=height,
        fps=fps,
        reasoning_call=reasoning_call,
        enable_model_authoring=enable_model_authoring,
        candidate_count=candidate_count,
        max_model_attempts=max_model_attempts,
    )
    normalized = dict(spec or {})
    if result.passed and result.selected_program is not None:
        normalized["open_visual_program"] = dict(result.selected_program)
        normalized["open_visual_program_candidates"] = [
            dict(item) for item in result.programs
        ]
        normalized["open_visual_tournament"] = result.tournament.to_dict()
        normalized["open_visual_authoring"] = {
            "version": GENERATIVE_AUTHORING_VERSION,
            "mode": result.authoring_mode,
            "model_attempts": result.model_attempts,
            "model_program_count": result.model_program_count,
            "deterministic_program_count": result.deterministic_program_count,
            "rejected_model_programs": list(result.rejected_model_programs),
            "warnings": list(result.warnings),
        }
    return normalized, result


def _system_prompt() -> str:
    return (
        "You are Vex's senior motion-design director and scene-graph engineer. "
        "Author executable visual explanations, not UI mockups and not template names. "
        "Your output is untrusted and will be rejected unless every factual element "
        "binds to supplied evidence. Prefer transformations that make the concept "
        "understandable with the audio muted. Return strict JSON only."
    )


def _authoring_prompt(
    spec: dict[str, Any],
    ir: dict[str, Any],
    *,
    candidate_count: int,
) -> str:
    creative_direction = dict(
        spec.get("creative_direction_program")
        or spec.get("creative_direction")
        or {}
    )
    directed_brief = dict(spec.get("directed_visual_brief") or {})
    context = {
        "renderer": str(spec.get("renderer_hint") or "auto"),
        "orientation": str(spec.get("orientation") or "landscape"),
        "style_pack": str(spec.get("style_pack") or "auto"),
        "creative_direction": creative_direction,
        "directed_visual_brief": directed_brief,
        "source_frame_analysis": dict(
            (spec.get("auto_visuals_director") or {}).get("source_frame_analysis")
            or {}
        ),
        "recent_visual_fingerprints": _history(spec)[-5:],
    }
    return (
        open_visual_program_prompt_block(ir, candidate_count=candidate_count)
        + "\n\nProduction context:\n"
        + json.dumps(context, ensure_ascii=True, sort_keys=True)
    )


def _history(spec: dict[str, Any]) -> list[dict[str, Any]]:
    values = (
        spec.get("open_visual_program_history")
        or spec.get("creative_direction_history")
        or spec.get("visual_world_history")
        or []
    )
    return [dict(item) for item in values if isinstance(item, dict)][-8:]


def _duration(spec: dict[str, Any]) -> float:
    explicit = _number(spec.get("duration"), 0.0)
    if explicit > 0:
        return max(1.2, min(explicit, 16.0))
    start = _number(spec.get("start"), 0.0)
    end = _number(spec.get("end"), 0.0)
    return max(1.2, min(end - start if end > start else 4.0, 16.0))


def _dedupe_programs(programs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    signatures: set[str] = set()
    program_ids: set[str] = set()
    for item in programs:
        fingerprint = open_visual_program_fingerprint(item)
        signature = str(fingerprint.get("signature") or "")
        program_id = str(item.get("program_id") or "")
        if not signature or signature in signatures or not program_id or program_id in program_ids:
            continue
        signatures.add(signature)
        program_ids.add(program_id)
        result.append(dict(item))
    return result[:4]


def _number(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


__all__ = [
    "GENERATIVE_AUTHORING_VERSION",
    "GenerativeAuthoringResult",
    "author_open_visual_programs",
    "compile_open_visual_program_for_spec",
]
