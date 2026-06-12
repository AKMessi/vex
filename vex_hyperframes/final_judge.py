from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import config


FINAL_JUDGE_VERSION = "hyperframes-independent-final-judge-v1"


@dataclass(frozen=True)
class FinalIndependentVerdict:
    version: str
    available: bool
    passed: bool
    score: float
    thesis: str
    issues: list[str] = field(default_factory=list)
    notes: str = ""
    model: str = ""
    error: str = ""
    local_gate_passed: bool = False
    vision_gate_passed: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["score"] = round(float(self.score), 4)
        return payload


def judge_final_candidate(
    frame_paths: list[Path],
    *,
    production_contract: dict[str, Any],
    scene_program: dict[str, Any],
    quality_report: dict[str, Any],
    critic_bundle: dict[str, Any],
    qa_mode: str,
    model_name: str | None = None,
) -> FinalIndependentVerdict:
    selected_model = str(
        model_name
        or getattr(config, "HYPERFRAMES_VISION_MODEL", "")
        or config.GEMINI_MODEL
        or ""
    ).strip()
    local_issues: list[str] = []
    if not bool(quality_report.get("passed")):
        local_issues.append("final_quality_gate_failed")
    if not bool(critic_bundle.get("passed")):
        local_issues.append("final_visual_critic_gate_failed")
    if int(critic_bundle.get("hard_failure_count") or 0) > 0:
        local_issues.append("final_candidate_has_hard_counterexamples")
    if not scene_program.get("program_signature"):
        local_issues.append("final_scene_program_unsigned")
    local_passed = not local_issues
    local_score = min(
        _bounded(quality_report.get("score")),
        _bounded(critic_bundle.get("score"), default=1.0),
    )
    if (
        not bool(getattr(config, "HYPERFRAMES_ENABLE_VISION_QA", False))
        or not config.GEMINI_API_KEY
    ):
        strict = str(qa_mode or "").strip().lower() == "vision"
        issues = [
            *local_issues,
            *(["independent_vision_judge_unavailable"] if strict else []),
        ]
        return FinalIndependentVerdict(
            version=FINAL_JUDGE_VERSION,
            available=False,
            passed=local_passed and not strict,
            score=local_score,
            thesis="",
            issues=issues,
            notes=(
                "Independent vision judge unavailable; artifact-only final gate used."
            ),
            model=selected_model,
            local_gate_passed=local_passed,
            vision_gate_passed=None,
        )
    max_frames = int(getattr(config, "HYPERFRAMES_MAX_CRITIC_FRAMES", 8))
    usable_frames = [
        Path(path)
        for path in frame_paths[:max_frames]
        if Path(path).is_file()
    ]
    if not usable_frames:
        strict = str(qa_mode or "").strip().lower() == "vision"
        return FinalIndependentVerdict(
            version=FINAL_JUDGE_VERSION,
            available=False,
            passed=local_passed and not strict,
            score=local_score,
            thesis="",
            issues=[
                *local_issues,
                *(["independent_judge_frames_unavailable"] if strict else []),
            ],
            notes="Independent vision judge had no final frames.",
            model=selected_model,
            local_gate_passed=local_passed,
            vision_gate_passed=None,
        )
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=config.GEMINI_API_KEY,
            http_options=config.google_genai_http_options(),
        )
        contents: list[Any] = [
            types.Part.from_text(
                text=_final_judge_prompt(
                    production_contract=production_contract,
                    scene_program=scene_program,
                    frame_count=len(usable_frames),
                )
            )
        ]
        for path in usable_frames:
            contents.append(
                types.Part.from_bytes(
                    data=path.read_bytes(),
                    mime_type="image/png",
                )
            )
        response = client.models.generate_content(
            model=selected_model,
            contents=contents,
            config=config.build_gemini_generation_config(
                (
                    "You are an independent final visual release judge. You have "
                    "no access to generation attempts, repair history, or prior "
                    "critic reports. Judge only the final frames and signed contract. "
                    "Return only JSON."
                ),
                model_name=selected_model,
            ),
        )
        payload = json.loads(
            _extract_json_object(getattr(response, "text", "") or "")
        )
        vision_passed = bool(payload.get("passed"))
        vision_score = _bounded(payload.get("score"))
        vision_issues = _strings(payload.get("issues"))
        passed = local_passed and vision_passed and not vision_issues
        return FinalIndependentVerdict(
            version=FINAL_JUDGE_VERSION,
            available=True,
            passed=passed,
            score=min(local_score, vision_score),
            thesis=" ".join(str(payload.get("thesis") or "").split())[:300],
            issues=[*local_issues, *vision_issues],
            notes=" ".join(str(payload.get("notes") or "").split())[:500],
            model=selected_model,
            local_gate_passed=local_passed,
            vision_gate_passed=vision_passed,
        )
    except Exception as exc:  # noqa: BLE001
        strict = str(qa_mode or "").strip().lower() == "vision"
        return FinalIndependentVerdict(
            version=FINAL_JUDGE_VERSION,
            available=False,
            passed=local_passed and not strict,
            score=local_score,
            thesis="",
            issues=[
                *local_issues,
                *(["independent_vision_judge_failed"] if strict else []),
            ],
            notes="Independent final vision request failed.",
            model=selected_model,
            error=" ".join(str(exc).split())[:500],
            local_gate_passed=local_passed,
            vision_gate_passed=None,
        )


def _final_judge_prompt(
    *,
    production_contract: dict[str, Any],
    scene_program: dict[str, Any],
    frame_count: int,
) -> str:
    contract = {
        "thesis": production_contract.get("thesis"),
        "takeaway": production_contract.get("takeaway"),
        "scene_type": production_contract.get("scene_type"),
        "required_labels": production_contract.get("required_labels"),
        "visual_claim_graph": production_contract.get("visual_claim_graph"),
        "screenshot_test": production_contract.get("screenshot_test"),
    }
    program = {
        "layout_family": scene_program.get("layout_family"),
        "elements": scene_program.get("elements"),
        "relations": scene_program.get("relations"),
    }
    return "\n".join(
        [
            f"You receive {frame_count} chronological frames from one final visual candidate.",
            "This is a release gate, not a brainstorming task.",
            "Fail if the thesis is unclear, any required object or relation is missing, any visible content is unsupported, hierarchy is weak, copy is unreadable, geometry collides, or the visual looks generic or amateur.",
            "Do not infer quality from the contract. Verify it in the pixels.",
            "Return exactly {passed:boolean,score:0..1,thesis:string,issues:string[],notes:string}.",
            "SIGNED CONTRACT:",
            json.dumps(contract, ensure_ascii=True, sort_keys=True),
            "SIGNED SCENE PROGRAM:",
            json.dumps(program, ensure_ascii=True, sort_keys=True),
        ]
    )


def _extract_json_object(raw_text: str) -> str:
    cleaned = str(raw_text or "").strip()
    fenced = re.search(
        r"```(?:json)?\s*(\{.*\})\s*```",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fenced:
        return fenced.group(1)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        return cleaned[start : end + 1]
    raise ValueError("Independent final judge did not return a JSON object.")


def _bounded(value: Any, *, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(number, 1.0))


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        cleaned = " ".join(str(item or "").split())[:300]
        if cleaned and cleaned not in result:
            result.append(cleaned)
    return result[:16]


__all__ = [
    "FINAL_JUDGE_VERSION",
    "FinalIndependentVerdict",
    "judge_final_candidate",
]
