from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import config


@dataclass(frozen=True)
class HyperframesVisionReport:
    available: bool
    passed: bool | None
    score: float | None
    notes: str
    missing_labels: list[str] = field(default_factory=list)
    semantic_issues: list[str] = field(default_factory=list)
    repair_directives: list[str] = field(default_factory=list)
    model: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def critique_hyperframes_frames(
    frame_paths: list[Path],
    *,
    production_contract: dict[str, Any],
    storyboard: list[dict[str, Any]],
    model_name: str | None = None,
) -> HyperframesVisionReport:
    enabled = bool(getattr(config, "HYPERFRAMES_ENABLE_VISION_QA", False))
    selected_model = str(
        model_name
        or getattr(config, "HYPERFRAMES_VISION_MODEL", "")
        or config.GEMINI_MODEL
        or ""
    ).strip()
    if not enabled:
        return HyperframesVisionReport(
            available=False,
            passed=None,
            score=None,
            notes="Vision QA is disabled.",
            model=selected_model,
        )
    if not config.GEMINI_API_KEY:
        return HyperframesVisionReport(
            available=False,
            passed=None,
            score=None,
            notes="Vision QA was skipped because GEMINI_API_KEY is not configured.",
            model=selected_model,
        )
    usable_frames = [path for path in frame_paths[:4] if Path(path).is_file()]
    if not usable_frames:
        return HyperframesVisionReport(
            available=False,
            passed=None,
            score=None,
            notes="Vision QA was skipped because no sampled frames were available.",
            model=selected_model,
        )
    try:
        from google import genai
        from google.genai import types

        prompt = _vision_prompt(production_contract, storyboard, len(usable_frames))
        contents: list[Any] = [types.Part.from_text(text=prompt)]
        for path in usable_frames:
            contents.append(
                types.Part.from_bytes(
                    data=Path(path).read_bytes(),
                    mime_type="image/png",
                )
            )
        client = genai.Client(
            api_key=config.GEMINI_API_KEY,
            http_options=config.google_genai_http_options(),
        )
        response = client.models.generate_content(
            model=selected_model,
            contents=contents,
            config=config.build_gemini_generation_config(
                (
                    "You are a strict motion-design QA system. Judge explanatory correctness "
                    "before aesthetics and return only one JSON object."
                ),
                model_name=selected_model,
            ),
        )
        payload = json.loads(_extract_json_object(getattr(response, "text", "") or ""))
        score = _bounded(payload.get("score"), 0.0)
        passed = bool(payload.get("passed")) and score >= 0.72
        return HyperframesVisionReport(
            available=True,
            passed=passed,
            score=score,
            notes=str(payload.get("notes") or "").strip(),
            missing_labels=_string_list(payload.get("missing_labels"), limit=10),
            semantic_issues=_string_list(payload.get("semantic_issues"), limit=10),
            repair_directives=_string_list(payload.get("repair_directives"), limit=8),
            model=selected_model,
        )
    except Exception as exc:  # noqa: BLE001
        return HyperframesVisionReport(
            available=False,
            passed=None,
            score=None,
            notes="Vision QA request failed.",
            model=selected_model,
            error=" ".join(str(exc).split())[:500],
        )


def _vision_prompt(
    contract: dict[str, Any],
    storyboard: list[dict[str, Any]],
    frame_count: int,
) -> str:
    return "\n".join(
        [
            f"Inspect {frame_count} chronological sampled frames from one HyperFrames visual.",
            "Return JSON with keys: passed, score, notes, missing_labels, semantic_issues, repair_directives.",
            "Score from 0 to 1.",
            "Fail if the frames are attractive but do not explain the contracted relationship.",
            "Fail invented metrics, entities, interface states, controls, or unsupported copy.",
            "Fail when the final sampled frame cannot communicate the idea as a still.",
            "Fail incoherent motion, object identity changes, unreadable text, clipping, or overlapping content.",
            f"Scene type: {contract.get('scene_type') or ''}",
            f"Thesis: {contract.get('thesis') or ''}",
            f"Takeaway: {contract.get('takeaway') or ''}",
            "Required labels: " + "; ".join(contract.get("required_labels") or []),
            "Required motion: " + "; ".join(contract.get("required_motion") or []),
            f"Screenshot test: {contract.get('screenshot_test') or ''}",
            "Forbidden content: " + "; ".join(contract.get("forbidden_content") or []),
            "Storyboard phases: "
            + "; ".join(
                f"{item.get('phase')}={item.get('visual_change')}"
                for item in storyboard[:8]
                if isinstance(item, dict)
            ),
        ]
    )


def _extract_json_object(raw_text: str) -> str:
    cleaned = str(raw_text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        return cleaned[start : end + 1]
    raise ValueError("Vision QA did not return a JSON object.")


def _string_list(value: Any, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        cleaned = " ".join(str(item or "").split()).strip()
        if cleaned and cleaned not in result:
            result.append(cleaned)
        if len(result) >= limit:
            break
    return result


def _bounded(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(number, 1.0))


__all__ = [
    "HyperframesVisionReport",
    "critique_hyperframes_frames",
]
