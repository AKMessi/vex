from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import config
from vex_hyperframes.inverse_decoder import (
    INVERSE_DECODER_VERSION,
    BlindFrameDecode,
    blind_decode_prompt,
    build_counterfactual_frames,
    evaluate_inverse_decode,
    parse_blind_decode,
)


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
    decoder_version: str = INVERSE_DECODER_VERSION
    decoded_claim: dict[str, Any] = field(default_factory=dict)
    thesis_score: float | None = None
    object_coverage: float | None = None
    relation_coverage: float | None = None
    sequence_score: float | None = None
    missing_relation_ids: list[str] = field(default_factory=list)
    counterfactual: dict[str, Any] = field(default_factory=dict)
    counterfactual_artifacts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def critique_hyperframes_frames(
    frame_paths: list[Path],
    *,
    production_contract: dict[str, Any],
    storyboard: list[dict[str, Any]],
    proof_encoding: str = "",
    model_name: str | None = None,
) -> HyperframesVisionReport:
    del storyboard
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
    max_frames = int(getattr(config, "HYPERFRAMES_MAX_CRITIC_FRAMES", 8))
    usable_frames = [
        Path(path)
        for path in frame_paths[:max_frames]
        if Path(path).is_file()
    ]
    if not usable_frames:
        return HyperframesVisionReport(
            available=False,
            passed=None,
            score=None,
            notes="Vision QA was skipped because no sampled frames were available.",
            model=selected_model,
        )
    counterfactual_enabled = bool(
        getattr(config, "HYPERFRAMES_ENABLE_COUNTERFACTUAL_QA", True)
    )
    try:
        from google import genai

        client = genai.Client(
            api_key=config.GEMINI_API_KEY,
            http_options=config.google_genai_http_options(),
        )
        decoded = _request_blind_decode(
            client,
            selected_model,
            usable_frames,
        )
        ablated_decode = None
        scrambled_decode = None
        artifact_payload: dict[str, Any] = {}
        if counterfactual_enabled:
            output_dir = usable_frames[0].parent / "inverse_decoder_counterfactuals"
            ablated_frames, scrambled_frames = build_counterfactual_frames(
                usable_frames,
                output_dir,
                encoding_family=proof_encoding,
            )
            ablated_decode = _request_blind_decode(
                client,
                selected_model,
                ablated_frames,
            )
            scrambled_decode = _request_blind_decode(
                client,
                selected_model,
                scrambled_frames,
            )
            artifact_payload = {
                "relation_ablation_frames": [
                    str(path) for path in ablated_frames
                ],
                "temporal_scramble_frames": [
                    str(path) for path in scrambled_frames
                ],
                "relation_ablation_decode": ablated_decode.to_dict(),
                "temporal_scramble_decode": scrambled_decode.to_dict(),
            }
        evaluation = evaluate_inverse_decode(
            decoded,
            production_contract=production_contract,
            relation_ablation_decode=ablated_decode,
            temporal_scramble_decode=scrambled_decode,
            min_score=float(
                getattr(config, "HYPERFRAMES_BLIND_DECODER_MIN_SCORE", 0.68)
            ),
            require_counterfactuals=counterfactual_enabled,
        )
        evaluation_payload = evaluation.to_dict()
        notes = (
            f"Blind decoder inferred: {decoded.thesis}"
            if decoded.thesis
            else "Blind decoder could not infer a coherent thesis."
        )
        return HyperframesVisionReport(
            available=True,
            passed=evaluation.passed,
            score=evaluation.score,
            notes=notes,
            missing_labels=evaluation.missing_labels,
            semantic_issues=evaluation.issues,
            repair_directives=evaluation.repair_directives,
            model=selected_model,
            decoded_claim=decoded.to_dict(),
            thesis_score=evaluation.thesis_score,
            object_coverage=evaluation.object_coverage,
            relation_coverage=evaluation.relation_coverage,
            sequence_score=evaluation.sequence_score,
            missing_relation_ids=evaluation.missing_relation_ids,
            counterfactual=dict(evaluation_payload["counterfactual"]),
            counterfactual_artifacts=artifact_payload,
        )
    except Exception as exc:  # noqa: BLE001
        return HyperframesVisionReport(
            available=False,
            passed=None,
            score=None,
            notes="Blind inverse-decoder QA request failed.",
            model=selected_model,
            error=" ".join(str(exc).split())[:500],
        )


def _request_blind_decode(
    client: Any,
    model_name: str,
    frame_paths: list[Path],
) -> BlindFrameDecode:
    from google.genai import types

    contents: list[Any] = [
        types.Part.from_text(text=blind_decode_prompt(len(frame_paths)))
    ]
    for path in frame_paths:
        contents.append(
            types.Part.from_bytes(
                data=Path(path).read_bytes(),
                mime_type="image/png",
            )
        )
    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=config.build_gemini_generation_config(
            (
                "You are a blind inverse-graphics decoder. Infer only what is visible "
                "in the supplied frames. Never assume an intended answer. Return only JSON."
            ),
            model_name=model_name,
        ),
    )
    payload = json.loads(
        _extract_json_object(getattr(response, "text", "") or "")
    )
    return parse_blind_decode(payload)


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
    raise ValueError("Blind inverse decoder did not return a JSON object.")


__all__ = [
    "HyperframesVisionReport",
    "critique_hyperframes_frames",
]
