"""Frame-conditioned suggestions; the model never edits pixels or executes code."""

from __future__ import annotations

import json
import copy
from pathlib import Path
from typing import Any
from vex_visuals.evidence import payload_digest, select_evidence_frames
from vex_visuals.repair import RepairLevel, TypedRepairOperation, VisualRepairPlan
from vex_visuals.verifier import (
    VisualVerifierReport,
    VisionRequest,
    _configured_provider_models,
)


def propose_frame_repairs(
    spec: dict[str, Any],
    report: VisualVerifierReport,
    frames: list[Path],
    *,
    round_index: int,
    request: VisionRequest | None = None,
) -> tuple[VisualRepairPlan | None, dict[str, Any]]:
    program = dict(spec.get("open_visual_program") or {})
    source_program_signature = str(program.get("signature") or "")
    if not frames or not program:
        return None, {"available": False, "reason": "missing_frame_or_program_evidence"}
    endpoints = (
        [(report.provider, report.model)]
        if report.available and report.provider
        else _configured_provider_models()
    )
    program = copy.deepcopy(program)
    for element in program.get("elements") or []:
        if isinstance(element.get("asset"), dict):
            element["asset"].pop("data_uri", None)
    program = {
        "palette": program.get("palette"),
        "canvas": program.get("canvas"),
        "element_columns": [
            "element_id",
            "type",
            "text",
            "layout",
            "style",
            "parent_id",
        ],
        "elements": [
            [
                item.get(key)
                for key in [
                    "element_id",
                    "type",
                    "text",
                    "layout",
                    "style",
                    "parent_id",
                ]
            ]
            for item in program.get("elements") or []
        ],
        "track_columns": ["track_id", "target_id", "property", "samples"],
        "tracks": [
            [
                track.get("track_id"),
                track.get("target_id"),
                track.get("property"),
                [
                    [key.get("t"), key.get("value"), key.get("easing", "linear")]
                    for key in track.get("keyframes") or []
                ],
            ]
            for track in program.get("tracks") or []
        ],
    }
    contract = spec.get("visual_communication_contract") or {}
    compact_contract = {
        key: contract.get(key) for key in ["thesis", "takeaway", "forbidden_claims"]
    }
    compact_contract["propositions"] = [
        {
            key: item.get(key)
            for key in ["proposition_id", "proposition", "exact_numbers"]
        }
        for item in contract.get("propositions") or []
    ]
    compact_report = {
        "state": report.state.value,
        "issues": report.issues,
        "dimensions": {
            dimension.name: {"score": dimension.score, "issues": dimension.issues}
            for dimension in [
                report.semantic,
                report.design,
                report.temporal,
                report.technical,
            ]
        },
    }
    prompt = "\n".join(
        [
            "You are a motion designer repairing the actual numbered QA frames. Return JSON only.",
            "Identify specific visible failures using frame_index, target_id, observed, and requirement.",
            "Propose at most 3 small operations and 3 short counterexamples (each field under 120 characters). Do not change evidence bindings, invent facts, add code, or remove required meaning.",
            "Allowed operations: move(target_id,x,y), resize(target_id,width,height), replace_text(target_id,text), set_style(target_id,style), set_motion(target_id,keyframes).",
            "Coordinates are normalized [0,1]. Style may use font_size,font_weight,stroke_width,opacity,radius,fill,stroke. Motion target_id must be an existing track ID.",
            "Input track samples are compact [t,value,easing] arrays. Output set_motion keyframes must be objects with t,value,easing fields.",
            "Return {counterexamples:[{frame_index,target_id,observed,requirement}],operations:[{op,target_id,...}]} or empty arrays if no supported improvement is needed.",
            "SOURCE CONTRACT: "
            + json.dumps(compact_contract, ensure_ascii=True, separators=(",", ":")),
            "CURRENT PROGRAM: "
            + json.dumps(program, ensure_ascii=True, separators=(",", ":")),
            "OBSERVED VERIFICATION: "
            + json.dumps(compact_report, ensure_ascii=True, separators=(",", ":")),
        ]
    )
    errors = []
    for provider, model in endpoints:
        try:
            if request:
                payload = request(provider, model, prompt, frames)
            else:
                from providers.multimodal import request_visual_json
                from vex_visuals.model_contracts import REPAIR_SCHEMA

                supplied_frames = (
                    select_evidence_frames(frames, limit=2)
                    if provider == "groq"
                    else frames
                )
                payload = request_visual_json(
                    provider,
                    model,
                    prompt,
                    supplied_frames,
                    role="repair",
                    schema=REPAIR_SCHEMA if provider == "groq" else None,
                )
            operations = payload.get("operations")
            counterexamples = payload.get("counterexamples")
            if (
                not isinstance(operations, list)
                or not isinstance(counterexamples, list)
                or len(operations) > 8
            ):
                raise ValueError("Invalid native frame-repair proposal")
            for operation in operations:
                if isinstance(operation, dict) and isinstance(
                    operation.get("style"), dict
                ):
                    operation["style"] = {
                        key: value
                        for key, value in operation["style"].items()
                        if value is not None
                    }
            allowed = {"move", "resize", "replace_text", "set_style", "set_motion"}
            if any(
                not isinstance(item, dict)
                or item.get("op") not in allowed
                or not item.get("target_id")
                for item in operations
            ):
                raise ValueError("Unsupported native frame-repair operation")
            diagnostics = {
                "available": True,
                "provider": provider,
                "model": model,
                "counterexamples": counterexamples[:12],
                "operation_count": len(operations),
                "supplied_frame_count": (
                    len(frames)
                    if request or provider != "groq"
                    else len(supplied_frames)
                ),
            }
            if not operations:
                return None, diagnostics
            operation = TypedRepairOperation(
                f"frame-repair-{round_index}",
                RepairLevel.EXECUTION,
                "frame_patch",
                "Native vision inspected QA frames and proposed targeted scene changes",
                parameters={"operations": operations,"source_program_signature":source_program_signature},
            )
            unsigned = {
                "version": "vex-typed-visual-repair-v1",
                "repair_id": f"{spec.get('visual_id', 'visual')}-native-{round_index}",
                "round_index": round_index,
                "source_state": report.state.value,
                "requires_concept_regeneration": False,
                "operations": [operation.to_dict()],
            }
            return (
                VisualRepairPlan(
                    **{
                        **unsigned,
                        "source_state": report.state,
                        "operations": [operation],
                        "signature": payload_digest(unsigned),
                    }
                ),
                diagnostics,
            )
        except (OSError, ValueError, TypeError, KeyError) as exc:
            errors.append(type(exc).__name__)
        except Exception as exc:
            # Preserve deterministic repair when an external model is unavailable.
            response = getattr(exc, "response", None)
            errors.append(f"{type(exc).__name__}:{getattr(response,'status_code','')}")
    return None, {"available": False, "errors": errors}
