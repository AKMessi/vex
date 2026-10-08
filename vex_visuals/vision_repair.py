"""Frame-conditioned suggestions; the model never edits pixels or executes code."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from vex_visuals.evidence import payload_digest
from vex_visuals.repair import RepairLevel, TypedRepairOperation, VisualRepairPlan
from vex_visuals.verifier import VisualVerifierReport, VisionRequest, _configured_provider_models, _default_vision_request


def propose_frame_repairs(spec: dict[str, Any], report: VisualVerifierReport, frames: list[Path], *, round_index: int, request: VisionRequest | None = None) -> tuple[VisualRepairPlan | None, dict[str, Any]]:
    program = dict(spec.get("open_visual_program") or {})
    if not frames or not program:
        return None, {"available": False, "reason": "missing_frame_or_program_evidence"}
    endpoints = [(report.provider, report.model)] if report.available and report.provider else _configured_provider_models()
    prompt = "\n".join([
        "You are a motion designer repairing the actual numbered QA frames. Return JSON only.",
        "Identify specific visible failures using frame_index, target_id, observed, and requirement.",
        "Propose at most 8 small operations. Do not change evidence bindings, invent facts, add code, or remove required meaning.",
        "Allowed operations: move(target_id,x,y), resize(target_id,width,height), replace_text(target_id,text), set_style(target_id,style), set_motion(target_id,keyframes).",
        "Coordinates are normalized [0,1]. Style may use font_size,font_weight,stroke_width,opacity,radius,fill,stroke. Motion target_id must be an existing track ID.",
        "Return {counterexamples:[{frame_index,target_id,observed,requirement}],operations:[{op,target_id,...}]} or empty arrays if no supported improvement is needed.",
        "SOURCE CONTRACT: " + json.dumps(spec.get("visual_communication_contract") or {}, ensure_ascii=True),
        "CURRENT PROGRAM: " + json.dumps(program, ensure_ascii=True),
        "OBSERVED VERIFICATION: " + json.dumps(report.to_dict(), ensure_ascii=True),
    ])
    errors = []
    for provider, model in endpoints:
        try:
            payload = (request or _default_vision_request)(provider, model, prompt, frames)
            operations = payload.get("operations")
            counterexamples = payload.get("counterexamples")
            if not isinstance(operations, list) or not isinstance(counterexamples, list) or len(operations) > 8:
                raise ValueError("Invalid native frame-repair proposal")
            allowed = {"move", "resize", "replace_text", "set_style", "set_motion"}
            if any(not isinstance(item, dict) or item.get("op") not in allowed or not item.get("target_id") for item in operations):
                raise ValueError("Unsupported native frame-repair operation")
            diagnostics = {"available": True, "provider": provider, "model": model, "counterexamples": counterexamples[:12], "operation_count": len(operations)}
            if not operations:
                return None, diagnostics
            operation = TypedRepairOperation(f"frame-repair-{round_index}", RepairLevel.EXECUTION, "frame_patch", "Native vision inspected QA frames and proposed targeted scene changes", parameters={"operations": operations})
            unsigned = {"version": "vex-typed-visual-repair-v1", "repair_id": f"{spec.get('visual_id', 'visual')}-native-{round_index}", "round_index": round_index, "source_state": report.state.value, "requires_concept_regeneration": False, "operations": [operation.to_dict()]}
            return VisualRepairPlan(**{**unsigned, "source_state": report.state, "operations": [operation], "signature": payload_digest(unsigned)}), diagnostics
        except (OSError, ValueError, TypeError, KeyError) as exc:
            errors.append(type(exc).__name__)
        except Exception as exc:
            # Preserve deterministic repair when an external model is unavailable.
            errors.append(type(exc).__name__)
    return None, {"available": False, "errors": errors}
