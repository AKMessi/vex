from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import config
from vex_hyperframes.counterexamples import (
    COUNTEREXAMPLE_VERSION,
    CriticReport,
    VisualCounterexample,
    VisualCriticBundle,
    parse_counterexamples,
)
from vex_hyperframes.scene_program import validate_scene_program


def run_visual_critics(
    frame_paths: list[Path],
    *,
    production_contract: dict[str, Any],
    visual_explanation_ir: dict[str, Any],
    scene_program: dict[str, Any],
    render_trace: dict[str, Any],
    quality_report: dict[str, Any],
    vision_report: dict[str, Any] | None,
    source_asset_grounding: dict[str, Any] | None = None,
    model_name: str | None = None,
) -> VisualCriticBundle:
    blind = build_blind_critic_report(
        vision_report or {},
        production_contract=production_contract,
        scene_program=scene_program,
        render_trace=render_trace,
    )
    grounded = build_local_grounded_critic(
        production_contract=production_contract,
        visual_explanation_ir=visual_explanation_ir,
        scene_program=scene_program,
        render_trace=render_trace,
        source_asset_grounding=source_asset_grounding or {},
    )
    design = build_local_design_critic(
        scene_program=scene_program,
        render_trace=render_trace,
        quality_report=quality_report,
    )
    vision_grounded, vision_design = _request_grounded_design_critique(
        frame_paths,
        production_contract=production_contract,
        visual_explanation_ir=visual_explanation_ir,
        scene_program=scene_program,
        render_trace=render_trace,
        model_name=model_name,
    )
    grounded = _merge_reports(grounded, vision_grounded)
    design = _merge_reports(design, vision_design)
    counterexamples = _deduplicate(
        [
            *blind.counterexamples,
            *grounded.counterexamples,
            *design.counterexamples,
        ]
    )
    hard_failure_count = sum(
        1 for item in counterexamples if item.severity == "hard_failure"
    )
    available_scores = [
        report.score
        for report in (blind, grounded, design)
        if report.score is not None
    ]
    score = (
        sum(float(item) for item in available_scores) / len(available_scores)
        if available_scores
        else 0.0
    )
    passed = (
        hard_failure_count == 0
        and all(report.passed is not False for report in (blind, grounded, design))
    )
    return VisualCriticBundle(
        version=COUNTEREXAMPLE_VERSION,
        passed=passed,
        score=score,
        blind=blind,
        grounded=grounded,
        design=design,
        counterexamples=counterexamples,
        hard_failure_count=hard_failure_count,
    )


def build_blind_critic_report(
    vision_report: dict[str, Any],
    *,
    production_contract: dict[str, Any],
    scene_program: dict[str, Any],
    render_trace: dict[str, Any],
) -> CriticReport:
    if not vision_report.get("available"):
        return CriticReport(
            critic="blind",
            available=False,
            passed=None,
            score=None,
            notes=str(vision_report.get("notes") or "Blind critic unavailable."),
            model=str(vision_report.get("model") or ""),
            error=str(vision_report.get("error") or ""),
        )
    graph = dict(production_contract.get("visual_claim_graph") or {})
    node_by_label = {
        _normalize(item.get("label")): str(item.get("node_id") or "")
        for item in graph.get("nodes") or []
        if isinstance(item, dict)
    }
    element_by_object = {
        str(item.get("object_id") or ""): str(item.get("element_id") or "")
        for item in scene_program.get("elements") or []
        if isinstance(item, dict)
    }
    raw: list[dict[str, Any]] = []
    for label in vision_report.get("missing_labels") or []:
        object_id = node_by_label.get(_normalize(label), "")
        element_id = element_by_object.get(object_id, "")
        raw.append(
            {
                "issue_type": "missing_object",
                "severity": "hard_failure",
                "summary": f"Blind viewer could not identify {label}.",
                "expected": f"The grounded object {label} is visually identifiable.",
                "observed": "The object was absent or visually ambiguous.",
                "element_ids": [element_id] if element_id else [],
                "confidence": 0.94,
                "allowed_repairs": [
                    "persist_element",
                    "strengthen_hierarchy",
                    "change_layout_family",
                ],
            }
        )
    for relation_id in vision_report.get("missing_relation_ids") or []:
        raw.append(
            {
                "issue_type": "missing_relation",
                "severity": "hard_failure",
                "summary": f"Blind viewer could not recover {relation_id}.",
                "expected": "The required relation is directly decodable from geometry and timing.",
                "observed": "The relation could not be inferred from the rendered frames.",
                "relation_ids": [relation_id],
                "confidence": 0.96,
                "allowed_repairs": [
                    "strengthen_relation",
                    "swap_proof_encoding",
                ],
            }
        )
    decoded = dict(vision_report.get("decoded_claim") or {})
    if decoded.get("unsupported_visual_claims"):
        raw.append(
            {
                "issue_type": "unsupported_content",
                "severity": "hard_failure",
                "summary": "Blind viewer inferred unsupported visual claims.",
                "expected": "Every implied claim is backed by signed evidence.",
                "observed": "; ".join(
                    str(item)
                    for item in decoded.get("unsupported_visual_claims") or []
                ),
                "confidence": 0.92,
                "allowed_repairs": ["remove_unsupported_content"],
            }
        )
    if float(vision_report.get("thesis_score") or 0.0) < 0.36:
        raw.append(
            {
                "issue_type": "ambiguous_thesis",
                "severity": "hard_failure",
                "summary": "Blind viewer could not recover the intended thesis.",
                "expected": str(production_contract.get("takeaway") or ""),
                "observed": str(decoded.get("thesis") or "No coherent thesis decoded."),
                "confidence": 0.9,
                "allowed_repairs": [
                    "strengthen_hierarchy",
                    "swap_proof_encoding",
                ],
            }
        )
    counterexamples = parse_counterexamples(
        raw,
        critic="blind",
        scene_program=scene_program,
        render_trace=render_trace,
    )
    return CriticReport(
        critic="blind",
        available=True,
        passed=bool(vision_report.get("passed")),
        score=_optional_score(vision_report.get("score")),
        counterexamples=counterexamples,
        notes=str(vision_report.get("notes") or ""),
        model=str(vision_report.get("model") or ""),
        error=str(vision_report.get("error") or ""),
    )


def build_local_grounded_critic(
    *,
    production_contract: dict[str, Any],
    visual_explanation_ir: dict[str, Any],
    scene_program: dict[str, Any],
    render_trace: dict[str, Any],
    source_asset_grounding: dict[str, Any],
) -> CriticReport:
    graph = dict(production_contract.get("visual_claim_graph") or {})
    validation = validate_scene_program(
        scene_program,
        ir=visual_explanation_ir,
        claim_graph=graph,
    )
    raw: list[dict[str, Any]] = []
    for error in validation.errors:
        issue_type = "weak_grounding"
        repairs = ["change_layout_family"]
        element_ids: list[str] = []
        relation_ids: list[str] = []
        if "overlap" in error or "bounds" in error:
            issue_type = "overlap"
            repairs = ["move_element", "change_layout_family"]
        elif "copy_is_not_grounded" in error or "evidence" in error:
            issue_type = "unsupported_content"
            repairs = ["remove_unsupported_content"]
        elif "relation" in error:
            issue_type = "missing_relation"
            repairs = ["strengthen_relation", "swap_proof_encoding"]
            relation_ids = [
                relation_id
                for relation_id in _ids_from_error(error)
                if relation_id.startswith("relation_")
            ]
        elif "object_coverage" in error or "unknown_object" in error:
            issue_type = "missing_object"
            repairs = ["persist_element", "change_layout_family"]
            element_ids = [
                element_id
                for element_id in _ids_from_error(error)
                if element_id.startswith("element_")
            ]
        raw.append(
            {
                "issue_type": issue_type,
                "severity": "hard_failure",
                "summary": error.replace("_", " "),
                "expected": "The signed scene program exactly preserves grounded objects and relations.",
                "observed": error,
                "element_ids": element_ids,
                "relation_ids": relation_ids,
                "confidence": 1.0,
                "allowed_repairs": repairs,
            }
        )
    scene_type = str(production_contract.get("scene_type") or "")
    if (
        scene_type == "grounded_interface_walkthrough"
        and (
            not source_asset_grounding.get("asset_path")
            or source_asset_grounding.get("embedded") is False
        )
    ):
        raw.append(
            {
                "issue_type": "source_asset_required",
                "severity": "hard_failure",
                "summary": "The interface explanation has no real source frame.",
                "expected": "A grounded interface visual uses an approved local source image.",
                "observed": "The scene would reconstruct UI from labels alone.",
                "confidence": 1.0,
                "allowed_repairs": ["bind_source_asset", "reroute_renderer"],
            }
        )
    counterexamples = parse_counterexamples(
        raw,
        critic="grounded",
        scene_program=scene_program,
        render_trace=render_trace,
    )
    score = max(0.0, 1.0 - 0.22 * len(counterexamples))
    return CriticReport(
        critic="grounded",
        available=True,
        passed=not any(
            item.severity == "hard_failure" for item in counterexamples
        ),
        score=score,
        counterexamples=counterexamples,
        notes="Deterministic evidence, source, and claim-graph checks completed.",
    )


def build_local_design_critic(
    *,
    scene_program: dict[str, Any],
    render_trace: dict[str, Any],
    quality_report: dict[str, Any],
) -> CriticReport:
    raw: list[dict[str, Any]] = []
    elements = [
        dict(item)
        for item in scene_program.get("elements") or []
        if isinstance(item, dict)
    ]
    for item in elements:
        element_id = str(item.get("element_id") or "")
        text = str(item.get("text") or "")
        if len(text) > 68:
            raw.append(
                {
                    "issue_type": "overflow",
                    "severity": "error",
                    "summary": f"{element_id} carries too much copy.",
                    "expected": "A concise label that remains readable at delivery size.",
                    "observed": f"{len(text)} visible characters.",
                    "element_ids": [element_id],
                    "confidence": 0.88,
                    "allowed_repairs": ["resize_element", "reduce_density"],
                }
            )
    if len(elements) >= 6:
        mean_area = sum(
            float(item.get("width") or 0.0) * float(item.get("height") or 0.0)
            for item in elements
        ) / len(elements)
        if mean_area > 0.072:
            raw.append(
                {
                    "issue_type": "density",
                    "severity": "error",
                    "summary": "The composition is too dense for rapid comprehension.",
                    "expected": "Clear grouping and sufficient negative space.",
                    "observed": f"{len(elements)} elements average {mean_area:.3f} frame area.",
                    "element_ids": [
                        str(item.get("element_id") or "") for item in elements
                    ],
                    "confidence": 0.9,
                    "allowed_repairs": [
                        "reduce_density",
                        "change_layout_family",
                    ],
                }
            )
    emphasis_values = [
        float(item.get("emphasis") or 0.5) for item in elements
    ]
    if len(emphasis_values) >= 3 and max(emphasis_values) - min(emphasis_values) < 0.12:
        raw.append(
            {
                "issue_type": "hierarchy",
                "severity": "warning",
                "summary": "The visual hierarchy is too flat.",
                "expected": "One proof-bearing object dominates, with supporting objects subordinate.",
                "observed": "Element emphasis values are nearly uniform.",
                "element_ids": [
                    str(item.get("element_id") or "") for item in elements
                ],
                "confidence": 0.84,
                "allowed_repairs": ["strengthen_hierarchy", "resize_element"],
            }
        )
    quality_issues = [
        *list(quality_report.get("issues") or []),
        *list(quality_report.get("semantic_issues") or []),
    ]
    for issue in quality_issues:
        normalized = str(issue or "").lower()
        if "edge" in normalized or "overflow" in normalized:
            issue_type = "overflow"
            repairs = ["move_element", "resize_element"]
        elif "static" in normalized or "motion" in normalized:
            issue_type = "motion"
            repairs = ["retime_reveal"]
        elif "sparse" in normalized or "dead space" in normalized:
            issue_type = "hierarchy"
            repairs = ["strengthen_hierarchy", "change_layout_family"]
        else:
            continue
        raw.append(
            {
                "issue_type": issue_type,
                "severity": "error",
                "summary": str(issue),
                "expected": "The render meets the premium design QA thresholds.",
                "observed": str(issue),
                "confidence": 0.82,
                "allowed_repairs": repairs,
            }
        )
    counterexamples = parse_counterexamples(
        raw,
        critic="design",
        scene_program=scene_program,
        render_trace=render_trace,
    )
    score = max(
        0.0,
        min(
            float(quality_report.get("score") or 1.0),
            1.0 - 0.1 * len(counterexamples),
        ),
    )
    return CriticReport(
        critic="design",
        available=True,
        passed=not any(
            item.severity in {"error", "hard_failure"}
            for item in counterexamples
        ),
        score=score,
        counterexamples=counterexamples,
        notes="Deterministic hierarchy, density, copy, and frame-quality checks completed.",
    )


def _request_grounded_design_critique(
    frame_paths: list[Path],
    *,
    production_contract: dict[str, Any],
    visual_explanation_ir: dict[str, Any],
    scene_program: dict[str, Any],
    render_trace: dict[str, Any],
    model_name: str | None,
) -> tuple[CriticReport, CriticReport]:
    selected_model = str(
        model_name
        or getattr(config, "HYPERFRAMES_VISION_MODEL", "")
        or config.GEMINI_MODEL
        or ""
    ).strip()
    if not bool(getattr(config, "HYPERFRAMES_ENABLE_VISION_QA", False)):
        return _unavailable_pair("Vision critics are disabled.", selected_model)
    if not config.GEMINI_API_KEY:
        return _unavailable_pair(
            "Vision critics skipped because GEMINI_API_KEY is not configured.",
            selected_model,
        )
    max_frames = int(getattr(config, "HYPERFRAMES_MAX_CRITIC_FRAMES", 8))
    usable_frames = [
        Path(path)
        for path in frame_paths[:max_frames]
        if Path(path).is_file()
    ]
    if not usable_frames:
        return _unavailable_pair(
            "Vision critics skipped because no sampled frames were available.",
            selected_model,
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
                text=_grounded_design_prompt(
                    production_contract=production_contract,
                    visual_explanation_ir=visual_explanation_ir,
                    scene_program=scene_program,
                    render_trace=render_trace,
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
                    "You are two independent visual critics: a grounded relevance "
                    "auditor and a senior motion design director. Return only JSON. "
                    "Never propose new facts, labels, entities, or metrics."
                ),
                model_name=selected_model,
            ),
        )
        payload = json.loads(
            _extract_json_object(getattr(response, "text", "") or "")
        )
        grounded = _report_from_vision_payload(
            payload.get("grounded"),
            critic="grounded",
            scene_program=scene_program,
            render_trace=render_trace,
            model=selected_model,
        )
        design = _report_from_vision_payload(
            payload.get("design"),
            critic="design",
            scene_program=scene_program,
            render_trace=render_trace,
            model=selected_model,
        )
        return grounded, design
    except Exception as exc:  # noqa: BLE001
        error = " ".join(str(exc).split())[:500]
        return (
            CriticReport(
                critic="grounded",
                available=False,
                passed=None,
                score=None,
                notes="Grounded vision critic request failed.",
                model=selected_model,
                error=error,
            ),
            CriticReport(
                critic="design",
                available=False,
                passed=None,
                score=None,
                notes="Design vision critic request failed.",
                model=selected_model,
                error=error,
            ),
        )


def _grounded_design_prompt(
    *,
    production_contract: dict[str, Any],
    visual_explanation_ir: dict[str, Any],
    scene_program: dict[str, Any],
    render_trace: dict[str, Any],
    frame_count: int,
) -> str:
    compact_contract = {
        "thesis": production_contract.get("thesis"),
        "takeaway": production_contract.get("takeaway"),
        "scene_type": production_contract.get("scene_type"),
        "required_labels": production_contract.get("required_labels"),
        "visual_claim_graph": production_contract.get("visual_claim_graph"),
        "forbidden_content": visual_explanation_ir.get("forbidden_content"),
    }
    compact_program = {
        "layout_family": scene_program.get("layout_family"),
        "elements": scene_program.get("elements"),
        "relations": scene_program.get("relations"),
        "captures": render_trace.get("captures"),
    }
    return "\n".join(
        [
            f"You will receive {frame_count} chronological rendered frames.",
            "Audit the pixels against the signed contract and scene program below.",
            "Grounded critic: reject irrelevant, fabricated, generic, or weakly evidenced imagery.",
            "Design critic: reject weak hierarchy, collisions, unreadable copy, arbitrary decoration, poor pacing, and amateur composition.",
            "Every issue must point to existing element_ids, relation_ids, frame_id, or normalized regions when possible.",
            "Allowed issue_type values: ambiguous_thesis, density, hierarchy, missing_object, missing_relation, motion, overflow, overlap, pacing, source_asset_required, unsupported_content, weak_grounding, weak_relation_encoding.",
            "Allowed severity values: info, warning, error, hard_failure.",
            "Allowed repairs: bind_source_asset, change_layout_family, move_element, persist_element, reduce_density, remove_unsupported_content, resize_element, reroute_renderer, retime_reveal, strengthen_hierarchy, strengthen_relation, swap_proof_encoding.",
            "Return exactly: {grounded:{passed,score,notes,counterexamples:[]},design:{passed,score,notes,counterexamples:[]}}.",
            "Each counterexample requires: issue_type,severity,summary,expected,observed,confidence,frame_id,element_ids,relation_ids,evidence_ids,regions,allowed_repairs.",
            "Do not invent replacement copy or visual facts.",
            "SIGNED CONTRACT:",
            json.dumps(compact_contract, ensure_ascii=True, sort_keys=True),
            "SCENE PROGRAM AND TRACE:",
            json.dumps(compact_program, ensure_ascii=True, sort_keys=True),
        ]
    )


def _report_from_vision_payload(
    payload: Any,
    *,
    critic: str,
    scene_program: dict[str, Any],
    render_trace: dict[str, Any],
    model: str,
) -> CriticReport:
    data = dict(payload) if isinstance(payload, dict) else {}
    counterexamples = parse_counterexamples(
        data.get("counterexamples"),
        critic=critic,
        scene_program=scene_program,
        render_trace=render_trace,
    )
    score = _optional_score(data.get("score"))
    passed_value = data.get("passed")
    passed = (
        bool(passed_value)
        if isinstance(passed_value, bool)
        else not any(
            item.severity in {"error", "hard_failure"}
            for item in counterexamples
        )
    )
    return CriticReport(
        critic=critic,
        available=True,
        passed=passed,
        score=score,
        counterexamples=counterexamples,
        notes=str(data.get("notes") or ""),
        model=model,
    )


def _merge_reports(local: CriticReport, vision: CriticReport) -> CriticReport:
    if not vision.available:
        return CriticReport(
            critic=local.critic,
            available=local.available,
            passed=local.passed,
            score=local.score,
            counterexamples=local.counterexamples,
            notes=" ".join(
                item
                for item in [local.notes, vision.notes]
                if item
            ),
            model=vision.model,
            error=vision.error,
        )
    counterexamples = _deduplicate(
        [*local.counterexamples, *vision.counterexamples]
    )
    scores = [
        score
        for score in (local.score, vision.score)
        if score is not None
    ]
    return CriticReport(
        critic=local.critic,
        available=True,
        passed=local.passed is not False and vision.passed is not False,
        score=sum(scores) / len(scores) if scores else None,
        counterexamples=counterexamples,
        notes=" ".join(item for item in [local.notes, vision.notes] if item),
        model=vision.model,
        error=vision.error,
    )


def _unavailable_pair(
    notes: str,
    model: str,
) -> tuple[CriticReport, CriticReport]:
    return (
        CriticReport("grounded", False, None, None, notes=notes, model=model),
        CriticReport("design", False, None, None, notes=notes, model=model),
    )


def _deduplicate(
    values: list[VisualCounterexample],
) -> list[VisualCounterexample]:
    result: list[VisualCounterexample] = []
    seen: set[tuple[Any, ...]] = set()
    severity_rank = {
        "info": 0,
        "warning": 1,
        "error": 2,
        "hard_failure": 3,
    }
    for item in sorted(
        values,
        key=lambda value: (
            -severity_rank.get(value.severity, 0),
            value.critic,
            value.counterexample_id,
        ),
    ):
        key = (
            item.issue_type,
            tuple(item.element_ids),
            tuple(item.relation_ids),
            item.frame_id,
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


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
    raise ValueError("Visual critics did not return a JSON object.")


def _ids_from_error(value: str) -> list[str]:
    return re.findall(r"(?:element|relation)_[a-zA-Z0-9_-]+", value)


def _normalize(value: Any) -> str:
    return re.sub(r"[^a-z0-9%+./-]+", " ", str(value or "").lower()).strip()


def _optional_score(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(number, 1.0))


__all__ = [
    "build_blind_critic_report",
    "build_local_design_critic",
    "build_local_grounded_critic",
    "run_visual_critics",
]
