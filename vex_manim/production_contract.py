from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from vex_manim.blueprint import SceneBlueprint
from vex_manim.briefs import SceneBrief
from vex_manim.visual_ir import StoryboardCritique, StoryboardFrame, VisualExplanationIR


GENERIC_LABELS = {
    "key idea",
    "input",
    "output",
    "outcome",
    "start",
    "before",
    "after",
    "mechanism",
    "signal",
    "proof",
    "core loop",
}


@dataclass
class ProductionVisualContract:
    visual_id: str
    scene_type: str
    thesis: str
    viewer_question: str
    problem_label: str
    mechanism_label: str
    resolution_label: str
    proof_label: str
    motion_spine: str
    required_beats: list[str] = field(default_factory=list)
    required_devices: list[str] = field(default_factory=list)
    copy_terms: list[str] = field(default_factory=list)
    screenshot_test: str = ""
    forbidden_patterns: list[str] = field(default_factory=list)
    quality_floor: float = 0.72
    passed: bool = True
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["quality_floor"] = round(float(self.quality_floor), 3)
        return payload


def _clean(text: Any, *, max_chars: int = 120) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip(" -,\n\t")
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip(" ,.;:-")
    return cleaned


def _phrase(text: Any, *, max_words: int = 4, max_chars: int = 34) -> str:
    cleaned = _clean(text, max_chars=max_chars * 2)
    if not cleaned:
        return ""
    tokens = re.findall(r"[A-Za-z0-9%+./-]+(?:'[A-Za-z0-9%+./-]+)?", cleaned)
    if not tokens:
        return cleaned[:max_chars].rstrip()
    skipped = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "so",
        "that",
        "the",
        "this",
        "to",
        "was",
        "with",
        "you",
        "your",
    }
    kept: list[str] = []
    for token in tokens:
        if not kept and token.lower() in skipped:
            continue
        kept.append(token)
        if len(kept) >= max_words:
            break
    candidate = " ".join(kept).strip() or " ".join(tokens[:max_words]).strip()
    if len(candidate) > max_chars:
        candidate = candidate[:max_chars].rstrip(" ,.;:-")
    return candidate


def _unique(values: list[str], *, limit: int = 8) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = _clean(value, max_chars=90)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
        if len(result) >= limit:
            break
    return result


def _object_values(ir: VisualExplanationIR) -> list[dict[str, Any]]:
    return [item.to_dict() for item in list(getattr(ir, "objects", []) or [])]


def _first_object_copy(objects: list[dict[str, Any]], roles: set[str], *fallbacks: Any) -> str:
    for item in objects:
        role = str(item.get("role") or "").strip()
        if role not in roles:
            continue
        for copy_item in list(item.get("copy") or []):
            phrase = _phrase(copy_item, max_words=4, max_chars=34)
            if phrase:
                return phrase
        meaning = _phrase(item.get("meaning"), max_words=4, max_chars=34)
        if meaning:
            return meaning
    for fallback in fallbacks:
        phrase = _phrase(fallback, max_words=4, max_chars=34)
        if phrase:
            return phrase
    return ""


def _required_devices(scene_type: str) -> list[str]:
    mapping = {
        "before_after_morph": [
            "dim wrong model",
            "bridge pulse",
            "matched transform",
            "bright resolved model",
        ],
        "signal_flow_system": [
            "source node",
            "traveling signal",
            "active mechanism hub",
            "outcome expansion",
        ],
        "guided_process_route": [
            "drawn route",
            "traveler marker",
            "step activation",
            "guided camera follow",
        ],
        "metric_proof": [
            "tracked metric",
            "drawn evidence curve",
            "moving proof marker",
            "threshold highlight",
        ],
        "interface_causality": [
            "assembled interface surface",
            "focus beam",
            "module-to-module signal",
            "result feedback trace",
        ],
        "concept_emphasis": [
            "kinetic phrase",
            "motion spine",
            "support convergence",
            "payoff lock",
        ],
    }
    return list(mapping.get(scene_type, mapping["concept_emphasis"]))


def _quality_floor(brief: SceneBrief, scene_type: str) -> float:
    base = 0.72 if brief.composition_mode == "replace" else 0.66
    if brief.animation_intensity == "high":
        base += 0.03
    elif brief.animation_intensity == "medium":
        base += 0.015
    if scene_type in {"before_after_morph", "signal_flow_system", "guided_process_route"}:
        base += 0.02
    return min(round(base, 3), 0.8)


def build_production_visual_contract(
    spec: dict[str, Any],
    brief: SceneBrief,
    ir: VisualExplanationIR,
    frames: list[StoryboardFrame],
    critique: StoryboardCritique,
    blueprint: SceneBlueprint,
) -> ProductionVisualContract:
    objects = _object_values(ir)
    scene_type = str(ir.scene_type or "concept_emphasis")
    problem_label = _first_object_copy(
        objects,
        {"before_state", "input", "process_step", "metric", "interface"},
        ir.misconception,
        brief.before_state,
        brief.headline,
    )
    mechanism_label = _first_object_copy(
        objects,
        {"causal_bridge", "mechanism", "motion_spine", "attention_driver", "chart", "focus"},
        ir.proof_signal,
        brief.cause,
        blueprint.focal_system,
    )
    resolution_label = _first_object_copy(
        objects,
        {"after_state", "outcome", "result", "contrast", "payoff", "core_idea"},
        ir.correct_model,
        brief.after_state,
        ir.claim,
    )
    proof_label = _phrase(ir.proof_signal or brief.effect or brief.deck or mechanism_label, max_words=5, max_chars=42)
    thesis = _clean(ir.visual_goal or brief.mental_model or brief.viewer_takeaway or brief.objective, max_chars=160)
    required_beats = _unique(
        [
            *list(ir.required_motion or []),
            *(frame.required_change for frame in frames),
        ],
        limit=6,
    )
    copy_terms = _unique(
        [
            _phrase(ir.claim, max_words=5, max_chars=44),
            problem_label,
            mechanism_label,
            resolution_label,
            proof_label,
            *[
                _phrase(line, max_words=4, max_chars=34)
                for item in objects
                for line in list(item.get("copy") or [])
            ],
        ],
        limit=8,
    )
    warnings: list[str] = []
    generic_count = sum(
        1
        for label in [problem_label, mechanism_label, resolution_label]
        if not label or label.strip().lower() in GENERIC_LABELS
    )
    if generic_count >= 2:
        warnings.append("semantic labels are too generic to carry a premium visual")
    if not critique.passed:
        warnings.extend(list(critique.fatal_issues or [])[:2])
    if len(required_beats) < 3:
        warnings.append("motion contract has too few concrete beats")
    passed = not warnings
    return ProductionVisualContract(
        visual_id=brief.visual_id,
        scene_type=scene_type,
        thesis=thesis,
        viewer_question=_clean(ir.viewer_question, max_chars=100),
        problem_label=problem_label,
        mechanism_label=mechanism_label,
        resolution_label=resolution_label,
        proof_label=proof_label,
        motion_spine=_clean(blueprint.focal_system or blueprint.archetype, max_chars=90),
        required_beats=required_beats,
        required_devices=_required_devices(scene_type),
        copy_terms=copy_terms,
        screenshot_test=(
            "A paused final frame must show the problem, mechanism, and payoff relationship "
            "without needing transcript text."
        ),
        forbidden_patterns=_unique(list(ir.forbidden_patterns or []) + list(brief.must_avoid or []), limit=10),
        quality_floor=_quality_floor(brief, scene_type),
        passed=passed,
        warnings=warnings[:6],
    )


def production_contract_prompt_block(contract: ProductionVisualContract | dict[str, Any] | None) -> str:
    if contract is None:
        return ""
    payload = contract.to_dict() if isinstance(contract, ProductionVisualContract) else dict(contract)
    lines = [
        "Production visual contract:",
        f"- Intuition thesis: {payload.get('thesis') or ''}",
        f"- Viewer question: {payload.get('viewer_question') or ''}",
        (
            "- Semantic labels: "
            f"problem={payload.get('problem_label') or '(none)'}; "
            f"mechanism={payload.get('mechanism_label') or '(none)'}; "
            f"resolution={payload.get('resolution_label') or '(none)'}; "
            f"proof={payload.get('proof_label') or '(none)'}"
        ),
        f"- Motion spine: {payload.get('motion_spine') or ''}",
        "- Required beats: " + "; ".join(list(payload.get("required_beats") or [])),
        "- Required devices: " + "; ".join(list(payload.get("required_devices") or [])),
        "- Copy terms to prefer: " + "; ".join(list(payload.get("copy_terms") or [])),
        f"- Screenshot test: {payload.get('screenshot_test') or ''}",
        f"- Quality floor: {float(payload.get('quality_floor') or 0.0):.2f}",
    ]
    warnings = list(payload.get("warnings") or [])
    if warnings:
        lines.append("- Contract warnings: " + "; ".join(str(item) for item in warnings))
    return "\n".join(lines)
