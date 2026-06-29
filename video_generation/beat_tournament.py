from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from video_generation.director import BeatContract, DirectorPackage
from video_generation.models import Beat, ScriptPlan, VideoGenerationRequest
from vex_hyperframes.variants import HyperframesVariant


BEAT_TOURNAMENT_VERSION = "hyperframes-beat-tournament-v1"

_LOW_VALUE_LABELS = {
    "action",
    "better",
    "clear",
    "context",
    "idea",
    "input",
    "output",
    "result",
    "signal",
    "simple",
    "system",
    "useful",
    "workflow",
}


@dataclass(frozen=True)
class VariantScore:
    variant_id: str
    variant_index: int
    score: float
    passed: bool
    reasons: list[str]
    penalties: list[str]
    medium_family: str = ""
    world_signature: str = ""
    template: str = ""
    scene_type: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BeatTournamentResult:
    version: str
    beat_id: str
    selected_variant_id: str
    selected_variant_index: int
    selected_score: float
    passed: bool
    selection_reason: str
    records: list[VariantScore]
    issues: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "beat_id": self.beat_id,
            "selected_variant_id": self.selected_variant_id,
            "selected_variant_index": self.selected_variant_index,
            "selected_score": self.selected_score,
            "passed": self.passed,
            "selection_reason": self.selection_reason,
            "records": [record.to_dict() for record in self.records],
            "issues": list(self.issues),
            "warnings": list(self.warnings),
        }


def select_directed_variant(
    variants: list[HyperframesVariant],
    *,
    beat: Beat,
    request: VideoGenerationRequest,
    plan: ScriptPlan,
    director_package: DirectorPackage | None,
    used_medium_families: list[str],
    used_world_signatures: set[str],
) -> tuple[HyperframesVariant | None, BeatTournamentResult]:
    contract = director_package.beat_contract(beat.beat_id) if director_package else None
    records = [
        _score_variant(
            variant,
            beat=beat,
            request=request,
            plan=plan,
            contract=contract,
            used_medium_families=used_medium_families,
            used_world_signatures=used_world_signatures,
        )
        for variant in variants
    ]
    eligible = [record for record in records if record.passed]
    if not eligible:
        issues = ["no_variant_passed_authoring_tournament"] if records else ["no_variants_to_score"]
        return None, BeatTournamentResult(
            version=BEAT_TOURNAMENT_VERSION,
            beat_id=beat.beat_id,
            selected_variant_id="",
            selected_variant_index=-1,
            selected_score=0.0,
            passed=False,
            selection_reason="no eligible variant cleared the authoring score floor",
            records=records,
            issues=issues,
            warnings=[],
        )
    selected_record = max(
        eligible,
        key=lambda record: (
            record.score,
            1 if record.medium_family not in used_medium_families[-2:] else 0,
            1 if record.world_signature not in used_world_signatures else 0,
            -record.variant_index,
        ),
    )
    selected_variant = next(
        variant
        for variant in variants
        if variant.variant_id == selected_record.variant_id
        and variant.variant_index == selected_record.variant_index
    )
    warnings: list[str] = []
    if selected_record.score < 0.74:
        warnings.append("selected_variant_cleared_floor_with_low_margin")
    return selected_variant, BeatTournamentResult(
        version=BEAT_TOURNAMENT_VERSION,
        beat_id=beat.beat_id,
        selected_variant_id=selected_record.variant_id,
        selected_variant_index=selected_record.variant_index,
        selected_score=selected_record.score,
        passed=True,
        selection_reason="highest scoring variant after semantic, diversity, and motion priors",
        records=records,
        issues=[],
        warnings=warnings,
    )


def _score_variant(
    variant: HyperframesVariant,
    *,
    beat: Beat,
    request: VideoGenerationRequest,
    plan: ScriptPlan,
    contract: BeatContract | None,
    used_medium_families: list[str],
    used_world_signatures: set[str],
) -> VariantScore:
    spec = dict(variant.spec)
    qa_contract = dict(spec.get("qa_contract") or {})
    ir = dict(spec.get("visual_explanation_ir") or {})
    world = dict(spec.get("visual_world_program") or {})
    scene_program = dict(spec.get("scene_program_v2") or {})
    semantic_frame = dict(spec.get("semantic_frame") or ir.get("semantic_frame") or {})
    required_labels = [
        str(item).strip().lower()
        for item in qa_contract.get("required_labels") or spec.get("required_labels") or []
        if str(item).strip()
    ]
    scene_type = str(ir.get("scene_type") or "")
    medium = str(world.get("medium_family") or "")
    signature = str(world.get("world_signature") or world.get("fingerprint") or "")
    template = str(spec.get("template") or "")
    score = 0.48
    reasons: list[str] = []
    penalties: list[str] = []

    if template.startswith("semantic_"):
        score += 0.08
        reasons.append("semantic_template")
    if len(required_labels) >= 2:
        score += 0.1
        reasons.append("source_grounded_labels")
    if len(semantic_frame) >= 2:
        score += 0.1
        reasons.append("rich_semantic_frame")
    if world:
        score += 0.09
        reasons.append("visual_world_program")
    if scene_program:
        score += 0.08
        reasons.append("scene_program_v2")
    if str(spec.get("proof_program_id") or ""):
        score += 0.07
        reasons.append("proof_program_candidate")
    if _scene_matches(beat.scene_type, scene_type, template):
        score += 0.07
        reasons.append("scene_type_matches_beat")
    if contract is not None:
        coverage = _contract_coverage(contract, required_labels, spec)
        if coverage >= 0.5:
            score += 0.08
            reasons.append("director_contract_covered")
        elif contract.required_objects:
            score -= 0.08
            penalties.append("director_contract_undercovered")
    if medium and medium not in used_medium_families[-2:]:
        score += 0.045
        reasons.append("medium_diversity")
    elif medium:
        score -= 0.055
        penalties.append("medium_repeats_recent_beats")
    if signature and signature not in used_world_signatures:
        score += 0.04
        reasons.append("world_signature_diversity")
    elif signature:
        score -= 0.08
        penalties.append("world_signature_duplicate")
    if _has_generic_labels(required_labels):
        score -= 0.11
        penalties.append("generic_required_labels")
    if _fake_metric_risk(beat, spec):
        score -= 0.08
        penalties.append("metric_without_source_number")
    if _caption_only_risk(spec):
        score -= 0.1
        penalties.append("caption_carries_explanation")
    if _style_matches_request(request, plan, spec):
        score += 0.035
        reasons.append("style_matches_request")

    rounded = round(max(0.0, min(score, 1.0)), 4)
    return VariantScore(
        variant_id=variant.variant_id,
        variant_index=variant.variant_index,
        score=rounded,
        passed=rounded >= 0.62 and "caption_carries_explanation" not in penalties,
        reasons=reasons,
        penalties=penalties,
        medium_family=medium,
        world_signature=signature,
        template=template,
        scene_type=scene_type,
    )


def _contract_coverage(
    contract: BeatContract,
    required_labels: list[str],
    spec: dict[str, Any],
) -> float:
    if not contract.required_objects:
        return 1.0
    haystack = _normalize(
        " ".join(
            [
                *required_labels,
                str(spec.get("context_text") or ""),
                str(spec.get("headline") or ""),
                str(spec.get("sentence_text") or ""),
            ]
        )
    )
    hits = 0
    for item in contract.required_objects:
        tokens = [token for token in _words(item) if len(token) >= 4]
        if not tokens:
            continue
        if any(token in haystack for token in tokens):
            hits += 1
    return hits / max(len(contract.required_objects), 1)


def _scene_matches(beat_scene: str, compiled_scene: str, template: str) -> bool:
    text = f"{compiled_scene} {template}".lower()
    if beat_scene == "process":
        return any(item in text for item in ("process", "flow", "transform", "causal"))
    if beat_scene == "contrast":
        return any(item in text for item in ("transform", "contrast", "state"))
    if beat_scene == "metric":
        return any(item in text for item in ("metric", "data", "proof"))
    if beat_scene == "proof":
        return any(item in text for item in ("proof", "evidence", "causal", "transform"))
    if beat_scene == "hook":
        return any(item in text for item in ("quote", "transform", "state", "proof"))
    return bool(compiled_scene or template)


def _has_generic_labels(labels: list[str]) -> bool:
    if not labels:
        return True
    generic = 0
    for label in labels:
        normalized = _normalize(label)
        if normalized in _LOW_VALUE_LABELS:
            generic += 1
    return generic / max(len(labels), 1) >= 0.5


def _fake_metric_risk(beat: Beat, spec: dict[str, Any]) -> bool:
    text = f"{beat.narration} {spec.get('context_text', '')}"
    return beat.scene_type == "metric" and not re.search(r"\d", text)


def _caption_only_risk(spec: dict[str, Any]) -> bool:
    qa_contract = dict(spec.get("qa_contract") or {})
    labels = " ".join(
        str(item)
        for item in (
            qa_contract.get("required_labels")
            or spec.get("required_labels")
            or []
        )
    )
    world = dict(spec.get("visual_world_program") or {})
    scene_program = dict(spec.get("scene_program_v2") or {})
    objects = (
        world.get("objects")
        or world.get("object_bindings")
        or scene_program.get("elements")
        or spec.get("semantic_objects")
        or []
    )
    return len(_words(labels)) >= 18 and not objects


def _style_matches_request(
    request: VideoGenerationRequest,
    plan: ScriptPlan,
    spec: dict[str, Any],
) -> bool:
    text = _normalize(
        " ".join(
            [
                request.style,
                plan.design_direction,
                str(spec.get("style_pack") or ""),
                str(spec.get("context_text") or ""),
            ]
        )
    )
    return any(item in text for item in ("cinematic", "kinetic", "signal", "proof", "premium"))


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9%+./-]+", " ", str(value or "").lower()).strip()


def _words(value: str) -> list[str]:
    return re.findall(r"[a-z0-9%+./-]+", str(value or "").lower())
