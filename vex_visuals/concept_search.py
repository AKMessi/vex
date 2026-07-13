from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from typing import Any, Callable, Iterable

from vex_visuals.communication_contract import CommunicationContract
from vex_visuals.open_visual_program import sign_open_visual_program


VISUAL_CONCEPT_VERSION = "vex-visual-concept-v1"
VISUAL_REFERENCE_BOARD_VERSION = "vex-visual-reference-board-v1"
VISUAL_CONCEPT_SEARCH_VERSION = "vex-visual-concept-search-v1"

CONCEPT_LANES = (
    "physical_transformation",
    "data_explanation",
    "spatial_metaphor",
    "editorial_kinetic",
    "source_grounded_collage",
    "dimensional_system",
)

_LANE_DIRECTIONS: dict[str, dict[str, Any]] = {
    "physical_transformation": {
        "medium": "spatial_metaphor",
        "metaphor": "The idea behaves like material passing through a purpose-built machine.",
        "composition": "A dominant transformation occupies the frame; labels annotate evidence instead of forming cards.",
        "visual_nouns": ["input material", "transformation gate", "resolved output"],
        "motion": "conservation_of_form",
        "assets": "code_native",
    },
    "data_explanation": {
        "medium": "data_sculpture",
        "metaphor": "Quantities and ranked states become a living data instrument.",
        "composition": "A single quantitative comparison controls scale, density, and focal contrast.",
        "visual_nouns": ["measured field", "rank marker", "comparison scale"],
        "motion": "data_driven_morph",
        "assets": "code_native",
    },
    "spatial_metaphor": {
        "medium": "spatial_stage",
        "metaphor": "Meaning is encoded through distance, convergence, containment, and depth.",
        "composition": "Foreground evidence moves through a deep stage toward one unmistakable resolution.",
        "visual_nouns": ["depth field", "semantic path", "focal destination"],
        "motion": "camera_guided_reveal",
        "assets": "generated_vector_optional",
    },
    "editorial_kinetic": {
        "medium": "editorial_motion",
        "metaphor": "A concise thesis is assembled from evidence-bearing typographic fragments.",
        "composition": "One dominant phrase and one supporting visual gesture create magazine-level hierarchy.",
        "visual_nouns": ["hero phrase", "evidence fragment", "graphic gesture"],
        "motion": "semantic_typography",
        "assets": "code_native",
    },
    "source_grounded_collage": {
        "medium": "evidence_collage",
        "metaphor": "Real source evidence is isolated, reframed, and connected to a visual explanation.",
        "composition": "A source crop is the hero; generated annotation and cutout layers explain only what is visible.",
        "visual_nouns": ["source crop", "evidence highlight", "explanatory cutout"],
        "motion": "evidence_reframing",
        "assets": "source_first",
    },
    "dimensional_system": {
        "medium": "dimensional_diagram",
        "metaphor": "The mechanism is a layered object whose internals separate and recombine.",
        "composition": "An exploded dimensional assembly reveals relationships without using a dashboard layout.",
        "visual_nouns": ["layered assembly", "internal channel", "resolved core"],
        "motion": "exploded_view_reassembly",
        "assets": "generated_vector_optional",
    },
}


@dataclass(frozen=True)
class SemanticEncoding:
    proposition_id: str
    visual_form: str
    visual_change: str
    proof_moment: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["proof_moment"] = round(float(self.proof_moment), 4)
        return payload


@dataclass(frozen=True)
class VisualConceptBrief:
    version: str
    concept_id: str
    lane: str
    title: str
    medium: str
    metaphor: str
    composition: str
    focal_beat: str
    visual_nouns: list[str]
    semantic_encodings: list[SemanticEncoding]
    motion_grammar: str
    art_direction: dict[str, str]
    asset_strategy: str
    risk_flags: list[str] = field(default_factory=list)
    authored_by: str = "deterministic"
    signature: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "semantic_encodings": [item.to_dict() for item in self.semantic_encodings],
        }


@dataclass(frozen=True)
class ReferenceFrame:
    frame_id: str
    fraction: float
    purpose: str
    visible_proposition_ids: list[str]
    composition: str
    transformation: str
    focal_element: str
    camera: str
    hold: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["fraction"] = round(float(self.fraction), 4)
        return payload


@dataclass(frozen=True)
class VisualReferenceBoard:
    version: str
    board_id: str
    concept_id: str
    frames: list[ReferenceFrame]
    palette_intent: str
    typography_intent: str
    texture_intent: str
    target_complexity: str
    signature: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "frames": [item.to_dict() for item in self.frames],
        }


@dataclass(frozen=True)
class ConceptCandidateScore:
    concept_id: str
    eligible: bool
    score: float
    semantic_coverage: float
    visual_specificity: float
    transformation_strength: float
    novelty: float
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("score", "semantic_coverage", "visual_specificity", "transformation_strength", "novelty"):
            payload[key] = round(float(payload[key]), 4)
        return payload


@dataclass(frozen=True)
class VisualConceptSearchResult:
    version: str
    selected_concept_id: str
    concepts: list[VisualConceptBrief]
    reference_boards: list[VisualReferenceBoard]
    scores: list[ConceptCandidateScore]
    model_attempts: int
    model_concept_count: int
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "concepts": [item.to_dict() for item in self.concepts],
            "reference_boards": [item.to_dict() for item in self.reference_boards],
            "scores": [item.to_dict() for item in self.scores],
        }


def author_visual_concepts(
    spec: dict[str, Any],
    contract: CommunicationContract | dict[str, Any],
    *,
    reasoning_call: Callable[[str, str, str, str], str] | None = None,
    enable_model_authoring: bool = True,
    candidate_count: int = 6,
    history: Iterable[dict[str, Any]] | None = None,
) -> VisualConceptSearchResult:
    contract_payload = contract.to_dict() if isinstance(contract, CommunicationContract) else dict(contract or {})
    count = max(3, min(int(candidate_count), len(CONCEPT_LANES)))
    deterministic = build_visual_concept_candidates(
        spec,
        contract_payload,
        candidate_count=count,
    )
    provider = str(spec.get("generation_provider") or "").strip().lower()
    model = str(spec.get("generation_model") or "").strip()
    authored: list[VisualConceptBrief] = []
    warnings: list[str] = []
    attempts = 0
    if enable_model_authoring and reasoning_call is not None and provider in {"claude", "gemini"} and model:
        attempts = 1
        try:
            raw = reasoning_call(
                provider,
                model,
                _concept_system_prompt(),
                visual_concept_prompt(spec, contract_payload, candidate_count=min(count, 4)),
            )
            authored, rejected = normalize_authored_visual_concepts(
                _extract_json_object(raw),
                contract_payload,
                visual_id=str(contract_payload.get("visual_id") or "visual"),
            )
            if rejected:
                warnings.append(f"model_concepts_rejected:{len(rejected)}")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"model_concept_authoring_unavailable:{type(exc).__name__}")
    elif enable_model_authoring:
        warnings.append("model_concept_authoring_not_configured")

    concepts = _dedupe_concepts([*authored, *deterministic])
    scores = score_visual_concepts(concepts, contract_payload, history=history)
    eligible_ids = [item.concept_id for item in scores if item.eligible]
    selected_id = eligible_ids[0] if eligible_ids else ""
    ordered_ids = [item.concept_id for item in scores]
    concept_by_id = {item.concept_id: item for item in concepts}
    ordered_concepts = [concept_by_id[item] for item in ordered_ids if item in concept_by_id]
    boards = [build_visual_reference_board(item, contract_payload) for item in ordered_concepts]
    return VisualConceptSearchResult(
        version=VISUAL_CONCEPT_SEARCH_VERSION,
        selected_concept_id=selected_id,
        concepts=ordered_concepts,
        reference_boards=boards,
        scores=scores,
        model_attempts=attempts,
        model_concept_count=len(authored),
        warnings=_unique(warnings, limit=12),
    )


def build_visual_concept_candidates(
    spec: dict[str, Any],
    contract: dict[str, Any],
    *,
    candidate_count: int = 6,
) -> list[VisualConceptBrief]:
    count = max(1, min(int(candidate_count), len(CONCEPT_LANES)))
    proposition_ids = [
        str(item.get("proposition_id") or "")
        for item in contract.get("propositions") or []
        if isinstance(item, dict) and item.get("proposition_id")
    ]
    thesis = _clean(contract.get("thesis") or contract.get("takeaway"), limit=140)
    source_available = bool(
        spec.get("source_frame_path")
        or spec.get("source_asset")
        or (spec.get("source_asset_grounding") or {}).get("path")
    )
    result: list[VisualConceptBrief] = []
    for index, lane in enumerate(CONCEPT_LANES[:count]):
        direction = _LANE_DIRECTIONS[lane]
        risks: list[str] = []
        if lane == "source_grounded_collage" and not source_available:
            risks.append("source_asset_unavailable_use_generated_evidence_texture")
        if lane == "data_explanation" and not any(
            item.get("exact_numbers")
            for item in contract.get("propositions") or []
            if isinstance(item, dict)
        ):
            risks.append("no_exact_metric_use_rank_or_state_not_invented_numbers")
        encodings = [
            SemanticEncoding(
                proposition_id=proposition_id,
                visual_form=_visual_form(lane, proposition_index),
                visual_change=_visual_change(lane, proposition_index, len(proposition_ids)),
                proof_moment=min(0.88, 0.18 + proposition_index * (0.58 / max(len(proposition_ids) - 1, 1))),
            )
            for proposition_index, proposition_id in enumerate(proposition_ids)
        ]
        unsigned = VisualConceptBrief(
            version=VISUAL_CONCEPT_VERSION,
            concept_id=f"{_safe_id(contract.get('visual_id'))}-concept-{index + 1:02d}",
            lane=lane,
            title=thesis or f"Visual concept {index + 1}",
            medium=str(direction["medium"]),
            metaphor=str(direction["metaphor"]),
            composition=str(direction["composition"]),
            focal_beat=_focal_beat(lane, contract),
            visual_nouns=list(direction["visual_nouns"]),
            semantic_encodings=encodings,
            motion_grammar=str(direction["motion"]),
            art_direction=_art_direction(lane, spec),
            asset_strategy=str(direction["assets"]),
            risk_flags=risks,
        )
        result.append(replace(unsigned, signature=visual_concept_signature(unsigned)))
    return result


def build_visual_reference_board(
    concept: VisualConceptBrief | dict[str, Any],
    contract: CommunicationContract | dict[str, Any],
) -> VisualReferenceBoard:
    brief = concept if isinstance(concept, VisualConceptBrief) else _concept_from_payload(dict(concept or {}))
    payload = contract.to_dict() if isinstance(contract, CommunicationContract) else dict(contract or {})
    proposition_ids = [item.proposition_id for item in brief.semantic_encodings]
    thirds = max(1, (len(proposition_ids) + 2) // 3)
    frames = [
        ReferenceFrame(
            frame_id="frame_01_premise",
            fraction=0.08,
            purpose="Establish one concrete premise with immediate hierarchy.",
            visible_proposition_ids=proposition_ids[:thirds],
            composition=brief.composition,
            transformation="The first evidence-bearing form enters without decorative motion.",
            focal_element=brief.visual_nouns[0] if brief.visual_nouns else "premise",
            camera="locked establishing view",
        ),
        ReferenceFrame(
            frame_id="frame_02_mechanism",
            fraction=0.38,
            purpose="Expose the mechanism and its causal relationship.",
            visible_proposition_ids=proposition_ids[: thirds * 2],
            composition=brief.composition,
            transformation=brief.focal_beat,
            focal_element=brief.visual_nouns[1] if len(brief.visual_nouns) > 1 else "mechanism",
            camera="guided push toward the proof-bearing region",
        ),
        ReferenceFrame(
            frame_id="frame_03_proof",
            fraction=0.7,
            purpose="Make the key transformation visually undeniable.",
            visible_proposition_ids=proposition_ids,
            composition=brief.composition,
            transformation=f"Resolve the concept using {brief.motion_grammar}.",
            focal_element=brief.visual_nouns[-1] if brief.visual_nouns else "result",
            camera="stable proof view",
        ),
        ReferenceFrame(
            frame_id="frame_04_hold",
            fraction=0.94,
            purpose="Hold the complete explanation long enough to read silently.",
            visible_proposition_ids=proposition_ids,
            composition=f"Resolved {brief.composition}",
            transformation="All proof-bearing elements settle; only subtle continuity motion remains.",
            focal_element=_clean(payload.get("takeaway"), limit=140) or "resolved takeaway",
            camera="locked final hold",
            hold=True,
        ),
    ]
    unsigned = VisualReferenceBoard(
        version=VISUAL_REFERENCE_BOARD_VERSION,
        board_id=f"{brief.concept_id}-board",
        concept_id=brief.concept_id,
        frames=frames,
        palette_intent=brief.art_direction.get("palette", "controlled contrast"),
        typography_intent=brief.art_direction.get("typography", "editorial hierarchy"),
        texture_intent=brief.art_direction.get("texture", "restrained material detail"),
        target_complexity="one hero mechanism, at most two supporting layers",
    )
    return replace(unsigned, signature=visual_reference_board_signature(unsigned))


def score_visual_concepts(
    concepts: Iterable[VisualConceptBrief],
    contract: dict[str, Any],
    *,
    history: Iterable[dict[str, Any]] | None = None,
) -> list[ConceptCandidateScore]:
    required = {
        str(item.get("proposition_id") or "")
        for item in contract.get("propositions") or []
        if isinstance(item, dict) and bool(item.get("required", True))
    }
    proposition_payloads = [
        dict(item)
        for item in contract.get("propositions") or []
        if isinstance(item, dict)
    ]
    history_items = [dict(item) for item in history or [] if isinstance(item, dict)]
    scores: list[ConceptCandidateScore] = []
    for concept in concepts:
        issues = validate_visual_concept(concept, contract)
        covered = {item.proposition_id for item in concept.semantic_encodings}
        semantic_coverage = len(required & covered) / max(len(required), 1)
        specificity = min(
            1.0,
            len([item for item in concept.visual_nouns if len(item.split()) >= 2]) / 3.0 * 0.55
            + min(len(concept.art_direction), 4) / 4.0 * 0.45,
        )
        transformation = min(
            1.0,
            (0.45 if concept.focal_beat else 0.0)
            + (0.35 if concept.motion_grammar not in {"fade", "slide", "generic"} else 0.0)
            + (0.2 if any(item.visual_change for item in concept.semantic_encodings) else 0.0),
        )
        novelty = _concept_novelty(concept, history_items)
        semantic_fit = _lane_semantic_fit(concept.lane, proposition_payloads)
        score = (
            semantic_coverage * 0.32
            + specificity * 0.16
            + transformation * 0.22
            + novelty * 0.12
            + semantic_fit * 0.18
        )
        if concept.risk_flags:
            score -= min(0.12, len(concept.risk_flags) * 0.035)
        scores.append(
            ConceptCandidateScore(
                concept_id=concept.concept_id,
                eligible=not issues and semantic_coverage >= 0.99 and transformation >= 0.7,
                score=max(0.0, min(score, 1.0)),
                semantic_coverage=semantic_coverage,
                visual_specificity=specificity,
                transformation_strength=transformation,
                novelty=novelty,
                issues=issues,
            )
        )
    return sorted(scores, key=lambda item: (item.eligible, item.score, item.concept_id), reverse=True)


def validate_visual_concept(concept: VisualConceptBrief, contract: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if concept.version != VISUAL_CONCEPT_VERSION:
        issues.append("unsupported_visual_concept_version")
    if concept.lane not in CONCEPT_LANES:
        issues.append("unsupported_visual_concept_lane")
    allowed = {
        str(item.get("proposition_id") or "")
        for item in contract.get("propositions") or []
        if isinstance(item, dict)
    }
    encoded = {item.proposition_id for item in concept.semantic_encodings}
    if not allowed.issubset(encoded):
        issues.append("visual_concept_missing_semantic_encoding")
    if encoded - allowed:
        issues.append("visual_concept_invented_proposition")
    if len(concept.visual_nouns) < 2:
        issues.append("visual_concept_has_weak_visual_vocabulary")
    if not concept.focal_beat or not concept.motion_grammar:
        issues.append("visual_concept_has_no_proof_bearing_motion")
    if concept.signature != visual_concept_signature(concept):
        issues.append("visual_concept_signature_mismatch")
    return _unique(issues, limit=20)


def normalize_authored_visual_concepts(
    raw: str | dict[str, Any],
    contract: dict[str, Any],
    *,
    visual_id: str,
) -> tuple[list[VisualConceptBrief], list[dict[str, Any]]]:
    payload = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
    accepted: list[VisualConceptBrief] = []
    rejected: list[dict[str, Any]] = []
    for index, item in enumerate(payload.get("concepts") or []):
        if not isinstance(item, dict):
            rejected.append({"index": index, "issues": ["concept_is_not_an_object"]})
            continue
        normalized = dict(item)
        normalized["version"] = VISUAL_CONCEPT_VERSION
        normalized["concept_id"] = _safe_id(normalized.get("concept_id") or f"{visual_id}-authored-{index + 1:02d}")
        normalized["authored_by"] = "model"
        normalized["semantic_encodings"] = [
            {
                "proposition_id": str(value.get("proposition_id") or ""),
                "visual_form": _clean(value.get("visual_form"), limit=180),
                "visual_change": _clean(value.get("visual_change"), limit=220),
                "proof_moment": max(0.0, min(_number(value.get("proof_moment"), 0.5), 1.0)),
            }
            for value in normalized.get("semantic_encodings") or []
            if isinstance(value, dict)
        ]
        try:
            concept = _concept_from_payload(normalized)
            concept = replace(concept, signature=visual_concept_signature(concept))
        except (TypeError, ValueError) as exc:
            rejected.append({"index": index, "issues": [f"concept_parse_failed:{exc}"]})
            continue
        issues = validate_visual_concept(concept, contract)
        if issues:
            rejected.append({"index": index, "concept_id": concept.concept_id, "issues": issues})
        else:
            accepted.append(concept)
    return accepted, rejected


def apply_concept_to_program(
    program: dict[str, Any],
    concept: VisualConceptBrief,
    board: VisualReferenceBoard,
) -> dict[str, Any]:
    payload = dict(program or {})
    payload["concept"] = {
        **dict(payload.get("concept") or {}),
        "title": concept.title,
        "medium": concept.medium,
        "metaphor": concept.metaphor,
        "composition": concept.composition,
    }
    quality = dict(payload.get("quality_contract") or {})
    quality.update(
        {
            "visual_concept_id": concept.concept_id,
            "visual_concept_signature": concept.signature,
            "visual_reference_board_signature": board.signature,
            "required_motion_grammar": concept.motion_grammar,
        }
    )
    payload["quality_contract"] = quality
    _apply_lane_vocabulary(payload, concept)
    return sign_open_visual_program(payload)


def visual_concept_prompt(
    spec: dict[str, Any],
    contract: dict[str, Any],
    *,
    candidate_count: int,
) -> str:
    context = {
        "renderer": spec.get("renderer_hint"),
        "orientation": spec.get("orientation"),
        "style_pack": spec.get("style_pack"),
        "source_asset_available": bool(spec.get("source_frame_path") or spec.get("source_asset")),
        "recent_visual_fingerprints": list(spec.get("open_visual_program_history") or [])[-6:],
    }
    compact_contract = {
        "visual_id": contract.get("visual_id"),
        "thesis": contract.get("thesis"),
        "takeaway": contract.get("takeaway"),
        "propositions": contract.get("propositions"),
        "temporal_sequence": contract.get("temporal_sequence"),
        "required_terms": contract.get("required_terms"),
        "forbidden_claims": contract.get("forbidden_claims"),
    }
    return "\n".join(
        [
            f"Create exactly {max(1, min(candidate_count, 4))} radically distinct visual concept briefs.",
            "These are visual explanations, not layouts, dashboards, node graphs, card grids, or template names.",
            "Each concept must choose one lane from: " + ", ".join(CONCEPT_LANES) + ".",
            "Use different lanes. Encode every proposition_id exactly once or more through visible form and visible change.",
            "The focal beat must explain the mechanism with muted audio. Decorative motion is insufficient.",
            "Do not invent numbers, entities, claims, assets, product interfaces, or outcomes.",
            "Return JSON with key concepts. Every concept requires: concept_id, lane, title, medium, metaphor, composition, focal_beat, visual_nouns, semantic_encodings, motion_grammar, art_direction, asset_strategy, risk_flags.",
            "semantic_encodings items require proposition_id, visual_form, visual_change, proof_moment in [0,1].",
            "art_direction requires palette, typography, texture, lighting.",
            "Communication contract:",
            json.dumps(compact_contract, ensure_ascii=True, sort_keys=True),
            "Production context:",
            json.dumps(context, ensure_ascii=True, sort_keys=True),
        ]
    )


def visual_concept_signature(concept: VisualConceptBrief | dict[str, Any]) -> str:
    payload = concept.to_dict() if isinstance(concept, VisualConceptBrief) else dict(concept or {})
    payload.pop("signature", None)
    return _signature(payload)


def visual_reference_board_signature(board: VisualReferenceBoard | dict[str, Any]) -> str:
    payload = board.to_dict() if isinstance(board, VisualReferenceBoard) else dict(board or {})
    payload.pop("signature", None)
    return _signature(payload)


def _concept_system_prompt() -> str:
    return (
        "You are Vex Visual Director, a world-class motion designer and information designer. "
        "Invent visual communication systems whose geometry and motion prove the supplied facts. "
        "Avoid UI-card compositions and transcript-as-headline shortcuts. Return strict JSON only."
    )


def _visual_form(lane: str, index: int) -> str:
    forms = {
        "physical_transformation": ["material cluster", "mechanical threshold", "resolved object"],
        "data_explanation": ["measured marks", "ranked distribution", "selected signal"],
        "spatial_metaphor": ["near field", "semantic corridor", "depth destination"],
        "editorial_kinetic": ["hero phrase fragment", "graphic evidence mark", "assembled thesis"],
        "source_grounded_collage": ["source crop", "evidence cutout", "grounded annotation"],
        "dimensional_system": ["outer layer", "internal channel", "assembled core"],
    }
    values = forms[lane]
    return values[index % len(values)]


def _visual_change(lane: str, index: int, total: int) -> str:
    progress = index / max(total - 1, 1)
    if progress < 0.34:
        phase = "establishes the initial state"
    elif progress < 0.67:
        phase = "undergoes the causal transformation"
    else:
        phase = "resolves into the final state"
    return f"The {_visual_form(lane, index)} {phase} through {_LANE_DIRECTIONS[lane]['motion']}."


def _focal_beat(lane: str, contract: dict[str, Any]) -> str:
    sequence = [str(item) for item in contract.get("temporal_sequence") or [] if str(item).strip()]
    core = sequence[len(sequence) // 2] if sequence else str(contract.get("takeaway") or contract.get("thesis") or "the result becomes visible")
    return f"At the central beat, {core}; the change is encoded through {_LANE_DIRECTIONS[lane]['motion']}."


def _art_direction(lane: str, spec: dict[str, Any]) -> dict[str, str]:
    style = str(spec.get("style_pack") or "auto").replace("_", " ")
    directions = {
        "physical_transformation": ("mineral neutrals with one electric accent", "compact grotesk labels", "tactile translucent material", "directional studio light"),
        "data_explanation": ("high-legibility multicolor data palette", "tabular numerals with restrained labels", "precise matte marks", "flat analytical light"),
        "spatial_metaphor": ("deep neutral field with warm and cool semantic accents", "editorial sans display", "volumetric haze used sparingly", "depth-defining rim light"),
        "editorial_kinetic": ("paper white, ink, signal red, and one secondary hue", "large editorial display with quiet captions", "print grain and crisp rules", "graphic flat light"),
        "source_grounded_collage": ("source-derived colors with contrasting annotation", "documentary caption typography", "paper edge and masked source texture", "source-consistent light"),
        "dimensional_system": ("charcoal, glass, brass, and cyan signal accents", "technical display with humanist labels", "glass, metal, and soft shadow", "controlled architectural light"),
    }
    palette, typography, texture, lighting = directions[lane]
    return {
        "palette": palette,
        "typography": typography,
        "texture": texture,
        "lighting": lighting,
        "style_context": style,
    }


def _concept_novelty(concept: VisualConceptBrief, history: list[dict[str, Any]]) -> float:
    if not history:
        return 1.0
    penalties: list[float] = []
    nouns = {item.lower() for item in concept.visual_nouns}
    for item in history:
        same_lane = str(item.get("lane") or item.get("medium") or "").lower() in {concept.lane, concept.medium.lower()}
        historical_nouns = {str(value).lower() for value in item.get("visual_nouns") or []}
        overlap = len(nouns & historical_nouns) / max(len(nouns | historical_nouns), 1)
        penalties.append((0.58 if same_lane else 0.0) + overlap * 0.42)
    return max(0.0, 1.0 - max(penalties, default=0.0))


def _lane_semantic_fit(lane: str, propositions: list[dict[str, Any]]) -> float:
    relation_count = sum(
        1 for item in propositions if str(item.get("proposition_type") or "") == "relation"
    )
    numeric_count = sum(1 for item in propositions if item.get("exact_numbers"))
    count = len(propositions)
    if lane == "physical_transformation":
        return 1.0 if relation_count else 0.76
    if lane == "data_explanation":
        return 0.92 if numeric_count else 0.54
    if lane == "spatial_metaphor":
        return 0.86 if relation_count else 0.72
    if lane == "editorial_kinetic":
        return 0.9 if count <= 3 else 0.68
    if lane == "source_grounded_collage":
        return 0.62
    if lane == "dimensional_system":
        return 0.84 if count >= 5 and relation_count >= 2 else 0.7
    return 0.5


def _apply_lane_vocabulary(program: dict[str, Any], concept: VisualConceptBrief) -> None:
    elements = [item for item in program.get("elements") or [] if isinstance(item, dict)]
    bound = [item for item in elements if not bool(item.get("decorative")) and item.get("type") != "text"]
    if concept.lane == "data_explanation" and bound:
        bound[0]["type"] = "chart"
        bound[0]["role"] = "measured_evidence"
    elif concept.lane == "source_grounded_collage" and bound:
        bound[0]["type"] = "image"
        bound[0]["role"] = "source_evidence"
    elif concept.lane == "dimensional_system":
        for depth, item in enumerate(bound, start=1):
            item["style"] = {**dict(item.get("style") or {}), "depth": depth, "shadow": 0.32}
    elif concept.lane == "editorial_kinetic":
        for item in bound:
            item["type"] = "text"
            item["role"] = "editorial_evidence"
    elif concept.lane == "spatial_metaphor":
        for depth, item in enumerate(bound, start=1):
            item["style"] = {**dict(item.get("style") or {}), "depth": depth, "blur": max(0, 3 - depth)}


def _concept_from_payload(payload: dict[str, Any]) -> VisualConceptBrief:
    return VisualConceptBrief(
        version=str(payload.get("version") or VISUAL_CONCEPT_VERSION),
        concept_id=_safe_id(payload.get("concept_id")),
        lane=str(payload.get("lane") or ""),
        title=_clean(payload.get("title"), limit=160),
        medium=_clean(payload.get("medium"), limit=80),
        metaphor=_clean(payload.get("metaphor"), limit=300),
        composition=_clean(payload.get("composition"), limit=300),
        focal_beat=_clean(payload.get("focal_beat"), limit=320),
        visual_nouns=_unique([_clean(item, limit=100) for item in payload.get("visual_nouns") or []], limit=10),
        semantic_encodings=[
            SemanticEncoding(
                proposition_id=str(item.get("proposition_id") or ""),
                visual_form=_clean(item.get("visual_form"), limit=180),
                visual_change=_clean(item.get("visual_change"), limit=220),
                proof_moment=max(0.0, min(_number(item.get("proof_moment"), 0.5), 1.0)),
            )
            for item in payload.get("semantic_encodings") or []
            if isinstance(item, dict)
        ],
        motion_grammar=_clean(payload.get("motion_grammar"), limit=100),
        art_direction={str(key): _clean(value, limit=180) for key, value in dict(payload.get("art_direction") or {}).items()},
        asset_strategy=_clean(payload.get("asset_strategy"), limit=80),
        risk_flags=_unique([_clean(item, limit=180) for item in payload.get("risk_flags") or []], limit=12),
        authored_by=_clean(payload.get("authored_by"), limit=40) or "model",
        signature=str(payload.get("signature") or ""),
    )


def _dedupe_concepts(concepts: list[VisualConceptBrief]) -> list[VisualConceptBrief]:
    result: list[VisualConceptBrief] = []
    seen_signatures: set[str] = set()
    seen_lanes: set[str] = set()
    for concept in concepts:
        semantic_signature = _signature(
            {
                "lane": concept.lane,
                "metaphor": concept.metaphor,
                "visual_nouns": concept.visual_nouns,
                "motion": concept.motion_grammar,
            }
        )
        if semantic_signature in seen_signatures:
            continue
        # Keep one deterministic candidate per lane, while allowing a model
        # concept to replace the deterministic baseline for that lane.
        if concept.lane in seen_lanes and concept.authored_by != "model":
            continue
        seen_signatures.add(semantic_signature)
        seen_lanes.add(concept.lane)
        result.append(concept)
    return result[:10]


def _extract_json_object(raw: str) -> str:
    text = str(raw or "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Visual concept authoring did not return a JSON object.")
    return text[start : end + 1]


def _signature(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _safe_id(value: Any) -> str:
    cleaned = "".join(character if character.isalnum() or character in {"-", "_"} else "-" for character in str(value or "visual"))
    return "-".join(part for part in cleaned.strip("-").split("-") if part)[:96] or "visual"


def _clean(value: Any, *, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit].strip()


def _number(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


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


__all__ = [
    "CONCEPT_LANES",
    "VISUAL_CONCEPT_SEARCH_VERSION",
    "VISUAL_CONCEPT_VERSION",
    "VISUAL_REFERENCE_BOARD_VERSION",
    "ConceptCandidateScore",
    "ReferenceFrame",
    "SemanticEncoding",
    "VisualConceptBrief",
    "VisualConceptSearchResult",
    "VisualReferenceBoard",
    "apply_concept_to_program",
    "author_visual_concepts",
    "build_visual_concept_candidates",
    "build_visual_reference_board",
    "normalize_authored_visual_concepts",
    "score_visual_concepts",
    "validate_visual_concept",
    "visual_concept_prompt",
    "visual_concept_signature",
    "visual_reference_board_signature",
]
