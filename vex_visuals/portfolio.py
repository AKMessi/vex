from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import Any, Iterable


VISUAL_PORTFOLIO_VERSION = "vex-visual-portfolio-v1"


@dataclass(frozen=True)
class VisualPortfolioIdentity:
    visual_id: str
    concept_id: str
    lane: str
    medium: str
    motion_grammar: str
    composition: str
    palette_signature: str
    renderer: str
    program_signature: str
    identity_signature: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VisualPortfolioReport:
    version: str
    visual_count: int
    score: float
    lane_entropy: float
    medium_entropy: float
    motion_entropy: float
    unique_identity_ratio: float
    lane_distribution: dict[str, int]
    medium_distribution: dict[str, int]
    motion_distribution: dict[str, int]
    renderer_distribution: dict[str, int]
    consecutive_repetitions: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "score",
            "lane_entropy",
            "medium_entropy",
            "motion_entropy",
            "unique_identity_ratio",
        ):
            payload[key] = round(float(payload[key]), 4)
        return payload


def extract_visual_portfolio_identity(
    item: dict[str, Any],
) -> VisualPortfolioIdentity:
    program = dict(item.get("open_visual_program") or {})
    concept = dict(program.get("concept") or {})
    quality = dict(program.get("quality_contract") or {})
    concept_id = str(quality.get("visual_concept_id") or "")
    concept_search = dict(item.get("visual_concept_search") or {})
    brief = next(
        (
            dict(candidate)
            for candidate in concept_search.get("concepts") or []
            if isinstance(candidate, dict)
            and str(candidate.get("concept_id") or "") == concept_id
        ),
        {},
    )
    lane = _clean(brief.get("lane") or concept.get("lane") or "unknown")
    medium = _clean(brief.get("medium") or concept.get("medium") or "unknown")
    motion = _clean(
        brief.get("motion_grammar")
        or quality.get("required_motion_grammar")
        or "unknown"
    )
    composition = _clean(
        brief.get("composition") or concept.get("composition") or "unknown"
    )
    palette = dict(program.get("palette") or {})
    palette_signature = _signature(
        {
            key: str(value).strip().casefold()
            for key, value in sorted(palette.items())
            if str(value).strip()
        }
    )
    renderer = _clean(item.get("renderer") or item.get("renderer_hint") or "unknown")
    program_signature = str(program.get("signature") or "")
    identity_payload = {
        "lane": lane,
        "medium": medium,
        "motion_grammar": motion,
        "composition": composition,
        "palette_signature": palette_signature,
        "renderer": renderer,
    }
    return VisualPortfolioIdentity(
        visual_id=str(item.get("visual_id") or ""),
        concept_id=concept_id,
        lane=lane,
        medium=medium,
        motion_grammar=motion,
        composition=composition,
        palette_signature=palette_signature,
        renderer=renderer,
        program_signature=program_signature,
        identity_signature=_signature(identity_payload),
    )


def evaluate_visual_portfolio(
    items: Iterable[dict[str, Any]],
) -> VisualPortfolioReport:
    identities = [
        extract_visual_portfolio_identity(dict(item))
        for item in items
        if isinstance(item, dict)
    ]
    count = len(identities)
    lanes = Counter(item.lane for item in identities)
    media = Counter(item.medium for item in identities)
    motions = Counter(item.motion_grammar for item in identities)
    renderers = Counter(item.renderer for item in identities)
    unique_ratio = (
        len({item.identity_signature for item in identities}) / count
        if count
        else 0.0
    )
    lane_entropy = _normalized_entropy(lanes, count)
    medium_entropy = _normalized_entropy(media, count)
    motion_entropy = _normalized_entropy(motions, count)
    if count <= 1:
        score = 1.0 if count else 0.0
    else:
        score = (
            lane_entropy * 0.3
            + medium_entropy * 0.25
            + motion_entropy * 0.2
            + unique_ratio * 0.2
            + min(len(renderers) / min(count, 2), 1.0) * 0.05
        )
    repetitions: list[dict[str, str]] = []
    for previous, current in zip(identities, identities[1:]):
        if (
            previous.lane == current.lane
            and previous.medium == current.medium
            and previous.motion_grammar == current.motion_grammar
            and previous.composition == current.composition
        ):
            repetitions.append(
                {
                    "visual_id": current.visual_id,
                    "compared_to_visual_id": previous.visual_id,
                    "lane": current.lane,
                    "medium": current.medium,
                    "motion_grammar": current.motion_grammar,
                }
            )
    warnings: list[str] = []
    if count >= 3 and lane_entropy < 0.42:
        warnings.append("portfolio_has_low_concept_lane_diversity")
    if count >= 3 and motion_entropy < 0.36:
        warnings.append("portfolio_has_low_motion_grammar_diversity")
    if repetitions:
        warnings.append("portfolio_contains_consecutive_creative_repetition")
    return VisualPortfolioReport(
        version=VISUAL_PORTFOLIO_VERSION,
        visual_count=count,
        score=max(0.0, min(score, 1.0)),
        lane_entropy=lane_entropy,
        medium_entropy=medium_entropy,
        motion_entropy=motion_entropy,
        unique_identity_ratio=unique_ratio,
        lane_distribution=dict(sorted(lanes.items())),
        medium_distribution=dict(sorted(media.items())),
        motion_distribution=dict(sorted(motions.items())),
        renderer_distribution=dict(sorted(renderers.items())),
        consecutive_repetitions=repetitions,
        warnings=warnings,
    )


def same_creative_grammar(
    first: VisualPortfolioIdentity,
    second: VisualPortfolioIdentity,
) -> bool:
    return bool(
        first.lane != "unknown"
        and first.lane == second.lane
        and first.medium == second.medium
        and first.motion_grammar == second.motion_grammar
        and first.composition == second.composition
    )


def _normalized_entropy(values: Counter[str], total: int) -> float:
    if total <= 1 or len(values) <= 1:
        return 1.0 if total == 1 else 0.0
    entropy = -sum(
        (count / total) * math.log(count / total)
        for count in values.values()
        if count > 0
    )
    return max(0.0, min(entropy / math.log(min(total, 6)), 1.0))


def _signature(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().casefold().split()) or "unknown"


__all__ = [
    "VISUAL_PORTFOLIO_VERSION",
    "VisualPortfolioIdentity",
    "VisualPortfolioReport",
    "evaluate_visual_portfolio",
    "extract_visual_portfolio_identity",
    "same_creative_grammar",
]
