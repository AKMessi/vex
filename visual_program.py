from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any


_STYLE_BY_VISUAL_TYPE = {
    "data_graphic": "bold_tech",
    "product_ui": "product_ui",
    "process": "signal_lab",
    "abstract_motion": "cinematic_night",
    "cutaway": "magazine_luxe",
    "location": "documentary_kinetic",
}

_CONCEPT_COLORS = (
    "#38BDF8",
    "#F59E0B",
    "#22C55E",
    "#FB7185",
    "#A78BFA",
    "#2DD4BF",
    "#FACC15",
    "#60A5FA",
)

_CONCEPT_MOTIFS = (
    "signal_route",
    "metric_orbit",
    "interface_stack",
    "contrast_beam",
    "decision_grid",
    "loop_trace",
    "cutaway_layers",
    "proof_ladder",
)


def _truncate(value: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)].rstrip(" ,.;:-") + "..."


@dataclass(frozen=True)
class VideoChapter:
    chapter_id: str
    start: float
    end: float
    summary: str
    card_ids: list[str] = field(default_factory=list)
    visual_density: str = "medium"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConceptMemory:
    concept_id: str
    label: str
    keywords: list[str]
    color: str
    motif: str
    first_card_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VisualBeat:
    card_id: str
    start: float
    end: float
    role: str
    text: str
    intuition_mode: str
    payoff: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VisualEpisode:
    episode_id: str
    start: float
    end: float
    card_ids: list[str]
    chapter_id: str
    purpose: str
    beats: list[VisualBeat]
    template_family: str
    continuity_group: str
    concept_ids: list[str]
    motif: str
    density_role: str
    transition_in: dict[str, Any]
    transition_out: dict[str, Any]
    qa_contract: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["beats"] = [beat.to_dict() for beat in self.beats]
        return payload


@dataclass(frozen=True)
class StyleBible:
    primary_style_pack: str
    motion_language: str
    continuity_rules: list[str]
    density_target: str
    recurring_motifs: list[str]
    concept_palette: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VisualNarrativeProgram:
    program_id: str
    duration_sec: float
    summary: str
    density_target: str
    style_bible: StyleBible
    chapters: list[VideoChapter]
    concept_memory: list[ConceptMemory]
    episodes: list[VisualEpisode]

    def to_dict(self) -> dict[str, Any]:
        return {
            "program_id": self.program_id,
            "duration_sec": self.duration_sec,
            "summary": self.summary,
            "density_target": self.density_target,
            "style_bible": self.style_bible.to_dict(),
            "chapters": [chapter.to_dict() for chapter in self.chapters],
            "concept_memory": [concept.to_dict() for concept in self.concept_memory],
            "episodes": [episode.to_dict() for episode in self.episodes],
        }


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9%+.-]+", str(text or "").lower())


def _clean_label(value: Any, *, max_chars: int = 42) -> str:
    cleaned = re.sub(r"\s+", " ", str(value or "")).strip(" ,.;:-")
    return _truncate(cleaned, max_chars) if cleaned else ""


def _card_id(card: dict[str, Any]) -> str:
    return str(card.get("card_id") or "").strip()


def _card_start(card: dict[str, Any]) -> float:
    return _as_float(card.get("start"), 0.0)


def _card_end(card: dict[str, Any]) -> float:
    return max(_card_start(card), _as_float(card.get("end"), _card_start(card)))


def _chapter_summary(cards: list[dict[str, Any]]) -> str:
    if not cards:
        return "Supporting explanation"
    first = cards[0]
    semantic = dict(first.get("semantic_frame") or {})
    takeaway = _clean_label(semantic.get("viewer_takeaway"), max_chars=54)
    if takeaway:
        return takeaway
    text = _clean_label(first.get("sentence_text"), max_chars=54)
    return text or "Supporting explanation"


def _visual_density(card_count: int, duration: float) -> str:
    if duration <= 0:
        return "medium"
    cards_per_minute = card_count / max(duration / 60.0, 0.1)
    if cards_per_minute >= 8.0:
        return "high"
    if cards_per_minute <= 3.0:
        return "low"
    return "medium"


def _build_chapters(cards: list[dict[str, Any]], clip_duration: float, scene_cuts: list[float]) -> list[VideoChapter]:
    if clip_duration <= 0:
        return []
    semantic_episode_ids = [
        str(card.get("semantic_episode_id") or "").strip()
        for card in cards
    ]
    if cards and all(semantic_episode_ids):
        grouped: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        current_id = ""
        for card in sorted(cards, key=_card_start):
            episode_id = str(card.get("semantic_episode_id") or "").strip()
            if current and episode_id != current_id:
                grouped.append(current)
                current = []
            current.append(card)
            current_id = episode_id
        if current:
            grouped.append(current)
        return [
            VideoChapter(
                chapter_id=str(group[0].get("semantic_episode_id") or f"chapter_{index:02d}"),
                start=round(min(_card_start(card) for card in group), 2),
                end=round(max(_as_float(card.get("end"), _card_start(card)) for card in group), 2),
                summary=_truncate(
                    str(group[0].get("semantic_episode_summary") or _chapter_summary(group)),
                    180,
                ),
                card_ids=[_card_id(card) for card in group if _card_id(card)],
                visual_density=_visual_density(
                    len(group),
                    max(
                        max(_as_float(card.get("end"), _card_start(card)) for card in group)
                        - min(_card_start(card) for card in group),
                        0.1,
                    ),
                ),
            )
            for index, group in enumerate(grouped, start=1)
        ]
    if clip_duration <= 35:
        boundaries = [0.0, clip_duration]
    else:
        chapter_count = max(2, min(5, round(clip_duration / 35.0)))
        raw_boundaries = [clip_duration * index / chapter_count for index in range(chapter_count + 1)]
        boundaries = [0.0]
        for boundary in raw_boundaries[1:-1]:
            nearby = [cut for cut in scene_cuts if abs(cut - boundary) <= 2.0]
            boundaries.append(round(min(nearby, key=lambda cut: abs(cut - boundary)) if nearby else boundary, 2))
        boundaries.append(clip_duration)
    chapters: list[VideoChapter] = []
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:]), start=1):
        if end - start < 0.5:
            continue
        chapter_cards = [card for card in cards if _card_start(card) >= start - 0.01 and _card_start(card) < end + 0.01]
        chapters.append(
            VideoChapter(
                chapter_id=f"chapter_{index:02d}",
                start=round(start, 2),
                end=round(end, 2),
                summary=_chapter_summary(chapter_cards),
                card_ids=[_card_id(card) for card in chapter_cards if _card_id(card)],
                visual_density=_visual_density(len(chapter_cards), end - start),
            )
        )
    return chapters


def _chapter_for_card(card: dict[str, Any], chapters: list[VideoChapter]) -> str:
    start = _card_start(card)
    for chapter in chapters:
        if chapter.start - 0.01 <= start <= chapter.end + 0.01:
            return chapter.chapter_id
    return chapters[-1].chapter_id if chapters else "chapter_01"


def _primary_style_pack(cards: list[dict[str, Any]]) -> str:
    if not cards:
        return "editorial_clean"
    scores: dict[str, float] = {}
    for card in cards:
        visual_type = str(card.get("visual_type_hint") or "").strip().lower()
        style = str(card.get("style_pack") or _STYLE_BY_VISUAL_TYPE.get(visual_type) or "editorial_clean")
        scores[style] = scores.get(style, 0.0) + max(_as_float(card.get("priority"), 0.0), 1.0)
    return max(scores, key=scores.get)


def _concept_label(card: dict[str, Any]) -> str:
    semantic = dict(card.get("semantic_frame") or {})
    for value in (
        semantic.get("viewer_takeaway"),
        semantic.get("after_state"),
        semantic.get("before_state"),
        *(card.get("keywords") or [])[:2],
        card.get("sentence_text"),
    ):
        label = _clean_label(value, max_chars=34)
        if label:
            return label
    return "Key idea"


def _concept_key(label: str) -> str:
    words = _as_words(label)
    return "_".join(words[:4]) or "concept"


def _build_concept_memory(cards: list[dict[str, Any]], *, limit: int = 8) -> list[ConceptMemory]:
    concepts: list[ConceptMemory] = []
    seen: set[str] = set()
    ranked = sorted(cards, key=lambda card: (_as_float(card.get("priority"), 0.0), -_card_start(card)), reverse=True)
    for card in ranked:
        label = _concept_label(card)
        key = _concept_key(label)
        if key in seen:
            continue
        index = len(concepts)
        concepts.append(
            ConceptMemory(
                concept_id=f"concept_{index + 1:02d}",
                label=label,
                keywords=[_clean_label(item, max_chars=24) for item in (card.get("keywords") or [])[:4] if _clean_label(item, max_chars=24)],
                color=_CONCEPT_COLORS[index % len(_CONCEPT_COLORS)],
                motif=_CONCEPT_MOTIFS[index % len(_CONCEPT_MOTIFS)],
                first_card_id=_card_id(card),
            )
        )
        seen.add(key)
        if len(concepts) >= limit:
            break
    return concepts


def _concepts_for_card(card: dict[str, Any], concepts: list[ConceptMemory]) -> list[str]:
    haystack = " ".join(
        [
            str(card.get("sentence_text") or ""),
            str(card.get("context_text") or ""),
            " ".join(str(item) for item in (card.get("keywords") or [])),
        ]
    ).lower()
    selected: list[str] = []
    for concept in concepts:
        needles = [concept.label, *concept.keywords]
        if any(str(needle).lower() in haystack for needle in needles if str(needle).strip()):
            selected.append(concept.concept_id)
        if len(selected) >= 2:
            break
    if selected:
        return selected
    return [concepts[0].concept_id] if concepts else []


def _template_family(card: dict[str, Any]) -> str:
    semantic = dict(card.get("semantic_frame") or {})
    mode = str(card.get("intuition_mode") or semantic.get("intuition_mode") or "").strip().lower()
    sentence = f"{card.get('sentence_text', '')} {card.get('context_text', '')}".lower()
    if re.search(r"\b(?:myth|misconception|wrong|false|truth|actually)\b", sentence):
        return "myth_buster"
    if re.search(r"\b(?:problem|pain|bottleneck|stuck|issue)\b", sentence) and re.search(r"\b(?:solution|fix|solve|instead|better)\b", sentence):
        return "problem_solution"
    if re.search(r"\b(?:risk|danger|threat|blind spot|failure|warning)\b", sentence):
        return "risk_radar"
    if re.search(r"\b(?:opportunity|leverage|channel|market|audience|landscape)\b", sentence):
        return "opportunity_map"
    if re.search(r"\b(?:score|grade|rating|readiness|criteria|quality)\b", sentence):
        return "scorecard"
    if re.search(r"\b(?:checklist|requirements?|must|need to)\b", sentence):
        return "checklist_reveal"
    if re.search(r"\b(?:focus|attention|signal|noise|priority)\b", sentence):
        return "focus_ring"
    if re.search(r"\b(?:momentum|accelerate|growth|trend|curve|spike)\b", sentence):
        return "momentum_wave"
    if re.search(r"\b(?:pipeline|x-?ray|stack|workflow)\b", sentence) and re.search(r"\b(?:inside|hidden|stage|layer)\b", sentence):
        return "pipeline_xray"
    if re.search(r"\b(?:blueprint|mechanism|how it works|under the hood)\b", sentence):
        return "mechanism_blueprint"
    if re.search(r"\b(?:if|else|branch|route)\b", sentence) and re.search(r"\b(?:then|otherwise|or)\b", sentence):
        return "decision_tree"
    if mode == "causal_chain":
        return "causal_chain"
    if mode == "process_route" and re.search(r"\b(loop|cycle|feedback|repeat|compound|iterate)\b", sentence):
        return "flywheel_loop"
    if mode == "process_route":
        return "kinetic_route"
    if mode == "misconception_flip" and re.search(r"\b(choice|choose|option|tradeoff|decision)\b", sentence):
        return "decision_matrix"
    if mode == "misconception_flip":
        return "contrast_ladder"
    if mode == "metric_proof":
        if re.search(r"\b(?:pulse|spike|threshold|live|feedback)\b", sentence):
            return "data_pulse"
        return "data_journey"
    if mode == "interface_walkthrough":
        return "interface_cascade"
    if re.search(r"\b(layer|inside|component|anatomy|breakdown|under the hood)\b", sentence):
        return "anatomy_cutaway"
    if re.search(r"\b(top|rank|priority|order|first|second|third)\b", sentence):
        return "stack_ranking"
    if re.search(r"\b(phase|chapter|sequence|timeline|moment|moments)\b", sentence):
        return "timeline_filmstrip"
    if re.search(r"\b(map|landscape|space|category|categories|ecosystem)\b", sentence):
        return "market_map"
    return "ribbon_quote"


def _transition_for_card(card: dict[str, Any], *, direction: str) -> dict[str, Any]:
    pause_key = "pause_before" if direction == "in" else "pause_after"
    pause = _as_float(card.get(pause_key), 0.0)
    scene_distance = _as_float(card.get("scene_distance"), 999.0)
    replace_safety = _as_float(card.get("replace_safety"), 0.0)
    if scene_distance <= 0.22:
        kind = "scene_match_cut"
        duration = 0.16
    elif pause >= 0.42 or replace_safety >= 0.68:
        kind = "soft_luma_fade"
        duration = 0.24
    elif pause >= 0.18:
        kind = "audio_bridge_fade"
        duration = 0.2
    else:
        kind = "micro_dissolve"
        duration = 0.14
    return {
        "kind": kind,
        "duration_sec": duration,
        "direction": direction,
        "reason": f"pause={pause:.2f}s scene_distance={scene_distance:.2f}s",
    }


def _beat_for_card(card: dict[str, Any]) -> VisualBeat:
    semantic = dict(card.get("semantic_frame") or {})
    text = (
        _clean_label(semantic.get("viewer_takeaway"), max_chars=54)
        or _clean_label(semantic.get("after_state"), max_chars=54)
        or _clean_label(card.get("sentence_text"), max_chars=54)
        or "Key idea"
    )
    return VisualBeat(
        card_id=_card_id(card),
        start=round(_card_start(card), 2),
        end=round(_card_end(card), 2),
        role=str(card.get("intuition_role") or semantic.get("intuition_role") or "supporting_example"),
        text=text,
        intuition_mode=str(card.get("intuition_mode") or semantic.get("intuition_mode") or "concept_emphasis"),
        payoff=round(_as_float(card.get("intuition_payoff") or semantic.get("intuition_payoff"), 0.0), 3),
    )


def _episode_cards(primary: dict[str, Any], cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    primary_start = _card_start(primary)
    primary_mode = str(primary.get("intuition_mode") or "").strip().lower()
    neighbors = [
        card
        for card in cards
        if _card_id(card) != _card_id(primary)
        and abs(_card_start(card) - primary_start) <= 5.5
        and (
            str(card.get("intuition_mode") or "").strip().lower() == primary_mode
            or _as_float(card.get("intuition_payoff"), 0.0) >= 0.74
        )
    ]
    ranked_neighbors = sorted(neighbors, key=lambda card: (abs(_card_start(card) - primary_start), -_as_float(card.get("priority"), 0.0)))
    selected = sorted([primary, *ranked_neighbors[:2]], key=_card_start)
    return selected


def _candidate_cards(cards: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    ranked = sorted(cards, key=lambda card: (_as_float(card.get("priority"), 0.0), -_card_start(card)), reverse=True)
    selected: list[dict[str, Any]] = []
    seen_novelty: set[str] = set()
    for card in ranked:
        card_id = _card_id(card)
        if not card_id:
            continue
        visualizability = _as_float(card.get("visualizability"), 0.0)
        payoff = _as_float(card.get("intuition_payoff"), 0.0)
        generic = _as_float(card.get("generic_penalty"), 0.0)
        if payoff < 0.5 or (visualizability < 0.38 and generic > 0.5):
            continue
        novelty = str(card.get("novelty_key") or card_id)
        if novelty in seen_novelty:
            continue
        if any(abs(_card_start(card) - _card_start(existing)) < 1.1 for existing in selected):
            continue
        selected.append(card)
        seen_novelty.add(novelty)
        if len(selected) >= budget:
            break
    return sorted(selected, key=_card_start)


def build_visual_narrative_program(
    cards: list[dict[str, Any]],
    *,
    clip_duration: float,
    max_visuals: int,
    scene_cuts: list[float] | None = None,
    prefer_premium: bool = False,
) -> VisualNarrativeProgram:
    scene_cuts = scene_cuts or []
    sorted_cards = sorted(cards, key=_card_start)
    chapters = _build_chapters(sorted_cards, clip_duration, scene_cuts)
    concepts = _build_concept_memory(sorted_cards)
    episode_budget = max(max_visuals * 2, max_visuals + 3)
    candidates = _candidate_cards(sorted_cards, episode_budget)
    episodes: list[VisualEpisode] = []
    used_cards: set[str] = set()
    for index, primary in enumerate(candidates, start=1):
        if _card_id(primary) in used_cards:
            continue
        grouped_cards = [card for card in _episode_cards(primary, sorted_cards) if _card_id(card) not in used_cards]
        if not grouped_cards:
            continue
        for card in grouped_cards:
            used_cards.add(_card_id(card))
        beats = [_beat_for_card(card) for card in grouped_cards]
        start = min(beat.start for beat in beats)
        end = max(beat.end for beat in beats)
        concept_ids = _concepts_for_card(primary, concepts)
        motif = next((concept.motif for concept in concepts if concept.concept_id in concept_ids), _CONCEPT_MOTIFS[index % len(_CONCEPT_MOTIFS)])
        density_role = "anchor" if _as_float(primary.get("intuition_payoff"), 0.0) >= 0.78 else "support"
        episodes.append(
            VisualEpisode(
                episode_id=f"episode_{len(episodes) + 1:03d}",
                start=round(start, 2),
                end=round(end, 2),
                card_ids=[beat.card_id for beat in beats],
                chapter_id=_chapter_for_card(primary, chapters),
                purpose=_truncate(str((primary.get("semantic_frame") or {}).get("mental_model") or primary.get("sentence_text") or "Make the spoken idea visible."), 180),
                beats=beats,
                template_family=_template_family(primary),
                continuity_group=f"{_chapter_for_card(primary, chapters)}:{concept_ids[0] if concept_ids else 'concept_00'}",
                concept_ids=concept_ids,
                motif=motif,
                density_role=density_role,
                transition_in=_transition_for_card(primary, direction="in"),
                transition_out=_transition_for_card(grouped_cards[-1], direction="out"),
                qa_contract={
                    "must_preserve_narrative_context": True,
                    "max_join_luma_jump": 0.28,
                    "min_boundary_motion_delta": 0.006 if prefer_premium else 0.004,
                    "avoid_duplicate_concept_within_sec": 9.0,
                    "source_to_visual_transition_required": True,
                },
            )
        )
        if len(episodes) >= episode_budget:
            break
    density_target = "high" if max_visuals >= 7 or clip_duration >= 90 else ("medium" if max_visuals >= 3 else "low")
    primary_style = _primary_style_pack(candidates or sorted_cards)
    concept_palette = {concept.concept_id: concept.color for concept in concepts}
    style_bible = StyleBible(
        primary_style_pack=primary_style,
        motion_language="contextual_continuity" if prefer_premium else "editorial_support",
        continuity_rules=[
            "Reuse concept colors and motifs across repeated ideas.",
            "Treat each full-screen visual as an episode with an entry and exit transition.",
            "Prefer visual sequences over isolated title-card slides when adjacent cards share a concept.",
            "Keep source-video rhythm visible through timing, pauses, and scene-cut alignment.",
        ],
        density_target=density_target,
        recurring_motifs=list(dict.fromkeys(episode.motif for episode in episodes))[:5],
        concept_palette=concept_palette,
    )
    summary_seed = " -> ".join(chapter.summary for chapter in chapters[:4] if chapter.summary)
    return VisualNarrativeProgram(
        program_id="visual_program_v1",
        duration_sec=round(clip_duration, 2),
        summary=_truncate(summary_seed or "Transcript-driven visual narrative program", 220),
        density_target=density_target,
        style_bible=style_bible,
        chapters=chapters,
        concept_memory=concepts,
        episodes=episodes,
    )


def _program_episode_map(program: VisualNarrativeProgram | dict[str, Any]) -> dict[str, dict[str, Any]]:
    payload = program.to_dict() if isinstance(program, VisualNarrativeProgram) else dict(program or {})
    episode_by_card: dict[str, dict[str, Any]] = {}
    for episode in payload.get("episodes") or []:
        if not isinstance(episode, dict):
            continue
        for card_id in episode.get("card_ids") or []:
            if str(card_id).strip():
                episode_by_card[str(card_id).strip()] = episode
    return episode_by_card


def apply_visual_program_to_specs(
    specs: list[dict[str, Any]],
    program: VisualNarrativeProgram | dict[str, Any],
    *,
    style_pack: str = "auto",
    enable_hyperframes_expansion: bool = True,
    premium_renderer_hint: str = "hyperframes",
) -> list[dict[str, Any]]:
    payload = program.to_dict() if isinstance(program, VisualNarrativeProgram) else dict(program or {})
    style_bible = dict(payload.get("style_bible") or {})
    primary_style = str(style_bible.get("primary_style_pack") or "editorial_clean")
    concept_memory = {
        str(item.get("concept_id") or ""): item
        for item in (payload.get("concept_memory") or [])
        if isinstance(item, dict)
    }
    episode_by_card = _program_episode_map(payload)
    enriched: list[dict[str, Any]] = []
    for spec in specs:
        normalized = dict(spec)
        card_id = str(normalized.get("card_id") or "").strip()
        episode = dict(episode_by_card.get(card_id) or {})
        concept_ids = [str(item) for item in (episode.get("concept_ids") or []) if str(item).strip()]
        concepts = [concept_memory[concept_id] for concept_id in concept_ids if concept_id in concept_memory]
        if episode:
            normalized["episode_id"] = episode.get("episode_id")
            normalized["program_context"] = {
                "program_id": payload.get("program_id"),
                "summary": payload.get("summary"),
                "density_target": payload.get("density_target"),
                "style_bible": style_bible,
            }
            normalized["episode_context"] = {
                "chapter_id": episode.get("chapter_id"),
                "purpose": episode.get("purpose"),
                "continuity_group": episode.get("continuity_group"),
                "density_role": episode.get("density_role"),
                "motif": episode.get("motif"),
                "template_family": episode.get("template_family"),
                "concepts": concepts,
            }
            normalized["visual_beats"] = list(episode.get("beats") or [])
            normalized["concept_ids"] = concept_ids
            normalized["continuity_group"] = episode.get("continuity_group")
            normalized["transition_in"] = dict(episode.get("transition_in") or {})
            normalized["transition_out"] = dict(episode.get("transition_out") or {})
            normalized["qa_contract"] = dict(episode.get("qa_contract") or {})
            normalized["background_motif"] = str(episode.get("motif") or normalized.get("background_motif") or "constellation")
            family = str(episode.get("template_family") or "").strip()
            expansion_templates = {
                "causal_chain",
                "flywheel_loop",
                "anatomy_cutaway",
                "stack_ranking",
                "decision_matrix",
                "contrast_ladder",
                "proof_sequence",
                "narrative_arc",
                "concept_map",
                "problem_solution",
                "myth_buster",
                "checklist_reveal",
                "risk_radar",
                "opportunity_map",
                "scorecard",
                "pipeline_xray",
                "decision_tree",
                "momentum_wave",
                "focus_ring",
                "timeline_filmstrip",
                "quote_breakdown",
                "market_map",
                "mechanism_blueprint",
                "data_pulse",
            }
            can_apply_family = enable_hyperframes_expansion or family not in expansion_templates
            if can_apply_family and family and str(normalized.get("template") or "") in {"ribbon_quote", "keyword_stack", "quote_focus"}:
                normalized["template"] = family
            if enable_hyperframes_expansion and family in expansion_templates:
                normalized["renderer_hint"] = (
                    premium_renderer_hint
                    if premium_renderer_hint in {"hyperframes", "remotion"}
                    else "hyperframes"
                )
            beats = [beat for beat in (episode.get("beats") or []) if isinstance(beat, dict)]
            if beats and normalized.get("template") in {
                "causal_chain",
                "flywheel_loop",
                "kinetic_route",
                "signal_network",
                "stack_ranking",
                "concept_map",
                "checklist_reveal",
                "pipeline_xray",
                "decision_tree",
                "timeline_filmstrip",
                "mechanism_blueprint",
                "market_map",
            }:
                beat_steps = [_clean_label(beat.get("text"), max_chars=28) for beat in beats]
                beat_steps = [step for step in beat_steps if step]
                if len(beat_steps) >= 2:
                    normalized["steps"] = beat_steps[:4]
            if style_pack in {"", "auto"}:
                normalized["style_pack"] = primary_style
        elif style_pack in {"", "auto"} and not normalized.get("style_pack"):
            normalized["style_pack"] = primary_style
        enriched.append(normalized)
    return enriched


def visual_program_prompt_block(program: VisualNarrativeProgram | dict[str, Any]) -> str:
    payload = program.to_dict() if isinstance(program, VisualNarrativeProgram) else dict(program or {})
    style_bible = dict(payload.get("style_bible") or {})
    episode_lines = []
    for episode in (payload.get("episodes") or [])[:24]:
        if not isinstance(episode, dict):
            continue
        beats = [
            str(beat.get("text") or "")
            for beat in (episode.get("beats") or [])
            if isinstance(beat, dict) and str(beat.get("text") or "").strip()
        ]
        episode_lines.append(
            (
                f"- {episode.get('episode_id')} {episode.get('start')}-{episode.get('end')}s "
                f"cards={', '.join(str(item) for item in (episode.get('card_ids') or []))} "
                f"family={episode.get('template_family')} group={episode.get('continuity_group')} "
                f"purpose={_truncate(str(episode.get('purpose') or ''), 120)} "
                f"beats={' | '.join(beats[:3])}"
            )
        )
    chapter_lines = [
        f"- {chapter.get('chapter_id')} {chapter.get('start')}-{chapter.get('end')}s: {chapter.get('summary')} density={chapter.get('visual_density')}"
        for chapter in (payload.get("chapters") or [])[:24]
        if isinstance(chapter, dict)
    ]
    return "\n".join(
        [
            f"Visual Narrative Program: {payload.get('summary', '')}",
            f"Density target: {payload.get('density_target', '')}",
            f"Primary style: {style_bible.get('primary_style_pack', '')}",
            f"Motion language: {style_bible.get('motion_language', '')}",
            "Chapters:",
            *chapter_lines,
            "Preferred visual episodes:",
            *episode_lines,
            "Continuity rules:",
            *[f"- {rule}" for rule in (style_bible.get("continuity_rules") or [])[:5]],
        ]
    )
