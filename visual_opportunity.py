from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from visual_copy_contract import (
    display_copy_issues,
    metric_value_is_visual_measure,
    normalize_display_copy,
)
from visual_explanation import visual_explanation_ir_signature
from vex_hyperframes.compiler import compile_hyperframes_plan


VISUAL_OPPORTUNITY_PLAN_VERSION = "visual-opportunity-plan-v1"
MAX_EPISODE_CARDS = 12
MAX_EPISODE_DURATION_SEC = 48.0
MAX_WINDOW_CARDS = 6
MAX_WINDOW_DURATION_SEC = 26.0
MIN_OPPORTUNITY_SCORE = 0.64
MIN_ASSISTIVE_OPPORTUNITY_SCORE = 0.5
MIN_ASSISTIVE_WINDOW_SIGNAL = 0.34
MAX_ASSISTIVE_GENERIC_PENALTY = 0.78
MIN_SELECTED_SPACING_SEC = 6.0

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "how",
    "i",
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
    "we",
    "what",
    "when",
    "which",
    "with",
    "you",
}
_TOPIC_BOUNDARY_PATTERN = re.compile(
    r"^(?:okay[, ]+)?(?:before we|let'?s move|moving on|the next topic|"
    r"now let'?s|now the|okay now)\b",
    flags=re.IGNORECASE,
)
_ACTION_PATTERN = re.compile(
    r"\b(?:adds?|applies?|attends?|builds?|calls?|checks?|chooses?|classifies?|"
    r"compares?|compresses?|connects?|converts?|creates?|decides?|enters?|"
    r"filters?|finds?|generates?|groups?|guesses?|highlights?|invokes?|keeps?|"
    r"handles?|hands?|learns?|links?|loads?|maps?|mak(?:e|es)|merg(?:e|es)|mixes?|"
    r"opens?|optimiz(?:e|es)|passes?|picks?|predicts?|queries?|reads?|"
    r"reach(?:es)?|reduces?|removes?|renders?|replaces?|returns?|routes?|runs?|"
    r"sampl(?:e|es)|scores?|selects?|sends?|stores?|summariz(?:e|es|ed)|trains?|"
    r"transforms?|turns?|updates?|validates?|writes?|becomes?)\b",
    flags=re.IGNORECASE,
)
_PROCESS_PATTERN = re.compile(
    r"\b(?:first|then|next|after|before|finally|again|pipeline|stage|step|"
    r"adds?|applies?|builds?|checks?|chooses?|compresses?|connects?|converts?|"
    r"creates?|filters?|generates?|groups?|links?|maps?|reads?|renders?|routes?|"
    r"runs?|scores?|selects?|summarizes?|turns?|updates?|validates?|writes?)\b",
    flags=re.IGNORECASE,
)
_ASSISTIVE_STRUCTURE_PATTERN = re.compile(
    r"\b(?:architecture|budget|cache|chain|component|context|decision|flow|graph|"
    r"input|layer|map|memory|model|output|path|pipeline|policy|pressure|process|"
    r"queue|retrieval|router|signal|state|system|token|trade[-\s]?off|workflow)\b",
    flags=re.IGNORECASE,
)
_CAUSE_PATTERN = re.compile(
    r"\b(?:because|therefore|causes?|forces?|leads?\s+to|results?\s+in)\b",
    flags=re.IGNORECASE,
)
_FRAGMENT_END_PATTERN = re.compile(
    r"\b(?:a|an|and|at|but|by|compared|for|from|in|into|of|on|or|per|than|the|to|with)\s*$",
    flags=re.IGNORECASE,
)
_LOW_SIGNAL_PATTERN = re.compile(
    r"\b(?:caught me off guard|genuinely|interesting|thought of sharing|"
    r"marketing view|specific thing|talk about|going to explain)\b",
    flags=re.IGNORECASE,
)
def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clean_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _truncate(value: Any, limit: int) -> str:
    text = _clean_space(value)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip(" ,;:-") + "..."


def _tokens(value: Any) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z0-9']+", str(value or "").lower())
        if len(token) >= 3 and token not in _STOPWORDS
    ]


def _content_terms(value: Any, *, limit: int = 10) -> list[str]:
    counts: dict[str, int] = {}
    for token in _tokens(value):
        counts[token] = counts.get(token, 0) + 1
    return [
        token
        for token, _ in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[:limit]
    ]


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(len(left | right), 1)


def _card_text(card: dict[str, Any]) -> str:
    return _clean_space(
        card.get("sentence_text")
        or card.get("source_sentence_text")
        or card.get("context_text")
    )


def _card_start(card: dict[str, Any]) -> float:
    return _as_float(card.get("start"), 0.0)


def _card_end(card: dict[str, Any]) -> float:
    return max(_as_float(card.get("end"), _card_start(card)), _card_start(card))


def _card_id(card: dict[str, Any]) -> str:
    return _clean_space(card.get("card_id"))


@dataclass(frozen=True)
class SemanticEpisode:
    episode_id: str
    start: float
    end: float
    card_ids: list[str]
    summary: str
    topic_terms: list[str]
    boundary_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OpportunityDecision:
    opportunity_id: str
    episode_id: str
    source_card_ids: list[str]
    start: float
    end: float
    score: float
    status: str
    reason: str
    scene_type: str
    semantic_signature: str
    card: dict[str, Any] = field(default_factory=dict)
    preflight: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VisualOpportunityPlan:
    version: str
    duration_sec: float
    requested_count: int
    recommended_count: int
    episodes: list[SemanticEpisode]
    selected: list[OpportunityDecision]
    reserves: list[OpportunityDecision]
    rejected: list[OpportunityDecision]

    @property
    def selected_cards(self) -> list[dict[str, Any]]:
        return [dict(item.card) for item in self.selected]

    @property
    def reserve_cards(self) -> list[dict[str, Any]]:
        return [dict(item.card) for item in self.reserves]

    def to_dict(self) -> dict[str, Any]:
        rejection_counts: dict[str, int] = {}
        for item in self.rejected:
            rejection_counts[item.reason] = rejection_counts.get(item.reason, 0) + 1
        return {
            "version": self.version,
            "duration_sec": round(self.duration_sec, 3),
            "requested_count": self.requested_count,
            "recommended_count": self.recommended_count,
            "episode_count": len(self.episodes),
            "selected_count": len(self.selected),
            "reserve_count": len(self.reserves),
            "rejected_count": len(self.rejected),
            "episodes": [item.to_dict() for item in self.episodes],
            "selected": [item.to_dict() for item in self.selected],
            "reserves": [item.to_dict() for item in self.reserves],
            "rejection_counts": rejection_counts,
            "rejected": [item.to_dict() for item in self.rejected[:80]],
        }


def build_semantic_episodes(
    cards: list[dict[str, Any]],
    *,
    clip_duration: float,
) -> list[SemanticEpisode]:
    ordered = sorted(
        (dict(card) for card in cards if _card_id(card)),
        key=_card_start,
    )
    if not ordered:
        return []
    grouped: list[tuple[list[dict[str, Any]], str]] = []
    current: list[dict[str, Any]] = []
    current_terms: set[str] = set()
    next_boundary_reason = "video_start"
    for card in ordered:
        text = _card_text(card)
        terms = set(_tokens(f"{text} {card.get('context_text', '')}"))
        boundary_reason = ""
        if current:
            previous = current[-1]
            gap = max(0.0, _card_start(card) - _card_end(previous))
            span = _card_end(card) - _card_start(current[0])
            overlap = _jaccard(current_terms, terms)
            if gap >= 6.0:
                boundary_reason = "subtitle_gap"
            elif len(current) >= MAX_EPISODE_CARDS:
                boundary_reason = "episode_card_limit"
            elif span >= MAX_EPISODE_DURATION_SEC:
                boundary_reason = "episode_duration_limit"
            elif len(current) >= 2 and _TOPIC_BOUNDARY_PATTERN.search(text):
                boundary_reason = "discourse_transition"
            elif len(current) >= 6 and gap >= 0.7 and overlap < 0.035:
                boundary_reason = "topic_shift"
        if boundary_reason:
            grouped.append((current, next_boundary_reason))
            current = []
            current_terms = set()
            next_boundary_reason = boundary_reason
        current.append(card)
        current_terms.update(terms)
    if current:
        grouped.append((current, next_boundary_reason))

    episodes: list[SemanticEpisode] = []
    for index, (episode_cards, boundary_reason) in enumerate(grouped, start=1):
        text = " ".join(_card_text(card) for card in episode_cards)
        episodes.append(
            SemanticEpisode(
                episode_id=f"semantic_episode_{index:03d}",
                start=round(_card_start(episode_cards[0]), 3),
                end=round(min(_card_end(episode_cards[-1]), clip_duration), 3),
                card_ids=[_card_id(card) for card in episode_cards],
                summary=_truncate(text, 180),
                topic_terms=_content_terms(text, limit=10),
                boundary_reason=boundary_reason,
            )
        )
    return episodes


def _window_source_text(cards: list[dict[str, Any]]) -> str:
    return _clean_space(" ".join(_card_text(card) for card in cards))


def _is_complete_label(value: str) -> bool:
    text = _clean_space(value)
    if not text or len(_tokens(text)) < 2:
        return False
    return not _FRAGMENT_END_PATTERN.search(text) and not display_copy_issues(
        text,
        role="label",
    )


def _metric_facts(cards: list[dict[str, Any]], source_text: str) -> list[dict[str, str]]:
    facts: list[dict[str, str]] = []
    seen: set[str] = set()
    for card in cards:
        for item in card.get("metric_facts") or []:
            if not isinstance(item, dict):
                continue
            value = _clean_space(item.get("value"))
            label = _clean_space(item.get("label") or value)
            key = value.lower()
            if not value or key in seen or value.replace(" ", "") not in source_text.replace(" ", ""):
                continue
            if not _is_complete_label(label):
                continue
            if not metric_value_is_visual_measure(value, label, source_text):
                continue
            facts.append({"value": value, "label": label})
            seen.add(key)
            if len(facts) >= 4:
                return facts
    return facts


def _action_clauses(source_text: str) -> list[str]:
    clauses = _action_segments(source_text)
    actions: list[str] = []
    seen: set[str] = set()
    for clause in clauses:
        cleaned = normalize_display_copy(clause)
        if not _ACTION_PATTERN.search(cleaned):
            continue
        if not (3 <= len(_tokens(cleaned)) <= 14):
            continue
        label = cleaned
        key = label.lower()
        if key in seen or not _is_complete_label(label):
            continue
        seen.add(key)
        actions.append(label)
        if len(actions) >= 5:
            break
    return actions


def _action_segments(source_text: str) -> list[str]:
    coarse = re.split(
        r"(?<=[.!?])\s+|[;]\s*|\b(?:and\s+then|so\s+then|then)\b",
        source_text,
        flags=re.IGNORECASE,
    )
    segments: list[str] = []
    for clause in coarse:
        text = _clean_space(clause)
        matches = list(_ACTION_PATTERN.finditer(text))
        if len(matches) < 2:
            segments.append(text)
            continue
        starts = [0]
        for match in matches[1:]:
            prefix = text[starts[-1] : match.start()].strip(" ,")
            if len(_tokens(prefix)) >= 2:
                starts.append(match.start())
        starts.append(len(text))
        for index in range(len(starts) - 1):
            segment = text[starts[index] : starts[index + 1]].strip(" ,")
            if segment:
                segments.append(segment)
    return segments


def _source_clauses(source_text: str) -> list[str]:
    clauses = re.split(
        r"(?<=[.!?])\s+|[;]\s*|\b(?:and then|then|so then)\b",
        source_text,
        flags=re.IGNORECASE,
    )
    cleaned_clauses: list[str] = []
    seen: set[str] = set()
    for clause in clauses:
        cleaned = normalize_display_copy(clause)
        if not (3 <= len(_tokens(cleaned)) <= 16):
            continue
        if not _is_complete_label(cleaned):
            continue
        label = cleaned
        key = label.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned_clauses.append(label)
        if len(cleaned_clauses) >= 5:
            break
    return cleaned_clauses


def _concept_phrases(source_text: str) -> list[str]:
    text = _clean_space(source_text)
    text = re.sub(
        r"^(?:the\s+)?(?:important\s+)?(?:idea|map|point|topic|part|thing)\s+"
        r"(?:is|means|shows|connects|links|maps)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    fragments = re.split(
        r",|\b(?:and|or|versus|vs\.?|while|between)\b",
        text,
        flags=re.IGNORECASE,
    )
    concepts: list[str] = []
    seen: set[str] = set()
    for fragment in fragments:
        cleaned = _clean_space(fragment).strip(" .,:;-")
        cleaned = re.sub(
            r"^(?:the|a|an|this|that|same|one|two|three|four)\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        token_count = len(_tokens(cleaned))
        if not (2 <= token_count <= 7):
            continue
        if _normalize_generic_label(cleaned):
            continue
        label = cleaned
        key = label.lower()
        if key in seen:
            continue
        seen.add(key)
        concepts.append(label)
        if len(concepts) >= 5:
            break
    return concepts


def _normalize_generic_label(value: str) -> bool:
    normalized = _clean_space(value).lower()
    return normalized in {
        "core idea",
        "main point",
        "specific thing",
        "the thing",
        "this thing",
        "something",
        "the concept",
    }


def _grounded_semantic_frame(
    cards: list[dict[str, Any]],
    source_text: str,
    metric_facts: list[dict[str, str]],
) -> dict[str, Any]:
    frame: dict[str, Any] = {}
    action_clauses = _action_clauses(source_text)
    if len(action_clauses) >= 2 and _PROCESS_PATTERN.search(source_text):
        frame["steps"] = (
            action_clauses[:-1]
            if len(action_clauses) >= 3
            else action_clauses
        )
        frame["input"] = action_clauses[0]
        frame["result"] = action_clauses[-1]
        frame["viewer_takeaway"] = action_clauses[-1]

    explicit_transform = re.search(
        r"\b(?:converts?|transforms?|compresses?)\s+(.{3,72}?)\s+(?:into|to)\s+(.{3,72}?)(?:[.,;]|$)",
        source_text,
        flags=re.IGNORECASE,
    )
    if explicit_transform:
        before = _clean_space(explicit_transform.group(1)).strip(" ,.;:-")
        after = _clean_space(explicit_transform.group(2)).strip(" ,.;:-")
        if _is_complete_label(before) and _is_complete_label(after):
            frame["before_state"] = _truncate(before, 72)
            frame["after_state"] = _truncate(after, 72)
            frame["viewer_takeaway"] = _truncate(after, 72)

    if _CAUSE_PATTERN.search(source_text) and len(action_clauses) >= 3:
        frame.setdefault("problem", action_clauses[0])
        frame.setdefault("mechanism", action_clauses[1])
        frame.setdefault("result", action_clauses[-1])

    for card in cards:
        semantic = card.get("semantic_frame")
        if not isinstance(semantic, dict):
            continue
        for key in ("decision", "low_branch", "high_branch", "screen", "focus", "action"):
            value = _clean_space(semantic.get(key))
            if (
                value
                and _is_complete_label(value)
                and value.lower() in source_text.lower()
            ):
                frame.setdefault(key, _truncate(value, 72))
    if "steps" not in frame:
        clause_steps = _source_clauses(source_text)
        if len(clause_steps) >= 2 and (
            _PROCESS_PATTERN.search(source_text)
            or _ASSISTIVE_STRUCTURE_PATTERN.search(source_text)
            or any(_ACTION_PATTERN.search(step) for step in clause_steps)
        ):
            frame["steps"] = clause_steps[:4]
            frame.setdefault("input", clause_steps[0])
            frame.setdefault("result", clause_steps[-1])
            frame.setdefault("viewer_takeaway", clause_steps[-1])
    if "steps" not in frame:
        concepts = _concept_phrases(source_text)
        if len(concepts) >= 2 and _ASSISTIVE_STRUCTURE_PATTERN.search(source_text):
            frame["steps"] = concepts[:4]
            frame.setdefault("input", concepts[0])
            frame.setdefault("result", concepts[-1])
            frame.setdefault("viewer_takeaway", concepts[-1])
    if metric_facts:
        frame.setdefault(
            "viewer_takeaway",
            _truncate(metric_facts[0].get("label") or metric_facts[0].get("value"), 72),
        )
    return frame


def _window_signal_score(cards: list[dict[str, Any]], source_text: str) -> float:
    priorities = [_as_float(card.get("priority"), 0.0) for card in cards]
    payoffs = [_as_float(card.get("intuition_payoff"), 0.0) for card in cards]
    visualizability = [_as_float(card.get("visualizability"), 0.0) for card in cards]
    generic = [_as_float(card.get("generic_penalty"), 0.0) for card in cards]
    source_need = [
        _as_float((card.get("source_frame_analysis") or {}).get("visual_need"), 0.5)
        for card in cards
    ]
    graph_opportunity = [
        _as_float((card.get("creative_graph_signals") or {}).get("graph_visual_opportunity"), 0.4)
        for card in cards
    ]
    explicit = (
        0.11
        if any(
            int(_as_float(card.get("numeric_hits"), 0.0)) > 0
            or _as_float(card.get("process_cues"), 0.0) >= 0.22
            or _as_float(card.get("contrast_cues"), 0.0) >= 0.22
            for card in cards
        )
        else 0.0
    )
    multi_sentence = 0.08 if len(cards) >= 2 else 0.0
    low_signal = 0.14 if _LOW_SIGNAL_PATTERN.search(source_text) else 0.0
    return max(
        0.0,
        min(
            1.0,
            (max(priorities, default=0.0) / 100.0) * 0.28
            + max(payoffs, default=0.0) * 0.18
            + max(visualizability, default=0.0) * 0.16
            + max(source_need, default=0.5) * 0.1
            + max(graph_opportunity, default=0.4) * 0.1
            + explicit
            + multi_sentence
            - max(generic, default=0.0) * 0.12
            - low_signal,
        ),
    )


def _has_explicit_visual_signal(cards: list[dict[str, Any]], source_text: str) -> bool:
    return bool(
        _PROCESS_PATTERN.search(source_text)
        or _CAUSE_PATTERN.search(source_text)
        or _ASSISTIVE_STRUCTURE_PATTERN.search(source_text)
        or any(
            int(_as_float(card.get("numeric_hits"), 0.0)) > 0
            or _as_float(card.get("process_cues"), 0.0) >= 0.18
            or _as_float(card.get("contrast_cues"), 0.0) >= 0.18
            or bool(card.get("metric_facts"))
            for card in cards
        )
    )


def _assistive_window_signal(cards: list[dict[str, Any]], source_text: str) -> float:
    tokens = _tokens(source_text)
    content_terms = _content_terms(source_text, limit=12)
    if len(tokens) < 5 or len(content_terms) < 3:
        return 0.0
    priorities = [_as_float(card.get("priority"), 0.0) for card in cards]
    payoffs = [_as_float(card.get("intuition_payoff"), 0.0) for card in cards]
    visualizability = [_as_float(card.get("visualizability"), 0.0) for card in cards]
    generic = [_as_float(card.get("generic_penalty"), 0.0) for card in cards]
    source_need = [
        _as_float((card.get("source_frame_analysis") or {}).get("visual_need"), 0.5)
        for card in cards
    ]
    graph_opportunity = [
        _as_float((card.get("creative_graph_signals") or {}).get("graph_visual_opportunity"), 0.4)
        for card in cards
    ]
    explicit = _has_explicit_visual_signal(cards, source_text)
    if max(generic, default=0.0) > MAX_ASSISTIVE_GENERIC_PENALTY and not explicit:
        return 0.0
    low_signal_penalty = 0.12 if _LOW_SIGNAL_PATTERN.search(source_text) and not explicit else 0.0
    multi_clause_bonus = 0.06 if len(_source_clauses(source_text)) >= 2 else 0.0
    concept_bonus = 0.05 if len(_concept_phrases(source_text)) >= 2 else 0.0
    return max(
        0.0,
        min(
            1.0,
            (max(priorities, default=0.0) / 100.0) * 0.2
            + max(payoffs, default=0.0) * 0.18
            + max(visualizability, default=0.0) * 0.18
            + max(source_need, default=0.5) * 0.16
            + max(graph_opportunity, default=0.4) * 0.12
            + (0.12 if explicit else 0.0)
            + (0.05 if len(cards) >= 2 else 0.0)
            + multi_clause_bonus
            + concept_bonus
            - max(generic, default=0.0) * 0.14
            - low_signal_penalty,
        ),
    )


def _allow_assistive_opportunity(
    card: dict[str, Any],
    cards: list[dict[str, Any]],
    *,
    source_text: str,
    signal_score: float,
    structural_strength: float,
    raw_preflight_passed: bool,
    strict_preflight_passed: bool,
    rejection_reason: str,
) -> tuple[bool, float]:
    assistive_signal = _assistive_window_signal(cards, source_text)
    content_terms = _content_terms(source_text, limit=12)
    explicit = _has_explicit_visual_signal(cards, source_text)
    generic_penalty = max(_as_float(item.get("generic_penalty"), 0.0) for item in cards)
    if len(content_terms) < 3:
        return False, 0.0
    if generic_penalty > MAX_ASSISTIVE_GENERIC_PENALTY and not explicit:
        return False, 0.0
    if _LOW_SIGNAL_PATTERN.search(source_text) and not explicit:
        return False, 0.0
    if rejection_reason in {
        "numeric_fact_lacks_source_provenance",
        "required_label_lacks_source_provenance",
        "fragmented_semantic_labels",
    }:
        return False, 0.0
    semantic_frame = card.get("semantic_frame") if isinstance(card.get("semantic_frame"), dict) else {}
    has_grounded_frame = bool(
        semantic_frame.get("steps")
        or semantic_frame.get("before_state")
        or semantic_frame.get("after_state")
        or semantic_frame.get("viewer_takeaway")
        or card.get("metric_facts")
    )
    if not has_grounded_frame and not explicit:
        return False, 0.0
    preflight_bonus = 0.08 if raw_preflight_passed else 0.0
    structural_floor = 0.06 if strict_preflight_passed else min(structural_strength, 0.18) * 0.18
    assistive_score = min(
        1.0,
        max(signal_score, assistive_signal) * 0.72
        + structural_floor
        + preflight_bonus,
    )
    return assistive_score >= MIN_ASSISTIVE_OPPORTUNITY_SCORE, assistive_score


def _opportunity_id(episode_id: str, source_card_ids: list[str]) -> str:
    seed = f"{episode_id}:{','.join(source_card_ids)}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]
    return f"visual_opportunity_{digest}"


def _episode_display_title(episode: SemanticEpisode) -> str:
    summary = _clean_space(episode.summary)
    definition = re.search(
        r"\b(?:also\s+known\s+as|known\s+as|called)\s+([^.!?]+)",
        summary,
        flags=re.IGNORECASE,
    )
    candidate = definition.group(1) if definition else re.split(r"[.!?]", summary, maxsplit=1)[0]
    candidate = _clean_space(candidate).strip(" ,.;:-")
    if not (2 <= len(_tokens(candidate)) <= 7) or len(candidate) > 64:
        return ""
    return candidate.title()


def _display_title_is_local(display_title: str, source_text: str) -> bool:
    candidate = normalize_display_copy(display_title)
    if display_copy_issues(candidate, role="title"):
        return False
    normalized_candidate = " ".join(re.findall(r"[a-z0-9]+", candidate.lower()))
    normalized_source = " ".join(re.findall(r"[a-z0-9]+", source_text.lower()))
    return bool(normalized_candidate and normalized_candidate in normalized_source)


def _opportunity_card(
    episode: SemanticEpisode,
    cards: list[dict[str, Any]],
) -> dict[str, Any]:
    source_text = _window_source_text(cards)
    source_card_ids = [_card_id(card) for card in cards]
    opportunity_id = _opportunity_id(episode.episode_id, source_card_ids)
    metric_facts = _metric_facts(cards, source_text)
    semantic_frame = _grounded_semantic_frame(cards, source_text, metric_facts)
    if len(semantic_frame.get("steps") or []) >= 2:
        metric_facts = []
    first = cards[0]
    last = cards[-1]
    planning_context_text = _clean_space(
        " ".join(
            part
            for part in [
                first.get("previous_text"),
                source_text,
                last.get("next_text"),
            ]
            if _clean_space(part)
        )
    )
    signal_score = _window_signal_score(cards, source_text)
    keywords: list[str] = []
    for card in cards:
        for keyword in card.get("keywords") or []:
            text = _clean_space(keyword)
            if text and text.lower() not in {item.lower() for item in keywords}:
                keywords.append(text)
    return {
        **dict(first),
        "card_id": opportunity_id,
        "start": round(_card_start(first), 3),
        "end": round(_card_end(last), 3),
        "sentence_text": _truncate(source_text, 420),
        "source_sentence_text": _truncate(source_text, 420),
        "context_text": _truncate(source_text, 520),
        "planning_context_text": _truncate(planning_context_text, 700),
        "previous_text": _truncate(first.get("previous_text"), 180),
        "next_text": _truncate(last.get("next_text"), 180),
        "semantic_frame": semantic_frame,
        "metric_facts": metric_facts,
        "keywords": keywords[:8] or _content_terms(source_text, limit=8),
        "numeric_hits": sum(int(_as_float(card.get("numeric_hits"), 0.0)) for card in cards),
        "sentence_numeric_hits": sum(
            int(_as_float(card.get("sentence_numeric_hits"), 0.0))
            for card in cards
        ),
        "process_cues": max(
            [_as_float(card.get("process_cues"), 0.0) for card in cards]
            + ([0.7] if _PROCESS_PATTERN.search(source_text) else [0.0])
        ),
        "sentence_process_cues": max(
            _as_float(card.get("sentence_process_cues"), 0.0) for card in cards
        ),
        "contrast_cues": max(
            _as_float(card.get("contrast_cues"), 0.0) for card in cards
        ),
        "sentence_contrast_cues": max(
            _as_float(card.get("sentence_contrast_cues"), 0.0) for card in cards
        ),
        "visualizability": round(
            max(
                signal_score,
                max(_as_float(card.get("visualizability"), 0.0) for card in cards),
            ),
            4,
        ),
        "intuition_payoff": round(
            max(
                signal_score,
                max(_as_float(card.get("intuition_payoff"), 0.0) for card in cards),
            ),
            4,
        ),
        "intuition_role": "core_mechanism",
        "priority": round(signal_score * 100.0, 3),
        "novelty_key": opportunity_id,
        "suggested_composition": "replace",
        "suggested_renderer": "hyperframes",
        "style_pack": first.get("style_pack") or "signal_lab",
        "semantic_episode_id": episode.episode_id,
        "semantic_episode_summary": episode.summary,
        "source_card_ids": source_card_ids,
        "opportunity_contract": {
            "version": VISUAL_OPPORTUNITY_PLAN_VERSION,
            "opportunity_id": opportunity_id,
            "episode_id": episode.episode_id,
            "source_card_ids": source_card_ids,
            "evidence_start": round(_card_start(first), 3),
            "evidence_end": round(_card_end(last), 3),
            "evidence_is_window_local": True,
            "why_visual": (
                "The subtitle window contains a source-grounded mechanism, transformation, "
                "process, decision, or quantitative relationship that can be inspected visually."
            ),
            "signal_score": round(signal_score, 4),
        },
    }


def _candidate_windows(
    episode: SemanticEpisode,
    episode_cards: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    candidates: list[tuple[float, list[dict[str, Any]]]] = []
    count = len(episode_cards)
    for start_index in range(count):
        for size in range(1, min(MAX_WINDOW_CARDS, count - start_index) + 1):
            window = episode_cards[start_index : start_index + size]
            duration = _card_end(window[-1]) - _card_start(window[0])
            if duration > MAX_WINDOW_DURATION_SEC:
                break
            source_text = _window_source_text(window)
            signal = _window_signal_score(window, source_text)
            has_structure = bool(
                _PROCESS_PATTERN.search(source_text)
                or _CAUSE_PATTERN.search(source_text)
                or any(card.get("metric_facts") for card in window)
                or re.search(
                    r"\b(?:from\b.+\bto|into|instead|otherwise|compared to|"
                    r"tokens?|cache|pipeline|layers?|blocks?|experts?)\b",
                    source_text,
                    flags=re.IGNORECASE,
                )
            )
            if not has_structure or signal < 0.42:
                assistive_signal = _assistive_window_signal(window, source_text)
                if assistive_signal < MIN_ASSISTIVE_WINDOW_SIGNAL:
                    continue
                signal = max(signal, assistive_signal)
            candidates.append((signal, window))
    candidates.sort(
        key=lambda item: (
            item[0],
            len(item[1]),
            -_card_start(item[1][0]),
        ),
        reverse=True,
    )
    unique: list[list[dict[str, Any]]] = []
    seen: set[tuple[str, ...]] = set()
    for _, window in candidates:
        key = tuple(_card_id(card) for card in window)
        if key in seen:
            continue
        seen.add(key)
        unique.append(window)
        if len(unique) >= 24:
            break
    return unique


def _preflight_strength(result: Any) -> tuple[bool, float, str]:
    if not result.passed:
        reason = str((result.issues or result.ir.rejection_reasons or ["semantic_preflight_rejected"])[0])
        return False, 0.0, reason
    ir = result.ir
    roles = {item.role for item in ir.objects}
    scene_type = str(ir.scene_type or "")
    minimum_objects = {
        "causal_intervention": 3,
        "guided_process": 3,
        "architecture_flow": 3,
        "decision_branch": 3,
        "narrative_progression": 3,
        "matched_state_transform": 2,
        "metric_delta": 2,
        "metric_intervention": 3,
        "metric_proof": 1,
        "set_partition": 3,
        "grounded_interface_walkthrough": 3,
    }.get(scene_type, 2)
    if len(ir.objects) < minimum_objects:
        return False, 0.0, "preflight_insufficient_explanatory_objects"
    if scene_type == "causal_intervention" and not (
        "result" in roles
        and "problem" in roles
        and ({"mechanism", "intervention"} & roles)
    ):
        return False, 0.0, "preflight_incomplete_causal_model"
    if scene_type in {"guided_process", "architecture_flow"} and "mechanism" not in roles:
        return False, 0.0, "preflight_missing_process_stages"
    if scene_type == "evidence_backed_quote":
        return False, 0.0, "preflight_quote_not_explanatory"
    relation_strength = min(len(ir.relations) / 3.0, 1.0)
    object_strength = min(len(ir.objects) / 4.0, 1.0)
    beat_strength = min(len(ir.beats) / 3.0, 1.0)
    scene_bonus = 0.12 if scene_type in {"set_partition", "architecture_flow", "guided_process"} else 0.06
    return (
        True,
        min(
            1.0,
            0.35 * object_strength
            + 0.3 * relation_strength
            + 0.23 * beat_strength
            + scene_bonus,
        ),
        "",
    )


def _metric_relation_quality(
    source_text: str,
    metric_facts: list[dict[str, Any]],
) -> tuple[bool, float]:
    value_count = len(
        {
            _clean_space(item.get("value")).lower()
            for item in metric_facts
            if _clean_space(item.get("value"))
        }
    )
    lowered = source_text.lower()
    single_baseline = bool(
        re.search(
            r"\b(?:only|requires?|at\s+the\s+cost\s+of|"
            r"work\s+is\s+eliminated)\b",
            lowered,
        )
    )
    explicit_comparison = bool(
        re.search(r"\b(?:compared\s+to|versus|vs\.?|rating\s+of)\b", lowered)
    )
    explicit_transition = bool(
        re.search(
            r"\b(?:falls?|drops?|rises?|increases?|decreases?)\s+from\b",
            lowered,
        )
    )
    ratio_replacement = bool(
        re.search(
            r"\binstead\s+of\b[^.!?]{0,80}\d+[^.!?]{0,60}\d+",
            lowered,
        )
    )
    measured_difference = bool(
        re.search(r"\b(?:difference|gap)\b", lowered)
        and value_count >= 3
    )
    passed = (
        (single_baseline and value_count >= 1)
        or (explicit_comparison and value_count >= 2)
        or (explicit_transition and value_count >= 2)
        or (ratio_replacement and value_count >= 2)
        or measured_difference
    )
    strength = min(0.14, 0.06 + max(0, value_count - 1) * 0.025) if passed else 0.0
    return passed, strength


def _decision_for_window(
    episode: SemanticEpisode,
    cards: list[dict[str, Any]],
) -> OpportunityDecision:
    card = _opportunity_card(episode, cards)
    opportunity_id = str(card["card_id"])
    display_title = _episode_display_title(episode)
    if not _display_title_is_local(display_title, str(card.get("sentence_text") or "")):
        display_title = ""
    if display_title:
        card["display_title"] = display_title
    preflight_spec = {
        "visual_id": opportunity_id,
        "card_id": opportunity_id,
        "sentence_text": card["sentence_text"],
        "context_text": card["context_text"],
        "semantic_frame": card["semantic_frame"],
        "metric_facts": card["metric_facts"],
        "visual_type_hint": card.get("visual_type_hint") or "",
        "duration": max(2.8, min(_card_end(cards[-1]) - _card_start(cards[0]), 7.0)),
        "composition_mode": "replace",
        "display_title": display_title,
    }
    result = compile_hyperframes_plan(preflight_spec)
    passed, structural_strength, rejection_reason = _preflight_strength(result)
    if (
        passed
        and result.ir.scene_type in {"metric_proof", "metric_delta"}
        and (
            len(cards) > 4
            or (_card_end(cards[-1]) - _card_start(cards[0])) > 16.0
        )
    ):
        passed = False
        rejection_reason = "preflight_metric_window_too_dense"
    metric_relation, metric_relation_bonus = _metric_relation_quality(
        card["sentence_text"],
        list(card.get("metric_facts") or []),
    )
    if (
        passed
        and result.ir.scene_type in {"metric_proof", "metric_delta"}
        and not metric_relation
    ):
        passed = False
        rejection_reason = "preflight_metric_lacks_explanatory_relation"
    signal_score = _as_float(
        (card.get("opportunity_contract") or {}).get("signal_score"),
        0.0,
    )
    if not (
        passed
        and result.ir.scene_type in {"metric_proof", "metric_delta"}
        and metric_relation
    ):
        metric_relation_bonus = 0.0
    score = min(
        1.0,
        signal_score * 0.58
        + structural_strength * 0.42
        + metric_relation_bonus
        + (0.06 if display_title else 0.0),
    )
    status = "candidate" if passed and score >= MIN_OPPORTUNITY_SCORE else "rejected"
    opportunity_tier = "primary" if status == "candidate" else "rejected"
    if status == "rejected":
        assistive_allowed, assistive_score = _allow_assistive_opportunity(
            card,
            cards,
            source_text=str(card.get("sentence_text") or ""),
            signal_score=signal_score,
            structural_strength=structural_strength,
            raw_preflight_passed=bool(result.passed),
            strict_preflight_passed=passed,
            rejection_reason=rejection_reason,
        )
        if assistive_allowed:
            status = "assistive_candidate"
            opportunity_tier = "assistive"
            if display_title:
                assistive_score = min(1.0, assistive_score + 0.04)
            score = max(score, assistive_score)
    reason = (
        ""
        if status == "candidate"
        else "assistive_source_grounded_visual"
        if status == "assistive_candidate"
        else rejection_reason
        or "opportunity_score_below_threshold"
    )
    semantic_signature = ""
    visual_ir = result.ir.to_dict()
    visual_ir_signature = visual_explanation_ir_signature(visual_ir)
    preflight: dict[str, Any] = {
        "passed": bool(result.passed),
        "issues": list(result.issues),
        "scene_type": result.ir.scene_type,
        "render_policy": result.ir.render_policy,
        "structural_strength": round(structural_strength, 4),
        "raw_preflight_passed": bool(result.passed),
        "strict_preflight_passed": bool(passed),
    }
    if result.production_contract is not None:
        semantic_signature = result.production_contract.semantic_signature
        preflight["semantic_signature"] = semantic_signature
        preflight["required_labels"] = list(result.production_contract.required_labels)
        preflight["required_relation_ids"] = list(
            result.production_contract.required_relation_ids
        )
    preflight["visual_explanation_ir_signature"] = visual_ir_signature
    contract = dict(card.get("opportunity_contract") or {})
    contract.update(
        {
            "score": round(score, 4),
            "scene_type": result.ir.scene_type,
            "semantic_signature": semantic_signature,
            "preflight_passed": bool(result.passed),
            "strict_preflight_passed": status == "candidate",
            "opportunity_tier": opportunity_tier,
            "visual_explanation_ir_signature": visual_ir_signature,
        }
    )
    card["opportunity_contract"] = contract
    card["opportunity_preflight"] = preflight
    card["visual_explanation_ir"] = visual_ir
    card["priority"] = round(score * 100.0, 3)
    card["visualizability"] = round(score, 4)
    card["intuition_payoff"] = round(score, 4)
    return OpportunityDecision(
        opportunity_id=opportunity_id,
        episode_id=episode.episode_id,
        source_card_ids=list(card.get("source_card_ids") or []),
        start=_as_float(card.get("start"), 0.0),
        end=_as_float(card.get("end"), 0.0),
        score=round(score, 4),
        status=status,
        reason=reason,
        scene_type=str(result.ir.scene_type or ""),
        semantic_signature=semantic_signature,
        card=card,
        preflight=preflight,
    )


def _overlaps(left: OpportunityDecision, right: OpportunityDecision) -> bool:
    return not (
        left.end <= right.start - 0.1
        or left.start >= right.end + 0.1
    )


def _schedule_opportunities(
    candidates: list[OpportunityDecision],
    *,
    requested_count: int,
    clip_duration: float,
) -> tuple[list[OpportunityDecision], list[OpportunityDecision]]:
    if not candidates or requested_count <= 0:
        return [], []
    duration_target = max(1, round(clip_duration / 48.0))
    target = min(requested_count, duration_target, len(candidates))
    by_episode: dict[str, list[OpportunityDecision]] = {}
    for candidate in candidates:
        by_episode.setdefault(candidate.episode_id, []).append(candidate)
    episode_heads = sorted(
        (
            max(items, key=lambda item: (item.score, -(item.end - item.start)))
            for items in by_episode.values()
        ),
        key=lambda item: (item.score, -item.start),
        reverse=True,
    )
    remaining = sorted(
        candidates,
        key=lambda item: (item.score, -(item.end - item.start), -item.start),
        reverse=True,
    )
    selected: list[OpportunityDecision] = []
    seen_signatures: set[str] = set()

    def add(candidate: OpportunityDecision) -> bool:
        if candidate.semantic_signature and candidate.semantic_signature in seen_signatures:
            return False
        if any(_overlaps(candidate, existing) for existing in selected):
            return False
        if any(
            abs(candidate.start - existing.start) < MIN_SELECTED_SPACING_SEC
            for existing in selected
        ):
            return False
        selected.append(candidate)
        if candidate.semantic_signature:
            seen_signatures.add(candidate.semantic_signature)
        return True

    for candidate in episode_heads:
        if len(selected) >= target:
            break
        add(candidate)
    for candidate in remaining:
        if len(selected) >= target:
            break
        add(candidate)
    selected.sort(key=lambda item: item.start)

    reserve_limit = max(requested_count, len(selected) * 2, 4)
    reserves: list[OpportunityDecision] = []
    selected_ids = {item.opportunity_id for item in selected}
    reserve_signatures: set[str] = set()
    reserve_episode_counts: dict[str, int] = {}
    for candidate in remaining:
        if candidate.opportunity_id in selected_ids:
            continue
        if candidate.semantic_signature and candidate.semantic_signature in reserve_signatures:
            continue
        if any(_overlaps(candidate, existing) for existing in reserves):
            continue
        if reserve_episode_counts.get(candidate.episode_id, 0) >= 2:
            continue
        reserves.append(candidate)
        reserve_episode_counts[candidate.episode_id] = (
            reserve_episode_counts.get(candidate.episode_id, 0) + 1
        )
        if candidate.semantic_signature:
            reserve_signatures.add(candidate.semantic_signature)
        if len(reserves) >= reserve_limit:
            break
    return selected, reserves


def build_visual_opportunity_plan(
    cards: list[dict[str, Any]],
    *,
    clip_duration: float,
    requested_count: int,
    blocked_card_ids: set[str] | None = None,
) -> VisualOpportunityPlan:
    blocked_card_ids = {
        _clean_space(card_id)
        for card_id in (blocked_card_ids or set())
        if _clean_space(card_id)
    }
    ordered = sorted(
        (dict(card) for card in cards if _card_id(card)),
        key=_card_start,
    )
    card_by_id = {_card_id(card): card for card in ordered}
    episodes = build_semantic_episodes(ordered, clip_duration=clip_duration)
    primary: list[OpportunityDecision] = []
    assistive: list[OpportunityDecision] = []
    rejected: list[OpportunityDecision] = []
    for episode in episodes:
        episode_cards = [
            card_by_id[card_id]
            for card_id in episode.card_ids
            if card_id in card_by_id
        ]
        for window in _candidate_windows(episode, episode_cards):
            decision = _decision_for_window(episode, window)
            if decision.opportunity_id in blocked_card_ids or any(
                card_id in blocked_card_ids
                for card_id in decision.source_card_ids
            ):
                rejected.append(
                    OpportunityDecision(
                        **{
                            **decision.to_dict(),
                            "status": "rejected",
                            "reason": "blocked_by_prior_failure_or_usage",
                        }
                    )
                )
            elif decision.status == "candidate":
                primary.append(decision)
            elif decision.status == "assistive_candidate":
                assistive.append(decision)
            else:
                rejected.append(decision)
    accepted = [*primary, *assistive]
    selected, reserves = _schedule_opportunities(
        accepted,
        requested_count=requested_count,
        clip_duration=clip_duration,
    )
    selected_ids = {item.opportunity_id for item in selected}
    reserve_ids = {item.opportunity_id for item in reserves}
    for candidate in accepted:
        if candidate.opportunity_id in selected_ids or candidate.opportunity_id in reserve_ids:
            continue
        rejected.append(
            OpportunityDecision(
                **{
                    **candidate.to_dict(),
                    "status": "rejected",
                    "reason": "not_selected_by_global_scheduler",
                }
            )
        )
    return VisualOpportunityPlan(
        version=VISUAL_OPPORTUNITY_PLAN_VERSION,
        duration_sec=clip_duration,
        requested_count=max(0, int(requested_count)),
        recommended_count=len(selected),
        episodes=episodes,
        selected=selected,
        reserves=reserves,
        rejected=rejected,
    )


__all__ = [
    "OpportunityDecision",
    "SemanticEpisode",
    "VISUAL_OPPORTUNITY_PLAN_VERSION",
    "VisualOpportunityPlan",
    "build_semantic_episodes",
    "build_visual_opportunity_plan",
]
