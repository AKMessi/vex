from __future__ import annotations

import json
import re
import subprocess
from typing import Any

import config
from broll_intelligence import (
    call_reasoning_model,
    extract_json_array,
    infer_visual_type,
    semantic_keywords,
    truncate,
    window_text,
)

SUPPORTED_TEMPLATES = {
    "data_journey": "A premium quantitative reveal with moving data, guided focus, and visual momentum.",
    "signal_network": "A premium process or system map with directional flow and network choreography.",
    "kinetic_route": "A premium timeline or journey beat staged along a route or guided path.",
    "spotlight_compare": "A premium contrast scene that morphs or spotlights the meaningful difference.",
    "interface_cascade": "A premium interface walkthrough with layered depth and focused camera attention.",
    "ribbon_quote": "A premium line or concept staged with kinetic type and directional motion.",
    "metric_callout": "Large value or strong claim with supporting context.",
    "keyword_stack": "Stacked short concepts with strong editorial styling.",
    "timeline_steps": "A short process or sequence laid out step by step.",
    "comparison_split": "Side-by-side contrast between two states.",
    "quote_focus": "A clean emphasis shot for a memorable phrase or line.",
    "system_flow": "A connected flow of stages or ideas.",
    "stat_grid": "A compact dashboard of 2-4 supporting stats or takeaways.",
}

STYLE_PACKS = {
    "editorial_clean": {
        "background": "#09111F",
        "panel_fill": "#12223C",
        "panel_stroke": "#5BC0EB",
        "accent": "#F59E0B",
        "accent_secondary": "#38BDF8",
        "glow": "#1D4ED8",
        "eyebrow_fill": "#14324D",
        "eyebrow_text": "#E0F2FE",
        "grid": "#214668",
        "text_primary": "#F8FAFC",
        "text_secondary": "#D6E3F3",
    },
    "bold_tech": {
        "background": "#07131D",
        "panel_fill": "#0D2435",
        "panel_stroke": "#34D399",
        "accent": "#38BDF8",
        "accent_secondary": "#A7F3D0",
        "glow": "#0EA5E9",
        "eyebrow_fill": "#103244",
        "eyebrow_text": "#D1FAE5",
        "grid": "#173A45",
        "text_primary": "#ECFEFF",
        "text_secondary": "#BAE6FD",
    },
    "documentary_kinetic": {
        "background": "#120B0C",
        "panel_fill": "#24151A",
        "panel_stroke": "#F97316",
        "accent": "#FACC15",
        "accent_secondary": "#FB7185",
        "glow": "#EA580C",
        "eyebrow_fill": "#3A1912",
        "eyebrow_text": "#FFEDD5",
        "grid": "#4A1F16",
        "text_primary": "#FFF7ED",
        "text_secondary": "#FED7AA",
    },
    "product_ui": {
        "background": "#08101E",
        "panel_fill": "#10223E",
        "panel_stroke": "#818CF8",
        "accent": "#22C55E",
        "accent_secondary": "#60A5FA",
        "glow": "#4F46E5",
        "eyebrow_fill": "#182A49",
        "eyebrow_text": "#E0E7FF",
        "grid": "#203257",
        "text_primary": "#F8FAFC",
        "text_secondary": "#C7D2FE",
    },
    "cinematic_night": {
        "background": "#050816",
        "panel_fill": "#101A34",
        "panel_stroke": "#A78BFA",
        "accent": "#F43F5E",
        "accent_secondary": "#F59E0B",
        "glow": "#7C3AED",
        "eyebrow_fill": "#20163C",
        "eyebrow_text": "#F3E8FF",
        "grid": "#241C4C",
        "text_primary": "#F8FAFC",
        "text_secondary": "#E9D5FF",
    },
    "signal_lab": {
        "background": "#08151A",
        "panel_fill": "#0E2328",
        "panel_stroke": "#2DD4BF",
        "accent": "#FBBF24",
        "accent_secondary": "#38BDF8",
        "glow": "#0F766E",
        "eyebrow_fill": "#13343B",
        "eyebrow_text": "#CCFBF1",
        "grid": "#18424A",
        "text_primary": "#F0FDFA",
        "text_secondary": "#99F6E4",
    },
    "magazine_luxe": {
        "background": "#140E12",
        "panel_fill": "#26171D",
        "panel_stroke": "#FB7185",
        "accent": "#F59E0B",
        "accent_secondary": "#FDBA74",
        "glow": "#BE185D",
        "eyebrow_fill": "#3A1B25",
        "eyebrow_text": "#FFE4E6",
        "grid": "#4A1D2B",
        "text_primary": "#FFF1F2",
        "text_secondary": "#FBCFE8",
    },
}

STYLE_PACK_HINTS = {
    "data_graphic": "bold_tech",
    "product_ui": "product_ui",
    "process": "signal_lab",
    "abstract_motion": "cinematic_night",
    "cutaway": "magazine_luxe",
    "location": "documentary_kinetic",
}

THEME_BY_VISUAL_TYPE = {
    "data_graphic": {
        "panel_stroke": "#38BDF8",
        "accent": "#FACC15",
    },
    "product_ui": {
        "panel_stroke": "#818CF8",
        "accent": "#22C55E",
    },
    "process": {
        "panel_stroke": "#34D399",
        "accent": "#F97316",
    },
    "abstract_motion": {
        "panel_stroke": "#A78BFA",
        "accent": "#F43F5E",
    },
}

RENDERER_HINTS_BY_TYPE = {
    "data_graphic": "manim",
    "product_ui": "manim",
    "process": "manim",
    "abstract_motion": "manim",
    "cutaway": "ffmpeg",
    "location": "ffmpeg",
}

PROCESS_MARKERS = {
    "first", "then", "next", "finally", "step", "steps", "process", "workflow", "system",
    "pipeline", "capture", "score", "render", "build", "turn", "convert", "flow",
}
CONTRAST_MARKERS = {
    "before", "after", "vs", "versus", "instead", "old", "new", "manual", "automated",
    "replace", "better", "worse", "from", "to", "shift", "compare",
}
GENERIC_ABSTRACT_TERMS = {
    "idea", "concept", "mindset", "approach", "thinking", "stuff", "things", "better",
    "growth", "learn", "lesson", "motivation", "strategy", "value", "important", "useful",
    "future", "success", "mind", "creative", "belief", "hard", "easy", "powerful",
}
FILLER_LEAD_WORDS = {
    "the", "a", "an", "this", "that", "these", "those", "we", "you", "it", "they", "our",
    "your", "their", "to", "for", "with", "by", "in", "on", "of",
}
TRAILING_TRIM_WORDS = {"with", "by", "to", "for", "and", "or", "of", "in", "on", "a", "an", "the", "then", "next", "finally"}
DISTILL_WORD_PATTERN = re.compile(r"[A-Za-z0-9%+.-]+(?:'[A-Za-z0-9%+.-]+)*")
BACKGROUND_MOTIFS = ("grid", "rings", "beams", "constellation", "bands")
PLAN_CACHE_VERSION = "2026-04-26-v3"
MIN_PREMIUM_REPLACE_DURATION_SEC = 2.4
LAYOUT_VARIANTS = {
    "data_journey": "arc_stage",
    "signal_network": "network_sweep",
    "kinetic_route": "route_curve",
    "spotlight_compare": "spotlight_stage",
    "interface_cascade": "cascade_focus",
    "ribbon_quote": "ribbon_sweep",
    "metric_callout": "hero_split",
    "keyword_stack": "stagger_stack",
    "timeline_steps": "elevated_timeline",
    "comparison_split": "offset_split",
    "quote_focus": "editorial_stage",
    "system_flow": "signal_chain",
    "stat_grid": "dashboard_mosaic",
}

PREMIUM_TEMPLATE_UPGRADES = {
    "metric_callout": "data_journey",
    "stat_grid": "data_journey",
    "timeline_steps": "kinetic_route",
    "system_flow": "signal_network",
    "comparison_split": "spotlight_compare",
    "quote_focus": "ribbon_quote",
    "keyword_stack": "ribbon_quote",
}

EDITORIAL_TEMPLATE_DOWNGRADES = {
    "data_journey": "metric_callout",
    "signal_network": "timeline_steps",
    "kinetic_route": "timeline_steps",
    "spotlight_compare": "comparison_split",
    "interface_cascade": "comparison_split",
    "ribbon_quote": "quote_focus",
}


def detect_scene_cuts(
    input_path: str,
    threshold: float = 0.34,
    min_gap_sec: float = 0.8,
) -> list[float]:
    command = [
        config.FFMPEG_PATH,
        "-i",
        input_path,
        "-filter:v",
        f"select=gt(scene\\,{threshold}),showinfo",
        "-an",
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        return []
    scene_cuts: list[float] = []
    for line in (result.stderr or "").splitlines():
        match = re.search(r"pts_time:([0-9.]+)", line)
        if not match:
            continue
        pts_time = float(match.group(1))
        if scene_cuts and pts_time - scene_cuts[-1] < min_gap_sec:
            continue
        scene_cuts.append(round(pts_time, 3))
    return scene_cuts


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9']+", str(text or "").lower())


def _proper_noun_count(text: str) -> int:
    return len(re.findall(r"\b[A-Z][a-zA-Z0-9]+\b", str(text or "")))


def _split_fragments(text: str, *, limit: int = 6) -> list[str]:
    raw = re.split(r"(?:[.;:!?]|\b(?:and then|then|next|finally|because|so|while)\b|,)", str(text or ""), flags=re.IGNORECASE)
    fragments: list[str] = []
    for part in raw:
        cleaned = re.sub(r"\s+", " ", part).strip(" -,\n\t")
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in FILLER_LEAD_WORDS:
            continue
        if cleaned not in fragments:
            fragments.append(cleaned)
        if len(fragments) >= limit:
            break
    return fragments


def _distill_phrase(text: str, *, max_words: int = 6, max_chars: int = 42) -> str:
    original = re.sub(r"\s+", " ", str(text or "")).strip()
    if not original:
        return ""
    words = DISTILL_WORD_PATTERN.findall(original)
    if not words:
        return original
    kept: list[str] = []
    for word in words:
        lowered = word.lower()
        if not kept and lowered in FILLER_LEAD_WORDS:
            continue
        kept.append(word)
        if len(kept) >= max_words:
            break
    candidate = " ".join(kept).strip()
    if not candidate:
        candidate = " ".join(words[:max_words]).strip()
    while candidate:
        tail = candidate.split(" ")[-1].lower()
        if tail not in TRAILING_TRIM_WORDS:
            break
        candidate = " ".join(candidate.split(" ")[:-1]).strip()
    while len(candidate) > max_chars and " " in candidate:
        candidate = " ".join(candidate.split(" ")[:-1]).strip()
    if len(candidate) > max_chars:
        candidate = truncate(candidate, max_chars)
    return candidate or original


def _display_case(text: str) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if not value:
        return ""
    if value.upper() == value and len(value) <= 6:
        return value
    return value[0].upper() + value[1:]


def _polish_visual_copy(text: Any, *, max_words: int, max_chars: int) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    return _display_case(_distill_phrase(value, max_words=max_words, max_chars=max_chars))


def _generic_penalty(text: str) -> float:
    tokens = _tokens(text)
    if not tokens:
        return 0.0
    abstract_hits = sum(1 for token in tokens if token in GENERIC_ABSTRACT_TERMS)
    ratio = abstract_hits / max(len(tokens), 1)
    return round(min(ratio * 1.35, 1.0), 3)


def _process_cue_score(text: str) -> float:
    tokens = set(_tokens(text))
    hits = len(tokens & PROCESS_MARKERS)
    return round(min(hits / 4.0, 1.0), 3)


def _contrast_cue_score(text: str) -> float:
    lowered = str(text or "").lower()
    hits = 0
    for marker in CONTRAST_MARKERS:
        if re.search(rf"\b{re.escape(marker)}\b", lowered):
            hits += 1
    if re.search(r"\bfrom\b.+\bto\b", lowered):
        hits += 2
    return round(min(hits / 4.0, 1.0), 3)


def _concrete_hit_score(text: str) -> float:
    tokens = set(_tokens(text))
    concrete_terms = {
        "screen", "dashboard", "timeline", "editor", "camera", "graph", "chart", "prompt",
        "workflow", "transcript", "scene", "clip", "audio", "export", "system", "render",
        "code", "video", "caption", "app", "panel", "browser", "interface",
    }
    hits = len(tokens & concrete_terms)
    return round(min(hits / 5.0, 1.0), 3)


def _visualizability_score(
    *,
    numeric_hits: int,
    process_cues: float,
    contrast_cues: float,
    concrete_hits: float,
    proper_nouns: int,
    generic_penalty: float,
    replace_safety: float,
) -> float:
    score = 0.24
    score += min(numeric_hits, 3) * 0.11
    score += process_cues * 0.24
    score += contrast_cues * 0.22
    score += concrete_hits * 0.18
    score += min(proper_nouns, 4) * 0.035
    score += replace_safety * 0.08
    score -= generic_penalty * 0.28
    return round(max(0.0, min(score, 1.0)), 3)


def _headline_from_card(card: dict[str, Any]) -> str:
    sentence = str(card.get("sentence_text") or "")
    fragments = _split_fragments(sentence)
    if int(card.get("numeric_hits") or 0) > 0:
        number_match = re.search(r"\b\d+(?:\.\d+)?(?:%|x)?\b", sentence, flags=re.IGNORECASE)
        if number_match:
            number = number_match.group(0)
            for fragment in fragments:
                if number in fragment:
                    return _display_case(_distill_phrase(fragment, max_words=6, max_chars=40))
            return _display_case(_distill_phrase(f"{number} {sentence}", max_words=6, max_chars=40))
    if float(card.get("process_cues") or 0.0) >= 0.35 and fragments:
        return _display_case(_distill_phrase(fragments[0], max_words=5, max_chars=36))
    if float(card.get("contrast_cues") or 0.0) >= 0.25 and fragments:
        return _display_case(_distill_phrase(fragments[0], max_words=6, max_chars=38))
    return _display_case(_distill_phrase(fragments[0] if fragments else sentence, max_words=6, max_chars=40))


def _supporting_lines_for_card(card: dict[str, Any]) -> list[str]:
    headline = _headline_from_card(card).lower()
    fragments = _split_fragments(f"{card.get('sentence_text', '')}. {card.get('context_text', '')}", limit=8)
    lines: list[str] = []
    for fragment in fragments:
        trimmed = _display_case(_distill_phrase(fragment, max_words=7, max_chars=44))
        lowered = trimmed.lower()
        if not trimmed or lowered == headline or lowered in lines:
            continue
        if headline in lowered or lowered in headline:
            continue
        lines.append(trimmed)
        if len(lines) >= 3:
            break
    if not lines:
        context = _display_case(
            _distill_phrase(str(card.get("context_text") or card.get("sentence_text") or ""), max_words=7, max_chars=44)
        )
        if context:
            lines.append(context)
    return lines[:3]


def _steps_for_card(card: dict[str, Any]) -> list[str]:
    headline = _headline_from_card(card).lower()
    fragments = _split_fragments(f"{card.get('sentence_text', '')}. {card.get('context_text', '')}", limit=8)
    steps = [_display_case(_distill_phrase(fragment, max_words=5, max_chars=28)) for fragment in fragments if fragment]
    deduped: list[str] = []
    for step in steps:
        lowered = step.lower()
        if lowered == headline or headline in lowered or lowered in headline:
            continue
        if not step or lowered in {item.lower() for item in deduped}:
            continue
        deduped.append(step)
        if len(deduped) >= 4:
            break
    return deduped[:4]


def _comparison_terms_for_card(card: dict[str, Any]) -> tuple[str, str, str, str]:
    sentence = str(card.get("sentence_text") or "")
    context = str(card.get("context_text") or "")
    lowered = sentence.lower()
    from_to = re.search(r"\bfrom\s+(.+?)\s+to\s+(.+?)(?:[.,;]|$)", sentence, flags=re.IGNORECASE)
    if from_to:
        left = _polish_visual_copy(from_to.group(1), max_words=5, max_chars=34)
        right = _polish_visual_copy(from_to.group(2), max_words=5, max_chars=34)
        return "From", "To", left, right
    versus = re.search(r"(.+?)\s+(?:vs|versus)\s+(.+?)(?:[.,;]|$)", sentence, flags=re.IGNORECASE)
    if versus:
        left = _polish_visual_copy(versus.group(1), max_words=5, max_chars=34)
        right = _polish_visual_copy(versus.group(2), max_words=5, max_chars=34)
        return "Option A", "Option B", left, right
    if "before" in lowered and "after" in lowered:
        return (
            "Before",
            "After",
            _polish_visual_copy(sentence, max_words=5, max_chars=34),
            _polish_visual_copy(context or sentence, max_words=5, max_chars=34),
        )
    return (
        "Old way",
        "New way",
        _polish_visual_copy(sentence, max_words=5, max_chars=34),
        _polish_visual_copy(context or sentence, max_words=5, max_chars=34),
    )


def _eyebrow_for_card(card: dict[str, Any], template: str) -> str:
    visual_type = str(card.get("visual_type_hint") or "")
    if template in {"metric_callout", "stat_grid"}:
        return "SIGNAL"
    if template in {"timeline_steps", "system_flow"}:
        return "WORKFLOW"
    if template == "comparison_split":
        return "SHIFT"
    if visual_type == "product_ui":
        return "INTERFACE"
    if visual_type == "abstract_motion":
        return "IDEA"
    return "INSIGHT"


def _deck_for_card(card: dict[str, Any], headline: str) -> str:
    for line in _supporting_lines_for_card(card):
        if line.lower() != headline.lower():
            return truncate(line, 46)
    return ""


def _background_motif(card: dict[str, Any], template: str, style_pack: str) -> str:
    visual_type = str(card.get("visual_type_hint") or "")
    if template in {"timeline_steps", "system_flow", "kinetic_route", "signal_network"}:
        return "grid"
    if template in {"comparison_split", "spotlight_compare"}:
        return "bands"
    if template == "data_journey":
        return "rings"
    if template == "interface_cascade":
        return "beams"
    if template == "ribbon_quote":
        return "constellation"
    if visual_type == "abstract_motion":
        return "rings"
    if style_pack in {"magazine_luxe", "documentary_kinetic"}:
        return "beams"
    return "constellation"


def _count_numbers(text: str) -> int:
    return len(re.findall(r"\b\d+(?:\.\d+)?(?:%|x)?\b", text, flags=re.IGNORECASE))


def _card_words(sentence: dict[str, Any], words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    start_index = int(sentence.get("word_start_index") or 0)
    end_index = int(sentence.get("word_end_index") or 0)
    if start_index > 0 and end_index >= start_index:
        return [word for word in words if start_index <= int(word.get("index") or 0) <= end_index]
    start_sec = float(sentence.get("start") or 0.0)
    end_sec = float(sentence.get("end") or start_sec)
    return [
        word
        for word in words
        if float(word.get("end") or 0.0) > start_sec and float(word.get("start") or 0.0) < end_sec
    ]


def _nearest_scene_distance(start_sec: float, end_sec: float, scene_cuts: list[float]) -> tuple[float | None, float]:
    if not scene_cuts:
        return None, 999.0
    anchor = start_sec
    nearest = min(scene_cuts, key=lambda cut: min(abs(cut - start_sec), abs(cut - end_sec)))
    return nearest, min(abs(nearest - anchor), abs(nearest - end_sec))


def _replace_safety(
    *,
    pause_before: float,
    pause_after: float,
    scene_distance: float,
    words_per_second: float,
    visual_type_hint: str,
) -> float:
    score = 0.22
    score += min(pause_before, 0.6) * 0.38
    score += min(pause_after, 0.6) * 0.46
    if scene_distance <= 0.35:
        score += 0.18
    if scene_distance <= 0.18:
        score += 0.06
    if words_per_second <= 2.0:
        score += 0.1
    if visual_type_hint in {"abstract_motion", "cutaway", "location"}:
        score += 0.08
    if visual_type_hint in {"data_graphic", "product_ui"}:
        score -= 0.06
    return round(max(0.0, min(score, 1.0)), 3)


def _default_style_pack(visual_type_hint: str) -> str:
    return STYLE_PACK_HINTS.get(visual_type_hint, "editorial_clean")


def _theme_for_card(card: dict[str, Any], style_pack: str) -> dict[str, str]:
    theme = dict(STYLE_PACKS.get(style_pack, STYLE_PACKS["editorial_clean"]))
    theme.update(THEME_BY_VISUAL_TYPE.get(str(card.get("visual_type_hint") or ""), {}))
    return theme


def _default_template(card: dict[str, Any]) -> str:
    visual_type = str(card.get("visual_type_hint") or "")
    numbers = int(card["sentence_numeric_hits"]) if "sentence_numeric_hits" in card else int(card.get("numeric_hits") or 0)
    process_cues = float(card["sentence_process_cues"]) if "sentence_process_cues" in card else float(card.get("process_cues") or 0.0)
    contrast_cues = float(card["sentence_contrast_cues"]) if "sentence_contrast_cues" in card else float(card.get("contrast_cues") or 0.0)
    if numbers >= 1 and contrast_cues < 0.34:
        return "data_journey"
    if process_cues >= 0.42:
        return "signal_network" if len(card.get("keywords") or []) >= 3 else "kinetic_route"
    if contrast_cues >= 0.4:
        return "spotlight_compare"
    if visual_type == "data_graphic":
        return "data_journey"
    if visual_type == "process":
        return "signal_network" if len(card.get("keywords") or []) >= 3 else "kinetic_route"
    if visual_type == "product_ui":
        return "interface_cascade"
    if visual_type == "abstract_motion":
        return "ribbon_quote"
    return "ribbon_quote" if numbers == 0 else "data_journey"


def _default_renderer_hint(card: dict[str, Any]) -> str:
    visual_type = str(card.get("visual_type_hint") or "")
    if visual_type in RENDERER_HINTS_BY_TYPE:
        return RENDERER_HINTS_BY_TYPE[visual_type]
    return "ffmpeg"


def _default_motion_preset(card: dict[str, Any], template: str) -> str:
    visual_type = str(card.get("visual_type_hint") or "")
    if template in {"timeline_steps", "system_flow", "kinetic_route", "signal_network"}:
        return "diagram_draw"
    if template in {"metric_callout", "stat_grid", "data_journey"}:
        return "kinetic_pop"
    if template in {"spotlight_compare", "interface_cascade"}:
        return "focus_shift"
    if template == "ribbon_quote":
        return "type_sweep"
    if visual_type == "abstract_motion":
        return "spotlight_sweep"
    return "gentle_rise"


def _upgrade_to_premium_template(card: dict[str, Any], template: str, composition_mode: str) -> str:
    if composition_mode != "replace":
        return template
    visual_type = str(card.get("visual_type_hint") or "")
    replace_safety = float(card.get("replace_safety") or 0.0)
    visualizability = float(card.get("visualizability") or 0.0)
    numeric_hits = int(card.get("numeric_hits") or 0)
    process_cues = float(card.get("process_cues") or 0.0)
    contrast_cues = float(card.get("contrast_cues") or 0.0)
    if visual_type == "location":
        return template
    if visual_type == "cutaway" and replace_safety < 0.68 and visualizability < 0.7:
        return template
    if template == "keyword_stack":
        if contrast_cues >= 0.18:
            return "spotlight_compare"
        if process_cues >= 0.18:
            return "signal_network"
        if numeric_hits >= 1:
            return "data_journey"
        return "ribbon_quote"
    if template == "comparison_split":
        return "spotlight_compare"
    if template in PREMIUM_TEMPLATE_UPGRADES:
        return PREMIUM_TEMPLATE_UPGRADES[template]
    return template


def _format_renderer_capabilities(capabilities: list[dict[str, Any]] | None) -> str:
    if not capabilities:
        return "No renderer metadata was provided."
    lines: list[str] = []
    for item in capabilities:
        name = str(item.get("name") or "unknown")
        available = bool(item.get("available"))
        templates = ", ".join(item.get("supported_templates") or [])
        reason = str(item.get("reason") or "")
        suffix = f" unavailable: {reason}" if not available and reason else ""
        lines.append(f"- {name} | available={available} | templates={templates}{suffix}")
    return "\n".join(lines)


def _visual_priority(card: dict[str, Any]) -> float:
    combined = f"{card['sentence_text']} {card['context_text']}".lower()
    tokens = re.findall(r"[a-zA-Z0-9']+", combined)
    if not tokens:
        return 0.0
    numbers = int(card.get("numeric_hits") or 0)
    specificity = min(len(set(tokens)) / max(len(tokens), 1), 1.0)
    visual_hits = len(card.get("keywords") or [])
    proper_nouns = _proper_noun_count(f"{card['sentence_text']} {card['context_text']}")
    cut_bonus = max(0.0, 1.0 - min(float(card.get("scene_distance") or 999.0), 1.0))
    replace_bonus = float(card.get("replace_safety") or 0.0)
    pause_bonus = min(float(card.get("pause_after") or 0.0), 0.6) * 9.0
    visualizability = float(card.get("visualizability") or 0.0)
    generic_penalty = float(card.get("generic_penalty") or 0.0)
    return round(
        24
        + numbers * 8.5
        + specificity * 20
        + min(visual_hits, 8) * 3.2
        + proper_nouns * 2.3
        + cut_bonus * 11
        + replace_bonus * 14
        + pause_bonus,
        2,
    ) + round(visualizability * 18 - generic_penalty * 14, 2)


def build_visual_context_cards(
    sentences: list[dict[str, Any]],
    transcript_segments: list[dict[str, Any]],
    clip_duration: float,
    *,
    words: list[dict[str, Any]] | None = None,
    scene_cuts: list[float] | None = None,
) -> list[dict[str, Any]]:
    words = words or []
    scene_cuts = scene_cuts or []
    cards: list[dict[str, Any]] = []
    total_sentences = len(sentences)
    for index, sentence in enumerate(sentences, start=1):
        start_sec = max(0.0, min(float(sentence.get("start") or 0.0), clip_duration))
        end_sec = max(start_sec + 0.12, min(float(sentence.get("end") or start_sec + 0.8), clip_duration))
        sentence_text = truncate(str(sentence.get("text") or ""), 220)
        if not sentence_text:
            continue
        context_text = truncate(
            window_text(
                transcript_segments,
                max(0.0, start_sec - 3.0),
                min(clip_duration, end_sec + 3.0),
            ),
            320,
        )
        card_words = _card_words(sentence, words)
        pause_before = 0.0
        pause_after = 0.0
        if index > 1:
            prev = sentences[index - 2]
            pause_before = max(0.0, start_sec - float(prev.get("end") or start_sec))
        if index < total_sentences:
            nxt = sentences[index]
            pause_after = max(0.0, float(nxt.get("start") or end_sec) - end_sec)
        word_count = len(card_words)
        words_per_second = round(word_count / max(end_sec - start_sec, 0.15), 2) if word_count else 0.0
        keywords = semantic_keywords(f"{sentence_text} {context_text}", limit=8)
        visual_type_hint = infer_visual_type(f"{sentence_text} {context_text}")
        nearest_scene_cut, scene_distance = _nearest_scene_distance(start_sec, end_sec, scene_cuts)
        sentence_numeric_hits = _count_numbers(sentence_text)
        numeric_hits = _count_numbers(f"{sentence_text} {context_text}")
        sentence_process_cues = _process_cue_score(sentence_text)
        process_cues = _process_cue_score(f"{sentence_text} {context_text}")
        sentence_contrast_cues = _contrast_cue_score(sentence_text)
        contrast_cues = _contrast_cue_score(f"{sentence_text} {context_text}")
        generic_penalty = _generic_penalty(f"{sentence_text} {context_text}")
        concrete_hits = _concrete_hit_score(f"{sentence_text} {context_text}")
        proper_nouns = _proper_noun_count(f"{sentence_text} {context_text}")
        replace_safety = _replace_safety(
            pause_before=pause_before,
            pause_after=pause_after,
            scene_distance=scene_distance,
            words_per_second=words_per_second,
            visual_type_hint=visual_type_hint,
        )
        visualizability = _visualizability_score(
            numeric_hits=numeric_hits,
            process_cues=process_cues,
            contrast_cues=contrast_cues,
            concrete_hits=concrete_hits,
            proper_nouns=proper_nouns,
            generic_penalty=generic_penalty,
            replace_safety=replace_safety,
        )
        suggested_composition = (
            "replace"
            if replace_safety >= 0.63 and visualizability >= 0.58 and visual_type_hint not in {"data_graphic", "product_ui"}
            else "picture_in_picture"
        )
        style_pack = _default_style_pack(visual_type_hint)
        row = {
            "card_id": f"visual_card_{index:03d}",
            "start": round(start_sec, 2),
            "end": round(end_sec, 2),
            "sentence_text": sentence_text,
            "context_text": context_text,
            "keywords": keywords,
            "visual_type_hint": visual_type_hint,
            "word_count": word_count,
            "words_per_second": words_per_second,
            "pause_before": round(pause_before, 3),
            "pause_after": round(pause_after, 3),
            "nearest_scene_cut": round(nearest_scene_cut, 3) if nearest_scene_cut is not None else None,
            "scene_distance": round(scene_distance, 3),
            "sentence_numeric_hits": sentence_numeric_hits,
            "numeric_hits": numeric_hits,
            "sentence_process_cues": sentence_process_cues,
            "process_cues": process_cues,
            "sentence_contrast_cues": sentence_contrast_cues,
            "contrast_cues": contrast_cues,
            "generic_penalty": generic_penalty,
            "concrete_hits": concrete_hits,
            "proper_nouns": proper_nouns,
            "replace_safety": replace_safety,
            "visualizability": visualizability,
            "suggested_composition": suggested_composition,
            "style_pack": style_pack,
            "suggested_renderer": _default_renderer_hint({"visual_type_hint": visual_type_hint}),
        }
        row["priority"] = _visual_priority(row)
        cards.append(row)
    return cards


def _format_cards_for_llm(cards: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for card in cards:
        lines.append(
            "\n".join(
                [
                    f"{card['card_id']} | {card['start']:.2f}-{card['end']:.2f} | priority={card['priority']:.2f}",
                    f"Sentence: {card['sentence_text']}",
                    f"Context: {card['context_text']}",
                    f"Keywords: {', '.join(card['keywords'])}",
                    f"Hint: {card['visual_type_hint']} | renderer={card['suggested_renderer']} | style={card['style_pack']}",
                    (
                        "Evidence: "
                        f"numbers={card['numeric_hits']} | "
                        f"visualizability={card['visualizability']:.2f} | "
                        f"generic_penalty={card['generic_penalty']:.2f} | "
                        f"wps={card['words_per_second']:.2f} | "
                        f"process={card['process_cues']:.2f} | "
                        f"contrast={card['contrast_cues']:.2f} | "
                        f"pause_before={card['pause_before']:.2f}s | "
                        f"pause_after={card['pause_after']:.2f}s | "
                        f"scene_distance={card['scene_distance']:.2f}s | "
                        f"replace_safety={card['replace_safety']:.2f}"
                    ),
                ]
            )
        )
    return "\n\n".join(lines)


def _snap_to_scene(
    value: float,
    scene_cuts: list[float],
    max_distance: float = 0.4,
    *,
    direction: str = "nearest",
) -> float:
    if not scene_cuts:
        return value
    if direction == "forward":
        forward = [cut for cut in scene_cuts if cut >= value]
        if forward:
            nearest = min(forward, key=lambda cut: abs(cut - value))
        else:
            nearest = min(scene_cuts, key=lambda cut: abs(cut - value))
    elif direction == "backward":
        backward = [cut for cut in scene_cuts if cut <= value]
        if backward:
            nearest = min(backward, key=lambda cut: abs(cut - value))
        else:
            nearest = min(scene_cuts, key=lambda cut: abs(cut - value))
    else:
        nearest = min(scene_cuts, key=lambda cut: abs(cut - value))
    if abs(nearest - value) <= max_distance:
        return nearest
    return value


def _expand_window_to_duration(
    start_sec: float,
    end_sec: float,
    *,
    clip_duration: float,
    target_duration_sec: float,
    scene_cuts: list[float],
) -> tuple[float, float]:
    current_duration = max(end_sec - start_sec, 0.0)
    if target_duration_sec <= 0.0 or current_duration >= target_duration_sec:
        return start_sec, end_sec
    center = (start_sec + end_sec) / 2.0
    expanded_start = max(0.0, center - target_duration_sec * 0.48)
    expanded_end = min(clip_duration, center + target_duration_sec * 0.52)
    if expanded_end - expanded_start < target_duration_sec:
        shortfall = target_duration_sec - (expanded_end - expanded_start)
        expanded_start = max(0.0, expanded_start - shortfall)
        if expanded_end - expanded_start < target_duration_sec:
            expanded_end = min(clip_duration, expanded_end + (target_duration_sec - (expanded_end - expanded_start)))
    snapped_start = _snap_to_scene(expanded_start, scene_cuts, max_distance=0.7, direction="backward")
    snapped_end = _snap_to_scene(expanded_end, scene_cuts, max_distance=0.7, direction="forward")
    if snapped_end - snapped_start >= max(target_duration_sec * 0.84, current_duration):
        return max(0.0, snapped_start), min(clip_duration, snapped_end)
    return max(0.0, expanded_start), min(clip_duration, expanded_end)


def _extract_emphasis_text(card: dict[str, Any]) -> str:
    text = str(card.get("sentence_text") or "")
    number_match = re.search(r"\b\d+(?:\.\d+)?(?:%|x)?\b", text, flags=re.IGNORECASE)
    if number_match:
        return number_match.group(0)
    keywords = list(card.get("keywords") or [])
    if keywords:
        return " ".join(keywords[:3])
    return truncate(text, 32)


def _coerce_string_list(raw: Any, limit: int, max_chars: int) -> list[str]:
    if isinstance(raw, list):
        values = [
            _polish_visual_copy(item, max_words=4 if max_chars <= 30 else 7, max_chars=max_chars)
            for item in raw
            if str(item).strip()
        ]
    elif str(raw or "").strip():
        values = [_polish_visual_copy(raw, max_words=4 if max_chars <= 30 else 7, max_chars=max_chars)]
    else:
        values = []
    return values[:limit]


def _normalize_visual_plan(
    raw_plan: list[dict[str, Any]],
    cards: list[dict[str, Any]],
    clip_duration: float,
    max_visuals: int,
    min_visual_sec: float,
    max_visual_sec: float,
    scene_cuts: list[float],
    available_renderers: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    card_map = {card["card_id"]: card for card in cards}
    known_renderers = {str(item.get("name") or "").strip().lower() for item in (available_renderers or [])}
    available_names = {
        str(item.get("name") or "").strip().lower()
        for item in (available_renderers or [])
        if bool(item.get("available"))
    }
    normalized: list[dict[str, Any]] = []
    template_counts: dict[str, int] = {}
    last_end = -999.0
    for index, item in enumerate(raw_plan, start=1):
        card = card_map.get(str(item.get("card_id") or "").strip())
        if card is None:
            continue
        start_sec = max(0.0, min(float(card["start"]) - 0.08, clip_duration))
        end_sec = min(clip_duration, float(card["end"]) + 0.22)
        start_sec = _snap_to_scene(start_sec, scene_cuts, direction="forward")
        end_sec = _snap_to_scene(end_sec, scene_cuts, direction="backward")
        if end_sec - start_sec < min_visual_sec:
            end_sec = min(clip_duration, start_sec + min_visual_sec)
        if end_sec - start_sec > max_visual_sec:
            end_sec = start_sec + max_visual_sec
        if end_sec <= start_sec or start_sec - last_end < 0.18:
            continue
        confidence = max(0.0, min(float(item.get("confidence", 0.58)), 1.0))
        if confidence < 0.35:
            continue
        template = str(item.get("template") or _default_template(card)).strip().lower()
        if template not in SUPPORTED_TEMPLATES:
            template = _default_template(card)
        if float(card.get("visualizability") or 0.0) < 0.46 and template in {"quote_focus", "keyword_stack", "ribbon_quote"}:
            continue
        if float(card.get("generic_penalty") or 0.0) > 0.68 and int(card.get("numeric_hits") or 0) == 0 and float(card.get("process_cues") or 0.0) < 0.3:
            continue
        max_count_for_template = 1 if template in {"quote_focus", "keyword_stack", "ribbon_quote"} else 2
        if template_counts.get(template, 0) >= max_count_for_template:
            continue
        composition_mode = str(item.get("composition_mode") or card["suggested_composition"]).strip().lower()
        if composition_mode in {"pip", "overlay", "picture-in-picture"}:
            composition_mode = "picture_in_picture"
        if composition_mode not in {"replace", "picture_in_picture"}:
            composition_mode = card["suggested_composition"]
        minimum_duration = min_visual_sec
        if composition_mode == "replace":
            minimum_duration = max(minimum_duration, MIN_PREMIUM_REPLACE_DURATION_SEC)
        start_sec, end_sec = _expand_window_to_duration(
            start_sec,
            end_sec,
            clip_duration=clip_duration,
            target_duration_sec=min(minimum_duration, max_visual_sec),
            scene_cuts=scene_cuts,
        )
        if end_sec - start_sec < minimum_duration:
            continue
        if composition_mode == "picture_in_picture":
            template = EDITORIAL_TEMPLATE_DOWNGRADES.get(template, template)
        template = _upgrade_to_premium_template(card, template, composition_mode)
        position = str(item.get("position") or "bottom_right").strip().lower()
        if position not in {"top_left", "top_right", "bottom_left", "bottom_right", "top", "bottom", "center"}:
            position = "bottom_right"
        scale = round(max(0.24, min(float(item.get("scale", 0.42) or 0.42), 0.8)), 3)
        slot_duration = end_sec - start_sec
        short_slot = slot_duration <= 2.8
        supporting_lines = _coerce_string_list(item.get("supporting_lines"), limit=3, max_chars=72)
        keywords = _coerce_string_list(item.get("keywords"), limit=4, max_chars=28)
        steps = _coerce_string_list(item.get("steps"), limit=4, max_chars=28)
        derived_headline = _headline_from_card(card)
        headline = _polish_visual_copy(
            item.get("headline") or derived_headline or card["sentence_text"],
            max_words=5 if short_slot else 6,
            max_chars=38 if short_slot else 48,
        )
        if headline.lower() == str(card.get("sentence_text") or "").strip().lower() or len(headline.split()) > 8:
            headline = derived_headline or truncate(headline, 42)
        emphasis_source = str(item.get("emphasis_text") or _extract_emphasis_text(card))
        emphasis_text = emphasis_source if re.fullmatch(r"\d+(?:\.\d+)?(?:%|x)?", emphasis_source, flags=re.IGNORECASE) else _polish_visual_copy(emphasis_source, max_words=4, max_chars=34)
        footer_text = _polish_visual_copy(
            item.get("footer_text") or _deck_for_card(card, headline) or card["context_text"],
            max_words=6 if short_slot else 8,
            max_chars=42 if short_slot else 58,
        )
        style_pack = str(item.get("style_pack") or card["style_pack"] or "editorial_clean").strip().lower()
        if style_pack not in STYLE_PACKS:
            style_pack = card["style_pack"]
        renderer_hint = str(item.get("renderer_hint") or card["suggested_renderer"] or "auto").strip().lower()
        if composition_mode == "replace" and renderer_hint in {"auto", "ffmpeg"} and template not in {"quote_focus", "keyword_stack", "metric_callout", "stat_grid", "timeline_steps", "comparison_split"}:
            renderer_hint = "manim"
        if composition_mode == "picture_in_picture" and template in {"metric_callout", "keyword_stack", "quote_focus", "stat_grid", "comparison_split", "timeline_steps"}:
            renderer_hint = "ffmpeg"
        if renderer_hint in known_renderers and renderer_hint not in available_names:
            renderer_hint = "auto"
        if renderer_hint not in known_renderers and renderer_hint != "auto":
            renderer_hint = card["suggested_renderer"]
        motion_preset = truncate(
            str(item.get("motion_preset") or _default_motion_preset(card, template)),
            32,
        )
        eyebrow = truncate(str(item.get("eyebrow") or _eyebrow_for_card(card, template)), 18).upper()
        deck = _polish_visual_copy(
            item.get("deck") or _deck_for_card(card, headline),
            max_words=6 if short_slot else 8,
            max_chars=36 if short_slot else 50,
        )
        background_motif = str(item.get("background_motif") or _background_motif(card, template, style_pack)).strip().lower()
        if background_motif not in BACKGROUND_MOTIFS:
            background_motif = _background_motif(card, template, style_pack)
        layout_variant = str(item.get("layout_variant") or LAYOUT_VARIANTS.get(template, "hero_split")).strip().lower()
        spec = {
            "visual_id": f"visual_{index:03d}",
            "card_id": card["card_id"],
            "start": round(start_sec, 2),
            "end": round(end_sec, 2),
            "duration": round(end_sec - start_sec, 2),
            "sentence_text": card["sentence_text"],
            "context_text": card["context_text"],
            "keywords": card["keywords"][:8],
            "visual_type_hint": card["visual_type_hint"],
            "template": template,
            "composition_mode": composition_mode,
            "position": position,
            "scale": scale,
            "eyebrow": eyebrow,
            "headline": headline,
            "deck": deck,
            "emphasis_text": emphasis_text,
            "supporting_lines": supporting_lines,
            "steps": steps,
            "quote_text": _polish_visual_copy(item.get("quote_text") or headline, max_words=10, max_chars=80),
            "left_label": truncate(str(item.get("left_label") or "Before"), 28),
            "right_label": truncate(str(item.get("right_label") or "After"), 28),
            "left_detail": _polish_visual_copy(item.get("left_detail") or card["sentence_text"], max_words=6, max_chars=42),
            "right_detail": _polish_visual_copy(item.get("right_detail") or card["context_text"], max_words=6, max_chars=42),
            "footer_text": footer_text,
            "style_pack": style_pack,
            "theme": _theme_for_card(card, style_pack),
            "renderer_hint": renderer_hint,
            "motion_preset": motion_preset,
            "background_motif": background_motif,
            "layout_variant": layout_variant,
            "rationale": truncate(str(item.get("rationale") or "Generated visual aligned to the active spoken beat."), 160),
            "confidence": round(confidence, 2),
            "importance": round(min(max(card["priority"] / 92.0, 0.25), 1.0), 2),
            "evidence": {
                "pause_before": card["pause_before"],
                "pause_after": card["pause_after"],
                "scene_distance": card["scene_distance"],
                "replace_safety": card["replace_safety"],
                "word_count": card["word_count"],
                "words_per_second": card["words_per_second"],
                "sentence_numeric_hits": card["sentence_numeric_hits"],
                "numeric_hits": card["numeric_hits"],
                "sentence_process_cues": card["sentence_process_cues"],
                "visualizability": card["visualizability"],
                "generic_penalty": card["generic_penalty"],
                "sentence_contrast_cues": card["sentence_contrast_cues"],
                "process_cues": card["process_cues"],
                "contrast_cues": card["contrast_cues"],
            },
        }
        if template in {"keyword_stack", "ribbon_quote"} and not keywords:
            spec["keywords"] = card["keywords"][:4] or [headline]
        if template in {"timeline_steps", "system_flow", "kinetic_route", "signal_network"} and not steps:
            spec["steps"] = _steps_for_card(card) or ([headline, emphasis_text, footer_text[:28]] if footer_text else [headline, emphasis_text])[:4]
        if template in {"metric_callout", "stat_grid", "data_journey"} and not supporting_lines:
            spec["supporting_lines"] = _supporting_lines_for_card(card)[:3]
        if short_slot:
            spec["supporting_lines"] = list(spec.get("supporting_lines") or [])[:2]
            spec["steps"] = list(spec.get("steps") or [])[:3]
            spec["quote_text"] = _polish_visual_copy(spec.get("quote_text") or headline, max_words=8, max_chars=58)
        if template in {"comparison_split", "spotlight_compare"}:
            left_label, right_label, left_detail, right_detail = _comparison_terms_for_card(card)
            spec["left_label"] = truncate(str(item.get("left_label") or left_label), 28)
            spec["right_label"] = truncate(str(item.get("right_label") or right_label), 28)
            spec["left_detail"] = truncate(str(item.get("left_detail") or left_detail), 72)
            spec["right_detail"] = truncate(str(item.get("right_detail") or right_detail), 72)
        normalized.append(spec)
        template_counts[template] = template_counts.get(template, 0) + 1
        last_end = end_sec
        if len(normalized) >= max_visuals:
            break
    return normalized


def _resequence_visual_ids(plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    resequenced: list[dict[str, Any]] = []
    for index, item in enumerate(plan, start=1):
        normalized = dict(item)
        normalized["visual_id"] = f"visual_{index:03d}"
        resequenced.append(normalized)
    return resequenced


def _candidate_pool(cards: list[dict[str, Any]], max_visuals: int) -> list[dict[str, Any]]:
    ranked = sorted(cards, key=lambda item: (item["priority"], -item["start"]), reverse=True)
    pool: list[dict[str, Any]] = []
    for card in ranked:
        visualizability = float(card.get("visualizability") or 0.0)
        numeric_hits = int(card.get("numeric_hits") or 0)
        process_cues = float(card.get("process_cues") or 0.0)
        contrast_cues = float(card.get("contrast_cues") or 0.0)
        generic_penalty = float(card.get("generic_penalty") or 0.0)
        if visualizability < 0.42 and numeric_hits == 0 and process_cues < 0.25 and contrast_cues < 0.25:
            continue
        if generic_penalty > 0.74 and visualizability < 0.55:
            continue
        if any(abs(card["start"] - existing["start"]) < 0.6 for existing in pool):
            continue
        pool.append(card)
        if len(pool) >= max(max_visuals * 3, 10):
            break
    return pool


def _should_use_fast_plan(candidate_cards: list[dict[str, Any]], max_visuals: int) -> bool:
    if not candidate_cards:
        return True
    head = candidate_cards[: max(max_visuals * 2, 4)]
    strong = 0
    ambiguous = 0
    for card in head:
        visualizability = float(card.get("visualizability") or 0.0)
        generic_penalty = float(card.get("generic_penalty") or 0.0)
        explicit_signal = (
            int(card.get("numeric_hits") or 0) > 0
            or float(card.get("process_cues") or 0.0) >= 0.42
            or float(card.get("contrast_cues") or 0.0) >= 0.4
        )
        if explicit_signal and visualizability >= 0.68 and generic_penalty <= 0.36:
            strong += 1
        if visualizability < 0.55 or generic_penalty > 0.58:
            ambiguous += 1
    needed = min(max_visuals, 3)
    return strong >= max(1, needed) and ambiguous <= 1


def _should_run_critic(plan: list[dict[str, Any]]) -> bool:
    if len(plan) >= 3:
        return True
    replace_count = sum(1 for item in plan if str(item.get("composition_mode") or "") == "replace")
    if replace_count >= 2:
        return True
    avg_confidence = sum(float(item.get("confidence") or 0.0) for item in plan) / max(len(plan), 1)
    if avg_confidence < 0.72:
        return True
    templates = [str(item.get("template") or "") for item in plan]
    if len(set(templates)) < len(templates):
        return True
    if any(template in {"quote_focus", "keyword_stack", "ribbon_quote"} for template in templates):
        return True
    headlines = [str(item.get("headline") or "").strip().lower() for item in plan if str(item.get("headline") or "").strip()]
    return len(set(headlines)) < len(headlines)


def fallback_visual_plan(
    cards: list[dict[str, Any]],
    clip_duration: float,
    max_visuals: int,
    min_visual_sec: float,
    max_visual_sec: float,
    scene_cuts: list[float],
    available_renderers: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    ranked = _candidate_pool(cards, max_visuals)
    fallback = []
    for card in ranked:
        template = _default_template(card)
        fallback.append(
            {
                "card_id": card["card_id"],
                "template": template,
                "renderer_hint": card["suggested_renderer"],
                "style_pack": card["style_pack"],
                "composition_mode": card["suggested_composition"],
                "eyebrow": _eyebrow_for_card(card, template),
                "headline": _headline_from_card(card),
                "deck": _deck_for_card(card, _headline_from_card(card)),
                "emphasis_text": _extract_emphasis_text(card),
                "supporting_lines": _supporting_lines_for_card(card),
                "keywords": card["keywords"][:4],
                "steps": _steps_for_card(card),
                "quote_text": truncate(card["sentence_text"], 120),
                "footer_text": _deck_for_card(card, _headline_from_card(card)),
                "left_label": _comparison_terms_for_card(card)[0],
                "right_label": _comparison_terms_for_card(card)[1],
                "left_detail": _comparison_terms_for_card(card)[2],
                "right_detail": _comparison_terms_for_card(card)[3],
                "position": "bottom_right",
                "scale": 0.42,
                "motion_preset": _default_motion_preset(card, template),
                "background_motif": _background_motif(card, template, str(card.get("style_pack") or "editorial_clean")),
                "layout_variant": LAYOUT_VARIANTS.get(template, "hero_split"),
                "rationale": "Fallback visual chosen from the strongest transcript beat when the model plan was unavailable.",
                "confidence": round(min(max(card["priority"] / 88.0, 0.45), 0.9), 2),
            }
        )
        if len(fallback) >= max_visuals:
            break
    return _resequence_visual_ids(
        _normalize_visual_plan(
            fallback,
            cards,
            clip_duration,
            max_visuals,
            min_visual_sec,
            max_visual_sec,
            scene_cuts,
            available_renderers,
        )
    )


def _backfill_plan_with_fallback(
    primary: list[dict[str, Any]],
    fallback: list[dict[str, Any]],
    *,
    max_visuals: int,
) -> list[dict[str, Any]]:
    merged = list(primary)
    seen_card_ids = {str(item.get("card_id") or "") for item in merged}
    for candidate in fallback:
        if len(merged) >= max_visuals:
            break
        card_id = str(candidate.get("card_id") or "")
        if card_id and card_id in seen_card_ids:
            continue
        start_sec = float(candidate.get("start") or 0.0)
        end_sec = float(candidate.get("end") or start_sec)
        if any(
            abs(start_sec - float(existing.get("start") or 0.0)) < 0.35
            or not (end_sec <= float(existing.get("start") or 0.0) - 0.12 or start_sec >= float(existing.get("end") or 0.0) + 0.12)
            for existing in merged
        ):
            continue
        merged.append(candidate)
        if card_id:
            seen_card_ids.add(card_id)
    merged = sorted(merged, key=lambda item: float(item.get("start") or 0.0))[:max_visuals]
    return _resequence_visual_ids(merged)


def analyze_visual_plan_with_llm(
    provider_name: str,
    model_name: str,
    cards: list[dict[str, Any]],
    clip_duration: float,
    max_visuals: int,
    min_visual_sec: float,
    max_visual_sec: float,
    scene_cuts: list[float],
    *,
    available_renderers: list[dict[str, Any]] | None = None,
    avoid_card_ids: set[str] | None = None,
    disable_fast_plan: bool = False,
) -> list[dict[str, Any]]:
    avoid_card_ids = {str(card_id).strip() for card_id in (avoid_card_ids or set()) if str(card_id).strip()}
    fallback = fallback_visual_plan(
        cards,
        clip_duration,
        max_visuals,
        min_visual_sec,
        max_visual_sec,
        scene_cuts,
        available_renderers,
    )
    if not cards:
        return fallback

    candidate_cards = _candidate_pool(cards, max_visuals)
    if not disable_fast_plan and _should_use_fast_plan(candidate_cards, max_visuals):
        return fallback
    template_lines = "\n".join(f"- {name}: {description}" for name, description in SUPPORTED_TEMPLATES.items())
    renderer_lines = _format_renderer_capabilities(available_renderers)
    avoid_card_line = (
        "Previously used card_ids to avoid unless absolutely necessary:\n"
        f"{', '.join(sorted(avoid_card_ids))}\n\n"
        if avoid_card_ids
        else ""
    )
    system_prompt = (
        "You are a senior motion graphics director planning precise generated visuals for an explainer video. "
        "Choose only transcript beats where a custom animation would make the spoken idea clearer. "
        "Prefer concise, literal, high-signal visuals with strong editorial taste. "
        "Do not create generic motivational cards. If a beat is vague or low-signal, skip it. "
        "Distill the copy. Do not simply repeat the spoken sentence as the headline. "
        "Use the evidence fields to decide when a full-screen replacement is safe versus when picture-in-picture is safer. "
        "Return ONLY a JSON array with at most {count} objects using these keys: "
        "card_id, template, renderer_hint, style_pack, composition_mode, eyebrow, headline, deck, emphasis_text, supporting_lines, "
        "steps, keywords, quote_text, left_label, right_label, left_detail, right_detail, footer_text, position, scale, "
        "motion_preset, background_motif, layout_variant, rationale, confidence."
    ).format(count=max_visuals)
    user_prompt = (
        f"Video duration: {clip_duration:.2f}s\n"
        f"Max visuals: {max_visuals}\n"
        f"Duration per visual: {min_visual_sec:.1f}s to {max_visual_sec:.1f}s\n"
        f"Detected scene cuts: {scene_cuts[:24]}\n\n"
        f"Supported templates:\n{template_lines}\n\n"
        "Available renderers:\n"
        f"{renderer_lines}\n\n"
        "Composition modes:\n"
        "- replace: full-screen generated cutaway\n"
        "- picture_in_picture: keep the source visible and place the visual in a corner\n\n"
        "Style packs:\n"
        "- editorial_clean\n- bold_tech\n- documentary_kinetic\n- product_ui\n- cinematic_night\n- signal_lab\n- magazine_luxe\n\n"
        f"{avoid_card_line}"
        f"Candidate transcript cards:\n{truncate(_format_cards_for_llm(candidate_cards), 8200)}\n\n"
        "Pick the strongest beats only. Avoid generic filler. "
        "Favor data_journey for quantitative replace beats, signal_network or kinetic_route for process beats, spotlight_compare for contrasts, interface_cascade for UI/product beats, and ribbon_quote only when the line is truly memorable. "
        "Use the older editorial templates mainly for picture-in-picture or lightweight overlays, not for premium full-screen generated visuals. "
        "Prefer manim for premium custom diagrams, motion systems, and custom replace visuals that can exploit the full Manim library, ffmpeg for simple clean picture-in-picture cards, and blender only for cinematic synthetic shots when available. "
        "Headlines should usually be 2 to 6 words, decks should be a short secondary line, and supporting lines should carry factual detail rather than generic hype. "
        "Return JSON array only."
    )
    try:
        director_raw = call_reasoning_model(provider_name, model_name, system_prompt, user_prompt)
        director_plan = json.loads(extract_json_array(director_raw))
    except Exception:
        return fallback
    normalized_director = _normalize_visual_plan(
        director_plan,
        cards,
        clip_duration,
        max_visuals,
        min_visual_sec,
        max_visual_sec,
        scene_cuts,
        available_renderers,
    )
    if not normalized_director:
        return fallback
    normalized_director = _backfill_plan_with_fallback(normalized_director, fallback, max_visuals=max_visuals)
    if not _should_run_critic(normalized_director):
        return _resequence_visual_ids(normalized_director or fallback)

    critic_system_prompt = (
        "You are a strict motion-design critic and QA lead. "
        "Your job is to reject generic, weak, repetitive, or mistimed visuals and return a tighter plan. "
        "Preserve only high-signal visuals, keep them concise, and fix backend/style/template choices when needed. "
        "Reject any card whose headline just repeats the narration or whose visual treatment feels bland. "
        "Return ONLY a JSON array with the same schema as the director."
    )
    critic_user_prompt = (
        "Renderer capabilities:\n"
        f"{renderer_lines}\n\n"
        f"{avoid_card_line}"
        "Original candidate cards:\n"
        f"{truncate(_format_cards_for_llm(candidate_cards), 6200)}\n\n"
        "Director plan to critique:\n"
        f"{json.dumps(normalized_director, indent=2)}\n\n"
        "Remove generic filler, reduce repetition, and make sure replacement shots only happen when the evidence looks safe. "
        "Return JSON array only."
    )
    try:
        critic_raw = call_reasoning_model(provider_name, model_name, critic_system_prompt, critic_user_prompt)
        critic_plan = json.loads(extract_json_array(critic_raw))
        normalized_critic = _normalize_visual_plan(
            critic_plan,
            cards,
            clip_duration,
            max_visuals,
            min_visual_sec,
            max_visual_sec,
            scene_cuts,
            available_renderers,
        )
        if normalized_critic:
            normalized_critic = _backfill_plan_with_fallback(normalized_critic, fallback, max_visuals=max_visuals)
        return _resequence_visual_ids(normalized_critic or normalized_director or fallback)
    except Exception:
        return _resequence_visual_ids(normalized_director or fallback)
