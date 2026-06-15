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
from visual_program import visual_program_prompt_block

SUPPORTED_TEMPLATES = {
    "data_journey": "A premium quantitative reveal with moving data, guided focus, and visual momentum.",
    "signal_network": "A premium process or system map with directional flow and network choreography.",
    "kinetic_route": "A premium timeline or journey beat staged along a route or guided path.",
    "spotlight_compare": "A premium contrast scene that morphs or spotlights the meaningful difference.",
    "interface_cascade": "A premium interface walkthrough with layered depth and focused camera attention.",
    "ribbon_quote": "A premium line or concept staged with kinetic type and directional motion.",
    "causal_chain": "A premium cause-and-effect sequence that makes the mechanism behind a claim legible.",
    "flywheel_loop": "A premium repeated-loop visual for feedback cycles, iteration, compounding, or habits.",
    "decision_matrix": "A premium decision scene that compares criteria, tradeoffs, and the better path.",
    "anatomy_cutaway": "A premium layered breakdown that reveals the parts inside a system, product, or idea.",
    "stack_ranking": "A premium ranked-priority scene for ordered takeaways, tiers, or weighted factors.",
    "contrast_ladder": "A premium before-to-after ladder that shows progressive improvement or escalation.",
    "proof_sequence": "A premium evidence chain that turns multiple proof points into one cumulative argument.",
    "narrative_arc": "A premium setup-conflict-payoff arc for story-driven explanation beats.",
    "concept_map": "A premium concept map that turns a broad idea into connected concrete anchors.",
    "problem_solution": "A premium problem-to-solution scene with a visible editorial pivot.",
    "myth_buster": "A premium misconception scene that separates the false belief from the useful truth.",
    "checklist_reveal": "A premium sequential checklist for practical criteria, requirements, or next actions.",
    "risk_radar": "A premium radar scan for threats, blind spots, risks, or attention areas.",
    "opportunity_map": "A premium opportunity field that shows where the highest-leverage path sits.",
    "scorecard": "A premium scorecard for grading tradeoffs, readiness, or quality dimensions.",
    "pipeline_xray": "A premium X-ray view through a workflow, stack, or hidden production pipeline.",
    "decision_tree": "A premium branching choice visual for if/then decisions and routing logic.",
    "momentum_wave": "A premium momentum wave for growth, acceleration, trend, or compounding beats.",
    "focus_ring": "A premium focus-ring visual for attention, priority, or narrowing from noise to signal.",
    "timeline_filmstrip": "A premium filmstrip timeline for sequential moments, phases, or narrative beats.",
    "quote_breakdown": "A premium quote breakdown that decomposes a memorable line into its moving parts.",
    "market_map": "A premium landscape map for categories, audiences, channels, or competitive spaces.",
    "mechanism_blueprint": "A premium blueprint view for how a mechanism, product, or system works.",
    "data_pulse": "A premium pulsing metric signal for live data, spikes, thresholds, or feedback.",
    "metric_callout": "Large value or strong claim with supporting context.",
    "keyword_stack": "Stacked short concepts with strong editorial styling.",
    "timeline_steps": "A short process or sequence laid out step by step.",
    "comparison_split": "Side-by-side contrast between two states.",
    "quote_focus": "A clean emphasis shot for a memorable phrase or line.",
    "system_flow": "A connected flow of stages or ideas.",
    "stat_grid": "A compact dashboard of 2-4 supporting stats or takeaways.",
    "three_d_title": "A deterministic Blender 3D title or concept reveal with extruded text and camera motion.",
    "floating_3d_label": "A transparent Blender overlay label/card for calling out a live shot without replacing it.",
    "object_orbit": "A Blender 3D object or placeholder orbit shot, optionally using a local model asset.",
    "logo_reveal": "A Blender 3D text/logo-style reveal for branded or chapter-intro beats.",
    "screen_pointer_3d": "A transparent Blender 3D arrow/pointer callout for pointing at on-screen UI or charts.",
    "data_tunnel": "A full-screen Blender abstract 3D data tunnel for AI, chips, networks, or data-flow ideas.",
    "product_model_spin": "A Blender turntable spin for a local product model asset or a clean placeholder object.",
}

BLENDER_3D_TEMPLATES = {
    "three_d_title",
    "floating_3d_label",
    "object_orbit",
    "logo_reveal",
    "screen_pointer_3d",
    "data_tunnel",
    "product_model_spin",
}
BLENDER_OVERLAY_TEMPLATES = {
    "floating_3d_label",
    "screen_pointer_3d",
    "product_model_spin",
}
BLENDER_TEMPLATE_DOWNGRADES = {
    "three_d_title": "ribbon_quote",
    "floating_3d_label": "quote_focus",
    "object_orbit": "interface_cascade",
    "logo_reveal": "ribbon_quote",
    "screen_pointer_3d": "quote_focus",
    "data_tunnel": "data_journey",
    "product_model_spin": "interface_cascade",
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
    "data_graphic": "hyperframes",
    "product_ui": "hyperframes",
    "process": "hyperframes",
    "abstract_motion": "hyperframes",
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
NEGATIVE_STATE_MARKERS = {
    "not", "never", "stuck", "forgot", "forget", "passive", "consume", "consuming",
    "watch", "watching", "tutorial", "tutorials", "read", "reading", "notes", "waste",
    "wrong", "problem", "without", "doesn't", "dont", "don't",
}
POSITIVE_STATE_MARKERS = {
    "build", "building", "ship", "shipping", "practice", "practicing", "start",
    "starting", "directly", "active", "learn", "learning", "retain", "retention",
    "apply", "applying", "feedback", "improve", "making", "create", "creating",
}
CAUSE_MARKERS = {
    "because", "when", "if", "so", "since", "after", "then", "consume", "watch", "read", "notes",
}
EFFECT_MARKERS = {
    "stuck", "forgot", "forget", "slow", "confused", "nothing", "doesn't", "dont", "don't",
}
FILLER_LEAD_WORDS = {
    "the", "a", "an", "this", "that", "these", "those", "we", "you", "it", "they", "our",
    "your", "their", "to", "for", "with", "by", "in", "on", "of", "but", "so", "because",
}
TRAILING_TRIM_WORDS = {
    "with", "by", "to", "for", "and", "or", "of", "in", "on", "a", "an", "the", "then", "next", "finally",
    "they", "them", "it", "its", "we", "you", "he", "she", "their", "our", "your",
    "is", "are", "was", "were", "be", "being", "been", "have", "has", "had", "do", "does", "did",
}
DISTILL_WORD_PATTERN = re.compile(r"[A-Za-z0-9%+.-]+(?:'[A-Za-z0-9%+.-]+)*")
BACKGROUND_MOTIFS = ("grid", "rings", "beams", "constellation", "bands")
PLAN_CACHE_VERSION = "2026-05-10-hyperframes-v1"
MIN_PREMIUM_REPLACE_DURATION_SEC = 2.4
METRIC_NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9.])(?P<value>\d+(?:\.\d+)?)(?:\s*(?P<unit>%|x|percent|thousand|million|billion|tokens?|parameters?|cache))?(?![A-Za-z0-9.])",
    flags=re.IGNORECASE,
)
METRIC_TEMPLATES = {
    "data_journey",
    "metric_callout",
    "stat_grid",
    "proof_sequence",
    "data_pulse",
    "momentum_wave",
}
LOW_SIGNAL_COPY_PATTERNS = (
    r"\bwe(?:'ll| will)\s+talk\s+about\b",
    r"\bin\s+the\s+video\s+later\b",
    r"\bthis\s+is\s+(?:one\s+)?(?:interesting|important)\b",
    r"\blet'?s\s+talk\s+about\b",
    r"\bwe(?:'re| are)\s+going\s+to\s+talk\b",
    r"\bmore\s+on\s+this\s+later\b",
    r"\bso\s+basically\b",
    r"^did\s+it\s+not\b",
)
LAYOUT_VARIANTS = {
    "data_journey": "arc_stage",
    "signal_network": "network_sweep",
    "kinetic_route": "route_curve",
    "spotlight_compare": "spotlight_stage",
    "interface_cascade": "cascade_focus",
    "ribbon_quote": "ribbon_sweep",
    "causal_chain": "cause_effect_trace",
    "flywheel_loop": "loop_orbit",
    "decision_matrix": "criteria_grid",
    "anatomy_cutaway": "layered_cutaway",
    "stack_ranking": "priority_stack",
    "contrast_ladder": "before_after_ladder",
    "proof_sequence": "evidence_chain",
    "narrative_arc": "story_arc",
    "concept_map": "concept_constellation",
    "problem_solution": "pivot_cards",
    "myth_buster": "truth_split",
    "checklist_reveal": "criteria_checklist",
    "risk_radar": "radar_scan",
    "opportunity_map": "opportunity_field",
    "scorecard": "quality_scorecard",
    "pipeline_xray": "pipeline_xray",
    "decision_tree": "branching_tree",
    "momentum_wave": "momentum_wave",
    "focus_ring": "focus_ring",
    "timeline_filmstrip": "filmstrip_timeline",
    "quote_breakdown": "quote_deconstruction",
    "market_map": "landscape_map",
    "mechanism_blueprint": "mechanism_blueprint",
    "data_pulse": "data_pulse",
    "metric_callout": "hero_split",
    "keyword_stack": "stagger_stack",
    "timeline_steps": "elevated_timeline",
    "comparison_split": "offset_split",
    "quote_focus": "editorial_stage",
    "system_flow": "signal_chain",
    "stat_grid": "dashboard_mosaic",
    "three_d_title": "cinematic_title",
    "floating_3d_label": "floating_overlay",
    "object_orbit": "orbit_stage",
    "logo_reveal": "logo_reveal",
    "screen_pointer_3d": "pointer_overlay",
    "data_tunnel": "tunnel_stage",
    "product_model_spin": "turntable_stage",
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

PREMIUM_FULLSCREEN_TEMPLATES = set(PREMIUM_TEMPLATE_UPGRADES.values()) | {
    "causal_chain",
    "flywheel_loop",
    "decision_matrix",
    "anatomy_cutaway",
    "stack_ranking",
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

EDITORIAL_TEMPLATE_DOWNGRADES = {
    "data_journey": "metric_callout",
    "signal_network": "timeline_steps",
    "kinetic_route": "timeline_steps",
    "spotlight_compare": "comparison_split",
    "interface_cascade": "comparison_split",
    "ribbon_quote": "quote_focus",
    "causal_chain": "system_flow",
    "flywheel_loop": "system_flow",
    "decision_matrix": "comparison_split",
    "anatomy_cutaway": "system_flow",
    "stack_ranking": "stat_grid",
    "contrast_ladder": "comparison_split",
    "proof_sequence": "stat_grid",
    "narrative_arc": "timeline_steps",
    "concept_map": "system_flow",
    "problem_solution": "comparison_split",
    "myth_buster": "comparison_split",
    "checklist_reveal": "timeline_steps",
    "risk_radar": "stat_grid",
    "opportunity_map": "system_flow",
    "scorecard": "stat_grid",
    "pipeline_xray": "system_flow",
    "decision_tree": "timeline_steps",
    "momentum_wave": "metric_callout",
    "focus_ring": "quote_focus",
    "timeline_filmstrip": "timeline_steps",
    "quote_breakdown": "quote_focus",
    "market_map": "system_flow",
    "mechanism_blueprint": "system_flow",
    "data_pulse": "metric_callout",
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


def _protect_numeric_periods(text: str) -> str:
    return re.sub(r"(?<=\d)\.(?=\d)", "<<DECIMAL_DOT>>", str(text or ""))


def _restore_numeric_periods(text: str) -> str:
    return str(text or "").replace("<<DECIMAL_DOT>>", ".")


def _metric_matches(text: str) -> list[re.Match[str]]:
    return list(METRIC_NUMBER_PATTERN.finditer(str(text or "")))


def _format_metric_value(match: re.Match[str]) -> str:
    value = match.group("value")
    unit = (match.group("unit") or "").strip()
    if not unit:
        return value
    return f"{value}{unit}" if unit.lower() in {"%", "x"} else f"{value} {unit}"


def _extract_metric_facts(text: str, *, limit: int = 4) -> list[dict[str, str]]:
    facts: list[dict[str, str]] = []
    seen: set[str] = set()
    fragments = _split_fragments(text, limit=10)
    for match in _metric_matches(text):
        value = _format_metric_value(match)
        key = value.lower()
        if key in seen:
            continue
        fragment = next(
            (
                item
                for item in fragments
                if match.group(0).replace(" ", "") in item.replace(" ", "")
                or value in item.replace(" ", "")
            ),
            str(text or ""),
        )
        label = _display_case(_distill_phrase(fragment, max_words=8, max_chars=64))
        facts.append({"value": value, "label": label or value})
        seen.add(key)
        if len(facts) >= limit:
            break
    return facts


def _has_terminal_punctuation(text: str) -> bool:
    return bool(re.search(r"[.!?]\s*$", str(text or "").strip()))


def _starts_like_continuation(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    first = value[0]
    if first.islower():
        return True
    return bool(
        re.match(
            r"^(?:cache|compared|required|requires|than|of|by|for|to|and|or|but|which|that|it)\b",
            value,
            flags=re.IGNORECASE,
        )
    )


def _should_merge_sentence_with_next(current: str, next_text: str) -> bool:
    if not str(current or "").strip() or not str(next_text or "").strip():
        return False
    if _is_low_signal_copy(current) and not _metric_matches(current):
        return False
    if not _has_terminal_punctuation(current) and _starts_like_continuation(next_text):
        return True
    if _metric_matches(current) and _starts_like_continuation(next_text):
        return True
    if re.search(r"\?\s*$", current) and _metric_matches(current) and _metric_matches(next_text):
        return True
    return False


def _is_low_signal_copy(text: Any) -> bool:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if not value:
        return True
    lowered = value.lower()
    if any(re.search(pattern, lowered) for pattern in LOW_SIGNAL_COPY_PATTERNS):
        return True
    tokens = _tokens(value)
    if not tokens:
        return True
    if len(tokens) <= 1 and not _metric_matches(value):
        return True
    filler_count = sum(1 for token in tokens if token in FILLER_LEAD_WORDS or token in TRAILING_TRIM_WORDS)
    if filler_count == len(tokens) and not _metric_matches(value):
        return True
    if re.fullmatch(r"\d+(?:\.\d+)?", value) and not re.search(r"%|x", value, flags=re.IGNORECASE):
        return True
    return False


def _is_bridge_sentence(text: str) -> bool:
    value = str(text or "").strip().lower()
    if not value:
        return True
    if any(re.search(pattern, value) for pattern in LOW_SIGNAL_COPY_PATTERNS):
        return True
    if re.search(r"\b(?:later|next|in a second|in a minute)\b", value) and not _metric_matches(value):
        return True
    return False


def _canonical_claim_key(text: str, metric_facts: list[dict[str, str]]) -> str:
    lowered = str(text or "").lower()
    metric_values = [str(item.get("value") or "").lower() for item in metric_facts if item.get("value")]
    if metric_values:
        anchors: list[str] = []
        if re.search(r"\bkv\b|\bcache\b", lowered):
            anchors.append("kv_cache")
        if re.search(r"\bdeep\s*seek\b|\bdeepseek\b", lowered):
            anchors.append("deepseek")
        if re.search(r"\bv\d+(?:\.\d+)?\b", lowered):
            anchors.append("version_compare")
        if anchors:
            return "metric:" + ":".join(sorted(set(metric_values + anchors)))
    keywords = semantic_keywords(text, limit=5)
    return "claim:" + "_".join(keywords[:4])


def _proper_noun_count(text: str) -> int:
    return len(re.findall(r"\b[A-Z][a-zA-Z0-9]+\b", str(text or "")))


def _split_fragments(text: str, *, limit: int = 6) -> list[str]:
    protected = _protect_numeric_periods(text)
    raw = re.split(
        r"(?:[.;:!?]|\b(?:and then|then|next|finally|because|so|while|but|instead)\b|,)",
        protected,
        flags=re.IGNORECASE,
    )
    fragments: list[str] = []
    for part in raw:
        cleaned = _restore_numeric_periods(re.sub(r"\s+", " ", part).strip(" -,\n\t"))
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in FILLER_LEAD_WORDS:
            continue
        if _is_low_signal_copy(cleaned):
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


def _limit_copy_words(text: str, *, max_words: int, max_chars: int) -> str:
    tokens = str(text or "").split()
    if not tokens:
        return ""
    clipped = " ".join(tokens[: max(max_words, 1)])
    if len(clipped) > max_chars:
        clipped = truncate(clipped, max_chars)
    return clipped.strip()


def _polish_visual_copy(text: Any, *, max_words: int, max_chars: int) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    polished = _display_case(_distill_phrase(value, max_words=max_words, max_chars=max_chars))
    return _display_case(_limit_copy_words(polished, max_words=max_words, max_chars=max_chars))


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
    metric_facts = list(card.get("metric_facts") or []) or _extract_metric_facts(sentence)
    if metric_facts:
        value = str(metric_facts[0].get("value") or "").strip()
        combined = f"{sentence} {card.get('context_text', '')}".lower()
        if value and re.search(r"\bkv\b|\bcache\b", combined):
            return f"{value} KV cache"
        if value and re.search(r"\b(?:latency|speed|memory|cost|tokens?|throughput|accuracy|quality)\b", combined):
            noun = next(
                (
                    token
                    for token in ("latency", "speed", "memory", "cost", "tokens", "throughput", "accuracy", "quality")
                    if re.search(rf"\b{token}\b", combined)
                ),
                "signal",
            )
            if re.search(rf"\b{re.escape(noun)}\b", value, flags=re.IGNORECASE):
                return _display_case(value)
            return _display_case(f"{value} {noun}")
        if value:
            for fragment in fragments:
                if value.replace(" ", "") in fragment.replace(" ", ""):
                    return _display_case(_distill_phrase(fragment, max_words=6, max_chars=40))
            return _display_case(_distill_phrase(f"{value} {sentence}", max_words=6, max_chars=40))
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
        if not trimmed or _is_low_signal_copy(trimmed) or lowered == headline or lowered in lines:
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
    steps = [
        _display_case(_distill_phrase(fragment, max_words=5, max_chars=28))
        for fragment in fragments
        if fragment and not _is_low_signal_copy(fragment)
    ]
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


def _marker_hits(text: str, markers: set[str]) -> int:
    lowered = str(text or "").lower()
    return sum(1 for marker in markers if re.search(rf"\b{re.escape(marker)}\b", lowered))


def _pick_marked_fragment(
    fragments: list[str],
    *,
    preferred_markers: set[str],
    avoid_texts: set[str] | None = None,
    max_words: int = 6,
    max_chars: int = 40,
) -> str:
    avoid_texts = {item.lower() for item in (avoid_texts or set()) if item}
    ranked = sorted(
        fragments,
        key=lambda fragment: (
            _marker_hits(fragment, preferred_markers),
            len(_tokens(fragment)),
        ),
        reverse=True,
    )
    for fragment in ranked:
        cleaned = _polish_visual_copy(fragment, max_words=max_words, max_chars=max_chars)
        lowered = cleaned.lower()
        if not cleaned or lowered in avoid_texts:
            continue
        if _marker_hits(fragment, preferred_markers) <= 0:
            continue
        return cleaned
    return ""


def _derive_semantic_frame(
    *,
    sentence_text: str,
    context_text: str,
    previous_text: str,
    next_text: str,
    visual_type_hint: str,
    numeric_hits: int,
    process_cues: float,
    contrast_cues: float,
) -> dict[str, Any]:
    story_window = truncate(" ".join(item for item in [previous_text, sentence_text, next_text] if item).strip(), 320)
    full_context = ". ".join(item for item in [previous_text, sentence_text, next_text, context_text] if item).strip()
    fragments = _split_fragments(full_context, limit=10)
    headline_seed = _display_case(_distill_phrase(sentence_text or context_text, max_words=6, max_chars=42))
    before_state = _pick_marked_fragment(
        fragments,
        preferred_markers=NEGATIVE_STATE_MARKERS,
        max_words=5,
        max_chars=34,
    )
    after_state = _pick_marked_fragment(
        fragments,
        preferred_markers=POSITIVE_STATE_MARKERS,
        avoid_texts={before_state},
        max_words=5,
        max_chars=34,
    )
    cause = _pick_marked_fragment(
        fragments,
        preferred_markers=CAUSE_MARKERS,
        avoid_texts={before_state, after_state},
        max_words=6,
        max_chars=42,
    )
    effect = _pick_marked_fragment(
        fragments,
        preferred_markers=EFFECT_MARKERS,
        avoid_texts={before_state, after_state, cause},
        max_words=6,
        max_chars=40,
    )
    if not before_state and fragments:
        before_state = _polish_visual_copy(fragments[0], max_words=5, max_chars=34)
    if not after_state:
        fallback_after = next_text or context_text or sentence_text
        after_state = _polish_visual_copy(fallback_after, max_words=5, max_chars=34)
    if not cause:
        cause = _polish_visual_copy(sentence_text or context_text, max_words=6, max_chars=42)
    if not effect:
        effect = _polish_visual_copy(context_text or next_text or sentence_text, max_words=6, max_chars=40)

    metric_facts = _extract_metric_facts(sentence_text)
    numeric_signal = bool(metric_facts or numeric_hits >= 2 or (numeric_hits >= 1 and visual_type_hint == "data_graphic"))
    ordinal_or_method = bool(re.search(r"\b(?:method|step|part|chapter|tip)\s+\d+\b", full_context, flags=re.IGNORECASE))
    negative_before = _marker_hits(before_state, NEGATIVE_STATE_MARKERS) > 0
    negative_in_sentence = bool(
        _marker_hits(sentence_text, NEGATIVE_STATE_MARKERS) > 0
        or re.search(r"\bnot\b|don't|doesn't|never\b", sentence_text, flags=re.IGNORECASE)
    )
    positive_after = bool(
        _marker_hits(after_state, POSITIVE_STATE_MARKERS) > 0
        and not re.search(r"\bnot\b|don't|doesn't|never\b", after_state, flags=re.IGNORECASE)
    )
    positive_in_sentence = bool(
        _marker_hits(sentence_text, POSITIVE_STATE_MARKERS) > 0
        and not negative_in_sentence
    )

    explicit_negation = bool(
        re.search(
            r"\bnot by\b|that's not learning|that is not learning|that's not\b|doesn't matter|don't learn|not learning\b",
            full_context,
            flags=re.IGNORECASE,
        )
    )
    if numeric_signal and not ordinal_or_method and not (explicit_negation and contrast_cues >= 0.12):
        intuition_mode = "metric_proof"
    elif before_state and after_state and before_state.lower() != after_state.lower() and negative_before and positive_after:
        intuition_mode = "misconception_flip" if (contrast_cues >= 0.12 or re.search(r"\bbut\b|\binstead\b|don't|doesn't", full_context, flags=re.IGNORECASE)) else "process_route"
    elif contrast_cues >= 0.24 and before_state and after_state and before_state.lower() != after_state.lower():
        intuition_mode = "misconception_flip"
    elif process_cues >= 0.42 and before_state and after_state:
        intuition_mode = "process_route"
    elif _marker_hits(full_context, CAUSE_MARKERS) > 0 and _marker_hits(full_context, EFFECT_MARKERS) > 0:
        intuition_mode = "causal_chain"
    elif visual_type_hint == "product_ui":
        intuition_mode = "interface_walkthrough"
    else:
        intuition_mode = "concept_emphasis"

    if intuition_mode == "metric_proof":
        mental_model = "Ground the spoken claim in concrete evidence the viewer can track."
        viewer_takeaway = headline_seed or after_state
        visual_metaphor = "tracked_metric"
    elif intuition_mode == "misconception_flip":
        mental_model = f"Show why {before_state} fails and why {after_state} works."
        viewer_takeaway = after_state or headline_seed
        visual_metaphor = "state_transition"
    elif intuition_mode == "process_route":
        mental_model = f"Show the journey from {before_state} to {after_state} as a repeatable process."
        viewer_takeaway = after_state or headline_seed
        visual_metaphor = "route_progression"
    elif intuition_mode == "causal_chain":
        mental_model = f"Make the causal link legible: {cause} -> {effect}."
        viewer_takeaway = effect or after_state or headline_seed
        visual_metaphor = "causal_flow"
    elif intuition_mode == "interface_walkthrough":
        mental_model = f"Show the interaction path that leads to {after_state or headline_seed}."
        viewer_takeaway = after_state or headline_seed
        visual_metaphor = "interface_cascade"
    else:
        mental_model = f"Give the abstract idea a concrete visual anchor around {headline_seed or after_state}."
        viewer_takeaway = headline_seed or after_state
        visual_metaphor = "kinetic_focus"

    positive_resolution = bool(
        after_state
        and after_state.lower() != before_state.lower()
        and (
            positive_after
            or _marker_hits(next_text, POSITIVE_STATE_MARKERS) > 0
            or re.search(r"\bstart\b|\bbuild\b|\blearn\b|\bship\b|\btarget(?:ed)?\b", after_state, flags=re.IGNORECASE)
        )
    )
    if intuition_mode in {"process_route", "causal_chain"}:
        intuition_role = "core_mechanism"
    elif intuition_mode == "misconception_flip":
        intuition_role = "core_mechanism" if positive_resolution and positive_in_sentence else "supporting_example"
    elif intuition_mode == "metric_proof":
        intuition_role = "supporting_example" if (explicit_negation or negative_in_sentence) and not positive_in_sentence else "concrete_proof"
    elif intuition_mode == "interface_walkthrough":
        intuition_role = "core_mechanism"
    else:
        intuition_role = "supporting_example"

    payoff = {
        "core_mechanism": 0.88,
        "concrete_proof": 0.74,
        "supporting_example": 0.48,
    }.get(intuition_role, 0.56)
    if intuition_mode == "process_route":
        payoff += 0.08
    elif intuition_mode == "causal_chain":
        payoff += 0.07
    elif intuition_mode == "misconception_flip":
        payoff += 0.04
    elif intuition_mode == "metric_proof":
        payoff += 0.02 if numeric_signal else -0.04
    payoff += min(process_cues, 0.55) * 0.08
    payoff += min(contrast_cues, 0.55) * 0.06
    if explicit_negation and intuition_role == "supporting_example":
        payoff -= 0.14
    if not positive_resolution and intuition_role != "core_mechanism":
        payoff -= 0.06
    if before_state and after_state and before_state.lower() == after_state.lower():
        payoff -= 0.08
    novelty_seed = viewer_takeaway or after_state or headline_seed or sentence_text
    novelty_key = re.sub(r"[^a-z0-9]+", "_", novelty_seed.lower()).strip("_")[:48]

    return {
        "intuition_mode": intuition_mode,
        "intuition_role": intuition_role,
        "intuition_payoff": round(max(0.0, min(payoff, 1.0)), 3),
        "novelty_key": novelty_key,
        "story_window": story_window,
        "before_state": before_state,
        "after_state": after_state,
        "cause": cause,
        "effect": effect,
        "mental_model": truncate(mental_model, 180),
        "viewer_takeaway": truncate(viewer_takeaway, 64),
        "visual_metaphor": visual_metaphor,
    }


def _comparison_terms_for_card(card: dict[str, Any]) -> tuple[str, str, str, str]:
    sentence = str(card.get("sentence_text") or "")
    context = str(card.get("context_text") or "")
    semantic_frame = dict(card.get("semantic_frame") or {})
    before_state = str(semantic_frame.get("before_state") or "").strip()
    after_state = str(semantic_frame.get("after_state") or "").strip()
    if before_state and after_state and before_state.lower() != after_state.lower():
        return (
            "Before",
            "After",
            _polish_visual_copy(before_state, max_words=5, max_chars=34),
            _polish_visual_copy(after_state, max_words=5, max_chars=34),
        )
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
    if template in {"metric_callout", "stat_grid", "data_journey", "proof_sequence", "scorecard", "data_pulse"}:
        return "SIGNAL"
    if template in {"timeline_steps", "system_flow", "kinetic_route", "signal_network", "causal_chain", "flywheel_loop", "pipeline_xray", "mechanism_blueprint", "decision_tree"}:
        return "WORKFLOW"
    if template in {"comparison_split", "spotlight_compare", "contrast_ladder", "problem_solution", "myth_buster"}:
        return "SHIFT"
    if template in {"decision_matrix", "scorecard"}:
        return "DECISION"
    if template in {"anatomy_cutaway", "quote_breakdown", "mechanism_blueprint", "pipeline_xray"}:
        return "BREAKDOWN"
    if template in {"stack_ranking", "checklist_reveal", "focus_ring"}:
        return "PRIORITY"
    if template in {"narrative_arc", "timeline_filmstrip"}:
        return "ARC"
    if template in {"risk_radar", "opportunity_map", "market_map", "concept_map"}:
        return "MAP"
    if template == "momentum_wave":
        return "MOMENTUM"
    if visual_type == "product_ui":
        return "INTERFACE"
    if visual_type == "abstract_motion":
        return "IDEA"
    return "INSIGHT"


def _deck_for_card(card: dict[str, Any], headline: str) -> str:
    semantic_frame = dict(card.get("semantic_frame") or {})
    takeaway = str(semantic_frame.get("viewer_takeaway") or "").strip()
    if takeaway and takeaway.lower() != headline.lower():
        return truncate(takeaway, 46)
    for line in _supporting_lines_for_card(card):
        if line.lower() != headline.lower():
            return truncate(line, 46)
    return ""


def _background_motif(card: dict[str, Any], template: str, style_pack: str) -> str:
    visual_type = str(card.get("visual_type_hint") or "")
    if template in {"timeline_steps", "system_flow", "kinetic_route", "signal_network", "causal_chain", "flywheel_loop", "pipeline_xray", "mechanism_blueprint", "decision_tree"}:
        return "grid"
    if template in {"comparison_split", "spotlight_compare", "decision_matrix", "contrast_ladder", "problem_solution", "myth_buster"}:
        return "bands"
    if template in {"data_journey", "proof_sequence", "stack_ranking", "scorecard", "risk_radar", "data_pulse", "momentum_wave"}:
        return "rings"
    if template in {"interface_cascade", "anatomy_cutaway", "quote_breakdown"}:
        return "beams"
    if template in {"ribbon_quote", "narrative_arc", "concept_map", "opportunity_map", "market_map", "timeline_filmstrip", "focus_ring"}:
        return "constellation"
    if visual_type == "abstract_motion":
        return "rings"
    if style_pack in {"magazine_luxe", "documentary_kinetic"}:
        return "beams"
    return "constellation"


def _count_numbers(text: str) -> int:
    return len(_metric_matches(text))


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
    combined_text = f"{card.get('sentence_text', '')} {card.get('context_text', '')}".lower()
    numbers = int(card["sentence_numeric_hits"]) if "sentence_numeric_hits" in card else int(card.get("numeric_hits") or 0)
    process_cues = float(card["sentence_process_cues"]) if "sentence_process_cues" in card else float(card.get("process_cues") or 0.0)
    contrast_cues = float(card["sentence_contrast_cues"]) if "sentence_contrast_cues" in card else float(card.get("contrast_cues") or 0.0)
    intuition_mode = str((card.get("semantic_frame") or {}).get("intuition_mode") or card.get("intuition_mode") or "").strip().lower()
    if re.search(r"\b(?:myth|misconception|wrong|false|truth|actually|not really)\b", combined_text):
        return "myth_buster"
    if re.search(r"\b(?:problem|pain|bottleneck|stuck|issue)\b", combined_text) and re.search(r"\b(?:solution|fix|solve|instead|better)\b", combined_text):
        return "problem_solution"
    if re.search(r"\b(?:risk|danger|threat|blind spot|failure|warning)\b", combined_text):
        return "risk_radar"
    if re.search(r"\b(?:opportunity|leverage|highest impact|where to|channel|market|audience)\b", combined_text):
        return "opportunity_map"
    if re.search(r"\b(?:score|grade|rating|quality|readiness|criteria)\b", combined_text):
        return "scorecard"
    if re.search(r"\b(?:checklist|requirement|requirements|must|need to|things to)\b", combined_text):
        return "checklist_reveal"
    if re.search(r"\b(?:if|else|branch|route|choose|decision tree)\b", combined_text) and re.search(r"\b(?:then|otherwise|or)\b", combined_text):
        return "decision_tree"
    if re.search(r"\b(?:momentum|accelerate|growth|grow|trend|curve|compound|spike)\b", combined_text):
        return "momentum_wave"
    if re.search(r"\b(?:focus|attention|priority|signal|noise|narrow)\b", combined_text):
        return "focus_ring"
    if re.search(r"\b(?:blueprint|mechanism|how it works|under the hood)\b", combined_text):
        return "mechanism_blueprint"
    if re.search(r"\b(?:x-?ray|pipeline|stack|workflow)\b", combined_text) and re.search(r"\b(?:inside|through|hidden|stage|layer)\b", combined_text):
        return "pipeline_xray"
    if intuition_mode == "metric_proof":
        if re.search(r"\b(?:pulse|spike|threshold|live|feedback)\b", combined_text):
            return "data_pulse"
        return "proof_sequence" if numbers >= 2 else "data_journey"
    if intuition_mode == "misconception_flip":
        if re.search(r"\b(?:choice|choose|option|trade[-\s]?off|decision|criteria)\b", combined_text):
            return "decision_matrix"
        return "contrast_ladder"
    if intuition_mode in {"process_route", "causal_chain"}:
        if intuition_mode == "causal_chain":
            return "causal_chain"
        if re.search(r"\b(?:loop|cycle|feedback|repeat|iterate|compound|flywheel)\b", combined_text):
            return "flywheel_loop"
        return "signal_network" if len(card.get("keywords") or []) >= 3 else "kinetic_route"
    if intuition_mode == "interface_walkthrough":
        return "interface_cascade"
    if re.search(r"\b(?:inside|layer|layers|component|anatomy|breakdown|under the hood)\b", combined_text):
        return "anatomy_cutaway"
    if re.search(r"\b(?:top|rank|ranking|priority|prioritize|first|second|third|tier)\b", combined_text):
        return "stack_ranking"
    if re.search(r"\b(?:story|setup|conflict|payoff|journey|arc)\b", combined_text):
        return "narrative_arc"
    if re.search(r"\b(?:phase|chapter|sequence|timeline|moment|moments)\b", combined_text):
        return "timeline_filmstrip"
    if re.search(r"\b(?:map|landscape|space|category|categories|ecosystem)\b", combined_text):
        return "market_map"
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
    return "hyperframes"


def _default_motion_preset(card: dict[str, Any], template: str) -> str:
    visual_type = str(card.get("visual_type_hint") or "")
    if template in {"timeline_steps", "system_flow", "kinetic_route", "signal_network", "causal_chain", "pipeline_xray", "mechanism_blueprint", "decision_tree"}:
        return "diagram_draw"
    if template == "flywheel_loop":
        return "loop_trace"
    if template in {"metric_callout", "stat_grid", "data_journey", "proof_sequence", "stack_ranking", "scorecard", "data_pulse", "risk_radar"}:
        return "kinetic_pop"
    if template in {"spotlight_compare", "interface_cascade", "decision_matrix", "contrast_ladder", "anatomy_cutaway", "problem_solution", "myth_buster", "focus_ring"}:
        return "focus_shift"
    if template in {"narrative_arc", "timeline_filmstrip"}:
        return "story_sweep"
    if template in {"ribbon_quote", "quote_breakdown"}:
        return "type_sweep"
    if template in {"concept_map", "opportunity_map", "market_map"}:
        return "map_reveal"
    if template == "momentum_wave":
        return "wave_rise"
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
            return "myth_buster" if re.search(r"\b(?:myth|wrong|false|truth)\b", f"{card.get('sentence_text', '')} {card.get('context_text', '')}", flags=re.IGNORECASE) else "contrast_ladder"
        if process_cues >= 0.18:
            return "mechanism_blueprint" if re.search(r"\b(?:mechanism|blueprint|under the hood)\b", f"{card.get('sentence_text', '')} {card.get('context_text', '')}", flags=re.IGNORECASE) else "causal_chain" if contrast_cues >= 0.18 else "signal_network"
        if numeric_hits >= 1:
            return "data_pulse" if re.search(r"\b(?:pulse|spike|threshold|feedback)\b", f"{card.get('sentence_text', '')} {card.get('context_text', '')}", flags=re.IGNORECASE) else "data_journey"
        return "ribbon_quote"
    if template == "comparison_split":
        return "contrast_ladder" if contrast_cues >= 0.34 else "spotlight_compare"
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
    consumed_indexes: set[int] = set()

    for zero_index, sentence in enumerate(sentences):
        index = zero_index + 1
        if zero_index in consumed_indexes:
            continue
        start_sec = max(0.0, min(float(sentence.get("start") or 0.0), clip_duration))
        end_sec = max(start_sec + 0.12, min(float(sentence.get("end") or start_sec + 0.8), clip_duration))
        raw_sentence_text = truncate(str(sentence.get("text") or ""), 220)
        if not raw_sentence_text:
            continue

        merged_indexes = [zero_index]
        claim_text = raw_sentence_text
        next_sentence = sentences[zero_index + 1] if zero_index + 1 < total_sentences else None
        next_sentence_text = truncate(str((next_sentence or {}).get("text") or ""), 220)
        if next_sentence and _should_merge_sentence_with_next(raw_sentence_text, next_sentence_text):
            claim_text = truncate(f"{raw_sentence_text.rstrip()} {next_sentence_text.lstrip()}", 260)
            end_sec = max(
                end_sec,
                min(float(next_sentence.get("end") or end_sec), clip_duration),
            )
            merged_indexes.append(zero_index + 1)
            consumed_indexes.add(zero_index + 1)

        own_metric_count = _count_numbers(claim_text)
        own_process_cues = _process_cue_score(claim_text)
        own_contrast_cues = _contrast_cue_score(claim_text)
        if _is_bridge_sentence(raw_sentence_text) and own_metric_count == 0 and own_process_cues < 0.18 and own_contrast_cues < 0.18:
            continue

        previous_index = max(0, zero_index - 1)
        next_index = min(total_sentences - 1, merged_indexes[-1] + 1)
        previous_text = (
            truncate(str(sentences[previous_index].get("text") or ""), 180)
            if zero_index > 0
            else ""
        )
        next_text = (
            truncate(str(sentences[next_index].get("text") or ""), 180)
            if next_index > merged_indexes[-1]
            else ""
        )
        context_text = truncate(
            window_text(
                transcript_segments,
                max(0.0, start_sec - 3.0),
                min(clip_duration, end_sec + 3.0),
            ),
            320,
        )
        card_words = [
            word
            for word in words
            if float(word.get("end") or 0.0) > start_sec and float(word.get("start") or 0.0) < end_sec
        ] or _card_words(sentence, words)
        pause_before = 0.0
        pause_after = 0.0
        if zero_index > 0:
            prev = sentences[zero_index - 1]
            pause_before = max(0.0, start_sec - float(prev.get("end") or start_sec))
        if merged_indexes[-1] + 1 < total_sentences:
            nxt = sentences[merged_indexes[-1] + 1]
            pause_after = max(0.0, float(nxt.get("start") or end_sec) - end_sec)
        word_count = len(card_words)
        words_per_second = round(word_count / max(end_sec - start_sec, 0.15), 2) if word_count else 0.0
        metric_facts = _extract_metric_facts(claim_text)
        cue_text = f"{claim_text} {previous_text} {next_text}"
        keywords = semantic_keywords(f"{claim_text} {context_text}", limit=8)
        visual_type_hint = infer_visual_type(f"{claim_text} {context_text}")
        nearest_scene_cut, scene_distance = _nearest_scene_distance(start_sec, end_sec, scene_cuts)
        sentence_numeric_hits = _count_numbers(claim_text)
        numeric_hits = sentence_numeric_hits
        sentence_process_cues = _process_cue_score(claim_text)
        process_cues = _process_cue_score(cue_text)
        sentence_contrast_cues = _contrast_cue_score(claim_text)
        contrast_cues = _contrast_cue_score(cue_text)
        generic_penalty = _generic_penalty(f"{claim_text} {context_text}")
        concrete_hits = _concrete_hit_score(f"{claim_text} {context_text}")
        proper_nouns = _proper_noun_count(f"{claim_text} {context_text}")
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
        semantic_frame = _derive_semantic_frame(
            sentence_text=claim_text,
            context_text=context_text,
            previous_text=previous_text,
            next_text=next_text,
            visual_type_hint=visual_type_hint,
            numeric_hits=numeric_hits,
            process_cues=process_cues,
            contrast_cues=contrast_cues,
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
            "sentence_text": claim_text,
            "source_sentence_text": raw_sentence_text,
            "merged_sentence_indexes": [idx + 1 for idx in merged_indexes],
            "context_text": context_text,
            "previous_text": previous_text,
            "next_text": next_text,
            "keywords": keywords,
            "metric_facts": metric_facts,
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
            "semantic_frame": semantic_frame,
            "intuition_mode": semantic_frame.get("intuition_mode", ""),
            "intuition_role": semantic_frame.get("intuition_role", ""),
            "intuition_payoff": float(semantic_frame.get("intuition_payoff") or 0.0),
            "novelty_key": _canonical_claim_key(claim_text, metric_facts) or semantic_frame.get("novelty_key", ""),
            "suggested_composition": suggested_composition,
            "style_pack": style_pack,
            "suggested_renderer": _default_renderer_hint({"visual_type_hint": visual_type_hint}),
        }
        row["priority"] = round(
            _visual_priority(row)
            + (float(row.get("intuition_payoff") or 0.0) - 0.5) * 26
            + (7.0 if str(row.get("intuition_role") or "") == "core_mechanism" else 0.0)
            - (10.0 if str(row.get("intuition_role") or "") == "supporting_example" else 0.0),
            2,
        )
        cards.append(row)
    return cards


def _format_cards_for_llm(cards: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for card in cards:
        source_frame = card.get("source_frame_analysis")
        source_frame = source_frame if isinstance(source_frame, dict) else {}
        raw_warnings = source_frame.get("warnings")
        source_warnings = raw_warnings if isinstance(raw_warnings, list) else []

        def source_value(key: str) -> float:
            try:
                return float(source_frame.get(key) or 0.0)
            except (TypeError, ValueError):
                return 0.0

        lines.append(
            "\n".join(
                [
                    f"{card['card_id']} | {card['start']:.2f}-{card['end']:.2f} | priority={card['priority']:.2f}",
                    (
                        "Opportunity: "
                        f"episode={card.get('semantic_episode_id', '')} "
                        f"scene={(card.get('opportunity_preflight') or {}).get('scene_type', '')} "
                        f"score={(card.get('opportunity_contract') or {}).get('score', '')}"
                    )
                    if card.get("opportunity_contract")
                    else "",
                    f"Sentence: {card['sentence_text']}",
                    f"Prev/Next: {card.get('previous_text', '')} || {card.get('next_text', '')}",
                    f"Context: {card['context_text']}",
                    (
                        "Intuition: "
                        f"mode={card.get('semantic_frame', {}).get('intuition_mode', '')} | "
                        f"role={card.get('semantic_frame', {}).get('intuition_role', '')} | "
                        f"payoff={card.get('semantic_frame', {}).get('intuition_payoff', '')} | "
                        f"before={card.get('semantic_frame', {}).get('before_state', '')} | "
                        f"after={card.get('semantic_frame', {}).get('after_state', '')} | "
                        f"takeaway={card.get('semantic_frame', {}).get('viewer_takeaway', '')}"
                    ),
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
                    (
                        "Source frame: "
                        f"type={source_frame.get('source_type', 'unknown')} | "
                        f"visual_need={source_value('visual_need'):.2f} | "
                        f"source_richness={source_value('source_richness'):.2f} | "
                        f"brightness={source_value('brightness'):.2f} | "
                        f"warnings={', '.join(str(item) for item in source_warnings)}"
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
    metric_facts = list(card.get("metric_facts") or [])
    if metric_facts:
        value = str(metric_facts[0].get("value") or "").strip()
        if value:
            return value
    text = str(card.get("sentence_text") or "")
    number_matches = _metric_matches(text)
    if number_matches:
        return _format_metric_value(number_matches[0])
    keywords = list(card.get("keywords") or [])
    if keywords:
        return " ".join(keywords[:3])
    return truncate(text, 32)


def _coerce_string_list(raw: Any, limit: int, max_chars: int, *, max_words: int | None = None) -> list[str]:
    resolved_max_words = max_words if max_words is not None else (4 if max_chars <= 30 else 7)
    if isinstance(raw, list):
        values = [
            _polish_visual_copy(item, max_words=resolved_max_words, max_chars=max_chars)
            for item in raw
            if str(item).strip()
        ]
    elif str(raw or "").strip():
        values = [_polish_visual_copy(raw, max_words=resolved_max_words, max_chars=max_chars)]
    else:
        values = []
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        lowered = value.lower()
        if not value or _is_low_signal_copy(value) or lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(value)
        if len(deduped) >= limit:
            break
    return deduped


def _promote_composition_for_premium(
    card: dict[str, Any],
    composition_mode: str,
    *,
    prefer_premium: bool,
) -> str:
    if not prefer_premium or composition_mode != "picture_in_picture":
        return composition_mode
    replace_safety = float(card.get("replace_safety") or 0.0)
    visualizability = float(card.get("visualizability") or 0.0)
    numeric_hits = int(card.get("numeric_hits") or 0)
    process_cues = float(card.get("process_cues") or 0.0)
    contrast_cues = float(card.get("contrast_cues") or 0.0)
    generic_penalty = float(card.get("generic_penalty") or 0.0)
    scene_distance = float(card.get("scene_distance") or 0.0)
    explicit_signal = numeric_hits > 0 or process_cues >= 0.34 or contrast_cues >= 0.28
    if replace_safety >= 0.72 and visualizability >= 0.46 and generic_penalty <= 0.56:
        return "replace"
    if replace_safety >= 0.56 and visualizability >= 0.5:
        return "replace"
    if process_cues >= 0.46 and scene_distance >= 1.8 and visualizability >= 0.45 and generic_penalty <= 0.32:
        return "replace"
    if replace_safety >= 0.48 and explicit_signal and visualizability >= 0.42:
        return "replace"
    if replace_safety >= 0.44 and numeric_hits >= 1 and visualizability >= 0.66:
        return "replace"
    return composition_mode


def _text_limits_for_visual(
    *,
    template: str,
    composition_mode: str,
    short_slot: bool,
    prefer_premium: bool,
) -> dict[str, int]:
    is_replace = composition_mode == "replace"
    premium_fullscreen = prefer_premium and is_replace and template in PREMIUM_FULLSCREEN_TEMPLATES
    route_like = template in {
        "timeline_steps",
        "system_flow",
        "kinetic_route",
        "signal_network",
        "causal_chain",
        "flywheel_loop",
        "narrative_arc",
        "concept_map",
        "checklist_reveal",
        "pipeline_xray",
        "decision_tree",
        "timeline_filmstrip",
        "mechanism_blueprint",
        "market_map",
    }
    compare_like = template in {
        "comparison_split",
        "spotlight_compare",
        "interface_cascade",
        "decision_matrix",
        "contrast_ladder",
        "anatomy_cutaway",
        "problem_solution",
        "myth_buster",
        "scorecard",
    }
    quote_like = template in {"quote_focus", "keyword_stack", "ribbon_quote", "quote_breakdown"}
    if premium_fullscreen:
        return {
            "headline_words": 4 if short_slot else 5,
            "headline_chars": 32 if short_slot else 40,
            "deck_words": 4 if short_slot else 6,
            "deck_chars": 28 if short_slot else 38,
            "support_words": 3 if route_like else 4,
            "support_chars": 22 if route_like else 28,
            "step_words": 3 if route_like else 4,
            "step_chars": 18 if route_like else 24,
            "detail_words": 4 if compare_like else 5,
            "detail_chars": 26 if compare_like else 32,
            "quote_words": 8 if quote_like else 10,
            "quote_chars": 56 if quote_like else 68,
            "footer_words": 4 if short_slot else 5,
            "footer_chars": 28 if short_slot else 34,
        }
    if is_replace:
        return {
            "headline_words": 5 if short_slot else 6,
            "headline_chars": 36 if short_slot else 46,
            "deck_words": 5 if short_slot else 7,
            "deck_chars": 34 if short_slot else 46,
            "support_words": 4 if route_like else 5,
            "support_chars": 28 if route_like else 36,
            "step_words": 4,
            "step_chars": 24 if route_like else 28,
            "detail_words": 5,
            "detail_chars": 34 if compare_like else 40,
            "quote_words": 10,
            "quote_chars": 72,
            "footer_words": 5 if short_slot else 6,
            "footer_chars": 32 if short_slot else 40,
        }
    return {
        "headline_words": 5 if short_slot else 6,
        "headline_chars": 38 if short_slot else 48,
        "deck_words": 6 if short_slot else 8,
        "deck_chars": 36 if short_slot else 50,
        "support_words": 5,
        "support_chars": 42,
        "step_words": 4,
        "step_chars": 28,
        "detail_words": 6,
        "detail_chars": 42,
        "quote_words": 10,
        "quote_chars": 80,
        "footer_words": 6 if short_slot else 8,
        "footer_chars": 42 if short_slot else 58,
    }


def _normalize_visual_plan(
    raw_plan: list[dict[str, Any]],
    cards: list[dict[str, Any]],
    clip_duration: float,
    max_visuals: int,
    min_visual_sec: float,
    max_visual_sec: float,
    scene_cuts: list[float],
    available_renderers: list[dict[str, Any]] | None,
    prefer_premium: bool = False,
) -> list[dict[str, Any]]:
    epsilon = 1e-3
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
    ordered_plan = sorted(
        list(raw_plan),
        key=lambda item: float((card_map.get(str(item.get("card_id") or "").strip()) or {}).get("start") or 0.0),
    )
    for index, item in enumerate(ordered_plan, start=1):
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
        metric_facts = list(card.get("metric_facts") or [])
        if template in METRIC_TEMPLATES and not metric_facts:
            template = "signal_network" if float(card.get("process_cues") or 0.0) >= 0.34 else "ribbon_quote"
        if (
            available_renderers is not None
            and template in BLENDER_3D_TEMPLATES
            and "blender" not in available_names
        ):
            template = BLENDER_TEMPLATE_DOWNGRADES.get(template, _default_template(card))
        hyperframes_expansion_templates = {
            "causal_chain",
            "flywheel_loop",
            "decision_matrix",
            "anatomy_cutaway",
            "stack_ranking",
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
        if template in hyperframes_expansion_templates and "hyperframes" not in available_names:
            template = EDITORIAL_TEMPLATE_DOWNGRADES.get(template, "timeline_steps")
        if float(card.get("visualizability") or 0.0) < 0.46 and template in {"quote_focus", "keyword_stack", "ribbon_quote"}:
            continue
        if float(card.get("generic_penalty") or 0.0) > 0.68 and int(card.get("numeric_hits") or 0) == 0 and float(card.get("process_cues") or 0.0) < 0.3:
            continue
        if template in {"quote_focus", "keyword_stack", "ribbon_quote"}:
            max_count_for_template = 1 if not prefer_premium else 2
        elif prefer_premium and template in PREMIUM_FULLSCREEN_TEMPLATES:
            max_count_for_template = 4
        else:
            max_count_for_template = 3
        if template_counts.get(template, 0) >= max_count_for_template:
            continue
        composition_mode = str(item.get("composition_mode") or card["suggested_composition"]).strip().lower()
        if composition_mode in {"pip", "overlay", "picture-in-picture"}:
            composition_mode = "picture_in_picture"
        if composition_mode not in {"replace", "picture_in_picture"}:
            composition_mode = card["suggested_composition"]
        if template in BLENDER_OVERLAY_TEMPLATES and not prefer_premium:
            composition_mode = "picture_in_picture"
        elif template in {"three_d_title", "object_orbit", "logo_reveal", "data_tunnel"}:
            composition_mode = "replace"
        if prefer_premium:
            composition_mode = "replace"
        else:
            composition_mode = _promote_composition_for_premium(
                card,
                composition_mode,
                prefer_premium=prefer_premium,
            )
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
        if end_sec - start_sec + epsilon < minimum_duration:
            continue
        if composition_mode == "picture_in_picture" and not prefer_premium:
            template = EDITORIAL_TEMPLATE_DOWNGRADES.get(template, template)
        template = _upgrade_to_premium_template(card, template, composition_mode)
        if (
            prefer_premium
            and composition_mode == "picture_in_picture"
            and template in PREMIUM_FULLSCREEN_TEMPLATES
            and (
                float(card.get("replace_safety") or 0.0) >= 0.42
                or (
                    float(card.get("process_cues") or 0.0) >= 0.44
                    and float(card.get("scene_distance") or 0.0) >= 1.6
                    and float(card.get("visualizability") or 0.0) >= 0.44
                )
            )
        ):
            composition_mode = "replace"
            start_sec, end_sec = _expand_window_to_duration(
                start_sec,
                end_sec,
                clip_duration=clip_duration,
                target_duration_sec=min(max(min_visual_sec, MIN_PREMIUM_REPLACE_DURATION_SEC), max_visual_sec),
                scene_cuts=scene_cuts,
            )
            if end_sec - start_sec + epsilon < max(min_visual_sec, MIN_PREMIUM_REPLACE_DURATION_SEC):
                continue
        if composition_mode == "replace":
            position = "center"
            scale = 1.0
        else:
            position = str(item.get("position") or "bottom_right").strip().lower()
            if position not in {"top_left", "top_right", "bottom_left", "bottom_right", "top", "bottom", "center", "center_left", "center_right"}:
                position = "bottom_right"
            scale = round(max(0.24, min(float(item.get("scale", 0.42) or 0.42), 0.8)), 3)
        slot_duration = end_sec - start_sec
        short_slot = slot_duration <= 2.8
        text_limits = _text_limits_for_visual(
            template=template,
            composition_mode=composition_mode,
            short_slot=short_slot,
            prefer_premium=prefer_premium,
        )
        supporting_lines = _coerce_string_list(
            item.get("supporting_lines"),
            limit=3,
            max_chars=text_limits["support_chars"],
            max_words=text_limits["support_words"],
        )
        keywords = _coerce_string_list(item.get("keywords"), limit=4, max_chars=24, max_words=3)
        steps = _coerce_string_list(
            item.get("steps"),
            limit=4,
            max_chars=text_limits["step_chars"],
            max_words=text_limits["step_words"],
        )
        derived_headline = _headline_from_card(card)
        headline = _polish_visual_copy(
            item.get("headline") or derived_headline or card["sentence_text"],
            max_words=text_limits["headline_words"],
            max_chars=text_limits["headline_chars"],
        )
        if headline.lower() == str(card.get("sentence_text") or "").strip().lower() or len(headline.split()) > 8:
            headline = derived_headline or truncate(headline, 42)
        emphasis_source = str(item.get("emphasis_text") or _extract_emphasis_text(card))
        emphasis_text = emphasis_source if re.fullmatch(r"\d+(?:\.\d+)?(?:%|x)?", emphasis_source, flags=re.IGNORECASE) else _polish_visual_copy(emphasis_source, max_words=4, max_chars=34)
        footer_text = _polish_visual_copy(
            item.get("footer_text") or _deck_for_card(card, headline) or card["context_text"],
            max_words=text_limits["footer_words"],
            max_chars=text_limits["footer_chars"],
        )
        style_pack = str(item.get("style_pack") or card["style_pack"] or "editorial_clean").strip().lower()
        if style_pack not in STYLE_PACKS:
            style_pack = card["style_pack"]
        renderer_hint = str(item.get("renderer_hint") or card["suggested_renderer"] or "auto").strip().lower()
        hyperframes_only_templates = {
            "data_journey",
            "signal_network",
            "kinetic_route",
            "spotlight_compare",
            "interface_cascade",
            "ribbon_quote",
            "causal_chain",
            "flywheel_loop",
            "decision_matrix",
            "anatomy_cutaway",
            "stack_ranking",
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
        if template in BLENDER_3D_TEMPLATES:
            renderer_hint = "blender"
        if composition_mode == "replace" and renderer_hint in {"auto", "ffmpeg"} and template not in {"quote_focus", "keyword_stack", "metric_callout", "stat_grid", "timeline_steps", "comparison_split"}:
            renderer_hint = "hyperframes"
        if template in hyperframes_only_templates:
            renderer_hint = "hyperframes"
        if not prefer_premium and composition_mode == "picture_in_picture" and template in {"metric_callout", "keyword_stack", "quote_focus", "stat_grid", "comparison_split", "timeline_steps"}:
            renderer_hint = "ffmpeg"
        if prefer_premium and renderer_hint in {"auto", "ffmpeg"}:
            renderer_hint = "hyperframes"
        if renderer_hint in known_renderers and renderer_hint not in available_names:
            renderer_hint = "auto"
        if known_renderers and renderer_hint not in known_renderers and renderer_hint != "auto":
            fallback_renderer = str(card.get("suggested_renderer") or "auto").strip().lower()
            renderer_hint = fallback_renderer if fallback_renderer in known_renderers else "auto"
        motion_preset = truncate(
            str(item.get("motion_preset") or _default_motion_preset(card, template)),
            32,
        )
        eyebrow = truncate(str(item.get("eyebrow") or _eyebrow_for_card(card, template)), 18).upper()
        deck = _polish_visual_copy(
            item.get("deck") or _deck_for_card(card, headline),
            max_words=text_limits["deck_words"],
            max_chars=text_limits["deck_chars"],
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
            "previous_text": card.get("previous_text", ""),
            "next_text": card.get("next_text", ""),
            "semantic_frame": dict(card.get("semantic_frame") or {}),
            "keywords": card["keywords"][:8],
            "metric_facts": metric_facts,
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
            "quote_text": _polish_visual_copy(
                item.get("quote_text") or headline,
                max_words=text_limits["quote_words"],
                max_chars=text_limits["quote_chars"],
            ),
            "left_label": truncate(str(item.get("left_label") or "Before"), 28),
            "right_label": truncate(str(item.get("right_label") or "After"), 28),
            "left_detail": _polish_visual_copy(
                item.get("left_detail") or card["sentence_text"],
                max_words=text_limits["detail_words"],
                max_chars=text_limits["detail_chars"],
            ),
            "right_detail": _polish_visual_copy(
                item.get("right_detail") or card["context_text"],
                max_words=text_limits["detail_words"],
                max_chars=text_limits["detail_chars"],
            ),
            "footer_text": footer_text,
            "style_pack": style_pack,
            "theme": _theme_for_card(card, style_pack),
            "renderer_hint": renderer_hint,
            "motion_preset": motion_preset,
            "text": str(item.get("text") or item.get("emphasis_text") or headline),
            "subtext": str(item.get("subtext") or item.get("deck") or deck),
            "label": str(item.get("label") or item.get("eyebrow") or eyebrow or headline),
            "camera_motion": str(
                item.get("camera_motion")
                or (
                    "orbit"
                    if template in {"object_orbit", "data_tunnel", "product_model_spin"}
                    else "static"
                    if template in {"floating_3d_label", "screen_pointer_3d"}
                    else "slow_push"
                )
            ).strip().lower(),
            "object_motion": str(
                item.get("object_motion")
                or (
                    "spin_y"
                    if template in {"object_orbit", "product_model_spin"}
                    else "float"
                    if template in {"floating_3d_label", "screen_pointer_3d"}
                    else "drop_in"
                    if template == "logo_reveal"
                    else "none"
                )
            ).strip().lower(),
            "alpha": template in BLENDER_OVERLAY_TEMPLATES and composition_mode == "picture_in_picture",
            "asset_path": str(item.get("asset_path") or "").strip(),
            "accent_color": str(item.get("accent_color") or "").strip(),
            "text_color": str(item.get("text_color") or "").strip(),
            "background_color": str(item.get("background_color") or "").strip(),
            "shadow": item.get("shadow", True),
            "transparent_background": template in BLENDER_OVERLAY_TEMPLATES and composition_mode == "picture_in_picture",
            "safe_area": item.get("safe_area", True),
            "background_motif": background_motif,
            "layout_variant": layout_variant,
            "generation_tier": "premium" if prefer_premium else "standard",
            "require_generated_scene": bool(prefer_premium and renderer_hint == "manim"),
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
                "metric_facts": metric_facts,
                "sentence_process_cues": card["sentence_process_cues"],
                "visualizability": card["visualizability"],
                "generic_penalty": card["generic_penalty"],
                "sentence_contrast_cues": card["sentence_contrast_cues"],
                "process_cues": card["process_cues"],
                "contrast_cues": card["contrast_cues"],
            },
        }
        for passthrough_key in (
            "planning_context_text",
            "semantic_episode_id",
            "semantic_episode_summary",
            "source_card_ids",
            "opportunity_contract",
            "opportunity_preflight",
        ):
            if passthrough_key in card:
                value = card.get(passthrough_key)
                spec[passthrough_key] = (
                    dict(value)
                    if isinstance(value, dict)
                    else list(value)
                    if isinstance(value, list)
                    else value
                )
        route_templates = {
            "timeline_steps",
            "system_flow",
            "kinetic_route",
            "signal_network",
            "causal_chain",
            "flywheel_loop",
            "narrative_arc",
            "concept_map",
            "checklist_reveal",
            "pipeline_xray",
            "decision_tree",
            "timeline_filmstrip",
            "mechanism_blueprint",
            "market_map",
        }
        if template in {"keyword_stack", "ribbon_quote"} and not keywords:
            spec["keywords"] = card["keywords"][:4] or [headline]
        if template in route_templates and not steps:
            spec["steps"] = _steps_for_card(card) or ([headline, emphasis_text, footer_text[:28]] if footer_text else [headline, emphasis_text])[:4]
        if template in route_templates:
            semantic_frame = dict(card.get("semantic_frame") or {})
            semantic_before = _polish_visual_copy(
                semantic_frame.get("before_state") or "",
                max_words=text_limits["step_words"],
                max_chars=text_limits["step_chars"],
            )
            semantic_after = _polish_visual_copy(
                semantic_frame.get("after_state") or "",
                max_words=text_limits["step_words"],
                max_chars=text_limits["step_chars"],
            )
            semantic_steps = []
            for value in [semantic_before, semantic_after]:
                lowered = value.lower()
                if value and lowered not in {item.lower() for item in semantic_steps}:
                    semantic_steps.append(value)
            existing_steps = list(spec.get("steps") or [])
            weak_route_steps = not existing_steps or (
                semantic_after
                and all(
                    str(step).strip().lower() in semantic_after.lower()
                    or semantic_after.lower().startswith(str(step).strip().lower())
                    for step in existing_steps
                    if str(step).strip()
                )
            )
            if semantic_steps and (weak_route_steps or len(existing_steps) < 2):
                spec["steps"] = semantic_steps[:3]
            if semantic_after and not list(spec.get("supporting_lines") or []):
                spec["supporting_lines"] = [semantic_after]
        if template in {"metric_callout", "stat_grid", "data_journey", "proof_sequence", "stack_ranking", "scorecard", "risk_radar", "data_pulse", "momentum_wave"} and not supporting_lines:
            spec["supporting_lines"] = _supporting_lines_for_card(card)[:3]
        if short_slot:
            spec["supporting_lines"] = list(spec.get("supporting_lines") or [])[:2]
            spec["steps"] = list(spec.get("steps") or [])[:3]
            spec["quote_text"] = _polish_visual_copy(spec.get("quote_text") or headline, max_words=8, max_chars=58)
        if template in {"comparison_split", "spotlight_compare", "decision_matrix", "contrast_ladder", "problem_solution", "myth_buster"}:
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
    seen_novelty: set[str] = set()
    for card in ranked:
        visualizability = float(card.get("visualizability") or 0.0)
        numeric_hits = int(card.get("numeric_hits") or 0)
        process_cues = float(card.get("process_cues") or 0.0)
        contrast_cues = float(card.get("contrast_cues") or 0.0)
        generic_penalty = float(card.get("generic_penalty") or 0.0)
        intuition_role = str(card.get("intuition_role") or "")
        intuition_payoff = float(card.get("intuition_payoff") or 0.0)
        novelty_key = str(card.get("novelty_key") or "")
        explicit_signal = numeric_hits > 0 or process_cues >= 0.2 or contrast_cues >= 0.2
        if visualizability < 0.34 and not explicit_signal:
            continue
        if generic_penalty > 0.82 and visualizability < 0.5 and not explicit_signal:
            continue
        if intuition_payoff < 0.48 and not explicit_signal:
            continue
        if intuition_role == "supporting_example" and intuition_payoff < 0.58 and visualizability < 0.62:
            continue
        if novelty_key and novelty_key in seen_novelty:
            continue
        if any(abs(card["start"] - existing["start"]) < 0.6 for existing in pool):
            continue
        if intuition_role == "core_mechanism" and any(
            str(existing.get("intuition_role") or "") == "core_mechanism"
            and abs(float(card["start"]) - float(existing["start"])) < 6.5
            for existing in pool
        ):
            continue
        pool.append(card)
        if novelty_key:
            seen_novelty.add(novelty_key)
        if len(pool) >= max(max_visuals * 4, 14):
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
        intuition_payoff = float(card.get("intuition_payoff") or 0.0)
        explicit_signal = (
            int(card.get("numeric_hits") or 0) > 0
            or float(card.get("process_cues") or 0.0) >= 0.42
            or float(card.get("contrast_cues") or 0.0) >= 0.4
        )
        if explicit_signal and visualizability >= 0.68 and generic_penalty <= 0.36 and intuition_payoff >= 0.68:
            strong += 1
        if visualizability < 0.55 or generic_penalty > 0.58 or intuition_payoff < 0.58:
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
    *,
    prefer_premium: bool = False,
) -> list[dict[str, Any]]:
    ranked = _candidate_pool(cards, max_visuals)
    fallback = []
    candidate_budget = max(max_visuals * 3, max_visuals + 2)
    for card in ranked:
        template = _default_template(card)
        composition_mode = _promote_composition_for_premium(
            card,
            str(card.get("suggested_composition") or "picture_in_picture"),
            prefer_premium=prefer_premium,
        )
        if prefer_premium:
            composition_mode = "replace"
        fallback.append(
            {
                "card_id": card["card_id"],
                "template": template,
                "renderer_hint": "hyperframes" if prefer_premium else card["suggested_renderer"],
                "style_pack": card["style_pack"],
                "composition_mode": composition_mode,
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
                "position": "center" if prefer_premium else "bottom_right",
                "scale": 1.0 if prefer_premium else 0.42,
                "motion_preset": _default_motion_preset(card, template),
                "background_motif": _background_motif(card, template, str(card.get("style_pack") or "editorial_clean")),
                "layout_variant": LAYOUT_VARIANTS.get(template, "hero_split"),
                "rationale": (
                    "Premium deterministic visual chosen from the strongest transcript beat when the model plan was unavailable."
                    if prefer_premium
                    else "Fallback visual chosen from the strongest transcript beat when the model plan was unavailable."
                ),
                "confidence": round(min(max(card["priority"] / 88.0, 0.45), 0.9), 2),
            }
        )
        if len(fallback) >= candidate_budget:
            break
    normalized_fallback = _normalize_visual_plan(
        fallback,
        cards,
        clip_duration,
        max_visuals,
        min_visual_sec,
        max_visual_sec,
        scene_cuts,
        available_renderers,
        prefer_premium=prefer_premium,
    )
    normalized_fallback = _prune_low_intuition_plan(
        normalized_fallback,
        cards,
        max_visuals=max_visuals,
        prefer_premium=prefer_premium,
    )
    return enforce_visual_semantic_contracts(_resequence_visual_ids(normalized_fallback), max_visuals=max_visuals)


def _prune_low_intuition_plan(
    plan: list[dict[str, Any]],
    cards: list[dict[str, Any]],
    *,
    max_visuals: int,
    prefer_premium: bool,
) -> list[dict[str, Any]]:
    if not plan:
        return []
    card_by_id = {str(card.get("card_id") or ""): card for card in cards}
    kept: list[dict[str, Any]] = []
    seen_novelty: set[str] = set()
    has_core = False
    metric_count = 0
    last_core_start: float | None = None
    for item in sorted(plan, key=lambda value: float(value.get("start") or 0.0)):
        card = card_by_id.get(str(item.get("card_id") or ""), {})
        intuition_role = str(card.get("intuition_role") or "")
        intuition_mode = str(card.get("intuition_mode") or "")
        intuition_payoff = float(card.get("intuition_payoff") or 0.0)
        novelty_key = str(card.get("novelty_key") or "")
        confidence = float(item.get("confidence") or 0.0)
        explicit_signal = (
            int(card.get("numeric_hits") or 0) > 0
            or float(card.get("process_cues") or 0.0) >= 0.22
            or float(card.get("contrast_cues") or 0.0) >= 0.22
            or float(card.get("visualizability") or 0.0) >= 0.66
        )
        min_payoff = 0.54 if prefer_premium else 0.5
        if intuition_payoff < min_payoff and not explicit_signal:
            continue
        if intuition_role == "supporting_example":
            if intuition_payoff < 0.64 and not explicit_signal:
                continue
        if has_core and intuition_role == "concrete_proof" and intuition_payoff < 0.68 and not explicit_signal:
            continue
        if intuition_mode == "metric_proof":
            if metric_count >= 3:
                continue
            if metric_count >= 1 and intuition_payoff < 0.7 and not explicit_signal:
                continue
        if novelty_key and novelty_key in seen_novelty:
            continue
        current_start = float(item.get("start") or 0.0)
        if intuition_role == "core_mechanism" and last_core_start is not None and abs(current_start - last_core_start) < 6.5:
            continue
        kept.append(item)
        if novelty_key:
            seen_novelty.add(novelty_key)
        if intuition_role == "core_mechanism":
            has_core = True
            last_core_start = current_start
        if intuition_mode == "metric_proof":
            metric_count += 1
        if len(kept) >= max_visuals:
            break
    if not kept:
        best = max(
            plan,
            key=lambda item: (
                float((card_by_id.get(str(item.get("card_id") or ""), {}) or {}).get("intuition_payoff") or 0.0),
                float(item.get("confidence") or 0.0),
            ),
        )
        kept = [best]
    return _resequence_visual_ids(kept)


def _backfill_plan_with_fallback(
    primary: list[dict[str, Any]],
    fallback: list[dict[str, Any]],
    *,
    max_visuals: int,
    prefer_premium: bool,
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


def enforce_visual_semantic_contracts(plan: list[dict[str, Any]], *, max_visuals: int | None = None) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for item in sorted(plan, key=lambda value: float(value.get("start") or 0.0)):
        normalized = dict(item)
        metric_facts = list(normalized.get("metric_facts") or (normalized.get("evidence") or {}).get("metric_facts") or [])
        sentence_text = str(normalized.get("sentence_text") or "")
        context_text = str(normalized.get("context_text") or "")
        claim_key = (
            _canonical_claim_key(f"{sentence_text} {context_text}", metric_facts)
            if metric_facts
            else str(normalized.get("card_id") or "").strip()
        )
        if claim_key and claim_key in seen_keys:
            continue
        template = str(normalized.get("template") or "").strip().lower()
        headline = str(normalized.get("headline") or "").strip()
        if _is_low_signal_copy(headline) and not metric_facts:
            continue
        if template in METRIC_TEMPLATES and not metric_facts:
            template = "signal_network" if float((normalized.get("evidence") or {}).get("process_cues") or 0.0) >= 0.34 else "ribbon_quote"
            normalized["template"] = template
        if metric_facts:
            normalized["metric_facts"] = metric_facts
            first_value = str(metric_facts[0].get("value") or "").strip()
            if first_value:
                normalized["emphasis_text"] = first_value
                if template in METRIC_TEMPLATES and re.search(r"\bkv\b|\bcache\b", f"{sentence_text} {context_text}", flags=re.IGNORECASE):
                    normalized["headline"] = f"{first_value} KV cache"
        for list_key in ("supporting_lines", "steps", "keywords"):
            values = []
            seen_values: set[str] = set()
            for value in normalized.get(list_key) or []:
                text = _polish_visual_copy(value, max_words=7 if list_key == "supporting_lines" else 4, max_chars=44 if list_key == "supporting_lines" else 30)
                lowered = text.lower()
                if not text or _is_low_signal_copy(text) or lowered in seen_values:
                    continue
                seen_values.add(lowered)
                values.append(text)
            normalized[list_key] = values
        cleaned.append(normalized)
        if claim_key:
            seen_keys.add(claim_key)
        if max_visuals is not None and len(cleaned) >= max_visuals:
            break
    return _resequence_visual_ids(cleaned)


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
    prefer_premium: bool = False,
    visual_program: dict[str, Any] | None = None,
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
        prefer_premium=prefer_premium,
    )
    if not cards:
        return fallback

    candidate_cards = _candidate_pool(cards, max_visuals)
    if not disable_fast_plan and not prefer_premium and _should_use_fast_plan(candidate_cards, max_visuals):
        return fallback
    template_lines = "\n".join(f"- {name}: {description}" for name, description in SUPPORTED_TEMPLATES.items())
    renderer_lines = _format_renderer_capabilities(available_renderers)
    program_block = visual_program_prompt_block(visual_program or {}) if visual_program else ""
    avoid_card_line = (
        "Previously used card_ids to avoid unless absolutely necessary:\n"
        f"{', '.join(sorted(avoid_card_ids))}\n\n"
        if avoid_card_ids
        else ""
    )
    composition_guidance = (
        "This run is producing premium generated visuals: every selected visual must use composition_mode='replace' "
        "so it becomes a full-screen cutaway. Do not choose picture_in_picture.\n"
        if prefer_premium
        else "Use the evidence fields to decide when a full-screen replacement is safe versus when picture-in-picture is safer.\n"
    )
    system_prompt = (
        "You are a senior motion graphics director planning precise generated visuals for an explainer video. "
        "Choose only transcript beats where a custom animation would make the spoken idea clearer. "
        "Aim for dense, useful coverage across the whole video: if many transcript beats have visualizable mechanisms, map them into multiple concise visual episodes instead of stopping after one or two. "
        "One excellent visual beats three mediocre ones, but strong videos should receive every meaningful visual opportunity that survives the evidence fields. "
        "Prefer concise, literal, high-signal visuals with strong editorial taste. "
        "Do not create generic motivational cards. If a beat is vague or low-signal, skip it. "
        "Distill the copy. Do not simply repeat the spoken sentence as the headline. "
        "Choose Blender only when a genuinely 3D visual helps: 3D title, model/object shot, logo reveal, floating 3D label, pointer, product spin, or cinematic data tunnel. "
        "Blender requests must use typed parameters only; never output raw Blender Python. "
        "Use overlay/picture_in_picture for floating labels, arrows, and product/model overlays; use replace for full-screen cinematic 3D insert shots. "
        f"{composition_guidance}"
        "Return ONLY a JSON array with at most {count} objects using these keys: "
        "card_id, template, renderer_hint, style_pack, composition_mode, eyebrow, headline, deck, emphasis_text, supporting_lines, "
        "steps, keywords, quote_text, left_label, right_label, left_detail, right_detail, footer_text, position, scale, "
        "motion_preset, camera_motion, object_motion, asset_path, label, subtext, accent_color, text_color, background_color, "
        "background_motif, layout_variant, rationale, confidence."
    ).format(count=max_visuals)
    program_section = f"{program_block}\n\n" if program_block else ""
    user_prompt = (
        f"Video duration: {clip_duration:.2f}s\n"
        f"Max visuals: {max_visuals}\n"
        f"Duration per visual: {min_visual_sec:.1f}s to {max_visual_sec:.1f}s\n"
        f"Detected scene cuts: {scene_cuts[:24]}\n\n"
        f"{program_section}"
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
        "Use the Visual Narrative Program when present: prefer its episode families, continuity groups, and transition intent unless the candidate evidence clearly contradicts it. "
        "Use intuition_role and intuition_payoff aggressively: core_mechanism beats are best, concrete_proof beats are optional, supporting_example beats are usually not worth a premium visual. "
        "Favor data_journey, data_pulse, proof_sequence, scorecard, or risk_radar for quantitative and proof beats; signal_network, kinetic_route, concept_map, mechanism_blueprint, pipeline_xray, causal_chain, decision_tree, or flywheel_loop for process beats; spotlight_compare, problem_solution, myth_buster, or contrast_ladder for contrasts; decision_matrix for tradeoffs; anatomy_cutaway or quote_breakdown for layered systems; interface_cascade for UI/product beats; momentum_wave for growth or compounding; focus_ring for attention/noise beats; timeline_filmstrip or narrative_arc for story beats; and ribbon_quote only when the line is truly memorable. "
        "Blender examples: three_d_title for 'rotating 3D title saying Attention Mechanism', floating_3d_label for 'label near the right side', data_tunnel for neural networks/GPU/data-flow mentions, screen_pointer_3d for '3D arrow pointing to the chart', and product_model_spin for 'spin this product model from 00:12 to 00:16'. "
        "Use the older editorial templates mainly for picture-in-picture or lightweight overlays, not for premium full-screen generated visuals. "
        "Prefer hyperframes for premium HTML/CSS motion slides, product UI scenes, process diagrams, comparisons, timelines, and data-driven explainers. Use manim only for formula-heavy math, geometry, axes, or scenes that genuinely need Manim's object model. Use ffmpeg for simple clean picture-in-picture cards, and blender only for cinematic synthetic shots when available. "
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
        prefer_premium=prefer_premium,
    )
    if not normalized_director:
        return fallback
    normalized_director = _prune_low_intuition_plan(
        normalized_director,
        cards,
        max_visuals=max_visuals,
        prefer_premium=prefer_premium,
    )
    normalized_director = _backfill_plan_with_fallback(
        normalized_director,
        fallback,
        max_visuals=max_visuals,
        prefer_premium=prefer_premium,
    )
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
        f"{program_section}"
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
            prefer_premium=prefer_premium,
        )
        if normalized_critic:
            normalized_critic = _prune_low_intuition_plan(
                normalized_critic,
                cards,
                max_visuals=max_visuals,
                prefer_premium=prefer_premium,
            )
            normalized_critic = _backfill_plan_with_fallback(
                normalized_critic,
                fallback,
                max_visuals=max_visuals,
                prefer_premium=prefer_premium,
            )
        return _resequence_visual_ids(normalized_critic or normalized_director or fallback)
    except Exception:
        return _resequence_visual_ids(normalized_director or fallback)
