from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
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
    "metric_callout": "Large value or claim with supporting context.",
    "keyword_stack": "A stacked set of short concepts or phrases.",
    "timeline_steps": "A short process broken into 2-4 steps.",
    "comparison_split": "A side-by-side before/after or old/new contrast.",
    "quote_focus": "A clean emphasis card for a memorable phrase or quote.",
}

DEFAULT_THEME = {
    "background": "#0B1020",
    "panel_fill": "#13203A",
    "panel_stroke": "#60A5FA",
    "accent": "#F59E0B",
    "text_primary": "#F8FAFC",
    "text_secondary": "#CBD5E1",
}

THEME_BY_VISUAL_TYPE = {
    "data_graphic": {
        "panel_stroke": "#38BDF8",
        "accent": "#F59E0B",
    },
    "product_ui": {
        "panel_stroke": "#A78BFA",
        "accent": "#22C55E",
    },
    "process": {
        "panel_stroke": "#34D399",
        "accent": "#F97316",
    },
    "abstract_motion": {
        "panel_stroke": "#F472B6",
        "accent": "#FBBF24",
    },
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


def _visual_priority(card: dict[str, Any]) -> float:
    combined = f"{card['sentence_text']} {card['context_text']}".lower()
    tokens = re.findall(r"[a-zA-Z0-9']+", combined)
    if not tokens:
        return 0.0
    numbers = len(re.findall(r"\b\d+(?:\.\d+)?%?\b", combined))
    specificity = min(len(set(tokens)) / max(len(tokens), 1), 1.0)
    visual_hits = len(card.get("keywords") or [])
    proper_nouns = len(re.findall(r"\b[A-Z][a-zA-Z0-9]+\b", f"{card['sentence_text']} {card['context_text']}"))
    return round(26 + numbers * 9 + specificity * 26 + min(visual_hits, 8) * 3 + proper_nouns * 2.5, 2)


def build_visual_context_cards(
    sentences: list[dict[str, Any]],
    transcript_segments: list[dict[str, Any]],
    clip_duration: float,
) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
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
        keywords = semantic_keywords(f"{sentence_text} {context_text}", limit=8)
        row = {
            "card_id": f"visual_card_{index:03d}",
            "start": round(start_sec, 2),
            "end": round(end_sec, 2),
            "sentence_text": sentence_text,
            "context_text": context_text,
            "keywords": keywords,
            "visual_type_hint": infer_visual_type(f"{sentence_text} {context_text}"),
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
                    f"Hint: {card['visual_type_hint']}",
                ]
            )
        )
    return "\n\n".join(lines)


def _theme_for_card(card: dict[str, Any]) -> dict[str, str]:
    theme = dict(DEFAULT_THEME)
    theme.update(THEME_BY_VISUAL_TYPE.get(str(card.get("visual_type_hint") or ""), {}))
    return theme


def _default_template(card: dict[str, Any]) -> str:
    visual_type = str(card.get("visual_type_hint") or "")
    if visual_type == "data_graphic":
        return "metric_callout"
    if visual_type == "process":
        return "timeline_steps"
    if visual_type == "product_ui":
        return "comparison_split"
    if visual_type == "abstract_motion":
        return "keyword_stack"
    return "quote_focus"


def _snap_to_scene(value: float, scene_cuts: list[float], max_distance: float = 0.4) -> float:
    if not scene_cuts:
        return value
    nearest = min(scene_cuts, key=lambda cut: abs(cut - value))
    if abs(nearest - value) <= max_distance:
        return nearest
    return value


def _extract_emphasis_text(card: dict[str, Any]) -> str:
    text = str(card.get("sentence_text") or "")
    number_match = re.search(r"\b\d+(?:\.\d+)?%?\b", text)
    if number_match:
        return number_match.group(0)
    keywords = list(card.get("keywords") or [])
    if keywords:
        return " ".join(keywords[:3])
    return truncate(text, 32)


def _coerce_string_list(raw: Any, limit: int, max_chars: int) -> list[str]:
    if isinstance(raw, list):
        values = [truncate(str(item), max_chars) for item in raw if str(item).strip()]
    elif str(raw or "").strip():
        values = [truncate(str(raw), max_chars)]
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
) -> list[dict[str, Any]]:
    card_map = {card["card_id"]: card for card in cards}
    normalized: list[dict[str, Any]] = []
    last_end = -999.0
    for index, item in enumerate(raw_plan, start=1):
        card = card_map.get(str(item.get("card_id") or "").strip())
        if card is None:
            continue
        start_sec = max(0.0, min(float(card["start"]) - 0.08, clip_duration))
        end_sec = min(clip_duration, float(card["end"]) + 0.22)
        start_sec = _snap_to_scene(start_sec, scene_cuts)
        end_sec = _snap_to_scene(end_sec, scene_cuts)
        if end_sec - start_sec < min_visual_sec:
            end_sec = min(clip_duration, start_sec + min_visual_sec)
        if end_sec - start_sec > max_visual_sec:
            end_sec = start_sec + max_visual_sec
        if end_sec <= start_sec or start_sec - last_end < 0.55:
            continue
        confidence = max(0.0, min(float(item.get("confidence", 0.58)), 1.0))
        if confidence < 0.35:
            continue
        template = str(item.get("template") or _default_template(card)).strip().lower()
        if template not in SUPPORTED_TEMPLATES:
            template = _default_template(card)
        composition_mode = str(item.get("composition_mode") or "replace").strip().lower()
        if composition_mode in {"pip", "overlay", "picture-in-picture"}:
            composition_mode = "picture_in_picture"
        if composition_mode not in {"replace", "picture_in_picture"}:
            composition_mode = "replace"
        position = str(item.get("position") or "bottom_right").strip().lower()
        if position not in {"top_left", "top_right", "bottom_left", "bottom_right", "top", "bottom", "center"}:
            position = "bottom_right"
        scale = round(max(0.24, min(float(item.get("scale", 0.42) or 0.42), 0.8)), 3)
        supporting_lines = _coerce_string_list(item.get("supporting_lines"), limit=3, max_chars=72)
        keywords = _coerce_string_list(item.get("keywords"), limit=4, max_chars=28)
        steps = _coerce_string_list(item.get("steps"), limit=4, max_chars=28)
        headline = truncate(str(item.get("headline") or card["sentence_text"]), 72)
        emphasis_text = truncate(str(item.get("emphasis_text") or _extract_emphasis_text(card)), 48)
        footer_text = truncate(str(item.get("footer_text") or card["context_text"]), 84)
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
            "headline": headline,
            "emphasis_text": emphasis_text,
            "supporting_lines": supporting_lines,
            "steps": steps,
            "quote_text": truncate(str(item.get("quote_text") or headline), 120),
            "left_label": truncate(str(item.get("left_label") or "Before"), 28),
            "right_label": truncate(str(item.get("right_label") or "After"), 28),
            "left_detail": truncate(str(item.get("left_detail") or card["sentence_text"]), 72),
            "right_detail": truncate(str(item.get("right_detail") or card["context_text"]), 72),
            "footer_text": footer_text,
            "theme": _theme_for_card(card),
            "rationale": truncate(str(item.get("rationale") or "Generated visual aligned to the active spoken beat."), 160),
            "confidence": round(confidence, 2),
        }
        if template == "keyword_stack" and not keywords:
            spec["keywords"] = card["keywords"][:4] or [headline]
        if template == "timeline_steps" and not steps:
            spec["steps"] = (
                [headline, emphasis_text, footer_text[:28]]
                if footer_text
                else [headline, emphasis_text]
            )[:4]
        if template == "metric_callout" and not supporting_lines:
            spec["supporting_lines"] = [truncate(card["sentence_text"], 72), truncate(card["context_text"], 72)]
        normalized.append(spec)
        last_end = end_sec
        if len(normalized) >= max_visuals:
            break
    return normalized


def fallback_visual_plan(
    cards: list[dict[str, Any]],
    clip_duration: float,
    max_visuals: int,
    min_visual_sec: float,
    max_visual_sec: float,
    scene_cuts: list[float],
) -> list[dict[str, Any]]:
    ranked = sorted(cards, key=lambda item: (item["priority"], -item["start"]), reverse=True)
    fallback = []
    for card in ranked:
        fallback.append(
            {
                "card_id": card["card_id"],
                "template": _default_template(card),
                "composition_mode": "picture_in_picture" if card["visual_type_hint"] in {"data_graphic", "product_ui"} else "replace",
                "headline": truncate(card["sentence_text"], 68),
                "emphasis_text": _extract_emphasis_text(card),
                "supporting_lines": [truncate(card["context_text"], 72)],
                "keywords": card["keywords"][:4],
                "steps": [truncate(term, 24) for term in card["keywords"][:3]],
                "quote_text": truncate(card["sentence_text"], 120),
                "footer_text": truncate(card["context_text"], 84),
                "left_label": "Before",
                "right_label": "After",
                "left_detail": truncate(card["sentence_text"], 72),
                "right_detail": truncate(card["context_text"], 72),
                "position": "bottom_right",
                "scale": 0.42,
                "rationale": "Fallback visual chosen from the strongest transcript sentence when the model plan was unavailable.",
                "confidence": round(min(max(card["priority"] / 88.0, 0.45), 0.9), 2),
            }
        )
        if len(fallback) >= max_visuals:
            break
    return _normalize_visual_plan(
        fallback,
        cards,
        clip_duration,
        max_visuals,
        min_visual_sec,
        max_visual_sec,
        scene_cuts,
    )


def analyze_visual_plan_with_llm(
    provider_name: str,
    model_name: str,
    cards: list[dict[str, Any]],
    clip_duration: float,
    max_visuals: int,
    min_visual_sec: float,
    max_visual_sec: float,
    scene_cuts: list[float],
) -> list[dict[str, Any]]:
    fallback = fallback_visual_plan(cards, clip_duration, max_visuals, min_visual_sec, max_visual_sec, scene_cuts)
    if not cards:
        return fallback
    template_lines = "\n".join(f"- {name}: {description}" for name, description in SUPPORTED_TEMPLATES.items())
    system_prompt = (
        "You are a senior motion graphics director planning precise generated visuals for an explainer video. "
        "Choose only transcript beats where a custom animation would make the spoken idea clearer. "
        "Prefer concise, literal, high-signal visuals. "
        "Return ONLY a JSON array with at most {count} objects using these keys: "
        "card_id, template, composition_mode, headline, emphasis_text, supporting_lines, steps, keywords, "
        "quote_text, left_label, right_label, left_detail, right_detail, footer_text, position, scale, rationale, confidence."
    ).format(count=max_visuals)
    user_prompt = (
        f"Video duration: {clip_duration:.2f}s\n"
        f"Max visuals: {max_visuals}\n"
        f"Duration per visual: {min_visual_sec:.1f}s to {max_visual_sec:.1f}s\n"
        f"Detected scene cuts: {scene_cuts[:24]}\n\n"
        f"Supported templates:\n{template_lines}\n\n"
        "Composition modes:\n"
        "- replace: full-screen generated cutaway\n"
        "- picture_in_picture: keep the source visible and place the visual in a corner\n\n"
        f"Transcript cards:\n{truncate(_format_cards_for_llm(cards), 7600)}\n\n"
        "Pick the strongest beats only. Avoid generic filler. "
        "If a sentence is abstract, choose keyword_stack or quote_focus. "
        "If it explains a process, choose timeline_steps. "
        "If it contrasts states, choose comparison_split. "
        "If it contains a number or strong claim, choose metric_callout. "
        "Return JSON array only."
    )
    try:
        raw_text = call_reasoning_model(provider_name, model_name, system_prompt, user_prompt)
        parsed = json.loads(extract_json_array(raw_text))
    except Exception:
        return fallback
    normalized = _normalize_visual_plan(
        parsed,
        cards,
        clip_duration,
        max_visuals,
        min_visual_sec,
        max_visual_sec,
        scene_cuts,
    )
    return normalized or fallback
