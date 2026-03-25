from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from google import genai
from google.genai import types

import config
from engine import apply_center_punch_ins, VideoEngineError, merge, probe_video, render_vertical_short, trim
from state import ProjectState, utc_now_iso
from tools.transcript import execute as transcribe
from tools.transcript_utils import optimize_caption_segments, parse_srt, write_srt_segments

VIRAL_TERMS = {
    "secret",
    "mistake",
    "mistakes",
    "crazy",
    "insane",
    "wild",
    "truth",
    "hack",
    "hacks",
    "controversial",
    "future",
    "never",
    "always",
    "easy",
    "hard",
    "why",
    "how",
    "biggest",
    "best",
    "worst",
    "nobody",
    "everyone",
    "million",
    "billion",
    "percent",
    "ai",
    "agent",
    "growth",
    "viral",
    "attention",
}
EMPHASIS_TERMS = {
    "must",
    "need",
    "important",
    "surprising",
    "unexpected",
    "warning",
    "problem",
    "opportunity",
    "proof",
    "story",
    "lesson",
    "formula",
    "framework",
    "strategy",
    "system",
    "trick",
    "tip",
}
PLATFORM_HASHTAGS = {
    "youtube_shorts": ["shorts", "youtubeshorts"],
    "tiktok": ["tiktok", "fyp"],
    "instagram_reels": ["reels", "instagramreels"],
}
VIRAL_SCORE_KEYS = ("hook_strength", "payoff", "novelty", "clarity", "shareability")
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "for",
    "with", "this", "that", "these", "those", "you", "your", "our", "their",
    "from", "into", "over", "under", "about", "just", "than", "then",
    "they", "them", "have", "has", "had", "was", "were", "are", "is",
    "be", "been", "being", "what", "when", "where", "which",
}


def _safe_stem(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip().lower())
    return re.sub(r"_+", "_", cleaned).strip("_") or "short"


def _extract_json_array(raw_text: str) -> str:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("The model did not return a JSON array.")
    return cleaned[start : end + 1]


def _extract_json_object(raw_text: str) -> str:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("The model did not return a JSON object.")
    return cleaned[start : end + 1]


def _truncate(text: str, limit: int) -> str:
    collapsed = re.sub(r"\s+", " ", text).strip()
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3].rstrip() + "..."


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9']+", text.lower())


def _heuristic_score(text: str, duration: float) -> float:
    tokens = _word_tokens(text)
    if not tokens:
        return 0.0
    lower = text.lower()
    unique_ratio = len(set(tokens)) / max(len(tokens), 1)
    numbers = len(re.findall(r"\b\d+(?:\.\d+)?%?\b", text))
    viral_hits = sum(1 for term in VIRAL_TERMS if term in lower)
    emphasis_hits = sum(1 for term in EMPHASIS_TERMS if term in lower)
    punctuation_hits = text.count("?") * 4 + text.count("!") * 2
    opener = " ".join(tokens[:12])
    opener_hits = sum(1 for term in VIRAL_TERMS | EMPHASIS_TERMS if term in opener)
    duration_penalty = abs(duration - 32.0) * 0.7
    score = (
        38.0
        + viral_hits * 6.0
        + emphasis_hits * 3.5
        + punctuation_hits
        + numbers * 2.5
        + opener_hits * 2.0
        + min(unique_ratio * 40.0, 18.0)
        - duration_penalty
    )
    return round(max(score, 1.0), 2)


def _clamp_score(value: float) -> int:
    return max(1, min(int(round(float(value))), 100))


def _heuristic_viral_score_breakdown(text: str, duration: float) -> dict[str, int]:
    tokens = _word_tokens(text)
    if not tokens:
        return {key: 1 for key in VIRAL_SCORE_KEYS} | {"overall": 1}
    lower = text.lower()
    unique_ratio = len(set(tokens)) / max(len(tokens), 1)
    numbers = len(re.findall(r"\b\d+(?:\.\d+)?%?\b", text))
    questions = text.count("?")
    exclaims = text.count("!")
    viral_hits = sum(1 for term in VIRAL_TERMS if term in lower)
    emphasis_hits = sum(1 for term in EMPHASIS_TERMS if term in lower)
    opener = " ".join(tokens[:12])
    opener_hits = sum(1 for term in VIRAL_TERMS | EMPHASIS_TERMS if term in opener)
    duration_fit = max(0.0, 1.0 - (abs(duration - 30.0) / 30.0))

    hook_strength = _clamp_score(42 + opener_hits * 11 + questions * 8 + exclaims * 4 + viral_hits * 5)
    payoff = _clamp_score(40 + emphasis_hits * 8 + numbers * 5 + unique_ratio * 26)
    novelty = _clamp_score(34 + viral_hits * 8 + numbers * 4 + unique_ratio * 22)
    clarity = _clamp_score(48 + duration_fit * 22 + unique_ratio * 18)
    shareability = _clamp_score(38 + viral_hits * 7 + questions * 5 + emphasis_hits * 4 + opener_hits * 5)
    overall = _clamp_score(
        hook_strength * 0.24
        + payoff * 0.22
        + novelty * 0.18
        + clarity * 0.18
        + shareability * 0.18
    )
    return {
        "overall": overall,
        "hook_strength": hook_strength,
        "payoff": payoff,
        "novelty": novelty,
        "clarity": clarity,
        "shareability": shareability,
    }


def _build_viral_explanations(text: str, duration: float, score_breakdown: dict[str, int]) -> list[str]:
    tokens = _word_tokens(text)
    lower = text.lower()
    numbers = len(re.findall(r"\b\d+(?:\.\d+)?%?\b", text))
    explanations: list[str] = []
    if any(term in lower for term in VIRAL_TERMS):
        explanations.append("Strong curiosity language gives the clip an immediate hook.")
    if numbers:
        explanations.append("Concrete numbers make the claim feel specific and easier to share.")
    if any(term in lower for term in EMPHASIS_TERMS):
        explanations.append("The transcript has a clear payoff or takeaway instead of vague chatter.")
    if 18.0 <= duration <= 45.0:
        explanations.append("The runtime fits short-form retention and replay behavior well.")
    if len(set(tokens)) / max(len(tokens), 1) > 0.72:
        explanations.append("The wording stays information-dense, which helps pacing and rewatch value.")
    if score_breakdown["hook_strength"] >= 80:
        explanations.append("The opener lands quickly enough to stop the scroll.")
    if score_breakdown["shareability"] >= 80:
        explanations.append("The idea is framed in a way viewers can easily quote or repost.")
    deduped: list[str] = []
    for explanation in explanations:
        if explanation not in deduped:
            deduped.append(explanation)
        if len(deduped) >= 4:
            break
    if not deduped:
        deduped.append("The clip is compact and understandable, which gives it baseline short-form potential.")
    return deduped


def _clip_transcript_text(segments: list[dict[str, float | str]]) -> str:
    return " ".join(str(segment["text"]).strip() for segment in segments if str(segment["text"]).strip()).strip()


def _fallback_viral_analysis(candidate: dict, selection: dict, clip_segments: list[dict[str, float | str]]) -> dict:
    transcript_text = _clip_transcript_text(clip_segments) or str(candidate.get("excerpt") or "")
    score_breakdown = _heuristic_viral_score_breakdown(transcript_text, float(candidate["duration"]))
    explanation = _build_viral_explanations(transcript_text, float(candidate["duration"]), score_breakdown)
    explanation.append(_truncate(str(selection.get("reason") or ""), 150)) if selection.get("reason") else None
    explanation = [item for item in explanation if item]
    return {
        "viral_score": score_breakdown,
        "viral_explanation": explanation[:4],
    }


def _normalize_viral_analysis(raw_analysis: dict, fallback: dict) -> dict:
    raw_scores = raw_analysis.get("viral_score") or {}
    normalized_scores = {
        key: _clamp_score(raw_scores.get(key, fallback["viral_score"][key]))
        for key in ("overall",) + VIRAL_SCORE_KEYS
    }
    explanation = [
        _truncate(str(item).strip(), 140)
        for item in raw_analysis.get("viral_explanation", [])
        if str(item).strip()
    ]
    if not explanation:
        explanation = list(fallback["viral_explanation"])
    return {
        "viral_score": normalized_scores,
        "viral_explanation": explanation[:4],
    }




def _format_timestamped_clip_segments(segments: list[dict[str, float | str]]) -> str:
    return "\n".join(
        f"{float(segment['start']):.2f}-{float(segment['end']):.2f}: {str(segment['text']).strip()}"
        for segment in segments
        if str(segment["text"]).strip()
    )


def _keyword_phrase(text: str, limit: int = 5) -> str:
    words: list[str] = []
    for token in _word_tokens(text):
        if token in STOPWORDS or len(token) < 3:
            continue
        if token not in words:
            words.append(token)
        if len(words) >= limit:
            break
    return " ".join(words) or _truncate(text, 50)


def _fallback_b_roll_suggestions(clip_segments: list[dict[str, float | str]]) -> list[dict]:
    suggestions: list[dict] = []
    last_end = -999.0
    for segment in clip_segments:
        text = str(segment["text"]).strip()
        if not text:
            continue
        start_sec = float(segment["start"])
        end_sec = float(segment["end"])
        if start_sec - last_end < 1.0:
            continue
        lower = text.lower()
        if any(term in lower for term in {"chart", "metric", "growth", "percent", "data", "revenue", "users"}):
            visual_type = "data_graphic"
            direction = "Show a quick chart, number card, or graph that reinforces the spoken metric."
        elif any(term in lower for term in {"tool", "app", "product", "website", "dashboard", "workflow", "agent"}):
            visual_type = "product_ui"
            direction = "Show a UI clip, dashboard screenshot, or product walkthrough beat tied to the claim."
        elif any(term in lower for term in {"customer", "team", "founder", "people", "creator", "audience"}):
            visual_type = "cutaway"
            direction = "Use a human cutaway or reaction-style insert that supports the point without covering captions."
        else:
            visual_type = "text_overlay"
            direction = "Use a quick illustrative insert or animated text card to underline the spoken takeaway."
        clip_end = min(end_sec, start_sec + 2.4)
        suggestions.append(
            {
                "start": round(max(start_sec, 0.0), 2),
                "end": round(max(clip_end, start_sec + 0.8), 2),
                "visual_type": visual_type,
                "search_query": _truncate(_keyword_phrase(text, limit=6), 70),
                "direction": _truncate(direction, 110),
                "rationale": _truncate("This beat has enough semantic density to support a reinforcing visual without distracting from the spoken payoff.", 130),
            }
        )
        last_end = clip_end
        if len(suggestions) >= 3:
            break
    if not suggestions and clip_segments:
        first = clip_segments[0]
        suggestions.append(
            {
                "start": round(float(first["start"]), 2),
                "end": round(min(float(first["end"]), float(first["start"]) + 2.0), 2),
                "visual_type": "text_overlay",
                "search_query": _truncate(_keyword_phrase(str(first["text"]), limit=6), 70),
                "direction": "Use a simple reinforcing visual or title card that clarifies the core point.",
                "rationale": "The opener is the safest place to reinforce context if no stronger B-roll beat stands out.",
            }
        )
    return suggestions


def _normalize_b_roll_suggestions(raw_suggestions: list[dict], fallback: list[dict], clip_duration: float) -> list[dict]:
    suggestions: list[dict] = []
    source = raw_suggestions or fallback
    for item in source:
        try:
            start_sec = max(0.0, min(float(item.get("start", 0.0)), clip_duration))
            end_sec = max(start_sec + 0.3, min(float(item.get("end", start_sec + 1.8)), clip_duration))
        except Exception:
            continue
        if end_sec <= start_sec:
            continue
        suggestions.append(
            {
                "start": round(start_sec, 2),
                "end": round(end_sec, 2),
                "visual_type": _truncate(str(item.get("visual_type") or "text_overlay"), 32),
                "search_query": _truncate(str(item.get("search_query") or "supporting visual"), 70),
                "direction": _truncate(str(item.get("direction") or "Add a supporting visual beat."), 110),
                "rationale": _truncate(str(item.get("rationale") or "Supports the spoken point visually."), 130),
            }
        )
        if len(suggestions) >= 4:
            break
    return suggestions or fallback[:3]


def _analyze_b_roll_with_llm(
    provider_name: str,
    model_name: str,
    candidate: dict,
    selection: dict,
    clip_segments: list[dict[str, float | str]],
    target_platform: str,
) -> list[dict]:
    fallback = _fallback_b_roll_suggestions(clip_segments)
    transcript_text = _clip_transcript_text(clip_segments) or str(candidate.get("excerpt") or "")
    timestamped_transcript = _format_timestamped_clip_segments(clip_segments)
    system_prompt = (
        "You are a short-form producer. Suggest strong B-roll beats for a short clip. "
        "Return ONLY a JSON array of up to 4 objects with keys start, end, visual_type, search_query, direction, rationale."
    )
    user_prompt = (
        f"Platform: {target_platform}\n"
        f"Candidate window: {candidate['start']:.2f}-{candidate['end']:.2f} ({candidate['duration']:.2f}s)\n"
        f"Draft title: {selection.get('title', '')}\n"
        f"Draft hook: {selection.get('hook', '')}\n\n"
        f"Transcript overview:\n{_truncate(transcript_text, 2200)}\n\n"
        f"Timestamped transcript:\n{_truncate(timestamped_transcript, 2600)}\n\n"
        "Prefer supportive visuals that reinforce the point without covering captions or feeling generic. Return JSON array only."
    )
    try:
        raw_text = _call_reasoning_model(provider_name, model_name, system_prompt, user_prompt)
        parsed = json.loads(_extract_json_array(raw_text))
    except Exception:
        return fallback
    return _normalize_b_roll_suggestions(parsed, fallback, clip_duration=float(candidate["duration"]))



def _fallback_punch_in_moments(clip_segments: list[dict[str, float | str]]) -> list[dict]:
    moments: list[dict] = []
    last_end = -999.0
    for segment in clip_segments:
        text = str(segment["text"]).strip()
        if not text:
            continue
        start_sec = float(segment["start"])
        end_sec = float(segment["end"])
        lower = text.lower()
        emphasis_score = 0
        emphasis_score += text.count("?") * 2
        emphasis_score += text.count("!")
        emphasis_score += sum(1 for term in EMPHASIS_TERMS if term in lower)
        emphasis_score += sum(1 for term in VIRAL_TERMS if term in lower)
        emphasis_score += len(re.findall(r"\b\d+(?:\.\d+)?%?\b", text))
        if emphasis_score < 2 or start_sec - last_end < 0.8:
            continue
        clip_end = min(end_sec, start_sec + 1.7)
        moments.append(
            {
                "start": round(max(start_sec, 0.0), 2),
                "end": round(max(clip_end, start_sec + 0.7), 2),
                "zoom": round(min(1.08 + emphasis_score * 0.015, 1.18), 2),
                "reason": _truncate("Emphasis-heavy line with a likely payoff beat.", 90),
            }
        )
        last_end = clip_end
        if len(moments) >= 3:
            break
    return moments


def _normalize_punch_in_moments(raw_moments: list[dict], fallback: list[dict], clip_duration: float) -> list[dict]:
    moments: list[dict] = []
    source = raw_moments or fallback
    for item in source:
        try:
            start_sec = max(0.0, min(float(item.get("start", 0.0)), clip_duration))
            end_sec = max(start_sec + 0.3, min(float(item.get("end", start_sec + 1.2)), clip_duration))
            zoom = max(1.05, min(float(item.get("zoom", 1.12)), 1.22))
        except Exception:
            continue
        if end_sec <= start_sec:
            continue
        moments.append(
            {
                "start": round(start_sec, 2),
                "end": round(end_sec, 2),
                "zoom": round(zoom, 2),
                "reason": _truncate(str(item.get("reason") or "Punch in on an emphasis beat."), 90),
            }
        )
        if len(moments) >= 4:
            break
    return moments


def _analyze_punch_in_with_llm(
    provider_name: str,
    model_name: str,
    candidate: dict,
    selection: dict,
    clip_segments: list[dict[str, float | str]],
    target_platform: str,
) -> list[dict]:
    fallback = _fallback_punch_in_moments(clip_segments)
    if not clip_segments:
        return fallback
    transcript_text = _clip_transcript_text(clip_segments) or str(candidate.get("excerpt") or "")
    timestamped_transcript = _format_timestamped_clip_segments(clip_segments)
    system_prompt = (
        "You are a short-form editor. Pick the best center punch-in moments for emphasis in a short clip. "
        "Return ONLY a JSON array of up to 4 objects with keys start, end, zoom, reason."
    )
    user_prompt = (
        f"Platform: {target_platform}\n"
        f"Candidate window: {candidate['start']:.2f}-{candidate['end']:.2f} ({candidate['duration']:.2f}s)\n"
        f"Draft title: {selection.get('title', '')}\n"
        f"Draft hook: {selection.get('hook', '')}\n\n"
        f"Transcript overview:\n{_truncate(transcript_text, 2200)}\n\n"
        f"Timestamped transcript:\n{_truncate(timestamped_transcript, 2600)}\n\n"
        "Choose moments where a subtle center punch-in would increase emphasis without feeling overedited. Return JSON array only."
    )
    try:
        raw_text = _call_reasoning_model(provider_name, model_name, system_prompt, user_prompt)
        parsed = json.loads(_extract_json_array(raw_text))
    except Exception:
        return fallback
    return _normalize_punch_in_moments(parsed, fallback, clip_duration=float(candidate["duration"]))

def _overlap_ratio(first: dict, second: dict) -> float:
    overlap = max(0.0, min(first["end"], second["end"]) - max(first["start"], second["start"]))
    if overlap <= 0:
        return 0.0
    shortest = min(first["end"] - first["start"], second["end"] - second["start"])
    return overlap / max(shortest, 0.001)


def _dedupe_candidates(candidates: list[dict], limit: int) -> list[dict]:
    selected: list[dict] = []
    for candidate in candidates:
        if all(_overlap_ratio(candidate, existing) < 0.68 for existing in selected):
            selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected


def _build_candidates(
    segments: list[dict[str, float | str]],
    min_duration_sec: float,
    max_duration_sec: float,
    limit: int = 28,
) -> list[dict]:
    candidates: list[dict] = []
    candidate_index = 1
    for start_index in range(len(segments)):
        start_sec = float(segments[start_index]["start"])
        text_parts: list[str] = []
        for end_index in range(start_index, len(segments)):
            segment = segments[end_index]
            text_parts.append(str(segment["text"]).strip())
            end_sec = float(segment["end"])
            duration = end_sec - start_sec
            if duration > max_duration_sec and end_index > start_index:
                break
            if duration < min_duration_sec:
                continue
            text = " ".join(part for part in text_parts if part).strip()
            if len(_word_tokens(text)) < 12:
                continue
            candidates.append(
                {
                    "candidate_id": f"cand_{candidate_index:02d}",
                    "start": round(start_sec, 2),
                    "end": round(end_sec, 2),
                    "duration": round(duration, 2),
                    "excerpt": _truncate(text, 320),
                    "heuristic_score": _heuristic_score(text, duration),
                }
            )
            candidate_index += 1
    if not candidates and segments:
        start_sec = float(segments[0]["start"])
        end_sec = float(segments[-1]["end"])
        text = " ".join(str(segment["text"]).strip() for segment in segments)
        candidates.append(
            {
                "candidate_id": "cand_01",
                "start": round(start_sec, 2),
                "end": round(end_sec, 2),
                "duration": round(end_sec - start_sec, 2),
                "excerpt": _truncate(text, 320),
                "heuristic_score": _heuristic_score(text, end_sec - start_sec),
            }
        )
    candidates.sort(key=lambda item: item["heuristic_score"], reverse=True)
    return _dedupe_candidates(candidates, limit=limit)


def _format_candidates_for_llm(candidates: list[dict]) -> str:
    lines: list[str] = []
    for candidate in candidates:
        lines.append(
            "\n".join(
                [
                    (
                        f"{candidate['candidate_id']} | {candidate['start']:.2f}-{candidate['end']:.2f} "
                        f"({candidate['duration']:.2f}s) | heuristic={candidate['heuristic_score']:.2f}"
                    ),
                    f"Excerpt: {candidate['excerpt']}",
                ]
            )
        )
    return "\n\n".join(lines)


def _call_reasoning_model(provider_name: str, model_name: str, system_prompt: str, user_prompt: str) -> str:
    if provider_name == "claude":
        from anthropic import Anthropic

        client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=model_name or config.CLAUDE_MODEL,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(block.text for block in response.content if getattr(block, "type", "") == "text")

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    response = client.models.generate_content(
        model=model_name or config.GEMINI_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return getattr(response, "text", "") or ""


def _default_title(candidate: dict) -> str:
    words = _word_tokens(candidate["excerpt"])
    if not words:
        return "High-signal short"
    return _truncate(" ".join(words[:8]).title(), 60)


def _default_hook(candidate: dict) -> str:
    sentence = candidate["excerpt"].split(".", 1)[0].strip()
    if len(sentence) >= 12:
        return _truncate(sentence, 90)
    return _truncate(candidate["excerpt"], 90)


def _fallback_selections(candidates: list[dict], count: int) -> list[dict]:
    selections: list[dict] = []
    for candidate in candidates[:count]:
        selections.append(
            {
                "candidate_id": candidate["candidate_id"],
                "score": round(min(candidate["heuristic_score"] + 8.0, 100.0), 2),
                "title": _default_title(candidate),
                "hook": _default_hook(candidate),
                "reason": "Selected from the top transcript windows using heuristic engagement signals.",
                "keywords": _word_tokens(candidate["excerpt"])[:5],
            }
        )
    return selections


def _select_shorts_with_llm(
    provider_name: str,
    model_name: str,
    candidates: list[dict],
    transcript_text: str,
    count: int,
    min_duration_sec: float,
    max_duration_sec: float,
    target_platform: str,
) -> list[dict]:
    candidate_map = {candidate["candidate_id"]: candidate for candidate in candidates}
    system_prompt = (
        "You are a short-form video strategist. Choose the most clip-worthy windows from a transcript candidate list. "
        "Prioritize sharp hooks, strong payoffs, novelty, specificity, controversy, and replay value. Diversify topics and avoid near-duplicates. "
        "Return ONLY a JSON array of objects with keys: candidate_id, score, title, hook, reason, keywords."
    )
    user_prompt = (
        f"Target platform: {target_platform}.\n"
        f"Need exactly {count} shorts. Each clip should stay between {min_duration_sec} and {max_duration_sec} seconds.\n\n"
        f"Transcript overview:\n{_truncate(transcript_text, 3500)}\n\n"
        f"Candidate windows:\n{_format_candidates_for_llm(candidates)}\n\n"
        "Choose the best candidates for viral-style shorts. Keep titles punchy, hooks conversational, reasons concrete, and keywords platform-friendly. "
        "Return JSON array only."
    )
    raw_text = _call_reasoning_model(provider_name, model_name, system_prompt, user_prompt)
    parsed = json.loads(_extract_json_array(raw_text))
    selections: list[dict] = []
    seen_ids: set[str] = set()
    for item in parsed:
        candidate_id = str(item.get("candidate_id", "")).strip()
        if candidate_id not in candidate_map or candidate_id in seen_ids:
            continue
        seen_ids.add(candidate_id)
        candidate = candidate_map[candidate_id]
        keywords = [str(keyword).strip() for keyword in item.get("keywords", []) if str(keyword).strip()]
        selections.append(
            {
                "candidate_id": candidate_id,
                "score": float(item.get("score", candidate["heuristic_score"])),
                "title": _truncate(str(item.get("title") or _default_title(candidate)), 72),
                "hook": _truncate(str(item.get("hook") or _default_hook(candidate)), 120),
                "reason": _truncate(str(item.get("reason") or "Strong transcript hook and payoff."), 220),
                "keywords": keywords[:6],
            }
        )
    return selections[:count]


def _analyze_viral_score_with_llm(
    provider_name: str,
    model_name: str,
    candidate: dict,
    selection: dict,
    clip_segments: list[dict[str, float | str]],
    target_platform: str,
) -> dict:
    fallback = _fallback_viral_analysis(candidate, selection, clip_segments)
    transcript_text = _clip_transcript_text(clip_segments) or str(candidate.get("excerpt") or "")
    system_prompt = (
        "You are a short-form video analyst. Score a short clip for short-form virality with concrete explainability. "
        "Return ONLY a JSON object with keys viral_score and viral_explanation. "
        "viral_score must contain overall, hook_strength, payoff, novelty, clarity, and shareability as integers from 1 to 100. "
        "viral_explanation must be an array of 3 or 4 concise bullet-style strings."
    )
    user_prompt = (
        f"Platform: {target_platform}\n"
        f"Candidate window: {candidate['start']:.2f}-{candidate['end']:.2f} ({candidate['duration']:.2f}s)\n"
        f"Draft title: {selection.get('title', '')}\n"
        f"Draft hook: {selection.get('hook', '')}\n"
        f"Why selected: {selection.get('reason', '')}\n\n"
        f"Transcript:\n{_truncate(transcript_text, 2400)}\n\n"
        "Return JSON only."
    )
    try:
        raw_text = _call_reasoning_model(provider_name, model_name, system_prompt, user_prompt)
        parsed = json.loads(_extract_json_object(raw_text))
    except Exception:
        return fallback
    return _normalize_viral_analysis(parsed, fallback)


def _clip_transcript_segments(
    segments: list[dict[str, float | str]],
    start_sec: float,
    end_sec: float,
) -> list[dict[str, float | str]]:
    clipped: list[dict[str, float | str]] = []
    for segment in segments:
        segment_start = float(segment["start"])
        segment_end = float(segment["end"])
        if segment_end <= start_sec or segment_start >= end_sec:
            continue
        clipped.append(
            {
                "start": round(max(segment_start, start_sec) - start_sec, 3),
                "end": round(min(segment_end, end_sec) - start_sec, 3),
                "text": str(segment["text"]).strip(),
            }
        )
    return [segment for segment in clipped if float(segment["end"]) > float(segment["start"])]


def _hashtags(keywords: list[str], target_platform: str) -> list[str]:
    tags: list[str] = []
    seen: set[str] = set()
    for keyword in PLATFORM_HASHTAGS.get(target_platform, []):
        normalized = _safe_stem(keyword).replace("_", "")
        if normalized and normalized not in seen:
            seen.add(normalized)
            tags.append(f"#{normalized}")
    for keyword in keywords:
        normalized = _safe_stem(keyword).replace("_", "")
        if normalized and normalized not in seen:
            seen.add(normalized)
            tags.append(f"#{normalized}")
        if len(tags) >= 8:
            break
    return tags


def _bundle_readme(project_name: str, manifest: dict) -> str:
    lines = [
        "# Auto Shorts Package",
        "",
        f"Project: {project_name}",
        f"Platform profile: {manifest['target_platform']}",
        f"Generated at: {manifest['created_at']}",
        f"Source video: {manifest['source_video']}",
        "",
        f"Shorts created: {len(manifest['shorts'])}",
        "",
    ]
    for item in manifest["shorts"]:
        lines.extend(
            [
                f"## {item['rank']}. {item['title']}",
                f"- Window: {item['start']}s to {item['end']}s",
                f"- Duration: {item['duration']}s",
                f"- Score: {item['score']}",
                f"- Viral score: {item['viral_score']['overall']}",
                f"- Hook: {item['hook']}",
                f"- Why it works: {item['reason']}",
                "- Viral explainability:",
            ]
        )
        for explanation in item.get("viral_explanation", []):
            lines.append(f"  - {explanation}")
        if item.get("b_roll_suggestions"):
            lines.append("- B-roll suggestions:")
            for suggestion in item["b_roll_suggestions"]:
                lines.append(
                    "  - "
                    f"{suggestion['start']}s-{suggestion['end']}s | {suggestion['visual_type']} | "
                    f"{suggestion['search_query']} | {suggestion['direction']}"
                )
        if item.get("punch_in_moments"):
            lines.append("- Punch-in moments:")
            for moment in item["punch_in_moments"]:
                lines.append(
                    "  - "
                    f"{moment['start']}s-{moment['end']}s | zoom {moment['zoom']}x | {moment['reason']}"
                )
        lines.extend([f"- Deliverable: {item['vertical_video_path']}", ""])
    if manifest.get("compilation_path"):
        lines.extend([f"Compilation: {manifest['compilation_path']}", ""])
    return "\n".join(lines)


def execute(params: dict, state: ProjectState) -> dict:
    transcript_path = Path(state.working_dir) / "transcript.txt"
    srt_path = Path(state.working_dir) / "transcript.srt"
    if not transcript_path.is_file() or not srt_path.is_file():
        transcribe_result = transcribe({}, state)
        state = transcribe_result["updated_state"]
        if not transcribe_result["success"]:
            return {
                "success": False,
                "message": transcribe_result["message"],
                "suggestion": None,
                "updated_state": state,
                "tool_name": "create_auto_shorts",
            }

    transcript_text = transcript_path.read_text(encoding="utf-8").strip()
    transcript_segments = parse_srt(srt_path)
    if not transcript_text or not transcript_segments:
        return {
            "success": False,
            "message": "Transcript generation succeeded, but no usable timestamped transcript segments were found.",
            "suggestion": None,
            "updated_state": state,
            "tool_name": "create_auto_shorts",
        }

    count = max(1, min(int(params.get("count", 3)), 8))
    min_duration_sec = max(12.0, float(params.get("min_duration_sec", 20.0)))
    max_duration_sec = max(min_duration_sec + 2.0, min(float(params.get("max_duration_sec", 45.0)), 90.0))
    include_compilation = bool(params.get("include_compilation", True))
    target_platform = str(params.get("target_platform", "youtube_shorts")).strip().lower()
    if target_platform not in {"youtube_shorts", "tiktok", "instagram_reels"}:
        target_platform = "youtube_shorts"

    candidates = _build_candidates(transcript_segments, min_duration_sec, max_duration_sec)
    if not candidates:
        return {
            "success": False,
            "message": "No viable short-form clip windows were found in the transcript.",
            "suggestion": None,
            "updated_state": state,
            "tool_name": "create_auto_shorts",
        }

    provider_name = (state.provider or config.PROVIDER or "gemini").strip().lower()
    if provider_name not in {"gemini", "claude"}:
        provider_name = "gemini"
    model_name = state.model or (
        config.CLAUDE_MODEL if provider_name == "claude" else config.GEMINI_MODEL
    )

    try:
        selections = _select_shorts_with_llm(
            provider_name=provider_name,
            model_name=model_name,
            candidates=candidates,
            transcript_text=transcript_text,
            count=min(count, len(candidates)),
            min_duration_sec=min_duration_sec,
            max_duration_sec=max_duration_sec,
            target_platform=target_platform,
        )
    except Exception:
        selections = []
    if not selections:
        selections = _fallback_selections(candidates, count=min(count, len(candidates)))

    candidate_map = {candidate["candidate_id"]: candidate for candidate in candidates}
    timestamp_label = utc_now_iso().replace(":", "-").replace("+00:00", "Z")
    bundle_dir = Path(state.output_dir) / f"{_safe_stem(state.project_name)}_auto_shorts_{timestamp_label}"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    created_shorts: list[dict] = []
    vertical_paths: list[str] = []
    failures: list[str] = []

    for rank, selection in enumerate(selections, start=1):
        candidate = candidate_map.get(selection["candidate_id"])
        if candidate is None:
            continue
        short_dir = bundle_dir / f"{rank:02d}_{_safe_stem(selection['title'])[:48]}"
        short_dir.mkdir(parents=True, exist_ok=True)
        try:
            raw_temp_path = trim(
                state.working_file,
                state.working_dir,
                float(candidate["start"]),
                float(candidate["end"]),
            )
            raw_clip_path = short_dir / "raw_clip.mp4"
            shutil.copy2(raw_temp_path, raw_clip_path)

            clip_segments = _clip_transcript_segments(
                transcript_segments,
                start_sec=float(candidate["start"]),
                end_sec=float(candidate["end"]),
            )
            transcript_txt_path = short_dir / "transcript.txt"
            transcript_txt_path.write_text(
                " ".join(str(segment["text"]).strip() for segment in clip_segments).strip() + "\n",
                encoding="utf-8",
            )
            caption_segments = optimize_caption_segments(clip_segments)
            captions_path = short_dir / "captions.srt"
            if caption_segments:
                write_srt_segments(captions_path, caption_segments)
                captions_arg = str(captions_path)
            else:
                captions_arg = None

            viral_analysis = _analyze_viral_score_with_llm(
                provider_name=provider_name,
                model_name=model_name,
                candidate=candidate,
                selection=selection,
                clip_segments=clip_segments,
                target_platform=target_platform,
            )
            b_roll_suggestions = _analyze_b_roll_with_llm(
                provider_name=provider_name,
                model_name=model_name,
                candidate=candidate,
                selection=selection,
                clip_segments=clip_segments,
                target_platform=target_platform,
            )
            punch_in_moments = _analyze_punch_in_with_llm(
                provider_name=provider_name,
                model_name=model_name,
                candidate=candidate,
                selection=selection,
                clip_segments=clip_segments,
                target_platform=target_platform,
            )
            motion_input_path = apply_center_punch_ins(
                str(raw_clip_path),
                state.working_dir,
                punch_in_moments,
            )
            motion_clip_path = None
            if motion_input_path != str(raw_clip_path):
                motion_clip_path = short_dir / "punch_in_clip.mp4"
                shutil.copy2(motion_input_path, motion_clip_path)

            vertical_temp_path = render_vertical_short(
                motion_input_path,
                state.working_dir,
                srt_path=captions_arg,
            )
            vertical_video_path = short_dir / f"{rank:02d}_{_safe_stem(selection['title'])}_{target_platform}.mp4"
            shutil.copy2(vertical_temp_path, vertical_video_path)
            vertical_paths.append(str(vertical_video_path))
            metadata = probe_video(str(vertical_video_path))
            hashtags = _hashtags(selection.get("keywords", []), target_platform)
            short_record = {
                "rank": rank,
                "title": selection["title"],
                "hook": selection["hook"],
                "reason": selection["reason"],
                "score": round(float(selection["score"]), 2),
                "viral_score": viral_analysis["viral_score"],
                "viral_explanation": viral_analysis["viral_explanation"],
                "b_roll_suggestions": b_roll_suggestions,
                "punch_in_moments": punch_in_moments,
                "motion_clip_path": str(motion_clip_path) if motion_clip_path else None,
                "start": round(float(candidate["start"]), 2),
                "end": round(float(candidate["end"]), 2),
                "duration": round(float(candidate["duration"]), 2),
                "heuristic_score": round(float(candidate["heuristic_score"]), 2),
                "keywords": selection.get("keywords", []),
                "hashtags": hashtags,
                "raw_clip_path": str(raw_clip_path),
                "vertical_video_path": str(vertical_video_path),
                "captions_path": str(captions_path) if clip_segments else None,
                "transcript_path": str(transcript_txt_path),
                "resolution": f"{metadata.get('width', 0)}x{metadata.get('height', 0)}",
            }
            (short_dir / "metadata.json").write_text(json.dumps(short_record, indent=2), encoding="utf-8")
            (short_dir / "broll_suggestions.json").write_text(
                json.dumps(b_roll_suggestions, indent=2),
                encoding="utf-8",
            )
            (short_dir / "punch_in_plan.json").write_text(
                json.dumps(punch_in_moments, indent=2),
                encoding="utf-8",
            )
            note_lines = [
                f"# {selection['title']}",
                "",
                f"Hook: {selection['hook']}",
                "",
                f"Why it works: {selection['reason']}",
                "",
                f"Viral score: {viral_analysis['viral_score']['overall']}",
                (
                    "Score breakdown: "
                    f"hook={viral_analysis['viral_score']['hook_strength']}, "
                    f"payoff={viral_analysis['viral_score']['payoff']}, "
                    f"novelty={viral_analysis['viral_score']['novelty']}, "
                    f"clarity={viral_analysis['viral_score']['clarity']}, "
                    f"shareability={viral_analysis['viral_score']['shareability']}"
                ),
                "",
                "Viral explainability:",
            ]
            for explanation in viral_analysis["viral_explanation"]:
                note_lines.append(f"- {explanation}")
            if b_roll_suggestions:
                note_lines.extend(["", "B-roll suggestions:"])
                for suggestion in b_roll_suggestions:
                    note_lines.append(
                        "- "
                        f"{suggestion['start']}s-{suggestion['end']}s | {suggestion['visual_type']} | "
                        f"query: {suggestion['search_query']} | {suggestion['direction']}"
                    )
            if punch_in_moments:
                note_lines.extend(["", "Punch-in moments:"])
                for moment in punch_in_moments:
                    note_lines.append(
                        "- "
                        f"{moment['start']}s-{moment['end']}s | zoom {moment['zoom']}x | {moment['reason']}"
                    )
            note_lines.extend(["", f"Suggested hashtags: {' '.join(hashtags)}"])
            (short_dir / "notes.md").write_text("\n".join(note_lines) + "\n", encoding="utf-8")
            created_shorts.append(short_record)
        except VideoEngineError as exc:
            failures.append(f"{selection['title']}: {exc}")

    if not created_shorts:
        return {
            "success": False,
            "message": "Auto shorts analysis completed, but FFmpeg failed to render every selected clip.",
            "suggestion": None,
            "updated_state": state,
            "tool_name": "create_auto_shorts",
        }

    compilation_path = None
    if include_compilation and len(vertical_paths) > 1:
        try:
            compilation_temp_path = merge(vertical_paths, state.working_dir)
            compilation_path = bundle_dir / "all_shorts_compilation.mp4"
            shutil.copy2(compilation_temp_path, compilation_path)
        except VideoEngineError as exc:
            failures.append(f"Compilation: {exc}")

    manifest = {
        "created_at": utc_now_iso(),
        "project_id": state.project_id,
        "project_name": state.project_name,
        "source_video": state.working_file,
        "target_platform": target_platform,
        "shorts": created_shorts,
        "candidate_count": len(candidates),
        "bundle_dir": str(bundle_dir),
        "compilation_path": str(compilation_path) if compilation_path else None,
        "transcript_path": str(transcript_path),
        "srt_path": str(srt_path),
        "failures": failures,
    }
    manifest_path = bundle_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (bundle_dir / "README.md").write_text(_bundle_readme(state.project_name, manifest) + "\n", encoding="utf-8")

    state.artifacts["latest_auto_shorts"] = {
        "created_at": manifest["created_at"],
        "manifest_path": str(manifest_path),
        "bundle_dir": str(bundle_dir),
        "count": len(created_shorts),
        "target_platform": target_platform,
    }
    history = list(state.artifacts.get("auto_shorts_history") or [])
    history.append(state.artifacts["latest_auto_shorts"])
    state.artifacts["auto_shorts_history"] = history[-10:]
    state.save()

    titles = ", ".join(item["title"] for item in created_shorts)
    failure_suffix = f" Failed extras: {'; '.join(failures)}" if failures else ""
    return {
        "success": True,
        "message": (
            f"Created {len(created_shorts)} auto shorts in {bundle_dir}. "
            f"Top picks: {titles}. Manifest: {manifest_path}.{failure_suffix}"
        ),
        "suggestion": None,
        "updated_state": state,
        "tool_name": "create_auto_shorts",
    }
