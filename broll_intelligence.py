from __future__ import annotations

import json
import ipaddress
import math
import re
import socket
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx
from google import genai
from google.genai import errors as genai_errors

import config

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "for",
    "with", "this", "that", "these", "those", "you", "your", "our", "their",
    "from", "into", "over", "under", "about", "just", "than", "then",
    "they", "them", "have", "has", "had", "was", "were", "are", "is",
    "be", "been", "being", "what", "when", "where", "which", "there",
    "really", "very", "actually", "kind", "sort",
}
VISUAL_TYPE_HINTS = {
    "data_graphic": ["analytics dashboard", "data chart", "business graph"],
    "product_ui": ["software dashboard", "app interface", "product workflow"],
    "cutaway": ["person working laptop", "team office", "meeting collaboration"],
    "process": ["hands working", "editing process", "workflow close up"],
    "location": ["office exterior", "warehouse interior", "studio workspace"],
    "abstract_motion": ["technology abstract", "cinematic background", "digital motion"],
}
MAX_STOCK_DOWNLOAD_BYTES = 512 * 1024 * 1024
MAX_STOCK_JSON_BYTES = 5 * 1024 * 1024
MAX_PEXELS_JSON_BYTES = MAX_STOCK_JSON_BYTES
PEXELS_API_HOST = "api.pexels.com"
PIXABAY_API_HOST = "pixabay.com"
COVERR_API_HOST = "api.coverr.co"
DEFAULT_STOCK_PROVIDER_ORDER = ("pexels", "pixabay", "coverr")
STOCK_PROVIDER_DISPLAY_NAMES = {
    "pexels": "Pexels",
    "pixabay": "Pixabay",
    "coverr": "Coverr",
}
STOCK_PROVIDER_KEY_NAMES = {
    "pexels": "PEXELS_API_KEY",
    "pixabay": "PIXABAY_API_KEY",
    "coverr": "COVERR_API_KEY",
}
VISUAL_KEYWORDS = {
    "data_graphic": {"data", "metric", "chart", "analytics", "revenue", "percent", "growth", "number"},
    "product_ui": {"app", "product", "website", "dashboard", "software", "workflow", "tool", "platform"},
    "cutaway": {"customer", "team", "founder", "people", "person", "creator", "audience", "meeting"},
    "process": {"build", "process", "system", "editing", "writing", "typing", "making"},
    "location": {"office", "factory", "city", "store", "studio", "room", "desk", "street"},
}
ABSTRACT_TERMS = {
    "mindset", "future", "idea", "concept", "strategy", "system", "growth", "attention", "belief",
    "lesson", "framework", "motivation", "creative", "thinking", "focus", "productivity",
}


@dataclass(frozen=True)
class StockBrollCandidate:
    provider: str
    provider_display_name: str
    provider_id: str
    title: str
    description: str
    tags: list[str]
    duration: float
    source_url: str
    download_url: str
    preview_url: str
    creator_name: str
    creator_url: str
    license_name: str
    license_url: str
    attribution_required: bool
    width: int
    height: int
    fps: float = 0.0
    quality: str = ""
    size_bytes: int = 0
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["file_info"] = {
            "id": self.provider_id,
            "link": self.download_url,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "quality": self.quality,
            "size": self.size_bytes,
            "file_type": "video/mp4",
        }
        return payload


def truncate(text: str, limit: int) -> str:
    collapsed = re.sub(r"\s+", " ", text).strip()
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3].rstrip() + "..."


def word_tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9']+", text.lower())


def semantic_keywords(text: str, limit: int = 8) -> list[str]:
    keywords: list[str] = []
    for token in word_tokens(text):
        if token in STOPWORDS or len(token) < 3:
            continue
        if token not in keywords:
            keywords.append(token)
        if len(keywords) >= limit:
            break
    return keywords


def keyword_phrase(text: str, limit: int = 5) -> str:
    keywords = semantic_keywords(text, limit=limit)
    return " ".join(keywords) or truncate(text, 50)


def safe_stem(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "project"


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def _as_int(value: Any, default: int = 0) -> int:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return default
    return number


def _split_tags(value: Any) -> list[str]:
    if isinstance(value, list):
        raw_tags = value
    else:
        raw_tags = str(value or "").split(",")
    tags: list[str] = []
    for item in raw_tags:
        tag = re.sub(r"\s+", " ", str(item).strip())
        if tag and tag.lower() not in {existing.lower() for existing in tags}:
            tags.append(truncate(tag, 48))
    return tags[:16]


def _provider_api_key(provider_name: str) -> str | None:
    if provider_name == "pexels":
        return config.PEXELS_API_KEY
    if provider_name == "pixabay":
        return config.PIXABAY_API_KEY
    if provider_name == "coverr":
        return config.COVERR_API_KEY
    return None


def normalize_stock_provider_names(value: Any = None) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        raw_values = [str(item) for item in value]
    else:
        raw = str(value or config.AUTO_BROLL_PROVIDERS or "auto").strip().lower()
        raw_values = re.split(r"[\s,]+", raw)

    names: list[str] = []
    for raw_name in raw_values:
        normalized = raw_name.strip().lower().replace("-", "_")
        if normalized in {"", "auto", "all"}:
            for provider in DEFAULT_STOCK_PROVIDER_ORDER:
                if provider not in names:
                    names.append(provider)
            continue
        aliases = {
            "pexel": "pexels",
            "pixabay_video": "pixabay",
            "cover": "coverr",
        }
        provider = aliases.get(normalized, normalized)
        if provider in DEFAULT_STOCK_PROVIDER_ORDER and provider not in names:
            names.append(provider)
    return names or list(DEFAULT_STOCK_PROVIDER_ORDER)


def configured_stock_provider_names(value: Any = None) -> list[str]:
    return [
        provider
        for provider in normalize_stock_provider_names(value)
        if _provider_api_key(provider)
    ]


def missing_stock_provider_keys(value: Any = None) -> list[str]:
    return [
        STOCK_PROVIDER_KEY_NAMES[provider]
        for provider in normalize_stock_provider_names(value)
        if not _provider_api_key(provider)
    ]


def stock_provider_status(value: Any = None) -> list[dict[str, Any]]:
    return [
        {
            "provider": provider,
            "display_name": STOCK_PROVIDER_DISPLAY_NAMES[provider],
            "configured": bool(_provider_api_key(provider)),
            "env_key": STOCK_PROVIDER_KEY_NAMES[provider],
        }
        for provider in normalize_stock_provider_names(value)
    ]


def extract_json_array(raw_text: str) -> str:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("The model did not return a JSON array.")
    return cleaned[start : end + 1]


def extract_json_object(raw_text: str) -> str:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("The model did not return a JSON object.")
    return cleaned[start : end + 1]


def _status_code_for_reasoning_error(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    if isinstance(response_status, int):
        return response_status
    return None


def _is_retryable_reasoning_error(exc: Exception) -> bool:
    if isinstance(exc, (genai_errors.ServerError, httpx.HTTPError, TimeoutError, OSError)):
        return True
    if isinstance(exc, (genai_errors.ClientError, genai_errors.APIError)):
        status_code = _status_code_for_reasoning_error(exc)
        if status_code in {408, 409, 425, 429}:
            return True
        if status_code is not None and status_code >= 500:
            return True
    message = str(exc).lower()
    retry_hints = (
        "internal error",
        "temporar",
        "timeout",
        "timed out",
        "connection reset",
        "service unavailable",
        "overloaded",
        "rate limit",
        "retry",
    )
    return any(hint in message for hint in retry_hints)


def _call_with_reasoning_retry(operation):
    max_attempts = max(1, int(config.LLM_REQUEST_MAX_RETRIES))
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            retryable = _is_retryable_reasoning_error(exc)
            if retryable and attempt < max_attempts:
                delay = float(config.LLM_RETRY_BASE_DELAY_SEC) * (2 ** (attempt - 1))
                time.sleep(delay)
                continue
            raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Reasoning model call failed without raising an exception.")


def call_reasoning_model(provider_name: str, model_name: str, system_prompt: str, user_prompt: str) -> str:
    config.configure_runtime_logging()
    if provider_name == "claude":
        from anthropic import Anthropic

        client = Anthropic(
            api_key=config.ANTHROPIC_API_KEY,
            timeout=config.ANTHROPIC_TIMEOUT_SEC,
        )
        response = _call_with_reasoning_retry(
            lambda: client.messages.create(
                model=model_name or config.CLAUDE_MODEL,
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
        )
        return "".join(block.text for block in response.content if getattr(block, "type", "") == "text")

    client = genai.Client(
        api_key=config.GEMINI_API_KEY,
        http_options=config.google_genai_http_options(),
    )
    response = _call_with_reasoning_retry(
        lambda: client.models.generate_content(
            model=model_name or config.GEMINI_MODEL,
            contents=user_prompt,
            config=config.build_gemini_generation_config(
                system_prompt,
                model_name=model_name or config.GEMINI_MODEL,
            ),
        )
    )
    return getattr(response, "text", "") or ""


def clip_text(segments: list[dict[str, float | str]]) -> str:
    return " ".join(str(segment["text"]).strip() for segment in segments if str(segment["text"]).strip()).strip()


def overlapping_segments(
    segments: list[dict[str, float | str]],
    start_sec: float,
    end_sec: float,
) -> list[dict[str, float | str]]:
    return [
        segment
        for segment in segments
        if float(segment["end"]) > start_sec and float(segment["start"]) < end_sec
    ]


def window_text(
    segments: list[dict[str, float | str]],
    start_sec: float,
    end_sec: float,
) -> str:
    return clip_text(overlapping_segments(segments, start_sec, end_sec))


def infer_visual_type(text: str) -> str:
    tokens = set(word_tokens(text))
    for visual_type, keywords in VISUAL_KEYWORDS.items():
        if tokens & keywords:
            return visual_type
    if tokens & ABSTRACT_TERMS:
        return "abstract_motion"
    return "cutaway"


def card_priority(card: dict) -> float:
    combined = f"{card['subtitle_text']} {card['context_text']}".lower()
    tokens = word_tokens(combined)
    if not tokens:
        return 0.0
    numbers = len(re.findall(r"\b\d+(?:\.\d+)?%?\b", combined))
    specificity = min(len(set(tokens)) / max(len(tokens), 1), 1.0)
    visual_hits = sum(1 for keyword_set in VISUAL_KEYWORDS.values() if set(tokens) & keyword_set)
    abstract_hits = sum(1 for term in ABSTRACT_TERMS if term in tokens)
    pronouns = sum(1 for token in tokens if token in {"this", "that", "it", "they", "them", "these"})
    return round(30 + numbers * 8 + specificity * 24 + visual_hits * 10 + abstract_hits * 3 - pronouns * 1.5, 2)


def _wrap_caption_words(words: list[str], max_chars_per_line: int, max_lines: int) -> str:
    lines: list[str] = []
    current: list[str] = []
    remaining_words = list(words)
    while remaining_words and len(lines) < max_lines:
        word = remaining_words.pop(0)
        candidate = " ".join(current + [word]).strip()
        if current and len(candidate) > max_chars_per_line:
            lines.append(" ".join(current))
            current = [word]
            continue
        current.append(word)
    if current:
        if len(lines) < max_lines:
            lines.append(" ".join(current))
        elif lines:
            lines[-1] = f"{lines[-1]} {' '.join(current)}".strip()
    if remaining_words and lines:
        lines[-1] = f"{lines[-1]} {' '.join(remaining_words)}".strip()
    return "\n".join(line.strip() for line in lines if line.strip())


def _caption_cards(
    segments: list[dict[str, float | str]],
    max_chars_per_line: int = 22,
    max_lines: int = 2,
    max_words_per_caption: int = 7,
    max_duration_sec: float = 1.8,
) -> list[dict[str, float | str]]:
    optimized: list[dict[str, float | str]] = []
    for segment in segments:
        start_sec = float(segment["start"])
        end_sec = float(segment["end"])
        text = re.sub(r"\s+", " ", str(segment["text"]).strip())
        words = [word for word in text.split(" ") if word]
        if not words or end_sec <= start_sec:
            continue
        duration = end_sec - start_sec
        caption_count = max(
            1,
            math.ceil(len(text) / float(max_chars_per_line * max_lines)),
            math.ceil(len(words) / float(max_words_per_caption)),
            math.ceil(duration / float(max_duration_sec)),
        )
        caption_count = min(caption_count, len(words))
        for index in range(caption_count):
            word_start = int(len(words) * index / caption_count)
            word_end = len(words) if index == caption_count - 1 else int(len(words) * (index + 1) / caption_count)
            caption_words = words[word_start:word_end]
            if not caption_words:
                continue
            piece_start = start_sec + duration * (index / caption_count)
            piece_end = start_sec + duration * ((index + 1) / caption_count)
            optimized.append(
                {
                    "start": round(piece_start, 3),
                    "end": round(piece_end, 3),
                    "text": _wrap_caption_words(caption_words, max_chars_per_line, max_lines),
                }
            )
    return [segment for segment in optimized if str(segment["text"]).strip()]


def build_context_cards(
    transcript_segments: list[dict[str, float | str]],
    clip_duration: float,
) -> list[dict]:
    subtitle_cards = _caption_cards(
        transcript_segments,
        max_chars_per_line=22,
        max_lines=2,
        max_words_per_caption=7,
        max_duration_sec=1.8,
    )
    cards: list[dict] = []
    for index, card in enumerate(subtitle_cards, start=1):
        start_sec = max(0.0, float(card["start"]))
        end_sec = min(clip_duration, float(card["end"]))
        subtitle_text = re.sub(r"\s+", " ", str(card["text"]).replace("\n", " ")).strip()
        context_text = window_text(transcript_segments, max(0.0, start_sec - 2.6), min(clip_duration, end_sec + 2.6))
        keywords = semantic_keywords(f"{subtitle_text} {context_text}", limit=10)
        row = {
            "card_id": f"card_{index:03d}",
            "start": round(start_sec, 2),
            "end": round(end_sec, 2),
            "subtitle_text": subtitle_text,
            "context_text": truncate(context_text, 260),
            "keywords": keywords,
            "visual_type_hint": infer_visual_type(f"{subtitle_text} {context_text}"),
        }
        row["priority"] = card_priority(row)
        cards.append(row)
    return cards


def format_cards_for_llm(cards: list[dict]) -> str:
    lines: list[str] = []
    for card in cards:
        lines.append(
            "\n".join(
                [
                    f"{card['card_id']} | {card['start']:.2f}-{card['end']:.2f} | priority={card['priority']:.2f}",
                    f"Subtitle: {card['subtitle_text']}",
                    f"Context: {card['context_text']}",
                    f"Keywords: {', '.join(card['keywords'])}",
                    f"Hint: {card['visual_type_hint']}",
                ]
            )
        )
    return "\n\n".join(lines)


def normalize_broll_plan(
    raw_plan: list[dict],
    cards: list[dict],
    clip_duration: float,
    max_overlays: int,
    min_overlay_sec: float,
    max_overlay_sec: float,
) -> list[dict]:
    card_map = {card["card_id"]: card for card in cards}
    suggestions: list[dict] = []
    last_end = -999.0
    for item in raw_plan:
        card = card_map.get(str(item.get("card_id") or "").strip())
        if card is None:
            continue
        start_sec = max(0.0, min(float(card["start"]) - 0.08, clip_duration))
        end_sec = min(clip_duration, float(card["end"]) + 0.22)
        if end_sec - start_sec < min_overlay_sec:
            end_sec = min(clip_duration, start_sec + min_overlay_sec)
        if end_sec - start_sec > max_overlay_sec:
            end_sec = start_sec + max_overlay_sec
        if end_sec <= start_sec or start_sec - last_end < 0.7:
            continue
        confidence = max(0.0, min(float(item.get("confidence", 0.55)), 1.0))
        if confidence < 0.38:
            continue
        suggestions.append(
            {
                "card_id": card["card_id"],
                "start": round(start_sec, 2),
                "end": round(end_sec, 2),
                "subtitle_text": card["subtitle_text"],
                "context_text": card["context_text"],
                "keywords": card["keywords"][:8],
                "visual_type": truncate(str(item.get("visual_type") or card["visual_type_hint"]), 32),
                "primary_query": truncate(str(item.get("primary_query") or keyword_phrase(card["subtitle_text"], 5)), 80),
                "backup_queries": [truncate(str(value), 80) for value in item.get("backup_queries", []) if str(value).strip()][:3],
                "must_include": [truncate(str(value), 24) for value in item.get("must_include", []) if str(value).strip()][:5],
                "avoid": [truncate(str(value), 24) for value in item.get("avoid", []) if str(value).strip()][:5],
                "direction": truncate(str(item.get("direction") or "Add a literal supporting cutaway tied to the active subtitle line."), 130),
                "rationale": truncate(str(item.get("rationale") or "Aligned to the subtitle beat and nearby transcript context."), 150),
                "confidence": round(confidence, 2),
            }
        )
        last_end = end_sec
        if len(suggestions) >= max_overlays:
            break
    return suggestions


def fallback_broll_plan(
    cards: list[dict],
    max_overlays: int,
    min_overlay_sec: float,
    max_overlay_sec: float,
    clip_duration: float,
) -> list[dict]:
    ranked = sorted(cards, key=lambda item: (item["priority"], item["start"]), reverse=True)
    chosen: list[dict] = []
    for card in ranked:
        if any(abs(card["start"] - float(existing.get("_anchor_start", 0.0))) < 1.1 for existing in chosen):
            continue
        chosen.append(
            {
                "card_id": card["card_id"],
                "_anchor_start": card["start"],
                "visual_type": card["visual_type_hint"],
                "primary_query": truncate(keyword_phrase(card["subtitle_text"], 5), 80),
                "backup_queries": [
                    truncate(" ".join(card["keywords"][:4]), 80),
                    truncate(keyword_phrase(card["context_text"], 5), 80),
                ],
                "must_include": card["keywords"][:4],
                "avoid": ["generic", "random"] if card["visual_type_hint"] != "abstract_motion" else [],
                "direction": "Anchor the cutaway to the active subtitle beat instead of a nearby generic moment.",
                "rationale": "Fallback selection anchored to the subtitle card so the visual change follows the spoken line.",
                "confidence": round(min(max(card["priority"] / 85.0, 0.42), 0.92), 2),
            }
        )
        if len(chosen) >= max_overlays:
            break
    return normalize_broll_plan(chosen, cards, clip_duration, max_overlays, min_overlay_sec, max_overlay_sec)


def analyze_broll_plan_with_llm(
    provider_name: str,
    model_name: str,
    cards: list[dict],
    clip_duration: float,
    max_overlays: int,
    min_overlay_sec: float,
    max_overlay_sec: float,
    orientation: str,
) -> list[dict]:
    fallback = fallback_broll_plan(cards, max_overlays, min_overlay_sec, max_overlay_sec, clip_duration)
    if not cards:
        return fallback
    system_prompt = (
        "You are a senior documentary editor designing stock B-roll insertions. "
        "Pick subtitle-anchored moments where the inserted footage should precisely reinforce the spoken line. "
        "Avoid generic nature or random office footage unless the line is truly abstract. "
        "Return ONLY a JSON array of up to {count} objects with keys: card_id, visual_type, primary_query, backup_queries, must_include, avoid, direction, rationale, confidence."
    ).format(count=max_overlays)
    user_prompt = (
        f"Video duration: {clip_duration:.2f}s\n"
        f"Orientation target: {orientation}\n"
        f"Need at most {max_overlays} B-roll inserts.\n"
        f"Each insert should stay between {min_overlay_sec:.1f}s and {max_overlay_sec:.1f}s.\n\n"
        f"Subtitle-aligned cards:\n{truncate(format_cards_for_llm(cards), 7200)}\n\n"
        "Choose only cards where a literal or semantically faithful stock visual would make the subtitle easier to feel and understand. "
        "Queries should be concrete and searchable on stock sites. Return JSON array only."
    )
    try:
        raw_text = call_reasoning_model(provider_name, model_name, system_prompt, user_prompt)
        parsed = json.loads(extract_json_array(raw_text))
    except Exception:
        return fallback
    normalized = normalize_broll_plan(parsed, cards, clip_duration, max_overlays, min_overlay_sec, max_overlay_sec)
    return normalized or fallback


def video_orientation(width: int, height: int) -> str:
    if height > width:
        return "portrait"
    if width > height:
        return "landscape"
    return "square"


def _json_rate_limit_headers(response, provider: str) -> dict[str, str]:  # noqa: ANN001
    if provider in {"pexels", "pixabay"}:
        return {
            "limit": response.headers.get("X-Ratelimit-Limit", ""),
            "remaining": response.headers.get("X-Ratelimit-Remaining", ""),
            "reset": response.headers.get("X-Ratelimit-Reset", ""),
        }
    if provider == "coverr":
        return {
            "limit": response.headers.get("X-RateLimit-Limit", ""),
            "remaining": response.headers.get("X-RateLimit-Remaining", ""),
            "reset": response.headers.get("X-RateLimit-Reset", ""),
        }
    return {}


def _read_stock_json_response(response, provider: str) -> tuple[dict, dict[str, str]]:  # noqa: ANN001
    raw_payload = response.read(MAX_STOCK_JSON_BYTES + 1)
    if len(raw_payload) > MAX_STOCK_JSON_BYTES:
        raise RuntimeError(f"{STOCK_PROVIDER_DISPLAY_NAMES.get(provider, provider)} API response was larger than the configured safety limit.")
    payload = json.loads(raw_payload.decode("utf-8"))
    return payload, _json_rate_limit_headers(response, provider)


def _provider_request_error(provider: str, exc: urllib.error.HTTPError) -> RuntimeError:
    label = STOCK_PROVIDER_DISPLAY_NAMES.get(provider, provider)
    details = exc.read().decode("utf-8", errors="ignore")
    if exc.code in {401, 403}:
        return RuntimeError(f"{label} API rejected the key. Check {STOCK_PROVIDER_KEY_NAMES.get(provider, 'the provider API key')}.")
    if exc.code == 429:
        return RuntimeError(f"{label} API rate limit exceeded. Wait for reset before retrying.")
    return RuntimeError(f"{label} API request failed with HTTP {exc.code}: {details or exc.reason}")


def pexels_get_json(url: str) -> tuple[dict, dict[str, str]]:
    if not config.PEXELS_API_KEY:
        raise RuntimeError("PEXELS_API_KEY is required for auto B-roll.")
    safe_url = _validated_public_https_url(url, allowed_hosts={PEXELS_API_HOST})
    request = urllib.request.Request(
        safe_url,
        headers={
            "Authorization": config.PEXELS_API_KEY,
            "Accept": "application/json",
            "User-Agent": "Vex/1.0 (+https://github.com/AKMessi/vex)",
        },
    )
    try:
        with _open_validated_https_request(request, timeout=30, allowed_hosts={PEXELS_API_HOST}) as response:
            return _read_stock_json_response(response, "pexels")
    except urllib.error.HTTPError as exc:
        raise _provider_request_error("pexels", exc) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach Pexels API: {exc.reason}") from exc


def pixabay_get_json(url: str) -> tuple[dict, dict[str, str]]:
    if not config.PIXABAY_API_KEY:
        raise RuntimeError("PIXABAY_API_KEY is required for Pixabay B-roll search.")
    safe_url = _validated_public_https_url(url, allowed_hosts={PIXABAY_API_HOST})
    request = urllib.request.Request(
        safe_url,
        headers={
            "Accept": "application/json",
            "User-Agent": "Vex/1.0 (+https://github.com/AKMessi/vex)",
        },
    )
    try:
        with _open_validated_https_request(request, timeout=30, allowed_hosts={PIXABAY_API_HOST}) as response:
            return _read_stock_json_response(response, "pixabay")
    except urllib.error.HTTPError as exc:
        raise _provider_request_error("pixabay", exc) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach Pixabay API: {exc.reason}") from exc


def coverr_get_json(url: str) -> tuple[dict, dict[str, str]]:
    if not config.COVERR_API_KEY:
        raise RuntimeError("COVERR_API_KEY is required for Coverr B-roll search.")
    safe_url = _validated_public_https_url(url, allowed_hosts={COVERR_API_HOST})
    request = urllib.request.Request(
        safe_url,
        headers={
            "Authorization": f"Bearer {config.COVERR_API_KEY}",
            "Accept": "application/json",
            "User-Agent": "Vex/1.0 (+https://github.com/AKMessi/vex)",
        },
    )
    try:
        with _open_validated_https_request(request, timeout=30, allowed_hosts={COVERR_API_HOST}) as response:
            return _read_stock_json_response(response, "coverr")
    except urllib.error.HTTPError as exc:
        raise _provider_request_error("coverr", exc) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach Coverr API: {exc.reason}") from exc


def search_pexels_videos(query: str, orientation: str, per_page: int = 8) -> tuple[list[dict], dict[str, str]]:
    params = urllib.parse.urlencode(
        {
            "query": query,
            "orientation": orientation,
            "size": "medium",
            "locale": "en-US",
            "per_page": min(max(per_page, 1), 80),
            "page": 1,
        }
    )
    payload, headers = pexels_get_json(f"https://api.pexels.com/v1/videos/search?{params}")
    return list(payload.get("videos") or []), headers


def search_pixabay_videos(query: str, orientation: str, per_page: int = 8) -> tuple[list[dict], dict[str, str]]:
    min_width = 720 if orientation == "portrait" else 960
    min_height = 960 if orientation == "portrait" else 540
    params = urllib.parse.urlencode(
        {
            "key": config.PIXABAY_API_KEY or "",
            "q": truncate(query, 100),
            "lang": "en",
            "video_type": "all",
            "min_width": min_width,
            "min_height": min_height,
            "safesearch": "true",
            "order": "popular",
            "per_page": min(max(per_page, 3), 80),
            "page": 1,
        }
    )
    payload, headers = pixabay_get_json(f"https://pixabay.com/api/videos/?{params}")
    return list(payload.get("hits") or []), headers


def search_coverr_videos(query: str, orientation: str, per_page: int = 8) -> tuple[list[dict], dict[str, str]]:
    params = urllib.parse.urlencode(
        {
            "query": truncate(query, 100),
            "sort": "popular",
            "urls": "true",
            "page_size": min(max(per_page, 1), 50),
            "page": 0,
        }
    )
    payload, headers = coverr_get_json(f"https://api.coverr.co/videos?{params}")
    hits = list(payload.get("hits") or [])
    if orientation == "portrait":
        vertical = [item for item in hits if bool(item.get("is_vertical"))]
        return (vertical or hits), headers
    if orientation == "landscape":
        landscape = [item for item in hits if not bool(item.get("is_vertical"))]
        return (landscape or hits), headers
    return hits, headers


def pick_video_file(video: dict, target_orientation: str, target_width: int, target_height: int) -> dict | None:
    best_file = None
    best_score = None
    for item in video.get("video_files") or []:
        if str(item.get("file_type") or "").lower() != "video/mp4":
            continue
        width = int(item.get("width") or 0)
        height = int(item.get("height") or 0)
        if width <= 0 or height <= 0:
            continue
        orientation_bonus = 18 if video_orientation(width, height) == target_orientation else 0
        quality = str(item.get("quality") or "").lower()
        quality_bonus = 20 if quality == "hd" else 8
        resolution_bonus = min((width * height) / max(target_width * target_height, 1), 3.0) * 12
        fps_bonus = min(float(item.get("fps") or 0.0), 60.0) / 10.0
        score = orientation_bonus + quality_bonus + resolution_bonus + fps_bonus
        if best_score is None or score > best_score:
            best_score = score
            best_file = item
    return best_file


def pick_pixabay_video_file(video: dict, target_orientation: str, target_width: int, target_height: int) -> dict | None:
    best_file = None
    best_score = None
    for quality, item in (video.get("videos") or {}).items():
        url = str(item.get("url") or "").strip()
        width = _as_int(item.get("width"))
        height = _as_int(item.get("height"))
        if not url or width <= 0 or height <= 0:
            continue
        orientation_bonus = 18 if video_orientation(width, height) == target_orientation else 0
        quality_rank = {"large": 24, "medium": 20, "small": 12, "tiny": 6}.get(str(quality), 4)
        resolution_bonus = min((width * height) / max(target_width * target_height, 1), 3.0) * 12
        size_penalty = min(_as_int(item.get("size")) / MAX_STOCK_DOWNLOAD_BYTES, 1.0) * 4
        score = orientation_bonus + quality_rank + resolution_bonus - size_penalty
        if best_score is None or score > best_score:
            best_score = score
            best_file = {
                "id": quality,
                "link": url,
                "width": width,
                "height": height,
                "fps": _as_float(item.get("fps")),
                "quality": quality,
                "size": _as_int(item.get("size")),
                "thumbnail": str(item.get("thumbnail") or ""),
                "file_type": "video/mp4",
            }
    return best_file


def pick_coverr_video_file(video: dict, target_orientation: str, target_width: int, target_height: int) -> dict | None:
    urls = video.get("urls") or {}
    download_url = str(urls.get("mp4_download") or urls.get("mp4") or "").strip()
    if not download_url:
        return None
    width = _as_int(video.get("max_width"))
    height = _as_int(video.get("max_height"))
    if width <= 0 or height <= 0:
        aspect_ratio = str(video.get("aspect_ratio") or "").strip()
        if ":" in aspect_ratio:
            left, right = aspect_ratio.split(":", 1)
            left_value = _as_float(left)
            right_value = _as_float(right)
            if left_value > 0 and right_value > 0:
                if bool(video.get("is_vertical")):
                    width, height = 1080, 1920
                else:
                    width = target_width or 1920
                    height = max(1, int(round(width * right_value / left_value)))
        if width <= 0 or height <= 0:
            width, height = target_width or 1920, target_height or 1080
    return {
        "id": str(video.get("id") or "coverr"),
        "link": download_url,
        "width": width,
        "height": height,
        "fps": 0.0,
        "quality": "curated",
        "size": 0,
        "thumbnail": str(video.get("thumbnail") or video.get("poster") or ""),
        "file_type": "video/mp4",
    }


def pexels_candidate_from_video(video: dict, file_info: dict) -> StockBrollCandidate:
    user = video.get("user") or {}
    return StockBrollCandidate(
        provider="pexels",
        provider_display_name="Pexels",
        provider_id=str(video.get("id") or ""),
        title=truncate(str(video.get("url") or "").rstrip("/").rsplit("/", 1)[-1].replace("-", " "), 120),
        description="",
        tags=list(slug_tokens(video)),
        duration=_as_float(video.get("duration")),
        source_url=str(video.get("url") or ""),
        download_url=str(file_info.get("link") or ""),
        preview_url=str(video.get("image") or ""),
        creator_name=str(user.get("name") or ""),
        creator_url=str(user.get("url") or ""),
        license_name="Pexels License",
        license_url="https://www.pexels.com/license/",
        attribution_required=True,
        width=_as_int(file_info.get("width")),
        height=_as_int(file_info.get("height")),
        fps=_as_float(file_info.get("fps")),
        quality=str(file_info.get("quality") or ""),
        size_bytes=_as_int(file_info.get("size")),
        raw={"video": video, "file_info": file_info},
    )


def pixabay_candidate_from_video(video: dict, file_info: dict) -> StockBrollCandidate:
    user = str(video.get("user") or "")
    user_id = str(video.get("user_id") or "").strip()
    creator_url = f"https://pixabay.com/users/{urllib.parse.quote(user)}-{user_id}/" if user and user_id else ""
    return StockBrollCandidate(
        provider="pixabay",
        provider_display_name="Pixabay",
        provider_id=str(video.get("id") or ""),
        title=truncate(str(video.get("tags") or "").split(",")[0].strip() or f"Pixabay video {video.get('id')}", 120),
        description=str(video.get("tags") or ""),
        tags=_split_tags(video.get("tags")),
        duration=_as_float(video.get("duration")),
        source_url=str(video.get("pageURL") or ""),
        download_url=str(file_info.get("link") or ""),
        preview_url=str(file_info.get("thumbnail") or video.get("picture_id") or ""),
        creator_name=user,
        creator_url=creator_url,
        license_name="Pixabay Content License",
        license_url="https://pixabay.com/service/license-summary/",
        attribution_required=True,
        width=_as_int(file_info.get("width")),
        height=_as_int(file_info.get("height")),
        fps=_as_float(file_info.get("fps")),
        quality=str(file_info.get("quality") or ""),
        size_bytes=_as_int(file_info.get("size")),
        raw={"video": video, "file_info": file_info},
    )


def coverr_candidate_from_video(video: dict, file_info: dict) -> StockBrollCandidate:
    tags = _split_tags(video.get("tags") or [])
    title = str(video.get("title") or "").strip()
    return StockBrollCandidate(
        provider="coverr",
        provider_display_name="Coverr",
        provider_id=str(video.get("id") or ""),
        title=truncate(title or f"Coverr video {video.get('id')}", 120),
        description=truncate(str(video.get("description") or ""), 220),
        tags=tags,
        duration=_as_float(video.get("duration")),
        source_url=f"https://coverr.co/videos/{video.get('id')}" if video.get("id") else "https://coverr.co",
        download_url=str(file_info.get("link") or ""),
        preview_url=str(file_info.get("thumbnail") or video.get("thumbnail") or video.get("poster") or ""),
        creator_name="Coverr",
        creator_url="https://coverr.co",
        license_name="Coverr License",
        license_url="https://coverr.co/license",
        attribution_required=True,
        width=_as_int(file_info.get("width")),
        height=_as_int(file_info.get("height")),
        fps=_as_float(file_info.get("fps")),
        quality=str(file_info.get("quality") or ""),
        size_bytes=_as_int(file_info.get("size")),
        raw={"video": video, "file_info": file_info},
    )


def _hostname_is_public(hostname: str) -> bool:
    normalized = hostname.strip().strip("[]").lower()
    if normalized in {"localhost", "localhost.localdomain"} or normalized.endswith(".localhost"):
        return False
    try:
        addresses = [ipaddress.ip_address(normalized)]
    except ValueError:
        try:
            infos = socket.getaddrinfo(normalized, None, type=socket.SOCK_STREAM)
        except socket.gaierror as exc:
            raise RuntimeError(f"Could not resolve stock media host: {hostname}") from exc
        addresses = [ipaddress.ip_address(info[4][0]) for info in infos]
    return all(
        not (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_multicast
            or address.is_reserved
            or address.is_unspecified
        )
        for address in addresses
    )


def _validated_public_https_url(url: str, *, allowed_hosts: set[str] | None = None) -> str:
    parsed = urllib.parse.urlparse(str(url or "").strip())
    if parsed.scheme.lower() != "https":
        raise RuntimeError("Stock media downloads must use HTTPS URLs.")
    if parsed.username or parsed.password:
        raise RuntimeError("Stock media URLs must not include credentials.")
    hostname = (parsed.hostname or "").lower()
    if not hostname:
        raise RuntimeError("Stock media URL is missing a host.")
    if allowed_hosts is not None and hostname not in allowed_hosts:
        allowed = ", ".join(sorted(allowed_hosts))
        raise RuntimeError(f"Unexpected stock media API host {hostname!r}; expected {allowed}.")
    if not _hostname_is_public(hostname):
        raise RuntimeError(f"Refusing to download stock media from non-public host: {hostname}")
    return urllib.parse.urlunparse(parsed)


class _ValidatedHttpsRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self, allowed_hosts: set[str] | None = None) -> None:
        self.allowed_hosts = allowed_hosts
        super().__init__()

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        safe_url = _validated_public_https_url(newurl, allowed_hosts=self.allowed_hosts)
        return super().redirect_request(req, fp, code, msg, headers, safe_url)


def _open_validated_https_request(
    request: urllib.request.Request,
    *,
    timeout: float,
    allowed_hosts: set[str] | None = None,
):
    opener = urllib.request.build_opener(_ValidatedHttpsRedirectHandler(allowed_hosts))
    response = opener.open(request, timeout=timeout)
    try:
        final_url = response.geturl()
        _validated_public_https_url(final_url, allowed_hosts=allowed_hosts)
    except Exception:
        response.close()
        raise
    return response


def download_file(url: str, destination: Path, *, max_bytes: int = MAX_STOCK_DOWNLOAD_BYTES) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    safe_url = _validated_public_https_url(url)
    request = urllib.request.Request(safe_url, headers={"User-Agent": "Vex/1.0"})
    temp_path = destination.with_name(f".{destination.name}.part")
    try:
        with _open_validated_https_request(request, timeout=60) as response:
            content_length = response.headers.get("Content-Length")
            if content_length:
                try:
                    declared_bytes = int(content_length)
                except ValueError as exc:
                    raise RuntimeError("Stock clip download reported an invalid Content-Length.") from exc
                if declared_bytes > max_bytes:
                    raise RuntimeError("Stock clip download is larger than the configured safety limit.")
            total = 0
            with temp_path.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise RuntimeError("Stock clip download exceeded the configured safety limit.")
                    output.write(chunk)
            if total <= 0:
                raise RuntimeError("Stock clip download was empty.")
            temp_path.replace(destination)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to download stock clip: {exc.reason}") from exc
    finally:
        temp_path.unlink(missing_ok=True)
    return destination


PROVIDER_VISUAL_TYPE_HINTS = {
    "pexels": VISUAL_TYPE_HINTS,
    "pixabay": {
        "data_graphic": ["data visualization", "technology dashboard", "analytics chart"],
        "product_ui": ["computer software", "app dashboard", "developer workspace"],
        "cutaway": ["person laptop", "office work", "team collaboration"],
        "process": ["hands typing", "creative process", "workflow"],
        "location": ["city office", "workplace interior", "studio"],
        "abstract_motion": ["technology background", "digital network", "abstract data"],
    },
    "coverr": {
        "data_graphic": ["data screen", "startup analytics", "technology"],
        "product_ui": ["software", "developer", "computer work"],
        "cutaway": ["business work", "creator workspace", "meeting"],
        "process": ["working", "building", "creative"],
        "location": ["office", "city", "studio"],
        "abstract_motion": ["technology", "network", "digital"],
    },
}


def query_variants(plan_item: dict, provider_name: str | None = None) -> list[str]:
    variants: list[str] = []
    for value in [str(plan_item.get("primary_query") or "").strip(), *[str(v).strip() for v in plan_item.get("backup_queries", []) if str(v).strip()]]:
        normalized = re.sub(r"\s+", " ", value).strip()
        if normalized and normalized not in variants:
            variants.append(normalized)
    if plan_item.get("must_include"):
        combined = " ".join([str(plan_item.get("primary_query") or "").strip(), *plan_item.get("must_include", [])]).strip()
        if combined and combined not in variants:
            variants.insert(1, truncate(combined, 80))
    provider_hints = PROVIDER_VISUAL_TYPE_HINTS.get(provider_name or "pexels", VISUAL_TYPE_HINTS)
    for hint in provider_hints.get(str(plan_item.get("visual_type") or "").lower(), []):
        if hint not in variants:
            variants.append(hint)
    return variants[:5]


def slug_tokens(video: dict) -> set[str]:
    url = str(video.get("url") or "")
    parts = [part for part in urllib.parse.urlparse(url).path.split("/") if part]
    return set(semantic_keywords(" ".join(parts), limit=12))


def candidate_semantic_tokens(candidate: dict) -> set[str]:
    source_path = urllib.parse.urlparse(str(candidate.get("source_url") or "")).path
    text = " ".join(
        [
            str(candidate.get("title") or ""),
            str(candidate.get("description") or ""),
            " ".join(str(tag) for tag in candidate.get("tags", [])),
            source_path.replace("/", " ").replace("-", " "),
        ]
    )
    return set(semantic_keywords(text, limit=24))


def _provider_intent_bonus(provider: str, visual_type: str) -> float:
    if provider == "coverr" and visual_type in {"cutaway", "process", "location"}:
        return 6.0
    if provider == "pixabay" and visual_type in {"abstract_motion", "data_graphic", "location"}:
        return 5.0
    if provider == "pexels" and visual_type in {"cutaway", "product_ui", "process"}:
        return 3.0
    return 0.0


def heuristic_candidate_score_for_candidate(
    plan_item: dict,
    candidate: dict,
    matched_query: str,
    query_rank: int,
    target_orientation: str,
    target_width: int,
    target_height: int,
) -> float:
    file_info = candidate.get("file_info") or {}
    width = int(candidate.get("width") or file_info.get("width") or 0)
    height = int(candidate.get("height") or file_info.get("height") or 0)
    quality = str(candidate.get("quality") or file_info.get("quality") or "").lower()
    duration = _as_float(candidate.get("duration"))
    overlay_duration = float(plan_item["end"]) - float(plan_item["start"])
    tokens = candidate_semantic_tokens(candidate)
    expected = set(
        semantic_keywords(
            " ".join(
                [
                    str(plan_item.get("subtitle_text") or ""),
                    str(plan_item.get("context_text") or ""),
                    str(plan_item.get("primary_query") or ""),
                    " ".join(plan_item.get("backup_queries", [])),
                ]
            ),
            limit=14,
        )
    )
    must_include = set(semantic_keywords(" ".join(str(token) for token in plan_item.get("must_include", []) if str(token).strip()), limit=12))
    avoid = set(semantic_keywords(" ".join(str(token) for token in plan_item.get("avoid", []) if str(token).strip()), limit=12))
    query_tokens = set(semantic_keywords(matched_query, limit=10))
    orientation_bonus = 18 if video_orientation(width, height) == target_orientation else 0
    quality_bonus = 18 if quality in {"hd", "large", "medium", "curated"} else 8
    resolution_bonus = min((width * height) / max(target_width * target_height, 1), 3.0) * 10
    duration_bonus = 10 if duration >= overlay_duration else 4
    overlap_bonus = len(tokens & expected) * 8
    must_bonus = len(tokens & must_include) * 12
    query_bonus = len(tokens & query_tokens) * 6
    avoid_penalty = len(tokens & avoid) * 12
    rank_bonus = max(0, 12 - query_rank * 3)
    provider_bonus = _provider_intent_bonus(str(candidate.get("provider") or ""), str(plan_item.get("visual_type") or ""))
    return round(
        orientation_bonus
        + quality_bonus
        + resolution_bonus
        + duration_bonus
        + overlap_bonus
        + must_bonus
        + query_bonus
        + rank_bonus
        + provider_bonus
        - avoid_penalty,
        2,
    )


def heuristic_candidate_score(
    plan_item: dict,
    video: dict,
    file_info: dict,
    matched_query: str,
    query_rank: int,
    target_orientation: str,
    target_width: int,
    target_height: int,
) -> float:
    candidate = pexels_candidate_from_video(video, file_info).to_dict()
    return heuristic_candidate_score_for_candidate(
        plan_item,
        candidate,
        matched_query,
        query_rank,
        target_orientation,
        target_width,
        target_height,
    )


def _search_pexels_candidates(
    query: str,
    target_orientation: str,
    target_width: int,
    target_height: int,
    per_page: int,
) -> tuple[list[dict], dict[str, str]]:
    videos, headers = search_pexels_videos(query, orientation=target_orientation, per_page=per_page)
    candidates: list[dict] = []
    for video in videos:
        file_info = pick_video_file(video, target_orientation, target_width, target_height)
        if file_info is not None:
            candidates.append(pexels_candidate_from_video(video, file_info).to_dict())
    return candidates, headers


def _search_pixabay_candidates(
    query: str,
    target_orientation: str,
    target_width: int,
    target_height: int,
    per_page: int,
) -> tuple[list[dict], dict[str, str]]:
    videos, headers = search_pixabay_videos(query, orientation=target_orientation, per_page=per_page)
    candidates: list[dict] = []
    for video in videos:
        file_info = pick_pixabay_video_file(video, target_orientation, target_width, target_height)
        if file_info is not None:
            candidates.append(pixabay_candidate_from_video(video, file_info).to_dict())
    return candidates, headers


def _search_coverr_candidates(
    query: str,
    target_orientation: str,
    target_width: int,
    target_height: int,
    per_page: int,
) -> tuple[list[dict], dict[str, str]]:
    videos, headers = search_coverr_videos(query, orientation=target_orientation, per_page=per_page)
    candidates: list[dict] = []
    for video in videos:
        file_info = pick_coverr_video_file(video, target_orientation, target_width, target_height)
        if file_info is not None:
            candidates.append(coverr_candidate_from_video(video, file_info).to_dict())
    return candidates, headers


STOCK_PROVIDER_SEARCHERS: dict[str, Callable[[str, str, int, int, int], tuple[list[dict], dict[str, str]]]] = {
    "pexels": _search_pexels_candidates,
    "pixabay": _search_pixabay_candidates,
    "coverr": _search_coverr_candidates,
}


def search_stock_provider(
    provider_name: str,
    query: str,
    target_orientation: str,
    target_width: int,
    target_height: int,
    per_page: int = 6,
) -> tuple[list[dict], dict[str, str]]:
    searcher = STOCK_PROVIDER_SEARCHERS.get(provider_name)
    if searcher is None:
        raise RuntimeError(f"Unsupported stock B-roll provider: {provider_name}")
    return searcher(query, target_orientation, target_width, target_height, per_page)


def collect_search_candidates(
    plan_item: dict,
    target_orientation: str,
    target_width: int,
    target_height: int,
    search_fn: Callable[..., tuple[list[dict], dict[str, str]]] | None = None,
    provider_names: Any = None,
) -> tuple[list[dict], dict[str, Any]]:
    candidates: list[dict] = []
    seen_keys: set[tuple[str, str]] = set()
    provider_headers: dict[str, dict[str, str]] = {}

    if search_fn is not None:
        latest_headers: dict[str, str] = {}
        for query_rank, query in enumerate(query_variants(plan_item, "pexels")):
            videos, latest_headers = search_fn(query, orientation=target_orientation, per_page=6)
            provider_headers["pexels"] = latest_headers
            for video in videos:
                video_id = str(video.get("id") or "")
                if video_id and ("pexels", video_id) in seen_keys:
                    continue
                file_info = pick_video_file(video, target_orientation, target_width, target_height)
                if file_info is None:
                    continue
                if video_id:
                    seen_keys.add(("pexels", video_id))
                candidate = pexels_candidate_from_video(video, file_info).to_dict()
                candidate.update({"matched_query": query, "query_rank": query_rank})
                candidate["score"] = heuristic_candidate_score_for_candidate(
                    plan_item,
                    candidate,
                    query,
                    query_rank,
                    target_orientation,
                    target_width,
                    target_height,
                )
                candidate["slug_tokens"] = sorted(candidate_semantic_tokens(candidate))
                candidates.append(candidate)
                if len(candidates) >= 10:
                    break
            if len(candidates) >= 10:
                break
        candidates.sort(key=lambda item: item["score"], reverse=True)
        for index, candidate in enumerate(candidates, start=1):
            candidate["result_id"] = f"cand_{index:02d}"
        return candidates, provider_headers

    active_providers = configured_stock_provider_names(provider_names)
    provider_order = {provider: index for index, provider in enumerate(active_providers)}
    for provider in active_providers:
        for query_rank, query in enumerate(query_variants(plan_item, provider)):
            try:
                provider_candidates, headers = search_stock_provider(
                    provider,
                    query,
                    target_orientation,
                    target_width,
                    target_height,
                    per_page=6,
                )
            except RuntimeError as exc:
                provider_headers.setdefault("_errors", {})[provider] = truncate(str(exc), 220)
                continue
            provider_headers[provider] = headers
            for candidate in provider_candidates:
                candidate_provider = str(candidate.get("provider") or provider)
                provider_id = str(candidate.get("provider_id") or "")
                key = (candidate_provider, provider_id or str(candidate.get("download_url") or ""))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                candidate.update({"matched_query": query, "query_rank": query_rank})
                candidate["score"] = heuristic_candidate_score_for_candidate(
                    plan_item,
                    candidate,
                    query,
                    query_rank,
                    target_orientation,
                    target_width,
                    target_height,
                )
                candidate["score"] += max(0, 4 - provider_order.get(candidate_provider, 4))
                candidate["slug_tokens"] = sorted(candidate_semantic_tokens(candidate))
                candidates.append(candidate)
                if len(candidates) >= 18:
                    break
            if len(candidates) >= 18:
                break
    candidates.sort(key=lambda item: item["score"], reverse=True)
    for index, candidate in enumerate(candidates, start=1):
        candidate["result_id"] = f"cand_{index:02d}"
    return candidates[:12], provider_headers


def format_candidate_summaries(candidates: list[dict]) -> str:
    lines: list[str] = []
    for candidate in candidates[:8]:
        file_info = candidate["file_info"]
        slug = ", ".join(candidate.get("slug_tokens") or []) or "none"
        lines.append(
            "\n".join(
                [
                    f"{candidate['result_id']} | {candidate.get('provider_display_name') or candidate.get('provider')} | score={candidate['score']:.2f} | query={candidate['matched_query']}",
                    f"Title: {candidate.get('title') or 'unknown'}",
                    f"Source: {candidate.get('source_url') or 'unknown'}",
                    f"Tags/tokens: {slug}",
                    f"Duration: {candidate.get('duration')}s | File: {file_info.get('width')}x{file_info.get('height')} {file_info.get('quality')} {file_info.get('fps')}fps",
                ]
            )
        )
    return "\n\n".join(lines)


def choose_candidate_with_llm(
    provider_name: str,
    model_name: str,
    plan_item: dict,
    candidates: list[dict],
) -> tuple[dict | None, str | None]:
    if not candidates:
        return None, None
    if len(candidates) == 1:
        return candidates[0], "Only viable candidate returned from configured stock search."
    system_prompt = (
        "You are selecting the best stock clip candidate for a precise subtitle-aligned B-roll insert. "
        "Choose the result whose semantics best match the subtitle and context. Prefer literal matches over generic mood footage, and avoid stock clips that only match a single vague word. "
        "Return ONLY a JSON object with keys result_id and reason."
    )
    user_prompt = (
        f"Subtitle: {plan_item['subtitle_text']}\n"
        f"Context: {plan_item['context_text']}\n"
        f"Primary query: {plan_item['primary_query']}\n"
        f"Backup queries: {', '.join(plan_item.get('backup_queries', []))}\n"
        f"Must include: {', '.join(plan_item.get('must_include', []))}\n"
        f"Avoid: {', '.join(plan_item.get('avoid', []))}\n\n"
        f"Candidates:\n{truncate(format_candidate_summaries(candidates), 5000)}\n\n"
        "Return JSON only."
    )
    try:
        raw_text = call_reasoning_model(provider_name, model_name, system_prompt, user_prompt)
        parsed = json.loads(extract_json_object(raw_text))
        chosen_id = str(parsed.get("result_id") or "").strip()
        reason = truncate(str(parsed.get("reason") or "Chosen by semantic reranking."), 160)
        chosen = next((candidate for candidate in candidates if candidate["result_id"] == chosen_id), None)
        if chosen is not None:
            return chosen, reason
    except Exception:
        pass
    return candidates[0], "Chosen by heuristic semantic score."


def ensure_writable_dir(candidates: list[Path]) -> Path:
    last_error: Exception | None = None
    for directory in candidates:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            probe = directory / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return directory
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError(f"No writable directory available for auto B-roll artifacts: {last_error}")


def writable_dir_candidates(base_working_dir: str, base_output_dir: str, project_id: str, label: str) -> list[Path]:
    safe_label = safe_stem(label)
    return [
        Path(base_working_dir) / safe_label,
        Path(base_output_dir) / safe_label,
        Path.cwd() / safe_label,
        Path(tempfile.gettempdir()) / "vex" / project_id / safe_label,
    ]
