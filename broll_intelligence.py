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
from pathlib import Path

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
MAX_PEXELS_JSON_BYTES = 5 * 1024 * 1024
PEXELS_API_HOST = "api.pexels.com"
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
            raw_payload = response.read(MAX_PEXELS_JSON_BYTES + 1)
            if len(raw_payload) > MAX_PEXELS_JSON_BYTES:
                raise RuntimeError("Pexels API response was larger than the configured safety limit.")
            payload = json.loads(raw_payload.decode("utf-8"))
            headers = {
                "limit": response.headers.get("X-Ratelimit-Limit", ""),
                "remaining": response.headers.get("X-Ratelimit-Remaining", ""),
                "reset": response.headers.get("X-Ratelimit-Reset", ""),
            }
            return payload, headers
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="ignore")
        if exc.code == 401:
            raise RuntimeError("Pexels API rejected the key. Check PEXELS_API_KEY.") from exc
        if exc.code == 429:
            raise RuntimeError("Pexels API rate limit exceeded. Wait for reset before retrying.") from exc
        raise RuntimeError(f"Pexels API request failed with HTTP {exc.code}: {details or exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach Pexels API: {exc.reason}") from exc


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


def query_variants(plan_item: dict) -> list[str]:
    variants: list[str] = []
    for value in [str(plan_item.get("primary_query") or "").strip(), *[str(v).strip() for v in plan_item.get("backup_queries", []) if str(v).strip()]]:
        normalized = re.sub(r"\s+", " ", value).strip()
        if normalized and normalized not in variants:
            variants.append(normalized)
    if plan_item.get("must_include"):
        combined = " ".join([str(plan_item.get("primary_query") or "").strip(), *plan_item.get("must_include", [])]).strip()
        if combined and combined not in variants:
            variants.insert(1, truncate(combined, 80))
    for hint in VISUAL_TYPE_HINTS.get(str(plan_item.get("visual_type") or "").lower(), []):
        if hint not in variants:
            variants.append(hint)
    return variants[:5]


def slug_tokens(video: dict) -> set[str]:
    url = str(video.get("url") or "")
    parts = [part for part in urllib.parse.urlparse(url).path.split("/") if part]
    return set(semantic_keywords(" ".join(parts), limit=12))


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
    width = int(file_info.get("width") or 0)
    height = int(file_info.get("height") or 0)
    quality = str(file_info.get("quality") or "").lower()
    duration = float(video.get("duration") or 0.0)
    overlay_duration = float(plan_item["end"]) - float(plan_item["start"])
    slug = slug_tokens(video)
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
    must_include = {token.lower() for token in plan_item.get("must_include", []) if str(token).strip()}
    avoid = {token.lower() for token in plan_item.get("avoid", []) if str(token).strip()}
    query_tokens = set(semantic_keywords(matched_query, limit=10))
    orientation_bonus = 18 if video_orientation(width, height) == target_orientation else 0
    quality_bonus = 18 if quality == "hd" else 8
    resolution_bonus = min((width * height) / max(target_width * target_height, 1), 3.0) * 10
    duration_bonus = 10 if duration >= overlay_duration else 4
    overlap_bonus = len(slug & expected) * 8
    must_bonus = len(slug & must_include) * 12
    query_bonus = len(slug & query_tokens) * 6
    avoid_penalty = len(slug & avoid) * 12
    rank_bonus = max(0, 12 - query_rank * 3)
    return round(orientation_bonus + quality_bonus + resolution_bonus + duration_bonus + overlap_bonus + must_bonus + query_bonus + rank_bonus - avoid_penalty, 2)


def collect_search_candidates(
    plan_item: dict,
    target_orientation: str,
    target_width: int,
    target_height: int,
    search_fn=search_pexels_videos,
) -> tuple[list[dict], dict[str, str]]:
    candidates: list[dict] = []
    seen_video_ids: set[int] = set()
    latest_headers: dict[str, str] = {}
    for query_rank, query in enumerate(query_variants(plan_item)):
        videos, latest_headers = search_fn(query, orientation=target_orientation, per_page=6)
        for video in videos:
            video_id = int(video.get("id") or 0)
            if video_id and video_id in seen_video_ids:
                continue
            file_info = pick_video_file(video, target_orientation, target_width, target_height)
            if file_info is None:
                continue
            seen_video_ids.add(video_id)
            candidates.append(
                {
                    "result_id": f"cand_{len(candidates)+1:02d}",
                    "video": video,
                    "file_info": file_info,
                    "matched_query": query,
                    "query_rank": query_rank,
                    "score": heuristic_candidate_score(plan_item, video, file_info, query, query_rank, target_orientation, target_width, target_height),
                    "slug_tokens": sorted(slug_tokens(video)),
                }
            )
            if len(candidates) >= 10:
                break
        if len(candidates) >= 10:
            break
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates, latest_headers


def format_candidate_summaries(candidates: list[dict]) -> str:
    lines: list[str] = []
    for candidate in candidates[:8]:
        video = candidate["video"]
        file_info = candidate["file_info"]
        slug = ", ".join(candidate["slug_tokens"]) or "none"
        lines.append(
            "\n".join(
                [
                    f"{candidate['result_id']} | score={candidate['score']:.2f} | query={candidate['matched_query']}",
                    f"URL: {video.get('url')}",
                    f"Slug tokens: {slug}",
                    f"Duration: {video.get('duration')}s | File: {file_info.get('width')}x{file_info.get('height')} {file_info.get('quality')} {file_info.get('fps')}fps",
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
        return candidates[0], "Only viable candidate returned from Pexels search."
    system_prompt = (
        "You are selecting the best stock clip candidate for a precise subtitle-aligned B-roll insert. "
        "Choose the result whose semantics best match the subtitle and context. Prefer literal matches over generic mood footage. "
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
