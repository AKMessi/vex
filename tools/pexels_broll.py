from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from google import genai
from google.genai import types

import config
from engine import VideoEngineError, apply_b_roll_overlays, probe_video
from state import ProjectState, utc_now_iso
from tools.transcript import execute as transcribe
from tools.transcript_utils import parse_srt

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "for",
    "with", "this", "that", "these", "those", "you", "your", "our", "their",
    "from", "into", "over", "under", "about", "just", "than", "then",
    "they", "them", "have", "has", "had", "was", "were", "are", "is",
    "be", "been", "being", "what", "when", "where", "which",
}
VISUAL_TYPE_HINTS = {
    "data_graphic": ["chart animation", "analytics screen", "numbers dashboard"],
    "product_ui": ["product dashboard", "app screen", "software workflow"],
    "cutaway": ["people working", "team meeting", "person thinking"],
    "abstract_motion": ["abstract motion", "technology background", "digital animation"],
    "nature": ["city timelapse", "nature motion", "cinematic background"],
}


def _truncate(text: str, limit: int) -> str:
    collapsed = re.sub(r"\s+", " ", text).strip()
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3].rstrip() + "..."


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9']+", text.lower())


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


def _format_timestamped_segments(segments: list[dict[str, float | str]]) -> str:
    return "\n".join(
        f"{float(segment['start']):.2f}-{float(segment['end']):.2f}: {str(segment['text']).strip()}"
        for segment in segments
        if str(segment["text"]).strip()
    )


def _fallback_b_roll_plan(
    segments: list[dict[str, float | str]],
    clip_duration: float,
    max_overlays: int,
    min_overlay_sec: float,
    max_overlay_sec: float,
) -> list[dict]:
    suggestions: list[dict] = []
    last_end = -999.0
    for segment in segments:
        text = str(segment["text"]).strip()
        if not text:
            continue
        start_sec = float(segment["start"])
        end_sec = float(segment["end"])
        if start_sec - last_end < 1.2:
            continue
        lower = text.lower()
        if any(term in lower for term in {"chart", "metric", "growth", "percent", "data", "revenue", "users"}):
            visual_type = "data_graphic"
            direction = "Use a metrics-focused stock visual that reinforces the spoken number or outcome."
        elif any(term in lower for term in {"tool", "app", "product", "website", "dashboard", "workflow", "agent"}):
            visual_type = "product_ui"
            direction = "Use a screen-centric or workflow stock shot that makes the software point feel concrete."
        elif any(term in lower for term in {"team", "founder", "people", "creator", "audience", "customer"}):
            visual_type = "cutaway"
            direction = "Use a human cutaway that supports the narration without overwhelming it."
        else:
            visual_type = "abstract_motion"
            direction = "Use a clean supporting motion shot that adds energy without distracting from the voice."
        target_duration = min(max(end_sec - start_sec, min_overlay_sec), max_overlay_sec)
        clip_end = min(start_sec + target_duration, clip_duration)
        suggestions.append(
            {
                "start": round(max(start_sec, 0.0), 2),
                "end": round(max(clip_end, start_sec + min_overlay_sec), 2),
                "visual_type": visual_type,
                "search_query": _truncate(_keyword_phrase(text, limit=6), 70),
                "direction": _truncate(direction, 120),
                "rationale": _truncate("This line has enough semantic density to support a visual cutaway while the original audio keeps carrying the story.", 140),
            }
        )
        last_end = clip_end
        if len(suggestions) >= max_overlays:
            break
    if not suggestions and clip_duration > 0:
        suggestions.append(
            {
                "start": 0.0,
                "end": round(min(max_overlay_sec, clip_duration), 2),
                "visual_type": "abstract_motion",
                "search_query": "cinematic background",
                "direction": "Use a broad establishing motion shot to add visual texture at the start.",
                "rationale": "Fallback establishing insert when transcript cues are weak.",
            }
        )
    return suggestions


def _normalize_b_roll_plan(
    raw_suggestions: list[dict],
    fallback: list[dict],
    clip_duration: float,
    max_overlays: int,
    min_overlay_sec: float,
    max_overlay_sec: float,
) -> list[dict]:
    suggestions: list[dict] = []
    source = raw_suggestions or fallback
    last_end = -999.0
    for item in source:
        try:
            start_sec = max(0.0, min(float(item.get("start", 0.0)), clip_duration))
            requested_end = float(item.get("end", start_sec + max_overlay_sec))
            end_sec = min(max(start_sec + min_overlay_sec, requested_end), clip_duration)
        except Exception:
            continue
        if end_sec <= start_sec or start_sec - last_end < 0.8:
            continue
        if end_sec - start_sec > max_overlay_sec:
            end_sec = start_sec + max_overlay_sec
        suggestions.append(
            {
                "start": round(start_sec, 2),
                "end": round(end_sec, 2),
                "visual_type": _truncate(str(item.get("visual_type") or "abstract_motion"), 32),
                "search_query": _truncate(str(item.get("search_query") or "supporting footage"), 80),
                "direction": _truncate(str(item.get("direction") or "Add a supporting stock visual."), 120),
                "rationale": _truncate(str(item.get("rationale") or "Supports the spoken point visually."), 140),
            }
        )
        last_end = end_sec
        if len(suggestions) >= max_overlays:
            break
    return suggestions or fallback[:max_overlays]


def _analyze_b_roll_plan_with_llm(
    provider_name: str,
    model_name: str,
    segments: list[dict[str, float | str]],
    clip_duration: float,
    max_overlays: int,
    min_overlay_sec: float,
    max_overlay_sec: float,
    orientation: str,
) -> list[dict]:
    fallback = _fallback_b_roll_plan(
        segments=segments,
        clip_duration=clip_duration,
        max_overlays=max_overlays,
        min_overlay_sec=min_overlay_sec,
        max_overlay_sec=max_overlay_sec,
    )
    system_prompt = (
        "You are a video editor planning stock B-roll cutaways. "
        "Return ONLY a JSON array of up to {count} objects with keys start, end, visual_type, search_query, direction, rationale. "
        "Pick beats where stock footage can strengthen the pacing while the original audio continues underneath."
    ).format(count=max_overlays)
    user_prompt = (
        f"Video duration: {clip_duration:.2f}s\n"
        f"Orientation target: {orientation}\n"
        f"Need at most {max_overlays} B-roll inserts.\n"
        f"Each insert must stay between {min_overlay_sec:.1f}s and {max_overlay_sec:.1f}s.\n\n"
        f"Timestamped transcript:\n{_truncate(_format_timestamped_segments(segments), 5000)}\n\n"
        "Prefer specific, searchable stock queries. Avoid overlapping ranges. Return JSON array only."
    )
    try:
        raw_text = _call_reasoning_model(provider_name, model_name, system_prompt, user_prompt)
        parsed = json.loads(_extract_json_array(raw_text))
    except Exception:
        return fallback
    return _normalize_b_roll_plan(
        raw_suggestions=parsed,
        fallback=fallback,
        clip_duration=clip_duration,
        max_overlays=max_overlays,
        min_overlay_sec=min_overlay_sec,
        max_overlay_sec=max_overlay_sec,
    )


def _video_orientation(width: int, height: int) -> str:
    if height > width:
        return "portrait"
    if width > height:
        return "landscape"
    return "square"


def _pexels_get_json(url: str) -> tuple[dict, dict[str, str]]:
    if not config.PEXELS_API_KEY:
        raise RuntimeError("PEXELS_API_KEY is required for auto B-roll.")
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": config.PEXELS_API_KEY,
            "Accept": "application/json",
            "User-Agent": "Vex/1.0 (+https://github.com/AKMessi/vex)",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
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


def _search_pexels_videos(query: str, orientation: str, per_page: int = 8) -> tuple[list[dict], dict[str, str]]:
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
    payload, headers = _pexels_get_json(f"https://api.pexels.com/v1/videos/search?{params}")
    return list(payload.get("videos") or []), headers


def _pick_video_file(video: dict, target_orientation: str, target_width: int, target_height: int) -> dict | None:
    best_file = None
    best_score = None
    for item in video.get("video_files") or []:
        if str(item.get("file_type") or "").lower() != "video/mp4":
            continue
        width = int(item.get("width") or 0)
        height = int(item.get("height") or 0)
        if width <= 0 or height <= 0:
            continue
        orientation_bonus = 18 if _video_orientation(width, height) == target_orientation else 0
        quality = str(item.get("quality") or "").lower()
        quality_bonus = 20 if quality == "hd" else 8
        resolution_bonus = min((width * height) / max(target_width * target_height, 1), 3.0) * 12
        fps_bonus = min(float(item.get("fps") or 0.0), 60.0) / 10.0
        score = orientation_bonus + quality_bonus + resolution_bonus + fps_bonus
        if best_score is None or score > best_score:
            best_score = score
            best_file = item
    return best_file


def _download_file(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Vex/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            destination.write_bytes(response.read())
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to download stock clip: {exc.reason}") from exc
    return destination


def _candidate_queries(plan_item: dict) -> list[str]:
    base_query = str(plan_item.get("search_query") or "").strip()
    visual_type = str(plan_item.get("visual_type") or "").strip().lower()
    queries: list[str] = []
    for value in [base_query, _keyword_phrase(base_query, limit=4), *(VISUAL_TYPE_HINTS.get(visual_type, []))]:
        normalized = re.sub(r"\s+", " ", str(value).strip())
        if normalized and normalized not in queries:
            queries.append(normalized)
    return queries[:4]


def _ensure_transcript_segments(state: ProjectState) -> tuple[Path, list[dict[str, float | str]]]:
    srt_path = Path(state.working_dir) / "transcript.srt"
    if not srt_path.exists():
        result = transcribe({}, state)
        if not result["success"]:
            raise RuntimeError(result["message"])
    segments = parse_srt(srt_path)
    if not segments:
        raise RuntimeError("Transcript was empty, so Vex could not plan B-roll beats.")
    return srt_path, segments


def execute(params: dict, state: ProjectState) -> dict:
    if not config.PEXELS_API_KEY:
        return {
            "success": False,
            "message": "PEXELS_API_KEY is missing. Set it in your environment or .env file to enable auto B-roll.",
            "suggestion": None,
            "updated_state": state,
            "tool_name": "add_auto_broll",
        }

    max_overlays = max(1, min(int(params.get("max_overlays", 5) or 5), 8))
    min_overlay_sec = max(0.8, min(float(params.get("min_overlay_sec", 1.2) or 1.2), 6.0))
    max_overlay_sec = max(min_overlay_sec, min(float(params.get("max_overlay_sec", 2.8) or 2.8), 8.0))

    try:
        srt_path, transcript_segments = _ensure_transcript_segments(state)
        metadata = state.metadata or probe_video(state.working_file)
        clip_duration = float(metadata.get("duration_sec") or 0.0)
        target_orientation = _video_orientation(int(metadata.get("width") or 0), int(metadata.get("height") or 0))
        provider_name = (state.provider or config.PROVIDER or "gemini").strip().lower()
        if provider_name not in {"gemini", "claude"}:
            provider_name = "gemini"
        model_name = state.model or (
            config.CLAUDE_MODEL if provider_name == "claude" else config.GEMINI_MODEL
        )

        plan = _analyze_b_roll_plan_with_llm(
            provider_name=provider_name,
            model_name=model_name,
            segments=transcript_segments,
            clip_duration=clip_duration,
            max_overlays=max_overlays,
            min_overlay_sec=min_overlay_sec,
            max_overlay_sec=max_overlay_sec,
            orientation=target_orientation,
        )

        cache_dir = Path(state.working_dir) / "pexels_cache"
        used_video_ids: set[int] = set()
        applied_overlays: list[dict] = []
        rate_limits: dict[str, str] = {}
        for plan_item in plan:
            selected_video = None
            selected_file = None
            selected_query = None
            for query in _candidate_queries(plan_item):
                videos, rate_limits = _search_pexels_videos(query, orientation=target_orientation, per_page=8)
                for video in videos:
                    video_id = int(video.get("id") or 0)
                    if video_id and video_id in used_video_ids:
                        continue
                    candidate_file = _pick_video_file(
                        video,
                        target_orientation=target_orientation,
                        target_width=int(metadata.get("width") or 1080),
                        target_height=int(metadata.get("height") or 1920),
                    )
                    if candidate_file is None:
                        continue
                    selected_video = video
                    selected_file = candidate_file
                    selected_query = query
                    break
                if selected_video is not None and selected_file is not None:
                    break
            if selected_video is None or selected_file is None or selected_query is None:
                continue

            used_video_ids.add(int(selected_video.get("id") or 0))
            file_token = str(selected_file.get("id") or f"{selected_file.get('width')}_{selected_file.get('height')}" or "stock")
            asset_path = cache_dir / f"pexels_{selected_video['id']}_{file_token}.mp4"
            if not asset_path.exists():
                _download_file(str(selected_file["link"]), asset_path)
            applied_overlays.append(
                {
                    "start": float(plan_item["start"]),
                    "end": float(plan_item["end"]),
                    "visual_type": plan_item["visual_type"],
                    "search_query": plan_item["search_query"],
                    "query_used": selected_query,
                    "direction": plan_item["direction"],
                    "rationale": plan_item["rationale"],
                    "asset_path": str(asset_path),
                    "pexels_video_id": selected_video.get("id"),
                    "pexels_url": selected_video.get("url"),
                    "creator_name": (selected_video.get("user") or {}).get("name"),
                    "creator_url": (selected_video.get("user") or {}).get("url"),
                    "preview_image": selected_video.get("image"),
                    "file_width": selected_file.get("width"),
                    "file_height": selected_file.get("height"),
                    "file_fps": selected_file.get("fps"),
                }
            )

        if not applied_overlays:
            return {
                "success": False,
                "message": "Vex planned B-roll beats, but Pexels did not return usable stock clips for the selected queries.",
                "suggestion": None,
                "updated_state": state,
                "tool_name": "add_auto_broll",
            }

        output_path = apply_b_roll_overlays(state.working_file, state.working_dir, applied_overlays)
        state.working_file = output_path
        state.metadata = probe_video(output_path)

        timestamp_label = utc_now_iso().replace(":", "-").replace("+00:00", "Z")
        bundle_stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", state.project_name).strip("_") or "project"
        bundle_dir = Path(state.output_dir) / f"{bundle_stem}_auto_broll_{timestamp_label}"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "created_at": utc_now_iso(),
            "project_id": state.project_id,
            "project_name": state.project_name,
            "source_video": state.source_files[0] if state.source_files else state.working_file,
            "working_file": state.working_file,
            "transcript_srt": str(srt_path),
            "pexels_attribution_required": True,
            "pexels_link": "https://www.pexels.com",
            "rate_limits": rate_limits,
            "overlays": applied_overlays,
        }
        manifest_path = bundle_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        credits_lines = [
            "# Pexels Attribution",
            "",
            "Photos and videos provided by Pexels: https://www.pexels.com",
            "",
        ]
        for index, item in enumerate(applied_overlays, start=1):
            credits_lines.extend(
                [
                    f"{index}. {item['start']:.2f}s-{item['end']:.2f}s",
                    f"   Query used: {item['query_used']}",
                    f"   Pexels video: {item.get('pexels_url') or 'unknown'}",
                    f"   Creator: {item.get('creator_name') or 'unknown'} ({item.get('creator_url') or 'n/a'})",
                    "",
                ]
            )
        (bundle_dir / "pexels_attribution.md").write_text("\n".join(credits_lines), encoding="utf-8")

        state.artifacts["latest_auto_broll"] = {
            "created_at": manifest["created_at"],
            "manifest_path": str(manifest_path),
            "bundle_dir": str(bundle_dir),
            "count": len(applied_overlays),
        }
        history = list(state.artifacts.get("auto_broll_history") or [])
        history.append(state.artifacts["latest_auto_broll"])
        state.artifacts["auto_broll_history"] = history[-10:]
        state.apply_operation(
            {
                "op": "add_auto_broll",
                "params": {
                    "max_overlays": max_overlays,
                    "min_overlay_sec": min_overlay_sec,
                    "max_overlay_sec": max_overlay_sec,
                },
                "timestamp": utc_now_iso(),
                "result_file": output_path,
                "description": f"Added {len(applied_overlays)} auto B-roll overlays from Pexels",
            }
        )
        return {
            "success": True,
            "message": (
                f"Added {len(applied_overlays)} auto B-roll overlays from Pexels. "
                f"Manifest: {manifest_path}"
            ),
            "suggestion": None,
            "updated_state": state,
            "tool_name": "add_auto_broll",
        }
    except (RuntimeError, VideoEngineError) as exc:
        return {
            "success": False,
            "message": str(exc),
            "suggestion": None,
            "updated_state": state,
            "tool_name": "add_auto_broll",
        }
