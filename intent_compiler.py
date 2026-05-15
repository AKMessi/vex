from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from agent_fast_actions import detect_fast_action
from edit_plan import EditPlan, ToolStep


PLAN_CONFIDENCE_THRESHOLD = 0.78

_MEDIA_EXTENSIONS = r"mp4|mov|mkv|webm|avi|m4v|mpg|mpeg"
_NUMBER = r"\d+(?:\.\d+)?"
_TIME_UNIT = (
    r"milliseconds?|msecs?|ms|seconds?|secs?|sec|s|minutes?|mins?|min|m|"
    r"hours?|hrs?|hr|h"
)
_COLON_TIME = r"\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?"
_TIME_TOKEN = rf"(?:{_COLON_TIME}|{_NUMBER}\s*(?:{_TIME_UNIT})?)"
_CHAIN_SPLIT_RE = re.compile(
    r"\s*(?:;|\b(?:and\s+then|then|after\s+that|also|plus|followed\s+by)\b|"
    r"\band\s+(?=(?:export|encode|convert|compress|burn|add|remove|trim|cut|speed|merge|mute|transcribe|create|make|extract|redo|undo|grade|color|colour)\b))\s*",
    re.IGNORECASE,
)

_PLATFORM_PRESETS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\byoutube\s*(?:1080p|hd)?\b|\byt\b", re.IGNORECASE), "youtube_1080p"),
    (re.compile(r"\b(?:youtube\s*)?4k\b|\buhd\b", re.IGNORECASE), "youtube_4k"),
    (re.compile(r"\binstagram\s+(?:square|feed|post)\b|\bsquare\b", re.IGNORECASE), "instagram_square"),
    (re.compile(r"\binstagram(?:\s+(?:reels?|stories?))?\b|\breels?\b|\binsta\s+reels?\b", re.IGNORECASE), "instagram_reels"),
    (re.compile(r"\btiktok\b|\btik\s*tok\b", re.IGNORECASE), "tiktok"),
    (re.compile(r"\btwitter\b|\bx\.com\b|\bfor\s+x\b", re.IGNORECASE), "twitter_x"),
    (re.compile(r"\bpodcast\b|\baudio\s+only\b|\bmp3\b", re.IGNORECASE), "podcast_audio"),
)


def compile_intent(user_message: str, state: Any | None = None) -> EditPlan | None:
    segments = _split_segments(user_message)
    if not segments:
        return None

    steps: list[ToolStep] = []
    confidences: list[float] = []
    reasons: list[str] = []
    for segment in segments:
        compiled = _compile_segment(segment, state=state)
        if compiled is None:
            return None
        step, confidence, reason = compiled
        if step.tool == "__compound__":
            steps.extend(list(step.params.get("steps") or []))
        else:
            steps.append(step)
        confidences.append(confidence)
        reasons.append(reason)

    if not steps:
        return None
    confidence = min(confidences)
    if confidence < PLAN_CONFIDENCE_THRESHOLD:
        return None
    heavy_tools = {
        "transcribe_video",
        "summarize_clip",
        "create_auto_shorts",
        "add_auto_broll",
        "add_auto_visuals",
        "auto_color_grade",
    }
    return EditPlan(
        steps=steps,
        source="deterministic_intent",
        confidence=round(confidence, 3),
        reason="; ".join(reasons),
        requires_llm=False,
        can_run_async=any(step.tool in heavy_tools for step in steps),
        final_response_mode="tool_summary",
    )


def _split_segments(user_message: str) -> list[str]:
    text = _strip_media_paths(user_message)
    pieces = [piece.strip(" ,.") for piece in _CHAIN_SPLIT_RE.split(text) if piece.strip(" ,.")]
    if len(pieces) <= 1:
        return [_strip_command_filler(text)] if _strip_command_filler(text) else []
    cleaned_pieces = [_strip_command_filler(piece) for piece in pieces if _strip_command_filler(piece)]
    if cleaned_pieces and all(_looks_like_encode_request(piece) for piece in cleaned_pieces):
        return [_strip_command_filler(text)]
    return cleaned_pieces


def _strip_media_paths(user_message: str) -> str:
    text = user_message.strip()
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    text = re.sub(rf'"[^"]*(?:[\\/]|\.({_MEDIA_EXTENSIONS})\b)[^"]*"', " ", text, flags=re.IGNORECASE)
    text = re.sub(rf"'[^']*(?:[\\/]|\.({_MEDIA_EXTENSIONS})\b)[^']*'", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[a-zA-Z]:\\\S+", " ", text)
    text = re.sub(rf"\S+\.({_MEDIA_EXTENSIONS})\b", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def _strip_command_filler(text: str) -> str:
    cleaned = text.strip().lower()
    cleaned = re.sub(r"^(please|can you|could you|vex|now|next|and)\s+", "", cleaned)
    cleaned = re.sub(r"\b(?:it|the video|this video|the clip|this clip|my video)\b", "it", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _compile_segment(segment: str, *, state: Any | None) -> tuple[ToolStep, float, str] | None:
    return (
        _compile_undo_redo(segment)
        or _compile_info(segment)
        or _compile_trim(segment, state=state)
        or _compile_silence_trim(segment)
        or _compile_speed(segment)
        or _compile_mute(segment)
        or _compile_color_grade(segment)
        or _compile_subtitles(segment, state=state)
        or _compile_transcribe(segment)
        or _compile_extract_audio(segment)
        or _compile_encode(segment)
        or _compile_export(segment)
        or _compile_auto_visuals(segment)
        or _compile_auto_broll(segment)
        or _compile_shorts(segment)
        or _compile_summarize(segment)
    )


def _compile_undo_redo(segment: str) -> tuple[ToolStep, float, str] | None:
    if re.fullmatch(r"(?:undo|undo last|undo last edit|revert last edit)", segment):
        return ToolStep("undo", {}, "undo last operation"), 0.99, "undo command"
    if re.fullmatch(r"(?:redo|redo last|redo last edit)", segment):
        return ToolStep("redo", {}, "redo last operation"), 0.99, "redo command"
    return None


def _compile_info(segment: str) -> tuple[ToolStep, float, str] | None:
    if re.search(r"\b(?:video\s+info|metadata|clip\s+info|inspect|probe)\b", segment):
        return ToolStep("get_video_info", {}, "inspect video metadata"), 0.92, "video info command"
    return None


def _compile_trim(segment: str, *, state: Any | None) -> tuple[ToolStep, float, str] | None:
    action = detect_fast_action(segment, getattr(state, "metadata", {}) if state is not None else {})
    if action is None:
        return None
    return ToolStep(action.tool_name, dict(action.params), action.reason), 0.94, action.reason


def _compile_silence_trim(segment: str) -> tuple[ToolStep, float, str] | None:
    if not re.search(r"\b(?:remove|trim|cut|delete|clean(?:\s+up)?)\b", segment):
        return None
    if not re.search(r"\b(?:silence|silent|pauses?|dead\s+air|ums?|uhs?|filler)\b", segment):
        return None
    params: dict[str, Any] = {}
    if re.search(r"\b(?:aggressive|high|hard)\b", segment):
        params["aggressiveness"] = "high"
    elif re.search(r"\b(?:gentle|light|low|conservative)\b", segment):
        params["aggressiveness"] = "low"
    else:
        params["aggressiveness"] = "medium"
    if re.search(r"\b(?:start|end|edges?|beginning)\b", segment):
        params["trim_edges"] = True
    return ToolStep("trim_silence", params, "remove silent pauses"), 0.9, "silence trim command"


def _compile_speed(segment: str) -> tuple[ToolStep, float, str] | None:
    if not re.search(r"\b(?:speed|faster|slower|slow\s+down|speed\s+up)\b", segment):
        return None
    factor = _extract_speed_factor(segment)
    if factor is None:
        return None
    params: dict[str, Any] = {"factor": factor}
    range_match = _extract_range(segment)
    if range_match is not None:
        params["start"], params["end"] = range_match
    return ToolStep("adjust_speed", params, f"adjust speed to {factor}x"), 0.86, "speed adjustment command"


def _extract_speed_factor(segment: str) -> float | None:
    by_match = re.search(rf"\b(?:to|by|at)\s+({_NUMBER})\s*x\b", segment)
    if by_match:
        return _bounded_float(by_match.group(1), 0.25, 4.0)
    bare_match = re.search(rf"\b({_NUMBER})\s*x\s+(?:speed|faster|slower)?\b", segment)
    if bare_match:
        value = _bounded_float(bare_match.group(1), 0.25, 4.0)
        if value is not None:
            return value
    percent_match = re.search(rf"\b({_NUMBER})\s*%\s+(?:speed|faster|slower)\b", segment)
    if percent_match:
        value = _bounded_float(str(float(percent_match.group(1)) / 100.0), 0.25, 4.0)
        if value is not None:
            return value
    half_match = re.search(r"\b(?:half\s+speed|half-speed)\b", segment)
    if half_match:
        return 0.5
    double_match = re.search(r"\b(?:double\s+speed|twice\s+as\s+fast)\b", segment)
    if double_match:
        return 2.0
    return None


def _compile_mute(segment: str) -> tuple[ToolStep, float, str] | None:
    if not re.search(r"\b(?:mute|silence audio|remove audio)\b", segment):
        return None
    range_match = _extract_range(segment)
    if range_match is None:
        return None
    start, end = range_match
    return ToolStep("mute_segment", {"start": start, "end": end}, "mute timed segment"), 0.88, "mute segment command"


def _compile_color_grade(segment: str) -> tuple[ToolStep, float, str] | None:
    color_intent = re.search(
        r"\b(?:auto\s+)?colou?r\s+(?:grade|grading|correct|correction|balance)\b|"
        r"\b(?:grade|correct|fix|balance)\s+(?:the\s+)?colou?rs?\b|"
        r"\bwhite\s+balance\b|"
        r"\bmake\s+(?:the\s+)?colou?rs?\s+(?:pop|better|cleaner|natural|vibrant)\b",
        segment,
    )
    look_intent = re.search(r"\b(?:cinematic|filmic|vibrant|warm|cool|documentary|punchy)\s+(?:look|grade|colou?r)\b", segment)
    if color_intent is None and look_intent is None:
        return None
    params: dict[str, Any] = {}
    look = _extract_color_grade_look(segment)
    if look is not None:
        params["look"] = look
    intensity = _extract_color_grade_intensity(segment)
    if intensity is not None:
        params["intensity"] = intensity
    return ToolStep("auto_color_grade", params, "apply automatic color grade"), 0.86, "auto color grade command"


def _compile_subtitles(segment: str, *, state: Any | None) -> tuple[ToolStep, float, str] | None:
    if not re.search(r"\b(?:burn|add|overlay|put)\s+(?:in\s+)?(?:captions?|subtitles?)\b", segment):
        return None
    params: dict[str, Any] = {}
    if "top" in segment:
        params["position"] = "top"
    elif "center" in segment or "middle" in segment:
        params["position"] = "center"
    elif "bottom" in segment:
        params["position"] = "bottom"
    steps: list[ToolStep] = []
    transcript_path = Path(str(getattr(state, "working_dir", "") or "")) / "transcript.srt"
    if state is not None and not transcript_path.is_file() and re.search(r"\b(?:add|create|generate|caption|subtitle)\b", segment):
        steps.append(ToolStep("transcribe_video", {}, "generate transcript for subtitles"))
    steps.append(ToolStep("burn_subtitles", params, "burn subtitles"))
    if len(steps) == 1:
        return steps[0], 0.84, "burn subtitles command"
    return ToolStep("__compound__", {"steps": steps}, "transcribe then burn subtitles"), 0.84, "subtitle command"


def _compile_transcribe(segment: str) -> tuple[ToolStep, float, str] | None:
    if re.search(r"\b(?:transcribe|generate transcript|make transcript)\b", segment):
        return ToolStep("transcribe_video", {}, "generate transcript"), 0.9, "transcription command"
    return None


def _compile_extract_audio(segment: str) -> tuple[ToolStep, float, str] | None:
    if not re.search(r"\b(?:extract|save|pull)\s+(?:the\s+)?audio\b", segment):
        return None
    fmt = "mp3"
    if "wav" in segment:
        fmt = "wav"
    elif "aac" in segment or "m4a" in segment:
        fmt = "aac"
    return ToolStep("extract_audio", {"format": fmt}, f"extract audio as {fmt}"), 0.88, "extract audio command"


def _compile_encode(segment: str) -> tuple[ToolStep, float, str] | None:
    if not _looks_like_encode_request(segment):
        return None
    return ToolStep("plan_encode", {"raw_request": segment}, "plan encode command"), 0.86, "encode command"


def _looks_like_encode_request(segment: str) -> bool:
    if re.search(r"\b(?:encode|transcode|convert|compress|re-encode|reencode|remux)\b", segment):
        return True
    if re.search(r"\b(?:reduce|shrink|lower)\s+(?:the\s+)?(?:file\s+)?size\b", segment):
        return True
    if re.search(r"\b(?:make|keep|get)\s+it\s+(?:smaller|under|below)\b", segment):
        return True
    if re.search(r"\b(?:to|as|into)\s+\.?(?:mp4|mov|mkv|webm|m4v)\b", segment):
        return True
    if re.search(r"\b(?:h\.?264|x264|h\.?265|x265|hevc|av1|vp9|prores)\b", segment):
        return True
    if re.search(r"\b(?:under|below|less than|target(?:ing)?)\s+\d+(?:\.\d+)?\s*(?:mb|mib|gb|gib)\b", segment):
        return True
    return False


def _compile_export(segment: str) -> tuple[ToolStep, float, str] | None:
    if not re.search(r"\b(?:export|render|save)\b", segment):
        return None
    preset = _detect_export_preset(segment)
    if preset is None:
        return None
    return ToolStep("export_video", {"preset_name": preset}, f"export using {preset}"), 0.9, "export command"


def _detect_export_preset(segment: str) -> str | None:
    explicit = re.search(
        r"\b(youtube_1080p|youtube_4k|instagram_reels|instagram_square|tiktok|twitter_x|podcast_audio|custom)\b",
        segment,
    )
    if explicit:
        return explicit.group(1)
    for pattern, preset in _PLATFORM_PRESETS:
        if pattern.search(segment):
            return preset
    return None


def _compile_auto_visuals(segment: str) -> tuple[ToolStep, float, str] | None:
    if not re.search(r"\b(?:auto\s+visuals?|generated\s+visuals?|supporting\s+visuals?|cutaways?|animations?)\b", segment):
        return None
    params: dict[str, Any] = {"force_fullscreen": True}
    count = _extract_count(segment)
    if count is not None:
        params["max_visuals"] = max(1, min(count, 6))
    renderer = _extract_renderer(segment)
    if renderer:
        params["renderer"] = renderer
    return ToolStep("add_auto_visuals", params, "add generated visuals"), 0.84, "auto visuals command"


def _compile_auto_broll(segment: str) -> tuple[ToolStep, float, str] | None:
    if not re.search(r"\b(?:b[-\s]?roll|stock footage|stock video|cutaways?)\b", segment):
        return None
    params: dict[str, Any] = {}
    count = _extract_count(segment)
    if count is not None:
        params["max_overlays"] = max(1, min(count, 8))
    return ToolStep("add_auto_broll", params, "add stock b-roll"), 0.84, "auto b-roll command"


def _compile_shorts(segment: str) -> tuple[ToolStep, float, str] | None:
    if not re.search(r"\b(?:shorts?|reels?|tiktoks?|viral clips?)\b", segment):
        return None
    params: dict[str, Any] = {}
    count = _extract_count(segment)
    if count is not None:
        params["count"] = max(1, min(count, 8))
    if "tiktok" in segment:
        params["target_platform"] = "tiktok"
    elif "instagram" in segment or "reel" in segment:
        params["target_platform"] = "instagram_reels"
    else:
        params["target_platform"] = "youtube_shorts"
    return ToolStep("create_auto_shorts", params, "create short-form clips"), 0.86, "shorts command"


def _compile_summarize(segment: str) -> tuple[ToolStep, float, str] | None:
    if not re.search(r"\b(?:summarize|condense|highlight|highlights|best moments)\b", segment):
        return None
    params: dict[str, Any] = {}
    duration = _extract_target_duration(segment)
    if duration is not None:
        params["target_duration_sec"] = duration
    return ToolStep("summarize_clip", params, "summarize clip"), 0.82, "summarize command"


def _extract_range(segment: str) -> tuple[str, str] | None:
    range_match = re.search(rf"\b(?:from\s+)?({_TIME_TOKEN})\s*(?:-|to|through|thru|until)\s*({_TIME_TOKEN})\b", segment)
    if range_match:
        return _time_to_seconds_label(range_match.group(1)), _time_to_seconds_label(range_match.group(2))
    return None


def _extract_count(segment: str) -> int | None:
    match = re.search(
        r"\b(\d{1,2})\s+(?:(?:youtube|instagram|tiktok|viral|generated|auto|stock)\s+)?"
        r"(?:visuals?|overlays?|b[-\s]?roll|shorts?|reels?|tiktoks?|clips?)\b",
        segment,
    )
    if not match:
        return None
    return int(match.group(1))


def _extract_renderer(segment: str) -> str | None:
    for renderer in ("hyperframes", "manim", "ffmpeg", "blender"):
        if re.search(rf"\b{renderer}\b", segment):
            return renderer
    return None


def _extract_color_grade_look(segment: str) -> str | None:
    if re.search(r"\b(?:cinematic|cinema|filmic|film)\b", segment):
        return "cinematic"
    if re.search(r"\b(?:vibrant|pop|poppy|colorful|colourful)\b", segment):
        return "vibrant"
    if re.search(r"\b(?:warm|warmer|golden)\b", segment):
        return "warm"
    if re.search(r"\b(?:cool|cooler|blue)\b", segment):
        return "cool"
    if re.search(r"\b(?:documentary|natural|neutral|clean|balanced)\b", segment):
        return "natural" if "documentary" not in segment else "documentary"
    if re.search(r"\b(?:punchy|high\s+contrast)\b", segment):
        return "punchy"
    return None


def _extract_color_grade_intensity(segment: str) -> float | None:
    match = re.search(rf"\b(?:intensity|strength)\s+(?:of\s+)?({_NUMBER})\b", segment)
    if match:
        return _bounded_float(match.group(1), 0.0, 1.5)
    if re.search(r"\b(?:subtle|gentle|light|mild|conservative)\b", segment):
        return 0.65
    if re.search(r"\b(?:strong|heavy|dramatic|bold|intense)\b", segment):
        return 1.25
    return None


def _extract_target_duration(segment: str) -> int | None:
    match = re.search(rf"\b(?:to|into|under|about|around)\s+({_TIME_TOKEN})\b", segment)
    if not match:
        return None
    seconds = int(round(_parse_time_seconds(match.group(1))))
    return seconds if seconds > 0 else None


def _time_to_seconds_label(label: str) -> str:
    seconds = round(_parse_time_seconds(label), 3)
    if seconds.is_integer():
        return str(int(seconds))
    return f"{seconds:.3f}".rstrip("0").rstrip(".")


def _parse_time_seconds(label: str) -> float:
    raw = label.strip().lower()
    if ":" in raw:
        parts = [float(part) for part in raw.split(":")]
        if len(parts) == 2:
            minutes, seconds = parts
            return minutes * 60 + seconds
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return hours * 3600 + minutes * 60 + seconds
        raise ValueError(f"Invalid timestamp: {label!r}")
    match = re.fullmatch(rf"({_NUMBER})\s*({_TIME_UNIT})?", raw)
    if not match:
        raise ValueError(f"Invalid timestamp: {label!r}")
    value = float(match.group(1))
    unit = (match.group(2) or "s").strip()
    if unit in {"ms", "msec", "msecs"} or unit.startswith("millisecond"):
        return value / 1000.0
    if unit in {"m", "min", "mins"} or unit.startswith("minute"):
        return value * 60.0
    if unit in {"h", "hr", "hrs"} or unit.startswith("hour"):
        return value * 3600.0
    return value


def _bounded_float(raw: str, minimum: float, maximum: float) -> float | None:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if minimum <= value <= maximum:
        return value
    return None
