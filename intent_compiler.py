from __future__ import annotations

import re
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
    r"\band\s+(?=(?:export|encode|convert|compress|burn|add|remove|trim|cut|speed|merge|mute|transcribe|create|make|generate|extract|redo|undo|grade|color|colour|zoom|effect)\b))\s*",
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
    "add_song",
    "add_visual_asset",
        "add_auto_effects",
        "auto_color_grade",
        "upscale_video",
        "generate_video",
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
    if re.search(r"\b(?:use|add|insert|put|overlay)\b", text, flags=re.IGNORECASE) and re.search(
        r"\.(?:html?|mp4|mov|m4v|webm|gif|png|jpe?g|webp|bmp)\b",
        text,
        flags=re.IGNORECASE,
    ):
        return re.sub(r"\s+", " ", text).strip()
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
        or _compile_auto_effects(segment)
        or _compile_subtitles(segment, state=state)
        or _compile_transcribe(segment)
        or _compile_generate_video(segment)
        or _compile_extract_audio(segment)
        or _compile_add_song(segment)
        or _compile_encode(segment)
        or _compile_upscale(segment)
        or _compile_export(segment)
        or _compile_manual_visual_asset(segment)
        or _compile_directed_hyperframes_visual(segment)
        or _compile_manual_blender_visual(segment)
        or _compile_auto_visuals(segment)
        or _compile_auto_broll(segment, state=state)
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
    if not re.search(r"\b(?:burn|add|overlay|put)\s+(?:in\s+)?(?:[a-z0-9_-]+\s+){0,4}(?:captions?|subtitles?)\b", segment):
        return None
    params: dict[str, Any] = {}
    if "top" in segment:
        params["position"] = "top"
    elif "center" in segment or "middle" in segment:
        params["position"] = "center"
    elif "bottom" in segment:
        params["position"] = "bottom"
    if re.search(r"\b(?:creator|bold|tiktok|reels?|shorts?)\s+(?:style\s+)?(?:captions?|subtitles?)\b", segment):
        params["style"] = "creator_bold"
    elif re.search(r"\b(?:cinematic|film|movie)\s+(?:style\s+)?(?:captions?|subtitles?)\b", segment):
        params["style"] = "cinematic"
    elif re.search(r"\b(?:glass|premium)\s+(?:style\s+)?(?:captions?|subtitles?)\b", segment):
        params["style"] = "glass"
    elif re.search(r"\b(?:karaoke|highlight(?:ed)?|focus)\s+(?:style\s+)?(?:captions?|subtitles?)\b", segment):
        params["style"] = "karaoke_focus"
    elif re.search(r"\b(?:minimal|simple|clean)\s+(?:style\s+)?(?:captions?|subtitles?)\b", segment):
        params["style"] = "minimal" if re.search(r"\b(?:minimal|simple)\b", segment) else "clean_pop"
    steps: list[ToolStep] = []
    transcript_path = None
    working_dir = str(getattr(state, "working_dir", "") or "").strip()
    if state is not None and working_dir:
        from tools.transcript_utils import transcript_artifact_path

        transcript_path = transcript_artifact_path(working_dir, "transcript.srt")
    if state is not None and transcript_path is None and re.search(r"\b(?:add|create|generate|caption|subtitle)\b", segment):
        steps.append(ToolStep("transcribe_video", {}, "generate transcript for subtitles"))
    steps.append(ToolStep("burn_subtitles", params, "burn subtitles"))
    if len(steps) == 1:
        return steps[0], 0.84, "burn subtitles command"
    return ToolStep("__compound__", {"steps": steps}, "transcribe then burn subtitles"), 0.84, "subtitle command"


def _compile_transcribe(segment: str) -> tuple[ToolStep, float, str] | None:
    if re.search(r"\b(?:transcribe|generate transcript|make transcript)\b", segment):
        return ToolStep("transcribe_video", {}, "generate transcript"), 0.9, "transcription command"
    return None


def _compile_generate_video(segment: str) -> tuple[ToolStep, float, str] | None:
    if not re.search(r"\b(?:generate|create|make|build|produce)\b", segment):
        return None
    if not re.search(r"\b(?:new\s+)?(?:hyperframes?\s+)?video\b", segment):
        return None
    if re.search(r"\b(?:shorts?|reels?|tiktoks?|viral clips?)\b", segment) and not re.search(
        r"\b(?:from\s+scratch|new\s+video|generate\s+(?:a\s+)?video|hyperframes?\s+video)\b",
        segment,
    ):
        return None
    prompt = _extract_generation_prompt(segment)
    if len(prompt.split()) < 2:
        return None
    params: dict[str, Any] = {"prompt": prompt}
    duration = _extract_target_duration(segment)
    if duration is not None:
        params["duration_sec"] = duration
    if re.search(r"\b(?:portrait|vertical|reels?|tiktok|shorts?)\b", segment):
        params["aspect"] = "portrait"
    elif re.search(r"\b(?:square|1:1)\b", segment):
        params["aspect"] = "square"
    elif re.search(r"\b(?:landscape|horizontal|16:9)\b", segment):
        params["aspect"] = "landscape"
    if re.search(r"\b(?:project\s+only|no\s+render|do\s+not\s+render|without\s+rendering)\b", segment):
        params["render"] = False
    if re.search(r"\b(?:silent|no\s+audio|without\s+audio|no\s+narration)\b", segment):
        params["generate_audio"] = False
    voice_match = re.search(r"\bvoice\s+([a-z]{2}_[a-z0-9_-]+)\b", segment)
    if voice_match:
        params["voice"] = voice_match.group(1)
    return ToolStep("generate_video", params, "generate new HyperFrames video"), 0.86, "generate video command"


def _compile_extract_audio(segment: str) -> tuple[ToolStep, float, str] | None:
    if not re.search(r"\b(?:extract|save|pull)\s+(?:the\s+)?audio\b", segment):
        return None
    fmt = "mp3"
    if "wav" in segment:
        fmt = "wav"
    elif "aac" in segment or "m4a" in segment:
        fmt = "aac"
    return ToolStep("extract_audio", {"format": fmt}, f"extract audio as {fmt}"), 0.88, "extract audio command"


def _compile_add_song(segment: str) -> tuple[ToolStep, float, str] | None:
    song_path = _extract_audio_asset_path(segment)
    song_intent = re.search(
        r"\b(?:add|put|insert|mix|use|attach)\s+(?:a\s+|the\s+|some\s+)?"
        r"(?:song|music|soundtrack|track|audio\s+bed|background\s+music)\b|"
        r"\b(?:background\s+music|music\s+bed|song\s+under|soundtrack)\b",
        segment,
    )
    if song_intent is None and song_path is None:
        return None
    if song_path is None and not re.search(r"\b(?:song|music|soundtrack|track)\b", segment):
        return None

    params: dict[str, Any] = {}
    if song_path is not None:
        params["song_path"] = song_path
    mode = _extract_song_mode(segment)
    if mode:
        params["mode"] = mode
    range_match = _extract_range(segment)
    if range_match is not None:
        params["start"], params["end"] = range_match
        if params.get("mode") != "replace":
            params["mode"] = "segment"
    volume = _extract_song_volume(segment)
    if volume is not None:
        params["volume"] = volume
    if re.search(r"\b(?:no\s+ducking|do\s+not\s+duck|without\s+ducking)\b", segment):
        params["ducking"] = "off"
    elif re.search(r"\b(?:duck|ducking|lower\s+(?:the\s+)?music\s+under\s+(?:speech|voice|dialogue|dialog))\b", segment):
        params["ducking"] = "on"
    if re.search(r"\b(?:do\s+not\s+loop|no\s+loop|without\s+looping)\b", segment):
        params["loop_policy"] = "trim"
    elif re.search(r"\b(?:loop|repeat)\b", segment):
        params["loop_policy"] = "loop"
    if re.search(r"\b(?:replace|swap)\s+(?:the\s+)?(?:audio|soundtrack|music)\b", segment):
        params["mode"] = "replace"
    return ToolStep("add_song", params, "add song to video"), 0.86, "add song command"


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


def _compile_upscale(segment: str) -> tuple[ToolStep, float, str] | None:
    if not re.search(r"\b(?:upscale|scale|resize)\b", segment):
        return None
    resolution = _extract_resolution(segment)
    if resolution is None:
        if re.search(r"\b1080p\b|\bfull\s*hd\b", segment):
            resolution = "1920x1080"
        elif re.search(r"\b4k\b|\buhd\b", segment):
            resolution = "3840x2160"
    if resolution is None:
        return None
    params: dict[str, Any] = {"resolution": resolution}
    if re.search(r"\b(?:fill|cover|crop)\b", segment):
        params["scale_mode"] = "fill"
    elif re.search(r"\b(?:stretch|distort)\b", segment):
        params["scale_mode"] = "stretch"
    else:
        params["scale_mode"] = "fit"
    return ToolStep("upscale_video", params, f"scale video to {resolution}"), 0.88, "upscale video command"


def _compile_manual_visual_asset(segment: str) -> tuple[ToolStep, float, str] | None:
    asset_path = _extract_visual_asset_path(segment)
    if asset_path is None:
        return None
    if not re.search(r"\b(?:use|add|insert|put|overlay)\b", segment):
        return None
    range_match = _extract_range(segment)
    if range_match is None:
        return None
    start, end = range_match
    mode = "replace"
    if re.search(r"\b(?:overlay|overlaid)\b", segment):
        mode = "overlay"
    if re.search(r"\b(?:pip|picture[-\s]?in[-\s]?picture)\b", segment):
        mode = "picture_in_picture"
    return (
        ToolStep(
            "add_visual_asset",
            {
                "asset_path": asset_path,
                "start": start,
                "end": end,
                "composition_mode": mode,
            },
            "insert manual visual asset",
        ),
        0.88,
        "manual visual asset command",
    )
    return None


def _compile_auto_visuals(segment: str) -> tuple[ToolStep, float, str] | None:
    if not re.search(r"\b(?:auto\s+visuals?|generated\s+visuals?|supporting\s+visuals?|cutaways?|animations?)\b", segment):
        return None
    params: dict[str, Any] = {"force_fullscreen": True}
    count = _extract_count(segment)
    if count is not None:
        requested_count = max(1, min(count, 32))
        params["max_visuals"] = requested_count
        params["requested_count"] = requested_count
        params["coverage_policy"] = "target_count"
    density = _extract_density(segment)
    if density:
        params["density"] = density
    renderer = _extract_renderer(segment)
    if renderer:
        params["renderer"] = renderer
    return ToolStep("add_auto_visuals", params, "add generated visuals"), 0.84, "auto visuals command"


def _compile_directed_hyperframes_visual(segment: str) -> tuple[ToolStep, float, str] | None:
    if not re.search(r"\bhyperframes?\b", segment):
        return None
    if not re.search(
        r"\b(?:add|create|make|insert|put|use|show|visuali[sz]e|depict|animate|custom|idea)\b",
        segment,
    ):
        return None
    if re.search(
        r"\b(?:auto\s+visuals?|generated\s+visuals?|supporting\s+visuals?)\b",
        segment,
    ) and not re.search(
        r"\b(?:show(?:ing)?|visuali[sz](?:e|ing)|depict(?:ing)?|animate|where|that|of|idea|as)\b",
        segment,
    ):
        return None

    idea = _extract_directed_hyperframes_idea(segment)
    if len(idea.split()) < 2:
        return None

    spec: dict[str, Any] = {
        "visual_idea": idea,
        "renderer_hint": "hyperframes",
        "composition_mode": "replace",
        "rationale": "User-directed HyperFrames visual idea.",
    }
    range_match = _extract_range(segment)
    if range_match is not None:
        spec["start"], spec["end"] = range_match
    else:
        at_match = re.search(rf"\b(?:at|from|starting\s+at)\s+({_TIME_TOKEN})\b", segment)
        duration_match = re.search(rf"\bfor\s+({_TIME_TOKEN})\b", segment)
        if at_match:
            duration = _parse_time_seconds(duration_match.group(1)) if duration_match else 4.0
            start_seconds = _parse_time_seconds(at_match.group(1))
            spec["start"] = _time_to_seconds_label(at_match.group(1))
            spec["end"] = _time_to_seconds_label(f"{start_seconds + duration}s")
    trigger_text = _extract_trigger_text(segment)
    if trigger_text and "start" not in spec:
        spec["trigger_text"] = trigger_text

    return (
        ToolStep(
            "add_auto_visuals",
            {
                "renderer": "hyperframes",
                "force_fullscreen": True,
                "max_visuals": 1,
                "directed_visual_specs": [spec],
            },
            "add directed HyperFrames visual",
        ),
        0.87,
        "directed hyperframes visual command",
    )


def _compile_manual_blender_visual(segment: str) -> tuple[ToolStep, float, str] | None:
    if not re.search(
        r"\b(?:3d|three[-\s]?d|blender|rotating|floating|arrow|pointer|data\s+tunnel|neural|gpu|chip|logo|product\s+model|model\s+spin|glb|gltf|obj|blend)\b",
        segment,
    ):
        return None
    if not re.search(r"\b(?:add|create|insert|put|make|spin|show)\b", segment):
        return None

    template = "three_d_title"
    composition_mode = "replace"
    camera_motion = "slow_push"
    object_motion = "none"
    visual_type_hint = "abstract_motion"
    if re.search(r"\b(?:floating|label|badge|callout)\b", segment):
        template = "floating_3d_label"
        composition_mode = "overlay"
        camera_motion = "static"
        object_motion = "float"
    if re.search(r"\b(?:arrow|pointer|pointing|chart)\b", segment):
        template = "screen_pointer_3d"
        composition_mode = "overlay"
        camera_motion = "static"
        object_motion = "float"
    if re.search(r"\b(?:data\s+tunnel|neural|network|gpu|chip|data\s+flow)\b", segment):
        template = "data_tunnel"
        composition_mode = "replace"
        camera_motion = "orbit"
        visual_type_hint = "abstract_motion"
    if re.search(r"\b(?:logo|brand)\b", segment):
        template = "logo_reveal"
        composition_mode = "replace"
        object_motion = "drop_in"
    if re.search(r"\b(?:product|model\s+spin|turntable)\b", segment):
        template = "product_model_spin"
        composition_mode = "overlay" if re.search(r"\boverlay\b", segment) else "replace"
        camera_motion = "orbit"
        object_motion = "spin_y"
        visual_type_hint = "product_ui"
    if re.search(r"\b(?:object\s+orbit|orbiting\s+object)\b", segment):
        template = "object_orbit"
        composition_mode = "replace"
        camera_motion = "orbit"
        object_motion = "spin_y"
    if re.search(r"\b(?:rotating|rotate|spin|spinning)\b", segment):
        object_motion = "spin_y"
        camera_motion = "orbit" if template not in {"floating_3d_label", "screen_pointer_3d"} else camera_motion

    range_match = _extract_range(segment)
    start = "0"
    end = "4"
    if range_match is not None:
        start, end = range_match
    else:
        at_match = re.search(rf"\b(?:at|from|starting\s+at)\s+({_TIME_TOKEN})\b", segment)
        duration_match = re.search(rf"\bfor\s+({_TIME_TOKEN})\b", segment)
        duration = _parse_time_seconds(duration_match.group(1)) if duration_match else 4.0
        if at_match:
            start_seconds = _parse_time_seconds(at_match.group(1))
            start = _time_to_seconds_label(at_match.group(1))
            end = _time_to_seconds_label(f"{start_seconds + duration}s")

    title = _extract_blender_visual_text(segment)
    asset_path = _extract_blender_asset_path(segment)
    trigger_text = _extract_trigger_text(segment)
    position = _extract_visual_position(segment)
    spec: dict[str, Any] = {
        "template": template,
        "composition_mode": composition_mode,
        "start": start,
        "end": end,
        "headline": title,
        "text": title,
        "label": title,
        "position": position,
        "style": "cinematic_dark" if template != "floating_3d_label" else "glass",
        "camera_motion": camera_motion,
        "object_motion": object_motion,
        "alpha": composition_mode == "overlay",
        "transparent_background": composition_mode == "overlay",
        "visual_type_hint": visual_type_hint,
        "rationale": "User-requested typed Blender 3D visual.",
    }
    if asset_path:
        spec["asset_path"] = asset_path
    if trigger_text:
        spec["trigger_text"] = trigger_text
        spec.pop("start", None)
        spec.pop("end", None)

    return (
        ToolStep(
            "add_auto_visuals",
            {
                "renderer": "blender",
                "force_fullscreen": composition_mode == "replace",
                "max_visuals": 1,
                "manual_visual_specs": [spec],
            },
            "add typed Blender 3D visual",
        ),
        0.86,
        "typed blender 3D visual command",
    )


def _compile_auto_effects(segment: str) -> tuple[ToolStep, float, str] | None:
    if not re.search(
        r"\badd_auto_effects\b|"
        r"\b(?:auto\s+)?(?:effects?|zooms?|zoom\s+effects?|punch[-\s]?ins?|camera\s+movement|emphasis\s+effects?)\b|"
        r"\bsubtitles?[-\s]?aware\s+(?:auto\s+)?(?:effects?|zooms?|emphasis)\b|"
        r"\bcaptions?[-\s]?aware\s+(?:auto\s+)?(?:effects?|zooms?|emphasis)\b",
        segment,
    ):
        return None
    params: dict[str, Any] = {}
    count = _extract_count(segment)
    if count is not None:
        params["max_effects"] = max(1, min(count, 32))
    if re.search(r"\b(?:subtle|gentle|light|low)\b", segment):
        params["density"] = "low"
        params["intensity"] = "subtle"
    elif re.search(r"\b(?:aggressive|high|strong|lots|many)\b", segment):
        params["density"] = "high"
        params["intensity"] = "high"
    else:
        params["density"] = "medium"
    if re.search(r"\b(?:only\s+zoom|zooms?\s+only|camera\s+only|no\s+style)\b", segment):
        params["include_style_effects"] = False
    return ToolStep("add_auto_effects", params, "add subtitle-aware auto effects"), 0.86, "auto effects command"


def _extract_blender_visual_text(segment: str) -> str:
    quoted = re.search(r'"([^"]{1,120})"|\'([^\']{1,120})\'', segment)
    if quoted:
        return (quoted.group(1) or quoted.group(2) or "").strip()
    saying_match = re.search(
        r"\b(?:saying|that\s+says|says)\s+(.+?)(?:\s+\b(?:at|from|for|near|on|using|with|when)\b|$)",
        segment,
    )
    if saying_match:
        return saying_match.group(1).strip(" ,.")
    label_match = re.search(
        r"\b(?:text|title|label)\s+(.+?)(?:\s+\b(?:at|from|for|near|on|using|with|when)\b|$)",
        segment,
    )
    if label_match:
        return label_match.group(1).strip(" ,.")
    if re.search(r"\bgpu\b", segment):
        return "GPU"
    if re.search(r"\bneural\b", segment):
        return "Neural Network"
    if re.search(r"\bchip\b", segment):
        return "GPU Chip"
    return "3D Visual"


def _extract_directed_hyperframes_idea(segment: str) -> str:
    quoted = re.search(r'"([^"]{3,360})"|\'([^\']{3,360})\'', segment)
    if quoted:
        return _clean_directed_hyperframes_idea(quoted.group(1) or quoted.group(2) or "")
    cleaned = segment
    cleaned = re.sub(
        rf"\b(?:from\s+)?{_TIME_TOKEN}\s*(?:-|to|through|thru|until)\s*{_TIME_TOKEN}\b",
        " ",
        cleaned,
    )
    cleaned = re.sub(rf"\b(?:at|from|starting\s+at)\s+{_TIME_TOKEN}\b", " ", cleaned)
    cleaned = re.sub(rf"\bfor\s+{_TIME_TOKEN}\b", " ", cleaned)
    cleaned = re.sub(
        r"\bwhen\s+i\s+say\s+.+?(?:\s*$|\s+(?:with|using|in)\b)",
        " ",
        cleaned,
    )
    cleaned = re.sub(
        r"\bwhen\s+(?:the\s+transcript\s+)?mentions?\s+.+?(?:\s*$|\s+(?:with|using|in)\b)",
        " ",
        cleaned,
    )
    cleaned = re.sub(
        r"\b(?:with|using|use|in)\s+hyperframes?\b|\bhyperframes?\s+(?:visuals?|animation|cutaway|renderer)\b|\bhyperframes?\b",
        " ",
        cleaned,
    )
    patterns = (
        r"\b(?:show|visuali[sz]e|depict|animate)\s+(.+)$",
        r"\b(?:visual|animation|cutaway)\s+(?:of|showing|where|that)\s+(.+)$",
        r"\bidea\s*:?\s+(.+)$",
        r"\bas\s+(.+)$",
        r"\b(?:about|around)\s+(.+)$",
    )
    for pattern in patterns:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            return _clean_directed_hyperframes_idea(match.group(1))
    return _clean_directed_hyperframes_idea(cleaned)


def _clean_directed_hyperframes_idea(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(value or "")).strip(" ,.;:()[]{}\"'")
    leading = re.compile(
        r"^(?:add|create|make|insert|put|use|a|an|the|custom|directed|generated|supporting|visuals?|animation|cutaway|to)\s+",
        flags=re.IGNORECASE,
    )
    previous = ""
    while cleaned and cleaned != previous:
        previous = cleaned
        cleaned = leading.sub("", cleaned).strip(" ,.;:")
    cleaned = re.sub(r"\b(?:using|with|in)\s*$", "", cleaned, flags=re.IGNORECASE)
    if len(cleaned) > 360:
        cleaned = cleaned[:359].rstrip(" ,.;:") + "..."
    return cleaned


def _extract_blender_asset_path(segment: str) -> str | None:
    match = re.search(r"(?P<path>(?:[./~\w-]+/)?[\w.-]+\.(?:glb|gltf|obj|blend))\b", segment)
    return match.group("path") if match else None


def _extract_trigger_text(segment: str) -> str | None:
    match = re.search(r"\bwhen\s+i\s+say\s+(.+?)(?:\s*$|\s+(?:from|at|for|with|using)\b)", segment)
    if match:
        return match.group(1).strip(" ,.")
    match = re.search(r"\bwhen\s+(?:the\s+transcript\s+)?mentions?\s+(.+?)(?:\s*$|\s+(?:from|at|for|with|using)\b)", segment)
    if match:
        return match.group(1).strip(" ,.")
    return None


def _extract_generation_prompt(segment: str) -> str:
    quoted = re.search(r'"([^"]{3,900})"|\'([^\']{3,900})\'', segment)
    if quoted:
        return (quoted.group(1) or quoted.group(2) or "").strip(" ,.")
    cleaned = segment
    cleaned = re.sub(
        r"\b(?:generate|create|make|build|produce)\s+(?:me\s+)?(?:a\s+)?(?:new\s+)?(?:hyperframes?\s+)?video\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\b(?:from\s+scratch|with\s+sound|with\s+audio|using\s+hyperframes?)\b", " ", cleaned)
    cleaned = re.sub(rf"\b(?:to|into|under|about|around)\s+{_TIME_TOKEN}\b", " ", cleaned)
    match = re.search(r"\b(?:about|on|for|explaining|that\s+explains)\s+(.+)$", cleaned)
    if match:
        cleaned = match.group(1)
    cleaned = re.sub(
        r"\b(?:portrait|vertical|landscape|horizontal|square|reels?|shorts?|tiktok|project\s+only|no\s+render|without\s+rendering|silent|no\s+audio|without\s+audio|voice\s+[a-z]{2}_[a-z0-9_-]+)\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", cleaned).strip(" ,.")


def _extract_visual_position(segment: str) -> str:
    if re.search(r"\btop\s+right\b", segment):
        return "top_right"
    if re.search(r"\btop\s+left\b", segment):
        return "top_left"
    if re.search(r"\bbottom\s+right\b", segment):
        return "bottom_right"
    if re.search(r"\bbottom\s+left\b", segment):
        return "bottom_left"
    if re.search(r"\bright\b", segment):
        return "center_right"
    if re.search(r"\bleft\b", segment):
        return "center_left"
    return "center"


def _compile_auto_broll(segment: str, *, state: Any | None = None) -> tuple[ToolStep, float, str] | None:
    if not re.search(r"\b(?:b[-\s]?roll|stock footage|stock video|cutaways?)\b", segment):
        return None
    params: dict[str, Any] = {}
    count = _extract_count(segment)
    if count is not None:
        requested_count = max(1, min(count, 24))
        params["max_overlays"] = requested_count
        params["requested_count"] = requested_count
        params["coverage_policy"] = "target_count"
    interval = _extract_interval_seconds(segment)
    if interval is not None:
        params["interval_sec"] = interval
        duration = float((getattr(state, "metadata", {}) or {}).get("duration_sec") or 0.0) if state is not None else 0.0
        if duration > 0:
            requested_count = max(1, min(int((duration + interval - 0.001) // interval), 24))
            params["max_overlays"] = requested_count
            params["requested_count"] = requested_count
        params["coverage_policy"] = "target_count"
    providers = [
        provider
        for provider in ("pexels", "pixabay", "coverr")
        if re.search(rf"\b{provider}\b", segment)
    ]
    if providers:
        params["providers"] = ",".join(providers)
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
        r"(?:visuals?|effects?|zooms?|overlays?|b[-\s]?roll|shorts?|reels?|tiktoks?|clips?)\b",
        segment,
    )
    if not match:
        return None
    return int(match.group(1))


def _extract_renderer(segment: str) -> str | None:
    if re.search(r"\b(?:both|hyperframes\s+(?:and|&|\+)\s+manim|manim\s+(?:and|&|\+)\s+hyperframes)\b", segment):
        return "both"
    if re.search(r"\b(?:remotion|react\s+video|react[-\s]?renderer)\b", segment):
        return "remotion"
    for renderer in ("hyperframes", "manim", "ffmpeg", "blender"):
        if re.search(rf"\b{renderer}\b", segment):
            return renderer
    return None


def _extract_density(segment: str) -> str | None:
    if re.search(r"\bchapter\s+coverage\b|\beach\s+chapter\b|\bcover\s+chapters\b", segment):
        return "chapter_coverage"
    if re.search(r"\b(?:dense|many|lots|frequent)\b", segment):
        return "dense"
    if re.search(r"\b(?:balanced|normal)\b", segment):
        return "balanced"
    if re.search(r"\b(?:sparse|light|few)\b", segment):
        return "sparse"
    return None


def _extract_interval_seconds(segment: str) -> float | None:
    match = re.search(rf"\bevery\s+({_TIME_TOKEN})\b", segment)
    if not match:
        return None
    seconds = _parse_time_seconds(match.group(1))
    if seconds <= 0:
        return None
    return max(1.0, min(seconds, 600.0))


def _extract_resolution(segment: str) -> str | None:
    match = re.search(r"\b(\d{3,5})\s*x\s*(\d{3,5})\b", segment)
    if not match:
        return None
    width = int(match.group(1))
    height = int(match.group(2))
    if width <= 0 or height <= 0:
        return None
    return f"{width}x{height}"


def _extract_visual_asset_path(segment: str) -> str | None:
    quoted = re.search(
        r'"([^"]+\.(?:html?|mp4|mov|m4v|webm|gif|png|jpe?g|webp|bmp))"|'
        r"'([^']+\.(?:html?|mp4|mov|m4v|webm|gif|png|jpe?g|webp|bmp))'",
        segment,
        flags=re.IGNORECASE,
    )
    if quoted:
        return (quoted.group(1) or quoted.group(2) or "").strip()
    match = re.search(
        r"(?P<path>(?:[./~\w-]+/)?[\w.-]+\.(?:html?|mp4|mov|m4v|webm|gif|png|jpe?g|webp|bmp))\b",
        segment,
        flags=re.IGNORECASE,
    )
    return match.group("path") if match else None


def _extract_audio_asset_path(segment: str) -> str | None:
    quoted = re.search(
        r'"([^"]+\.(?:aac|aiff|flac|m4a|mp3|ogg|opus|wav|wma))"|'
        r"'([^']+\.(?:aac|aiff|flac|m4a|mp3|ogg|opus|wav|wma))'",
        segment,
        flags=re.IGNORECASE,
    )
    if quoted:
        return (quoted.group(1) or quoted.group(2) or "").strip()
    match = re.search(
        r"(?P<path>(?:[./~\w-]+/)?[\w.-]+\.(?:aac|aiff|flac|m4a|mp3|ogg|opus|wav|wma))\b",
        segment,
        flags=re.IGNORECASE,
    )
    return match.group("path") if match else None


def _extract_song_mode(segment: str) -> str | None:
    if re.search(r"\b(?:replace|swap)\s+(?:the\s+)?(?:audio|soundtrack|music)\b", segment):
        return "replace"
    if re.search(r"\bintro\s*(?:and|&|\+)\s*outro\b|\bbookends?\b", segment):
        return "intro_outro"
    if re.search(r"\b(?:intro|opening|start)\s+(?:song|music|soundtrack|sting|cue)\b", segment):
        return "intro"
    if re.search(r"\b(?:outro|ending|closing|end)\s+(?:song|music|soundtrack|sting|cue)\b", segment):
        return "outro"
    if re.search(r"\b(?:highlight|montage|hype|energetic)\b", segment):
        return "highlight"
    if re.search(r"\b(?:background|under(?:neath)?|bed)\b", segment):
        return "background"
    return None


def _extract_song_volume(segment: str) -> float | None:
    match = re.search(
        rf"\b(?:volume|music\s+volume|song\s+volume)\s+(?:to\s+|at\s+)?({_NUMBER})(?:\s*%)?\b|"
        rf"\b(?:to|at)\s+({_NUMBER})(?:\s*%)?\s+(?:volume|music\s+volume|song\s+volume)\b",
        segment,
    )
    if not match:
        if re.search(r"\b(?:quiet|subtle|low)\b", segment):
            return 0.14
        if re.search(r"\b(?:loud|strong|prominent)\b", segment):
            return 0.42
        return None
    value = float(match.group(1) or match.group(2))
    if "%" in match.group(0):
        value /= 100.0
    elif value > 1.5:
        value /= 100.0
    return max(0.0, min(value, 1.5))


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
    match = re.search(rf"\b(?:to|into|under|about|around|in)\s+({_TIME_TOKEN})\b", segment)
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
