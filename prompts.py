from __future__ import annotations

from typing import Any

SYSTEM_PROMPT_TEMPLATE = """You are Vex, a precise and efficient video editing assistant. You are concise, terminal-friendly, and occasionally witty without being verbose.

Rules:
1. If video metadata is missing and the request needs metadata, call get_video_info before making editing decisions. Simple explicit timestamp edits do not need metadata first.
2. Break complex requests into multiple sequential tool calls when needed.
3. For a simple one-tool edit, call exactly that tool and stop. After tools finish, answer directly; do not add a second planning step unless another tool is clearly required.
4. Suggestions must be formatted exactly as: [SUGGESTION]: <text> - reply 'yes' to apply or continue.
5. Originals are safe. Never modify original source files; use the working copy only.
6. Reference prior timeline operations by name when relevant.
7. If the request is ambiguous, ask exactly one clarifying question before acting.
8. Keep responses plain text, concise, and REPL-friendly.
9. When the user replies 'yes' after a [SUGGESTION], apply it immediately.
10. When the user asks for reels, TikToks, YouTube Shorts, viral clips, or auto-cut social highlights, prefer create_auto_shorts over summarize_clip.
10a. When the user asks to add stock footage, cutaways, supporting visuals, or B-roll, prefer add_auto_broll if stock-provider footage fits the request. Use providers=pexels, pixabay, coverr, or a comma-separated subset only when the user names a provider; otherwise leave providers as auto.
10b. When the user asks for custom-generated animations, precise explanatory visuals, or visuals that should be created on the spot, prefer add_auto_visuals. If the user explicitly describes their own Hyperframes visual idea, call add_auto_visuals with renderer=hyperframes and directed_visual_specs containing the visual_idea plus optional start/end or trigger_text; the idea is art direction only, while transcript evidence remains the source of truth. If the user explicitly asks for Hyperframes, use renderer=hyperframes and do not mix in Manim. If the user explicitly asks for Manim, use renderer=manim and do not mix in Hyperframes. Use renderer=both only when the user asks for both.
10c. When the user asks to encode, transcode, convert formats, compress file size, target a file size, or generate an FFmpeg command, call plan_encode first. Never write or execute a raw FFmpeg shell command yourself. If an encode plan is pending and the user replies yes, call run_pending_encode.
10d. When the user asks to auto color grade, color correct, fix colors, white balance, make colors pop, warm/cool the image, or apply a cinematic look, prefer auto_color_grade.
10e. When the user asks for auto zooms, punch-ins, camera movement, subtitle-aware emphasis, or automatic effects tied to captions/subtitles, prefer add_auto_effects.
10f. When the user asks to generate a brand-new video from a prompt, script, topic, or narration without editing an existing source video, call generate_video. This is an audio-first native Hyperframes motion generator: pass prompt or script, optional title/duration/aspect/fps/quality/render_resolution/voice/style/music, and let the tool produce the synced video project and render. For public showcase/proof videos, prefer quality=high, fps=60, and render_resolution=4k when runtime cost is acceptable.
11. If any tool fails, do not guess the cause from prior conversation. Use the exact tool error message from the latest tool result, and say when you are unsure.
11a. If a tool fails during a chained workflow, stop and report the failure instead of continuing into downstream dependent tools unless the user explicitly asked to continue with partial results.

--- CURRENT PROJECT STATE ---
Project: {project_name}
Provider: {provider} / {model}
Working file: {working_file}
Duration: {duration}s | {width}x{height} | {fps}fps
Timeline ops applied: {timeline_count}
Last operation: {last_operation}
---
"""

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "get_video_info",
        "description": "Inspect the current working video and return metadata.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "trim_clip",
        "description": "Trim the working video to a specific time range.",
        "parameters": {
            "type": "object",
            "properties": {
                "start": {
                    "type": "string",
                    "description": "Start timestamp like '0:30', '30', or '30s'.",
                },
                "end": {
                    "type": "string",
                    "description": "Optional end timestamp like '1:45'.",
                },
            },
            "required": ["start"],
        },
    },
    {
        "name": "merge_clips",
        "description": "Merge the current working clip with one or more external video clips.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Additional file paths to concatenate after the current working clip.",
                }
            },
            "required": ["file_paths"],
        },
    },
    {
        "name": "adjust_speed",
        "description": "Adjust playback speed for the whole clip or for a specific segment.",
        "parameters": {
            "type": "object",
            "properties": {
                "factor": {"type": "number", "description": "Speed factor between 0.25 and 4.0."},
                "start": {"type": "string", "description": "Optional segment start."},
                "end": {"type": "string", "description": "Optional segment end."},
            },
            "required": ["factor"],
        },
    },
    {
        "name": "add_transition",
        "description": "Add a fade-style transition. For a single clip, 'crossfade' at position='between' behaves as a fade-through-black transition.",
        "parameters": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["fade_in", "fade_out", "crossfade"],
                },
                "duration": {"type": "number", "description": "Transition duration in seconds."},
                "position": {
                    "type": "string",
                    "enum": ["start", "end", "between"],
                },
            },
            "required": ["type", "duration", "position"],
        },
    },
    {
        "name": "add_text_overlay",
        "description": "Overlay text on the working video.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "position": {
                    "type": "string",
                    "enum": [
                        "top",
                        "center",
                        "bottom",
                        "top_left",
                        "top_right",
                        "bottom_left",
                        "bottom_right",
                    ],
                },
                "start": {"type": "string"},
                "end": {"type": "string"},
                "font_size": {"type": "integer", "default": 48},
                "color": {"type": "string", "default": "white"},
                "background_opacity": {"type": "number", "default": 0.0},
            },
            "required": ["text", "position", "start", "end"],
        },
    },
    {
        "name": "extract_audio",
        "description": "Extract audio from the current working video.",
        "parameters": {
            "type": "object",
            "properties": {
                "format": {"type": "string", "enum": ["mp3", "wav", "aac"], "default": "mp3"},
            },
            "required": [],
        },
    },
    {
        "name": "replace_audio",
        "description": "Replace or mix audio on the current working video.",
        "parameters": {
            "type": "object",
            "properties": {
                "audio_path": {"type": "string"},
                "mix_with_original": {"type": "boolean", "default": False},
                "mix_ratio": {"type": "number", "default": 0.5},
            },
            "required": ["audio_path"],
        },
    },
    {
        "name": "mute_segment",
        "description": "Mute a section of audio in the current working video.",
        "parameters": {
            "type": "object",
            "properties": {
                "start": {"type": "string"},
                "end": {"type": "string"},
            },
            "required": ["start", "end"],
        },
    },
    {
        "name": "trim_silence",
        "description": "Remove dead-air pauses from the video while preserving natural speech cadence. Useful for cleaning up raw footage, podcasts, and screen recordings.",
        "parameters": {
            "type": "object",
            "properties": {
                "aggressiveness": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "Controls the default silence duration and threshold. Default medium.",
                },
                "min_silence_duration": {
                    "type": "number",
                    "description": "Minimum silence duration in seconds to remove. Default 0.5.",
                },
                "silence_threshold_db": {
                    "type": "number",
                    "description": "Volume threshold in dB below which audio is considered silent. Default -35.0.",
                },
                "speech_padding_ms": {
                    "type": "number",
                    "description": "Speech padding to preserve around cuts in milliseconds. Default 120.",
                },
                "merge_gap_ms": {
                    "type": "number",
                    "description": "Merge nearby silence cuts separated by less than this gap in milliseconds. Default 180.",
                },
                "min_keep_duration_ms": {
                    "type": "number",
                    "description": "Minimum speech segment length to preserve between cuts in milliseconds. Default 280.",
                },
                "trim_edges": {
                    "type": "boolean",
                    "description": "Whether to also trim silent pauses at the very start or end. Default false.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "auto_color_grade",
        "description": "Analyze sampled frames from the current working video and apply a professional automatic color grade using deterministic FFmpeg filters. Records a local color quality contract, creative QA report, and creative-run registry entry. Use for color correction, white balance, exposure/contrast/saturation cleanup, and looks such as natural, vibrant, cinematic, warm, cool, documentary, or punchy.",
        "parameters": {
            "type": "object",
            "properties": {
                "look": {
                    "type": "string",
                    "enum": ["auto", "natural", "vibrant", "cinematic", "warm", "cool", "documentary", "punchy"],
                    "description": "Desired grade look. Default auto resolves to a natural correction.",
                },
                "intensity": {
                    "type": "number",
                    "description": "Grade strength from 0.0 to 1.5. Default 1.0.",
                },
                "sample_count": {
                    "type": "integer",
                    "description": "Number of frames to sample for analysis, clamped from 1 to 15. Default 9.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "burn_subtitles",
        "description": "Burn subtitles from an SRT file directly onto the video. Automatically uses the transcript generated by transcribe_video if no SRT path is provided.",
        "parameters": {
            "type": "object",
            "properties": {
                "srt_path": {
                    "type": "string",
                    "description": "Optional path to SRT file. Defaults to the transcript generated in the current project.",
                },
                "font_size": {
                    "type": "integer",
                    "description": "Optional absolute font size. If omitted, the selected subtitle style chooses a responsive size.",
                },
                "font_color": {
                    "type": "string",
                    "description": "Optional text color override such as white, yellow, or #F8FAFC.",
                },
                "outline_color": {
                    "type": "string",
                    "description": "Optional outline color override.",
                },
                "style": {
                    "type": "string",
                    "enum": ["clean_pop", "creator_bold", "cinematic", "glass", "karaoke_focus", "minimal"],
                    "description": "Production subtitle style preset. Default clean_pop. Use creator_bold for shorts/Reels, cinematic for subtle film-like captions, glass for premium explainer captions, karaoke_focus for high-energy highlighted captions, minimal for simple subtitles.",
                },
                "emphasis_color": {
                    "type": "string",
                    "description": "Optional accent/emphasis color override, e.g. #FACC15.",
                },
                "background_opacity": {
                    "type": "number",
                    "description": "Optional backplate opacity from 0 to 1 for styles with a caption box.",
                },
                "max_words_per_caption": {
                    "type": "integer",
                    "description": "Optional maximum words per displayed caption chunk. Lower values feel punchier.",
                },
                "max_lines": {
                    "type": "integer",
                    "description": "Optional maximum caption lines, usually 1 or 2.",
                },
                "case": {
                    "type": "string",
                    "enum": ["normal", "uppercase", "title"],
                    "description": "Optional text casing override.",
                },
                "position": {
                    "type": "string",
                    "enum": ["bottom", "center", "top"],
                    "description": "Subtitle position. Default bottom.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "summarize_clip",
        "description": "Automatically trim a long video down to the best moments fitting a target duration. Uses AI to analyze the transcript and select the most valuable segments. Will auto-transcribe first if needed.",
        "parameters": {
            "type": "object",
            "properties": {
                "target_duration_sec": {
                    "type": "number",
                    "description": "Target output duration in seconds. Default 60.",
                }
            },
            "required": [],
        },
    },
    {
        "name": "create_auto_shorts",
        "description": "Create multiple short-form vertical clips from the current working video. Auto-transcribes if needed, builds a local Video Understanding Graph, scores transcript windows for retention, visual opportunity, hook strength, payoff, novelty, clarity, shareability, thesis alignment, standalone clarity, story completeness, pacing, and topic diversity, uses the active reasoning model for final selection, packages each short with captions, QA reports, metadata, and a manifest bundle, and records the run in the local creative registry.",
        "parameters": {
            "type": "object",
            "properties": {
                "count": {
                    "type": "integer",
                    "description": "How many shorts to generate. Default 3.",
                },
                "min_duration_sec": {
                    "type": "number",
                    "description": "Minimum duration per short. Default 20.",
                },
                "max_duration_sec": {
                    "type": "number",
                    "description": "Maximum duration per short. Default 45.",
                },
                "target_platform": {
                    "type": "string",
                    "enum": ["youtube_shorts", "tiktok", "instagram_reels"],
                    "description": "Platform profile used for packaging and metadata. Default youtube_shorts.",
                },
                "include_compilation": {
                    "type": "boolean",
                    "description": "Whether to also render a merged compilation of the generated shorts. Default true.",
                },
                "subtitle_style": {
                    "type": "string",
                    "enum": ["clean_pop", "creator_bold", "cinematic", "glass", "karaoke_focus", "minimal"],
                    "description": "Subtitle style preset for rendered shorts. Defaults to the platform profile, usually creator_bold.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "generate_video",
        "description": "Generate a brand-new native Hyperframes video from a prompt or script without requiring a source video. The tool builds an audio-first script, TTS narration, timing/beat graph, native per-beat Hyperframes compositions, motion cues, transitions, captions, optional music, QA report, manifest, and final synced render.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Plain-English video idea, topic, or creative brief.",
                },
                "script": {
                    "type": "string",
                    "description": "Optional exact narration script. When present, preserve it as the spoken audio source.",
                },
                "title": {
                    "type": "string",
                    "description": "Optional video title.",
                },
                "duration_sec": {
                    "type": "number",
                    "description": "Target duration in seconds. Default 24, clamped from 6 to 180.",
                },
                "aspect": {
                    "type": "string",
                    "enum": ["landscape", "portrait", "square"],
                    "description": "Output frame shape. Use portrait for Shorts/Reels/TikTok. Default landscape.",
                },
                "fps": {
                    "type": "number",
                    "description": "Render frame rate. Use 60 for showcase motion when runtime cost is acceptable; default 30.",
                },
                "quality": {
                    "type": "string",
                    "enum": ["draft", "standard", "high"],
                    "description": "Hyperframes render quality. Use high for final proof/showcase videos; default standard.",
                },
                "render_resolution": {
                    "type": "string",
                    "enum": ["landscape", "portrait", "landscape-4k", "portrait-4k", "4k"],
                    "description": "Optional Hyperframes render preset. Use 4k, landscape-4k, or portrait-4k for showcase exports.",
                },
                "voice": {
                    "type": "string",
                    "description": "Hyperframes TTS voice ID such as af_heart, af_nova, am_adam, bf_emma. Default af_heart.",
                },
                "voice_speed": {
                    "type": "number",
                    "description": "Speech speed multiplier, clamped from 0.65 to 1.35. Default 1.0.",
                },
                "style": {
                    "type": "string",
                    "description": "Optional art direction such as clean_kinetic, premium_explainer, signal_lab, or viral_short.",
                },
                "background_music_path": {
                    "type": "string",
                    "description": "Optional local music file to mix under the narration.",
                },
                "music_volume": {
                    "type": "number",
                    "description": "Background music volume from 0 to 1. Default 0.12.",
                },
                "render": {
                    "type": "boolean",
                    "description": "Whether to render the final video. Default true. Use false for project-only generation.",
                },
                "generate_audio": {
                    "type": "boolean",
                    "description": "Whether to generate narration audio with Hyperframes TTS. Default true.",
                },
                "strict_audio_timing": {
                    "type": "boolean",
                    "description": "Whether transcription failure should fail the run instead of falling back to estimated audio timing. Default false.",
                },
                "workers": {
                    "type": "string",
                    "description": "Optional Hyperframes render worker count, such as 1, 2, or auto.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "add_auto_broll",
        "description": "Use B-roll Director v2 to plan subtitle-aligned, graph-aware stock cutaways, fetch matching clips from configured providers such as Pexels, Pixabay, and Coverr, verify candidate visual fit, run a final abruptness/semantic QA gate, and splice approved clips into the current working video while preserving the original audio.",
        "parameters": {
            "type": "object",
            "properties": {
                "max_overlays": {
                    "type": "integer",
                    "description": "Maximum number of stock inserts to add. Default 5.",
                },
                "min_overlay_sec": {
                    "type": "number",
                    "description": "Minimum duration for each B-roll insert in seconds. Default 1.2.",
                },
                "max_overlay_sec": {
                    "type": "number",
                    "description": "Maximum duration for each B-roll insert in seconds. Default 2.8.",
                },
                "providers": {
                    "type": "string",
                    "description": "Optional stock provider selection: auto, pexels, pixabay, coverr, or comma-separated names such as pexels,pixabay. Default auto.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "add_auto_visuals",
        "description": "Build a graph-backed video-level visual narrative program, plan transcript-aligned generated visuals, choose the best supported animation backend per visual, score the visual plan with local creative QA, composite context-aware explanatory cutaways into the working video, and record the run in the local creative registry.",
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["generated_only", "hybrid", "stock_only"],
                    "description": "How Vex should handle supporting visuals. Default generated_only.",
                },
                "renderer": {
                    "type": "string",
                    "enum": ["auto", "hyperframes", "manim", "both", "ffmpeg", "blender"],
                    "description": "Preferred animation backend. hyperframes and manim are strict single-renderer modes. both lets Vex choose between Hyperframes and Manim per visual. auto may use any available backend.",
                },
                "style_pack": {
                    "type": "string",
                    "enum": ["auto", "editorial_clean", "bold_tech", "documentary_kinetic", "product_ui", "cinematic_night", "signal_lab", "magazine_luxe"],
                    "description": "Preferred visual art direction. Default auto.",
                },
                "max_visuals": {
                    "type": "integer",
                    "description": "Maximum number of generated visuals to add. Default 5, capped at 12.",
                },
                "min_visual_sec": {
                    "type": "number",
                    "description": "Minimum duration for each generated visual in seconds. Default 1.4.",
                },
                "max_visual_sec": {
                    "type": "number",
                    "description": "Maximum duration for each generated visual in seconds. Default 3.6.",
                },
                "force_fullscreen": {
                    "type": "boolean",
                    "description": "Force generated visuals to replace the full frame instead of corner picture-in-picture. Default true for generated, hyperframes, and manim visuals.",
                },
                "visual_idea": {
                    "type": "string",
                    "description": "Optional one-off user art direction for a directed Hyperframes visual, such as 'particles compress into four memory blocks'. When set, use renderer=hyperframes and ground the actual labels/claims in the transcript.",
                },
                "start": {
                    "type": "string",
                    "description": "Optional start timestamp for a directed visual_idea, such as '12', '00:12', or '1:03'.",
                },
                "end": {
                    "type": "string",
                    "description": "Optional end timestamp for a directed visual_idea, such as '16', '00:16', or '1:08'.",
                },
                "trigger_text": {
                    "type": "string",
                    "description": "Optional transcript phrase used to time a directed visual_idea when start/end are omitted.",
                },
                "directed_visual_specs": {
                    "type": "array",
                    "description": "Optional list of explicit Hyperframes visual ideas. Each item may include visual_idea, start/end or trigger_text, and composition_mode. Treat visual_idea as art direction only; factual labels and relationships must come from transcript evidence.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "visual_idea": {
                                "type": "string",
                                "description": "User-described visual metaphor, medium, motion, or art direction.",
                            },
                            "start": {
                                "type": "string",
                                "description": "Optional start timestamp.",
                            },
                            "end": {
                                "type": "string",
                                "description": "Optional end timestamp.",
                            },
                            "trigger_text": {
                                "type": "string",
                                "description": "Optional transcript phrase used to choose timing when start/end are omitted.",
                            },
                            "composition_mode": {
                                "type": "string",
                                "enum": ["replace", "overlay", "picture_in_picture"],
                                "description": "How to composite the directed visual. Default replace.",
                            },
                        },
                    },
                },
            },
            "required": [],
        },
    },
    {
        "name": "add_auto_effects",
        "description": "Plan subtitle-aware emphasis effects from caption timing and transcript signals, then render camera motion and style accents such as punch-ins, punch-outs, slow pushes, pans, impact pulses, freeze accents, subtle shake, vignette, flash, focus, and subtitle highlight effects in one deterministic FFmpeg pass.",
        "parameters": {
            "type": "object",
            "properties": {
                "density": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "How frequently to apply effects. Default medium.",
                },
                "intensity": {
                    "type": "string",
                    "enum": ["subtle", "medium", "high", "strong"],
                    "description": "How strong camera movement and accents should feel. Default medium.",
                },
                "max_effects": {
                    "type": "integer",
                    "description": "Maximum effects to add, clamped from 1 to 32. Default 12.",
                },
                "include_style_effects": {
                    "type": "boolean",
                    "description": "Whether to include v2-style accents such as vignette, flash, focus, and subtitle highlight. Default true.",
                },
                "subtitle_position": {
                    "type": "string",
                    "enum": ["bottom", "center", "top"],
                    "description": "Where subtitles are expected to sit, used for highlight safe zones. Default bottom or latest burn_subtitles position.",
                },
                "refresh_existing": {
                    "type": "boolean",
                    "description": "Whether to remove a prior auto-effects pass before replanning. Default true.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "export_video",
        "description": "Export the current working video using a named preset.",
        "parameters": {
            "type": "object",
            "properties": {
                "preset_name": {"type": "string"},
                "custom_settings": {"type": "object"},
            },
            "required": ["preset_name"],
        },
    },
    {
        "name": "plan_encode",
        "description": "Plan a safe, metadata-aware FFmpeg encode/transcode/conversion command from plain English. This stores a pending plan and does not execute it.",
        "parameters": {
            "type": "object",
            "properties": {
                "raw_request": {
                    "type": "string",
                    "description": "The user's plain-English encode/compression/conversion request.",
                },
                "target_format": {
                    "type": "string",
                    "enum": ["mp4", "mov", "mkv", "webm", "m4v"],
                    "description": "Desired output container. Default mp4.",
                },
                "video_codec": {
                    "type": "string",
                    "enum": ["h264", "hevc", "av1", "vp9", "prores"],
                    "description": "Desired video codec family. Default h264 for compatibility.",
                },
                "audio_codec": {
                    "type": "string",
                    "enum": ["aac", "mp3", "opus", "copy", "none"],
                    "description": "Desired audio codec. Use none to strip audio.",
                },
                "quality": {
                    "type": "string",
                    "enum": ["max", "high", "balanced", "small"],
                    "description": "Quality-size preference. Default balanced.",
                },
                "optimize_for": {
                    "type": "string",
                    "enum": ["compatibility_quality", "smallest_size", "fastest"],
                    "description": "Primary optimization target. Default compatibility_quality.",
                },
                "target_size_mb": {
                    "type": "number",
                    "description": "Approximate target output size in megabytes. Enables two-pass encoding.",
                },
                "max_width": {"type": "integer", "description": "Optional maximum output width."},
                "max_height": {"type": "integer", "description": "Optional maximum output height."},
                "fps": {"type": "number", "description": "Optional output frame rate cap."},
                "strip_audio": {"type": "boolean", "description": "Whether to remove audio."},
                "copy_streams": {
                    "type": "boolean",
                    "description": "Whether to remux/copy streams instead of re-encoding when possible.",
                },
            },
            "required": ["raw_request"],
        },
    },
    {
        "name": "run_pending_encode",
        "description": "Execute the latest pending encode plan after the user confirms it.",
        "parameters": {
            "type": "object",
            "properties": {
                "plan_id": {
                    "type": "string",
                    "description": "Optional pending encode plan id to confirm.",
                }
            },
            "required": [],
        },
    },
    {
        "name": "undo",
        "description": "Undo the most recent timeline operation.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "redo",
        "description": "Redo the most recently undone timeline operation.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "transcribe_video",
        "description": "Generate a transcript for the current working video using Whisper.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
]


def build_system_prompt(state: Any) -> str:
    metadata = state.metadata or {}
    return SYSTEM_PROMPT_TEMPLATE.format(
        project_name=state.project_name,
        provider=state.provider,
        model=state.model,
        working_file=state.working_file,
        duration=metadata.get("duration_sec", "unknown"),
        width=metadata.get("width", "unknown"),
        height=metadata.get("height", "unknown"),
        fps=metadata.get("fps", "unknown"),
        timeline_count=len(state.timeline),
        last_operation=state.timeline[-1]["description"] if state.timeline else "none",
    )
