from __future__ import annotations

from typing import Any

SYSTEM_PROMPT_TEMPLATE = """You are Vex, a precise and efficient video editing assistant. You are concise, terminal-friendly, and occasionally witty without being verbose.

Rules:
1. If video metadata is missing, call get_video_info before making editing decisions.
2. Break complex requests into multiple sequential tool calls when needed.
3. After tools finish, reflect on whether the request fully succeeded and whether a follow-up is useful.
4. Suggestions must be formatted exactly as: [SUGGESTION]: <text> - reply 'yes' to apply or continue.
5. Originals are safe. Never modify original source files; use the working copy only.
6. Reference prior timeline operations by name when relevant.
7. If the request is ambiguous, ask exactly one clarifying question before acting.
8. Keep responses plain text, concise, and REPL-friendly.
9. When the user replies 'yes' after a [SUGGESTION], apply it immediately.
10. When the user asks for reels, TikToks, YouTube Shorts, viral clips, or auto-cut social highlights, prefer create_auto_shorts over summarize_clip.
10a. When the user asks to add stock footage, cutaways, supporting visuals, or B-roll, prefer add_auto_broll if Pexels-driven footage fits the request.
10b. When the user asks for custom-generated animations, precise explanatory visuals, or visuals that should be created on the spot, prefer add_auto_visuals. Let it choose the best supported renderer unless the user explicitly asks for one.
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
                "output_path": {"type": "string"},
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
                    "description": "Font size. Default 24.",
                },
                "font_color": {
                    "type": "string",
                    "description": "Text color (white, yellow, black, etc). Default white.",
                },
                "outline_color": {
                    "type": "string",
                    "description": "Outline color. Default black.",
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
        "description": "Create multiple short-form vertical clips from the current working video. Auto-transcribes if needed, ranks the most interesting moments with the active reasoning model, packages each short with captions and metadata, and writes a manifest bundle to the project's output directory.",
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
            },
            "required": [],
        },
    },
    {
        "name": "add_auto_broll",
        "description": "Plan subtitle-aligned, transcript-aware B-roll beats, fetch matching stock clips from Pexels, semantically rerank the results, and splice them into the current working video while preserving the original audio.",
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
            },
            "required": [],
        },
    },
    {
        "name": "add_auto_visuals",
        "description": "Plan transcript-aligned generated visuals, choose the best supported free animation backend per visual, and composite the results into the working video for precise custom explanatory cutaways.",
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
                    "enum": ["auto", "manim", "ffmpeg", "blender"],
                    "description": "Preferred animation backend. Default auto, which lets Vex choose per visual.",
                },
                "style_pack": {
                    "type": "string",
                    "enum": ["auto", "editorial_clean", "bold_tech", "documentary_kinetic", "product_ui", "cinematic_night", "signal_lab", "magazine_luxe"],
                    "description": "Preferred visual art direction. Default auto.",
                },
                "max_visuals": {
                    "type": "integer",
                    "description": "Maximum number of generated visuals to add. Default 3.",
                },
                "min_visual_sec": {
                    "type": "number",
                    "description": "Minimum duration for each generated visual in seconds. Default 1.4.",
                },
                "max_visual_sec": {
                    "type": "number",
                    "description": "Maximum duration for each generated visual in seconds. Default 3.6.",
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
                "output_path": {"type": "string"},
                "custom_settings": {"type": "object"},
            },
            "required": ["preset_name"],
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
