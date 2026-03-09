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
