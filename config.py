from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv
from google.genai import types

PROVIDER = "gemini"
GEMINI_API_KEY = None
GEMINI_MODEL = "gemma-4-31b-it"
ANTHROPIC_API_KEY = None
CLAUDE_MODEL = "claude-sonnet-4-5"
PEXELS_API_KEY = None
AGENT_PROJECTS_DIR = os.path.expanduser("~/.video-agent/projects/")
FFMPEG_PATH = "ffmpeg"
BLENDER_PATH = "blender"
WHISPER_MODEL = "base"
VERSION = "1.0.0"
GENAI_TIMEOUT_SEC = 90
ANTHROPIC_TIMEOUT_SEC = 90.0
MANIM_PREVIEW_TIMEOUT_SEC = 75
MANIM_FINAL_TIMEOUT_SEC = 240
LLM_REQUEST_MAX_RETRIES = 3
LLM_RETRY_BASE_DELAY_SEC = 1.5


def gemini_supports_thinking_config(model_name: str | None = None) -> bool:
    normalized = (model_name or GEMINI_MODEL or "").strip().lower()
    return normalized.startswith("gemini")


def build_gemini_generation_config(
    system_prompt: str,
    *,
    model_name: str | None = None,
    tools: list[types.Tool] | None = None,
) -> types.GenerateContentConfig:
    kwargs: dict[str, object] = {
        "system_instruction": system_prompt,
    }
    if tools:
        kwargs["tools"] = tools
    if gemini_supports_thinking_config(model_name):
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    return types.GenerateContentConfig(**kwargs)


def google_genai_http_options() -> types.HttpOptions:
    return types.HttpOptions(timeout=GENAI_TIMEOUT_SEC * 1000)


def configure_runtime_logging() -> None:
    noisy_loggers = (
        "google",
        "google.genai",
        "google.genai.models",
        "google.genai._api_client",
        "google_genai",
        "google_genai.models",
        "google_genai._api_client",
        "httpx",
        "httpcore",
        "urllib3",
    )
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def _print_and_exit(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def _ffmpeg_install_instructions() -> str:
    return (
        "FFmpeg was not found in PATH.\n"
        "Install instructions:\n"
        "  macOS:   brew install ffmpeg\n"
        "  Ubuntu:  sudo apt install ffmpeg\n"
        "  Windows: install from https://ffmpeg.org/download.html and add ffmpeg/bin to PATH"
    )


def reload_settings() -> None:
    load_dotenv()

    global PROVIDER
    global GEMINI_API_KEY
    global GEMINI_MODEL
    global ANTHROPIC_API_KEY
    global CLAUDE_MODEL
    global PEXELS_API_KEY
    global AGENT_PROJECTS_DIR
    global FFMPEG_PATH
    global BLENDER_PATH
    global WHISPER_MODEL
    global GENAI_TIMEOUT_SEC
    global ANTHROPIC_TIMEOUT_SEC
    global MANIM_PREVIEW_TIMEOUT_SEC
    global MANIM_FINAL_TIMEOUT_SEC
    global LLM_REQUEST_MAX_RETRIES
    global LLM_RETRY_BASE_DELAY_SEC

    PROVIDER = os.getenv("PROVIDER", "gemini").strip().lower()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemma-4-31b-it")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5")
    PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
    AGENT_PROJECTS_DIR = os.path.expanduser(
        os.getenv("AGENT_PROJECTS_DIR", "~/.video-agent/projects/")
    )
    FFMPEG_PATH = os.getenv("FFMPEG_PATH", "ffmpeg")
    BLENDER_PATH = os.getenv("BLENDER_PATH", "blender")
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
    GENAI_TIMEOUT_SEC = max(15, int(os.getenv("GENAI_TIMEOUT_SEC", "90")))
    ANTHROPIC_TIMEOUT_SEC = max(15.0, float(os.getenv("ANTHROPIC_TIMEOUT_SEC", "90")))
    MANIM_PREVIEW_TIMEOUT_SEC = max(30, int(os.getenv("MANIM_PREVIEW_TIMEOUT_SEC", "75")))
    MANIM_FINAL_TIMEOUT_SEC = max(MANIM_PREVIEW_TIMEOUT_SEC, int(os.getenv("MANIM_FINAL_TIMEOUT_SEC", "240")))
    LLM_REQUEST_MAX_RETRIES = max(1, int(os.getenv("LLM_REQUEST_MAX_RETRIES", "3")))
    LLM_RETRY_BASE_DELAY_SEC = max(0.5, float(os.getenv("LLM_RETRY_BASE_DELAY_SEC", "1.5")))


def validate_config() -> None:
    reload_settings()

    if PROVIDER not in {"gemini", "claude"}:
        _print_and_exit(
            f"Invalid PROVIDER={PROVIDER!r}. Valid options are: 'gemini', 'claude'."
        )

    if PROVIDER == "gemini" and not GEMINI_API_KEY:
        _print_and_exit(
            "GEMINI_API_KEY is required when PROVIDER=gemini. "
            "Set it in your environment or .env file."
        )

    if PROVIDER == "claude" and not ANTHROPIC_API_KEY:
        _print_and_exit(
            "ANTHROPIC_API_KEY is required when PROVIDER=claude. "
            "Set it in your environment or .env file."
        )

    if shutil.which(FFMPEG_PATH) is None:
        _print_and_exit(_ffmpeg_install_instructions())

    Path(AGENT_PROJECTS_DIR).mkdir(parents=True, exist_ok=True)
