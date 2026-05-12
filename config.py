from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from dotenv import load_dotenv

if TYPE_CHECKING:
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
HYPERFRAMES_NPX_PATH = "npx"
HYPERFRAMES_CLI_PACKAGE = "hyperframes"
HYPERFRAMES_LINT_TIMEOUT_SEC = 90
HYPERFRAMES_RENDER_TIMEOUT_SEC = 300
HYPERFRAMES_RENDER_QUALITY = ""
HYPERFRAMES_VARIANT_COUNT = 3
HYPERFRAMES_QA_MODE = "hybrid"
HYPERFRAMES_ENABLE_VISION_QA = True
HYPERFRAMES_MIN_QUALITY_SCORE = 0.78
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
    tools: list["types.Tool"] | None = None,
) -> "types.GenerateContentConfig":
    from google.genai import types

    automatic_function_calling = (
        types.AutomaticFunctionCallingConfig(disable=True)
        if tools
        else None
    )
    thinking_config = None
    if gemini_supports_thinking_config(model_name):
        thinking_config = types.ThinkingConfig(thinking_budget=0)
    return types.GenerateContentConfig(
        system_instruction=system_prompt,
        tools=cast(Any, tools or None),
        automatic_function_calling=automatic_function_calling,
        thinking_config=thinking_config,
    )


def google_genai_http_options() -> "types.HttpOptions":
    from google.genai import types

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


def _env_int(name: str, default: int, *, minimum: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        _print_and_exit(f"Invalid {name}={raw!r}. Expected an integer value.")
    return max(minimum, value)


def _env_float(name: str, default: float, *, minimum: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError:
        _print_and_exit(f"Invalid {name}={raw!r}. Expected a numeric value.")
    return max(minimum, value)


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
    global HYPERFRAMES_NPX_PATH
    global HYPERFRAMES_CLI_PACKAGE
    global HYPERFRAMES_LINT_TIMEOUT_SEC
    global HYPERFRAMES_RENDER_TIMEOUT_SEC
    global HYPERFRAMES_RENDER_QUALITY
    global HYPERFRAMES_VARIANT_COUNT
    global HYPERFRAMES_QA_MODE
    global HYPERFRAMES_ENABLE_VISION_QA
    global HYPERFRAMES_MIN_QUALITY_SCORE
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
    HYPERFRAMES_NPX_PATH = os.getenv("HYPERFRAMES_NPX_PATH", "npx")
    HYPERFRAMES_CLI_PACKAGE = os.getenv("HYPERFRAMES_CLI_PACKAGE", "hyperframes").strip() or "hyperframes"
    HYPERFRAMES_LINT_TIMEOUT_SEC = _env_int("HYPERFRAMES_LINT_TIMEOUT_SEC", 90, minimum=15)
    HYPERFRAMES_RENDER_TIMEOUT_SEC = _env_int("HYPERFRAMES_RENDER_TIMEOUT_SEC", 300, minimum=30)
    HYPERFRAMES_RENDER_QUALITY = os.getenv("HYPERFRAMES_RENDER_QUALITY", "").strip().lower()
    HYPERFRAMES_VARIANT_COUNT = min(_env_int("HYPERFRAMES_VARIANT_COUNT", 3, minimum=1), 5)
    HYPERFRAMES_QA_MODE = os.getenv("HYPERFRAMES_QA_MODE", "hybrid").strip().lower()
    if HYPERFRAMES_QA_MODE not in {"local", "hybrid", "vision"}:
        HYPERFRAMES_QA_MODE = "hybrid"
    HYPERFRAMES_ENABLE_VISION_QA = (
        os.getenv("HYPERFRAMES_ENABLE_VISION_QA", "true").strip().lower()
        not in {"0", "false", "no", "off"}
    )
    HYPERFRAMES_MIN_QUALITY_SCORE = min(
        _env_float("HYPERFRAMES_MIN_QUALITY_SCORE", 0.78, minimum=0.0),
        1.0,
    )
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
    GENAI_TIMEOUT_SEC = _env_int("GENAI_TIMEOUT_SEC", 90, minimum=15)
    ANTHROPIC_TIMEOUT_SEC = _env_float("ANTHROPIC_TIMEOUT_SEC", 90.0, minimum=15.0)
    MANIM_PREVIEW_TIMEOUT_SEC = _env_int("MANIM_PREVIEW_TIMEOUT_SEC", 75, minimum=30)
    MANIM_FINAL_TIMEOUT_SEC = max(
        MANIM_PREVIEW_TIMEOUT_SEC,
        _env_int("MANIM_FINAL_TIMEOUT_SEC", 240, minimum=30),
    )
    LLM_REQUEST_MAX_RETRIES = _env_int("LLM_REQUEST_MAX_RETRIES", 3, minimum=1)
    LLM_RETRY_BASE_DELAY_SEC = _env_float("LLM_RETRY_BASE_DELAY_SEC", 1.5, minimum=0.5)


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
