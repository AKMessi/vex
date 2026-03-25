from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

PROVIDER = "gemini"
GEMINI_API_KEY = None
GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
ANTHROPIC_API_KEY = None
CLAUDE_MODEL = "claude-sonnet-4-5"
PEXELS_API_KEY = None
AGENT_PROJECTS_DIR = os.path.expanduser("~/.video-agent/projects/")
FFMPEG_PATH = "ffmpeg"
WHISPER_MODEL = "base"
VERSION = "1.0.0"


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
    global WHISPER_MODEL

    PROVIDER = os.getenv("PROVIDER", "gemini").strip().lower()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5")
    PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
    AGENT_PROJECTS_DIR = os.path.expanduser(
        os.getenv("AGENT_PROJECTS_DIR", "~/.video-agent/projects/")
    )
    FFMPEG_PATH = os.getenv("FFMPEG_PATH", "ffmpeg")
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")


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
