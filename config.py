from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*_args, **_kwargs) -> bool:
        return False


if TYPE_CHECKING:
    from google.genai import types

PROVIDER = "gemini"
GEMINI_API_KEY = None
GEMINI_MODEL = "gemma-4-31b-it"
ANTHROPIC_API_KEY = None
CLAUDE_MODEL = "claude-sonnet-4-5"
OPENAI_COMPAT_BASE_URL = "http://localhost:11434/v1"
OPENAI_COMPAT_API_KEY = ""
OPENAI_COMPAT_MODEL = "qwen2.5-coder:14b"
OPENAI_COMPAT_TIMEOUT_SEC = 120.0
OPENAI_COMPAT_MAX_TOKENS = 4096
OPENAI_COMPAT_TEMPERATURE = 0.2
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "qwen2.5-coder:14b"
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
LM_STUDIO_MODEL = ""
LLAMA_CPP_BASE_URL = "http://localhost:8080/v1"
LLAMA_CPP_MODEL = ""
PEXELS_API_KEY = None
AGENT_PROJECTS_DIR = os.path.expanduser("~/.video-agent/projects/")
FFMPEG_PATH = "ffmpeg"
ENCODE_VALIDATION_TIMEOUT_SEC = 300
BLENDER_PATH = "blender"
BLENDER_RENDER_TIMEOUT_SEC = 300
HYPERFRAMES_CLI_PATH = "hyperframes"
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
MANIM_ALLOW_LLM_CODEGEN = False
LLM_REQUEST_MAX_RETRIES = 3
LLM_RETRY_BASE_DELAY_SEC = 1.5
SUPPORTED_PROVIDERS = {"gemini", "claude", "openai_compatible", "ollama", "lmstudio", "llama_cpp"}
LOCAL_LLM_PROVIDERS = {"openai_compatible", "ollama", "lmstudio", "llama_cpp"}


def normalize_provider_name(name: str | None) -> str:
    normalized = (name or "gemini").strip().lower().replace("-", "_")
    aliases = {
        "openai_compat": "openai_compatible",
        "openai_compatible": "openai_compatible",
        "ollama": "ollama",
        "lm_studio": "lmstudio",
        "lmstudio": "lmstudio",
        "llamacpp": "llama_cpp",
        "llama_cpp": "llama_cpp",
        "llama.cpp": "llama_cpp",
    }
    return aliases.get(normalized, normalized)


def local_llm_base_url(provider_name: str | None = None) -> str:
    provider = normalize_provider_name(provider_name or PROVIDER)
    if provider == "ollama":
        return OLLAMA_BASE_URL or OPENAI_COMPAT_BASE_URL
    if provider == "lmstudio":
        return LM_STUDIO_BASE_URL or OPENAI_COMPAT_BASE_URL
    if provider == "llama_cpp":
        return LLAMA_CPP_BASE_URL or OPENAI_COMPAT_BASE_URL
    return OPENAI_COMPAT_BASE_URL


def local_llm_model(provider_name: str | None = None) -> str:
    provider = normalize_provider_name(provider_name or PROVIDER)
    if provider == "ollama":
        return OLLAMA_MODEL or OPENAI_COMPAT_MODEL
    if provider == "lmstudio":
        return LM_STUDIO_MODEL or OPENAI_COMPAT_MODEL
    if provider == "llama_cpp":
        return LLAMA_CPP_MODEL or OPENAI_COMPAT_MODEL
    return OPENAI_COMPAT_MODEL


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


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    _print_and_exit(f"Invalid {name}={raw!r}. Expected a boolean value.")


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
    global OPENAI_COMPAT_BASE_URL
    global OPENAI_COMPAT_API_KEY
    global OPENAI_COMPAT_MODEL
    global OPENAI_COMPAT_TIMEOUT_SEC
    global OPENAI_COMPAT_MAX_TOKENS
    global OPENAI_COMPAT_TEMPERATURE
    global OLLAMA_BASE_URL
    global OLLAMA_MODEL
    global LM_STUDIO_BASE_URL
    global LM_STUDIO_MODEL
    global LLAMA_CPP_BASE_URL
    global LLAMA_CPP_MODEL
    global PEXELS_API_KEY
    global AGENT_PROJECTS_DIR
    global FFMPEG_PATH
    global ENCODE_VALIDATION_TIMEOUT_SEC
    global BLENDER_PATH
    global BLENDER_RENDER_TIMEOUT_SEC
    global HYPERFRAMES_CLI_PATH
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
    global MANIM_ALLOW_LLM_CODEGEN
    global LLM_REQUEST_MAX_RETRIES
    global LLM_RETRY_BASE_DELAY_SEC

    PROVIDER = normalize_provider_name(os.getenv("PROVIDER", "gemini"))
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemma-4-31b-it")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5")
    OPENAI_COMPAT_BASE_URL = os.getenv("OPENAI_COMPAT_BASE_URL", "http://localhost:11434/v1").strip().rstrip("/")
    OPENAI_COMPAT_API_KEY = os.getenv("OPENAI_COMPAT_API_KEY", "").strip()
    OPENAI_COMPAT_MODEL = os.getenv("OPENAI_COMPAT_MODEL", "qwen2.5-coder:14b").strip()
    OPENAI_COMPAT_TIMEOUT_SEC = _env_float("OPENAI_COMPAT_TIMEOUT_SEC", 120.0, minimum=15.0)
    OPENAI_COMPAT_MAX_TOKENS = _env_int("OPENAI_COMPAT_MAX_TOKENS", 4096, minimum=256)
    OPENAI_COMPAT_TEMPERATURE = min(_env_float("OPENAI_COMPAT_TEMPERATURE", 0.2, minimum=0.0), 2.0)
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1").strip().rstrip("/")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", OPENAI_COMPAT_MODEL).strip()
    LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1").strip().rstrip("/")
    LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", OPENAI_COMPAT_MODEL).strip()
    LLAMA_CPP_BASE_URL = os.getenv("LLAMA_CPP_BASE_URL", "http://localhost:8080/v1").strip().rstrip("/")
    LLAMA_CPP_MODEL = os.getenv("LLAMA_CPP_MODEL", OPENAI_COMPAT_MODEL).strip()
    PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
    AGENT_PROJECTS_DIR = os.path.expanduser(
        os.getenv("AGENT_PROJECTS_DIR", "~/.video-agent/projects/")
    )
    FFMPEG_PATH = os.getenv("FFMPEG_PATH", "ffmpeg")
    ENCODE_VALIDATION_TIMEOUT_SEC = _env_int("ENCODE_VALIDATION_TIMEOUT_SEC", 300, minimum=15)
    BLENDER_PATH = os.getenv("BLENDER_PATH", "blender")
    BLENDER_RENDER_TIMEOUT_SEC = _env_int("BLENDER_RENDER_TIMEOUT_SEC", 300, minimum=30)
    HYPERFRAMES_CLI_PATH = os.getenv("HYPERFRAMES_CLI_PATH", "hyperframes").strip() or "hyperframes"
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
    MANIM_ALLOW_LLM_CODEGEN = _env_bool("MANIM_ALLOW_LLM_CODEGEN", False)
    LLM_REQUEST_MAX_RETRIES = _env_int("LLM_REQUEST_MAX_RETRIES", 3, minimum=1)
    LLM_RETRY_BASE_DELAY_SEC = _env_float("LLM_RETRY_BASE_DELAY_SEC", 1.5, minimum=0.5)


def validate_config(*, require_provider: bool = True) -> None:
    reload_settings()

    if PROVIDER not in SUPPORTED_PROVIDERS:
        _print_and_exit(
            "Invalid PROVIDER="
            f"{PROVIDER!r}. Valid options are: "
            "'gemini', 'claude', 'openai_compatible', 'ollama', 'lmstudio', 'llama_cpp'."
        )

    if require_provider and PROVIDER == "gemini" and not GEMINI_API_KEY:
        _print_and_exit(
            "GEMINI_API_KEY is required when PROVIDER=gemini. "
            "Set it in your environment or .env file."
        )

    if require_provider and PROVIDER == "claude" and not ANTHROPIC_API_KEY:
        _print_and_exit(
            "ANTHROPIC_API_KEY is required when PROVIDER=claude. "
            "Set it in your environment or .env file."
        )

    if require_provider and PROVIDER in LOCAL_LLM_PROVIDERS:
        base_url = local_llm_base_url(PROVIDER)
        model_name = local_llm_model(PROVIDER)
        if not base_url.startswith(("http://", "https://")):
            _print_and_exit(
                f"Invalid local LLM base URL {base_url!r}. Expected an http:// or https:// URL."
            )
        if not model_name:
            _print_and_exit(
                f"A model name is required when PROVIDER={PROVIDER}. "
                "Set OPENAI_COMPAT_MODEL or the provider-specific model variable."
            )

    if shutil.which(FFMPEG_PATH) is None:
        _print_and_exit(_ffmpeg_install_instructions())

    Path(AGENT_PROJECTS_DIR).mkdir(parents=True, exist_ok=True)
