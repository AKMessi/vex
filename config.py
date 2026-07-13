from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from vex_runtime import __version__

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
PIXABAY_API_KEY = None
COVERR_API_KEY = None
AUTO_BROLL_PROVIDERS = "auto"
AUTO_BROLL_MAX_OVERLAYS = 24
AUTO_VISUALS_MAX_VISUALS = 32
AUTO_VISUALS_RENDERER_TOURNAMENT_SIZE = 2
AUTO_VISUALS_COMPOSITE_SIMILARITY_FLOOR = 0.72
VISUAL_COMPOSITE_SIMILARITY_FLOOR = 0.72
OPEN_VISUAL_PROGRAM_ENABLED = True
OPEN_VISUAL_PROGRAM_LLM_AUTHORING = True
OPEN_VISUAL_PROGRAM_CANDIDATES = 3
OPEN_VISUAL_PROGRAM_AUTHORING_ATTEMPTS = 2
OPEN_VISUAL_PROGRAM_MIN_SCORE = 0.78
VISUAL_DIRECTOR_ENABLED = True
VISUAL_DIRECTOR_VERIFICATION_MODE = "balanced"
VISUAL_DIRECTOR_VISION_MODEL = ""
VISUAL_DIRECTOR_CLAUDE_VISION_MODEL = ""
VISUAL_DIRECTOR_VERIFIER_RETRIES = 2
VISUAL_DIRECTOR_CIRCUIT_FAILURES = 2
VISUAL_DIRECTOR_CIRCUIT_COOLDOWN_SEC = 45.0
VISUAL_DIRECTOR_PAIRWISE_TOP_K = 3
VISUAL_DIRECTOR_MAX_REPAIR_ROUNDS = 2
VISUAL_DIRECTOR_MIN_REPAIR_DELTA = 0.025
VISUAL_DIRECTOR_RENDER_CANDIDATES = 2
AGENT_PROJECTS_DIR = os.path.expanduser("~/.video-agent/projects/")
FFMPEG_PATH = "ffmpeg"
FFMPEG_PROBE_TIMEOUT_SEC = 30
FFMPEG_RENDER_TIMEOUT_SEC = 7200
ENCODE_VALIDATION_TIMEOUT_SEC = 300
BLENDER_PATH = "blender"
BLENDER_RENDER_TIMEOUT_SEC = 3600
HYPERFRAMES_CLI_PATH = "hyperframes"
HYPERFRAMES_LINT_TIMEOUT_SEC = 90
HYPERFRAMES_RENDER_TIMEOUT_SEC = 0
HYPERFRAMES_RENDER_QUALITY = ""
HYPERFRAMES_VARIANT_COUNT = 3
HYPERFRAMES_PROOF_CANDIDATE_COUNT = 4
HYPERFRAMES_QA_MODE = "hybrid"
HYPERFRAMES_ENABLE_VISION_QA = True
HYPERFRAMES_ENABLE_COUNTERFACTUAL_QA = True
HYPERFRAMES_VISION_MODEL = ""
HYPERFRAMES_BLIND_DECODER_MIN_SCORE = 0.68
HYPERFRAMES_MIN_QUALITY_SCORE = 0.78
HYPERFRAMES_ENABLE_CEGIS = True
HYPERFRAMES_MAX_CRITIC_FRAMES = 8
HYPERFRAMES_MAX_REPAIR_ROUNDS = 3
HYPERFRAMES_MIN_REPAIR_DELTA = 0.025
REMOTION_RENDER_TIMEOUT_SEC = 0
REMOTION_RENDER_CONCURRENCY = ""
WHISPER_MODEL = "base"
WHISPER_PYTHON_PATH = ""
WHISPER_TRANSCRIBE_TIMEOUT_SEC = 7200
VERSION = __version__
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


def _env_timeout_sec(name: str, default: int, *, minimum: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        _print_and_exit(f"Invalid {name}={raw!r}. Expected an integer timeout in seconds.")
    if value <= 0:
        return 0
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
    global PIXABAY_API_KEY
    global COVERR_API_KEY
    global AUTO_BROLL_PROVIDERS
    global AUTO_BROLL_MAX_OVERLAYS
    global AUTO_VISUALS_MAX_VISUALS
    global AUTO_VISUALS_RENDERER_TOURNAMENT_SIZE
    global AUTO_VISUALS_COMPOSITE_SIMILARITY_FLOOR
    global VISUAL_COMPOSITE_SIMILARITY_FLOOR
    global OPEN_VISUAL_PROGRAM_ENABLED
    global OPEN_VISUAL_PROGRAM_LLM_AUTHORING
    global OPEN_VISUAL_PROGRAM_CANDIDATES
    global OPEN_VISUAL_PROGRAM_AUTHORING_ATTEMPTS
    global OPEN_VISUAL_PROGRAM_MIN_SCORE
    global VISUAL_DIRECTOR_ENABLED
    global VISUAL_DIRECTOR_VERIFICATION_MODE
    global VISUAL_DIRECTOR_VISION_MODEL
    global VISUAL_DIRECTOR_CLAUDE_VISION_MODEL
    global VISUAL_DIRECTOR_VERIFIER_RETRIES
    global VISUAL_DIRECTOR_CIRCUIT_FAILURES
    global VISUAL_DIRECTOR_CIRCUIT_COOLDOWN_SEC
    global VISUAL_DIRECTOR_PAIRWISE_TOP_K
    global VISUAL_DIRECTOR_MAX_REPAIR_ROUNDS
    global VISUAL_DIRECTOR_MIN_REPAIR_DELTA
    global VISUAL_DIRECTOR_RENDER_CANDIDATES
    global AGENT_PROJECTS_DIR
    global FFMPEG_PATH
    global FFMPEG_PROBE_TIMEOUT_SEC
    global FFMPEG_RENDER_TIMEOUT_SEC
    global ENCODE_VALIDATION_TIMEOUT_SEC
    global BLENDER_PATH
    global BLENDER_RENDER_TIMEOUT_SEC
    global HYPERFRAMES_CLI_PATH
    global HYPERFRAMES_LINT_TIMEOUT_SEC
    global HYPERFRAMES_RENDER_TIMEOUT_SEC
    global HYPERFRAMES_RENDER_QUALITY
    global HYPERFRAMES_VARIANT_COUNT
    global HYPERFRAMES_PROOF_CANDIDATE_COUNT
    global HYPERFRAMES_QA_MODE
    global HYPERFRAMES_ENABLE_VISION_QA
    global HYPERFRAMES_ENABLE_COUNTERFACTUAL_QA
    global HYPERFRAMES_VISION_MODEL
    global HYPERFRAMES_BLIND_DECODER_MIN_SCORE
    global HYPERFRAMES_MIN_QUALITY_SCORE
    global HYPERFRAMES_ENABLE_CEGIS
    global HYPERFRAMES_MAX_CRITIC_FRAMES
    global HYPERFRAMES_MAX_REPAIR_ROUNDS
    global HYPERFRAMES_MIN_REPAIR_DELTA
    global REMOTION_RENDER_TIMEOUT_SEC
    global REMOTION_RENDER_CONCURRENCY
    global WHISPER_MODEL
    global WHISPER_PYTHON_PATH
    global WHISPER_TRANSCRIBE_TIMEOUT_SEC
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
    PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY")
    COVERR_API_KEY = os.getenv("COVERR_API_KEY")
    AUTO_BROLL_PROVIDERS = os.getenv("AUTO_BROLL_PROVIDERS", "auto").strip().lower() or "auto"
    AUTO_BROLL_MAX_OVERLAYS = min(
        _env_int("AUTO_BROLL_MAX_OVERLAYS", 24, minimum=1),
        64,
    )
    AUTO_VISUALS_MAX_VISUALS = min(
        _env_int("AUTO_VISUALS_MAX_VISUALS", 32, minimum=1),
        96,
    )
    AUTO_VISUALS_RENDERER_TOURNAMENT_SIZE = min(
        _env_int("AUTO_VISUALS_RENDERER_TOURNAMENT_SIZE", 2, minimum=1),
        3,
    )
    legacy_composite_floor = os.getenv(
        "AUTO_VISUALS_COMPOSITE_SIMILARITY_FLOOR",
        "0.72",
    )
    visual_composite_floor = os.getenv(
        "VISUAL_COMPOSITE_SIMILARITY_FLOOR",
        legacy_composite_floor,
    )
    try:
        parsed_composite_floor = float(visual_composite_floor)
    except ValueError:
        _print_and_exit(
            "Invalid VISUAL_COMPOSITE_SIMILARITY_FLOOR="
            f"{visual_composite_floor!r}. Expected a numeric value."
        )
    VISUAL_COMPOSITE_SIMILARITY_FLOOR = max(
        0.5,
        min(parsed_composite_floor, 0.98),
    )
    AUTO_VISUALS_COMPOSITE_SIMILARITY_FLOOR = (
        VISUAL_COMPOSITE_SIMILARITY_FLOOR
    )
    OPEN_VISUAL_PROGRAM_ENABLED = _env_bool(
        "OPEN_VISUAL_PROGRAM_ENABLED",
        True,
    )
    OPEN_VISUAL_PROGRAM_LLM_AUTHORING = _env_bool(
        "OPEN_VISUAL_PROGRAM_LLM_AUTHORING",
        True,
    )
    OPEN_VISUAL_PROGRAM_CANDIDATES = min(
        _env_int("OPEN_VISUAL_PROGRAM_CANDIDATES", 3, minimum=1),
        4,
    )
    OPEN_VISUAL_PROGRAM_AUTHORING_ATTEMPTS = min(
        _env_int("OPEN_VISUAL_PROGRAM_AUTHORING_ATTEMPTS", 2, minimum=1),
        2,
    )
    OPEN_VISUAL_PROGRAM_MIN_SCORE = min(
        _env_float("OPEN_VISUAL_PROGRAM_MIN_SCORE", 0.78, minimum=0.0),
        1.0,
    )
    VISUAL_DIRECTOR_ENABLED = _env_bool("VISUAL_DIRECTOR_ENABLED", True)
    VISUAL_DIRECTOR_VERIFICATION_MODE = os.getenv(
        "VISUAL_DIRECTOR_VERIFICATION_MODE",
        "balanced",
    ).strip().lower()
    if VISUAL_DIRECTOR_VERIFICATION_MODE not in {"strict", "balanced", "off"}:
        _print_and_exit(
            "Invalid VISUAL_DIRECTOR_VERIFICATION_MODE="
            f"{VISUAL_DIRECTOR_VERIFICATION_MODE!r}. Expected strict, balanced, or off."
        )
    VISUAL_DIRECTOR_VISION_MODEL = os.getenv(
        "VISUAL_DIRECTOR_VISION_MODEL",
        HYPERFRAMES_VISION_MODEL or GEMINI_MODEL,
    ).strip()
    VISUAL_DIRECTOR_CLAUDE_VISION_MODEL = os.getenv(
        "VISUAL_DIRECTOR_CLAUDE_VISION_MODEL",
        CLAUDE_MODEL,
    ).strip()
    VISUAL_DIRECTOR_VERIFIER_RETRIES = min(
        _env_int("VISUAL_DIRECTOR_VERIFIER_RETRIES", 2, minimum=1),
        3,
    )
    VISUAL_DIRECTOR_CIRCUIT_FAILURES = min(
        _env_int("VISUAL_DIRECTOR_CIRCUIT_FAILURES", 2, minimum=1),
        8,
    )
    VISUAL_DIRECTOR_CIRCUIT_COOLDOWN_SEC = min(
        _env_float("VISUAL_DIRECTOR_CIRCUIT_COOLDOWN_SEC", 45.0, minimum=1.0),
        600.0,
    )
    VISUAL_DIRECTOR_PAIRWISE_TOP_K = min(
        _env_int("VISUAL_DIRECTOR_PAIRWISE_TOP_K", 3, minimum=1),
        4,
    )
    VISUAL_DIRECTOR_MAX_REPAIR_ROUNDS = min(
        _env_int("VISUAL_DIRECTOR_MAX_REPAIR_ROUNDS", 2, minimum=0),
        4,
    )
    VISUAL_DIRECTOR_MIN_REPAIR_DELTA = min(
        _env_float("VISUAL_DIRECTOR_MIN_REPAIR_DELTA", 0.025, minimum=0.0),
        0.25,
    )
    VISUAL_DIRECTOR_RENDER_CANDIDATES = min(
        _env_int("VISUAL_DIRECTOR_RENDER_CANDIDATES", 2, minimum=1),
        3,
    )
    AGENT_PROJECTS_DIR = os.path.expanduser(
        os.getenv("AGENT_PROJECTS_DIR", "~/.video-agent/projects/")
    )
    FFMPEG_PATH = os.getenv("FFMPEG_PATH", "ffmpeg")
    FFMPEG_PROBE_TIMEOUT_SEC = _env_int("FFMPEG_PROBE_TIMEOUT_SEC", 30, minimum=5)
    FFMPEG_RENDER_TIMEOUT_SEC = _env_timeout_sec(
        "FFMPEG_RENDER_TIMEOUT_SEC",
        7200,
        minimum=30,
    )
    ENCODE_VALIDATION_TIMEOUT_SEC = _env_int("ENCODE_VALIDATION_TIMEOUT_SEC", 300, minimum=15)
    BLENDER_PATH = os.getenv("BLENDER_PATH", "blender")
    BLENDER_RENDER_TIMEOUT_SEC = _env_timeout_sec("BLENDER_RENDER_TIMEOUT_SEC", 3600, minimum=30)
    HYPERFRAMES_CLI_PATH = os.getenv("HYPERFRAMES_CLI_PATH", "hyperframes").strip() or "hyperframes"
    HYPERFRAMES_LINT_TIMEOUT_SEC = _env_int("HYPERFRAMES_LINT_TIMEOUT_SEC", 90, minimum=15)
    HYPERFRAMES_RENDER_TIMEOUT_SEC = _env_timeout_sec("HYPERFRAMES_RENDER_TIMEOUT_SEC", 0, minimum=30)
    HYPERFRAMES_RENDER_QUALITY = os.getenv("HYPERFRAMES_RENDER_QUALITY", "").strip().lower()
    HYPERFRAMES_VARIANT_COUNT = min(_env_int("HYPERFRAMES_VARIANT_COUNT", 3, minimum=1), 5)
    HYPERFRAMES_PROOF_CANDIDATE_COUNT = min(
        _env_int("HYPERFRAMES_PROOF_CANDIDATE_COUNT", 4, minimum=1),
        8,
    )
    HYPERFRAMES_QA_MODE = os.getenv("HYPERFRAMES_QA_MODE", "hybrid").strip().lower()
    if HYPERFRAMES_QA_MODE not in {"local", "hybrid", "vision"}:
        HYPERFRAMES_QA_MODE = "hybrid"
    HYPERFRAMES_ENABLE_VISION_QA = (
        os.getenv("HYPERFRAMES_ENABLE_VISION_QA", "true").strip().lower()
        not in {"0", "false", "no", "off"}
    )
    HYPERFRAMES_ENABLE_COUNTERFACTUAL_QA = _env_bool(
        "HYPERFRAMES_ENABLE_COUNTERFACTUAL_QA",
        True,
    )
    HYPERFRAMES_VISION_MODEL = os.getenv(
        "HYPERFRAMES_VISION_MODEL",
        GEMINI_MODEL,
    ).strip()
    HYPERFRAMES_BLIND_DECODER_MIN_SCORE = min(
        _env_float("HYPERFRAMES_BLIND_DECODER_MIN_SCORE", 0.68, minimum=0.0),
        1.0,
    )
    HYPERFRAMES_MIN_QUALITY_SCORE = min(
        _env_float("HYPERFRAMES_MIN_QUALITY_SCORE", 0.78, minimum=0.0),
        1.0,
    )
    HYPERFRAMES_ENABLE_CEGIS = _env_bool("HYPERFRAMES_ENABLE_CEGIS", True)
    HYPERFRAMES_MAX_CRITIC_FRAMES = min(
        _env_int("HYPERFRAMES_MAX_CRITIC_FRAMES", 8, minimum=3),
        12,
    )
    HYPERFRAMES_MAX_REPAIR_ROUNDS = min(
        _env_int("HYPERFRAMES_MAX_REPAIR_ROUNDS", 3, minimum=0),
        5,
    )
    HYPERFRAMES_MIN_REPAIR_DELTA = min(
        _env_float("HYPERFRAMES_MIN_REPAIR_DELTA", 0.025, minimum=0.0),
        0.25,
    )
    REMOTION_RENDER_TIMEOUT_SEC = _env_timeout_sec(
        "REMOTION_RENDER_TIMEOUT_SEC",
        0,
        minimum=30,
    )
    REMOTION_RENDER_CONCURRENCY = os.getenv(
        "REMOTION_RENDER_CONCURRENCY",
        "",
    ).strip()
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
    WHISPER_PYTHON_PATH = os.getenv("WHISPER_PYTHON_PATH", "").strip()
    WHISPER_TRANSCRIBE_TIMEOUT_SEC = _env_int(
        "WHISPER_TRANSCRIBE_TIMEOUT_SEC",
        7200,
        minimum=30,
    )
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
