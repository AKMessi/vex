from __future__ import annotations

import os
import re
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import config
from state import utc_now_iso


VIDEO_GENERATION_VERSION = "hyperframes-video-generation-v1"
DEFAULT_DURATION_SEC = 24.0
MIN_DURATION_SEC = 6.0
MAX_DURATION_SEC = 180.0
DEFAULT_VOICE = "af_heart"
VALID_QUALITIES = {"draft", "standard", "high"}
VALID_FORMATS = {"mp4", "webm", "mov", "gif"}
VALID_ASPECTS = {"landscape", "portrait", "square"}
VOICE_RE = re.compile(r"^[a-z]{2}_[a-z0-9_ -]{2,40}$", re.IGNORECASE)


@dataclass(frozen=True)
class VideoGenerationRequest:
    prompt: str
    script: str = ""
    title: str = ""
    duration_sec: float = DEFAULT_DURATION_SEC
    aspect: str = "landscape"
    width: int = 1920
    height: int = 1080
    fps: int = 30
    quality: str = "standard"
    output_format: str = "mp4"
    render_resolution: str = ""
    voice: str = DEFAULT_VOICE
    voice_speed: float = 1.0
    language: str = ""
    generate_audio: bool = True
    transcribe_audio: bool = True
    render: bool = True
    auto_install_runtime: bool = True
    strict_audio_timing: bool = False
    output_dir: str = ""
    project_id: str = ""
    background_music_path: str = ""
    music_volume: float = 0.12
    style: str = "clean_kinetic"
    audience: str = ""
    cta: str = ""
    workers: str = ""
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScriptPlan:
    title: str
    narration: str
    design_direction: str
    source: str
    prompt: str
    audience: str = ""
    cta: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TimedWord:
    text: str
    start: float
    end: float
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Beat:
    beat_id: str
    index: int
    start: float
    end: float
    title: str
    narration: str
    caption: str
    scene_type: str
    keywords: list[str] = field(default_factory=list)
    visual_metaphor: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["duration"] = round(self.duration, 3)
        return payload


@dataclass(frozen=True)
class BeatGraph:
    version: str
    duration_sec: float
    source: str
    beats: list[Beat]
    words: list[TimedWord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "duration_sec": round(self.duration_sec, 3),
            "source": self.source,
            "beats": [beat.to_dict() for beat in self.beats],
            "words": [word.to_dict() for word in self.words],
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class GeneratedVideoResult:
    project_dir: str
    manifest_path: str
    output_path: str = ""
    index_path: str = ""
    script_path: str = ""
    storyboard_path: str = ""
    design_path: str = ""
    beat_graph_path: str = ""
    audio_path: str = ""
    transcript_path: str = ""
    qa_path: str = ""
    rendered: bool = False
    has_audio: bool = False
    duration_sec: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generated_videos_root() -> Path:
    base = Path(config.AGENT_PROJECTS_DIR).expanduser().resolve(strict=False).parent
    return base / "generated_videos"


def normalize_generation_request(params: dict[str, Any] | None) -> VideoGenerationRequest:
    payload = dict(params or {})
    prompt = _clean_text(payload.get("prompt") or payload.get("idea") or payload.get("topic"), limit=900)
    script = _clean_text(payload.get("script") or payload.get("narration"), limit=6000)
    if not prompt and script:
        prompt = _derive_prompt_from_script(script)
    if not prompt and not script:
        raise ValueError("generate_video requires a prompt, idea, topic, script, or narration.")

    title = _clean_text(payload.get("title"), limit=90)
    duration = _bounded_float(payload.get("duration_sec") or payload.get("duration"), DEFAULT_DURATION_SEC)
    duration = max(MIN_DURATION_SEC, min(duration, MAX_DURATION_SEC))
    aspect = str(payload.get("aspect") or payload.get("aspect_ratio") or "landscape").strip().lower()
    if aspect in {"vertical", "portrait_9_16", "9:16", "reels", "shorts", "tiktok"}:
        aspect = "portrait"
    elif aspect in {"1:1", "square"}:
        aspect = "square"
    elif aspect in {"16:9", "horizontal", "landscape"}:
        aspect = "landscape"
    if aspect not in VALID_ASPECTS:
        aspect = "landscape"
    width, height = _resolution_for(aspect)
    explicit_resolution = _clean_text(payload.get("resolution"), limit=32)
    if explicit_resolution:
        width, height = _parse_resolution(explicit_resolution, fallback=(width, height))

    fps = int(max(1, min(_bounded_float(payload.get("fps"), 30.0), 240)))
    quality = str(payload.get("quality") or "standard").strip().lower()
    if quality not in VALID_QUALITIES:
        quality = "standard"
    output_format = str(payload.get("format") or payload.get("output_format") or "mp4").strip().lower().lstrip(".")
    if output_format not in VALID_FORMATS:
        output_format = "mp4"
    if output_format == "gif" and _as_bool(payload.get("generate_audio"), True):
        output_format = "mp4"
    render_resolution = _normalize_render_resolution(
        payload.get("render_resolution")
        or payload.get("export_resolution")
        or payload.get("render_preset"),
        aspect=aspect,
    )

    voice = _clean_text(payload.get("voice") or DEFAULT_VOICE, limit=48).replace(" ", "_")
    if not VOICE_RE.fullmatch(voice):
        voice = DEFAULT_VOICE
    voice_speed = max(0.65, min(_bounded_float(payload.get("voice_speed") or payload.get("speed"), 1.0), 1.35))
    language = _clean_text(payload.get("language") or payload.get("lang"), limit=16).lower()
    generate_audio = _as_bool(payload.get("generate_audio"), True)
    transcribe_audio = _as_bool(payload.get("transcribe_audio"), generate_audio)
    render = _as_bool(payload.get("render"), True)
    auto_install_runtime = _as_bool(payload.get("auto_install_runtime"), True)
    strict_audio_timing = _as_bool(payload.get("strict_audio_timing"), False)

    output_dir = _clean_path_text(payload.get("output_dir") or payload.get("output"))
    project_id = _clean_text(payload.get("project_id"), limit=80)
    if project_id and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", project_id):
        project_id = ""
    background_music_path = _clean_path_text(
        payload.get("background_music_path") or payload.get("music_path")
    )
    music_volume = max(0.0, min(_bounded_float(payload.get("music_volume"), 0.12), 1.0))
    style = _clean_text(payload.get("style") or "clean_kinetic", limit=64).lower().replace(" ", "_")
    audience = _clean_text(payload.get("audience"), limit=160)
    cta = _clean_text(payload.get("cta"), limit=180)
    workers = _clean_text(payload.get("workers"), limit=16).lower()
    if workers and not re.fullmatch(r"(?:auto|[1-9]\d?)", workers):
        workers = ""

    return VideoGenerationRequest(
        prompt=prompt,
        script=script,
        title=title,
        duration_sec=round(duration, 3),
        aspect=aspect,
        width=width,
        height=height,
        fps=fps,
        quality=quality,
        output_format=output_format,
        render_resolution=render_resolution,
        voice=voice,
        voice_speed=round(voice_speed, 3),
        language=language,
        generate_audio=generate_audio,
        transcribe_audio=transcribe_audio and generate_audio,
        render=render,
        auto_install_runtime=auto_install_runtime,
        strict_audio_timing=strict_audio_timing,
        output_dir=output_dir,
        project_id=project_id,
        background_music_path=background_music_path,
        music_volume=round(music_volume, 3),
        style=style,
        audience=audience,
        cta=cta,
        workers=workers,
    )


def make_project_dir(request: VideoGenerationRequest) -> Path:
    root = Path(request.output_dir).expanduser().resolve(strict=False) if request.output_dir else generated_videos_root()
    root.mkdir(parents=True, exist_ok=True)
    slug = _safe_slug(request.title or request.prompt or request.script or "generated-video")
    project_id = request.project_id or uuid.uuid4().hex
    project_dir = root / f"{project_id[:12]}_{slug[:52]}"
    counter = 2
    candidate = project_dir
    while candidate.exists():
        candidate = root / f"{project_id[:12]}_{slug[:44]}_{counter}"
        counter += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _resolution_for(aspect: str) -> tuple[int, int]:
    if aspect == "portrait":
        return 1080, 1920
    if aspect == "square":
        return 1080, 1080
    return 1920, 1080


def _parse_resolution(raw: str, *, fallback: tuple[int, int]) -> tuple[int, int]:
    match = re.fullmatch(r"(\d{3,5})\s*x\s*(\d{3,5})", raw.strip().lower())
    if not match:
        return fallback
    width = int(match.group(1))
    height = int(match.group(2))
    if width < 320 or height < 320 or width > 7680 or height > 7680:
        return fallback
    return width, height


def _normalize_render_resolution(value: object, *, aspect: str) -> str:
    raw = _clean_text(value, limit=32).lower().replace("_", "-")
    if not raw:
        return ""
    if raw in {"best", "showcase", "uhd", "4k"}:
        return "portrait-4k" if aspect == "portrait" else "landscape-4k"
    aliases = {
        "1080": "portrait" if aspect == "portrait" else "landscape",
        "1080p": "portrait" if aspect == "portrait" else "landscape",
        "hd": "portrait" if aspect == "portrait" else "landscape",
        "landscape": "landscape",
        "portrait": "portrait",
        "vertical": "portrait",
        "landscape-4k": "landscape-4k",
        "portrait-4k": "portrait-4k",
        "4k-landscape": "landscape-4k",
        "4k-portrait": "portrait-4k",
    }
    return aliases.get(raw, "")


def _safe_slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", str(value or "")).strip("-_").lower()
    return cleaned or "generated-video"


def _derive_prompt_from_script(script: str) -> str:
    first_sentence = re.split(r"(?<=[.!?])\s+", script.strip(), maxsplit=1)[0]
    return _clean_text(first_sentence, limit=180) or "Generated video"


def _clean_text(value: object, *, limit: int) -> str:
    cleaned = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 1)].rstrip(" ,.;:") + "..."


def _clean_path_text(value: object) -> str:
    cleaned = str(value or "").strip().strip('"\'')
    if not cleaned:
        return ""
    return os.path.expanduser(cleaned)


def _bounded_float(value: object, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    if number != number:
        return default
    return number


def _as_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default
