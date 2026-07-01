from __future__ import annotations

import json
import math
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import config
from engine import parse_timestamp


SONG_MIX_DIRECTOR_VERSION = "song-mix-director-v1"
SONG_MIX_QA_VERSION = "song-mix-qa-v1"


@dataclass(frozen=True)
class SongMixSkill:
    skill_id: str
    label: str
    description: str
    preserve_original_audio: bool
    default_ducking: bool
    default_volume: float
    default_fade_sec: float
    target_lufs: float
    qa_floor: float
    hard_gates: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SongPlacement:
    placement_id: str
    start: float
    end: float
    song_start: float = 0.0
    fade_in: float = 0.0
    fade_out: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["duration"] = round(self.duration, 3)
        return payload


@dataclass(frozen=True)
class SongMixPlan:
    version: str
    selected_skill_id: str
    mode: str
    video_duration_sec: float
    song_duration_sec: float
    source_has_audio: bool
    preserve_original_audio: bool
    ducking_enabled: bool
    loop_song: bool
    loop_policy: str
    normalize_loudness: bool
    music_volume: float
    output_lufs: float
    song_lufs: float
    placements: list[SongPlacement]
    skill_graph: dict[str, Any]
    quality_contract: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    @property
    def max_placement_duration(self) -> float:
        return max((placement.duration for placement in self.placements), default=0.0)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["placements"] = [placement.to_dict() for placement in self.placements]
        payload["max_placement_duration"] = round(self.max_placement_duration, 3)
        return payload


SONG_MIX_SKILLS: tuple[SongMixSkill, ...] = (
    SongMixSkill(
        skill_id="voiceover_bed",
        label="Voiceover Bed",
        description="Mix a song under existing spoken audio while preserving intelligibility.",
        preserve_original_audio=True,
        default_ducking=True,
        default_volume=0.18,
        default_fade_sec=1.2,
        target_lufs=-16.0,
        qa_floor=0.82,
        hard_gates=[
            "output_has_audio",
            "video_duration_preserved",
            "source_audio_preserved",
            "no_obvious_clipping",
        ],
    ),
    SongMixSkill(
        skill_id="silent_video_soundtrack",
        label="Silent Video Soundtrack",
        description="Add a full soundtrack when the source video has no usable audio.",
        preserve_original_audio=False,
        default_ducking=False,
        default_volume=0.42,
        default_fade_sec=1.0,
        target_lufs=-15.0,
        qa_floor=0.80,
        hard_gates=["output_has_audio", "video_duration_preserved", "no_obvious_clipping"],
    ),
    SongMixSkill(
        skill_id="replace_soundtrack",
        label="Replace Soundtrack",
        description="Replace the source soundtrack with the selected song.",
        preserve_original_audio=False,
        default_ducking=False,
        default_volume=0.72,
        default_fade_sec=0.8,
        target_lufs=-15.0,
        qa_floor=0.80,
        hard_gates=["output_has_audio", "video_duration_preserved", "no_obvious_clipping"],
    ),
    SongMixSkill(
        skill_id="intro_sting",
        label="Intro Sting",
        description="Add a short opening song cue, preserving source audio when present.",
        preserve_original_audio=True,
        default_ducking=True,
        default_volume=0.24,
        default_fade_sec=0.45,
        target_lufs=-16.0,
        qa_floor=0.80,
        hard_gates=["output_has_audio", "video_duration_preserved", "no_obvious_clipping"],
    ),
    SongMixSkill(
        skill_id="outro_sting",
        label="Outro Sting",
        description="Add a short closing song cue with a clean tail fade.",
        preserve_original_audio=True,
        default_ducking=True,
        default_volume=0.26,
        default_fade_sec=0.7,
        target_lufs=-16.0,
        qa_floor=0.80,
        hard_gates=["output_has_audio", "video_duration_preserved", "no_obvious_clipping"],
    ),
    SongMixSkill(
        skill_id="intro_outro_sting",
        label="Intro/Outro Sting",
        description="Bookend the video with opening and closing song cues.",
        preserve_original_audio=True,
        default_ducking=True,
        default_volume=0.24,
        default_fade_sec=0.55,
        target_lufs=-16.0,
        qa_floor=0.80,
        hard_gates=["output_has_audio", "video_duration_preserved", "no_obvious_clipping"],
    ),
    SongMixSkill(
        skill_id="highlight_montage",
        label="Highlight Montage",
        description="Use a more present song bed for montage or high-energy sections.",
        preserve_original_audio=True,
        default_ducking=True,
        default_volume=0.30,
        default_fade_sec=0.8,
        target_lufs=-15.0,
        qa_floor=0.80,
        hard_gates=["output_has_audio", "video_duration_preserved", "no_obvious_clipping"],
    ),
    SongMixSkill(
        skill_id="segment_music",
        label="Segment Music",
        description="Add the song only to a bounded time range.",
        preserve_original_audio=True,
        default_ducking=True,
        default_volume=0.22,
        default_fade_sec=0.55,
        target_lufs=-16.0,
        qa_floor=0.80,
        hard_gates=["output_has_audio", "video_duration_preserved", "no_obvious_clipping"],
    ),
)

_SKILLS_BY_ID = {skill.skill_id: skill for skill in SONG_MIX_SKILLS}


def build_song_mix_plan(
    *,
    params: dict[str, Any],
    source_metadata: dict[str, Any],
    song_metadata: dict[str, Any],
) -> SongMixPlan:
    video_duration = _positive_float(source_metadata.get("duration_sec"))
    if video_duration <= 0.0:
        raise ValueError("The current working video does not have a valid duration.")
    song_duration = _positive_float(song_metadata.get("duration_sec"))
    if song_duration <= 0.0 or not bool(song_metadata.get("has_audio")):
        raise ValueError("The selected song file does not contain a readable audio stream.")

    source_has_audio = bool(source_metadata.get("has_audio"))
    mode = _normalize_mode(params.get("mode") or params.get("song_mode"))
    if mode == "auto":
        mode = "background"
    if params.get("start") is not None or params.get("end") is not None:
        if mode == "background":
            mode = "segment"

    skill = _select_skill(mode=mode, source_has_audio=source_has_audio, params=params)
    preserve_original = _as_bool(
        params.get("preserve_original_audio"),
        default=skill.preserve_original_audio and source_has_audio,
    ) and source_has_audio
    if mode == "replace":
        preserve_original = False

    ducking = _normalize_ducking(params.get("ducking"), default=skill.default_ducking and preserve_original)
    volume = _bounded_float(
        params.get("volume") or params.get("music_volume") or params.get("song_volume"),
        default=skill.default_volume,
        low=0.0,
        high=1.5,
    )
    loop_policy = _normalize_loop_policy(params.get("loop_policy") or params.get("loop"))
    normalize_loudness = _as_bool(params.get("normalize"), default=True)
    placements = _build_placements(
        mode=mode,
        params=params,
        video_duration=video_duration,
        song_duration=song_duration,
        default_fade=skill.default_fade_sec,
    )
    max_placement_duration = max((placement.duration for placement in placements), default=0.0)
    loop_song = (
        loop_policy == "loop"
        or (loop_policy == "auto" and song_duration + 0.25 < max_placement_duration)
    )
    warnings: list[str] = []
    if loop_song:
        warnings.append("song_looped_to_cover_requested_duration")
    if preserve_original and not ducking:
        warnings.append("source_audio_preserved_without_ducking")

    skill_graph = _skill_graph(
        selected_skill=skill,
        mode=mode,
        source_has_audio=source_has_audio,
        preserve_original=preserve_original,
        ducking=ducking,
        loop_song=loop_song,
        placements=placements,
    )
    quality_contract = _quality_contract(skill)
    return SongMixPlan(
        version=SONG_MIX_DIRECTOR_VERSION,
        selected_skill_id=skill.skill_id,
        mode=mode,
        video_duration_sec=round(video_duration, 3),
        song_duration_sec=round(song_duration, 3),
        source_has_audio=source_has_audio,
        preserve_original_audio=preserve_original,
        ducking_enabled=ducking,
        loop_song=loop_song,
        loop_policy=loop_policy,
        normalize_loudness=normalize_loudness,
        music_volume=round(volume, 4),
        output_lufs=skill.target_lufs,
        song_lufs=min(skill.target_lufs - 1.0, -16.0),
        placements=placements,
        skill_graph=skill_graph,
        quality_contract=quality_contract,
        warnings=warnings,
    )


def evaluate_song_mix_output(
    *,
    source_path: str,
    output_path: str,
    source_metadata: dict[str, Any],
    output_metadata: dict[str, Any],
    plan: SongMixPlan,
    audio_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stats = audio_stats if audio_stats is not None else collect_audio_stats(output_path)
    source_duration = _positive_float(source_metadata.get("duration_sec"))
    output_duration = _positive_float(output_metadata.get("duration_sec"))
    fps = _positive_float(source_metadata.get("fps")) or 30.0
    tolerance = max(0.35, 3.0 / max(fps, 1.0), source_duration * 0.015)
    duration_delta = abs(output_duration - source_duration)
    resolution_preserved = (
        int(output_metadata.get("width") or 0) == int(source_metadata.get("width") or 0)
        and int(output_metadata.get("height") or 0) == int(source_metadata.get("height") or 0)
    )

    issues: list[str] = []
    warnings: list[str] = []
    if output_duration <= 0.0:
        issues.append("song_mix_output_has_invalid_duration")
    elif duration_delta > tolerance:
        issues.append("song_mix_duration_drift")
    if not resolution_preserved:
        issues.append("song_mix_resolution_changed")
    if not bool(output_metadata.get("has_audio")):
        issues.append("song_mix_output_has_no_audio")
    if plan.preserve_original_audio and not bool(source_metadata.get("has_audio")):
        issues.append("song_mix_requested_source_audio_but_source_is_silent")
    if plan.preserve_original_audio and not plan.ducking_enabled:
        warnings.append("source_audio_preserved_without_ducking")
    if not Path(output_path).is_file():
        issues.append("song_mix_output_missing")
    elif Path(output_path).stat().st_size <= 0:
        issues.append("song_mix_output_empty")

    max_volume = _optional_float(stats.get("max_volume_db") if isinstance(stats, dict) else None)
    mean_volume = _optional_float(stats.get("mean_volume_db") if isinstance(stats, dict) else None)
    stats_available = bool(isinstance(stats, dict) and stats.get("available"))
    if stats_available and max_volume is not None and max_volume >= -0.05:
        issues.append("song_mix_audio_may_clip")
    if stats_available and mean_volume is not None and mean_volume < -45.0:
        warnings.append("song_mix_audio_is_very_quiet")
    if not stats_available:
        warnings.append("song_mix_audio_stats_unavailable")

    gate_count = 5
    passed_gates = sum(
        [
            output_duration > 0.0 and duration_delta <= tolerance,
            resolution_preserved,
            bool(output_metadata.get("has_audio")),
            Path(output_path).is_file() and Path(output_path).stat().st_size > 0 if Path(output_path).exists() else False,
            max_volume is None or max_volume < -0.05,
        ]
    )
    score = max(0.0, min(passed_gates / gate_count - len(issues) * 0.08 - len(warnings) * 0.015, 1.0))
    passed = not issues and score >= float(plan.quality_contract.get("qa_floor", 0.8))
    return {
        "version": SONG_MIX_QA_VERSION,
        "passed": passed,
        "score": round(score, 4),
        "issues": _unique(issues),
        "warnings": _unique([*warnings, *plan.warnings]),
        "evidence": {
            "source_path": str(source_path),
            "output_path": str(output_path),
            "source_duration_sec": round(source_duration, 4),
            "output_duration_sec": round(output_duration, 4),
            "duration_delta_sec": round(duration_delta, 4),
            "duration_tolerance_sec": round(tolerance, 4),
            "resolution_preserved": resolution_preserved,
            "source_has_audio": bool(source_metadata.get("has_audio")),
            "output_has_audio": bool(output_metadata.get("has_audio")),
            "preserve_original_audio": plan.preserve_original_audio,
            "ducking_enabled": plan.ducking_enabled,
            "loop_song": plan.loop_song,
            "audio_stats": stats,
        },
    }


def collect_audio_stats(media_path: str) -> dict[str, Any]:
    command = [
        config.FFMPEG_PATH,
        "-hide_banner",
        "-nostdin",
        "-i",
        str(media_path),
        "-map",
        "0:a:0",
        "-af",
        "volumedetect",
        "-f",
        "null",
        "-",
    ]
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=90,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": str(exc), "command": command}
    stderr = result.stderr or ""
    if result.returncode != 0:
        return {"available": False, "error": _truncate(stderr, 400), "command": command}
    mean = _match_db(stderr, r"mean_volume:\s*(-?\d+(?:\.\d+)?)\s*dB")
    peak = _match_db(stderr, r"max_volume:\s*(-?\d+(?:\.\d+)?)\s*dB")
    return {
        "available": mean is not None or peak is not None,
        "mean_volume_db": mean,
        "max_volume_db": peak,
        "command": command,
    }


def write_song_mix_notes(path: Path, *, plan: SongMixPlan, qa: dict[str, Any]) -> None:
    lines = [
        "# Song Mix Notes",
        "",
        f"Skill: {plan.selected_skill_id}",
        f"Mode: {plan.mode}",
        f"Preserve source audio: {'yes' if plan.preserve_original_audio else 'no'}",
        f"Ducking: {'yes' if plan.ducking_enabled else 'no'}",
        f"Loop song: {'yes' if plan.loop_song else 'no'}",
        f"Music volume: {plan.music_volume:.3f}",
        f"QA: {qa.get('score', 0.0):.3f} ({'passed' if qa.get('passed') else 'failed'})",
        "",
        "## Placements",
        "",
    ]
    for placement in plan.placements:
        lines.append(
            f"- {placement.placement_id}: {placement.start:.2f}s-{placement.end:.2f}s "
            f"(fade {placement.fade_in:.2f}s/{placement.fade_out:.2f}s)"
        )
    if qa.get("issues"):
        lines.extend(["", "## Issues", ""])
        lines.extend(f"- {issue}" for issue in qa.get("issues") or [])
    if qa.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in qa.get("warnings") or [])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plan_from_manifest_payload(payload: dict[str, Any]) -> SongMixPlan:
    placements = [
        SongPlacement(
            placement_id=str(item.get("placement_id") or f"placement_{index:02d}"),
            start=float(item.get("start") or 0.0),
            end=float(item.get("end") or 0.0),
            song_start=float(item.get("song_start") or 0.0),
            fade_in=float(item.get("fade_in") or 0.0),
            fade_out=float(item.get("fade_out") or 0.0),
        )
        for index, item in enumerate(payload.get("placements") or [], start=1)
        if isinstance(item, dict)
    ]
    return SongMixPlan(
        version=str(payload.get("version") or SONG_MIX_DIRECTOR_VERSION),
        selected_skill_id=str(payload.get("selected_skill_id") or "voiceover_bed"),
        mode=str(payload.get("mode") or "background"),
        video_duration_sec=float(payload.get("video_duration_sec") or 0.0),
        song_duration_sec=float(payload.get("song_duration_sec") or 0.0),
        source_has_audio=bool(payload.get("source_has_audio")),
        preserve_original_audio=bool(payload.get("preserve_original_audio")),
        ducking_enabled=bool(payload.get("ducking_enabled")),
        loop_song=bool(payload.get("loop_song")),
        loop_policy=str(payload.get("loop_policy") or "auto"),
        normalize_loudness=bool(payload.get("normalize_loudness")),
        music_volume=float(payload.get("music_volume") or 0.0),
        output_lufs=float(payload.get("output_lufs") or -16.0),
        song_lufs=float(payload.get("song_lufs") or -17.0),
        placements=placements,
        skill_graph=dict(payload.get("skill_graph") or {}),
        quality_contract=dict(payload.get("quality_contract") or {}),
        warnings=[str(item) for item in payload.get("warnings") or []],
    )


def _select_skill(*, mode: str, source_has_audio: bool, params: dict[str, Any]) -> SongMixSkill:
    if not source_has_audio and mode != "replace":
        return _SKILLS_BY_ID["silent_video_soundtrack"]
    if mode == "replace":
        return _SKILLS_BY_ID["replace_soundtrack"]
    if mode == "intro":
        return _SKILLS_BY_ID["intro_sting"]
    if mode == "outro":
        return _SKILLS_BY_ID["outro_sting"]
    if mode == "intro_outro":
        return _SKILLS_BY_ID["intro_outro_sting"]
    if mode == "highlight":
        return _SKILLS_BY_ID["highlight_montage"]
    if mode == "segment":
        return _SKILLS_BY_ID["segment_music"]
    energy = str(params.get("energy") or params.get("mood") or "").strip().lower()
    if energy in {"high", "hype", "energetic", "montage"}:
        return _SKILLS_BY_ID["highlight_montage"]
    return _SKILLS_BY_ID["voiceover_bed"]


def _build_placements(
    *,
    mode: str,
    params: dict[str, Any],
    video_duration: float,
    song_duration: float,
    default_fade: float,
) -> list[SongPlacement]:
    explicit_start = _optional_time(params.get("start"))
    explicit_end = _optional_time(params.get("end"))
    fade_in_raw = _optional_float(params.get("fade_in"))
    fade_out_raw = _optional_float(params.get("fade_out"))

    if mode == "intro":
        length = min(video_duration, _bounded_float(params.get("duration"), 6.0, low=0.25, high=video_duration))
        return [_placement("intro", 0.0, length, default_fade, fade_in_raw, fade_out_raw)]
    if mode == "outro":
        length = min(video_duration, _bounded_float(params.get("duration"), 7.0, low=0.25, high=video_duration))
        return [_placement("outro", max(0.0, video_duration - length), video_duration, default_fade, fade_in_raw, fade_out_raw)]
    if mode == "intro_outro":
        length = min(video_duration / 2.0, _bounded_float(params.get("duration"), 6.0, low=0.25, high=video_duration))
        return [
            _placement("intro", 0.0, length, default_fade, fade_in_raw, fade_out_raw),
            _placement("outro", max(length, video_duration - length), video_duration, default_fade, fade_in_raw, fade_out_raw),
        ]

    if mode in {"segment", "highlight"}:
        start = explicit_start if explicit_start is not None else 0.0
        if explicit_end is not None:
            end = explicit_end
        else:
            requested = _bounded_float(
                params.get("duration"),
                default=min(song_duration, max(4.0, video_duration - start)),
                low=0.25,
                high=max(0.25, video_duration - start),
            )
            end = start + requested
        start = max(0.0, min(start, video_duration))
        end = max(start + 0.1, min(end, video_duration))
        return [_placement("segment", start, end, default_fade, fade_in_raw, fade_out_raw)]

    start = explicit_start if explicit_start is not None else 0.0
    end = explicit_end if explicit_end is not None else video_duration
    start = max(0.0, min(start, video_duration))
    end = max(start + 0.1, min(end, video_duration))
    return [_placement("background", start, end, default_fade, fade_in_raw, fade_out_raw)]


def _placement(
    placement_id: str,
    start: float,
    end: float,
    default_fade: float,
    fade_in_raw: float | None,
    fade_out_raw: float | None,
) -> SongPlacement:
    duration = max(end - start, 0.0)
    max_fade = max(0.0, duration * 0.42)
    fade_in = min(max(0.0, fade_in_raw if fade_in_raw is not None else default_fade), max_fade)
    fade_out = min(max(0.0, fade_out_raw if fade_out_raw is not None else default_fade), max_fade)
    return SongPlacement(
        placement_id=placement_id,
        start=round(start, 3),
        end=round(end, 3),
        fade_in=round(fade_in, 3),
        fade_out=round(fade_out, 3),
    )


def _skill_graph(
    *,
    selected_skill: SongMixSkill,
    mode: str,
    source_has_audio: bool,
    preserve_original: bool,
    ducking: bool,
    loop_song: bool,
    placements: list[SongPlacement],
) -> dict[str, Any]:
    return {
        "version": SONG_MIX_DIRECTOR_VERSION,
        "selected_skill_id": selected_skill.skill_id,
        "skills": [skill.to_dict() for skill in SONG_MIX_SKILLS],
        "decision_trace": [
            f"mode={mode}",
            f"source_has_audio={source_has_audio}",
            f"preserve_original_audio={preserve_original}",
            f"ducking_enabled={ducking}",
            f"loop_song={loop_song}",
            f"placements={len(placements)}",
        ],
    }


def _quality_contract(skill: SongMixSkill) -> dict[str, Any]:
    return {
        "version": SONG_MIX_QA_VERSION,
        "skill_id": skill.skill_id,
        "qa_floor": skill.qa_floor,
        "hard_gates": list(skill.hard_gates),
        "soft_gates": [
            "music_starts_and_ends_on_requested_timing",
            "speech_bed_uses_ducking_when_preserving_source_audio",
            "looping_is_explicit_when_song_is_shorter_than_requested_window",
            "rendered_mix_has_auditable_loudness_stats",
        ],
    }


def _normalize_mode(value: object) -> str:
    normalized = str(value or "auto").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "bed": "background",
        "background_music": "background",
        "soundtrack": "background",
        "replace_audio": "replace",
        "replace_song": "replace",
        "opening": "intro",
        "start": "intro",
        "ending": "outro",
        "end": "outro",
        "bookend": "intro_outro",
        "bookends": "intro_outro",
        "introoutro": "intro_outro",
        "montage": "highlight",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"auto", "background", "replace", "intro", "outro", "intro_outro", "segment", "highlight"}:
        return "auto"
    return normalized


def _normalize_loop_policy(value: object) -> str:
    if isinstance(value, bool):
        return "loop" if value else "trim"
    normalized = str(value or "auto").strip().lower().replace("-", "_")
    if normalized in {"yes", "true", "on", "repeat"}:
        return "loop"
    if normalized in {"no", "false", "off", "none"}:
        return "trim"
    if normalized in {"auto", "loop", "trim", "pad"}:
        return normalized
    return "auto"


def _normalize_ducking(value: object, *, default: bool) -> bool:
    normalized = str(value or "").strip().lower()
    if normalized == "auto" or value is None:
        return default
    return _as_bool(value, default=default)


def _optional_time(value: object) -> float | None:
    if value is None:
        return None
    try:
        return max(0.0, float(parse_timestamp(value)))
    except (TypeError, ValueError):
        return None


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _positive_float(value: object) -> float:
    number = _optional_float(value)
    return number if number is not None and number > 0.0 else 0.0


def _bounded_float(value: object, default: float, *, low: float, high: float) -> float:
    number = _optional_float(value)
    if number is None:
        number = default
    return max(low, min(number, high))


def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _as_bool(value: object, *, default: bool) -> bool:
    parsed = _optional_bool(value)
    return default if parsed is None else parsed


def _match_db(text: str, pattern: str) -> float | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return round(float(match.group(1)), 3)
    except ValueError:
        return None


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _truncate(value: str, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip(" ,.;:") + "..."


def manifest_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)
