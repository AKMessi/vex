from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
import subprocess
from typing import Any, Callable

import config
from engine import probe_video


COMPOSITE_QA_VERSION = "visual-composite-qa-v1"
FRAME_WIDTH = 48
FRAME_HEIGHT = 27
FRAME_BYTES = FRAME_WIDTH * FRAME_HEIGHT * 3


@dataclass(frozen=True)
class CompositeSample:
    visual_id: str
    output_time_sec: float
    asset_time_sec: float
    similarity: float
    output_brightness: float
    output_contrast: float
    passed: bool
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CompositeQualityReport:
    version: str
    passed: bool
    score: float
    issues: list[str]
    warnings: list[str]
    metadata_checks: dict[str, Any]
    samples: list[CompositeSample]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["samples"] = [sample.to_dict() for sample in self.samples]
        return payload


def evaluate_visual_composite(
    source_path: str,
    output_path: str,
    overlays: list[dict[str, Any]],
    *,
    source_metadata: dict[str, Any] | None = None,
    output_metadata: dict[str, Any] | None = None,
    similarity_floor: float | None = None,
    frame_extractor: Callable[[str, float], bytes | None] | None = None,
) -> CompositeQualityReport:
    source_meta = dict(source_metadata or probe_video(source_path))
    output_meta = dict(output_metadata or probe_video(output_path))
    floor = max(
        0.5,
        min(
            float(
                similarity_floor
                if similarity_floor is not None
                else config.VISUAL_COMPOSITE_SIMILARITY_FLOOR
            ),
            0.98,
        ),
    )
    extractor = frame_extractor or _extract_fill_frame
    issues: list[str] = []
    warnings: list[str] = []

    source_duration = _finite_float(source_meta.get("duration_sec"), 0.0)
    output_duration = _finite_float(output_meta.get("duration_sec"), 0.0)
    duration_tolerance = max(
        0.25,
        3.0 / max(_finite_float(source_meta.get("fps"), 30.0), 1.0),
    )
    duration_delta = abs(output_duration - source_duration)
    resolution_preserved = (
        int(output_meta.get("width") or 0) == int(source_meta.get("width") or 0)
        and int(output_meta.get("height") or 0)
        == int(source_meta.get("height") or 0)
    )
    audio_preserved = not bool(source_meta.get("has_audio")) or bool(
        output_meta.get("has_audio")
    )
    if output_duration <= 0.0:
        issues.append("composite_output_has_invalid_duration")
    elif duration_delta > duration_tolerance:
        issues.append("composite_duration_drift")
    if not resolution_preserved:
        issues.append("composite_resolution_changed")
    if not audio_preserved:
        issues.append("composite_dropped_source_audio")
    if int(output_meta.get("size_bytes") or 0) <= 0:
        issues.append("composite_output_is_empty")

    samples: list[CompositeSample] = []
    for overlay in overlays:
        if str(overlay.get("compose_mode") or "").strip().lower() != "replace":
            continue
        visual_id = str(overlay.get("visual_id") or "visual")
        asset_path = str(overlay.get("asset_path") or "").strip()
        start = _finite_float(overlay.get("start"), 0.0)
        end = _finite_float(overlay.get("end"), start)
        duration = max(0.0, end - start)
        if not asset_path or not Path(asset_path).is_file():
            samples.append(
                CompositeSample(
                    visual_id=visual_id,
                    output_time_sec=round(start + duration * 0.5, 3),
                    asset_time_sec=round(duration * 0.5, 3),
                    similarity=0.0,
                    output_brightness=0.0,
                    output_contrast=0.0,
                    passed=False,
                    issues=["composite_qa_asset_missing"],
                )
            )
            continue
        output_time = max(0.0, min(start + duration * 0.5, output_duration))
        asset_time = max(0.0, duration * 0.5)
        output_frame = extractor(output_path, output_time)
        asset_frame = extractor(asset_path, asset_time)
        sample_issues: list[str] = []
        if output_frame is None or len(output_frame) != FRAME_BYTES:
            sample_issues.append("composite_output_frame_unavailable")
        if asset_frame is None or len(asset_frame) != FRAME_BYTES:
            sample_issues.append("composite_asset_frame_unavailable")
        similarity = (
            _frame_similarity(output_frame, asset_frame)
            if not sample_issues
            else 0.0
        )
        brightness, contrast = _luma_metrics(output_frame)
        if not sample_issues and similarity < floor:
            sample_issues.append("composite_visual_not_present_as_rendered")
        if output_frame is not None and (
            (brightness < 0.012 and contrast < 0.012)
            or (brightness > 0.992 and contrast < 0.008)
        ):
            sample_issues.append("composite_frame_is_effectively_blank")
        samples.append(
            CompositeSample(
                visual_id=visual_id,
                output_time_sec=round(output_time, 3),
                asset_time_sec=round(asset_time, 3),
                similarity=round(similarity, 4),
                output_brightness=round(brightness, 4),
                output_contrast=round(contrast, 4),
                passed=not sample_issues,
                issues=sample_issues,
            )
        )

    failed_samples = [sample for sample in samples if not sample.passed]
    if failed_samples:
        issues.append("composite_visual_sample_failed")
    if not samples:
        warnings.append("no_fullscreen_replacement_samples_required")
    metadata_score = sum(
        [
            output_duration > 0.0 and duration_delta <= duration_tolerance,
            resolution_preserved,
            audio_preserved,
            int(output_meta.get("size_bytes") or 0) > 0,
        ]
    ) / 4.0
    sample_score = (
        sum(sample.similarity for sample in samples) / len(samples)
        if samples
        else 1.0
    )
    score = max(0.0, min(metadata_score * 0.45 + sample_score * 0.55, 1.0))
    return CompositeQualityReport(
        version=COMPOSITE_QA_VERSION,
        passed=not issues,
        score=round(score, 4),
        issues=_unique(issues),
        warnings=_unique(warnings),
        metadata_checks={
            "source_duration_sec": round(source_duration, 4),
            "output_duration_sec": round(output_duration, 4),
            "duration_delta_sec": round(duration_delta, 4),
            "duration_tolerance_sec": round(duration_tolerance, 4),
            "resolution_preserved": resolution_preserved,
            "audio_preserved": audio_preserved,
            "source_has_audio": bool(source_meta.get("has_audio")),
            "output_has_audio": bool(output_meta.get("has_audio")),
            "similarity_floor": round(floor, 4),
        },
        samples=samples,
    )


def _extract_fill_frame(video_path: str, time_sec: float) -> bytes | None:
    command = [
        config.FFMPEG_PATH,
        "-v",
        "error",
        "-ss",
        f"{max(float(time_sec), 0.0):.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-vf",
        (
            f"scale={FRAME_WIDTH}:{FRAME_HEIGHT}:force_original_aspect_ratio=increase,"
            f"crop={FRAME_WIDTH}:{FRAME_HEIGHT},setsar=1,format=rgb24"
        ),
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or len(result.stdout) != FRAME_BYTES:
        return None
    return bytes(result.stdout)


def _frame_similarity(left: bytes, right: bytes) -> float:
    if len(left) != FRAME_BYTES or len(right) != FRAME_BYTES:
        return 0.0
    mean_absolute_error = sum(
        abs(left_value - right_value)
        for left_value, right_value in zip(left, right, strict=True)
    ) / FRAME_BYTES
    return max(0.0, min(1.0 - mean_absolute_error / 255.0, 1.0))


def _luma_metrics(frame: bytes | None) -> tuple[float, float]:
    if frame is None or len(frame) != FRAME_BYTES:
        return 0.0, 0.0
    luma = [
        (
            frame[index] * 0.2126
            + frame[index + 1] * 0.7152
            + frame[index + 2] * 0.0722
        )
        / 255.0
        for index in range(0, len(frame), 3)
    ]
    mean = sum(luma) / len(luma)
    variance = sum((value - mean) ** 2 for value in luma) / len(luma)
    return mean, math.sqrt(variance)


def _finite_float(value: object, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


__all__ = [
    "COMPOSITE_QA_VERSION",
    "CompositeQualityReport",
    "CompositeSample",
    "evaluate_visual_composite",
]
