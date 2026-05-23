from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from effects.schema import EffectInstance, EffectPlan
from smartie.schema import SmartieAttentionPoint, SmartieBundle


@dataclass
class _Candidate:
    start: float
    end: float
    x: float
    y: float
    score: float
    cue: str
    source_count: int = 1

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def plan_smartie_attention_effects(
    bundle_or_points: SmartieBundle | Iterable[SmartieAttentionPoint],
    *,
    duration: float | None = None,
    width: int | None = None,
    height: int | None = None,
    fps: float | None = None,
    max_segments: int = 28,
    min_shot_duration: float = 0.8,
    max_shot_duration: float = 3.2,
    max_zoom_speed_per_sec: float = 0.92,
) -> EffectPlan:
    if isinstance(bundle_or_points, SmartieBundle):
        bundle = bundle_or_points
        points = list(bundle.attention_points)
        metadata = bundle.manifest.declared_metadata
        duration = duration if duration is not None else _optional_float(metadata.get("duration_sec"))
        width = width if width is not None else _optional_int(metadata.get("width"))
        height = height if height is not None else _optional_int(metadata.get("height"))
        fps = fps if fps is not None else _optional_float(metadata.get("fps"))
    else:
        bundle = None
        points = list(bundle_or_points)

    clip_duration = _resolve_duration(points, duration)
    if clip_duration <= 0.0:
        return _empty_plan(points, duration=clip_duration, reason="missing_duration")

    smoothed = _smooth_points(points)
    candidates = _build_candidates(
        smoothed,
        clip_duration=clip_duration,
        min_shot_duration=max(0.35, min_shot_duration),
        max_shot_duration=max(max_shot_duration, min_shot_duration),
    )
    if not candidates:
        return _empty_plan(points, duration=clip_duration, reason="no_useful_attention")

    candidates = _merge_candidates(candidates)
    candidates = _resolve_overlaps(candidates, min_gap_sec=0.12, min_duration_sec=min_shot_duration)
    candidates.sort(key=lambda candidate: (-candidate.score, candidate.start))
    candidates = sorted(candidates[: max(1, max_segments)], key=lambda candidate: candidate.start)

    effects: list[EffectInstance] = []
    previous_scale = 1.0
    for index, candidate in enumerate(candidates, start=1):
        if candidate.duration < min_shot_duration - 0.001:
            continue
        scale = _target_scale(
            candidate.score,
            candidate.duration,
            max_zoom_speed_per_sec=max_zoom_speed_per_sec,
            previous_scale=previous_scale,
        )
        if scale < 1.045:
            continue
        focus_x, focus_y = _bounded_focus(candidate.x, candidate.y, scale=scale)
        previous_scale = scale
        effects.append(
            EffectInstance(
                effect_id=f"smart_zoom_{index:03d}",
                effect_type="smart_zoom_segment",
                start=round(candidate.start, 3),
                end=round(candidate.end, 3),
                priority=round(candidate.score, 3),
                source_card_id=f"smartie_attention_{index:03d}",
                reason=_reason_for_candidate(candidate),
                params={
                    "focus_x": round(focus_x, 5),
                    "focus_y": round(focus_y, 5),
                    "target_scale": round(scale, 4),
                    "max_scale": round(scale, 4),
                    "easing": "ease_in_out_hold",
                    "smoothing": "intent_aware",
                    "confidence": round(candidate.score, 3),
                    "cue": candidate.cue,
                    "source_points": candidate.source_count,
                },
                signals={
                    "smartie_cue": candidate.cue,
                    "attention_score": round(candidate.score, 3),
                    "source_points": candidate.source_count,
                    "fps": fps,
                    "source_width": width,
                    "source_height": height,
                },
            )
        )

    return EffectPlan(
        effects=effects,
        source="smartie_attention",
        metadata={
            "planner": "smartie_attention_v1",
            "input_points": len(points),
            "candidate_segments": len(candidates),
            "clip_duration_sec": round(clip_duration, 3),
            "min_shot_duration_sec": round(min_shot_duration, 3),
            "max_zoom_speed_per_sec": round(max_zoom_speed_per_sec, 3),
            "source_bundle": str(bundle.root) if bundle is not None else None,
        },
    )


def _empty_plan(
    points: list[SmartieAttentionPoint],
    *,
    duration: float,
    reason: str,
) -> EffectPlan:
    return EffectPlan(
        effects=[],
        source="smartie_attention",
        metadata={
            "planner": "smartie_attention_v1",
            "input_points": len(points),
            "candidate_segments": 0,
            "clip_duration_sec": round(max(duration, 0.0), 3),
            "empty_reason": reason,
        },
    )


def _resolve_duration(points: list[SmartieAttentionPoint], duration: float | None) -> float:
    if duration is not None and math.isfinite(duration) and duration > 0:
        return float(duration)
    if not points:
        return 0.0
    return max(point.time + max(point.duration, 0.0) for point in points) + 0.5


def _smooth_points(points: list[SmartieAttentionPoint]) -> list[SmartieAttentionPoint]:
    smoothed: list[SmartieAttentionPoint] = []
    last_x: float | None = None
    last_y: float | None = None
    for point in sorted(points, key=lambda candidate: candidate.time):
        if point.x is None or point.y is None:
            smoothed.append(point)
            continue
        cue = point.cue.lower()
        alpha = 0.68 if cue in {"click", "keyboard"} else 0.42 if cue == "dwell" else 0.32
        if last_x is None or last_y is None:
            next_x, next_y = point.x, point.y
        else:
            next_x = (alpha * point.x) + ((1.0 - alpha) * last_x)
            next_y = (alpha * point.y) + ((1.0 - alpha) * last_y)
        last_x, last_y = next_x, next_y
        smoothed.append(
            SmartieAttentionPoint(
                time=point.time,
                x=max(0.0, min(next_x, 1.0)),
                y=max(0.0, min(next_y, 1.0)),
                confidence=point.confidence,
                cue=point.cue,
                duration=point.duration,
                raw=point.raw,
            )
        )
    return smoothed


def _build_candidates(
    points: list[SmartieAttentionPoint],
    *,
    clip_duration: float,
    min_shot_duration: float,
    max_shot_duration: float,
) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    for point in points:
        if point.x is None or point.y is None:
            continue
        score = _attention_score(point)
        if score < 0.34:
            continue
        cue = point.cue.lower().strip() or "attention"
        base_duration = {
            "click": 1.22,
            "keyboard": 1.08,
            "dwell": 1.7,
            "motion": 0.9,
        }.get(cue, 1.05)
        if point.duration > 0:
            base_duration = max(base_duration, min(point.duration + 0.24, max_shot_duration))
        shot_duration = max(min_shot_duration, min(base_duration + score * 0.42, max_shot_duration))
        pre_roll = 0.16 if cue == "click" else 0.1
        start = max(0.0, point.time - pre_roll)
        end = min(clip_duration, start + shot_duration)
        if end - start < min_shot_duration and end >= clip_duration:
            start = max(0.0, end - min_shot_duration)
        if end - start < min_shot_duration:
            continue
        candidates.append(
            _Candidate(
                start=round(start, 3),
                end=round(end, 3),
                x=max(0.0, min(point.x, 1.0)),
                y=max(0.0, min(point.y, 1.0)),
                score=score,
                cue=cue,
            )
        )
    return candidates


def _attention_score(point: SmartieAttentionPoint) -> float:
    cue = point.cue.lower().strip()
    score = max(0.0, min(float(point.confidence), 1.0))
    score += {
        "click": 0.32,
        "keyboard": 0.24,
        "dwell": 0.18,
        "attention": 0.08,
        "focus": 0.08,
        "motion": -0.04,
    }.get(cue, 0.02)
    if point.duration >= 0.8:
        score += 0.08
    return round(max(0.0, min(score, 1.0)), 3)


def _merge_candidates(candidates: list[_Candidate]) -> list[_Candidate]:
    if not candidates:
        return []
    merged: list[_Candidate] = []
    for candidate in sorted(candidates, key=lambda item: item.start):
        if not merged:
            merged.append(candidate)
            continue
        previous = merged[-1]
        gap = candidate.start - previous.end
        distance = math.dist((candidate.x, candidate.y), (previous.x, previous.y))
        if gap <= 0.46 and distance <= 0.18:
            total = previous.source_count + candidate.source_count
            prev_weight = previous.score * previous.source_count
            next_weight = candidate.score * candidate.source_count
            weight_sum = max(prev_weight + next_weight, 0.001)
            previous.end = max(previous.end, candidate.end)
            previous.x = ((previous.x * prev_weight) + (candidate.x * next_weight)) / weight_sum
            previous.y = ((previous.y * prev_weight) + (candidate.y * next_weight)) / weight_sum
            previous.score = max(previous.score, candidate.score, (previous.score + candidate.score) / 2.0)
            previous.source_count = total
            if _cue_rank(candidate.cue) > _cue_rank(previous.cue):
                previous.cue = candidate.cue
            continue
        merged.append(candidate)
    return merged


def _resolve_overlaps(
    candidates: list[_Candidate],
    *,
    min_gap_sec: float,
    min_duration_sec: float,
) -> list[_Candidate]:
    resolved: list[_Candidate] = []
    for candidate in sorted(candidates, key=lambda item: item.start):
        if not resolved:
            resolved.append(candidate)
            continue
        previous = resolved[-1]
        if candidate.start >= previous.end + min_gap_sec:
            resolved.append(candidate)
            continue
        distance = math.dist((candidate.x, candidate.y), (previous.x, previous.y))
        if distance <= 0.14:
            previous.end = max(previous.end, candidate.end)
            previous.score = max(previous.score, candidate.score)
            previous.source_count += candidate.source_count
            continue
        if candidate.score <= previous.score:
            trimmed_previous_end = min(previous.end, candidate.start - min_gap_sec)
            if trimmed_previous_end - previous.start >= min_duration_sec:
                previous.end = round(trimmed_previous_end, 3)
                resolved.append(candidate)
            continue
        shifted_start = previous.end + min_gap_sec
        if candidate.end - shifted_start >= min_duration_sec:
            candidate.start = round(shifted_start, 3)
            resolved.append(candidate)
    return [candidate for candidate in resolved if candidate.end - candidate.start >= min_duration_sec]


def _target_scale(
    score: float,
    duration: float,
    *,
    max_zoom_speed_per_sec: float,
    previous_scale: float,
) -> float:
    scale = 1.08 + (0.76 * max(0.0, min(score, 1.0)))
    ramp_window = max(min(duration * 0.28, 0.72), 0.22)
    speed_limited = 1.0 + (max_zoom_speed_per_sec * ramp_window)
    scale = min(scale, speed_limited, 2.15)
    if abs(scale - previous_scale) > 0.55:
        scale = previous_scale + (0.55 if scale > previous_scale else -0.55)
    return round(max(1.0, scale), 4)


def _bounded_focus(x: float, y: float, *, scale: float) -> tuple[float, float]:
    visible = 1.0 / max(scale, 1.0)
    x_margin = min(0.5, visible / 2.0)
    y_margin = min(0.5, visible / 2.0)
    return (
        max(x_margin, min(float(x), 1.0 - x_margin)),
        max(y_margin, min(float(y), 1.0 - y_margin)),
    )


def _reason_for_candidate(candidate: _Candidate) -> str:
    cue = candidate.cue.replace("_", " ")
    return (
        f"Smartie {cue} telemetry held attention near "
        f"{candidate.x:.2f},{candidate.y:.2f} with confidence {candidate.score:.2f}."
    )


def _cue_rank(cue: str) -> int:
    return {"click": 5, "keyboard": 4, "dwell": 3, "attention": 2, "focus": 2, "motion": 1}.get(cue, 0)


def _optional_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _optional_int(value: object) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result
