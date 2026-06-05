from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from effects.schema import EffectInstance, EffectPlan


MOTION_PLAN_VERSION = "motion-director-v1"

TASTE_PROFILES: dict[str, dict[str, float | bool | str]] = {
    "clean_documentary": {
        "max_scale_delta": 0.105,
        "max_pan_offset": 0.075,
        "max_zoom_velocity": 0.105,
        "style_strength": 0.74,
        "allow_flash": False,
        "allow_shake": False,
    },
    "viral_commentary": {
        "max_scale_delta": 0.135,
        "max_pan_offset": 0.095,
        "max_zoom_velocity": 0.145,
        "style_strength": 0.92,
        "allow_flash": True,
        "allow_shake": True,
    },
    "tutorial_focus": {
        "max_scale_delta": 0.085,
        "max_pan_offset": 0.055,
        "max_zoom_velocity": 0.085,
        "style_strength": 0.62,
        "allow_flash": False,
        "allow_shake": False,
    },
    "cinematic_subtle": {
        "max_scale_delta": 0.075,
        "max_pan_offset": 0.05,
        "max_zoom_velocity": 0.075,
        "style_strength": 0.58,
        "allow_flash": False,
        "allow_shake": False,
    },
    "high_energy_shorts": {
        "max_scale_delta": 0.15,
        "max_pan_offset": 0.105,
        "max_zoom_velocity": 0.16,
        "style_strength": 0.98,
        "allow_flash": True,
        "allow_shake": True,
    },
}


@dataclass(frozen=True)
class DirectedCameraSegment:
    effect_id: str
    source_card_id: str
    effect_type: str
    start: float
    end: float
    target_scale: float
    anchor_x: float
    anchor_y: float
    pan_x: float
    pan_y: float
    easing: str
    velocity_score: float
    safety_score: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DirectedStyleLayer:
    effect_id: str
    modifier: str
    start: float
    end: float
    opacity: float
    safety_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MotionDirectorReport:
    passed: bool
    score: float
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MotionPlan:
    version: str
    taste_profile: str
    clip_duration: float
    width: int
    height: int
    max_scale: float
    max_pan_offset: float
    max_zoom_velocity: float
    camera_segments: list[DirectedCameraSegment]
    style_layers: list[DirectedStyleLayer]
    qa: MotionDirectorReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "taste_profile": self.taste_profile,
            "clip_duration": round(float(self.clip_duration), 3),
            "width": int(self.width),
            "height": int(self.height),
            "max_scale": round(float(self.max_scale), 4),
            "max_pan_offset": round(float(self.max_pan_offset), 4),
            "max_zoom_velocity": round(float(self.max_zoom_velocity), 4),
            "camera_segments": [segment.to_dict() for segment in self.camera_segments],
            "style_layers": [layer.to_dict() for layer in self.style_layers],
            "qa": self.qa.to_dict(),
        }


def direct_effect_plan(
    plan: EffectPlan,
    *,
    clip_duration: float,
    width: int,
    height: int,
) -> tuple[EffectPlan, MotionPlan]:
    metadata = dict(plan.metadata or {})
    taste_profile = _choose_taste_profile(metadata, clip_duration=clip_duration, width=width, height=height)
    profile = TASTE_PROFILES[taste_profile]
    max_scale_delta = _profile_float(profile, "max_scale_delta")
    max_pan_offset = _profile_float(profile, "max_pan_offset")
    max_zoom_velocity = _profile_float(profile, "max_zoom_velocity")
    base_style_strength = _profile_float(profile, "style_strength")
    allow_flash = bool(profile.get("allow_flash"))
    allow_shake = bool(profile.get("allow_shake"))

    directed_effects: list[EffectInstance] = []
    camera_segments: list[DirectedCameraSegment] = []
    style_layers: list[DirectedStyleLayer] = []
    warnings: list[str] = []
    issues: list[str] = []

    for index, effect in enumerate(plan.effects):
        duration = max(float(effect.end) - float(effect.start), 0.001)
        context_risk = _bounded(effect.params.get("context_risk"), 0.0)
        scene_stability = _bounded(effect.params.get("context_scene_stability"), 0.55)
        role = str(effect.params.get("context_role") or "support")
        effect_type = str(effect.effect_type)
        effective_type = effect_type
        if effect_type == "subtle_shake" and (not allow_shake or context_risk >= 0.42):
            effective_type = "impact_pulse"
            warnings.append(f"{effect.effect_id or index}: converted risky shake into impact_pulse.")
        target_delta = max(0.0, _as_float(effect.params.get("max_scale"), 1.06) - 1.0)
        role_delta_cap = _role_scale_cap(role)
        risk_multiplier = max(0.45, 1.0 - context_risk * 0.65)
        stability_multiplier = 1.0 + max(scene_stability - 0.62, 0.0) * 0.18
        target_delta = min(target_delta * risk_multiplier * stability_multiplier, max_scale_delta, role_delta_cap)
        velocity = target_delta / duration
        if velocity > max_zoom_velocity:
            target_delta = max_zoom_velocity * duration
            warnings.append(f"{effect.effect_id or index}: softened zoom speed to motion-director limit.")
        target_scale = round(1.0 + max(0.018, target_delta), 4)
        anchor_x, anchor_y = _anchor_for_effect(effect, role=role)
        pan_x, pan_y = _pan_for_effect(effective_type, index=index, max_pan_offset=max_pan_offset, context_risk=context_risk)
        easing = _easing_for_effect(effective_type, str(effect.params.get("motion_shape") or ""))
        safety_score = _segment_safety_score(
            context_risk=context_risk,
            scene_stability=scene_stability,
            target_scale=target_scale,
            pan_x=pan_x,
            pan_y=pan_y,
        )
        velocity_score = max(0.0, min(1.0, 1.0 - velocity / max(max_zoom_velocity * 1.4, 0.001)))
        effect_id = effect.effect_id or f"effect_{index + 1:03d}"
        segment = DirectedCameraSegment(
            effect_id=effect_id,
            source_card_id=effect.source_card_id,
            effect_type=effective_type,
            start=round(float(effect.start), 3),
            end=round(float(effect.end), 3),
            target_scale=target_scale,
            anchor_x=anchor_x,
            anchor_y=anchor_y,
            pan_x=round(pan_x, 4),
            pan_y=round(pan_y, 4),
            easing=easing,
            velocity_score=round(velocity_score, 4),
            safety_score=round(safety_score, 4),
            reason=_motion_reason(effect, taste_profile=taste_profile, safety_score=safety_score),
        )
        camera_segments.append(segment)

        modifiers = list(effect.modifiers)
        if not allow_flash:
            modifiers = [modifier for modifier in modifiers if modifier != "flash_accent"]
        if context_risk >= 0.48:
            modifiers = [modifier for modifier in modifiers if modifier not in {"flash_accent", "focus_blur"}]
        style_strength = round(max(0.32, min(base_style_strength * (1.0 - context_risk * 0.28), 1.0)), 3)
        for modifier in modifiers:
            style_layers.append(
                DirectedStyleLayer(
                    effect_id=effect_id,
                    modifier=str(modifier),
                    start=round(float(effect.start), 3),
                    end=round(float(effect.end), 3),
                    opacity=style_strength,
                    safety_score=round(max(0.0, min(safety_score + 0.08, 1.0)), 4),
                )
            )
        params = dict(effect.params)
        params.update(
            {
                "max_scale": target_scale,
                "anchor_x": anchor_x,
                "anchor_y": anchor_y,
                "pan_x": round(pan_x, 4),
                "pan_y": round(pan_y, 4),
                "motion_shape": easing,
                "motion_director_version": MOTION_PLAN_VERSION,
                "motion_taste_profile": taste_profile,
                "motion_safety_score": round(safety_score, 4),
                "style_strength": style_strength,
            }
        )
        directed_effects.append(
            EffectInstance(
                effect_id=effect_id,
                effect_type=effective_type,
                start=effect.start,
                end=effect.end,
                priority=effect.priority,
                source_card_id=effect.source_card_id,
                reason=effect.reason,
                params=params,
                modifiers=modifiers,
                signals=effect.signals,
            )
        )

    qa = _evaluate_motion_plan(
        camera_segments,
        style_layers,
        clip_duration=clip_duration,
        max_scale=1.0 + max_scale_delta,
        max_pan_offset=max_pan_offset,
        max_zoom_velocity=max_zoom_velocity,
        inherited_issues=issues,
        inherited_warnings=warnings,
    )
    motion_plan = MotionPlan(
        version=MOTION_PLAN_VERSION,
        taste_profile=taste_profile,
        clip_duration=clip_duration,
        width=width,
        height=height,
        max_scale=1.0 + max_scale_delta,
        max_pan_offset=max_pan_offset,
        max_zoom_velocity=max_zoom_velocity,
        camera_segments=camera_segments,
        style_layers=style_layers,
        qa=qa,
    )
    directed_metadata = {
        **metadata,
        "motion_director": {
            "version": MOTION_PLAN_VERSION,
            "taste_profile": taste_profile,
            "qa": qa.to_dict(),
        },
        "motion_plan": motion_plan.to_dict(),
    }
    return (
        EffectPlan(
            effects=directed_effects,
            version=plan.version,
            compiler_version=plan.compiler_version,
            source=plan.source,
            metadata=directed_metadata,
        ),
        motion_plan,
    )


def validate_motion_plan_payload(payload: dict[str, Any]) -> MotionDirectorReport:
    segments = [
        item for item in payload.get("camera_segments") or []
        if isinstance(item, dict)
    ]
    layers = [
        item for item in payload.get("style_layers") or []
        if isinstance(item, dict)
    ]
    max_scale = _as_float(payload.get("max_scale"), 1.16)
    max_pan_offset = _as_float(payload.get("max_pan_offset"), 0.1)
    max_zoom_velocity = _as_float(payload.get("max_zoom_velocity"), 0.14)
    issues: list[str] = []
    warnings: list[str] = []
    previous_end = -1.0
    for index, segment in enumerate(sorted(segments, key=lambda item: _as_float(item.get("start"), 0.0)), start=1):
        label = str(segment.get("effect_id") or f"segment_{index:03d}")
        start = _as_float(segment.get("start"), 0.0)
        end = _as_float(segment.get("end"), start)
        target_scale = _as_float(segment.get("target_scale"), 1.0)
        pan_x = abs(_as_float(segment.get("pan_x"), 0.0))
        pan_y = abs(_as_float(segment.get("pan_y"), 0.0))
        if end <= start:
            issues.append(f"{label} has invalid timing.")
            continue
        if start < previous_end + 0.08:
            warnings.append(f"{label} starts close to the previous directed motion segment.")
        previous_end = max(previous_end, end)
        if target_scale > max_scale + 0.002:
            issues.append(f"{label} exceeds directed max scale.")
        if max(pan_x, pan_y) > max_pan_offset + 0.002:
            issues.append(f"{label} exceeds directed pan bounds.")
        velocity = max(0.0, target_scale - 1.0) / max(end - start, 0.001)
        if velocity > max_zoom_velocity * 1.18:
            warnings.append(f"{label} has fast zoom velocity.")
        anchor_x = _as_float(segment.get("anchor_x"), 0.5)
        anchor_y = _as_float(segment.get("anchor_y"), 0.5)
        if not 0.18 <= anchor_x <= 0.82 or not 0.18 <= anchor_y <= 0.82:
            issues.append(f"{label} has unsafe camera anchor.")
    flash_count = sum(1 for layer in layers if str(layer.get("modifier") or "") == "flash_accent")
    if flash_count > 2:
        warnings.append(f"flash_accent appears {flash_count} times in the directed style plan.")
    score = _qa_score(issues=issues, warnings=warnings, segment_count=len(segments))
    return MotionDirectorReport(
        passed=not issues,
        score=score,
        issues=issues,
        warnings=warnings,
        metrics={
            "segment_count": len(segments),
            "style_layer_count": len(layers),
            "flash_count": flash_count,
            "max_scale": round(max_scale, 4),
            "max_pan_offset": round(max_pan_offset, 4),
            "max_zoom_velocity": round(max_zoom_velocity, 4),
        },
    )


def _evaluate_motion_plan(
    segments: list[DirectedCameraSegment],
    layers: list[DirectedStyleLayer],
    *,
    clip_duration: float,
    max_scale: float,
    max_pan_offset: float,
    max_zoom_velocity: float,
    inherited_issues: list[str],
    inherited_warnings: list[str],
) -> MotionDirectorReport:
    payload = {
        "camera_segments": [segment.to_dict() for segment in segments],
        "style_layers": [layer.to_dict() for layer in layers],
        "clip_duration": clip_duration,
        "max_scale": max_scale,
        "max_pan_offset": max_pan_offset,
        "max_zoom_velocity": max_zoom_velocity,
    }
    report = validate_motion_plan_payload(payload)
    issues = [*inherited_issues, *report.issues]
    warnings = [*inherited_warnings, *report.warnings]
    score = _qa_score(issues=issues, warnings=warnings, segment_count=len(segments))
    metrics = dict(report.metrics)
    metrics["average_segment_safety"] = round(
        sum(segment.safety_score for segment in segments) / max(len(segments), 1),
        4,
    )
    return MotionDirectorReport(
        passed=not issues,
        score=score,
        issues=issues,
        warnings=warnings,
        metrics=metrics,
    )


def _choose_taste_profile(
    metadata: dict[str, Any],
    *,
    clip_duration: float,
    width: int,
    height: int,
) -> str:
    requested = str(metadata.get("taste_profile") or metadata.get("effect_taste_profile") or "").strip().lower()
    if requested in TASTE_PROFILES:
        return requested
    rhythm = str(metadata.get("timeline_rhythm") or "").strip().lower()
    density = str(metadata.get("density") or "").strip().lower()
    intensity = _as_float(metadata.get("intensity"), 1.0)
    aspect = width / max(height, 1)
    if aspect < 0.82 or (clip_duration <= 75 and density == "high" and intensity >= 1.1):
        return "high_energy_shorts"
    if rhythm == "fast" or density == "high":
        return "viral_commentary"
    if rhythm == "slow" or intensity <= 0.78:
        return "cinematic_subtle"
    if width >= 1280 and height >= 720 and clip_duration >= 45:
        return "clean_documentary"
    return "tutorial_focus"


def _anchor_for_effect(effect: EffectInstance, *, role: str) -> tuple[float, float]:
    subtitle_position = str(effect.params.get("subtitle_position") or "bottom")
    anchor_x = _bounded(effect.params.get("anchor_x"), 0.5)
    anchor_y = _bounded(effect.params.get("anchor_y"), 0.5)
    if subtitle_position == "bottom":
        anchor_y = min(anchor_y, 0.46)
    elif subtitle_position == "top":
        anchor_y = max(anchor_y, 0.54)
    if role in {"hook", "proof"}:
        anchor_x = 0.5
    elif role == "contrast":
        anchor_x = 0.46
    elif role == "payoff":
        anchor_x = 0.52
    return round(max(0.24, min(anchor_x, 0.76)), 3), round(max(0.26, min(anchor_y, 0.74)), 3)


def _pan_for_effect(
    effect_type: str,
    *,
    index: int,
    max_pan_offset: float,
    context_risk: float,
) -> tuple[float, float]:
    direction = -1.0 if index % 2 else 1.0
    risk_scale = max(0.35, 1.0 - context_risk)
    if effect_type == "micro_pan":
        return direction * max_pan_offset * 0.72 * risk_scale, 0.0
    if effect_type == "snap_reframe":
        return direction * max_pan_offset * 0.94 * risk_scale, 0.0
    if effect_type == "subtle_shake":
        return direction * max_pan_offset * 0.35 * risk_scale, max_pan_offset * 0.18 * risk_scale
    return 0.0, 0.0


def _easing_for_effect(effect_type: str, existing: str) -> str:
    if existing in {"ease_hold", "soft_peak", "soft_hold", "ease_in_out_hold"}:
        return existing
    if effect_type == "impact_pulse":
        return "impact"
    if effect_type == "freeze_accent":
        return "ease_hold"
    if effect_type in {"slow_push", "punch_in", "punch_out"}:
        return "ease_hold"
    if effect_type in {"micro_pan", "snap_reframe"}:
        return "soft_peak"
    return "smooth"


def _segment_safety_score(
    *,
    context_risk: float,
    scene_stability: float,
    target_scale: float,
    pan_x: float,
    pan_y: float,
) -> float:
    score = 0.86
    score -= context_risk * 0.34
    score += max(scene_stability - 0.5, 0.0) * 0.18
    score -= max(target_scale - 1.12, 0.0) * 1.7
    score -= max(abs(pan_x), abs(pan_y)) * 0.75
    return max(0.0, min(score, 1.0))


def _motion_reason(effect: EffectInstance, *, taste_profile: str, safety_score: float) -> str:
    return (
        f"{effect.effect_type} directed with {taste_profile} profile; "
        f"safety={safety_score:.2f}; {effect.reason}"
    )


def _role_scale_cap(role: str) -> float:
    return {
        "hook": 0.13,
        "proof": 0.12,
        "contrast": 0.115,
        "process": 0.095,
        "payoff": 0.105,
        "focus": 0.085,
        "support": 0.072,
    }.get(role, 0.085)


def _qa_score(*, issues: list[str], warnings: list[str], segment_count: int) -> float:
    score = 1.0 - len(issues) * 0.22 - len(warnings) * 0.045
    if segment_count == 0:
        score -= 0.18
    return round(max(0.0, min(score, 1.0)), 4)


def _profile_float(profile: dict[str, Any], key: str) -> float:
    return _as_float(profile.get(key), 0.0)


def _bounded(value: Any, default: float) -> float:
    return max(0.0, min(_as_float(value, default), 1.0))


def _as_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number
