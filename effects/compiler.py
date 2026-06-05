from __future__ import annotations

import math

from effects.schema import EffectInstance, EffectPlan


def build_effect_filter_graph(
    plan: EffectPlan,
    *,
    duration: float,
    width: int,
    height: int,
    fps: float,
    has_audio: bool,
) -> str:
    effects = [effect for effect in plan.effects if effect.end > effect.start]
    if not effects:
        raise ValueError("Effect plan does not contain any renderable effects.")
    fps_value = max(15, int(math.ceil(fps or 30.0)))
    motion_segments = _motion_plan_segments(plan)
    if motion_segments:
        scale_expr = _motion_scale_expr(motion_segments)
        x_expr = _bounded_crop_position_expr("x", _motion_crop_position_expr(motion_segments, axis="x"))
        y_expr = _bounded_crop_position_expr("y", _motion_crop_position_expr(motion_segments, axis="y"))
    else:
        scale_expr = _combined_scale_expr(effects)
        x_expr = _bounded_crop_expr("x", _combined_crop_offset_expr(effects, axis="x"))
        y_expr = _bounded_crop_expr("y", _combined_crop_offset_expr(effects, axis="y"))
    filters = [
        (
            f"[0:v]fps={fps_value},"
            f"scale=w='max({width}\\,trunc(iw*({scale_expr})/2)*2)':"
            f"h='max({height}\\,trunc(ih*({scale_expr})/2)*2)':eval=frame,"
            f"crop={width}:{height}:x='{x_expr}':y='{y_expr}',setsar=1"
        )
    ]
    filters.extend(_style_filters(effects))
    return ",".join(filters) + "[v]"


def _motion_plan_segments(plan: EffectPlan) -> list[dict]:
    motion_plan = plan.metadata.get("motion_plan") if isinstance(plan.metadata, dict) else None
    if not isinstance(motion_plan, dict):
        return []
    if str(motion_plan.get("version") or "") != "motion-director-v1":
        return []
    segments = [
        dict(item)
        for item in motion_plan.get("camera_segments") or []
        if isinstance(item, dict)
    ]
    return [item for item in segments if _as_float(item.get("end"), 0.0) > _as_float(item.get("start"), 0.0)]


def _motion_scale_expr(segments: list[dict]) -> str:
    terms: list[str] = []
    for segment in segments:
        target_scale = _as_float(segment.get("target_scale"), 1.0)
        delta = max(0.0, target_scale - 1.0)
        if delta <= 0.0001:
            continue
        terms.append(f"{delta:.5f}*({_motion_segment_shape_expr(segment)})")
    if not terms:
        return "1"
    return f"1+{_max_expr(terms)}"


def _motion_crop_position_expr(segments: list[dict], *, axis: str) -> str:
    terms: list[str] = []
    anchor_key = "anchor_x" if axis == "x" else "anchor_y"
    pan_key = "pan_x" if axis == "x" else "pan_y"
    for segment in segments:
        shape = _motion_segment_shape_expr(segment)
        anchor_offset = _as_float(segment.get(anchor_key), 0.5) - 0.5
        pan = _as_float(segment.get(pan_key), 0.0)
        if abs(anchor_offset) > 0.0001:
            terms.append(f"{anchor_offset:.5f}*({shape})")
        if abs(pan) > 0.0001:
            terms.append(f"{pan:.5f}*({_motion_segment_pan_shape_expr(segment)})")
    if not terms:
        return "0.5"
    return f"min(max(0.5+{'+'.join(terms)}\\,0.0)\\,1.0)"


def _motion_segment_shape_expr(segment: dict) -> str:
    start = _as_float(segment.get("start"), 0.0)
    end = _as_float(segment.get("end"), start)
    duration = max(end - start, 0.1)
    u = f"((t-{start:.5f})/{duration:.5f})"
    between = _between_expr(start, end)
    easing = str(segment.get("easing") or "smooth").strip().lower()
    if easing == "impact":
        body = f"pow(sin(PI*{u})\\,2)"
    elif easing in {"ease_hold", "soft_hold", "ease_in_out_hold"}:
        body = f"min(1\\,min(max(0\\,{u}/0.18)\\,max(0\\,(1-{u})/0.18)))"
    elif easing == "soft_peak":
        body = f"(0.5-0.5*cos(2*PI*{u}))"
    else:
        body = f"(0.5-0.5*cos(2*PI*{u}))"
    return f"if({between}\\,{body}\\,0)"


def _motion_segment_pan_shape_expr(segment: dict) -> str:
    start = _as_float(segment.get("start"), 0.0)
    end = _as_float(segment.get("end"), start)
    duration = max(end - start, 0.1)
    u = f"((t-{start:.5f})/{duration:.5f})"
    between = _between_expr(start, end)
    effect_type = str(segment.get("effect_type") or "")
    if effect_type == "micro_pan":
        body = f"sin(PI*{u})"
    elif effect_type == "snap_reframe":
        body = f"(0.5-0.5*cos(2*PI*{u}))"
    elif effect_type == "subtle_shake":
        body = f"sin(PI*{u})*sin(28*t)"
    else:
        body = _motion_segment_shape_expr(segment)
        return body
    return f"if({between}\\,{body}\\,0)"


def _max_expr(terms: list[str]) -> str:
    if not terms:
        return "0"
    expr = terms[0]
    for term in terms[1:]:
        expr = f"max({expr}\\,{term})"
    return expr


def _combined_scale_expr(effects: list[EffectInstance]) -> str:
    deltas: list[str] = []
    for effect in effects:
        max_scale = _effect_scale(effect)
        delta = max(0.0, max_scale - 1.0)
        if delta <= 0.0001:
            continue
        deltas.append(f"{delta:.5f}*({_motion_shape_expr(effect)})")
    if not deltas:
        return "1"
    return "1+" + "+".join(deltas)


def _combined_crop_offset_expr(effects: list[EffectInstance], *, axis: str) -> str:
    offsets: list[str] = []
    for index, effect in enumerate(effects):
        offset = _crop_offset_expr(effect, axis=axis, direction=-1 if index % 2 else 1)
        if offset:
            offsets.append(offset)
    if not offsets:
        return "0"
    return "+".join(offsets)


def _motion_shape_expr(effect: EffectInstance) -> str:
    motion_shape = str(effect.params.get("motion_shape") or "").strip().lower()
    if motion_shape in {"ease_hold", "ease_in_out_hold", "soft_hold"}:
        return _window_shape(effect, "ease_hold")
    if motion_shape in {"soft_peak", "gentle_peak"}:
        return _window_shape(effect, "soft_peak")
    if effect.effect_type == "impact_pulse":
        return _window_shape(effect, "impact")
    if effect.effect_type == "subtle_shake":
        return _window_shape(effect, "shake")
    if effect.effect_type == "freeze_accent":
        return _window_shape(effect, "hold")
    return _window_shape(effect, "smooth")


def _window_shape(effect: EffectInstance, mode: str) -> str:
    start = float(effect.start)
    end = float(effect.end)
    duration = max(end - start, 0.1)
    u = f"((t-{start:.5f})/{duration:.5f})"
    between = _between_expr(start, end)
    if mode == "impact":
        body = f"pow(sin(PI*{u})\\,2)"
    elif mode == "shake":
        body = f"(0.5-0.5*cos(2*PI*{u}))"
    elif mode == "hold":
        body = f"min(1\\,sin(PI*{u})*1.35)"
    elif mode == "ease_hold":
        body = f"min(1\\,min(max(0\\,{u}/0.18)\\,max(0\\,(1-{u})/0.18)))"
    elif mode == "soft_peak":
        body = f"(0.5-0.5*cos(2*PI*{u}))"
    else:
        body = f"(0.5-0.5*cos(2*PI*{u}))"
    return f"if({between}\\,{body}\\,0)"


def _crop_offset_expr(effect: EffectInstance, *, axis: str, direction: int) -> str:
    start = float(effect.start)
    end = float(effect.end)
    duration = max(end - start, 0.1)
    u = f"((t-{start:.5f})/{duration:.5f})"
    between = _between_expr(start, end)
    sign = -1 if direction < 0 else 1
    if effect.effect_type == "micro_pan" and axis == "x":
        return f"{0.14 * sign:.5f}*if({between}\\,sin(2*PI*{u})\\,0)"
    if effect.effect_type == "snap_reframe" and axis == "x":
        return f"{0.18 * sign:.5f}*if({between}\\,sin(PI*{u})\\,0)"
    if effect.effect_type == "subtle_shake":
        amp = _param(effect, "shake_amplitude", 0.055) * (0.75 if axis == "y" else 1.0)
        frequency = 43 if axis == "y" else 55
        return f"{amp:.5f}*if({between}\\,sin(PI*{u})*sin({frequency}*t)\\,0)"
    return ""


def _bounded_crop_expr(axis: str, offset_expr: str) -> str:
    base = f"(in_{'w' if axis == 'x' else 'h'}-out_{'w' if axis == 'x' else 'h'})"
    return f"min(max({base}*(0.5+({offset_expr}))\\,0)\\,{base})"


def _bounded_crop_position_expr(axis: str, position_expr: str) -> str:
    base = f"(in_{'w' if axis == 'x' else 'h'}-out_{'w' if axis == 'x' else 'h'})"
    return f"min(max({base}*({position_expr})\\,0)\\,{base})"


def _style_filters(effects: list[EffectInstance]) -> list[str]:
    filters: list[str] = []
    for effect in effects:
        for modifier in [*effect.modifiers, effect.effect_type]:
            filters.extend(_modifier_filters(modifier, effect))
    return filters


def _modifier_filters(modifier: str, effect: EffectInstance) -> list[str]:
    enable = f"enable='{_between_expr(effect.start, effect.end)}'"
    style_strength = max(0.0, min(_param(effect, "style_strength", 1.0), 1.0))
    if modifier == "vignette":
        vertical_opacity = 0.09 * style_strength
        horizontal_opacity = 0.065 * style_strength
        return [
            f"drawbox=x=0:y=0:w=iw:h=ih*0.12:color=black@{vertical_opacity:.3f}:t=fill:{enable}",
            f"drawbox=x=0:y=ih*0.88:w=iw:h=ih*0.12:color=black@{vertical_opacity:.3f}:t=fill:{enable}",
            f"drawbox=x=0:y=0:w=iw*0.08:h=ih:color=black@{horizontal_opacity:.3f}:t=fill:{enable}",
            f"drawbox=x=iw*0.92:y=0:w=iw*0.08:h=ih:color=black@{horizontal_opacity:.3f}:t=fill:{enable}",
        ]
    if modifier == "focus_blur":
        contrast = 1.0 + 0.026 * style_strength
        saturation = 1.0 + 0.014 * style_strength
        return [f"eq=contrast={contrast:.3f}:saturation={saturation:.3f}:{enable}"]
    if modifier == "flash_accent":
        flash_end = min(float(effect.start) + max(min(effect.duration * 0.22, 0.18), 0.06), float(effect.end))
        opacity = 0.075 * style_strength
        return [
            (
                f"drawbox=x=0:y=0:w=iw:h=ih:color=white@{opacity:.3f}:t=fill:"
                f"enable='{_between_expr(effect.start, flash_end)}'"
            )
        ]
    if modifier == "subtitle_highlight" and bool(effect.params.get("subtitle_highlight_enabled")):
        position = str(effect.params.get("subtitle_position") or "bottom")
        if position == "top":
            y_expr = "ih*0.095"
        elif position == "center":
            y_expr = "ih*0.425"
        else:
            y_expr = "ih*0.785"
        return [
            (
                f"drawbox=x=iw*0.08:y={y_expr}:w=iw*0.84:h=ih*0.13:"
                f"color=black@{0.14 * style_strength:.3f}:t=fill:{enable}"
            )
        ]
    return []


def _between_expr(start: float, end: float) -> str:
    return f"between(t\\,{float(start):.5f}\\,{float(end):.5f})"


def _param(effect: EffectInstance, key: str, default: float) -> float:
    try:
        return float(effect.params.get(key, default))
    except (TypeError, ValueError):
        return default


def _as_float(value: object, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number


def _effect_scale(effect: EffectInstance) -> float:
    return _param(effect, "max_scale", 1.0)
