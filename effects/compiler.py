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


def _style_filters(effects: list[EffectInstance]) -> list[str]:
    filters: list[str] = []
    for effect in effects:
        for modifier in [*effect.modifiers, effect.effect_type]:
            filters.extend(_modifier_filters(modifier, effect))
    return filters


def _modifier_filters(modifier: str, effect: EffectInstance) -> list[str]:
    enable = f"enable='{_between_expr(effect.start, effect.end)}'"
    if modifier == "vignette":
        return [
            f"drawbox=x=0:y=0:w=iw:h=ih*0.12:color=black@0.10:t=fill:{enable}",
            f"drawbox=x=0:y=ih*0.88:w=iw:h=ih*0.12:color=black@0.10:t=fill:{enable}",
            f"drawbox=x=0:y=0:w=iw*0.08:h=ih:color=black@0.08:t=fill:{enable}",
            f"drawbox=x=iw*0.92:y=0:w=iw*0.08:h=ih:color=black@0.08:t=fill:{enable}",
        ]
    if modifier == "focus_blur":
        return [f"eq=contrast=1.035:saturation=1.018:{enable}"]
    if modifier == "flash_accent":
        flash_end = min(float(effect.start) + max(min(effect.duration * 0.22, 0.18), 0.06), float(effect.end))
        return [
            (
                "drawbox=x=0:y=0:w=iw:h=ih:color=white@0.10:t=fill:"
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
                f"color=black@0.18:t=fill:{enable}"
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


def _effect_scale(effect: EffectInstance) -> float:
    return _param(effect, "max_scale", 1.0)
