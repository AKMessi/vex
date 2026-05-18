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
    boundaries = sorted({0.0, round(duration, 3), *[round(effect.start, 3) for effect in effects], *[round(effect.end, 3) for effect in effects]})
    segments = [
        (boundaries[index], boundaries[index + 1])
        for index in range(len(boundaries) - 1)
        if boundaries[index + 1] - boundaries[index] > 0.02
    ]
    if not segments:
        raise ValueError("Effect plan produced no renderable timeline segments.")

    filter_parts: list[str] = []
    filter_parts.append(f"[0:v]split={len(segments)}" + "".join(f"[v{index}]" for index in range(len(segments))))
    if has_audio:
        filter_parts.append(f"[0:a]asplit={len(segments)}" + "".join(f"[a{index}]" for index in range(len(segments))))

    concat_inputs: list[str] = []
    for index, (start_sec, end_sec) in enumerate(segments):
        effect = _active_effect(effects, start_sec, end_sec)
        segment_duration = end_sec - start_sec
        video_filter = f"[v{index}]trim={start_sec:.3f}:{end_sec:.3f},setpts=PTS-STARTPTS,fps={fps_value}"
        if effect is None:
            video_filter += _identity_filters(width, height)
        else:
            video_filter += _effect_filters(effect, segment_duration, width, height, fps_value)
        video_filter += f"[v{index}o]"
        filter_parts.append(video_filter)
        concat_inputs.append(f"[v{index}o]")
        if has_audio:
            filter_parts.append(f"[a{index}]atrim={start_sec:.3f}:{end_sec:.3f},asetpts=PTS-STARTPTS[a{index}o]")
            concat_inputs.append(f"[a{index}o]")

    filter_parts.append(
        f"{''.join(concat_inputs)}concat=n={len(segments)}:v=1:a={'1' if has_audio else '0'}[v]"
        + ("[a]" if has_audio else "")
    )
    return ";".join(filter_parts)


def _active_effect(effects: list[EffectInstance], start_sec: float, end_sec: float) -> EffectInstance | None:
    for effect in effects:
        if start_sec >= effect.start - 0.001 and end_sec <= effect.end + 0.001:
            return effect
    return None


def _identity_filters(width: int, height: int) -> str:
    return (
        f",scale={width}:{height}:force_original_aspect_ratio=decrease"
        f",pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1"
    )


def _effect_filters(effect: EffectInstance, segment_duration: float, width: int, height: int, fps: int) -> str:
    filters = ""
    if effect.effect_type == "freeze_accent":
        frame_count = max(2, int(math.ceil(segment_duration * fps)))
        filters += (
            ",select='eq(n\\,0)'"
            f",loop=loop={frame_count}:size=1:start=0"
            f",setpts=N/({fps}*TB),trim=start=0:duration={segment_duration:.3f},fps={fps}"
        )
    scale_expr = _scale_expr(effect, segment_duration)
    filters += (
        f",scale=w='trunc(iw*({scale_expr})/2)*2':h='trunc(ih*({scale_expr})/2)*2':eval=frame"
        f",crop={width}:{height}:x='{_crop_x_expr(effect, segment_duration)}':y='{_crop_y_expr(effect, segment_duration)}'"
        ",setsar=1"
    )
    for modifier in effect.modifiers:
        modifier_filter = _modifier_filter(modifier, segment_duration, effect.params)
        if modifier_filter:
            filters += f",{modifier_filter}"
    if effect.effect_type in {"vignette", "focus_blur", "flash_accent", "subtitle_highlight"}:
        own_filter = _modifier_filter(effect.effect_type, segment_duration, effect.params)
        if own_filter:
            filters += f",{own_filter}"
    return filters


def _scale_expr(effect: EffectInstance, duration: float) -> str:
    max_scale = _param(effect, "max_scale", 1.08)
    delta = max_scale - 1.0
    d = max(duration, 0.1)
    ramp = max(min(d * 0.35, 0.32), 0.08)
    if effect.effect_type == "punch_out":
        return f"max(1\\,{max_scale:.5f}-{delta:.5f}*min(t/{ramp:.5f}\\,1))"
    if effect.effect_type == "slow_push":
        return f"1+{delta:.5f}*(t/{d:.5f})"
    if effect.effect_type == "impact_pulse":
        return f"1+{delta:.5f}*sin(PI*t/{d:.5f})"
    if effect.effect_type in {"micro_pan", "snap_reframe", "subtle_shake", "freeze_accent"}:
        return f"{max_scale:.5f}"
    return f"min({max_scale:.5f}\\,1+{delta:.5f}*min(t/{ramp:.5f}\\,1))"


def _crop_x_expr(effect: EffectInstance, duration: float) -> str:
    center = "(in_w-out_w)/2"
    d = max(duration, 0.1)
    if effect.effect_type == "micro_pan":
        return f"(in_w-out_w)*(0.50+0.16*sin(PI*t/{d:.5f}))"
    if effect.effect_type == "snap_reframe":
        return f"(in_w-out_w)*if(lt(t\\,{d * 0.42:.5f})\\,0.38\\,0.62)"
    if effect.effect_type == "subtle_shake":
        amp = _param(effect, "shake_amplitude", 0.07)
        return f"(in_w-out_w)*(0.5+{amp:.5f}*sin(55*t))"
    return center


def _crop_y_expr(effect: EffectInstance, duration: float) -> str:
    center = "(in_h-out_h)/2"
    if effect.effect_type == "subtle_shake":
        amp = _param(effect, "shake_amplitude", 0.07) * 0.65
        return f"(in_h-out_h)*(0.5+{amp:.5f}*sin(43*t))"
    return center


def _modifier_filter(modifier: str, duration: float, params: dict) -> str:
    if modifier == "vignette":
        return "vignette=PI/5:eval=frame"
    if modifier == "focus_blur":
        return "unsharp=5:5:0.65:3:3:0.25"
    if modifier == "flash_accent":
        flash_duration = min(max(duration * 0.28, 0.08), 0.22)
        return f"drawbox=x=0:y=0:w=iw:h=ih:color=white@0.16:t=fill:enable='between(t\\,0\\,{flash_duration:.3f})'"
    if modifier == "subtitle_highlight":
        position = str(params.get("subtitle_position") or "bottom")
        if position == "top":
            y_expr = "ih*0.095"
        elif position == "center":
            y_expr = "ih*0.425"
        else:
            y_expr = "ih*0.785"
        return f"drawbox=x=iw*0.08:y={y_expr}:w=iw*0.84:h=ih*0.13:color=yellow@0.14:t=fill"
    return ""


def _param(effect: EffectInstance, key: str, default: float) -> float:
    try:
        return float(effect.params.get(key, default))
    except (TypeError, ValueError):
        return default
