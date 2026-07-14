from __future__ import annotations

import html
import math
from dataclasses import dataclass
from typing import Any

from vex_hyperframes.safety import validate_authored_html_safety
from vex_visuals.open_visual_program import validate_open_visual_program


@dataclass(frozen=True)
class CompiledOpenVisualStage:
    html: str
    metadata: dict[str, Any]


def compile_open_visual_stage(
    program: dict[str, Any],
    *,
    ir: dict[str, Any],
) -> CompiledOpenVisualStage:
    validation = validate_open_visual_program(program, ir=ir)
    if not validation.passed:
        raise ValueError(
            "Unsafe Open Visual Program: " + "; ".join(validation.errors)
        )
    elements = [
        dict(item)
        for item in program.get("elements") or []
        if isinstance(item, dict)
    ]
    tracks = [
        dict(item)
        for item in program.get("tracks") or []
        if isinstance(item, dict)
    ]
    relations = [
        dict(item)
        for item in program.get("relations") or []
        if isinstance(item, dict)
    ]
    palette = dict(program.get("palette") or {})
    canvas = dict(program.get("canvas") or {})
    canvas_width = max(320, int(_number(canvas.get("width"), 1280)))
    canvas_height = max(180, int(_number(canvas.get("height"), 720)))
    tracks_by_target: dict[str, list[dict[str, Any]]] = {}
    for item in tracks:
        tracks_by_target.setdefault(str(item.get("target_id") or ""), []).append(
            item
        )
    element_markup = "\n".join(
        _element_markup(
            item,
            tracks_by_target.get(str(item.get("element_id") or ""), []),
            palette=palette,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
        )
        for item in elements
    )
    relation_markup = _relation_markup(relations, elements)
    concept = dict(program.get("concept") or {})
    background = _palette_color(palette, "background", "#F4F0E8")
    surface = _palette_color(palette, "surface", "#FFFDF8")
    ink = _palette_color(palette, "ink", "#111111")
    muted = _palette_color(palette, "muted", "#4B4740")
    accent = _palette_color(palette, "accent", "#F04438")
    accent_secondary = _palette_color(
        palette,
        "accent_secondary",
        "#1E5EFF",
    )
    grid = _palette_color(palette, "grid", "#C8C0B4")
    canvas_text = _contrast_foreground(background, ink, surface)
    surface_text = _contrast_foreground(surface, ink, background)
    visible_labels = [
        str(item.get("text") or "").strip()
        for item in elements
        if str(item.get("text") or "").strip()
    ]
    semantic_object_ids = sorted(
        {
            str((item.get("binding") or {}).get("id") or "")
            for item in elements
            if str((item.get("binding") or {}).get("kind") or "") == "object"
            and str((item.get("binding") or {}).get("id") or "")
        }
    )
    semantic_relation_ids = sorted(
        {
            str((item.get("binding") or {}).get("id") or "")
            for item in relations
            if str((item.get("binding") or {}).get("kind") or "")
            == "relation"
            and str((item.get("binding") or {}).get("id") or "")
        }
    )
    fragment = f"""
      <style>
        .ovp-stage {{ position:absolute; inset:0; overflow:hidden; background:{background}; color:{canvas_text}; --ovp-bg:{background}; --ovp-surface:{surface}; --ovp-ink:{ink}; --ovp-muted:{muted}; --ovp-accent:{accent}; --ovp-accent-2:{accent_secondary}; --ovp-grid:{grid}; --ovp-canvas-text:{canvas_text}; --ovp-surface-text:{surface_text}; }}
        .ovp-stage::before {{ content:""; position:absolute; inset:0; opacity:.22; background-image:linear-gradient(var(--ovp-grid) 1px,transparent 1px),linear-gradient(90deg,var(--ovp-grid) 1px,transparent 1px); background-size:58px 58px; transform:translate3d(calc(var(--p,0) * -28px),calc(var(--p,0) * -16px),0); }}
        .ovp-element {{ position:absolute; box-sizing:border-box; display:grid; place-items:center; text-align:center; line-height:1.06; overflow:hidden; transform-origin:center; will-change:transform,opacity; }}
        .ovp-actor {{ position:absolute; inset:0; display:grid; place-items:center; padding:inherit; text-align:inherit; }}
        .ovp-element.ovp-text {{ overflow:visible; text-align:left; place-items:center start; }}
        .ovp-element.ovp-source-signal::before {{ content:""; position:absolute; inset:20%; background:repeating-linear-gradient(180deg,var(--ovp-accent-2) 0 4px,transparent 4px 11px); opacity:.72; }}
        .ovp-element.ovp-transformation-gate {{ clip-path:polygon(0 0,100% 12%,72% 88%,0 100%); }}
        .ovp-element.ovp-transformation-gate::before {{ content:""; width:46%; aspect-ratio:1; border:6px solid currentColor; border-radius:50%; box-shadow:0 0 28px color-mix(in srgb,var(--ovp-accent-2) 62%,transparent); transform:rotate(calc(var(--route-progress,0) * 180deg)); }}
        .ovp-element.ovp-transformation-gate::after {{ content:""; position:absolute; width:18%; aspect-ratio:1; background:var(--ovp-accent-2); transform:rotate(45deg); }}
        .ovp-element.ovp-compressed-representation::before {{ content:""; position:absolute; inset:9%; border:2px solid color-mix(in srgb,currentColor 48%,transparent); transform:translate(-7px,7px); }}
        .ovp-element.ovp-selection-result::after {{ content:""; position:absolute; left:12px; right:12px; top:calc(16% + var(--route-progress,0) * 66%); height:3px; background:var(--ovp-accent); box-shadow:0 0 18px var(--ovp-accent); }}
        .ovp-element.ovp-resolved-outcome::before {{ content:""; width:68px; height:68px; border:5px solid var(--ovp-accent-2); outline:3px solid color-mix(in srgb,var(--ovp-accent-2) 22%,transparent); outline-offset:8px; border-radius:50%; background:radial-gradient(circle,var(--ovp-accent) 0 12px,transparent 13px); }}
        .ovp-element strong {{ position:relative; z-index:2; overflow-wrap:anywhere; }}
        .ovp-relation {{ fill:none; stroke:var(--ovp-accent); stroke-linecap:round; stroke-width:4; pathLength:1; stroke-dasharray:1; stroke-dashoffset:calc(1 - var(--route-progress,0)); filter:drop-shadow(0 0 8px color-mix(in srgb,var(--ovp-accent) 48%,transparent)); }}
        .ovp-progress {{ position:absolute; left:0; bottom:0; width:calc(var(--route-progress,0) * 100%); height:5px; background:var(--ovp-accent); }}
      </style>
      <section class="ovp-stage" data-open-visual-program="{_escape(program.get("program_id"))}" data-open-visual-signature="{_escape(program.get("signature"))}" data-open-visual-medium="{_escape(concept.get("medium"))}">
        {relation_markup}
        {element_markup}
      </section>
    """
    safety = validate_authored_html_safety(fragment)
    if not safety.safe:
        raise ValueError(
            "Compiled Open Visual Program HTML failed safety validation: "
            + "; ".join(safety.errors)
        )
    return CompiledOpenVisualStage(
        html=fragment,
        metadata={
            "generation_mode": "open_visual_program",
            "open_visual_program_id": str(program.get("program_id") or ""),
            "open_visual_program_signature": str(program.get("signature") or ""),
            "open_visual_medium": str(concept.get("medium") or ""),
            "open_visual_metaphor": str(concept.get("metaphor") or ""),
            "open_visual_validation": validation.to_dict(),
            "object_coverage": validation.object_coverage,
            "relation_coverage": validation.relation_coverage,
            "grounded_copy_ratio": validation.grounded_text_ratio,
            "motion_coverage": validation.motion_coverage,
            "semantic_fitness": validation.semantic_fitness,
            "visible_labels": visible_labels,
            "semantic_object_ids": semantic_object_ids,
            "semantic_relation_ids": semantic_relation_ids,
            "fingerprint": dict(validation.fingerprint),
            "safety": safety.to_dict(),
        },
    )


def _element_markup(
    element: dict[str, Any],
    tracks: list[dict[str, Any]],
    *,
    palette: dict[str, Any],
    canvas_width: int,
    canvas_height: int,
) -> str:
    layout = dict(element.get("layout") or {})
    style = dict(element.get("style") or {})
    element_type = _safe_class(element.get("type") or "shape")
    role = _safe_class(element.get("role") or "evidence")
    binding = dict(element.get("binding") or {})
    text_value = str(element.get("text") or "")
    framed = element_type in {"chart", "group", "image", "metric", "shape", "token"}
    background_key = style.get("fill") or ("surface" if framed else "background")
    background = _style_color(background_key, palette, fallback="surface" if framed else "background")
    preferred_foreground = _style_color(
        style.get("color") or ("text" if style.get("fill") == "text" else "ink"),
        palette,
        fallback="ink",
    )
    foreground = _contrast_foreground(
        background,
        preferred_foreground,
        _palette_color(palette, "ink", "#111111"),
        _palette_color(palette, "surface", "#FFFDF8"),
    )
    semantic_font_floor = (
        34.0
        if role == "title"
        else 22.0
        if text_value and not bool(element.get("decorative"))
        else 13.0
    )
    requested_font_size = max(
        semantic_font_floor,
        min(_number(style.get("font_size"), 30.0), 110.0),
    )
    font_size = _fitted_font_size(
        text_value,
        layout,
        requested=requested_font_size,
        framed=framed,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        minimum=semantic_font_floor,
    )
    translate_x = _track_delta(tracks, "translate_x") * 1280
    translate_y = _track_delta(tracks, "translate_y") * 720
    scale_start, scale_end = _track_endpoints(tracks, "scale", 1.0)
    rotation_start, rotation_end = _track_endpoints(tracks, "rotation", 0.0)
    opacity_track = _track(tracks, "opacity")
    delay, span = _track_window(opacity_track or _first_track(tracks))
    animation = "slide-right" if abs(translate_x) >= abs(translate_y) and abs(translate_x) > 1 else "rise"
    y = max(-120, min(120, -translate_y if abs(translate_y) > 1 else 24))
    css = [
        f"left:{_percent(layout.get('x'))}",
        f"top:{_percent(layout.get('y'))}",
        f"width:{_percent(layout.get('width'), 0.1)}",
        f"height:{_percent(layout.get('height'), 0.1)}",
        f"color:{foreground}",
        f"font-size:clamp({semantic_font_floor:.0f}px,{font_size / 12:.3f}vw,{font_size:.1f}px)",
        f"font-weight:{int(max(300, min(_number(style.get('font_weight'), 750), 950)))}",
        (
            "transform:translate3d("
            f"calc(var(--route-progress,0) * {translate_x:.3f}px),"
            f"calc(var(--route-progress,0) * {translate_y:.3f}px),0) "
            f"scale(calc({scale_start:.4f} + var(--route-progress,0) * {scale_end - scale_start:.4f})) "
            f"rotate(calc({rotation_start:.3f}deg + var(--route-progress,0) * {rotation_end - rotation_start:.3f}deg))"
        ),
    ]
    if framed:
        css.extend(
            [
                f"background:{background}",
                f"border:{max(1, int(_number(style.get('stroke_width'), 2)))}px solid {_style_color(style.get('stroke') or 'accent', palette, fallback='accent')}",
                f"border-radius:{max(0, int(_number(style.get('radius'), 0)))}px",
                "padding:clamp(8px,1.2vw,20px)",
            ]
        )
    label_attr = (
        f' data-vex-required-label="{_escape(text_value)}"' if text_value else ""
    )
    repeat = max(1, min(int(_number(element.get("repeat"), 1)), 24))
    if element_type == "particle":
        content = "".join(
            f'<i style="position:absolute;left:{(index * 37) % 94}%;top:{(index * 53) % 88}%;width:{5 + index % 4}px;height:{5 + index % 4}px;border-radius:50%;background:{_accent_var(index)};"></i>'
            for index in range(repeat)
        )
    elif element_type == "chart":
        content = "".join(
            f'<i data-bar="{0.34 + ((index * 29) % 58) / 100:.3f}" style="display:block;position:absolute;left:{10 + index * (80 / max(repeat, 3)):.2f}%;bottom:12%;width:{64 / max(repeat, 3):.2f}%;height:calc(var(--bar-progress,.1) * 76%);background:{_accent_var(index)};"></i>'
            for index in range(max(repeat, 3))
        ) + (f"<strong>{_escape(text_value)}</strong>" if text_value else "")
    elif role in {"source-signal", "transformation-gate"}:
        content = ""
    else:
        content = f"<strong>{_escape(text_value)}</strong>" if text_value else ""
    if framed:
        content += '<i class="ovp-progress" aria-hidden="true"></i>'
    return (
        f'<div class="ovp-element ovp-{element_type} ovp-{role}" '
        f'data-vex-node-id="{_escape(binding.get("id"))}"{label_attr} '
        f'style="{";".join(css)}">'
        f'<div class="ovp-actor" data-anim="{animation}" '
        f'data-delay="{delay:.4f}" data-span="{span:.4f}" '
        f'data-y="{y:.2f}" data-scale=".94">{content}</div></div>'
    )


def _relation_markup(
    relations: list[dict[str, Any]],
    elements: list[dict[str, Any]],
) -> str:
    by_id = {
        str(item.get("element_id") or ""): dict(item.get("layout") or {})
        for item in elements
    }
    paths: list[str] = []
    for index, relation in enumerate(relations):
        source = by_id.get(str(relation.get("source_id") or ""))
        target = by_id.get(str(relation.get("target_id") or ""))
        if source is None or target is None:
            continue
        x1 = (_number(source.get("x"), 0.0) + _number(source.get("width"), 0.1) / 2) * 1000
        y1 = (_number(source.get("y"), 0.0) + _number(source.get("height"), 0.1) / 2) * 1000
        x2 = (_number(target.get("x"), 0.0) + _number(target.get("width"), 0.1) / 2) * 1000
        y2 = (_number(target.get("y"), 0.0) + _number(target.get("height"), 0.1) / 2) * 1000
        bend = max(36.0, abs(x2 - x1) * 0.35)
        binding = dict(relation.get("binding") or {})
        paths.append(
            f'<path class="ovp-relation relation-{index + 1}" '
            f'd="M{x1:.2f},{y1:.2f} C{x1 + bend:.2f},{y1:.2f} {x2 - bend:.2f},{y2:.2f} {x2:.2f},{y2:.2f}" '
            f'data-vex-required-edge="{_escape(binding.get("id") or relation.get("relation_id"))}"></path>'
        )
    return (
        '<svg viewBox="0 0 1000 1000" preserveAspectRatio="none" '
        'style="position:absolute;inset:0;width:100%;height:100%;overflow:visible;pointer-events:none">'
        + "".join(paths)
        + "</svg>"
    )


def _track(tracks: list[dict[str, Any]], property_name: str) -> dict[str, Any] | None:
    return next(
        (item for item in tracks if str(item.get("property") or "") == property_name),
        None,
    )


def _first_track(tracks: list[dict[str, Any]]) -> dict[str, Any] | None:
    return tracks[0] if tracks else None


def _track_window(track: dict[str, Any] | None) -> tuple[float, float]:
    keyframes = [
        dict(item)
        for item in (track or {}).get("keyframes") or []
        if isinstance(item, dict)
    ]
    if len(keyframes) < 2:
        return 0.08, 0.64
    start = max(0.0, min(_number(keyframes[0].get("t"), 0.08), 0.9))
    end = max(start + 0.05, min(_number(keyframes[-1].get("t"), 0.72), 1.0))
    return start, max(0.05, end - start)


def _track_endpoints(
    tracks: list[dict[str, Any]],
    property_name: str,
    default: float,
) -> tuple[float, float]:
    item = _track(tracks, property_name)
    frames = [
        dict(frame)
        for frame in (item or {}).get("keyframes") or []
        if isinstance(frame, dict)
    ]
    if len(frames) < 2:
        return default, default
    return _number(frames[0].get("value"), default), _number(frames[-1].get("value"), default)


def _track_delta(tracks: list[dict[str, Any]], property_name: str) -> float:
    start, end = _track_endpoints(tracks, property_name, 0.0)
    return end - start


def _style_color(
    value: Any,
    palette: dict[str, Any],
    *,
    fallback: str,
) -> str:
    key = str(value or "").strip().lower()
    palette_keys = {
        "accent": "accent",
        "accent_secondary": "accent_secondary",
        "background": "background",
        "grid": "grid",
        "ink": "ink",
        "muted": "muted",
        "surface": "surface",
        "text": "ink",
    }
    palette_key = palette_keys.get(key)
    if palette_key:
        return _palette_color(palette, palette_key, _palette_fallback(palette_key))
    return _color(key, _palette_color(palette, fallback, _palette_fallback(fallback)))


def _palette_color(
    palette: dict[str, Any],
    key: str,
    fallback: str,
) -> str:
    return _color(palette.get(key), fallback)


def _palette_fallback(key: str) -> str:
    return {
        "accent": "#F04438",
        "accent_secondary": "#1E5EFF",
        "background": "#F4F0E8",
        "grid": "#C8C0B4",
        "ink": "#111111",
        "muted": "#4B4740",
        "surface": "#FFFDF8",
    }.get(key, "#111111")


def _contrast_foreground(background: str, *preferred: str) -> str:
    candidates: list[str] = []
    for value in (*preferred, "#FFFFFF", "#111111"):
        normalized = _color(value, "")
        if normalized and normalized.lower() not in {item.lower() for item in candidates}:
            candidates.append(normalized)
    if not candidates:
        return "#FFFFFF"
    for candidate in candidates:
        if _contrast_ratio(background, candidate) >= 4.5:
            return candidate
    return max(candidates, key=lambda value: _contrast_ratio(background, value))


def _contrast_ratio(first: str, second: str) -> float:
    lighter, darker = sorted(
        (_relative_luminance(first), _relative_luminance(second)),
        reverse=True,
    )
    return (lighter + 0.05) / (darker + 0.05)


def _relative_luminance(value: str) -> float:
    cleaned = _color(value, "#000000").lstrip("#")
    if len(cleaned) == 3:
        cleaned = "".join(character * 2 for character in cleaned)
    channels = [int(cleaned[index : index + 2], 16) / 255 for index in (0, 2, 4)]
    linear = [
        channel / 12.92
        if channel <= 0.04045
        else ((channel + 0.055) / 1.055) ** 2.4
        for channel in channels
    ]
    return linear[0] * 0.2126 + linear[1] * 0.7152 + linear[2] * 0.0722


def _fitted_font_size(
    text: str,
    layout: dict[str, Any],
    *,
    requested: float,
    framed: bool,
    canvas_width: int,
    canvas_height: int,
    minimum: float = 13.0,
) -> float:
    if not text.strip():
        return requested
    horizontal_padding = 40.0 if framed else 0.0
    vertical_padding = 32.0 if framed else 0.0
    width = max(
        48.0,
        _number(layout.get("width"), 0.1) * canvas_width - horizontal_padding,
    )
    height = max(
        24.0,
        _number(layout.get("height"), 0.1) * canvas_height - vertical_padding,
    )
    glyph_count = max(1, len(" ".join(text.split())))
    area_fit = math.sqrt((width * height) / (glyph_count * 0.62 * 1.12)) * 0.9
    longest_word = max((len(word) for word in text.split()), default=1)
    word_fit = width / max(longest_word * 0.64, 1.0) * 0.9
    return max(minimum, min(requested, area_fit, word_fit))


def _accent_var(index: int) -> str:
    return "var(--accent-2)" if index % 2 else "var(--accent)"


def _percent(value: Any, default: float = 0.0) -> str:
    return f"{max(0.0, min(_number(value, default), 1.0)) * 100:.4f}%"


def _safe_class(value: Any) -> str:
    return "-".join(
        part for part in str(value or "item").lower().replace("_", "-").split("-") if part.isalnum()
    ) or "item"


def _color(value: Any, fallback: str) -> str:
    cleaned = str(value or "").strip()
    if cleaned.startswith("#") and len(cleaned) in {4, 7, 9}:
        return cleaned
    return fallback


def _escape(value: Any) -> str:
    return html.escape(str(value or ""), quote=True)


def _number(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


__all__ = ["CompiledOpenVisualStage", "compile_open_visual_stage"]
