from __future__ import annotations

import html
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
    tracks_by_target: dict[str, list[dict[str, Any]]] = {}
    for item in tracks:
        tracks_by_target.setdefault(str(item.get("target_id") or ""), []).append(
            item
        )
    element_markup = "\n".join(
        _element_markup(
            item,
            tracks_by_target.get(str(item.get("element_id") or ""), []),
        )
        for item in elements
    )
    relation_markup = _relation_markup(relations, elements)
    palette = dict(program.get("palette") or {})
    concept = dict(program.get("concept") or {})
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
        .ovp-stage {{ position:absolute; inset:0; overflow:hidden; color:{_color(palette.get("ink"), "var(--text)")}; }}
        .ovp-stage::before {{ content:""; position:absolute; inset:0; opacity:.22; background-image:linear-gradient({_color(palette.get("grid"), "var(--stroke)")} 1px,transparent 1px),linear-gradient(90deg,{_color(palette.get("grid"), "var(--stroke)")} 1px,transparent 1px); background-size:58px 58px; transform:translate3d(calc(var(--p,0) * -28px),calc(var(--p,0) * -16px),0); }}
        .ovp-element {{ position:absolute; box-sizing:border-box; display:grid; place-items:center; text-align:center; line-height:1.06; overflow:hidden; transform-origin:center; will-change:transform,opacity; }}
        .ovp-actor {{ position:absolute; inset:0; display:grid; place-items:center; padding:inherit; text-align:inherit; }}
        .ovp-element.ovp-text {{ overflow:visible; text-align:left; place-items:center start; }}
        .ovp-element.ovp-source-signal::before {{ content:""; position:absolute; inset:20%; background:repeating-linear-gradient(180deg,var(--accent-2) 0 4px,transparent 4px 11px); opacity:.72; }}
        .ovp-element.ovp-transformation-gate {{ clip-path:polygon(0 0,100% 12%,72% 88%,0 100%); }}
        .ovp-element.ovp-transformation-gate::before {{ content:""; width:46%; aspect-ratio:1; border:6px solid var(--text); border-radius:50%; box-shadow:0 0 28px color-mix(in srgb,var(--accent-2) 62%,transparent); transform:rotate(calc(var(--route-progress,0) * 180deg)); }}
        .ovp-element.ovp-transformation-gate::after {{ content:""; position:absolute; width:18%; aspect-ratio:1; background:var(--accent-2); transform:rotate(45deg); }}
        .ovp-element.ovp-compressed-representation::before {{ content:""; position:absolute; inset:9%; border:2px solid color-mix(in srgb,var(--text) 48%,transparent); transform:translate(-7px,7px); }}
        .ovp-element.ovp-selection-result::after {{ content:""; position:absolute; left:12px; right:12px; top:calc(16% + var(--route-progress,0) * 66%); height:3px; background:var(--accent); box-shadow:0 0 18px var(--accent); }}
        .ovp-element strong {{ position:relative; z-index:2; overflow-wrap:anywhere; }}
        .ovp-relation {{ fill:none; stroke:var(--accent); stroke-linecap:round; stroke-width:4; pathLength:1; stroke-dasharray:1; stroke-dashoffset:calc(1 - var(--route-progress,0)); filter:drop-shadow(0 0 8px color-mix(in srgb,var(--accent) 48%,transparent)); }}
        .ovp-progress {{ position:absolute; left:0; bottom:0; width:calc(var(--route-progress,0) * 100%); height:5px; background:var(--accent); }}
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
) -> str:
    layout = dict(element.get("layout") or {})
    style = dict(element.get("style") or {})
    element_type = _safe_class(element.get("type") or "shape")
    role = _safe_class(element.get("role") or "evidence")
    binding = dict(element.get("binding") or {})
    text_value = str(element.get("text") or "")
    framed = element_type in {"chart", "group", "image", "metric", "shape", "token"}
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
        f"color:{_style_color(style.get('color') or ('text' if style.get('fill') == 'text' else 'ink'))}",
        f"font-size:clamp(13px,{max(13.0, min(_number(style.get('font_size'), 30.0), 110.0)) / 12:.3f}vw,{max(13.0, min(_number(style.get('font_size'), 30.0), 110.0)):.1f}px)",
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
                f"background:{_style_color(style.get('fill') or 'surface')}",
                f"border:{max(1, int(_number(style.get('stroke_width'), 2)))}px solid {_style_color(style.get('stroke') or 'accent')}",
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


def _style_color(value: Any) -> str:
    key = str(value or "").strip().lower()
    variables = {
        "accent": "var(--accent)",
        "accent_secondary": "var(--accent-2)",
        "background": "var(--bg)",
        "grid": "var(--stroke)",
        "ink": "var(--text)",
        "muted": "var(--muted)",
        "surface": "var(--panel)",
        "text": "var(--text)",
    }
    return variables.get(key, _color(key, "var(--text)"))


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
