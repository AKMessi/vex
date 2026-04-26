from __future__ import annotations

from typing import Any

from renderers.base import RenderedAsset, RendererStatus, VisualRenderer, VisualRendererError
from renderers.blender_renderer import BlenderRenderer
from renderers.ffmpeg_renderer import FFmpegRenderer
from renderers.manim_renderer import ManimRenderer

_RENDERERS: dict[str, VisualRenderer] = {
    "manim": ManimRenderer(),
    "ffmpeg": FFmpegRenderer(),
    "blender": BlenderRenderer(),
}


def get_renderer(name: str) -> VisualRenderer:
    normalized = (name or "manim").strip().lower()
    if normalized not in _RENDERERS:
        raise VisualRendererError(f"Unsupported renderer: {name}")
    return _RENDERERS[normalized]


def list_renderers() -> list[VisualRenderer]:
    return list(_RENDERERS.values())


def renderer_capabilities() -> list[dict[str, Any]]:
    return [renderer.capability_summary() for renderer in list_renderers()]


def available_renderers() -> list[VisualRenderer]:
    return [renderer for renderer in list_renderers() if renderer.availability().available]


def resolve_renderer(
    spec: dict[str, Any],
    *,
    preferred: str = "auto",
    allow_unavailable: bool = False,
    exclude: set[str] | None = None,
) -> tuple[VisualRenderer, str]:
    preferred_name = (preferred or "auto").strip().lower()
    exclude = {name.strip().lower() for name in (exclude or set())}
    if preferred_name and preferred_name != "auto" and preferred_name not in exclude:
        preferred_renderer = get_renderer(preferred_name)
        preferred_status = preferred_renderer.availability()
        if preferred_status.available or allow_unavailable:
            preferred_score = preferred_renderer.score_spec(spec)
            if preferred_score >= 0.0:
                return (
                    preferred_renderer,
                    f"{preferred_renderer.name} was explicitly preferred for {spec.get('template', 'visual')} "
                    f"({spec.get('visual_type_hint', 'general')}).",
                )
    candidates: list[VisualRenderer]
    if preferred_name and preferred_name != "auto":
        candidates = [get_renderer(preferred_name)] + [renderer for renderer in list_renderers() if renderer.name != preferred_name]
    else:
        candidates = list_renderers()

    best_renderer: VisualRenderer | None = None
    best_score = -999.0
    best_reason = ""
    unavailable_notes: list[str] = []
    for renderer in candidates:
        if renderer.name in exclude:
            continue
        status = renderer.availability()
        if not status.available and not allow_unavailable:
            unavailable_notes.append(f"{renderer.name}: {status.reason}")
            continue
        score = renderer.score_spec(spec)
        if score > best_score:
            best_score = score
            best_renderer = renderer
            best_reason = (
                f"{renderer.name} scored {score:.2f} for {spec.get('template', 'visual')} "
                f"({spec.get('visual_type_hint', 'general')})."
            )
    if best_renderer is None:
        detail = "; ".join(unavailable_notes) or "No renderer reported availability."
        raise VisualRendererError(f"No renderer could render this visual. {detail}")
    return best_renderer, best_reason


__all__ = [
    "RenderedAsset",
    "RendererStatus",
    "VisualRenderer",
    "VisualRendererError",
    "available_renderers",
    "get_renderer",
    "list_renderers",
    "renderer_capabilities",
    "resolve_renderer",
]
