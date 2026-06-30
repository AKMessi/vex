from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from renderers.base import (
    RenderedAsset,
    RendererStatus,
    VisualRenderer,
    VisualRendererError,
    render_with_manifest,
)
from renderers.blender_renderer import BlenderRenderer
from renderers.blender_spec import BlenderVisualSpec
from renderers.ffmpeg_renderer import FFmpegRenderer
from renderers.hyperframes_renderer import HyperframesRenderer
from renderers.manim_renderer import ManimRenderer

_RENDERERS: dict[str, VisualRenderer] = {
    "hyperframes": HyperframesRenderer(),
    "manim": ManimRenderer(),
    "ffmpeg": FFmpegRenderer(),
    "blender": BlenderRenderer(),
}


@dataclass(frozen=True)
class RendererMatch:
    renderer: VisualRenderer
    score: float
    reason: str
    explicitly_preferred: bool = False


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
    matches = rank_renderers(
        spec,
        preferred=preferred,
        allow_unavailable=allow_unavailable,
        exclude=exclude,
    )
    selected = matches[0]
    return selected.renderer, selected.reason


def rank_renderers(
    spec: dict[str, Any],
    *,
    preferred: str = "auto",
    allow_unavailable: bool = False,
    exclude: set[str] | None = None,
) -> list[RendererMatch]:
    preferred_name = (preferred or "auto").strip().lower()
    exclude = {name.strip().lower() for name in (exclude or set())}
    if preferred_name not in {"", "auto"}:
        get_renderer(preferred_name)
    candidates: list[VisualRenderer]
    if preferred_name not in {"", "auto"}:
        candidates = [get_renderer(preferred_name)] + [renderer for renderer in list_renderers() if renderer.name != preferred_name]
    else:
        candidates = list_renderers()

    matches: list[tuple[int, int, RendererMatch]] = []
    unavailable_notes: list[str] = []
    rejected_notes: list[str] = []
    template = str(spec.get("template") or "visual").strip().lower() or "visual"
    for order, renderer in enumerate(candidates):
        if renderer.name in exclude:
            continue
        status = renderer.availability()
        if not status.available and not allow_unavailable:
            unavailable_notes.append(f"{renderer.name}: {status.reason}")
            continue
        score = renderer.score_spec(spec)
        if score < 0.0:
            if not renderer.supports(spec):
                rejected_notes.append(
                    f"{renderer.name}: template {template!r} is unsupported"
                )
            else:
                rejected_notes.append(
                    f"{renderer.name}: rejected template {template!r} for this rendering route"
                )
            continue
        explicitly_preferred = (
            preferred_name not in {"", "auto"} and renderer.name == preferred_name
        )
        reason = (
            (
                f"{renderer.name} was explicitly preferred for {spec.get('template', 'visual')} "
                f"({spec.get('visual_type_hint', 'general')}); capability score {score:.2f}."
            )
            if explicitly_preferred
            else (
                f"{renderer.name} scored {score:.2f} for {spec.get('template', 'visual')} "
                f"({spec.get('visual_type_hint', 'general')})."
            )
        )
        preference_rank = 0 if explicitly_preferred else 1
        matches.append(
            (
                preference_rank,
                order,
                RendererMatch(
                    renderer=renderer,
                    score=round(float(score), 4),
                    reason=reason,
                    explicitly_preferred=explicitly_preferred,
                ),
            )
        )
    if not matches:
        detail = "; ".join([*unavailable_notes, *rejected_notes])
        if not detail:
            detail = "No registered renderer accepted this visual specification."
        raise VisualRendererError(f"No renderer could render this visual. {detail}")
    matches.sort(key=lambda item: (item[0], -item[2].score, item[1]))
    return [item[2] for item in matches]


__all__ = [
    "RenderedAsset",
    "RendererMatch",
    "RendererStatus",
    "VisualRenderer",
    "VisualRendererError",
    "BlenderVisualSpec",
    "available_renderers",
    "get_renderer",
    "list_renderers",
    "renderer_capabilities",
    "render_with_manifest",
    "rank_renderers",
    "resolve_renderer",
]
