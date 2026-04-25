from __future__ import annotations

from renderers.base import RenderedAsset, VisualRendererError
from renderers.manim_renderer import ManimRenderer

_RENDERERS = {
    "manim": ManimRenderer(),
}


def get_renderer(name: str):
    normalized = (name or "manim").strip().lower()
    if normalized not in _RENDERERS:
        raise VisualRendererError(f"Unsupported renderer: {name}")
    return _RENDERERS[normalized]


__all__ = ["RenderedAsset", "VisualRendererError", "get_renderer"]
