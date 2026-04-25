from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


class VisualRendererError(RuntimeError):
    pass


@dataclass
class RenderedAsset:
    asset_path: str
    width: int
    height: int
    duration_sec: float
    renderer: str
    job_dir: str
    script_path: str


class VisualRenderer:
    name = "base"

    def render(
        self,
        spec: dict[str, Any],
        render_root: Path,
        width: int,
        height: int,
        fps: float,
    ) -> RenderedAsset:
        raise NotImplementedError
