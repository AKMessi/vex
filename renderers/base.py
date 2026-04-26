from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class VisualRendererError(RuntimeError):
    pass


@dataclass
class RendererStatus:
    available: bool
    reason: str = ""


@dataclass
class RenderedAsset:
    asset_path: str
    width: int
    height: int
    duration_sec: float
    renderer: str
    job_dir: str
    script_path: str
    artifact_paths: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class VisualRenderer:
    name = "base"
    supported_templates: set[str] = set()

    def render(
        self,
        spec: dict[str, Any],
        render_root: Path,
        width: int,
        height: int,
        fps: float,
    ) -> RenderedAsset:
        raise NotImplementedError

    def availability(self) -> RendererStatus:
        return RendererStatus(True, "")

    def supports(self, spec: dict[str, Any]) -> bool:
        if not self.supported_templates:
            return True
        return str(spec.get("template") or "").strip().lower() in self.supported_templates

    def score_spec(self, spec: dict[str, Any]) -> float:
        if not self.supports(spec):
            return -1.0
        return 0.5

    def capability_summary(self) -> dict[str, Any]:
        status = self.availability()
        return {
            "name": self.name,
            "available": status.available,
            "reason": status.reason,
            "supported_templates": sorted(self.supported_templates),
        }
