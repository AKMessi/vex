"""Typed evidence and publication stages reusable across visual workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from renderers.base import RenderedAsset
from vex_visuals.evidence import (
    build_verification_receipt,
    evidence_capture_plan,
    select_evidence_frames,
)
from vex_hyperframes.qa import extract_quality_frames


@dataclass(frozen=True)
class CapturedEvidence:
    asset_path: str
    renderer: str
    frame_paths: tuple[Path, ...]
    reused_renderer_frames: bool


@dataclass(frozen=True)
class PublicationEvidence:
    passed: bool
    receipt: dict[str, Any]


def capture_rendered_evidence(
    spec: dict[str, Any],
    asset: RenderedAsset,
    *,
    output_dir: Path,
    extractor: Callable = extract_quality_frames,
) -> CapturedEvidence:
    artifact_paths = dict(asset.artifact_paths or {})
    existing: list[Path] = []
    for key in ("render_qa_frame_paths", "qa_frame_paths"):
        values = artifact_paths.get(key) or []
        if isinstance(values, (str, Path)):
            values = [values]
        for value in values:
            path = Path(str(value))
            if path.is_file() and path not in existing:
                existing.append(path)
    if len(existing) >= 4:
        frames = select_evidence_frames(existing, limit=8)
        return CapturedEvidence(asset.asset_path, asset.renderer, tuple(frames), True)
    plan = evidence_capture_plan(dict(spec.get("open_visual_program") or {}))
    generated = extractor(
        asset.asset_path,
        output_dir,
        duration_sec=max(float(asset.duration_sec or 0), 0.1),
        frame_count=8,
        capture_plan=plan or None,
        fps=float((asset.metadata or {}).get("fps") or 30),
    )
    return CapturedEvidence(
        asset.asset_path, asset.renderer, tuple(generated or existing), False
    )


def finalize_publication_evidence(
    asset: RenderedAsset,
    spec: dict[str, Any],
    frames: list[Path],
    verdict: dict[str, Any],
) -> PublicationEvidence:
    receipt = build_verification_receipt(asset, spec, frames, verdict)
    return PublicationEvidence(bool(receipt["passed"]), receipt)
