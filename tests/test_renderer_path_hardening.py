from __future__ import annotations

import json
from pathlib import Path

import pytest

from renderers.base import (
    RenderedAsset,
    VisualRenderer,
    VisualRendererError,
    render_job_manifest_path,
    render_with_manifest,
    safe_render_job_dir,
)


def test_safe_render_job_dir_sanitizes_path_traversal(tmp_path: Path) -> None:
    render_root = tmp_path / "renders"

    job_dir = safe_render_job_dir(render_root, "../../outside")

    assert job_dir == (render_root / "outside").resolve()
    assert job_dir.parent == render_root.resolve()


def test_safe_render_job_dir_limits_untrusted_identifier_length(tmp_path: Path) -> None:
    render_root = tmp_path / "renders"

    job_dir = safe_render_job_dir(render_root, "a" * 200)

    assert job_dir.parent == render_root.resolve()
    assert len(job_dir.name) == 96


def test_render_with_manifest_records_success(tmp_path: Path) -> None:
    renderer = _SuccessfulRenderer()
    render_root = tmp_path / "renders"

    asset = render_with_manifest(
        renderer,
        {"visual_id": "scene-1", "template": "metric_callout", "duration": 2.0},
        render_root=render_root,
        width=1920,
        height=1080,
        fps=30,
    )

    manifest_path = Path(asset.artifact_paths["render_job_manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_path == render_job_manifest_path(asset.job_dir)
    assert manifest["status"] == "succeeded"
    assert manifest["renderer"] == "test_renderer"
    assert manifest["output_path"] == asset.asset_path
    assert manifest["metadata"]["quality_score"] == 0.91


def test_render_with_manifest_records_failure(tmp_path: Path) -> None:
    renderer = _FailingRenderer()
    render_root = tmp_path / "renders"

    with pytest.raises(VisualRendererError, match="render failed"):
        render_with_manifest(
            renderer,
            {"visual_id": "scene-2", "template": "metric_callout"},
            render_root=render_root,
            width=1920,
            height=1080,
            fps=30,
        )

    manifest_path = render_job_manifest_path(render_root / "scene-2")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["error"] == "render failed"


def test_render_with_manifest_normalizes_unexpected_backend_failure(tmp_path: Path) -> None:
    renderer = _UnexpectedFailingRenderer()
    render_root = tmp_path / "renders"

    with pytest.raises(VisualRendererError, match="test_renderer renderer failed: backend crashed"):
        render_with_manifest(
            renderer,
            {"visual_id": "scene-3", "template": "metric_callout"},
            render_root=render_root,
            width=1920,
            height=1080,
            fps=30,
        )

    manifest = json.loads(
        render_job_manifest_path(render_root / "scene-3").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "failed"
    assert manifest["error"] == "backend crashed"


class _SuccessfulRenderer(VisualRenderer):
    name = "test_renderer"

    def render(
        self,
        spec: dict,
        render_root: Path,
        width: int,
        height: int,
        fps: float,
    ) -> RenderedAsset:
        job_dir = safe_render_job_dir(render_root, spec["visual_id"])
        job_dir.mkdir(parents=True, exist_ok=True)
        output_path = job_dir / "scene.mp4"
        script_path = job_dir / "script.txt"
        output_path.write_bytes(b"video")
        script_path.write_text("render", encoding="utf-8")
        return RenderedAsset(
            asset_path=str(output_path),
            width=width,
            height=height,
            duration_sec=2.0,
            renderer=self.name,
            job_dir=str(job_dir),
            script_path=str(script_path),
            metadata={"quality_score": 0.91},
        )


class _FailingRenderer(VisualRenderer):
    name = "test_renderer"

    def render(
        self,
        spec: dict,
        render_root: Path,
        width: int,
        height: int,
        fps: float,
    ) -> RenderedAsset:
        raise VisualRendererError("render failed")


class _UnexpectedFailingRenderer(VisualRenderer):
    name = "test_renderer"

    def render(
        self,
        spec: dict,
        render_root: Path,
        width: int,
        height: int,
        fps: float,
    ) -> RenderedAsset:
        raise OSError("backend crashed")
