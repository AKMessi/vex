from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class VisualRendererError(RuntimeError):
    pass


RENDER_JOB_SCHEMA_VERSION = 1
RENDER_JOB_MANIFEST_NAME = "render_job.json"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_render_job_dir(render_root: Path, spec_id: object) -> Path:
    root = Path(render_root).expanduser().resolve(strict=False)
    raw_id = str(spec_id or "visual")
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", raw_id).strip("_")[:96] or "visual"
    candidate = (root / cleaned).resolve(strict=False)
    root_text = os.path.normcase(os.path.abspath(str(root)))
    candidate_text = os.path.normcase(os.path.abspath(str(candidate)))
    try:
        if os.path.commonpath([candidate_text, root_text]) != root_text:
            raise ValueError
    except ValueError as exc:
        raise VisualRendererError("Renderer job directory escaped the render root.") from exc
    return candidate


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
    artifact_paths: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RenderJobManifest:
    job_id: str
    renderer: str
    spec_id: str
    status: str
    render_root: str
    job_dir: str
    width: int
    height: int
    fps: float
    created_at: str
    updated_at: str
    started_at: str = ""
    finished_at: str = ""
    output_path: str = ""
    script_path: str = ""
    error: str = ""
    artifact_paths: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    request: dict[str, Any] = field(default_factory=dict)
    schema_version: int = RENDER_JOB_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "job_id": self.job_id,
            "renderer": self.renderer,
            "spec_id": self.spec_id,
            "status": self.status,
            "render_root": self.render_root,
            "job_dir": self.job_dir,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "output_path": self.output_path,
            "script_path": self.script_path,
            "error": self.error,
            "artifact_paths": self.artifact_paths,
            "metadata": self.metadata,
            "request": self.request,
        }


def render_job_manifest_path(job_dir: Path | str) -> Path:
    return Path(job_dir) / RENDER_JOB_MANIFEST_NAME


def begin_render_job(
    renderer_name: str,
    spec: dict[str, Any],
    *,
    render_root: Path,
    width: int,
    height: int,
    fps: float,
) -> RenderJobManifest:
    spec_id = str(spec.get("visual_id") or spec.get("id") or "visual")
    job_dir = safe_render_job_dir(render_root, spec_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    now = utc_now_iso()
    manifest = RenderJobManifest(
        job_id=_render_job_id(renderer_name, spec_id, now),
        renderer=str(renderer_name or "unknown"),
        spec_id=spec_id,
        status="running",
        render_root=str(Path(render_root).expanduser().resolve(strict=False)),
        job_dir=str(job_dir),
        width=int(width),
        height=int(height),
        fps=float(fps),
        created_at=now,
        updated_at=now,
        started_at=now,
        request={
            "template": spec.get("template"),
            "visual_type_hint": spec.get("visual_type_hint"),
            "composition_mode": spec.get("composition_mode"),
            "start": spec.get("start"),
            "end": spec.get("end"),
            "duration": spec.get("duration"),
        },
    )
    write_render_job_manifest(manifest)
    return manifest


def complete_render_job(manifest: RenderJobManifest, asset: RenderedAsset) -> RenderJobManifest:
    now = utc_now_iso()
    manifest_path = render_job_manifest_path(manifest.job_dir)
    manifest.status = "succeeded"
    manifest.updated_at = now
    manifest.finished_at = now
    manifest.output_path = str(asset.asset_path or "")
    manifest.script_path = str(asset.script_path or "")
    manifest.artifact_paths = {
        **dict(asset.artifact_paths or {}),
        "render_job_manifest_path": str(manifest_path),
    }
    manifest.metadata = dict(asset.metadata or {})
    write_render_job_manifest(manifest)
    asset.artifact_paths = dict(manifest.artifact_paths)
    return manifest


def fail_render_job(manifest: RenderJobManifest, error: object) -> RenderJobManifest:
    now = utc_now_iso()
    manifest.status = "failed"
    manifest.updated_at = now
    manifest.finished_at = now
    manifest.error = str(error)
    write_render_job_manifest(manifest)
    return manifest


def write_render_job_manifest(manifest: RenderJobManifest) -> Path:
    manifest.schema_version = RENDER_JOB_SCHEMA_VERSION
    path = render_job_manifest_path(manifest.job_dir)
    _atomic_write_json(path, manifest.to_dict())
    return path


def render_with_manifest(
    renderer: "VisualRenderer",
    spec: dict[str, Any],
    *,
    render_root: Path,
    width: int,
    height: int,
    fps: float,
) -> RenderedAsset:
    if spec.get("open_visual_program"):
        from vex_visuals.assets import bind_program_assets
        spec = bind_program_assets(spec)
    manifest = begin_render_job(
        renderer.name,
        spec,
        render_root=render_root,
        width=width,
        height=height,
        fps=fps,
    )
    try:
        from vex_runtime.visual_run import current_visual_run
        run = current_visual_run()
        def render():
            if run:
                run.consume("renders")
                if spec.get("isolate_renderer", True) and renderer.name in {"remotion", "hyperframes", "manim", "blender"}:
                    from vex_runtime.renderer_workers import render_isolated
                    return render_isolated(run,renderer.name,spec,render_root=render_root,width=width,height=height,fps=fps)
            return asdict(renderer.render(spec, render_root=render_root, width=width, height=height, fps=fps))
        if run:
            from importlib.resources import files
            from vex_visuals.evidence import file_digest
            runtime_files = ["remotion_entry.jsx", "remotion_scene_graph.jsx", "visual_motion.mjs", "visual_telemetry.mjs", "fonts/Inter.ttf"]
            versions = {name:file_digest(files("renderers").joinpath(name)) for name in runtime_files}
            for package, names in {"vex_hyperframes":["composer.py","open_visual_runtime.py","visual_world_renderer.py"], "vex_visuals":["scene_graph.py","open_visual_program.py","motion_state.py"]}.items():
                versions.update({package+"/"+name:file_digest(files(package).joinpath(name)) for name in names})
            import config
            policy = {key:getattr(config,key,None) for key in ["HYPERFRAMES_QA_MODE","HYPERFRAMES_MIN_QUALITY_SCORE","HYPERFRAMES_RENDER_QUALITY","HYPERFRAMES_ENABLE_VISION_QA","HYPERFRAMES_VISION_MODEL","VISUAL_DIRECTOR_GROQ_VISION_MODEL"]}
            policy["vision_configured"] = bool(config.GROQ_API_KEY or config.GEMINI_API_KEY or config.ANTHROPIC_API_KEY)
            payload, hit = run.stage("render:"+renderer.name, {"spec":spec,"width":width,"height":height,"fps":fps,"runtime":versions,"policy":policy},render,output_paths=lambda value:[value["asset_path"]])
            asset = RenderedAsset(**payload)
            asset.metadata["stage_cache_hit"] = hit
        else:
            asset = RenderedAsset(**render())
    except Exception as exc:
        fail_render_job(manifest, exc)
        if isinstance(exc, VisualRendererError):
            raise
        raise VisualRendererError(f"{renderer.name} renderer failed: {exc}") from exc
    complete_render_job(manifest, asset)
    return asset


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

    def render_with_manifest(
        self,
        spec: dict[str, Any],
        *,
        render_root: Path,
        width: int,
        height: int,
        fps: float,
    ) -> RenderedAsset:
        return render_with_manifest(
            self,
            spec,
            render_root=render_root,
            width=width,
            height=height,
            fps=fps,
        )


def _render_job_id(renderer_name: str, spec_id: str, created_at: str) -> str:
    encoded = f"{renderer_name}:{spec_id}:{created_at}".encode("utf-8")
    return f"render_{hashlib.sha256(encoded).hexdigest()[:16]}"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            json.dump(payload, temp_file, indent=2)
            temp_file.write("\n")
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)
