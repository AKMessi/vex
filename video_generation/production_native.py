"""Materialize validated scenes using Vex's trusted native renderer entry points."""
from __future__ import annotations

import importlib.metadata
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

from vex_runtime.locking import exclusive_file_lock
from vex_visuals.evidence import file_digest, payload_digest
from video_generation.film_direction import parse_film_direction
from video_generation.film_project import _render_native, _stage_native_source
from video_generation.models import VideoGenerationRequest
from video_generation.remotion_project import _write_json
from video_generation.renderer import HyperframesVideoRuntime


def materialize_scene(root: Path, scene: dict, timeline: dict, quality: str, workers: str) -> Path:
    if root.resolve() != root:
        raise ValueError("Native assets must stay inside the production workspace.")
    recipe = scene["native_recipe"]
    # Revalidate saved input before the worker sees it, including bundle and CLI input.
    direction = parse_film_direction({"palette": recipe["palette"], "shots": [recipe]}, "", "", authored=True)
    shot = direction.shots[0]
    if shot.producer != scene.get("producer") or shot.producer not in {"manim", "hyperframes"}:
        raise ValueError("The native recipe must match the shot's renderer.")
    request = VideoGenerationRequest(prompt="", renderer="hybrid", quality=quality, workers=workers,
                                     **{key: timeline[key] for key in ("width", "height", "fps")})
    frames = scene.get("source_duration_frames", scene["duration_frames"])
    payload = {**asdict(shot), "palette": direction.palette,
               "duration": frames / request.fps, "fps": request.fps,
               "width": request.width, "height": request.height}
    runtime = HyperframesVideoRuntime(auto_install=False)
    compiler = Path(__file__).with_name("film_project.py")
    if shot.producer == "manim":
        try:
            fingerprint = {"version": importlib.metadata.version("manim"),
                           "source": file_digest(Path(__file__).with_name("film_manim_runtime.py"))}
        except importlib.metadata.PackageNotFoundError as exc:
            raise ValueError("Install the Vex Manim extra to render this scene.") from exc
    else:
        motion = compiler.parent.parent / "renderers" / "film_motion.mjs"
        fingerprint = {"cli": file_digest(Path(runtime.ensure_available())), "quality": quality,
                       "motion": file_digest(motion)}
    identity = payload_digest({"scene": payload, "renderer": fingerprint, "compiler": file_digest(compiler)})
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{identity}.mp4"
    receipt = target.with_suffix(".sha256")
    with exclusive_file_lock(target.with_suffix(".lock"), timeout_sec=600):
        if target.resolve() != target or receipt.resolve() != receipt:
            raise ValueError("The cached renderer asset escapes its workspace.")
        if target.is_file() and receipt.is_file() and receipt.read_text(encoding="ascii") == file_digest(target):
            return target
        job = Path(tempfile.mkdtemp(prefix="scene-", dir=root))
        try:
            _write_json(job / "shot.json", payload)
            source = _stage_native_source(job, payload, shot.kind)
            output, _ = _render_native(job, source, runtime, request, shot.kind, frames)
            digest = file_digest(output)
            output.replace(target)
            digest_file = job / "digest"
            digest_file.write_text(digest, encoding="ascii")
            digest_file.replace(receipt)
        finally:
            shutil.rmtree(job, ignore_errors=True)
    return target


def materialize_sandbox(root: Path, scene: dict, timeline: dict) -> Path:
    from vex_runtime.native_sandbox import render_source, sandbox_image

    if root.resolve() != root:
        raise ValueError("Sandbox renders must stay in the production workspace.")
    root.mkdir(parents=True, exist_ok=True)
    request = {name: timeline[name] for name in ("width", "height", "fps")}
    request["frames"] = scene.get("source_duration_frames", scene["duration_frames"])
    native = scene["sandbox_source"]
    identity = payload_digest([sandbox_image(), native, request])
    target = root / f"{identity}.mp4"
    stamp = root / f"{identity}.sha256"
    with exclusive_file_lock(root / f"{identity}.lock", timeout_sec=400):
        if target.resolve() != target or stamp.resolve() != stamp:
            raise ValueError("Sandbox cache path escaped its workspace.")
        if target.is_file() and stamp.is_file() and stamp.read_text(encoding="ascii") == file_digest(target):
            return target
        target.unlink(missing_ok=True)
        stamp.unlink(missing_ok=True)
        receipt = render_source(native["source"], target, engine=native["engine"], **request)
        stamp.write_text(receipt["sha256"], encoding="ascii")
    return target
