"""Verify a scoped film repair against its baseline on the same frame clock."""
from __future__ import annotations

import json
import shutil
from collections.abc import Collection
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image

from vex_visuals.evidence import file_digest, payload_digest
from vex_runtime.visual_run import consume_visual_resource
from video_generation.film_artwork import validate_artwork
from video_generation.remotion_project import _write_json, run_remotion_job, stage_remotion_runtime
from video_generation.remotion_revisions import _project_asset
from video_generation.renderer import CommandRecord, VideoGenerationRuntimeError


def compare_film_repair(
    baseline: Path, candidate: Path, job: Path, changed_ids: set[str], commands: list[CommandRecord],
    *, workers: str, auto_install_runtime: bool,
) -> dict[str, Any]:
    started = perf_counter()
    timelines = [json.loads(path.read_text(encoding="utf-8"))["timeline"] for path in (baseline, candidate)]
    before, after = timelines
    locked = ("width", "height", "fps", "duration_frames", "captions", "audio")
    if any(before[key] != after[key] for key in locked):
        raise ValueError("Film repair changed the canvas, clock, captions or audio.")
    scene_keys = ("id", "start_frame", "duration_frames", "producer", "standfirst")
    if ([tuple(shot[key] for key in scene_keys) for shot in before["scenes"]]
            != [tuple(shot[key] for key in scene_keys) for shot in after["scenes"]]):
        raise ValueError("Film repair changed shot identity, timing, renderer or narration.")

    scenes = before["scenes"]
    if not 1 <= len(changed_ids) <= 3 or changed_ids - {scene["id"] for scene in scenes}:
        raise ValueError("Film comparison needs one to three known repair targets.")
    if any(a != b for a, b in zip(scenes, after["scenes"]) if a["id"] not in changed_ids):
        raise ValueError("Film repair changed a shot outside its edit scope.")
    samples = _sample_frames(scenes, after["scenes"], changed_ids)
    frames = sorted(frame for shot_frames in samples.values() for frame in shot_frames)

    job.mkdir()
    stage_remotion_runtime(job)
    public = job / "public"
    public.mkdir()
    variants = []
    variant_ids = ("baseline", "candidate")
    for label, path, timeline in zip(variant_ids, (baseline, candidate), timelines, strict=True):
        variants.append({
            "candidate_id": label,
            "input_props": _preview_props(timeline, path.parent, public, samples.keys()),
        })
    _write_json(job / "input_props.json", variants[0]["input_props"])
    (job / "variants.json").write_text(json.dumps(variants, ensure_ascii=False), encoding="utf-8")
    _write_json(job / "render_request.json", {
        "render_mode": "project_stills", "video_format": "mp4", "video_quality": "draft",
        "frames": frames, "candidate_input_props_file": "variants.json",
    })
    consume_visual_resource("renders", 2)
    command = run_remotion_job(
        job, workers=workers, auto_install_runtime=auto_install_runtime, name="film_repair_comparison",
    )
    commands.append(command)
    if command.returncode:
        raise VideoGenerationRuntimeError(f"Film comparison could not render. See {command.log_path}")

    shots = []
    issues = []
    for shot_id, shot_frames in samples.items():
        evidence = [_compare_frame(job, frame) for frame in shot_frames]
        targeted = shot_id in changed_ids
        visible = any(frame["visible_change"] for frame in evidence)
        if targeted != visible:
            issues.append(f"{'no_visible_change' if targeted else 'change_outside_scope'}:{shot_id}")
        shots.append({"id": shot_id, "targeted": targeted, "frames": evidence})
    receipt = {
        "version": "vex-film-paired-frames-v1",
        "passed": not issues,
        "issues": issues,
        "shots": shots,
        "sampled_frames_per_variant": len(frames),
        "input_sha256": {label: payload_digest({"timeline": timeline}) for label, timeline in zip(variant_ids, timelines)},
        "comparison_wall_seconds": round(perf_counter() - started, 3),
    }
    result = {**receipt, "receipt_sha256": payload_digest(receipt)}
    _write_json(job / "comparison.json", result)
    return result


def _sample_frames(before: list[dict], after: list[dict], changed_ids: set[str]) -> dict[str, list[int]]:
    samples = {}
    optional = {}
    for index, (scene, candidate) in enumerate(zip(before, after, strict=True)):
        shot_id = scene["id"]
        start = scene["start_frame"]
        last = scene["duration_frames"] - 1
        if shot_id in changed_ids:
            anchors = {start, start + round(last * .55), start + last}
            samples[shot_id] = sorted(anchors)
            changed_key_frames = {start + round(last * time) for time in _changed_key_times(scene, candidate)}
            alternatives = changed_key_frames - anchors or {start + round(last * .28)} - anchors
            if alternatives:
                optional[shot_id] = max(alternatives, key=lambda f: (min(abs(f - a) for a in anchors), -f))
        elif index and before[index - 1]["id"] in changed_ids:
            samples[shot_id] = [start]
    spare = 12 - sum(map(len, samples.values()))
    for shot_id, frame in list(optional.items())[:spare]:
        samples[shot_id] = sorted([*samples[shot_id], frame])
    return samples


def _changed_key_times(before: dict, after: dict) -> set[float]:
    def keys(scene: dict) -> set[tuple]:
        return {
            (layer["id"], *key)
            for layer in (scene.get("artwork") or {}).get("layers", [])
            for key in layer.get("keys", [])
        }

    return {key[1] for key in keys(before) ^ keys(after) if key[-1] > 0}


def reviewed_frames(
    composition: Path, delivery: dict[str, Any], shot_ids: Collection[str],
) -> dict[str, list[tuple[int, Path]]]:
    evidence = {}
    evidence_root = (composition / "delivery_frames").resolve()
    for shot in delivery["shots"]:
        if shot["id"] not in shot_ids:
            continue
        frames = []
        for frame in shot["frames"]:
            path = (composition / frame["path"]).resolve(strict=True)
            if path.parent != evidence_root or file_digest(path) != frame["sha256"]:
                raise ValueError(f"Reviewed frame evidence changed for {shot['id']}.")
            frames.append((frame["frame"], path))
        evidence[shot["id"]] = frames
    if evidence.keys() != set(shot_ids) or any(not frames for frames in evidence.values()):
        raise ValueError("The original delivery is missing reviewed frames for the repair.")
    return evidence


def _preview_props(timeline: dict, root: Path, public: Path, sampled_ids: Collection[str]) -> dict:
    scenes = []
    for scene in timeline["scenes"]:
        preview = dict(scene)
        if scene["id"] not in sampled_ids:
            preview.pop("media", None)
            preview.pop("artwork", None)
        if "artwork" in preview:
            validate_artwork(preview["artwork"])
        if media := preview.get("media"):
            source = _project_asset((root / "public").resolve(), media["src"])
            name = file_digest(source) + source.suffix.lower()
            if not (public / name).exists():
                shutil.copyfile(source, public / name)
            preview["media"] = {**media, "src": name}
        scenes.append(preview)
    return {"timeline": {**timeline, "scenes": scenes, "audio": {"narration": "", "music": "", "music_volume": 0}}}


def _compare_frame(job: Path, frame: int) -> dict:
    name = f"frame_{frame:08d}.png"
    baseline = job / "remotion_preview_frames" / "baseline" / name
    candidate = job / "remotion_preview_frames" / "candidate" / name
    with Image.open(baseline) as original, Image.open(candidate) as updated:
        if original.size != updated.size:
            raise ValueError("Paired frames have different dimensions.")
        delta = np.abs(
            np.asarray(original.convert("RGB"), dtype=np.int16) - np.asarray(updated.convert("RGB"), dtype=np.int16),
        ).max(axis=2)
    changed = int(np.count_nonzero(delta >= 8))
    return {
        "frame": frame,
        "baseline": {"path": baseline.relative_to(job).as_posix(), "sha256": file_digest(baseline)},
        "candidate": {"path": candidate.relative_to(job).as_posix(), "sha256": file_digest(candidate)},
        "changed_pixel_fraction": round(changed / delta.size, 6),
        "visible_change": changed >= max(8, delta.size * .0002),
    }
