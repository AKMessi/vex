"""Portable, deduplicated edit history with checksum-based media relinking."""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

from engine import probe_video
from state import ProjectState
from vex_runtime.edit_graph import EditGraph
from vex_runtime.locking import exclusive_file_lock
from vex_runtime.production_bundle import DELIVERIES
from vex_runtime.production_edits import open_timeline
from vex_runtime.studio_projects import create_project
from vex_visuals.evidence import file_digest
from video_generation.production_timeline import audio_assets
from video_generation.remotion_revisions import _project_asset, _read_object, _validate_timeline, _write


VERSION = "vex-edit-history-v1"
HASH = re.compile(r"[a-f0-9]{64}")
VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".flv", ".m4v"}
MAX_CUTS = 200


def _object_path(root: Path, key: object) -> Path:
    if not isinstance(key, str) or not HASH.fullmatch(key):
        raise ValueError("Invalid media checksum in archive.")
    path = root / "objects" / key
    if path.resolve() != path:
        raise ValueError("Archive media must stay in the object directory.")
    return path


def _copy_verified(source: Path, target: Path, digest: str) -> None:
    target.parent.mkdir(exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=".asset-", dir=target.parent)
    os.close(handle)
    staged = Path(name)
    try:
        shutil.copyfile(source, staged)
        if file_digest(staged) != digest:
            raise ValueError("Media changed while it was copied.")
        os.replace(staged, target)
    finally:
        staged.unlink(missing_ok=True)


def _add_object(root: Path, source: Path) -> str:
    identity = file_digest(source)
    target = _object_path(root, identity)
    if not target.is_file():
        _copy_verified(source, target, identity)
    return identity


def _media_refs(timeline: dict) -> set[str]:
    return audio_assets(timeline) | {shot["media"]["src"] for shot in timeline["scenes"] if shot.get("media")}


def export_history(state: ProjectState, destination: Path) -> Path:
    operations = [*state.timeline, *reversed(state.redo_stack)]
    if len(operations) >= MAX_CUTS:
        raise ValueError(f"An archive can store up to {MAX_CUTS} cuts.")
    target = destination.expanduser().resolve()
    target.mkdir(parents=True, exist_ok=False)
    cuts = []
    for index in range(len(operations) + 1):
        operation = operations[index - 1] if index else {}
        active = index == len(state.timeline)
        video = state.working_file if active else operation.get("result_file") if index else state.source_files[0]
        if not isinstance(video, str) or not video:
            raise ValueError(f"Cut {index + 1} has no retained video.")
        media = Path(video).resolve(strict=True)
        if media.suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError(f"Cut {index + 1} has an unsupported video format.")
        snapshot = replace(state, timeline=operations[:index], working_file=str(media))
        entry = {"video": _add_object(target, media), "suffix": media.suffix.lower(),
                 "label": str(operation.get("description") or "Source")[:240]}
        try:
            source, timeline = open_timeline(snapshot)
        except ValueError:
            source = None
        if source is not None:
            entry["timeline"] = timeline
            entry["media"] = {name: _add_object(target, _project_asset(source / "public", name))
                              for name in sorted(_media_refs(timeline))}
        cuts.append(entry)
    document = {"version": VERSION, "title": state.project_name, "active": len(state.timeline),
                "direction": state.artifacts.get("production_direction"), "cuts": cuts}
    _write(target / "archive.json", document)
    return target


def _load_archive(root: Path) -> tuple[dict, set[str]]:
    path = root / "archive.json"
    if path.resolve(strict=True) != path or path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("The project archive manifest is unavailable.")
    metadata = _read_object(path)
    cuts, active = metadata.get("cuts"), metadata.get("active")
    if (metadata.get("version") != VERSION or not isinstance(cuts, list) or not 1 <= len(cuts) <= MAX_CUTS
            or type(active) is not int or not 0 <= active < len(cuts)):
        raise ValueError("Unsupported project archive.")
    keys = set()
    for cut in cuts:
        if not isinstance(cut, dict) or not isinstance(cut.get("suffix"), str) or cut["suffix"] not in VIDEO_SUFFIXES:
            raise ValueError("An archived cut has an invalid format.")
        refs = cut.get("media", {})
        if not isinstance(refs, dict):
            raise ValueError("Invalid archived media inventory.")
        if "timeline" in cut:
            from video_generation.remotion_revisions import ASSET_NAME

            timeline = _validate_timeline({"timeline": cut["timeline"]})
            if cut["suffix"] not in {Path(filename).suffix for filename in DELIVERIES} or (
                refs.keys() != _media_refs(timeline)
                or any(not ASSET_NAME.fullmatch(name) or name.endswith(".") for name in refs)
            ):
                raise ValueError("The composition does not match its archived media.")
        elif refs:
            raise ValueError("A baked cut cannot refer to composition media.")
        for key in (cut.get("video"), *refs.values()):
            _object_path(root, key)
            keys.add(key)
    return metadata, keys


def relink_archive(root: Path, directory: Path) -> dict:
    root = root.resolve(strict=True)
    directory = directory.resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("Choose a folder containing the relocated media.")
    _, keys = _load_archive(root)
    missing = {key for key in keys if not _object_path(root, key).exists()}
    repaired = []
    for parent, directories, names in os.walk(directory, followlinks=False):
        directories.sort()
        for name in sorted(names):
            if not missing:
                return {"relinked": repaired, "missing": []}
            candidate = Path(parent) / name
            if candidate.is_symlink() or not candidate.is_file() or not candidate.resolve().is_relative_to(directory):
                continue
            digest = file_digest(candidate)
            if digest in missing:
                _copy_verified(candidate, _object_path(root, digest), digest)
                repaired.append(digest)
                missing.remove(digest)
    return {"relinked": repaired, "missing": sorted(missing)}


def relink_project_source(state: ProjectState, relocated: Path) -> Path:
    """Link the primary footage again using the bytes in its original managed copy."""
    workspace = Path(state.working_dir).resolve()
    original = Path(state.source_files[0])
    managed = workspace / f"source_{original.name}"
    if managed.resolve(strict=True) != managed or not managed.is_file():
        raise ValueError("The managed original is unavailable. Relink from a portable archive instead.")
    candidate = relocated.resolve(strict=True)
    if not candidate.is_file() or candidate.suffix.lower() not in VIDEO_SUFFIXES:
        raise ValueError("Choose the relocated video file.")
    if file_digest(candidate) != file_digest(managed):
        raise ValueError("This file does not match the original video checksum.")
    with exclusive_file_lock(workspace / ".source-relink.lock"):
        state.source_files[0] = str(candidate)
        state.save()
    return candidate


def open_history(root: Path, provider: str, model: str) -> ProjectState:
    root = root.resolve(strict=True)
    archive, keys = _load_archive(root)
    for key in keys:
        path = _object_path(root, key)
        if not path.is_file() or file_digest(path) != key:
            raise ValueError(f"Media object {key} is missing or different. Relink the archive first.")

    def initialize(state: ProjectState) -> None:
        workspace = Path(state.working_dir)
        cache = workspace / ".archive-media"
        cache.mkdir()
        for key in keys:
            _copy_verified(_object_path(root, key), cache / key, key)
        cut_root = workspace / "production_cuts"
        cut_root.mkdir()
        videos = []
        operations = []
        for index, cut in enumerate(archive["cuts"]):
            folder = Path(tempfile.mkdtemp(prefix="remotion-", dir=cut_root))
            output = folder / f"final{cut['suffix']}"
            shutil.copyfile(cache / cut["video"], output)
            report = folder / "revision.json"
            _write(report, {"version": "vex-remotion-revision-v1", "status": "verified",
                            "output": output.name, "receipt": {"output_sha256": cut["video"]}})
            if "timeline" in cut:
                _write(folder / "input_props.json", {"timeline": cut["timeline"]})
                _write(folder / "render_request.json", {"render_mode": "video", "video_quality": "standard",
                                                        "video_format": output.suffix[1:]})
                for name, key in cut["media"].items():
                    destination = folder / "public" / name
                    destination.parent.mkdir(exist_ok=True)
                    try:
                        os.link(cache / key, destination)
                    except OSError:
                        shutil.copyfile(cache / key, destination)
                if index == 0:
                    state.artifacts["generated_source"] = {"manifest_path": str(report)}
            if index:
                operations.append({"op": "edit_production" if "timeline" in cut else "archived_checkpoint",
                                   "params": {"cut": folder.name}, "previous_file": str(videos[-1]),
                                   "result_file": str(output), "description": str(cut.get("label", ""))[:240]})
            videos.append(output)
        active = archive["active"]
        state.source_files = [str(videos[0])]
        state.timeline = operations[:active]
        state.redo_stack = list(reversed(operations[active:]))
        state.working_file = str(videos[active])
        state.metadata = probe_video(state.working_file)
        state.edit_graph = EditGraph.from_source(state.working_file, duration=state.metadata["duration_sec"],
                                                 fps=state.metadata["fps_ratio"] or state.metadata["fps"],
                                                 provenance="rendered_anchor").to_dict()
        direction = archive.get("direction")
        if (isinstance(direction, dict) and isinstance(direction.get("notes"), str)
                and isinstance(direction.get("locked_shots"), list)):
            state.artifacts["production_direction"] = {
                "notes": direction["notes"][:3000],
                "locked_shots": [item for item in direction["locked_shots"][:200] if isinstance(item, str)],
            }

    first = archive["cuts"][0]
    title = str(archive.get("title") or "Restored project")[:120]
    return create_project(str(_object_path(root, first["video"])), title, provider, model,
                          managed_source=True, initialize=initialize)
