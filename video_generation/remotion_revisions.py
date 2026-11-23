from __future__ import annotations

import json
import math
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

from engine import probe_video
from vex_visuals.evidence import file_digest, payload_digest
from video_generation.models import VALID_FORMATS, VALID_QUALITIES
from video_generation.remotion_incremental import render_revision_delivery
from video_generation.remotion_project import (
    TIMELINE_VERSION, run_remotion_job, stage_remotion_runtime,
)


IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_TYPES = {".mp4", ".mov", ".webm"}
ASSET_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,149}")


def _composition_dir(project: Path) -> Path:
    source = project.expanduser().resolve(strict=True)
    if (source / "remotion" / "input_props.json").is_file():
        report = source / "film_repair.json"
        accepted = report.is_file() and _read_object(report).get("accepted") is True
        relative = "film_repair/remotion" if accepted else "remotion"
        composition = (source / relative).resolve(strict=True)
        if not composition.is_relative_to(source):
            raise ValueError("The selected composition must be within the generated project.")
        return composition
    return source


def inspect_remotion_project(project: Path) -> dict[str, Any]:
    source = _composition_dir(project)
    timeline = _validate_timeline(_read_object(source / "input_props.json"), allow_pending=True)
    shots = [
        {
            "id": scene["id"],
            "producer": scene.get("pending_producer") or scene.get("producer") or "remotion",
            "pending": bool(scene.get("pending_producer")),
            "start_frame": scene["start_frame"],
            "end_frame": scene["start_frame"] + scene["duration_frames"],
        }
        for scene in timeline["scenes"]
    ]
    return {
        "version": "vex-production-inspection-v1",
        "composition": str(source),
        "width": timeline["width"], "height": timeline["height"],
        "fps": timeline["fps"], "duration_frames": timeline["duration_frames"],
        "caption_count": len(timeline["captions"]),
        "audio": {role: bool(timeline["audio"].get(role)) for role in ("narration", "music")},
        "shots": shots,
    }


def revise_remotion_project(
    project: Path,
    *,
    replacements: dict[str, Path],
    output_dir: Path | None = None,
    workers: str | None = None,
    quality: str | None = None,
    output_format: str | None = None,
    preview_frames: tuple[int, ...] = (),
    stage_only: bool = False,
    full_render: bool = False,
) -> tuple[Path, Path, dict[str, Any]]:
    source = _composition_dir(project)
    props = _read_object(source / "input_props.json")
    original_digest = payload_digest(props)
    settings = _read_object(source / "render_request.json")
    timeline = _validate_timeline(props, allow_pending=True)
    if (
        settings.get("render_mode") != "video"
        or settings.get("video_format") not in VALID_FORMATS
        or settings.get("video_quality") not in VALID_QUALITIES
    ):
        raise ValueError("Invalid Remotion video export settings.")
    output_format = output_format or settings["video_format"]
    quality = quality or settings["video_quality"]
    if output_format not in VALID_FORMATS or quality not in VALID_QUALITIES:
        raise ValueError("Invalid output format or quality.")
    if stage_only and preview_frames:
        raise ValueError("Choose either staging only or frame previews.")
    frames = tuple(sorted(set(preview_frames)))
    if len(frames) > 12 or any(type(frame) is not int or not 0 <= frame < timeline["duration_frames"] for frame in frames):
        raise ValueError("Select up to 12 unique frame numbers within the timeline.")
    concurrency = str(workers or settings.get("workers") or "")
    if concurrency and not re.fullmatch(r"auto|[1-9][0-9]?", concurrency):
        raise ValueError("Workers must be auto or an integer between 1 and 99.")
    ids = {scene["id"] for scene in timeline["scenes"]}
    if unknown := sorted(replacements.keys() - ids):
        raise ValueError(f"Unknown shot IDs: {', '.join(unknown)}. Available shots: {', '.join(sorted(ids))}.")
    public = (source / "public").resolve()
    if not public.is_relative_to(source):
        raise ValueError("The project media directory must stay within the composition.")
    assets: dict[str, Path] = {}
    for role in ("narration", "music"):
        if name := timeline["audio"].get(role):
            assets[name] = _project_asset(public, name)
    if output_format == "gif" and assets:
        raise ValueError("GIF delivery requires a silent timeline.")
    fps = timeline["fps"]
    if output_format == "gif" and fps > 50:
        raise ValueError("GIF delivery supports at most 50 fps.")
    for scene in timeline["scenes"]:
        if scene["id"] in replacements:
            path = replacements[scene["id"]].expanduser().resolve(strict=True)
            suffix = path.suffix.lower()
            if not path.is_file() or suffix not in IMAGE_TYPES | VIDEO_TYPES:
                raise ValueError(f"Unsupported replacement media: {path}")
            name = f"shot_{file_digest(path)}{suffix}"
            kind = "image" if suffix in IMAGE_TYPES else "video"
            scene["media"] = {"src": name, "kind": kind, "fit": "cover", "trim_start_frame": 0}
            scene.pop("pending_producer", None)
            if scene.get("producer"):
                scene["producer"] = "local_media"
        elif media := scene.get("media"):
            if not isinstance(media, dict):
                raise ValueError(f"Invalid media for shot {scene['id']}.")
            path = _project_asset(public, media.get("src"))
        else:
            continue
        _validate_shot_media(scene["media"], path, fps, scene["duration_frames"])
        assets[scene["media"]["src"]] = path

    parent = source.parent if (source / "revision.json").is_file() else source.parent / "revisions"
    if output_dir is not None:
        parent = output_dir.expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    job = Path(tempfile.mkdtemp(prefix="remotion-", dir=parent))
    stage_remotion_runtime(job)
    if assets:
        (job / "public").mkdir()
    asset_receipts = {}
    for filename, path in sorted(assets.items()):
        target = job / "public" / filename
        shutil.copyfile(path, target)
        asset_receipts[filename] = file_digest(target)
    timeline_path = job / "input_props.json"
    settings_path = job / "render_request.json"
    delivery_settings = {
        "render_mode": "video", "video_format": output_format, "video_quality": quality, "workers": concurrency,
    }
    revision_path = job / "revision.json"
    revision = {
        "version": "vex-remotion-revision-v1",
        "source": str(source),
        "source_timeline_sha256": original_digest,
        "replaced_shots": sorted(replacements),
        "asset_sha256": asset_receipts,
        "status": "preparing",
    }
    _write(revision_path, revision)
    try:
        from video_generation.film_project import materialize_pending_shots

        native_commands = materialize_pending_shots(source, job, timeline, quality=quality, workers=concurrency)
        for scene in timeline["scenes"]:
            if media := scene.get("media"):
                if media["src"] not in asset_receipts:
                    asset_receipts[media["src"]] = file_digest(job / "public" / media["src"])
        _validate_timeline(props)
        _write(timeline_path, props)
        _write(settings_path, delivery_settings)
        revision.update(timeline_sha256=payload_digest(props), native_commands=[item.to_dict() for item in native_commands])
        if stage_only:
            receipt = {"passed": True, "status": "prepared"}
            _write(revision_path, {**revision, "status": "prepared", "receipt": receipt})
            return job, revision_path, receipt
        if frames:
            try:
                _write(settings_path, {**delivery_settings, "render_mode": "project_stills", "frames": list(frames)})
                command = run_remotion_job(job, workers=concurrency, auto_install_runtime=True, name="remotion_preview")
            finally:
                _write(settings_path, delivery_settings)
            previews = job / "remotion_preview_frames"
            preview_paths = [previews / f"frame_{frame:08d}.png" for frame in frames]
            if command.returncode or any(not path.is_file() for path in preview_paths):
                raise RuntimeError(f"Frame preview failed. See {command.log_path}")
            receipt = {"passed": True, "status": "previewed", "frames": [
                {"frame": frame, "path": path.relative_to(job).as_posix(), "sha256": file_digest(path)}
                for frame, path in zip(frames, preview_paths, strict=True)
            ]}
            _write(revision_path, {**revision, "status": "previewed", "receipt": receipt, "command": command.to_dict()})
            return previews, revision_path, receipt
        video, metadata, commands, receipt, rendering = render_revision_delivery(
            source, job, output_format=output_format, workers=concurrency, full_render=full_render,
        )
    except Exception as exc:
        _write(revision_path, {**revision, "status": "failed", "error": str(exc)})
        raise RuntimeError(f"Remotion revision failed. See {revision_path}. {exc}") from exc
    _write(revision_path, {
        **revision,
        "status": "verified" if receipt["passed"] else "rejected",
        "output": video.name,
        "metadata": metadata,
        "command": commands[-1].to_dict(),
        "commands": [record.to_dict() for record in commands],
        "rendering": rendering,
        "receipt": receipt,
    })
    return video, revision_path, receipt


def _validate_timeline(props: dict[str, Any], *, allow_pending: bool = False) -> dict[str, Any]:
    timeline = props.get("timeline")
    if not isinstance(timeline, dict) or timeline.get("version") != TIMELINE_VERSION:
        raise ValueError("Unsupported Remotion project timeline version.")
    for key, minimum, maximum in (
        ("width", 320, 7680), ("height", 320, 7680), ("fps", 1, 240),
        ("duration_frames", 1, 2**31 - 1),
    ):
        value = timeline.get(key)
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"Invalid timeline {key}.")
    if timeline["width"] % 2 or timeline["height"] % 2:
        raise ValueError("Remotion video canvas dimensions must be even.")
    scenes = timeline.get("scenes")
    captions = timeline.get("captions")
    audio = timeline.get("audio")
    if not isinstance(scenes, list) or not scenes or not isinstance(captions, list) or not isinstance(audio, dict):
        raise ValueError("Timeline requires scenes, captions, and audio tracks.")
    volume = audio.get("music_volume", 0)
    if type(volume) not in {int, float} or not 0 <= volume <= 1:
        raise ValueError("Music volume must be between 0 and 1.")
    for role in ("narration", "music"):
        if not isinstance(audio.get(role, ""), str):
            raise ValueError(f"Invalid {role} asset name.")
    ids = set()
    cursor = 0
    for scene in scenes:
        start, end = _span(scene)
        pending = scene.get("pending_producer")
        if pending and (not allow_pending or pending not in {"manim", "hyperframes"}):
            raise ValueError("The timeline has an unsupported or unprepared native shot.")
        shot_id = scene.get("id")
        if not isinstance(shot_id, str) or not shot_id or shot_id in ids or start != cursor:
            raise ValueError("Shots must have unique IDs and cover the timeline contiguously.")
        program = scene.get("program")
        if program is not None and not isinstance(program, dict):
            raise ValueError(f"Invalid program for shot {shot_id}.")
        standfirst = scene.get("standfirst", "")
        if not isinstance(standfirst, str):
            raise ValueError(f"Invalid text for shot {shot_id}.")
        if "artwork" in scene:
            from video_generation.film_artwork import validate_artwork
            scene["artwork"] = validate_artwork(scene["artwork"])
        if program is None and not standfirst.strip() and not any(scene.get(key) for key in ("media", "artwork", "editorial", "pending_producer")):
            raise ValueError(f"Shot {shot_id} has no program, artwork, text, or media.")
        ids.add(shot_id)
        cursor = end
    if cursor != timeline["duration_frames"]:
        raise ValueError("The final shot must end on the timeline boundary.")
    cursor = 0
    for caption in captions:
        start, end = _span(caption)
        text = caption.get("text")
        if start < cursor or end > timeline["duration_frames"] or not isinstance(text, str) or not text.strip():
            raise ValueError("Captions must have text and ordered, non-overlapping in-bounds intervals.")
        cursor = end
    return timeline


def _span(item: Any) -> tuple[int, int]:
    if not isinstance(item, dict):
        raise ValueError("Expected a frame interval object.")
    start, count = item.get("start_frame"), item.get("duration_frames")
    if type(start) is not int or type(count) is not int or start < 0 or count <= 0:
        raise ValueError("Frame intervals require a nonnegative start and positive frame count.")
    return start, start + count


def _validate_shot_media(media: Any, path: Path, fps: int, frames: int) -> None:
    if not isinstance(media, dict) or media.get("fit") not in {"cover", "contain"}:
        raise ValueError("Shot media fit must be cover or contain.")
    trim = media.get("trim_start_frame", 0)
    if type(trim) is not int or trim < 0:
        raise ValueError("Clip trim_start_frame must be a nonnegative frame index.")
    extension = path.suffix.lower()
    if media.get("kind") == "image" and extension in IMAGE_TYPES and trim == 0:
        with Image.open(path) as image:
            image.verify()
    elif media.get("kind") == "video" and extension in VIDEO_TYPES:
        probe = probe_video(str(path))
        duration = probe.get("video_duration_sec") or probe["duration_sec"]
        available_frames = math.floor(float(duration) * fps + 0.00001)
        if not probe["width"] or available_frames < trim + frames:
            raise ValueError(f"{path.name} does not contain enough video for this shot ({(trim + frames) / fps:.3f}s required).")
    else:
        raise ValueError("Shot media must be a PNG, JPEG, WebP, MP4, MOV, or WebM file.")


def _project_asset(public: Path, name: Any) -> Path:
    if not isinstance(name, str) or not ASSET_NAME.fullmatch(name) or name.endswith("."):
        raise ValueError("Project assets must be simple filenames within the public directory.")
    path = (public / name).resolve(strict=True)
    if not path.is_relative_to(public) or not path.is_file():
        raise ValueError(f"Invalid project asset {name}.")
    return path


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object.")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
