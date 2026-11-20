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
from video_generation.remotion_delivery import check_remotion_delivery
from video_generation.remotion_project import TIMELINE_VERSION, render_remotion_project, stage_remotion_runtime


IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_TYPES = {".mp4", ".mov", ".webm"}
ASSET_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,149}")


def revise_remotion_project(
    project: Path,
    *,
    replacements: dict[str, Path],
    output_dir: Path | None = None,
    workers: str | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    source = project.expanduser().resolve(strict=True)
    if (source / "remotion" / "input_props.json").is_file():
        source = source / "remotion"
    props = _read_object(source / "input_props.json")
    settings = _read_object(source / "render_request.json")
    timeline = _validate_timeline(props)
    if (
        settings.get("render_mode") != "video"
        or settings.get("video_format") not in VALID_FORMATS
        or settings.get("video_quality") not in VALID_QUALITIES
    ):
        raise ValueError("Invalid Remotion video export settings.")
    output_format = settings["video_format"]
    concurrency = str(workers or settings.get("workers") or "")
    if concurrency and not re.fullmatch(r"auto|[1-9][0-9]?", concurrency):
        raise ValueError("Workers must be auto or an integer between 1 and 99.")
    original_digest = payload_digest(props)
    ids = {scene["id"] for scene in timeline["scenes"]}
    if unknown := sorted(replacements.keys() - ids):
        raise ValueError(f"Unknown shot IDs: {', '.join(unknown)}. Available shots: {', '.join(sorted(ids))}.")
    public = (source / "public").resolve()
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
        elif media := scene.get("media"):
            if not isinstance(media, dict):
                raise ValueError(f"Invalid media for shot {scene['id']}.")
            path = _project_asset(public, media.get("src"))
        else:
            continue
        _validate_shot_media(scene["media"], path, fps, scene["duration_frames"])
        assets[scene["media"]["src"]] = path

    parent = output_dir.expanduser().resolve() if output_dir else source.parent / "revisions"
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
    _write(timeline_path, props)
    _write(job / "render_request.json", {
        "render_mode": "video",
        "video_format": output_format,
        "video_quality": settings["video_quality"],
        "workers": concurrency,
    })
    revision_path = job / "revision.json"
    revision = {
        "version": "vex-remotion-revision-v1",
        "source": str(source),
        "source_timeline_sha256": original_digest,
        "timeline_sha256": payload_digest(props),
        "replaced_shots": sorted(replacements),
        "asset_sha256": asset_receipts,
        "status": "rendering",
    }
    _write(revision_path, revision)
    try:
        video, metadata, command = render_remotion_project(
            job, output_format=output_format, workers=concurrency,
        )
        receipt = check_remotion_delivery(timeline_path, video, metadata)
    except Exception as exc:
        _write(revision_path, {**revision, "status": "failed", "error": str(exc)})
        raise RuntimeError(f"Remotion revision failed. See {revision_path}. {exc}") from exc
    _write(revision_path, {
        **revision,
        "status": "verified" if receipt["passed"] else "rejected",
        "output": video.name,
        "metadata": metadata,
        "command": command.to_dict(),
        "receipt": receipt,
    })
    return video, revision_path, receipt


def _validate_timeline(props: dict[str, Any]) -> dict[str, Any]:
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
        shot_id = scene.get("id")
        if not isinstance(shot_id, str) or not shot_id or shot_id in ids or start != cursor:
            raise ValueError("Shots must have unique IDs and cover the timeline contiguously.")
        program = scene.get("program")
        if program is not None and not isinstance(program, dict):
            raise ValueError(f"Invalid program for shot {shot_id}.")
        standfirst = scene.get("standfirst", "")
        if not isinstance(standfirst, str):
            raise ValueError(f"Invalid text for shot {shot_id}.")
        if program is None and not standfirst.strip() and not scene.get("media"):
            raise ValueError(f"Shot {shot_id} has no program, text, or media.")
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
        available_frames = math.floor(float(probe["duration_sec"]) * fps + 0.00001)
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
