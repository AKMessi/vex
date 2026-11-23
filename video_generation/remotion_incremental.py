"""Reuse a verified delivery when changes fit within a few locked shot ranges."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

import config
from engine import VideoEngineError, _ffmpeg_render_timeout_sec, probe_video
from vex_visuals.evidence import file_digest, payload_digest
from video_generation.remotion_project import render_remotion_project, run_remotion_job
from video_generation.renderer import CommandRecord, VideoGenerationRuntimeError, _run_logged


LOGGER = logging.getLogger(__name__)
RECEIPT_NAME = "delivery.json"
RUNTIME_FILES = (
    "entry.jsx", "remotion_scene_graph.jsx", "film_motion.mjs", "visual_motion.mjs",
    "visual_telemetry.mjs", "visual_fonts.mjs",
)
MAX_RANGES = 3
MAX_REUSE_PASSES = 2
Range = tuple[int, int]


@dataclass(frozen=True)
class ShotReuse:
    output: Path
    ranges: tuple[Range, ...]
    changed: frozenset[str]
    delivery: dict[str, Any]
    passes: int
    bundle_fingerprint: str


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object.")
    return value


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _local_file(root: Path, relative: str) -> Path:
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Render input escapes its job directory.")
    return path


def _dependencies(directory: Path, timeline: dict) -> dict[str, dict[str, str]]:
    from video_generation.production_timeline import audio_assets, picture_assets

    root = directory.resolve(strict=True)
    media_names = audio_assets(timeline) | picture_assets(timeline)
    media_names.difference_update({None, ""})
    if any(not isinstance(name, str) or "/" in name or "\\" in name for name in media_names):
        raise ValueError("A staged media asset must use a local filename.")
    runtime = {name: file_digest(_local_file(root, name)) for name in RUNTIME_FILES}
    assets = {
        name: file_digest(_local_file(root, f"public/{name}"))
        for name in sorted(media_names)
    }
    return {"runtime": runtime, "assets": assets}


def record_delivery(timeline_path: Path, output: Path, delivery: dict, *, reuse_passes: int = 0) -> None:
    """Seal a successful MP4 to its authored timeline, runtime and staged media."""
    if output.name != "final.mp4" or not delivery["passed"]:
        return
    root = output.parent
    try:
        props = _read(timeline_path)
        settings = _read(root / "render_request.json")
        if (
            settings.get("render_mode") != "video" or settings.get("video_format") != "mp4"
            or payload_digest(props) != delivery["timeline_sha256"]
        ):
            return
        bundle = _read(root / "remotion_result.json").get("bundle_fingerprint")
        if not isinstance(bundle, str) or len(bundle) != 64:
            return
        record = {
            "version": 1,
            "dependencies": _dependencies(root, props["timeline"]),
            "quality": settings["video_quality"],
            "reuse_passes": reuse_passes,
            "bundle_fingerprint": bundle,
            "delivery": delivery,
        }
        temporary = root / "delivery.tmp"
        _write(temporary, {**record, "sha256": payload_digest(record)})
        os.replace(temporary, root / RECEIPT_NAME)
    except (OSError, ValueError, KeyError, TypeError):
        LOGGER.debug("Could not record a reusable final delivery for %s", root, exc_info=True)


def _shot_ranges(before: dict, after: dict) -> tuple[tuple[Range, ...], frozenset[str]]:
    visual_fields = {"scenes", "captions", "overlays"}
    if {key: value for key, value in before.items() if key not in visual_fields} != {
        key: value for key, value in after.items() if key not in visual_fields
    }:
        raise ValueError("The revision changed the canvas, clock or audio.")
    # Caption IDs exist solely for editing. Compare the values the renderer consumes.
    def visible_cues(timeline):
        try:
            return {(cue["start_frame"], cue["duration_frames"], cue["text"]) for cue in timeline["captions"]}
        except (KeyError, TypeError) as exc:
            raise ValueError("Cannot compare captions or audio for frame reuse.") from exc

    intervals = [(start, start + frames) for start, frames, _ in visible_cues(before) ^ visible_cues(after)]
    old_overlays = {row["id"]: row for row in before.get("overlays", [])}
    new_overlays = {row["id"]: row for row in after.get("overlays", [])}
    for identity in old_overlays.keys() | new_overlays.keys():
        old, new = old_overlays.get(identity), new_overlays.get(identity)
        if old != new:
            intervals.extend((row["start_frame"], row["start_frame"] + row["duration_frames"])
                             for row in (old, new) if row)
    identity = ("id", "start_frame", "duration_frames")
    ranges: list[Range] = []
    changed = set()
    for old, new in zip(before["scenes"], after["scenes"], strict=True):
        if any(old[key] != new[key] for key in identity) or old.get("pending_producer") or new.get("pending_producer"):
            raise ValueError("The revision changed shot boundaries or has a pending shot.")
        start = new["start_frame"]
        end = start + new["duration_frames"]
        if old != new:
            intervals.append((start, end))
    for start, end in sorted(intervals):
        if ranges and ranges[-1][1] >= start:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], end))
        else:
            ranges.append((start, end))
    for shot in after["scenes"]:
        start = shot["start_frame"]
        if any(left < start + shot["duration_frames"] and start < right for left, right in ranges):
            changed.add(shot["id"])
    if not ranges or len(ranges) > MAX_RANGES or sum(end - start for start, end in ranges) >= before["duration_frames"]:
        raise ValueError("The revision needs a full render for its shot coverage.")
    return tuple(ranges), frozenset(changed)


def _plan_reuse(source: Path, job: Path) -> ShotReuse:
    from video_generation.production_timeline import audio_assets

    record = _read(source / RECEIPT_NAME)
    signature = record.pop("sha256", "")
    delivery = record["delivery"]
    if record.get("version") != 1 or signature != payload_digest(record) or delivery.get("passed") is not True:
        raise ValueError("The previous delivery receipt is invalid.")
    passes = record.get("reuse_passes", 0)
    if type(passes) is not int or not 0 <= passes < MAX_REUSE_PASSES:
        raise ValueError("Render from the composition to refresh the picture after repeated revisions.")
    before_props = _read(source / "input_props.json")
    after_props = _read(job / "input_props.json")
    before, after = before_props["timeline"], after_props["timeline"]
    if (
        delivery["timeline_sha256"] != payload_digest(before_props)
        or {key: value for key, value in before_props.items() if key != "timeline"}
        != {key: value for key, value in after_props.items() if key != "timeline"}
        or record["quality"] != _read(job / "render_request.json").get("video_quality")
    ):
        raise ValueError("The source timeline or export settings changed.")
    ranges, changed = _shot_ranges(before, after)
    original = _dependencies(source, before)
    candidate = _dependencies(job, after)
    if original != record["dependencies"] or original["runtime"] != candidate["runtime"]:
        raise ValueError("The source media or rendering runtime changed.")
    retained = audio_assets(before)
    retained.update(
        old["media"]["src"] for old, new in zip(before["scenes"], after["scenes"], strict=True)
        if old == new and old.get("media")
    )
    retained.update(row["media"]["src"] for row in before.get("overlays", []) if row in after.get("overlays", []))
    if any(original["assets"][name] != candidate["assets"][name] for name in retained if name):
        raise ValueError("An unchanged shot or audio track has different media.")
    output = _local_file(source.resolve(), "final.mp4")
    if file_digest(output) != delivery["output_sha256"]:
        raise ValueError("The previous export no longer matches its receipt.")
    if not _matches_clock(probe_video(str(output)), before, before["duration_frames"]):
        raise ValueError("The previous export has an incompatible MP4 frame clock.")
    return ShotReuse(output, ranges, changed, delivery, passes, record["bundle_fingerprint"])


def _matches_clock(metadata: dict, timeline: dict, frames: int) -> bool:
    fps = timeline["fps"]
    return (
        metadata.get("codec") == "h264" and metadata.get("pix_fmt") == "yuv420p"
        and metadata.get("width") == timeline["width"] and metadata.get("height") == timeline["height"]
        and metadata.get("video_frame_count") == frames
        and abs(float(metadata.get("fps") or 0) - fps) < 0.01
        and abs(float(metadata.get("video_duration_sec") or 0) * fps - frames) < 0.1
    )


def _splice_filter(ranges: tuple[Range, ...], total_frames: int, fps: int) -> str:
    segments: list[tuple[int, int, int]] = []
    cursor = 0
    for index, (start, end) in enumerate(ranges, 1):
        if cursor < start:
            segments.append((0, cursor, start))
        segments.append((index, 0, end - start))
        cursor = end
    if cursor < total_frames:
        segments.append((0, cursor, total_frames))
    base_count = sum(source == 0 for source, _, _ in segments)
    chain = []
    if base_count > 1:
        chain.append(f"[0:v:0]split={base_count}" + "".join(f"[base{i}]" for i in range(base_count)))
    base_index = 0
    for index, (source, start, end) in enumerate(segments):
        if source == 0 and base_count > 1:
            label = f"[base{base_index}]"
            base_index += 1
        else:
            label = f"[{source}:v:0]"
        chain.append(f"{label}trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS,setsar=1[part{index}]")
    ordered = "".join(f"[part{index}]" for index in range(len(segments)))
    chain.append(f"{ordered}concat=n={len(segments)}:v=1:a=0,settb=1/{fps},setpts=N[out]")
    return ";".join(chain)


def _unchanged_frames_match(source: Path, job: Path, previous: dict, current: dict, changed: frozenset[str]) -> bool:
    for before, after in zip(previous["shots"], current["shots"], strict=True):
        if before["id"] != after["id"]:
            return False
        if before["id"] in changed:
            continue
        if not before["frames"] or len(before["frames"]) != len(after["frames"]):
            return False
        for original, revised in zip(before["frames"], after["frames"], strict=True):
            if original["frame"] != revised["frame"]:
                return False
            prior = _local_file(source.resolve(), original["path"])
            if file_digest(prior) != original["sha256"]:
                return False
            with Image.open(prior) as old_frame, Image.open(job / revised["path"]) as new_frame:
                if old_frame.size != new_frame.size:
                    return False
                delta = np.abs(
                    np.asarray(old_frame.convert("RGB"), dtype=np.int16)
                    - np.asarray(new_frame.convert("RGB"), dtype=np.int16),
                )
                if float(delta.mean()) > 3 or float(np.mean(delta.max(axis=2) > 24)) > 0.03:
                    return False
    return True


def _audio_hash(path: Path, job: Path, label: str) -> str:
    command = [
        config.FFMPEG_PATH, "-nostdin", "-v", "error", "-i", str(path),
        "-map", "0:a:0", "-c:a", "copy", "-f", "hash", "-hash", "sha256", "-",
    ]
    log_name = f"remotion_audio_{label}"
    result = _run_logged(command, cwd=job, name=log_name, log_path=job / f"{log_name}.log",
                         timeout=_ffmpeg_render_timeout_sec())
    checksum = result.stdout.strip()
    if result.returncode or not checksum.startswith("SHA256="):
        raise ValueError("Could not verify the copied audio track.")
    return checksum


def _attach_rendering(delivery: dict, rendering: dict) -> dict:
    evidence = {**delivery, "rendering": rendering}
    evidence.pop("receipt_sha256", None)
    return {**evidence, "receipt_sha256": payload_digest(evidence)}


def render_revision_delivery(
    source: Path, job: Path, *,
    output_format: str, workers: str = "", auto_install_runtime: bool = True, full_render: bool = False,
) -> tuple[Path, dict, list[CommandRecord], dict, dict]:
    from video_generation.remotion_delivery import check_remotion_delivery

    timeline_path = job / "input_props.json"
    settings_path = job / "render_request.json"
    timeline = _read(timeline_path)["timeline"]
    total = timeline["duration_frames"]
    commands = []
    reason = "full_render_requested" if full_render else "format_requires_full_render"
    if output_format == "mp4" and not full_render:
        try:
            plan = _plan_reuse(source, job)
            settings = settings_path.read_bytes()
            try:
                _write(settings_path, {**json.loads(settings), "render_mode": "video_ranges", "frame_ranges": plan.ranges})
                record = run_remotion_job(
                    job, workers=workers, auto_install_runtime=auto_install_runtime, name="remotion_shot_ranges",
                )
                commands.append(record)
            finally:
                settings_path.write_bytes(settings)
            if record.returncode:
                raise RuntimeError(f"Shot rendering failed. See {record.log_path}")
            runner = _read(job / "remotion_result.json")
            expected = [{"start": start, "end": end} for start, end in plan.ranges]
            if runner.get("rendered_ranges") != expected:
                raise ValueError("The renderer returned different frame ranges.")
            if runner.get("bundle_fingerprint") != plan.bundle_fingerprint:
                raise ValueError("The revision loaded a different renderer dependency bundle.")
            command = [config.FFMPEG_PATH, "-hide_banner", "-nostdin", "-v", "error", "-i", str(plan.output)]
            for index, (start, end) in enumerate(plan.ranges):
                clip = job / "shot-ranges" / f"{index}.mp4"
                clip_metadata = probe_video(str(clip))
                if clip_metadata.get("has_audio") or not _matches_clock(clip_metadata, timeline, end - start):
                    raise ValueError("A shot clip has an incompatible frame clock.")
                command.extend(["-i", str(clip)])
            output = job / "final.mp4"
            fps = timeline["fps"]
            command.extend([
                "-filter_complex", _splice_filter(plan.ranges, total, fps),
                "-map", "[out]", "-map", "0:a:0?", "-c:a", "copy", "-c:v", "libx264",
                "-preset", "fast", "-crf", "14", "-pix_fmt", "yuv420p", "-r", str(fps),
                "-colorspace", "bt709", "-color_range", "tv", "-movflags", "+faststart", "-y", str(output),
            ])
            splice = _run_logged(
                command, cwd=job, name="remotion_splice", log_path=job / "remotion_splice.log",
                timeout=_ffmpeg_render_timeout_sec(),
            )
            commands.append(splice)
            if splice.returncode:
                raise RuntimeError(f"Could not assemble the shot ranges. See {splice.log_path}")
            metadata = probe_video(str(output))
            if not _matches_clock(metadata, timeline, total):
                raise ValueError("The assembled output has an incompatible frame clock.")
            delivery = check_remotion_delivery(timeline_path, output, metadata, reusable=False)
            if not delivery["passed"] or not _unchanged_frames_match(source, job, plan.delivery, delivery, plan.changed):
                raise ValueError("The assembled output did not preserve the unchanged shots.")
            audio_copied = bool(metadata.get("has_audio"))
            if audio_copied and _audio_hash(plan.output, job, "source") != _audio_hash(output, job, "revision"):
                raise ValueError("The copied audio track does not match the source.")
            count = sum(end - start for start, end in plan.ranges)
            rendering = {
                "mode": "shot_ranges", "rendered_frames": count, "reused_frames": total - count,
                "ranges": expected, "audio_copied": audio_copied,
            }
            delivery = _attach_rendering(delivery, rendering)
            record_delivery(timeline_path, output, delivery, reuse_passes=plan.passes + 1)
            return output, metadata, commands, delivery, rendering
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, VideoEngineError) as exc:
            reason = "no_verified_source" if not (source / RECEIPT_NAME).is_file() else str(exc)
            LOGGER.debug("Full render fallback for %s: %s", job, reason)

    output, metadata, record = render_remotion_project(
        job, output_format=output_format, workers=workers, auto_install_runtime=auto_install_runtime,
    )
    commands.append(record)
    delivery = check_remotion_delivery(timeline_path, output, metadata, reusable=False)
    rendering = {
        "mode": "full", "rendered_frames": total, "reused_frames": 0, "reason": reason,
    }
    delivery = _attach_rendering(delivery, rendering)
    record_delivery(timeline_path, output, delivery)
    return output, metadata, commands, delivery, rendering
