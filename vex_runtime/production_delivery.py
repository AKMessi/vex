"""Publish a saved production cut without editing the project."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import config
from engine import _run_command, probe_video
from state import ProjectState
from vex_runtime.edit_graph import rational
from vex_runtime.production_edits import verified_composition
from vex_visuals.evidence import file_digest, payload_digest
from video_generation.production_captions import serialize_captions
from video_generation.production_timeline import _frame, apply_timeline_edits
from video_generation.remotion_revisions import _read_object, _validate_timeline, revise_remotion_project


VERSION = "vex-delivery-v1"
MANIFEST = "delivery.json"
VIDEO_FILES = {"cut.mp4", "cut-captioned.mp4", "cut-clean.mp4"}
TEXT_FILES = {"captions.srt", "captions.vtt", "chapters.vtt"}


def _require_clock(metadata: dict, timeline: dict, frames: int) -> None:
    fps = timeline["fps"]
    count = metadata["video_frame_count"]
    duration = metadata.get("video_duration_sec") or metadata["duration_sec"]
    # WebM often reports duration but no frame count in the stream header.
    tolerance = 0.1 if count is not None else max(1, fps * 0.05)
    if (
        (metadata["width"], metadata["height"]) != (timeline["width"], timeline["height"])
        or abs(metadata["fps"] - fps) >= 0.01
        or (count is not None and count != frames)
        or abs(duration * fps - frames) > tolerance
    ):
        raise ValueError("The saved video and production timeline disagree on the frame clock.")


def _compatible_video(metadata: dict) -> bool:
    return metadata["codec"] == "h264" and metadata["pix_fmt"] == "yuv420p"


def _write_video(picture: Path, sound: Path, picture_info: dict, sound_info: dict, output: Path, *,
                 start: int, end: int, fps: int, total_frames: int) -> str:
    full = start == 0 and end == total_frames
    copy_picture = full and _compatible_video(picture_info)
    has_sound = sound_info["has_audio"]
    copy_sound = full and (not has_sound or sound_info["audio_codec"] == "aac")
    separate_sound = has_sound and picture != sound
    command = [config.FFMPEG_PATH, "-hide_banner", "-loglevel", "error", "-nostdin", "-xerror", "-i", str(picture)]
    if separate_sound:
        command.extend(["-i", str(sound)])
    command.extend(["-map", "0:v:0", "-map_metadata", "-1", "-map_chapters", "-1"])
    if copy_picture:
        command.extend(["-c:v", "copy"])
    else:
        command.extend([
            "-vf", f"trim=start_frame={start}:end_frame={end},settb=1/{fps},setpts=N",
            "-r", str(fps), "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
        ])
    if has_sound:
        command.extend(["-map", f"{int(separate_sound)}:a:0"])
        if copy_sound:
            command.extend(["-c:a", "copy"])
        else:
            first, last = (round(frame * 48_000 / fps) for frame in (start, end))
            command.extend([
                "-af", f"aresample=48000:async=1:first_pts=0,apad,atrim=start_sample={first}:end_sample={last},asetpts=PTS-STARTPTS",
                "-c:a", "aac", "-b:a", "192k",
            ])
    _run_command([*command, "-movflags", "+faststart", str(output)], "Delivery video could not be exported")
    return "remuxed" if copy_picture and copy_sound else "encoded"


def _package_file(bundle: ZipFile, name: str, path: Path) -> dict:
    size = 0
    digest = hashlib.sha256()
    with path.open("rb") as stream, bundle.open(name, "w", force_zip64=path.stat().st_size >= 2**31 - 1) as member:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            member.write(block)
            digest.update(block)
            size += len(block)
    return {"bytes": size, "sha256": digest.hexdigest()}


def _sidecars(folder: Path, timeline: dict, start: int, end: int) -> tuple[list[Path], int]:
    fps = timeline["fps"]
    captions = []
    for cue in timeline["captions"]:
        left = max(start, cue["start_frame"])
        right = min(end, cue["start_frame"] + cue["duration_frames"])
        if left < right:
            captions.append({"start_frame": left - start, "duration_frames": right - left, "text": cue["text"]})
    files = []
    if captions:
        for kind in ("srt", "vtt"):
            path = folder / f"captions.{kind}"
            path.write_text(serialize_captions(captions, fps, kind=kind), encoding="utf-8", newline="\n")
            files.append(path)
    markers = {}
    for marker in timeline.get("markers", []):
        frame, title = marker["start_frame"], marker["text"].strip()
        if start <= frame < end and title:
            markers.setdefault(frame - start, []).append(title)
    if markers:
        starts = sorted(markers)
        chapters = [{"start_frame": left, "duration_frames": right - left, "text": " / ".join(markers[left])}
                    for left, right in zip(starts, [*starts[1:], end - start], strict=True)]
        path = folder / "chapters.vtt"
        path.write_text(serialize_captions(chapters, fps, kind="vtt"), encoding="utf-8", newline="\n")
        files.append(path)
    return files, len(captions)


def _sealed_source(manifest: Path, master: Path, props_digest: str, sha256: str) -> dict:
    report = _read_object(manifest)
    receipt = report.get("receipt") or report.get("qa", {}).get("evidence", {}).get("visual_quality") or {}
    if receipt.get("output_sha256", sha256) != sha256 or receipt.get("timeline_sha256", props_digest) != props_digest:
        raise ValueError("The saved production no longer matches its delivery receipt.")
    assets = report.get("asset_sha256", {})
    recorded = master.parent / MANIFEST
    if not assets and recorded.is_file():
        record = _read_object(recorded)
        delivery = record.get("delivery", {})
        if delivery.get("output_sha256") == sha256 and delivery.get("timeline_sha256") == props_digest:
            assets = record.get("dependencies", {}).get("assets", {})
    return assets


def publish_delivery(state: ProjectState, destination: Path, request: Mapping, *,
                     on_progress: Callable[[str, float], None] | None = None) -> dict:
    """Write one ZIP for a pinned revision and a half-open range of its frame clock."""
    if type(request.get("revision")) is not int or request["revision"] != state.revision:
        raise ValueError("The cut changed. Refresh it before exporting.")
    mode = request.get("captions", "burned")
    if mode not in ("burned", "clean", "both"):
        raise ValueError("Choose burned, clean, or both for the video captions.")
    destination = destination.expanduser().absolute()
    if destination.suffix.lower() != ".zip":
        raise ValueError("Choose a .zip destination for the delivery package.")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("The delivery destination already exists.")
    manifest, master, trims = verified_composition(state)
    source = master.parent
    props_path = source / "input_props.json"
    props = _read_object(props_path)
    props_digest = payload_digest(props)
    master_sha = file_digest(master)
    expected_assets = _sealed_source(manifest, master, props_digest, master_sha)
    base = _validate_timeline(props)
    timeline = apply_timeline_edits(base, trims)
    fps, frames = timeline["fps"], timeline["duration_frames"]
    first, last = (request.get(key) for key in ("start", "end"))
    start = 0 if first is None else _frame(rational(first), fps)
    end = frames if last is None else _frame(rational(last), fps)
    if not 0 <= start < end <= frames:
        raise ValueError("Select at least one frame within the saved cut. The end time is exclusive.")
    offset = sum(_frame(trim["start"], fps) for trim in trims)
    master_info = probe_video(str(master))
    _require_clock(master_info, base, base["duration_frames"])
    if on_progress:
        on_progress("Preparing delivery files", 0.12)
    with tempfile.TemporaryDirectory(prefix=".vex-delivery-", dir=destination.parent) as temporary:
        folder = Path(temporary)
        sidecars, caption_count = _sidecars(folder, timeline, start, end)
        files = {path.name: path for path in sidecars}
        clean, clean_info = master, master_info
        if mode != "burned" and caption_count:
            if on_progress:
                on_progress("Rendering without editable captions", 0.20)
            clean, clean_manifest, receipt = revise_remotion_project(
                source, replacements={}, include_captions=False, output_format="mp4", output_dir=folder / "render",
            )
            if not receipt["passed"]:
                raise ValueError("The clean video did not pass render verification.")
            clean_inputs = _read_object(clean_manifest)
            if (clean_inputs["source_timeline_sha256"] != props_digest or any(
                clean_inputs["asset_sha256"].get(name) != digest for name, digest in expected_assets.items()
            )):
                raise ValueError("The clean render used different inputs from the saved production.")
            clean_info = probe_video(str(clean))
            _require_clock(clean_info, base, base["duration_frames"])
        if on_progress:
            on_progress("Exporting the selected frames", 0.65)
        videos = []
        variants = ([("cut-captioned.mp4", master, master_info), ("cut-clean.mp4", clean, clean_info)]
                    if mode == "both" and caption_count else
                    [("cut.mp4", master, master_info) if mode == "burned" else ("cut.mp4", clean, clean_info)])
        for filename, picture, picture_info in variants:
            if (picture == master and master.suffix.lower() == ".mp4" and offset + start == 0
                    and offset + end == base["duration_frames"] and _compatible_video(master_info)
                    and (not master_info["has_audio"] or master_info["audio_codec"] == "aac")):
                output, info, method = master, master_info, "copied"
            else:
                output = folder / filename
                method = _write_video(picture, master, picture_info, master_info, output, start=offset + start,
                                      end=offset + end, fps=fps, total_frames=base["duration_frames"])
                info = probe_video(str(output))
            _require_clock(info, timeline, end - start)
            if (not _compatible_video(info) or info["video_frame_count"] != end - start
                    or info["has_audio"] != master_info["has_audio"] or (info["has_audio"] and info["audio_codec"] != "aac")):
                raise ValueError("The delivery video failed its codec, frame count, or audio checks.")
            videos.append({"file": filename, "method": method, "audio": info["has_audio"]})
            files[filename] = output
        if on_progress:
            on_progress("Packaging the verified delivery", 0.90)
        report = {
            "version": VERSION, "project_id": state.project_id, "revision": state.revision,
            "cut_sha256": master_sha, "timeline_sha256": payload_digest(timeline),
            "start_frame": start, "end_frame": end, "fps": fps, "width": timeline["width"], "height": timeline["height"],
            "caption_mode": mode, "caption_count": caption_count, "videos": videos,
        }
        archive = folder / "delivery.zip"
        with ZipFile(archive, "w") as bundle:
            report["files"] = {name: _package_file(bundle, name, path) for name, path in files.items()}
            archived_master = next((report["files"][name]["sha256"] for name, path in files.items() if path == master), None)
            if (archived_master or file_digest(master)) != master_sha or payload_digest(_read_object(props_path)) != props_digest:
                raise ValueError("The master video or timeline changed during export. Retry from the saved cut.")
            bundle.writestr(MANIFEST, json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
        # Publish atomically without overwriting another export.
        if os.name == "nt":
            archive.rename(destination)
        else:
            os.link(archive, destination)
    return report


def read_delivery(path: Path, *, verify: bool = False) -> dict:
    """Read a delivery inventory, optionally checking every packaged byte without extracting it."""
    try:
        with ZipFile(path) as bundle:
            if bundle.getinfo(MANIFEST).file_size > 64 * 1024:
                raise ValueError("The delivery manifest is too large.")
            report = json.loads(bundle.read(MANIFEST))
            if not isinstance(report, dict) or report.get("version") != VERSION or not isinstance(report.get("files"), dict):
                raise ValueError("Unsupported Vex delivery manifest.")
            if (any(type(report.get(key)) is not int for key in ("revision", "start_frame", "end_frame", "fps", "width", "height"))
                    or not 0 <= report["start_frame"] < report["end_frame"] or report["fps"] <= 0
                    or report.get("caption_mode") not in ("burned", "clean", "both")):
                raise ValueError("Invalid Vex delivery frame clock or caption mode.")
            files = report["files"]
            names = bundle.namelist()
            if (not VIDEO_FILES.intersection(files) or files.keys() - VIDEO_FILES - TEXT_FILES
                    or len(names) != len(files) + 1 or set(names) != {MANIFEST, *files}):
                raise ValueError("The ZIP does not match the delivery inventory.")
            if verify:
                for filename, receipt in files.items():
                    if not isinstance(receipt, dict) or bundle.getinfo(filename).file_size != receipt.get("bytes"):
                        raise ValueError(f"Delivery size check failed for {filename}.")
                    with bundle.open(filename) as stream:
                        digest = hashlib.file_digest(stream, "sha256").hexdigest()
                    if digest != receipt.get("sha256"):
                        raise ValueError(f"Delivery checksum failed for {filename}.")
    except (BadZipFile, KeyError, UnicodeDecodeError, RuntimeError, NotImplementedError) as exc:
        raise ValueError("The Vex delivery ZIP is damaged or unsupported.") from exc
    return report
