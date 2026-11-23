"""Stage local media and adapt source-based edits to the Studio composition."""
from __future__ import annotations

import shutil
from pathlib import Path

import config
from engine import _run_command, probe_video
from state import ProjectState
from tools.path_security import resolve_existing_project_file
from vex_runtime.edit_graph import EditGraph
from vex_visuals.evidence import file_digest, payload_digest


class EditorialMedia:
    def __init__(self, public: Path):
        self.public = public
        self.public.mkdir(parents=True, exist_ok=True)
        self._probe = {}
        self._staged = {}

    def probe(self, path: Path) -> dict:
        if path not in self._probe:
            self._probe[path] = probe_video(str(path))
        return self._probe[path]

    def stage(self, path: Path, *, audio: bool = False) -> str:
        from video_generation.remotion_revisions import IMAGE_TYPES

        cache_key = path, audio
        if cache_key in self._staged:
            return self._staged[cache_key]
        suffix = path.suffix.lower()
        if not audio and suffix in IMAGE_TYPES:
            direct = True
        else:
            metadata = self.probe(path)
            if audio and not metadata["has_audio"]:
                raise ValueError(f"{path.name} contains no audio stream.")
            if not audio and not metadata["width"]:
                raise ValueError(f"{path.name} contains no picture stream.")
            if audio:
                direct = suffix in {".wav", ".mp3", ".m4a", ".mp4", ".webm"}
            else:
                direct = (suffix == ".mp4" and metadata["codec"] == "h264") or (
                    suffix == ".webm" and metadata["codec"] in {"vp8", "vp9", "av1"}
                )
        target = self.public / (file_digest(path) + (suffix if direct else ".m4a" if audio else ".mp4"))
        if not target.exists():
            if direct:
                shutil.copyfile(path, target)
            else:
                encoding = (["-vn", "-map", "0:a:0", "-c:a", "aac", "-b:a", "192k"] if audio else
                            ["-map", "0:v:0", "-map", "0:a:0?", "-c:v", "libx264", "-crf", "15", "-preset", "fast",
                             "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p", "-c:a", "aac"])
                _run_command([config.FFMPEG_PATH, "-nostdin", "-v", "error", "-protocol_whitelist", "file",
                              "-i", str(path), *encoding, "-movflags", "+faststart", str(target)],
                             "Could not prepare editorial media for Studio")
        self._staged[cache_key] = target.name
        return target.name


def footage_timeline(state: ProjectState, depot: EditorialMedia) -> dict:
    from video_generation.remotion_project import TIMELINE_VERSION

    working = Path(state.working_file).resolve(strict=True)
    metadata = depot.probe(working)
    width, height = (max(320, metadata[edge] + metadata[edge] % 2) for edge in ("width", "height"))
    fps = round(metadata["fps"])
    if not 1 <= fps <= 240 or not metadata["width"]:
        raise ValueError("Opening footage requires a video stream with a supported frame clock.")
    graph = EditGraph.from_mapping(state.edit_graph) if state.edit_graph else None
    if graph is None:
        graph = EditGraph.from_source(working, duration=metadata.get("video_duration_sec") or metadata["duration_sec"],
                                      fps=metadata["fps"], provenance="rendered_anchor")
    output_frames = round(graph.duration * fps)
    # Use the exact frame count of the current delivery for a matching clock.
    if graph.provenance == "rendered_anchor" and metadata["fps"] == fps and metadata.get("video_frame_count"):
        output_frames = metadata["video_frame_count"]
    if output_frames <= 0:
        raise ValueError("The footage has no frames to edit.")
    scenes, sound, overlays = [], [], []
    start = 0
    for index, span in enumerate(graph.spans):
        end = output_frames if index == len(graph.spans) - 1 else min(output_frames, round(span.output_end * fps))
        if end <= start:
            continue
        original = resolve_existing_project_file(graph.sources[span.source_id].media_path, state)
        src = depot.stage(original)
        rate = float((span.source_end - span.source_start) / span.duration)
        trim_start = round(span.source_start * fps)
        media = {"src": src, "kind": "video", "fit": "contain", "trim_start_frame": trim_start, "playback_rate": rate}
        identity = "source_" + payload_digest([span.source_id, str(span.source_start), str(span.output_start)])[:24]
        scenes.append({"id": identity, "start_frame": start, "duration_frames": end - start,
                       "producer": "local_media", "media": media})
        if depot.probe(original)["has_audio"]:
            sound.append({"id": identity, "src": src, "start_frame": start, "duration_frames": end - start,
                          "trim_start_frame": trim_start, "playback_rate": rate})
        start = end
    from video_generation.remotion_revisions import IMAGE_TYPES

    for layer in graph.layers:
        original = resolve_existing_project_file(layer.asset_path, state)
        if layer.checksum_sha256 and file_digest(original) != layer.checksum_sha256:
            raise ValueError("Original visual overlay changed after it was approved.")
        start, end = round(layer.output_start * fps), min(output_frames, round(layer.output_end * fps))
        if end <= start:
            continue
        # Assign a lane that stays above earlier overlapping visual layers.
        track = max((row["track"] for row in overlays
                     if row["start_frame"] < end and row["start_frame"] + row["duration_frames"] > start), default=0) + 1
        is_image = original.suffix.lower() in IMAGE_TYPES
        asset_metadata = {} if is_image else depot.probe(original)
        overlays.append({"id": layer.layer_id, "track": track, "start_frame": start, "duration_frames": end - start,
                         "mode": layer.compose_mode, "position": layer.position, "scale": layer.scale, "margin": layer.margin,
                         "fade_in_frames": round(layer.transition_in_sec * fps),
                         "fade_out_frames": round(layer.transition_out_sec * fps),
                         "media": {"src": depot.stage(original), "kind": "image" if is_image else "video", "fit": "cover",
                                   "transparent": bool(asset_metadata.get("has_alpha")),
                                   "trim_start_frame": 0 if is_image else round(layer.asset_start * fps)}})
    return {"version": TIMELINE_VERSION, "width": width, "height": height, "fps": fps, "duration_frames": output_frames,
            "source_provenance": graph.provenance, "captions": [], "scenes": scenes,
            "overlays": sorted(overlays, key=lambda row: (row["track"], row["start_frame"])),
            "audio": {"narration": "", "music": "", "music_volume": 0,
                      "tracks": [{"id": "source_audio", "name": "Source audio", "clips": sound}] if sound else []}}


def deliver_editorial(job: Path, timeline: dict, *, existing: Path | None = None) -> Path:
    from video_generation.remotion_delivery import check_remotion_delivery
    from video_generation.remotion_project import render_remotion_project, stage_remotion_runtime
    from video_generation.remotion_revisions import _project_asset, _validate_shot_media, _validate_timeline, _write

    _validate_timeline({"timeline": timeline})
    for item in (*timeline["scenes"], *timeline.get("overlays", [])):
        if media := item.get("media"):
            asset = _project_asset(job / "public", media["src"])
            _validate_shot_media(media, asset, timeline["fps"], item["duration_frames"], item.get("source_offset_frames", 0))
    stage_remotion_runtime(job)
    props = job / "input_props.json"
    _write(props, {"timeline": timeline})
    _write(job / "render_request.json", {"render_mode": "video", "video_format": "mp4", "video_quality": "standard"})
    if existing is not None:
        output = job / "final.mp4"
        shutil.copyfile(existing, output)
        metadata = probe_video(str(output))
    else:
        output, metadata, _ = render_remotion_project(job, output_format="mp4")
    receipt = check_remotion_delivery(props, output, metadata, reusable=existing is None)
    if not receipt["passed"]:
        raise ValueError(f"Editorial delivery failed verification: {', '.join(receipt['issues'])}")
    _write(job / "revision.json", {"version": "vex-remotion-revision-v1", "status": "verified",
                                   "output": output.name, "receipt": receipt})
    return output
