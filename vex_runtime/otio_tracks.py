"""Translate editable Studio lanes to and from ordinary OTIO tracks."""
from __future__ import annotations

import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from urllib.parse import quote

import config
from engine import probe_video
from state import ProjectState
from vex_runtime.editorial_sources import EditorialMedia, deliver_editorial
from vex_runtime.otio_interop import _library, _local_media
from vex_runtime.studio_projects import create_project
from vex_visuals.evidence import file_digest
from video_generation.remotion_project import TIMELINE_VERSION
from video_generation.remotion_revisions import IMAGE_TYPES, _project_asset, _validate_timeline


VERSION = "vex-otio-tracks-v1"
OVERLAY_FIELDS = ("mode", "position", "scale", "margin", "fade_in_frames", "fade_out_frames",
                  "fade_offset_frames", "fade_span_frames")
GAIN_FIELDS = ("gain", "fade_in_frames", "fade_out_frames", "fade_offset_frames", "fade_span_frames")


def _native(value):
    if isinstance(value, Mapping):
        return {key: _native(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [_native(item) for item in value]
    return value


def _vex_metadata(owner):
    value = owner.metadata.get("vex", {})
    return value if isinstance(value, Mapping) else {}


def build_editorial_otio(title: str, timeline: dict, source: Path, workspace: Path) -> str:
    otio = _library()
    rate = timeline["fps"]
    cut = otio.schema.Timeline(name=title)
    cut.global_start_time = otio.opentime.RationalTime(0, rate)
    cut.metadata["vex"] = {"mode": VERSION, "width": timeline["width"], "height": timeline["height"], "fps": rate}
    workspace.mkdir(parents=True, exist_ok=True)
    resources = {}

    def reference(path):
        if path not in resources:
            digest = file_digest(path)
            target = workspace / f"{digest}{path.suffix.lower()}"
            if target.resolve() != target or (target.is_file() and file_digest(target) != digest):
                raise ValueError("Exported editorial media was changed. Choose a new handoff folder.")
            if not target.exists():
                shutil.copyfile(path, target)
            resources[path] = quote(target.relative_to(workspace.parent).as_posix())
        return resources[path]

    def time(frame):
        return otio.opentime.RationalTime(frame, rate)

    def window(start, frames):
        return otio.opentime.TimeRange(time(start), time(frames))

    def clip(path, start, frames, *, identity="", name="", playback=1, metadata=None):
        result = otio.schema.Clip(name=name or path.name, source_range=window(start, frames),
                                  media_reference=otio.schema.ExternalReference(target_url=reference(path)),
                                  metadata={"vex": {"id": identity, **(metadata or {})}})
        if playback != 1:
            result.effects.append(otio.schema.LinearTimeWarp(time_scalar=playback))
        return result

    def track(name, kind, rows, convert, metadata=None):
        lane = otio.schema.Track(name=name, kind=kind, metadata={"vex": metadata or {}})
        cursor = 0
        for row in rows:
            if row["start_frame"] > cursor:
                lane.append(otio.schema.Gap(duration=time(row["start_frame"] - cursor)))
            lane.append(convert(row))
            cursor = row["start_frame"] + row["duration_frames"]
        cut.tracks.append(lane)

    public = source / "public"
    needs_plate = any((not row.get("media") and not row.get("gap")) or row.get("transition") == "accent_wipe"
                      for row in timeline["scenes"])
    plate = None
    if needs_plate:
        workspace.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="vex-otio-picture-") as directory:
            job = Path(directory)
            (job / "public").mkdir()
            for name in {row["media"]["src"] for row in timeline["scenes"] if row.get("media")}:
                shutil.copyfile(_project_asset(public, name), job / "public" / name)
            clean = {**timeline, "captions": [], "overlays": [], "audio": {"narration": "", "music": "", "music_volume": 0}}
            output = deliver_editorial(job, clean)
            reference(output)
            plate = workspace / f"{file_digest(output)}.mp4"

    def picture(row):
        if row.get("gap"):
            return otio.schema.Gap(duration=time(row["duration_frames"]))
        media = row.get("media")
        if media and row.get("transition") != "accent_wipe":
            playback = media.get("playback_rate", 1)
            trim = media.get("trim_start_frame", 0) + round(row.get("source_offset_frames", 0) * playback)
            return clip(_project_asset(public, media["src"]), trim, row["duration_frames"],
                        identity=row["id"], playback=playback, metadata={"fit": media["fit"], "producer": row["producer"]})
        data = {key: row[key] for key in ("artwork", "editorial", "program", "standfirst", "source_offset_frames",
                                         "source_duration_frames", "transition", "transition_color") if key in row}
        return clip(plate, row["start_frame"], row["duration_frames"], identity=row["id"], name=row["id"],
                    metadata={"recipe": data, "original_start": row["start_frame"], "original_frames": row["duration_frames"],
                              "producer": row.get("producer", "remotion")})

    track("Picture", otio.schema.TrackKind.Video, timeline["scenes"], picture)
    for number in sorted({row["track"] for row in timeline.get("overlays", [])}):
        rows = [row for row in timeline["overlays"] if row["track"] == number]
        track(f"Overlay {number}", otio.schema.TrackKind.Video, rows,
              lambda row: clip(_project_asset(public, row["media"]["src"]), row["media"]["trim_start_frame"],
                               row["duration_frames"], identity=row["id"], playback=row["media"].get("playback_rate", 1),
                               metadata={"fit": row["media"]["fit"], **{key: row[key] for key in OVERLAY_FIELDS if key in row}}))

    def sound(row):
        return clip(_project_asset(public, row["src"]), row["trim_start_frame"], row["duration_frames"],
                    identity=row.get("id", ""), playback=row.get("playback_rate", 1),
                    metadata={key: row[key] for key in GAIN_FIELDS if key in row})

    audio = timeline["audio"]
    tracks = list(audio.get("tracks", []))
    voice = list(audio.get("voice_clips", []))
    if audio.get("narration"):
        voice = [{"src": audio["narration"], "start_frame": 0, "trim_start_frame": 0,
                  "duration_frames": timeline["duration_frames"]}]
    if voice:
        tracks.append({"id": "vex_narration", "name": "Narration", "gain": audio.get("voice_gain", 1), "clips": voice})
    if name := audio.get("music"):
        length = max(1, round(probe_video(str(_project_asset(public, name)))["duration_sec"] * rate))
        total = timeline["duration_frames"]
        if (total + length - 1) // length > 4000:
            raise ValueError("The looping music is too short to export as editable OTIO clips.")
        fade = min(max(1, round(rate * .6)), max(1, total // 2))
        tracks.append({"id": "vex_music", "name": "Music", "gain": audio.get("music_volume", 0), "clips": [
            {"src": name, "start_frame": start, "trim_start_frame": 0, "duration_frames": min(length, total - start),
             "fade_in_frames": fade, "fade_out_frames": fade, "fade_offset_frames": start, "fade_span_frames": total}
            for start in range(0, total, length)]})
    for lane in tracks:
        track(lane["name"], otio.schema.TrackKind.Audio, lane["clips"], sound,
              {"id": lane["id"], "gain": lane.get("gain", 1)})
    for cue in timeline["captions"]:
        cut.tracks.markers.append(otio.schema.Marker(name=cue["text"], marked_range=window(cue["start_frame"], cue["duration_frames"]),
                                                    metadata={"vex_caption": True, "id": cue.get("id", "")}))
    for marker in timeline.get("markers", []):
        cut.tracks.markers.append(otio.schema.Marker(name=marker["text"], marked_range=window(marker["start_frame"], 0)))
    return otio.core.serialize_json_to_string(cut) + "\n"


def import_editorial_otio(path: Path, root: Path, document, name: str | None, provider: str, model: str) -> ProjectState:
    otio = _library()
    settings = _vex_metadata(document)
    own = settings.get("mode") == VERSION
    tracks = list(document.tracks)
    if (not 1 <= len(tracks) <= 64 or any(not isinstance(track, otio.schema.Track)
            or track.kind not in (otio.schema.TrackKind.Video, otio.schema.TrackKind.Audio) for track in tracks)
            or any(item.source_range or item.effects or not item.enabled for item in (document.tracks, *tracks))
            or sum(len(lane) for lane in tracks) > 4000):
        raise ValueError("OTIO import needs flat, enabled picture and audio lanes within the track limit.")
    rate = document.global_start_time.rate if document.global_start_time else 30
    fps = round(rate)
    if not document.global_start_time:
        first = next(iter(document.find_clips()), None)
        if first is not None and first.source_range:
            fps = round(first.source_range.duration.rate)
    if not 1 <= fps <= 120:
        raise ValueError("OTIO import supports a Studio frame grid from 1 to 120 fps.")

    def frames(time):
        if not time.is_valid_time():
            raise ValueError("OTIO contains an invalid time range.")
        return round(time.to_seconds() * fps)

    project_root = Path(config.AGENT_PROJECTS_DIR).expanduser().resolve()
    project_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".import-otio-", dir=project_root) as directory:
        stage = Path(directory)
        depot = EditorialMedia(stage / "public")
        scenes, overlays, sound, dissolves = [], [], [], []
        end, video_index = 0, 0
        dimensions = None
        for lane in document.tracks:
            video = lane.kind == otio.schema.TrackKind.Video
            index = video_index
            video_index += int(video)
            cursor, rows = 0, []
            lane_metadata = _vex_metadata(lane) if own else {}
            for item in lane:
                if isinstance(item, otio.schema.Transition):
                    if not video or index != 0 or item.transition_type != otio.schema.TransitionTypes.SMPTE_Dissolve:
                        raise ValueError("Only picture dissolves on the primary video track are supported.")
                    dissolves.append((cursor, frames(item.in_offset), frames(item.out_offset)))
                    continue
                if not isinstance(item, (otio.schema.Clip, otio.schema.Gap)) or not item.enabled:
                    raise ValueError("This OTIO contains a nested, disabled or unsupported editorial item.")
                if isinstance(item, otio.schema.Gap) and item.effects:
                    raise ValueError("OTIO gaps cannot have clip effects.")
                duration = frames(item.duration())
                if duration <= 0:
                    raise ValueError("Every OTIO clip and gap requires at least one frame.")
                identity = f"lane_{len(sound)}_{index}_{cursor}"
                row = {"start_frame": cursor, "duration_frames": duration}
                cursor += duration
                if isinstance(item, otio.schema.Gap):
                    if video and index == 0:
                        scenes.append({**row, "id": identity, "gap": True, "producer": "local_media"})
                    continue
                if not isinstance(item.media_reference, otio.schema.ExternalReference) or not item.source_range:
                    raise ValueError("OTIO clips must reference a local file and declare a source range.")
                original = _local_media(item.media_reference.target_url, path.parent, root, extended=True)
                meta = _vex_metadata(item) if own else {}
                identity = str(meta.get("id") or identity)
                trim = item.source_range.start_time
                if item.media_reference.available_range:
                    trim -= item.media_reference.available_range.start_time
                trim_start = frames(trim)
                effects = list(item.effects)
                if len(effects) > 1 or (effects and type(effects[0]) is not otio.schema.LinearTimeWarp):
                    raise ValueError("OTIO clip effects must use a supported constant playback speed.")
                rate_multiplier = effects[0].time_scalar if effects else 1
                if trim_start < 0 or not .125 <= rate_multiplier <= 8:
                    raise ValueError("OTIO clip has an unsupported source position or speed.")
                picture = video and original.suffix.lower() in IMAGE_TYPES
                src = depot.stage(original, audio=not video)
                if video:
                    media = {"src": src, "kind": "image" if picture else "video", "trim_start_frame": 0 if picture else trim_start,
                             "fit": meta.get("fit", "contain"), "playback_rate": rate_multiplier}
                    if not picture:
                        info = depot.probe(original)
                        media["transparent"] = bool(info.get("has_alpha"))
                        dimensions = dimensions or (info["width"], info["height"])
                    if index == 0:
                        recipe = meta.get("recipe")
                        producer = meta.get("producer")
                        producer = producer if producer in ("remotion", "manim", "hyperframes", "local_media") else "local_media"
                        if (hasattr(recipe, "keys") and recipe.get("artwork") and trim_start == meta.get("original_start")
                                and duration == meta.get("original_frames") and rate_multiplier == 1
                                and recipe.get("transition", "cut") == "cut"):
                            # SVG artwork goes through the normal composition validator.
                            from video_generation.film_artwork import validate_artwork
                            artwork = validate_artwork(_native(recipe["artwork"]))
                            scenes.append({**row, "artwork": artwork, "id": identity, "producer": producer,
                                           **{key: recipe[key] for key in ("source_offset_frames", "source_duration_frames") if key in recipe}})
                        else:
                            scenes.append({**row, "media": media, "id": identity, "producer": producer})
                    else:
                        overlays.append({**row, "id": identity, "track": index, "media": media,
                                         **{key: meta[key] for key in OVERLAY_FIELDS if key in meta}})
                else:
                    rows.append({**row, "id": identity, "src": src, "trim_start_frame": trim_start,
                                 "playback_rate": rate_multiplier, **{key: meta[key] for key in GAIN_FIELDS if key in meta}})
            if not video:
                sound.append({"id": str(lane_metadata.get("id") or f"audio_{len(sound)}"),
                              "name": (lane.name or f"Audio {len(sound) + 1}")[:120],
                              "gain": lane_metadata.get("gain", 1), "clips": rows})
            end = max(end, cursor)
        if not scenes or end > fps * 3600:
            raise ValueError("OTIO import needs a picture track and supports a timeline up to one hour.")
        last = scenes[-1]["start_frame"] + scenes[-1]["duration_frames"]
        if last < end:
            scenes.append({"id": "end_gap", "gap": True, "start_frame": last, "duration_frames": end - last,
                           "producer": "local_media"})
        if dissolves:
            for overlay in overlays:
                overlay["track"] += 2
        for boundary, lead, tail in dissolves:
            before = next((shot for shot in scenes if shot["start_frame"] + shot["duration_frames"] == boundary), None)
            after = next((shot for shot in scenes if shot["start_frame"] == boundary), None)
            if (not before or not after or not before.get("media") or not after.get("media")
                    or lead < 0 or tail < 0 or not lead + tail or lead > before["duration_frames"] or tail > after["duration_frames"]):
                raise ValueError("A dissolve needs valid bounds and source handles on both neighboring clips.")
            left, right = dict(before["media"]), dict(after["media"])
            if left["kind"] == "video":
                left["trim_start_frame"] += round(before["duration_frames"] * left["playback_rate"])
            if right["kind"] == "video":
                right["trim_start_frame"] -= round(lead * right["playback_rate"])
            if right["trim_start_frame"] < 0:
                raise ValueError("The incoming dissolve clip needs more media before its in point.")
            # The incoming clip fades across the edit. The outgoing clip remains
            # underneath it until the fade has finished on the other side.
            if tail:
                overlays.append({"id": f"dissolve_out_{boundary}", "track": 1, "start_frame": boundary,
                                 "duration_frames": tail, "media": left})
            overlays.append({"id": f"dissolve_in_{boundary}", "track": 2, "start_frame": boundary - lead,
                             "duration_frames": lead + tail, "fade_in_frames": lead + tail, "media": right})
        overlays.sort(key=lambda row: (row["track"], row["start_frame"]))
        width, height = dimensions or (1920, 1080)
        if own:
            width, height = settings.get("width", width), settings.get("height", height)
        if type(width) is not int or type(height) is not int:
            raise ValueError("Invalid OTIO canvas dimensions.")
        width, height = (max(320, edge + edge % 2) for edge in (width, height))
        captions, markers = [], []
        for marker in document.tracks.markers:
            start, count = frames(marker.marked_range.start_time), frames(marker.marked_range.duration)
            if 0 <= start < end and marker.name:
                if own and marker.metadata.get("vex_caption") is True and count > 0:
                    captions.append({"start_frame": start, "duration_frames": min(count, end - start), "text": marker.name[:280]})
                else:
                    markers.append({"start_frame": start, "text": marker.name[:280]})
        timeline = {"version": TIMELINE_VERSION, "fps": fps, "width": width, "height": height, "duration_frames": end,
                    "scenes": scenes, "overlays": overlays, "captions": sorted(captions, key=lambda row: row["start_frame"]),
                    "markers": markers, "audio": {"narration": "", "music": "", "music_volume": 0, "tracks": sound}}
        _validate_timeline({"timeline": timeline})
        delivery = deliver_editorial(stage, timeline)

        def initialize(state):
            composition = Path(state.working_dir) / "imported_timeline"
            shutil.copytree(stage, composition)
            state.artifacts["generated_source"] = {"manifest_path": str(composition / "revision.json")}

        return create_project(str(delivery), name or document.name or path.stem, provider, model,
                              managed_source=True, initialize=initialize)
