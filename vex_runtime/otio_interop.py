"""Native OpenTimelineIO handoff for local editorial cuts."""
from __future__ import annotations

import shutil
import tempfile
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import url2pathname

from engine import probe_video
from state import ProjectState
from vex_runtime.edit_graph import ClipSpan, EditGraph, SourceNode, rational


MAX_CLIPS = 200
MAX_TIMELINE_BYTES = 8 * 1024 * 1024


def _library():
    try:
        import opentimelineio
    except ImportError as exc:
        raise ValueError("Install the Vex interchange extra with pip install 'vex-video[interop]'.") from exc
    return opentimelineio


def build_otio(state: ProjectState, payload: dict) -> str:
    otio = _library()
    graph = EditGraph.from_mapping(payload["edit_graph"]) if payload["edit_graph"] else None
    media = payload["media"]
    rate = float(graph.fps if graph else media["fps"])
    records = []
    mode = "rendered-cut"
    if state.artifacts.get("generated_source"):
        from vex_runtime.production_edits import open_timeline

        try:
            _, composition = open_timeline(state)
        except (OSError, ValueError):
            composition = None
        if composition:
            mode = "baked-shots"
            rate = composition["fps"]
            for index, shot in enumerate(composition["scenes"], 1):
                records.append((f"Shot {index}", Path(state.working_file), shot["start_frame"],
                                shot["duration_frames"], {"shot_id": shot["id"], "producer": shot.get("producer", "remotion")}))
    if not records and graph and not graph.layers and all(
        span.duration == span.source_end - span.source_start for span in graph.spans
    ):
        mode = graph.provenance
        for span in graph.spans:
            source = Path(graph.sources[span.source_id].media_path)
            records.append((source.name, source, float(span.source_start * graph.fps),
                            float(span.duration * graph.fps), {}))
    if not records:
        path = Path(state.working_file)
        records.append((path.name, path, 0, round(media["duration_sec"] * rate), {}))
    timeline = otio.schema.Timeline(name=state.project_name)
    timeline.global_start_time = otio.opentime.RationalTime(0, rate)
    timeline.metadata["vex"] = {"mode": mode, "fps": rate, "width": media["width"], "height": media["height"],
                                "linked_source_audio": True}
    track = otio.schema.Track(name="Picture", kind=otio.schema.TrackKind.Video)
    timeline.tracks.append(track)
    linked_audio = otio.schema.Track(name="Source audio", kind=otio.schema.TrackKind.Audio)
    audio_presence = {}
    for name, path, start, length, metadata in records:
        if length <= 0:
            raise ValueError("The project has no positive-duration picture to export.")
        path = path.resolve(strict=True)
        if path not in audio_presence:
            audio_presence[path] = probe_video(str(path))["has_audio"]
        window = otio.opentime.TimeRange(otio.opentime.RationalTime(start, rate), otio.opentime.RationalTime(length, rate))
        track.append(otio.schema.Clip(name=name, media_reference=otio.schema.ExternalReference(target_url=path.as_uri()),
                                      source_range=window, metadata=metadata))
        if audio_presence[path]:
            linked_audio.append(otio.schema.Clip(name=name, source_range=window,
                                media_reference=otio.schema.ExternalReference(target_url=path.as_uri())))
        else:
            linked_audio.append(otio.schema.Gap(duration=window.duration))
    if any(audio_presence.values()):
        timeline.tracks.append(linked_audio)
    return otio.core.serialize_json_to_string(timeline) + "\n"


def _local_media(reference: str, timeline_dir: Path, media_root: Path) -> Path:
    if reference.startswith(("//", "\\\\")):
        raise ValueError("OTIO import requires local media references.")
    path = Path(reference)
    if not path.is_absolute():
        parsed = urlsplit(reference)
        if parsed.scheme not in {"", "file"} or parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError("OTIO import requires local media references.")
        path = timeline_dir / url2pathname(parsed.path)
    if not path.resolve().is_relative_to(media_root):
        raise ValueError("Referenced media is outside the selected media directory.")
    path = path.resolve(strict=True)
    if not path.is_file() or path.suffix.lower() not in {".mov", ".mp4", ".webm", ".mkv"}:
        raise ValueError("OTIO import requires MOV, MP4, WebM or MKV picture media.")
    return path


def _read_tracks(path: Path, media_root: Path):
    otio = _library()
    if path.suffix.lower() != ".otio" or not 0 < path.stat().st_size <= MAX_TIMELINE_BYTES:
        raise ValueError("Choose an .otio file up to 8 MiB.")
    try:
        timeline = otio.core.deserialize_json_from_string(path.read_text(encoding="utf-8-sig"))
    except (ValueError, RuntimeError) as exc:
        raise ValueError("The file does not contain a valid OpenTimelineIO timeline.") from exc
    if not isinstance(timeline, otio.schema.Timeline):
        raise ValueError("Choose an OTIO Timeline document.")
    tracks = list(timeline.tracks)
    if len(tracks) not in {1, 2} or any(not isinstance(track, otio.schema.Track) for track in tracks):
        raise ValueError("OTIO import supports one picture track and optional linked source audio.")
    video_tracks = [track for track in tracks if track.kind == otio.schema.TrackKind.Video]
    audio_tracks = [track for track in tracks if track.kind == otio.schema.TrackKind.Audio]
    if len(video_tracks) != 1 or len(audio_tracks) != len(tracks) - 1:
        raise ValueError("OTIO import supports one picture track and optional linked source audio.")
    for item in [timeline.tracks, *tracks]:
        if item.source_range or item.effects or not item.enabled:
            raise ValueError("Track-level trims, effects, and disabled tracks are unsupported.")
    video = video_tracks[0]
    linked_audio = audio_tracks[0] if audio_tracks else None
    if not 1 <= len(video) <= MAX_CLIPS:
        raise ValueError(f"The OTIO picture track must contain 1 to {MAX_CLIPS} clips.")
    if linked_audio is not None and len(linked_audio) != len(video):
        raise ValueError("The OTIO audio track must match the picture edits.")

    def read_clip(clip):
        if (not isinstance(clip, otio.schema.Clip) or not clip.enabled or clip.effects
                or not isinstance(clip.media_reference, otio.schema.ExternalReference)):
            raise ValueError("OTIO import supports straight cuts with local media and no clip effects.")
        window = clip.source_range
        if window is None or not window.duration.is_valid_time() or not window.start_time.is_valid_time():
            raise ValueError("Each OTIO clip must specify a valid source range.")
        reference = _local_media(clip.media_reference.target_url, path.parent, media_root)
        available = clip.media_reference.available_range
        if available is not None:
            window = otio.opentime.TimeRange(window.start_time - available.start_time, window.duration)
        return reference, window

    records = [read_clip(clip) for clip in video]
    if linked_audio is not None:
        for audio, (reference, window) in zip(linked_audio, records, strict=True):
            if isinstance(audio, otio.schema.Gap) and audio.enabled and not audio.effects and audio.duration() == window.duration:
                continue
            if read_clip(audio) != (reference, window):
                raise ValueError("The OTIO audio track must use the same cuts and media as the picture.")
    origin = timeline.global_start_time
    rate = Fraction(origin.rate if origin else records[0][1].duration.rate).limit_denominator(100_000)
    if not 1 <= rate <= 120:
        raise ValueError("OTIO import supports frame rates from 1 to 120 fps.")
    return timeline.name, rate, records, linked_audio


def import_otio(path: Path, media_root: Path | None, name: str | None, provider: str, model: str) -> ProjectState:
    from vex_runtime.edit_graph_renderer import render_edit_graph
    from vex_runtime.studio_projects import create_project

    path = path.expanduser().resolve(strict=True)
    root = media_root.expanduser().resolve(strict=True) if media_root else path.parent
    title, rate, records, linked_audio = _read_tracks(path, root)
    otio = _library()
    sources, spans, source_info = {}, [], {}
    cursor = Fraction(0)
    with tempfile.TemporaryDirectory(prefix="vex-otio-") as directory:
        staging = Path(directory)
        for index, (source, window) in enumerate(records):
            if source not in source_info:
                with source.open("rb") as stream:
                    header = stream.read(12)
                if not (header.startswith(b"\x1a\x45\xdf\xa3") or header[4:8] in {b"ftyp", b"moov", b"mdat", b"wide", b"free"}):
                    raise ValueError("The referenced file is not a supported local video container.")
                identity = f"media_{len(sources) + 1:03d}"
                staged = staging / f"{identity}{source.suffix.lower()}"
                shutil.copyfile(source, staged)
                info = probe_video(str(staged))
                if not info["width"] or not info["height"] or not info["fps"]:
                    raise ValueError("Every OTIO picture clip must contain a video stream.")
                duration = rational(info.get("video_duration_sec") or info["duration_rational"])
                if info.get("video_frame_count") and info.get("fps_ratio"):
                    duration = info["video_frame_count"] / rational(info["fps_ratio"])
                sources[identity] = SourceNode(identity, str(staged), duration)
                source_info[source] = (identity, info)
            identity, info = source_info[source]
            if linked_audio is not None and isinstance(linked_audio[index], otio.schema.Gap) and info["has_audio"]:
                raise ValueError("Audio gaps over clips with embedded audio are unsupported.")
            start = Fraction(window.start_time.to_seconds()).limit_denominator(10**6)
            duration = Fraction(window.duration.to_seconds()).limit_denominator(10**6)
            if (start < 0 or duration <= 0 or abs(start * rate - round(start * rate)) > Fraction(1, 1000)
                    or abs(duration * rate - round(duration * rate)) > Fraction(1, 1000)):
                raise ValueError("OTIO source windows must align to the timeline frame grid.")
            start = round(start * rate) / rate
            duration = round(duration * rate) / rate
            spans.append(ClipSpan(identity, start, start + duration, cursor, cursor + duration))
            cursor += duration
        first = source_info[records[0][0]][1]
        width, height = (first[key] + first[key] % 2 for key in ("width", "height"))
        if max(width, height) > 7680 or cursor > 3600:
            raise ValueError("OTIO import supports a canvas up to 7680 pixels and a cut up to one hour.")
        graph = EditGraph(sources, tuple(spans), rate)
        delivery = render_edit_graph(graph, staging, width=width, height=height)
        result = probe_video(delivery)
        if result["video_frame_count"] != round(graph.duration * rate):
            raise ValueError("The imported cut did not preserve the expected frame count.")

        def initialize(state: ProjectState) -> None:
            target = Path(state.working_dir) / "media"
            target.mkdir()
            managed = {}
            for identity, source in sources.items():
                asset = target / Path(source.media_path).name
                shutil.copy2(source.media_path, asset)
                managed[identity] = replace(source, media_path=str(asset))
            state.source_files = [source.media_path for source in managed.values()]
            state.edit_graph = replace(graph, sources=managed).to_dict()

        return create_project(delivery, name or title or path.stem, provider, model, managed_source=True,
                              initialize=initialize)
