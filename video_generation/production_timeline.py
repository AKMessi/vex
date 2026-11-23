"""Frame-addressed edits to a verified production composition."""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
from typing import Any

from vex_runtime.edit_graph import rational
from vex_visuals.evidence import payload_digest


def cue_id(cue: dict) -> str:
    return cue.get("id") or "cue_" + payload_digest(cue)[:20]


def audio_assets(timeline: dict) -> set[str]:
    audio = timeline["audio"]
    return {name for name in (audio.get("narration"), audio.get("music"),
                              *(clip["src"] for clip in audio.get("voice_clips", []))) if name}


def _voice_clips(timeline: dict) -> list[dict]:
    audio = timeline["audio"]
    if "voice_clips" in audio:
        return audio["voice_clips"]
    if audio.get("narration"):
        return [{"src": audio["narration"], "start_frame": 0, "trim_start_frame": 0,
                 "duration_frames": timeline["duration_frames"]}]
    return []


def _intervals(items: list[dict], start: int, end: int, target: int, *, voice: bool) -> list[dict]:
    mapped = []
    for item in items:
        left = max(start, item["start_frame"])
        right = min(end, item["start_frame"] + item["duration_frames"])
        if left >= right:
            continue
        result = {**item, "start_frame": target + left - start, "duration_frames": right - left}
        if voice:
            result["trim_start_frame"] += left - item["start_frame"]
        else:
            result["id"] = cue_id(item)
            if left != item["start_frame"] or right != item["start_frame"] + item["duration_frames"]:
                result["id"] += f"_{left}_{right}"
        mapped.append(result)
    return mapped


def _reassemble(timeline: dict, shots: list[tuple[dict, int, int]]) -> None:
    scenes, captions, voice = [], [], []
    cursor = 0
    before_voice = _voice_clips(timeline)
    for source, start, end in shots:
        scene = {**source, "start_frame": cursor, "duration_frames": end - start}
        offset = start - source["start_frame"]
        if offset or end - start != source["duration_frames"]:
            scene["source_offset_frames"] = source.get("source_offset_frames", 0) + offset
            scene["source_duration_frames"] = source.get("source_duration_frames", source["duration_frames"])
        scenes.append(scene)
        captions.extend(_intervals(timeline["captions"], start, end, cursor, voice=False))
        voice.extend(_intervals(before_voice, start, end, cursor, voice=True))
        cursor += end - start
    timeline.update(scenes=scenes, captions=captions, duration_frames=cursor)
    if timeline["audio"].get("narration") or before_voice:
        timeline["audio"].update(narration="", voice_clips=voice)


def _frame(value: object, fps: int) -> int:
    seconds = rational(value)
    # Round once to the composition grid, using a deterministic half-up rule.
    return int((seconds * fps + Fraction(1, 2)) // 1)


def apply_timeline_edits(timeline: dict, operations: list[dict[str, Any]]) -> dict:
    return compile_timeline_edits(timeline, operations)[0]


def compile_timeline_edits(timeline: dict, operations: list[dict[str, Any]]) -> tuple[dict, dict[str, dict]]:
    if not isinstance(operations, list) or len(operations) > 32:
        raise ValueError("A production patch supports up to 32 operations.")
    result = deepcopy(timeline)
    speech = {}
    for operation in operations:
        if not isinstance(operation, dict):
            raise ValueError("Each production operation must be an object.")
        kind = operation.get("op")
        scenes = result["scenes"]
        by_id = {shot["id"]: shot for shot in scenes}
        if kind in {"trim", "remove_range"}:
            start = _frame(operation.get("start"), result["fps"])
            end = result["duration_frames"] if operation.get("end") is None else _frame(operation["end"], result["fps"])
            if not 0 <= start < end <= result["duration_frames"]:
                raise ValueError("Trim must retain a positive range within the cut.")
            if kind == "trim" and start == 0 and end == result["duration_frames"]:
                continue
            if kind == "remove_range" and start == 0 and end == result["duration_frames"]:
                raise ValueError("The cut must retain at least one frame.")
            windows = [(start, end)] if kind == "trim" else [(0, start), (end, result["duration_frames"])]
            retained = []
            for shot in scenes:
                pieces = []
                for lower, upper in windows:
                    left = max(lower, shot["start_frame"])
                    right = min(upper, shot["start_frame"] + shot["duration_frames"])
                    if left < right:
                        pieces.append((left, right))
                for index, (left, right) in enumerate(pieces):
                    piece = shot if index == 0 else {**shot, "id": "shot_" + payload_digest([shot["id"], left])[:24]}
                    retained.append((piece, left, right))
            _reassemble(result, retained)
        elif kind == "reorder":
            order = operation.get("shots")
            if (not isinstance(order, list) or any(not isinstance(item, str) for item in order)
                    or len(order) != len(scenes) or set(order) != by_id.keys()):
                raise ValueError("Provide every shot ID once in the desired order.")
            if order != list(by_id):
                _reassemble(result, [
                    (shot, shot["start_frame"], shot["start_frame"] + shot["duration_frames"])
                    for shot in (by_id[item] for item in order)
                ])
        elif kind == "split":
            identity = operation.get("id")
            shot = by_id.get(identity) if isinstance(identity, str) else None
            frame = _frame(operation.get("at"), result["fps"])
            if shot is None or not shot["start_frame"] < frame < shot["start_frame"] + shot["duration_frames"]:
                raise ValueError("Split must be inside the selected shot.")
            retained = []
            for row in scenes:
                start = row["start_frame"]
                end = start + row["duration_frames"]
                if row is shot:
                    tail = {**row, "id": "shot_" + payload_digest([identity, frame])[:24]}
                    retained.extend(((row, start, frame), (tail, frame, end)))
                else:
                    retained.append((row, start, end))
            _reassemble(result, retained)
        elif kind == "caption":
            cue = next((row for row in result["captions"] if cue_id(row) == operation.get("id")), None)
            value = operation.get("text")
            if cue is None or not isinstance(value, str) or not 0 < len(value.strip()) <= 280:
                raise ValueError("Choose a caption and enter 1 to 280 characters.")
            if value.strip() != cue["text"]:
                cue.update(id=cue_id(cue), text=value.strip())
        elif kind == "canvas":
            width, height = operation.get("width"), operation.get("height")
            if any(type(edge) is not int or edge % 2 or not 320 <= edge <= 7680 for edge in (width, height)):
                raise ValueError("Canvas dimensions must be even integers between 320 and 7680.")
            result.update(width=width, height=height)
        elif kind == "music_gain":
            gain = operation.get("gain")
            if not result["audio"].get("music") or type(gain) not in {int, float} or not 0 <= gain <= 1:
                raise ValueError("Music gain must be between zero and one and requires a music track.")
            result["audio"]["music_volume"] = gain
        elif kind == "voice_gain":
            gain = operation.get("gain")
            if not _voice_clips(result) or type(gain) not in {int, float} or not 0 <= gain <= 1:
                raise ValueError("Voice gain must be between zero and one and requires a voice track.")
            result["audio"]["voice_gain"] = gain
        elif kind == "artwork":
            from video_generation.film_artwork import validate_artwork

            shot = by_id.get(operation.get("id")) if isinstance(operation.get("id"), str) else None
            if shot is None or shot.get("media") or shot.get("pending_producer"):
                raise ValueError("Choose a browser-authored shot to edit its artwork.")
            shot.update(program=None, artwork=validate_artwork(operation.get("artwork")))
        elif kind in {"manim", "hyperframes"}:
            from dataclasses import asdict

            from video_generation.film_direction import parse_film_direction

            shot = by_id.get(operation.get("id")) if isinstance(operation.get("id"), str) else None
            if shot is None:
                raise ValueError("Select a shot for native authoring.")
            palette = operation.get("palette", {"background": "#111821", "text": "#F5F2E9", "accent": "#5EEAD4"})
            if kind == "manim":
                geometry = operation.get("geometry")
                if not isinstance(geometry, dict):
                    raise ValueError("Supply an editable Manim geometry recipe.")
                direction = {key: geometry.get(key, []) for key in ("action", "intent", "marks", "labels")}
                direction["kind"] = "geometry"
            else:
                direction = {"artwork": operation.get("artwork"), "intent": "Show authored motion graphics"}
            model = parse_film_direction({"palette": palette, "shots": [
                {**direction, "renderer": kind, "narration": ""}],}, "", "", authored=True)
            recipe = {**asdict(model.shots[0]), "palette": palette}
            for field in ("artwork", "program", "editorial", "media", "source_offset_frames", "source_duration_frames"):
                shot.pop(field, None)
            shot.update(producer=kind, native_recipe=recipe)
            shot.pop("sandbox_source", None)
        elif kind == "native_code":
            from vex_runtime.native_sandbox import ENGINES, MAX_SOURCE, sandbox_image

            identity, engine, source = operation.get("id"), operation.get("engine"), operation.get("source")
            shot = by_id.get(identity) if isinstance(identity, str) else None
            if (shot is None or not isinstance(engine, str) or engine not in ENGINES or not isinstance(source, str)
                    or not 0 < len(source.encode("utf-8")) <= MAX_SOURCE):
                raise ValueError("Choose a shot, a renderer, and a bounded UTF-8 source.")
            sandbox_image()
            for field in ("artwork", "program", "editorial", "media", "native_recipe",
                          "source_offset_frames", "source_duration_frames"):
                shot.pop(field, None)
            shot.update(producer=engine, sandbox_source={"engine": engine, "source": source})
        elif kind == "narration":
            from video_generation.models import DEFAULT_VOICE, VOICE_RE

            shot = by_id.get(operation.get("id")) if isinstance(operation.get("id"), str) else None
            value = operation.get("text")
            voice = operation.get("voice") or result.get("voice_settings", {}).get("voice", DEFAULT_VOICE)
            if (shot is None or not isinstance(value, str) or not 0 < len(value.strip()) <= 280
                    or not isinstance(voice, str) or not VOICE_RE.fullmatch(voice)):
                raise ValueError("Choose a shot, enter up to 280 characters and specify a supported voice.")
            value = " ".join(value.split())
            start = shot["start_frame"]
            length = shot["duration_frames"]
            end = start + length
            words = value.split()
            if len(words) > max(1, int(length / result["fps"] * 4)):
                raise ValueError("The sentence is too long for the shot. Write a shorter line.")
            spec = {"voice": voice, "text": value, "speed": result.get("voice_settings", {}).get("speed", 1.0),
                    "frames": length, "fps": result["fps"]}
            name = f"speech_{payload_digest(spec)}.wav"
            speech[name] = spec
            result["audio"].update(voice_clips=_without_interval(_voice_clips(result), start, end,
                                                                 result["duration_frames"], voice=True),
                                   narration="")
            result["audio"]["voice_clips"].append({"src": name, "start_frame": start, "trim_start_frame": 0,
                                                  "duration_frames": length})
            result["audio"]["voice_clips"].sort(key=lambda item: item["start_frame"])
            result["captions"] = _without_interval(result["captions"], start, end, result["duration_frames"], voice=False)
            groups = [words[index:index + 6] for index in range(0, len(words), 6)]
            for index, group in enumerate(groups):
                left = start + length * index // len(groups)
                right = start + length * (index + 1) // len(groups)
                result["captions"].append({"start_frame": left, "duration_frames": right - left, "text": " ".join(group)})
            result["captions"].sort(key=lambda item: item["start_frame"])
            shot["standfirst"] = value
        else:
            raise ValueError(f"Unsupported production edit: {kind!r}.")
    return result, speech


def _without_interval(items: list[dict], start: int, end: int, duration: int, *, voice: bool) -> list[dict]:
    return _intervals(items, 0, start, 0, voice=voice) + _intervals(items, end, duration, end, voice=voice)
