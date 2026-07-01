from __future__ import annotations

import logging
import math
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Callable

import ffmpeg

import config
from color_grading import (
    ColorGradePlanningError,
    build_shot_aware_color_grade_plan,
    validate_color_grade_output,
    validate_color_grade_output_by_shots,
)
from subtitles import compile_subtitles_to_ass

LOGGER = logging.getLogger(__name__)


class VideoEngineError(Exception):
    def __init__(self, message: str, command: str = "") -> None:
        super().__init__(message)
        self.command = command


def _ffprobe_binary() -> str:
    ffmpeg_path = Path(config.FFMPEG_PATH)
    if ffmpeg_path.name.lower().startswith("ffmpeg"):
        candidate = ffmpeg_path.with_name(ffmpeg_path.name.replace("ffmpeg", "ffprobe", 1))
        if shutil.which(str(candidate)):
            return str(candidate)
    return "ffprobe"


def _unique_path(working_dir: str, suffix: str) -> str:
    Path(working_dir).mkdir(parents=True, exist_ok=True)
    return str(Path(working_dir) / f"{uuid.uuid4().hex}{suffix}")


def _run_command(command: list[str], message: str) -> None:
    command_text = " ".join(command)
    LOGGER.debug("Running ffmpeg command: %s", command_text)
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise VideoEngineError(
            f"{message}: {result.stderr.strip() or result.stdout.strip()}",
            command=command_text,
        )


def _run_ffmpeg(stream, message: str) -> None:
    command = ffmpeg.compile(stream, cmd=config.FFMPEG_PATH, overwrite_output=True)
    LOGGER.debug("Running ffmpeg command: %s", " ".join(command))
    try:
        ffmpeg.run(
            stream,
            cmd=config.FFMPEG_PATH,
            overwrite_output=True,
            capture_stdout=True,
            capture_stderr=True,
        )
    except ffmpeg.Error as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else ""
        raise VideoEngineError(f"{message}: {stderr.strip()}", command=" ".join(command)) from exc


def parse_timestamp(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        raise ValueError("Invalid timestamp: None")
    raw = str(value).strip()
    if not raw:
        raise ValueError("Invalid timestamp: ''")
    if raw.endswith("s"):
        raw = raw[:-1]
    if raw.count(":") == 0:
        try:
            return float(raw)
        except ValueError as exc:
            raise ValueError(f"Invalid timestamp: {value!r}") from exc
    parts = raw.split(":")
    if len(parts) not in {2, 3}:
        raise ValueError(f"Invalid timestamp: {value!r}")
    try:
        numbers = [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"Invalid timestamp: {value!r}") from exc
    if len(numbers) == 2:
        minutes, seconds = numbers
        return minutes * 60 + seconds
    hours, minutes, seconds = numbers
    return hours * 3600 + minutes * 60 + seconds


def _fps_to_float(rate: str) -> float:
    if not rate or rate == "0/0":
        return 0.0
    if "/" in rate:
        numerator, denominator = rate.split("/", 1)
        return round(float(numerator) / float(denominator), 3) if float(denominator) else 0.0
    return float(rate)


def _silent_audio(duration: float, working_dir: str) -> str:
    temp_path = _unique_path(working_dir, ".m4a")
    command = [
        config.FFMPEG_PATH,
        "-f",
        "lavfi",
        "-i",
        "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-t",
        str(duration),
        "-c:a",
        "aac",
        "-y",
        temp_path,
    ]
    _run_command(command, "Failed to generate silent audio")
    return temp_path


def probe_video(path: str) -> dict:
    info = ffmpeg.probe(path, cmd=_ffprobe_binary())
    format_info = info.get("format", {})
    streams = info.get("streams", [])
    video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), {})
    audio_stream = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    video_bitrate = video_stream.get("bit_rate") or format_info.get("bit_rate")
    audio_bitrate = audio_stream.get("bit_rate") if audio_stream else None
    return {
        "duration_sec": float(format_info.get("duration") or video_stream.get("duration") or 0.0),
        "fps": _fps_to_float(video_stream.get("avg_frame_rate", "0/0")),
        "width": int(video_stream.get("width") or 0),
        "height": int(video_stream.get("height") or 0),
        "codec": video_stream.get("codec_name", "unknown"),
        "profile": video_stream.get("profile"),
        "pix_fmt": video_stream.get("pix_fmt"),
        "video_bit_rate": int(video_bitrate or 0),
        "has_audio": audio_stream is not None,
        "audio_codec": audio_stream.get("codec_name", "unknown") if audio_stream else None,
        "audio_bit_rate": int(audio_bitrate or 0) if audio_bitrate else 0,
        "audio_channels": int(audio_stream.get("channels") or 0) if audio_stream else 0,
        "size_bytes": int(format_info.get("size") or os.path.getsize(path)),
        "format": format_info.get("format_name", "unknown"),
        "format_long_name": format_info.get("format_long_name", "unknown"),
    }


def _video_has_audio(path: str) -> bool:
    return bool(probe_video(path).get("has_audio"))


def _ffconcat_file_line(path: str) -> str:
    normalized = Path(path).resolve().as_posix()
    escaped = normalized.replace("'", r"'\''")
    return f"file '{escaped}'"


def trim(input_path: str, working_dir: str, start_sec: float, end_sec: float | None) -> str:
    output_path = _unique_path(working_dir, ".mp4")
    stream = ffmpeg.input(input_path, ss=max(start_sec, 0.0))
    output_kwargs = {"vcodec": "libx264", "acodec": "aac", "movflags": "+faststart"}
    if end_sec is not None:
        output_kwargs["t"] = max(end_sec - start_sec, 0.0)
    stream = ffmpeg.output(stream, output_path, **output_kwargs)
    _run_ffmpeg(stream, "Failed to trim video")
    return output_path


def _normalize_for_concat(input_path: str, working_dir: str, resolution: tuple[int, int], fps: float) -> str:
    width, height = resolution
    output_path = _unique_path(working_dir, ".mp4")
    metadata = probe_video(input_path)
    input_stream = ffmpeg.input(input_path)
    video = input_stream.video.filter("scale", width, height, force_original_aspect_ratio="decrease")
    video = video.filter("pad", width, height, "(ow-iw)/2", "(oh-ih)/2", color="black")
    video = video.filter("fps", fps=math.ceil(fps) if fps else 30)
    if metadata["has_audio"]:
        audio = input_stream.audio.filter("aresample", 44100)
    else:
        audio = ffmpeg.input(_silent_audio(metadata["duration_sec"], working_dir)).audio
    stream = ffmpeg.output(
        video,
        audio,
        output_path,
        vcodec="libx264",
        acodec="aac",
        pix_fmt="yuv420p",
        movflags="+faststart",
    )
    _run_ffmpeg(stream, "Failed to normalize clip for concat")
    return output_path


def merge(input_paths: list[str], working_dir: str) -> str:
    if not input_paths:
        raise VideoEngineError("At least one input path is required for merge.")
    metadata = [probe_video(path) for path in input_paths]
    target_resolution = (metadata[0]["width"], metadata[0]["height"])
    target_fps = metadata[0]["fps"] or 30.0
    normalized_files = [
        _normalize_for_concat(path, working_dir, target_resolution, target_fps) for path in input_paths
    ]
    concat_list = Path(_unique_path(working_dir, ".txt"))
    concat_list.write_text(
        "\n".join(_ffconcat_file_line(path) for path in normalized_files),
        encoding="utf-8",
    )
    output_path = _unique_path(working_dir, ".mp4")
    command = [
        config.FFMPEG_PATH,
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list),
        "-c",
        "copy",
        "-y",
        output_path,
    ]
    _run_command(command, "Failed to merge clips")
    return output_path


def extract_segments(
    input_path: str,
    working_dir: str,
    segments: list[tuple[float, float]],
) -> str:
    if not segments:
        raise VideoEngineError("At least one segment is required to extract highlights.")
    normalized_segments = sorted(segments, key=lambda item: item[0])
    trimmed_paths = [
        trim(input_path, working_dir, start_sec=max(start_sec, 0.0), end_sec=max(end_sec, 0.0))
        for start_sec, end_sec in normalized_segments
        if end_sec > start_sec
    ]
    if not trimmed_paths:
        raise VideoEngineError("No valid highlight segments were selected.")
    if len(trimmed_paths) == 1:
        return trimmed_paths[0]
    return merge(trimmed_paths, working_dir)


def _speed_audio_filter(factor: float) -> str:
    filters: list[str] = []
    remaining = factor
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    filters.append(f"atempo={remaining:.5f}")
    return ",".join(filters)


def adjust_speed(
    input_path: str,
    working_dir: str,
    factor: float,
    segment_start: float | None,
    segment_end: float | None,
) -> str:
    output_path = _unique_path(working_dir, ".mp4")
    has_audio = _video_has_audio(input_path)
    if segment_start is None and segment_end is None:
        if has_audio:
            filter_complex = f"[0:v]setpts={1/factor:.8f}*PTS[v];[0:a]{_speed_audio_filter(factor)}[a]"
        else:
            filter_complex = f"[0:v]setpts={1/factor:.8f}*PTS[v]"
        command = [
            config.FFMPEG_PATH,
            "-i",
            input_path,
            "-filter_complex",
            filter_complex,
            "-map",
            "[v]",
        ]
        if has_audio:
            command.extend(["-map", "[a]", "-c:a", "aac"])
        else:
            command.append("-an")
        command.extend(["-c:v", "libx264", "-y", output_path])
        _run_command(command, "Failed to adjust video speed")
        return output_path

    start = segment_start or 0.0
    end = segment_end if segment_end is not None else probe_video(input_path)["duration_sec"]
    if has_audio:
        filter_complex = (
            f"[0:v]split=3[v1][v2][v3];"
            f"[0:a]asplit=3[a1][a2][a3];"
            f"[v1]trim=0:{start},setpts=PTS-STARTPTS[v1o];"
            f"[a1]atrim=0:{start},asetpts=PTS-STARTPTS[a1o];"
            f"[v2]trim={start}:{end},setpts={1/factor:.8f}*(PTS-STARTPTS)[v2o];"
            f"[a2]atrim={start}:{end},asetpts=PTS-STARTPTS,{_speed_audio_filter(factor)}[a2o];"
            f"[v3]trim={end},setpts=PTS-STARTPTS[v3o];"
            f"[a3]atrim={end},asetpts=PTS-STARTPTS[a3o];"
            f"[v1o][a1o][v2o][a2o][v3o][a3o]concat=n=3:v=1:a=1[v][a]"
        )
    else:
        filter_complex = (
            f"[0:v]split=3[v1][v2][v3];"
            f"[v1]trim=0:{start},setpts=PTS-STARTPTS[v1o];"
            f"[v2]trim={start}:{end},setpts={1/factor:.8f}*(PTS-STARTPTS)[v2o];"
            f"[v3]trim={end},setpts=PTS-STARTPTS[v3o];"
            f"[v1o][v2o][v3o]concat=n=3:v=1:a=0[v]"
        )
    command = [
        config.FFMPEG_PATH,
        "-i",
        input_path,
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-c:v",
        "libx264",
    ]
    if has_audio:
        command.extend(["-map", "[a]", "-c:a", "aac"])
    else:
        command.append("-an")
    command.extend(["-y", output_path])
    _run_command(command, "Failed to adjust segment speed")
    return output_path


def fade_in(input_path: str, working_dir: str, duration: float) -> str:
    output_path = _unique_path(working_dir, ".mp4")
    has_audio = _video_has_audio(input_path)
    command = [
        config.FFMPEG_PATH,
        "-i",
        input_path,
        "-vf",
        f"fade=t=in:st=0:d={duration}",
    ]
    if has_audio:
        command.extend(["-af", f"afade=t=in:st=0:d={duration}"])
    command.extend(["-c:v", "libx264"])
    if has_audio:
        command.extend(["-c:a", "aac"])
    else:
        command.append("-an")
    command.extend(["-y", output_path])
    _run_command(command, "Failed to add fade in")
    return output_path


def fade_out(input_path: str, working_dir: str, duration: float) -> str:
    output_path = _unique_path(working_dir, ".mp4")
    clip_info = probe_video(input_path)
    has_audio = bool(clip_info.get("has_audio"))
    start = max(clip_info["duration_sec"] - duration, 0.0)
    command = [
        config.FFMPEG_PATH,
        "-i",
        input_path,
        "-vf",
        f"fade=t=out:st={start}:d={duration}",
    ]
    if has_audio:
        command.extend(["-af", f"afade=t=out:st={start}:d={duration}"])
    command.extend(["-c:v", "libx264"])
    if has_audio:
        command.extend(["-c:a", "aac"])
    else:
        command.append("-an")
    command.extend(["-y", output_path])
    _run_command(command, "Failed to add fade out")
    return output_path


def crossfade(input1: str, input2: str, working_dir: str, duration: float) -> str:
    output_path = _unique_path(working_dir, ".mp4")
    clip_info = probe_video(input1)
    offset = max(clip_info["duration_sec"] - duration, 0.0)
    command = [
        config.FFMPEG_PATH,
        "-i",
        input1,
        "-i",
        input2,
        "-filter_complex",
        (
            f"[0:v][1:v]xfade=transition=fade:duration={duration}:offset={offset}[v];"
            f"[0:a][1:a]acrossfade=d={duration}[a]"
        ),
        "-map",
        "[v]",
        "-map",
        "[a]",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-y",
        output_path,
    ]
    _run_command(command, "Failed to crossfade clips")
    return output_path


def add_text(
    input_path: str,
    working_dir: str,
    text: str,
    position: str,
    font_size: int,
    color: str,
    start_sec: float,
    end_sec: float,
    bg_opacity: float,
) -> str:
    from moviepy.editor import ColorClip, CompositeVideoClip, TextClip, VideoFileClip

    output_path = _unique_path(working_dir, ".mp4")
    base = VideoFileClip(input_path)
    duration = max(end_sec - start_sec, 0.0)
    text_clip = TextClip(text, fontsize=font_size, color=color, method="caption", size=(int(base.w * 0.8), None))
    text_clip = text_clip.set_start(start_sec).set_duration(duration)
    pos_map = {
        "top": ("center", "top"),
        "center": ("center", "center"),
        "bottom": ("center", "bottom"),
        "top_left": ("left", "top"),
        "top_right": ("right", "top"),
        "bottom_left": ("left", "bottom"),
        "bottom_right": ("right", "bottom"),
    }
    text_clip = text_clip.set_position(pos_map[position])
    layers = [base]
    if bg_opacity > 0:
        background = (
            ColorClip(size=(text_clip.w + 40, text_clip.h + 20), color=(0, 0, 0))
            .set_opacity(bg_opacity)
            .set_start(start_sec)
            .set_duration(duration)
            .set_position(pos_map[position])
        )
        layers.append(background)
    layers.append(text_clip)
    final = CompositeVideoClip(layers)
    final.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile=str(Path(working_dir) / f"{uuid.uuid4().hex}_temp-audio.m4a"),
        remove_temp=True,
        logger=None,
    )
    base.close()
    final.close()
    return output_path


def extract_audio(input_path: str, working_dir: str, fmt: str) -> str:
    if not _video_has_audio(input_path):
        raise VideoEngineError("The current video has no audio stream to extract.")
    suffix = ".m4a" if fmt == "aac" else f".{fmt}"
    output_path = _unique_path(working_dir, suffix)
    codec = {"mp3": "libmp3lame", "wav": "pcm_s16le", "aac": "aac"}[fmt]
    stream = ffmpeg.output(ffmpeg.input(input_path).audio, output_path, acodec=codec)
    _run_ffmpeg(stream, "Failed to extract audio")
    return output_path


def replace_audio(video_path: str, audio_path: str, working_dir: str, mix: bool, mix_ratio: float) -> str:
    output_path = _unique_path(working_dir, ".mp4")
    has_original_audio = _video_has_audio(video_path)
    if mix and has_original_audio:
        filter_complex = (
            f"[0:a]volume={1 - mix_ratio:.3f}[orig];"
            f"[1:a]volume={mix_ratio:.3f}[new];"
            f"[orig][new]amix=inputs=2:duration=first:dropout_transition=2[a]"
        )
        command = [
            config.FFMPEG_PATH,
            "-i",
            video_path,
            "-i",
            audio_path,
            "-filter_complex",
            filter_complex,
            "-map",
            "0:v",
            "-map",
            "[a]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            "-y",
            output_path,
        ]
    else:
        command = [
            config.FFMPEG_PATH,
            "-i",
            video_path,
            "-i",
            audio_path,
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            "-y",
            output_path,
        ]
    _run_command(command, "Failed to replace audio")
    return output_path


def add_song_to_video(
    video_path: str,
    song_path: str,
    working_dir: str,
    song_mix_plan: dict[str, Any],
    *,
    filtergraph_path: str | None = None,
) -> str:
    output_path = _unique_path(working_dir, ".mp4")
    source_metadata = probe_video(video_path)
    duration = max(float(song_mix_plan.get("video_duration_sec") or source_metadata.get("duration_sec") or 0.0), 0.0)
    if duration <= 0.0:
        raise VideoEngineError("Cannot add a song because the source video duration is invalid.")
    has_original_audio = bool(source_metadata.get("has_audio")) and bool(
        song_mix_plan.get("preserve_original_audio")
    )
    filter_graph = _build_song_mix_filter_graph(song_mix_plan, has_original_audio=has_original_audio)
    if filtergraph_path:
        Path(filtergraph_path).write_text(filter_graph, encoding="utf-8")

    command = [
        config.FFMPEG_PATH,
        "-hide_banner",
        "-nostdin",
        "-i",
        video_path,
    ]
    if bool(song_mix_plan.get("loop_song")):
        command.extend(["-stream_loop", "-1"])
    command.extend(
        [
            "-i",
            song_path,
            "-filter_complex",
            filter_graph,
            "-map",
            "0:v:0",
            "-map",
            "[a]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            "-t",
            f"{duration:.3f}",
            "-y",
            output_path,
        ]
    )
    _run_command(command, "Failed to add song to video")
    return output_path


def _build_song_mix_filter_graph(song_mix_plan: dict[str, Any], *, has_original_audio: bool) -> str:
    duration = max(float(song_mix_plan.get("video_duration_sec") or 0.0), 0.001)
    placements = [
        dict(item)
        for item in song_mix_plan.get("placements") or []
        if isinstance(item, dict)
    ]
    if not placements:
        raise VideoEngineError("Song mix plan has no placements.")
    normalize_loudness = _metadata_bool(song_mix_plan.get("normalize_loudness"), True)
    music_volume = max(0.0, min(float(song_mix_plan.get("music_volume") or 0.0), 1.5))
    song_lufs = float(song_mix_plan.get("song_lufs") or -17.0)
    output_lufs = float(song_mix_plan.get("output_lufs") or -16.0)

    filter_parts: list[str] = []
    song_base_filters = [
        "aresample=48000",
        "aformat=sample_fmts=fltp:channel_layouts=stereo",
    ]
    if normalize_loudness:
        song_base_filters.append(f"loudnorm=I={song_lufs:.1f}:TP=-2.0:LRA=11")
    filter_parts.append(f"[1:a]{','.join(song_base_filters)}[songbase]")
    if len(placements) > 1:
        split_outputs = "".join(f"[songsrc{index}]" for index in range(len(placements)))
        filter_parts.append(f"[songbase]asplit={len(placements)}{split_outputs}")
    else:
        filter_parts.append("[songbase]anull[songsrc0]")

    music_labels: list[str] = []
    for index, placement in enumerate(placements):
        start = max(0.0, min(float(placement.get("start") or 0.0), duration))
        end = max(start + 0.001, min(float(placement.get("end") or start), duration))
        placement_duration = max(end - start, 0.001)
        fade_in = max(0.0, min(float(placement.get("fade_in") or 0.0), placement_duration * 0.45))
        fade_out = max(0.0, min(float(placement.get("fade_out") or 0.0), placement_duration * 0.45))
        song_start = max(0.0, float(placement.get("song_start") or 0.0))
        delay_ms = max(0, int(round(start * 1000.0)))
        placement_filters = [
            f"atrim=start={song_start:.3f}:duration={placement_duration:.3f}",
            "asetpts=PTS-STARTPTS",
        ]
        if fade_in > 0.001:
            placement_filters.append(f"afade=t=in:st=0:d={fade_in:.3f}")
        if fade_out > 0.001:
            fade_start = max(placement_duration - fade_out, 0.0)
            placement_filters.append(f"afade=t=out:st={fade_start:.3f}:d={fade_out:.3f}")
        placement_filters.extend(
            [
                f"volume={music_volume:.5f}",
                f"adelay={delay_ms}:all=1",
                f"apad=whole_dur={duration:.3f}",
                f"atrim=0:{duration:.3f}",
                "asetpts=PTS-STARTPTS",
            ]
        )
        label = f"music{index}"
        filter_parts.append(f"[songsrc{index}]{','.join(placement_filters)}[{label}]")
        music_labels.append(f"[{label}]")

    if len(music_labels) > 1:
        filter_parts.append(
            f"{''.join(music_labels)}amix=inputs={len(music_labels)}:"
            "duration=first:dropout_transition=0:normalize=0[musicbed]"
        )
    else:
        filter_parts.append(f"{music_labels[0]}anull[musicbed]")

    if has_original_audio:
        filter_parts.append("[0:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[orig]")
        if _metadata_bool(song_mix_plan.get("ducking_enabled"), False):
            filter_parts.append(
                "[musicbed][orig]sidechaincompress=threshold=0.035:"
                "ratio=8:attack=20:release=350:makeup=1[ducked]"
            )
            filter_parts.append(
                '[orig][ducked]amix=inputs=2:duration=first:dropout_transition=0:weights="1 1":normalize=0[mixed]'
            )
        else:
            filter_parts.append(
                '[orig][musicbed]amix=inputs=2:duration=first:dropout_transition=0:weights="1 1":normalize=0[mixed]'
            )
    else:
        filter_parts.append("[musicbed]anull[mixed]")

    final_filters = ["alimiter=limit=0.95"]
    if normalize_loudness:
        final_filters.append(f"loudnorm=I={output_lufs:.1f}:TP=-1.5:LRA=11")
    final_filters.append("aresample=48000")
    filter_parts.append(f"[mixed]{','.join(final_filters)}[a]")
    return ";".join(filter_parts)


def mute_segment(input_path: str, working_dir: str, start_sec: float, end_sec: float) -> str:
    if not _video_has_audio(input_path):
        return input_path
    output_path = _unique_path(working_dir, ".mp4")
    command = [
        config.FFMPEG_PATH,
        "-i",
        input_path,
        "-af",
        f"volume=enable='between(t,{start_sec},{end_sec})':volume=0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-y",
        output_path,
    ]
    _run_command(command, "Failed to mute audio segment")
    return output_path


def _merge_time_ranges(ranges: list[tuple[float, float]], gap_sec: float = 0.0) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start_sec, end_sec in sorted(ranges, key=lambda item: item[0]):
        if end_sec <= start_sec:
            continue
        if not merged or start_sec > merged[-1][1] + gap_sec:
            merged.append([start_sec, end_sec])
            continue
        merged[-1][1] = max(merged[-1][1], end_sec)
    return [(start_sec, end_sec) for start_sec, end_sec in merged]


def _invert_time_ranges(duration: float, removal_ranges: list[tuple[float, float]]) -> list[tuple[float, float]]:
    keep_ranges: list[tuple[float, float]] = []
    cursor = 0.0
    for start_sec, end_sec in _merge_time_ranges(removal_ranges):
        start_sec = max(0.0, min(start_sec, duration))
        end_sec = max(0.0, min(end_sec, duration))
        if start_sec > cursor:
            keep_ranges.append((cursor, start_sec))
        cursor = max(cursor, end_sec)
    if cursor < duration:
        keep_ranges.append((cursor, duration))
    return [(start_sec, end_sec) for start_sec, end_sec in keep_ranges if end_sec - start_sec > 0.02]


def _metadata_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _transition_duration_sec(value: Any, *, default: float = 0.0) -> float:
    if not isinstance(value, dict):
        return default
    kind = str(value.get("kind") or "").strip().lower()
    if kind in {"", "scene_match_cut", "hard_cut"}:
        return 0.0
    try:
        duration = float(value.get("duration_sec", default))
    except (TypeError, ValueError):
        duration = default
    return max(0.0, min(duration, 0.45))


def _normalize_visual_overlays(
    overlays: list[dict[str, Any]],
    duration: float,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in sorted(overlays, key=lambda candidate: float(candidate.get("start", 0.0))):
        asset_path = str(item.get("asset_path") or "").strip()
        if not asset_path or not Path(asset_path).is_file():
            continue
        start_sec = max(0.0, min(float(item.get("start", 0.0)), duration))
        if start_sec >= duration:
            continue
        end_sec = min(duration, max(start_sec + 0.1, min(float(item.get("end", start_sec + 1.5)), duration)))
        if end_sec <= start_sec:
            continue
        if normalized and start_sec < float(normalized[-1]["end"]):
            continue
        force_fullscreen = (
            _metadata_bool(item.get("force_fullscreen"))
            or _metadata_bool(item.get("fullscreen"))
            or _metadata_bool(item.get("full_screen"))
        )
        compose_mode = str(item.get("compose_mode") or item.get("composition_mode") or "replace").strip().lower()
        if compose_mode in {"fullscreen", "full_screen", "full-screen", "replace_fullscreen", "full_screen_replace"}:
            compose_mode = "replace"
        elif compose_mode in {"alpha_overlay", "full_frame_overlay"}:
            compose_mode = "overlay"
        elif compose_mode == "overlay":
            compose_mode = "overlay" if _metadata_bool(item.get("has_alpha")) else "picture_in_picture"
        elif compose_mode in {"pip", "picture-in-picture"}:
            compose_mode = "picture_in_picture"
        if force_fullscreen:
            compose_mode = "replace"
        if compose_mode not in {"replace", "picture_in_picture", "overlay"}:
            compose_mode = "replace"
        if compose_mode in {"replace", "overlay"}:
            scale = 1.0
            margin = 0
            position = "center"
            transition_in_sec = 0.0 if compose_mode == "overlay" else _transition_duration_sec(item.get("transition_in"))
            transition_out_sec = 0.0 if compose_mode == "overlay" else _transition_duration_sec(item.get("transition_out"))
        else:
            scale = max(0.22, min(float(item.get("scale", item.get("pip_scale", 0.42)) or 0.42), 0.85))
            margin = int(max(16, min(float(item.get("margin", max(min(width, height) * 0.04, 24.0))), 160)))
            position = str(item.get("position") or "bottom_right").strip().lower()
            if position not in {"top_left", "top_right", "bottom_left", "bottom_right", "top", "bottom", "center", "center_left", "center_right"}:
                position = "bottom_right"
            transition_in_sec = 0.0
            transition_out_sec = 0.0
        normalized.append(
            {
                "start": round(start_sec, 3),
                "end": round(end_sec, 3),
                "asset_path": asset_path,
                "compose_mode": compose_mode,
                "force_fullscreen": force_fullscreen,
                "scale": round(scale, 3),
                "margin": margin,
                "position": position,
                "transition_in_sec": round(transition_in_sec, 3),
                "transition_out_sec": round(transition_out_sec, 3),
            }
        )
    return normalized


def _pip_overlay_position_expr(position: str, margin: int) -> tuple[str, str]:
    margin_expr = str(max(margin, 0))
    if position == "top_left":
        return margin_expr, margin_expr
    if position == "top_right":
        return f"W-w-{margin_expr}", margin_expr
    if position == "bottom_left":
        return margin_expr, f"H-h-{margin_expr}"
    if position == "top":
        return "(W-w)/2", margin_expr
    if position == "center_left":
        return margin_expr, "(H-h)/2"
    if position == "center":
        return "(W-w)/2", "(H-h)/2"
    if position == "center_right":
        return f"W-w-{margin_expr}", "(H-h)/2"
    if position == "bottom":
        return "(W-w)/2", f"H-h-{margin_expr}"
    return f"W-w-{margin_expr}", f"H-h-{margin_expr}"


def apply_visual_overlays(
    input_path: str,
    working_dir: str,
    overlays: list[dict[str, Any]],
) -> str:
    if not overlays:
        return input_path

    clip_info = probe_video(input_path)
    duration = max(float(clip_info["duration_sec"]), 0.0)
    width = int(clip_info.get("width") or 0)
    height = int(clip_info.get("height") or 0)
    fps = float(clip_info.get("fps") or 30.0) or 30.0
    if duration <= 0.0 or width <= 0 or height <= 0:
        return input_path

    normalized = _normalize_visual_overlays(overlays, duration, width, height)
    if not normalized:
        return input_path

    boundaries = sorted(
        {
            0.0,
            duration,
            *[float(item["start"]) for item in normalized],
            *[float(item["end"]) for item in normalized],
        }
    )
    segments = [
        (boundaries[index], boundaries[index + 1])
        for index in range(len(boundaries) - 1)
        if boundaries[index + 1] - boundaries[index] > 0.02
    ]
    if not segments:
        return input_path

    unique_assets: list[str] = []
    asset_indexes: dict[str, int] = {}
    for item in normalized:
        asset_path = str(item["asset_path"])
        if asset_path not in asset_indexes:
            asset_indexes[asset_path] = len(unique_assets) + 1
            unique_assets.append(asset_path)

    command = [config.FFMPEG_PATH, "-i", input_path]
    for asset_path in unique_assets:
        command.extend(["-stream_loop", "-1", "-i", asset_path])

    filter_parts: list[str] = []
    concat_inputs: list[str] = []
    for index, (start_sec, end_sec) in enumerate(segments):
        segment_duration = end_sec - start_sec
        active_overlay = next(
            (
                item
                for item in normalized
                if start_sec >= float(item["start"]) - 0.001 and end_sec <= float(item["end"]) + 0.001
            ),
            None,
        )
        if active_overlay is None:
            filter_parts.append(
                (
                    f"[0:v]trim={start_sec:.3f}:{end_sec:.3f},setpts=PTS-STARTPTS,"
                    f"fps={math.ceil(fps)},scale={width}:{height}:force_original_aspect_ratio=decrease,"
                    f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1[v{index}]"
                )
            )
        else:
            input_index = asset_indexes[str(active_overlay["asset_path"])]
            if str(active_overlay.get("compose_mode")) == "picture_in_picture":
                pip_width = max(160, min(int(round(width * float(active_overlay.get("scale", 0.42)))), max(width - 64, 160)))
                margin = int(active_overlay.get("margin", 24))
                x_pos, y_pos = _pip_overlay_position_expr(
                    str(active_overlay.get("position") or "bottom_right"),
                    margin,
                )
                filter_parts.append(
                    (
                        f"[0:v]trim={start_sec:.3f}:{end_sec:.3f},setpts=PTS-STARTPTS,"
                        f"fps={math.ceil(fps)},scale={width}:{height}:force_original_aspect_ratio=decrease,"
                        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1[base{index}]"
                    )
                )
                filter_parts.append(
                    (
                        f"[{input_index}:v]trim=0:{segment_duration:.3f},setpts=PTS-STARTPTS,"
                        f"fps={math.ceil(fps)},scale={pip_width}:-2:force_original_aspect_ratio=decrease,"
                        f"setsar=1,format=rgba[ov{index}]"
                    )
                )
                filter_parts.append(f"[base{index}][ov{index}]overlay={x_pos}:{y_pos}:shortest=1[v{index}]")
            elif str(active_overlay.get("compose_mode")) == "overlay":
                filter_parts.append(
                    (
                        f"[0:v]trim={start_sec:.3f}:{end_sec:.3f},setpts=PTS-STARTPTS,"
                        f"fps={math.ceil(fps)},scale={width}:{height}:force_original_aspect_ratio=decrease,"
                        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1[base{index}]"
                    )
                )
                filter_parts.append(
                    (
                        f"[{input_index}:v]trim=0:{segment_duration:.3f},setpts=PTS-STARTPTS,"
                        f"fps={math.ceil(fps)},scale={width}:{height}:force_original_aspect_ratio=decrease,"
                        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=0x00000000,setsar=1,format=rgba[ov{index}]"
                    )
                )
                filter_parts.append(f"[base{index}][ov{index}]overlay=0:0:shortest=1,format=yuv420p[v{index}]")
            else:
                transition_in_sec = min(float(active_overlay.get("transition_in_sec") or 0.0), segment_duration * 0.35)
                transition_out_sec = min(float(active_overlay.get("transition_out_sec") or 0.0), segment_duration * 0.35)
                if transition_in_sec > 0.001 or transition_out_sec > 0.001:
                    filter_parts.append(
                        (
                            f"[0:v]trim={start_sec:.3f}:{end_sec:.3f},setpts=PTS-STARTPTS,"
                            f"fps={math.ceil(fps)},scale={width}:{height}:force_original_aspect_ratio=decrease,"
                            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1[base{index}]"
                        )
                    )
                    fade_filters = ""
                    if transition_in_sec > 0.001:
                        fade_filters += f",fade=t=in:st=0:d={transition_in_sec:.3f}:alpha=1"
                    if transition_out_sec > 0.001:
                        fade_start = max(segment_duration - transition_out_sec, 0.0)
                        fade_filters += f",fade=t=out:st={fade_start:.3f}:d={transition_out_sec:.3f}:alpha=1"
                    filter_parts.append(
                        (
                            f"[{input_index}:v]trim=0:{segment_duration:.3f},setpts=PTS-STARTPTS,"
                            f"fps={math.ceil(fps)},scale={width}:{height}:force_original_aspect_ratio=increase,"
                            f"crop={width}:{height},setsar=1,format=rgba{fade_filters}[ov{index}]"
                        )
                    )
                    filter_parts.append(f"[base{index}][ov{index}]overlay=0:0:shortest=1,format=yuv420p[v{index}]")
                else:
                    filter_parts.append(
                        (
                            f"[{input_index}:v]trim=0:{segment_duration:.3f},setpts=PTS-STARTPTS,"
                            f"fps={math.ceil(fps)},scale={width}:{height}:force_original_aspect_ratio=increase,"
                            f"crop={width}:{height},setsar=1[v{index}]"
                        )
                    )
        concat_inputs.append(f"[v{index}]")

    filter_parts.append(f"{''.join(concat_inputs)}concat=n={len(segments)}:v=1:a=0[v]")
    output_path = _unique_path(working_dir, ".mp4")
    command.extend(
        [
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            "[v]",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-shortest",
            "-y",
            output_path,
        ]
    )
    _run_command(command, "Failed to apply visual overlays")
    return output_path


def apply_b_roll_overlays(
    input_path: str,
    working_dir: str,
    overlays: list[dict[str, float | str]],
) -> str:
    return apply_visual_overlays(
        input_path,
        working_dir,
        [
            {
                **item,
                "compose_mode": "replace",
            }
            for item in overlays
        ],
    )


def trim_silence(
    input_path: str,
    working_dir: str,
    min_silence_duration: float = 0.5,
    silence_threshold_db: float = -35.0,
    speech_padding_sec: float = 0.12,
    merge_gap_sec: float = 0.18,
    min_keep_duration_sec: float = 0.28,
    trim_edges: bool = False,
) -> str:
    command = [
        config.FFMPEG_PATH,
        "-i",
        input_path,
        "-af",
        f"silencedetect=n={silence_threshold_db}dB:d={min_silence_duration}",
        "-f",
        "null",
        "-",
    ]
    command_text = " ".join(command)
    LOGGER.debug("Running ffmpeg command: %s", command_text)
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise VideoEngineError(
            f"Failed to detect silence: {result.stderr.strip() or result.stdout.strip()}",
            command=command_text,
        )

    silence_start_pattern = re.compile(r"silence_start:\s*([0-9.]+)")
    silence_end_pattern = re.compile(r"silence_end:\s*([0-9.]+)")
    silence_ranges: list[tuple[float, float]] = []
    pending_start: float | None = None
    for line in result.stderr.splitlines():
        start_match = silence_start_pattern.search(line)
        if start_match:
            pending_start = float(start_match.group(1))
            continue
        end_match = silence_end_pattern.search(line)
        if end_match and pending_start is not None:
            silence_ranges.append((pending_start, float(end_match.group(1))))
            pending_start = None

    clip_duration = probe_video(input_path)["duration_sec"]
    if pending_start is not None:
        silence_ranges.append((pending_start, clip_duration))

    normalized_silences = _merge_time_ranges(silence_ranges)
    removal_ranges: list[tuple[float, float]] = []
    for silence_start, silence_end in normalized_silences:
        if silence_end - silence_start < min_silence_duration:
            continue
        if not trim_edges and silence_start <= max(speech_padding_sec, 0.06):
            continue
        if not trim_edges and silence_end >= clip_duration - max(speech_padding_sec, 0.06):
            continue
        adjusted_start = 0.0 if trim_edges and silence_start <= speech_padding_sec else silence_start + speech_padding_sec
        adjusted_end = (
            clip_duration if trim_edges and silence_end >= clip_duration - speech_padding_sec else silence_end - speech_padding_sec
        )
        adjusted_start = max(0.0, min(adjusted_start, clip_duration))
        adjusted_end = max(0.0, min(adjusted_end, clip_duration))
        if adjusted_end - adjusted_start < 0.08:
            continue
        removal_ranges.append((adjusted_start, adjusted_end))

    removal_ranges = _merge_time_ranges(removal_ranges, gap_sec=merge_gap_sec)
    if not removal_ranges:
        return input_path

    while True:
        keep_segments = _invert_time_ranges(clip_duration, removal_ranges)
        changed = False
        for index, (keep_start, keep_end) in enumerate(keep_segments):
            if keep_end - keep_start >= min_keep_duration_sec:
                continue
            if 0 < index < len(keep_segments) - 1 and index <= len(removal_ranges) - 1:
                left_start, _ = removal_ranges[index - 1]
                _, right_end = removal_ranges[index]
                removal_ranges[index - 1] = (left_start, right_end)
                del removal_ranges[index]
                changed = True
                break
            if index == 0 and trim_edges and removal_ranges:
                removal_ranges[0] = (0.0, removal_ranges[0][1])
                changed = True
                break
            if index == len(keep_segments) - 1 and trim_edges and removal_ranges:
                removal_ranges[-1] = (removal_ranges[-1][0], clip_duration)
                changed = True
                break
        if not changed:
            break
        removal_ranges = _merge_time_ranges(removal_ranges, gap_sec=merge_gap_sec)

    keep_segments = _invert_time_ranges(clip_duration, removal_ranges)
    if not keep_segments:
        return input_path
    removed_duration = clip_duration - sum(end_sec - start_sec for start_sec, end_sec in keep_segments)
    if removed_duration < 0.12:
        return input_path
    return extract_segments(input_path, working_dir, keep_segments)


def _escape_subtitles_path(path: str) -> str:
    normalized = Path(path).resolve().as_posix()
    return normalized.replace("\\", "/").replace(":", r"\:").replace("'", r"\'")


def burn_subtitles(
    input_path: str,
    working_dir: str,
    srt_path: str,
    font_size: int | None = None,
    font_color: str | None = None,
    outline_color: str | None = None,
    position: str = "bottom",
    style: str = "clean_pop",
    emphasis_color: str | None = None,
    background_opacity: float | None = None,
    max_words_per_caption: int | None = None,
    max_lines: int | None = None,
    case: str | None = None,
) -> str:
    output_path = _unique_path(working_dir, ".mp4")
    metadata = probe_video(input_path)
    width = int(metadata.get("width") or 1920)
    height = int(metadata.get("height") or 1080)
    ass_path = _unique_path(working_dir, ".ass")
    compile_subtitles_to_ass(
        srt_path,
        ass_path,
        width=width,
        height=height,
        style_name=style,
        position=position,
        font_size=font_size,
        font_color=font_color,
        outline_color=outline_color,
        emphasis_color=emphasis_color,
        background_opacity=background_opacity,
        max_words_per_caption=max_words_per_caption,
        max_lines=max_lines,
        case=case,
    )
    filter_path = _escape_subtitles_path(ass_path)
    command = [
        config.FFMPEG_PATH,
        "-i",
        input_path,
        "-vf",
        f"ass='{filter_path}'",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-y",
        output_path,
    ]
    _run_command(command, "Failed to burn subtitles into video")
    return output_path


def render_vertical_short(
    input_path: str,
    working_dir: str,
    srt_path: str | None = None,
    subtitle_font_size: int | None = None,
    subtitle_font_color: str | None = None,
    subtitle_outline_color: str | None = None,
    subtitle_style: str = "creator_bold",
) -> str:
    output_path = _unique_path(working_dir, ".mp4")
    filter_parts = [
        "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,boxblur=20:2,eq=brightness=-0.10:saturation=1.15[bg]",
        "[0:v]scale=1080:1400:force_original_aspect_ratio=decrease[fg]",
        "[bg][fg]overlay=(W-w)/2:(H-h)/2[stage]",
    ]
    if srt_path:
        ass_path = _unique_path(working_dir, ".ass")
        compile_subtitles_to_ass(
            srt_path,
            ass_path,
            width=1080,
            height=1920,
            style_name=subtitle_style,
            position="bottom",
            font_size=subtitle_font_size,
            font_color=subtitle_font_color,
            outline_color=subtitle_outline_color,
        )
        filter_path = _escape_subtitles_path(ass_path)
        filter_parts.append(f"[stage]ass='{filter_path}'[v]")
    else:
        filter_parts.append("[stage]null[v]")
    command = [
        config.FFMPEG_PATH,
        "-i",
        input_path,
        "-filter_complex",
        ";".join(filter_parts),
        "-map",
        "[v]",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-y",
        output_path,
    ]
    _run_command(command, "Failed to render vertical short")
    return output_path


def apply_center_punch_ins(
    input_path: str,
    working_dir: str,
    moments: list[dict[str, float | str]],
) -> str:
    if not moments:
        return input_path
    clip_info = probe_video(input_path)
    duration = max(float(clip_info["duration_sec"]), 0.0)
    normalized_moments: list[dict[str, float]] = []
    for moment in moments:
        start_sec = max(0.0, min(float(moment.get("start", 0.0)), duration))
        end_sec = max(start_sec + 0.1, min(float(moment.get("end", start_sec + 0.8)), duration))
        if end_sec <= start_sec:
            continue
        zoom = max(1.03, min(float(moment.get("zoom", 1.12)), 1.35))
        if normalized_moments and start_sec < normalized_moments[-1]["end"]:
            normalized_moments[-1]["end"] = max(normalized_moments[-1]["end"], end_sec)
            normalized_moments[-1]["zoom"] = max(normalized_moments[-1]["zoom"], zoom)
            continue
        normalized_moments.append({"start": start_sec, "end": end_sec, "zoom": zoom})
    if not normalized_moments:
        return input_path

    boundaries = sorted({0.0, duration, *[item["start"] for item in normalized_moments], *[item["end"] for item in normalized_moments]})
    segments = [
        (boundaries[index], boundaries[index + 1])
        for index in range(len(boundaries) - 1)
        if boundaries[index + 1] - boundaries[index] > 0.02
    ]
    if not segments:
        return input_path

    filter_parts: list[str] = []
    width = int(clip_info.get("width") or 0)
    height = int(clip_info.get("height") or 0)
    if width <= 0 or height <= 0:
        return input_path
    has_audio = bool(clip_info.get("has_audio"))
    filter_parts.append(f"[0:v]split={len(segments)}" + "".join(f"[v{index}]" for index in range(len(segments))))
    if has_audio:
        filter_parts.append(f"[0:a]asplit={len(segments)}" + "".join(f"[a{index}]" for index in range(len(segments))))
    concat_inputs: list[str] = []
    for index, (start_sec, end_sec) in enumerate(segments):
        active_moment = next(
            (
                moment
                for moment in normalized_moments
                if start_sec >= moment["start"] - 0.001 and end_sec <= moment["end"] + 0.001
            ),
            None,
        )
        video_filter = f"[v{index}]trim={start_sec}:{end_sec},setpts=PTS-STARTPTS"
        if active_moment is not None:
            zoom = active_moment["zoom"]
            zoom_width = max(int(round(width * zoom)), width)
            zoom_height = max(int(round(height * zoom)), height)
            video_filter += (
                f",scale={zoom_width}:{zoom_height}"
                f",crop={width}:{height}"
            )
        video_filter += ",setsar=1"
        video_filter += f"[v{index}o]"
        filter_parts.append(video_filter)
        concat_inputs.append(f"[v{index}o]")
        if has_audio:
            filter_parts.append(f"[a{index}]atrim={start_sec}:{end_sec},asetpts=PTS-STARTPTS[a{index}o]")
            concat_inputs.append(f"[a{index}o]")

    concat_parts = "".join(concat_inputs)
    filter_parts.append(
        f"{concat_parts}concat=n={len(segments)}:v=1:a={'1' if has_audio else '0'}[v]"
        + ("[a]" if has_audio else "")
    )
    output_path = _unique_path(working_dir, ".mp4")
    command = [
        config.FFMPEG_PATH,
        "-i",
        input_path,
        "-filter_complex",
        ";".join(filter_parts),
        "-map",
        "[v]",
    ]
    if has_audio:
        command.extend(["-map", "[a]"])
    command.extend(
        [
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-y",
            output_path,
        ]
    )
    _run_command(command, "Failed to apply center punch-ins")
    return output_path


def apply_timed_effects(
    input_path: str,
    working_dir: str,
    effect_plan: dict[str, Any],
    *,
    filtergraph_path: str | None = None,
) -> str:
    from effects.compiler import build_effect_filter_graph
    from effects.schema import EffectPlan

    plan = EffectPlan.from_dict(effect_plan)
    if not plan.effects:
        return input_path
    clip_info = probe_video(input_path)
    duration = max(float(clip_info["duration_sec"]), 0.0)
    width = int(clip_info.get("width") or 0)
    height = int(clip_info.get("height") or 0)
    fps = float(clip_info.get("fps") or 30.0) or 30.0
    has_audio = bool(clip_info.get("has_audio"))
    if duration <= 0.0 or width <= 0 or height <= 0:
        return input_path

    filter_graph = build_effect_filter_graph(
        plan,
        duration=duration,
        width=width,
        height=height,
        fps=fps,
        has_audio=has_audio,
    )
    if filtergraph_path:
        Path(filtergraph_path).write_text(filter_graph, encoding="utf-8")

    output_path = _unique_path(working_dir, ".mp4")
    command = [
        config.FFMPEG_PATH,
        "-i",
        input_path,
        "-filter_complex",
        filter_graph,
        "-map",
        "[v]",
    ]
    if has_audio:
        command.extend(["-map", "0:a?"])
    else:
        command.append("-an")
    command.extend(
        [
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-y",
            output_path,
        ]
    )
    _run_command(command, "Failed to apply timed effects")
    return output_path


def apply_color_grade(
    input_path: str,
    working_dir: str,
    filter_graph: str,
    *,
    render_mode: str = "vf",
    output_label: str = "[vout]",
) -> str:
    if not str(filter_graph or "").strip():
        raise VideoEngineError("Color grade filter graph is empty.")
    output_path = _unique_path(working_dir, ".mp4")
    normalized_mode = str(render_mode or "vf").strip().lower()
    if normalized_mode == "filter_complex":
        label = str(output_label or "[vout]").strip()
        if not label.startswith("["):
            label = f"[{label}]"
        command = [
            config.FFMPEG_PATH,
            "-i",
            input_path,
            "-filter_complex",
            filter_graph,
            "-map",
            label,
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-shortest",
            "-y",
            output_path,
        ]
    else:
        command = [
            config.FFMPEG_PATH,
            "-i",
            input_path,
            "-vf",
            filter_graph,
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-y",
            output_path,
        ]
    _run_command(command, "Failed to apply color grade")
    return output_path


def auto_color_grade(
    input_path: str,
    working_dir: str,
    *,
    look: str = "auto",
    intensity: float = 1.0,
    sample_count: int = 9,
    mode: str = "auto",
    max_shots: int = 18,
    candidate_count: int = 4,
) -> tuple[str, dict[str, Any]]:
    metadata = probe_video(input_path)
    plan = build_shot_aware_color_grade_plan(
        input_path,
        metadata,
        look=look,
        intensity=intensity,
        sample_count=sample_count,
        mode=mode,
        max_shots=max_shots,
        candidate_count=candidate_count,
    )
    output_path = apply_color_grade(
        input_path,
        working_dir,
        plan.filter_graph,
        render_mode=plan.render_mode,
        output_label=plan.output_label or "[vout]",
    )
    plan_payload = plan.to_dict()
    try:
        output_metadata = probe_video(output_path)
        plan_payload["validation"] = validate_color_grade_output(
            output_path,
            output_metadata,
            sample_count=min(max(sample_count, 3), 5),
        )
        manifest = dict(plan_payload.get("manifest") or {})
        manifest_shots = list(manifest.get("shots") or [])
        if manifest_shots:
            plan_payload["validation"]["shot_validation"] = validate_color_grade_output_by_shots(
                output_path,
                output_metadata,
                manifest_shots,
                sample_count=3,
            )
    except (ColorGradePlanningError, VideoEngineError, OSError) as exc:
        validation_warning = f"Could not validate graded output: {exc}"
        plan_payload["validation"] = {
            "passed": False,
            "score": 0.0,
            "warnings": [validation_warning],
            "analysis": {},
        }
        plan_payload.setdefault("warnings", []).append(validation_warning)
    return output_path, plan_payload


def export(
    input_path: str,
    output_path: str,
    preset: dict,
    progress_callback: Callable[[float], None] | None = None,
) -> str:
    metadata = probe_video(input_path)
    duration = max(metadata["duration_sec"], 0.001)
    _validate_export_request(input_path, output_path, preset, metadata)
    final_path = Path(output_path).expanduser().resolve()
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temporary_export_path(final_path)
    command = _build_export_command(input_path, str(temp_path), preset)
    try:
        _run_export_command(command, duration, progress_callback=progress_callback)
        _validate_export_output(input_path, str(temp_path), preset, metadata, command)
    except VideoEngineError as exc:
        if not _should_retry_export_with_low_memory_x264(exc, preset):
            _remove_partial_output(str(temp_path))
            raise
        _remove_partial_output(str(temp_path))
        retry_command = _build_export_command(input_path, str(temp_path), preset, low_memory_x264=True)
        LOGGER.warning(
            "Retrying export with low-memory x264 settings after encoder allocation failure."
        )
        try:
            _run_export_command(retry_command, duration, progress_callback=progress_callback)
            _validate_export_output(input_path, str(temp_path), preset, metadata, retry_command)
        except VideoEngineError:
            _remove_partial_output(str(temp_path))
            raise
    try:
        os.replace(temp_path, final_path)
    except OSError as exc:
        _remove_partial_output(str(temp_path))
        raise VideoEngineError(f"Export failed while moving validated output into place: {exc}") from exc
    return str(final_path)


def _build_export_command(
    input_path: str,
    output_path: str,
    preset: dict,
    *,
    low_memory_x264: bool = False,
) -> list[str]:
    command = [config.FFMPEG_PATH, "-hide_banner", "-nostdin", "-i", input_path]
    if preset.get("audio_only"):
        command.extend(["-map", "0:a:0?"])
        if preset.get("audio_codec"):
            command.extend(["-vn", "-c:a", preset["audio_codec"]])
        if preset.get("audio_bitrate"):
            command.extend(["-b:a", preset["audio_bitrate"]])
    else:
        command.extend(["-map", "0:v:0", "-map", "0:a?"])
        if preset.get("resolution"):
            command.extend(
                [
                    "-vf",
                    _export_scale_filter(
                        str(preset["resolution"]),
                        scale_mode=str(preset.get("scale_mode") or "fit"),
                    ),
                ]
            )
        if preset.get("fps"):
            command.extend(["-r", str(preset["fps"])])
        video_codec = preset.get("video_codec")
        if video_codec:
            command.extend(["-c:v", video_codec])
        if preset.get("audio_codec"):
            command.extend(["-c:a", preset["audio_codec"]])
        if preset.get("video_bitrate"):
            command.extend(["-b:v", preset["video_bitrate"]])
        if preset.get("audio_bitrate"):
            command.extend(["-b:a", preset["audio_bitrate"]])
        if video_codec == "libx264":
            command.extend(["-pix_fmt", "yuv420p"])
            x264_preset = str(preset.get("x264_preset") or preset.get("preset") or "").strip()
            if x264_preset and not low_memory_x264:
                command.extend(["-preset", x264_preset])
            if low_memory_x264:
                command.extend(
                    [
                        "-preset",
                        "veryfast",
                        "-threads",
                        "2",
                        "-x264-params",
                        "rc-lookahead=10:sync-lookahead=0:sliced-threads=1",
                    ]
                )
        command.extend(["-max_muxing_queue_size", "1024"])
        if _supports_faststart(preset):
            command.extend(["-movflags", "+faststart"])
    command.extend(["-y", output_path])
    return command


def _validate_export_request(
    input_path: str,
    output_path: str,
    preset: dict,
    metadata: dict[str, Any],
) -> None:
    if not input_path or not Path(input_path).is_file():
        raise VideoEngineError(f"Export input file does not exist: {input_path}")
    if Path(input_path).expanduser().resolve() == Path(output_path).expanduser().resolve():
        raise VideoEngineError("Export output path must be different from the input video path.")
    if preset.get("audio_only") and not metadata.get("has_audio"):
        raise VideoEngineError("Cannot export an audio-only preset because the source has no audio stream.")
    if not preset.get("audio_only") and (int(metadata.get("width") or 0) <= 0 or int(metadata.get("height") or 0) <= 0):
        raise VideoEngineError("Cannot export video because the source has no readable video stream.")
    if preset.get("resolution"):
        _parse_export_resolution(str(preset["resolution"]))
        _normalize_export_scale_mode(preset.get("scale_mode", "fit"))
    if preset.get("fps"):
        try:
            fps = float(preset["fps"])
        except (TypeError, ValueError) as exc:
            raise VideoEngineError(f"Invalid export FPS: {preset.get('fps')!r}") from exc
        if fps <= 0 or fps > 240:
            raise VideoEngineError("Export FPS must be greater than 0 and no more than 240.")
    for key in ("video_bitrate", "audio_bitrate"):
        if preset.get(key):
            try:
                _bitrate_to_bits(str(preset[key]))
            except ValueError as exc:
                raise VideoEngineError(f"Invalid export bitrate for {key}: {preset.get(key)!r}") from exc


def _temporary_export_path(final_path: Path) -> Path:
    suffix = final_path.suffix or ".tmp"
    return final_path.with_name(f".{final_path.stem}.{uuid.uuid4().hex}.tmp{suffix}")


def _parse_export_resolution(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"\s*(\d{2,5})\s*x\s*(\d{2,5})\s*", str(value or ""), flags=re.IGNORECASE)
    if not match:
        raise VideoEngineError(f"Invalid export resolution {value!r}. Use WIDTHxHEIGHT, for example 1920x1080.")
    width = int(match.group(1))
    height = int(match.group(2))
    if width <= 0 or height <= 0:
        raise VideoEngineError("Export resolution must be positive.")
    if width % 2 or height % 2:
        raise VideoEngineError("Export resolution must use even width and height for video codec compatibility.")
    return width, height


def _normalize_export_scale_mode(value: object) -> str:
    mode = str(value or "fit").strip().lower()
    if mode in {"fit", "contain", "letterbox"}:
        return "fit"
    if mode in {"fill", "cover", "crop"}:
        return "fill"
    if mode in {"stretch", "distort"}:
        return "stretch"
    raise VideoEngineError("Export scale_mode must be one of: fit, fill, stretch.")


def _export_scale_filter(resolution: str, scale_mode: str = "fit") -> str:
    width, height = _parse_export_resolution(resolution)
    mode = _normalize_export_scale_mode(scale_mode)
    if mode == "fit":
        return (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease:flags=lanczos,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
            "setsar=1"
        )
    if mode == "fill":
        return (
            f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={width}:{height},"
            "setsar=1"
        )
    return f"scale={width}:{height}:flags=lanczos,setsar=1"


def _supports_faststart(preset: dict) -> bool:
    suffix = str(preset.get("format") or "").strip().lower().lstrip(".")
    return suffix in {"mp4", "m4v", "mov", ""}


def _run_export_command(
    command: list[str],
    duration: float,
    *,
    progress_callback: Callable[[float], None] | None = None,
) -> None:
    command_text = " ".join(command)
    LOGGER.debug("Running ffmpeg command: %s", command_text)
    try:
        process = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True)
    except OSError as exc:
        raise VideoEngineError(f"Failed to launch export command: {exc}", command=command_text) from exc
    if process.stderr is None:
        raise VideoEngineError("Failed to launch export command.", command=command_text)
    stderr_lines: list[str] = []
    for line in process.stderr:
        stderr_lines.append(line)
        if "time=" in line and progress_callback:
            marker = line.split("time=", 1)[1].split()[0]
            try:
                seconds = parse_timestamp(marker)
            except ValueError:
                continue
            progress_callback(min(seconds / duration, 1.0))
    if process.wait() != 0:
        stderr_text = "".join(stderr_lines).strip()
        message = f"Export failed: {stderr_text}" if stderr_text else "Export failed."
        raise VideoEngineError(message, command=command_text)
    if progress_callback:
        progress_callback(1.0)


def _validate_export_output(
    input_path: str,
    output_path: str,
    preset: dict,
    source_metadata: dict[str, Any],
    command: list[str],
) -> None:
    path = Path(output_path)
    issues: list[str] = []
    if not path.is_file():
        raise VideoEngineError(f"Export failed validation: output was not created: {path}", command=" ".join(command))
    output_size = path.stat().st_size
    if output_size <= 0:
        raise VideoEngineError(f"Export failed validation: output is empty: {path}", command=" ".join(command))
    try:
        output_metadata = probe_video(str(path))
    except Exception as exc:  # noqa: BLE001
        raise VideoEngineError(f"Export failed validation: ffprobe could not read output: {exc}", command=" ".join(command)) from exc

    if preset.get("audio_only"):
        _validate_audio_only_export_output(preset, output_metadata, issues)
    else:
        _validate_video_export_output(preset, source_metadata, output_metadata, issues)
    _validate_export_duration(source_metadata, output_metadata, issues)
    _validate_export_decode(str(path), issues)

    if issues:
        rendered = "; ".join(issues[:4])
        if len(issues) > 4:
            rendered += f"; plus {len(issues) - 4} more issue(s)"
        raise VideoEngineError(f"Export failed validation: {rendered}", command=" ".join(command))
    LOGGER.debug(
        "Export validation passed for %s (%s bytes, source=%s)",
        output_path,
        output_size,
        input_path,
    )


def _validate_audio_only_export_output(
    preset: dict,
    output_metadata: dict[str, Any],
    issues: list[str],
) -> None:
    if not output_metadata.get("has_audio"):
        issues.append("audio-only export has no readable audio stream")
    if int(output_metadata.get("width") or 0) > 0 or int(output_metadata.get("height") or 0) > 0:
        issues.append("audio-only export unexpectedly contains a video stream")
    expected_audio = _expected_export_audio_codec(preset)
    observed_audio = str(output_metadata.get("audio_codec") or "").lower()
    if expected_audio and observed_audio and observed_audio not in _audio_codec_aliases(expected_audio):
        issues.append(f"audio codec {observed_audio!r} does not match expected {expected_audio!r}")
    _validate_export_container(preset, output_metadata, issues)


def _validate_video_export_output(
    preset: dict,
    source_metadata: dict[str, Any],
    output_metadata: dict[str, Any],
    issues: list[str],
) -> None:
    output_width = int(output_metadata.get("width") or 0)
    output_height = int(output_metadata.get("height") or 0)
    if output_width <= 0 or output_height <= 0:
        issues.append("video export has no readable video stream")
    if preset.get("resolution"):
        expected_width, expected_height = _parse_export_resolution(str(preset["resolution"]))
        if abs(output_width - expected_width) > 2 or abs(output_height - expected_height) > 2:
            issues.append(
                f"output dimensions {output_width}x{output_height} do not match requested "
                f"{expected_width}x{expected_height}"
            )
    expected_fps = _positive_float(preset.get("fps"))
    observed_fps = _positive_float(output_metadata.get("fps"))
    if expected_fps and observed_fps and abs(observed_fps - expected_fps) > max(0.5, expected_fps * 0.05):
        issues.append(f"output FPS {observed_fps:.3f} does not match requested {expected_fps:.3f}")
    expected_video = _expected_export_video_codec(preset, source_metadata)
    observed_video = str(output_metadata.get("codec") or "").lower()
    if expected_video and observed_video and observed_video not in _video_codec_aliases(expected_video):
        issues.append(f"video codec {observed_video!r} does not match expected {expected_video!r}")
    if source_metadata.get("has_audio"):
        if not output_metadata.get("has_audio"):
            issues.append("source has audio but exported output has no audio stream")
        else:
            expected_audio = _expected_export_audio_codec(preset)
            observed_audio = str(output_metadata.get("audio_codec") or "").lower()
            if expected_audio and observed_audio and observed_audio not in _audio_codec_aliases(expected_audio):
                issues.append(f"audio codec {observed_audio!r} does not match expected {expected_audio!r}")
    _validate_export_container(preset, output_metadata, issues)


def _validate_export_container(
    preset: dict,
    output_metadata: dict[str, Any],
    issues: list[str],
) -> None:
    expected = str(preset.get("format") or "").strip().lower().lstrip(".")
    if not expected:
        return
    accepted = {
        "mp4": {"mov", "mp4", "m4a", "3gp", "3g2", "mj2"},
        "m4v": {"mov", "mp4", "m4a", "3gp", "3g2", "mj2"},
        "mov": {"mov", "mp4", "m4a", "3gp", "3g2", "mj2"},
        "mp3": {"mp3"},
        "wav": {"wav"},
        "aac": {"aac", "adts"},
        "mkv": {"matroska", "webm"},
        "webm": {"matroska", "webm"},
    }.get(expected)
    if not accepted:
        return
    observed = {token.strip().lower() for token in str(output_metadata.get("format") or "").split(",") if token.strip()}
    if not observed.intersection(accepted):
        issues.append(f"container {output_metadata.get('format')!r} does not match expected {expected!r}")


def _validate_export_duration(
    source_metadata: dict[str, Any],
    output_metadata: dict[str, Any],
    issues: list[str],
) -> None:
    source_duration = _positive_float(source_metadata.get("duration_sec"))
    output_duration = _positive_float(output_metadata.get("duration_sec"))
    if not source_duration or not output_duration:
        return
    tolerance = max(1.0, source_duration * 0.03)
    if abs(output_duration - source_duration) > tolerance:
        issues.append(f"output duration {output_duration:.2f}s differs from source {source_duration:.2f}s")


def _validate_export_decode(output_path: str, issues: list[str]) -> None:
    command = [
        config.FFMPEG_PATH,
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-xerror",
        "-i",
        output_path,
        "-map",
        "0",
        "-f",
        "null",
        "-",
    ]
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=max(15, int(config.ENCODE_VALIDATION_TIMEOUT_SEC)),
        )
    except subprocess.TimeoutExpired:
        issues.append(f"full decode validation timed out after {int(config.ENCODE_VALIDATION_TIMEOUT_SEC)} seconds")
        return
    except OSError as exc:
        issues.append(f"could not launch full decode validation: {exc}")
        return
    if result.returncode != 0:
        stderr = " ".join((result.stderr or "").split())
        if len(stderr) > 300:
            stderr = stderr[:300] + "..."
        issues.append(f"full decode validation failed{': ' + stderr if stderr else ''}")


def _expected_export_video_codec(preset: dict, source_metadata: dict[str, Any]) -> str:
    codec = str(preset.get("video_codec") or "").strip().lower()
    if codec == "copy":
        return str(source_metadata.get("codec") or "").lower()
    return {
        "libx264": "h264",
        "libx265": "hevc",
        "libaom-av1": "av1",
        "libvpx-vp9": "vp9",
        "prores_ks": "prores",
    }.get(codec, codec)


def _expected_export_audio_codec(preset: dict) -> str:
    codec = str(preset.get("audio_codec") or "").strip().lower()
    if codec == "copy":
        return ""
    return {
        "libmp3lame": "mp3",
        "libopus": "opus",
    }.get(codec, codec)


def _video_codec_aliases(codec: str) -> set[str]:
    return {
        "h264": {"h264"},
        "hevc": {"hevc", "h265"},
        "av1": {"av1"},
        "vp9": {"vp9"},
        "prores": {"prores"},
    }.get(codec, {codec})


def _audio_codec_aliases(codec: str) -> set[str]:
    return {
        "aac": {"aac"},
        "mp3": {"mp3"},
        "opus": {"opus"},
    }.get(codec, {codec})


def _positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _should_retry_export_with_low_memory_x264(exc: VideoEngineError, preset: dict) -> bool:
    if preset.get("audio_only"):
        return False
    if preset.get("video_codec") != "libx264":
        return False
    message = str(exc).lower()
    retry_markers = (
        "malloc",
        "cannot allocate memory",
        "not enough memory",
        "error submitting video frame to the encoder",
        "generic error in an external library",
    )
    return any(marker in message for marker in retry_markers)


def _remove_partial_output(output_path: str) -> None:
    try:
        Path(output_path).unlink(missing_ok=True)
    except OSError:
        LOGGER.warning("Could not remove partial export output before retry: %s", output_path)


def extract_frame(input_path: str, working_dir: str, timestamp_sec: float) -> str:
    output_path = _unique_path(working_dir, ".jpg")
    stream = ffmpeg.output(ffmpeg.input(input_path, ss=timestamp_sec), output_path, vframes=1)
    _run_ffmpeg(stream, "Failed to extract frame")
    return output_path


def _bitrate_to_bits(rate: str | None) -> int:
    if not rate:
        return 0
    raw = rate.strip().lower()
    if raw.endswith("k"):
        return int(float(raw[:-1]) * 1000)
    if raw.endswith("m"):
        return int(float(raw[:-1]) * 1_000_000)
    return int(float(raw))


def estimate_output_size(input_path: str, preset: dict) -> int:
    duration = probe_video(input_path)["duration_sec"]
    video_bitrate = _bitrate_to_bits(preset.get("video_bitrate"))
    audio_bitrate = _bitrate_to_bits(preset.get("audio_bitrate"))
    return int((video_bitrate + audio_bitrate) * duration / 8)


def check_disk_space(path: str, required_bytes: int) -> bool:
    destination = Path(path)
    base = destination if destination.is_dir() else destination.parent
    while not base.exists() and base != base.parent:
        base = base.parent
    usage = shutil.disk_usage(base)
    return usage.free >= required_bytes
