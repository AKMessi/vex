from __future__ import annotations

import logging
import math
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Callable

import ffmpeg
from moviepy.editor import ColorClip, CompositeVideoClip, TextClip, VideoFileClip

import config

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
    return {
        "duration_sec": float(format_info.get("duration") or video_stream.get("duration") or 0.0),
        "fps": _fps_to_float(video_stream.get("avg_frame_rate", "0/0")),
        "width": int(video_stream.get("width") or 0),
        "height": int(video_stream.get("height") or 0),
        "codec": video_stream.get("codec_name", "unknown"),
        "has_audio": audio_stream is not None,
        "size_bytes": int(format_info.get("size") or os.path.getsize(path)),
        "format": format_info.get("format_name", "unknown"),
    }


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
        "\n".join(f"file '{Path(path).as_posix()}'" for path in normalized_files),
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
    if segment_start is None and segment_end is None:
        command = [
            config.FFMPEG_PATH,
            "-i",
            input_path,
            "-filter_complex",
            f"[0:v]setpts={1/factor:.8f}*PTS[v];[0:a]{_speed_audio_filter(factor)}[a]",
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
        _run_command(command, "Failed to adjust video speed")
        return output_path

    start = segment_start or 0.0
    end = segment_end if segment_end is not None else probe_video(input_path)["duration_sec"]
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
    command = [
        config.FFMPEG_PATH,
        "-i",
        input_path,
        "-filter_complex",
        filter_complex,
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
    _run_command(command, "Failed to adjust segment speed")
    return output_path


def fade_in(input_path: str, working_dir: str, duration: float) -> str:
    output_path = _unique_path(working_dir, ".mp4")
    command = [
        config.FFMPEG_PATH,
        "-i",
        input_path,
        "-vf",
        f"fade=t=in:st=0:d={duration}",
        "-af",
        f"afade=t=in:st=0:d={duration}",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-y",
        output_path,
    ]
    _run_command(command, "Failed to add fade in")
    return output_path


def fade_out(input_path: str, working_dir: str, duration: float) -> str:
    output_path = _unique_path(working_dir, ".mp4")
    clip_info = probe_video(input_path)
    start = max(clip_info["duration_sec"] - duration, 0.0)
    command = [
        config.FFMPEG_PATH,
        "-i",
        input_path,
        "-vf",
        f"fade=t=out:st={start}:d={duration}",
        "-af",
        f"afade=t=out:st={start}:d={duration}",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-y",
        output_path,
    ]
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
    suffix = ".m4a" if fmt == "aac" else f".{fmt}"
    output_path = _unique_path(working_dir, suffix)
    codec = {"mp3": "libmp3lame", "wav": "pcm_s16le", "aac": "aac"}[fmt]
    stream = ffmpeg.output(ffmpeg.input(input_path).audio, output_path, acodec=codec)
    _run_ffmpeg(stream, "Failed to extract audio")
    return output_path


def replace_audio(video_path: str, audio_path: str, working_dir: str, mix: bool, mix_ratio: float) -> str:
    output_path = _unique_path(working_dir, ".mp4")
    if mix:
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


def mute_segment(input_path: str, working_dir: str, start_sec: float, end_sec: float) -> str:
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


def trim_silence(
    input_path: str,
    working_dir: str,
    min_silence_duration: float = 0.5,
    silence_threshold_db: float = -35.0,
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

    keep_segments: list[tuple[float, float]] = []
    cursor = 0.0
    for silence_start, silence_end in silence_ranges:
        if silence_start > cursor:
            keep_segments.append((cursor, silence_start))
        cursor = max(cursor, silence_end)
    if cursor < clip_duration:
        keep_segments.append((cursor, clip_duration))

    keep_segments = [
        (start_sec, end_sec)
        for start_sec, end_sec in keep_segments
        if end_sec - start_sec > 0.0
    ]
    if len(keep_segments) < 2:
        return input_path
    return extract_segments(input_path, working_dir, keep_segments)


def _ass_color(value: str) -> str:
    color_map = {
        "white": "00FFFFFF",
        "black": "00000000",
        "yellow": "0000FFFF",
        "red": "000000FF",
        "green": "0000FF00",
        "blue": "00FF0000",
        "cyan": "00FFFF00",
        "magenta": "00FF00FF",
    }
    return color_map.get(str(value).strip().lower(), color_map["white"])


def _escape_subtitles_path(path: str) -> str:
    normalized = Path(path).resolve().as_posix()
    return normalized.replace("\\", "/").replace(":", r"\:").replace("'", r"\'")


def burn_subtitles(
    input_path: str,
    working_dir: str,
    srt_path: str,
    font_size: int = 24,
    font_color: str = "white",
    outline_color: str = "black",
    position: str = "bottom",
) -> str:
    output_path = _unique_path(working_dir, ".mp4")
    position_styles = {
        "bottom": {"Alignment": "2", "MarginV": "30"},
        "center": {"Alignment": "5", "MarginV": "30"},
        "top": {"Alignment": "8", "MarginV": "30"},
    }
    style = position_styles.get(position, position_styles["bottom"])
    filter_path = _escape_subtitles_path(srt_path)
    force_style = (
        f"Fontsize={font_size},"
        f"PrimaryColour=&H{_ass_color(font_color)},"
        f"OutlineColour=&H{_ass_color(outline_color)},"
        "Outline=2,"
        f"Alignment={style['Alignment']},"
        f"MarginV={style['MarginV']}"
    )
    command = [
        config.FFMPEG_PATH,
        "-i",
        input_path,
        "-vf",
        f"subtitles='{filter_path}':force_style='{force_style}'",
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
    subtitle_font_size: int = 10,
    subtitle_font_color: str = "white",
    subtitle_outline_color: str = "black",
) -> str:
    output_path = _unique_path(working_dir, ".mp4")
    filter_parts = [
        "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,boxblur=20:2,eq=brightness=-0.10:saturation=1.15[bg]",
        "[0:v]scale=1080:1400:force_original_aspect_ratio=decrease[fg]",
        "[bg][fg]overlay=(W-w)/2:(H-h)/2[stage]",
        "[stage]drawbox=x=88:y=1600:w=904:h=190:color=black@0.28:t=fill[base]",
    ]
    if srt_path:
        filter_path = _escape_subtitles_path(srt_path)
        force_style = (
            f"Fontsize={subtitle_font_size},"
            f"PrimaryColour=&H{_ass_color(subtitle_font_color)},"
            f"OutlineColour=&H{_ass_color(subtitle_outline_color)},"
            "BackColour=&H66000000,"
            "Bold=1,"
            "Outline=1,"
            "Shadow=0,"
            "BorderStyle=4,"
            "Alignment=2,"
            "MarginL=140,"
            "MarginR=140,"
            "MarginV=92,"
            "Spacing=0.05,"
            "WrapStyle=2"
        )
        filter_parts.append(
            f"[base]subtitles='{filter_path}':original_size=1080x1920:force_style='{force_style}'[v]"
        )
    else:
        filter_parts.append("[base]null[v]")
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


def export(
    input_path: str,
    output_path: str,
    preset: dict,
    progress_callback: Callable[[float], None] | None = None,
) -> str:
    metadata = probe_video(input_path)
    duration = max(metadata["duration_sec"], 0.001)
    command = [config.FFMPEG_PATH, "-i", input_path]
    if preset.get("audio_only"):
        if preset.get("audio_codec"):
            command.extend(["-vn", "-c:a", preset["audio_codec"]])
        if preset.get("audio_bitrate"):
            command.extend(["-b:a", preset["audio_bitrate"]])
    else:
        if preset.get("resolution"):
            command.extend(["-vf", f"scale={preset['resolution'].replace('x', ':')}"])
        if preset.get("fps"):
            command.extend(["-r", str(preset["fps"])])
        if preset.get("video_codec"):
            command.extend(["-c:v", preset["video_codec"]])
        if preset.get("audio_codec"):
            command.extend(["-c:a", preset["audio_codec"]])
        if preset.get("video_bitrate"):
            command.extend(["-b:v", preset["video_bitrate"]])
        if preset.get("audio_bitrate"):
            command.extend(["-b:a", preset["audio_bitrate"]])
        command.extend(["-movflags", "+faststart"])
    command.extend(["-y", output_path])
    command_text = " ".join(command)
    LOGGER.debug("Running ffmpeg command: %s", command_text)
    process = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True)
    if process.stderr is None:
        raise VideoEngineError("Failed to launch export command.", command=command_text)
    for line in process.stderr:
        if "time=" in line and progress_callback:
            marker = line.split("time=", 1)[1].split()[0]
            try:
                seconds = parse_timestamp(marker)
            except ValueError:
                continue
            progress_callback(min(seconds / duration, 1.0))
    if process.wait() != 0:
        raise VideoEngineError("Export failed.", command=command_text)
    if progress_callback:
        progress_callback(1.0)
    return output_path


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
