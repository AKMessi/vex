from __future__ import annotations

import logging
import math
import os
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
