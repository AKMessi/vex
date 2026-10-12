"""Compile source spans and approved visuals into one FFmpeg render."""

from __future__ import annotations
from pathlib import Path
import uuid
import config
from vex_runtime.edit_graph import EditGraph, rational_text
from vex_visuals.evidence import file_digest


def render_edit_graph(
    graph: EditGraph,
    working_dir: str | Path,
    *,
    width: int,
    height: int,
    fidelity: str = "balanced",
) -> str:
    from engine import (
        probe_video,
        _run_command,
        _speed_audio_filter,
        _pip_overlay_position_expr,
    )

    root = Path(working_dir)
    root.mkdir(parents=True, exist_ok=True)
    output = root / (
        f"visual_graph_{uuid.uuid4().hex}." + ("mov" if fidelity == "master" else "mp4")
    )
    command = [config.FFMPEG_PATH, "-hide_banner", "-nostdin"]
    sources = list(graph.sources.values())
    source_index = {source.source_id: index for index, source in enumerate(sources)}
    has_audio = {
        source.source_id: bool(probe_video(source.media_path).get("has_audio"))
        for source in sources
    }
    for source in sources:
        command.extend(["-i", source.media_path])
    for layer in graph.layers:
        if (
            layer.checksum_sha256
            and file_digest(layer.asset_path) != layer.checksum_sha256
        ):
            raise ValueError("Approved visual asset changed before graph rendering")
        command.extend(["-i", layer.asset_path])
    filters = []
    concat = []
    audio = []
    want_audio = any(has_audio.values())
    fps = rational_text(graph.fps)
    for index, span in enumerate(graph.spans):
        source = source_index[span.source_id]
        ratio = float(span.duration / (span.source_end - span.source_start))
        filters.append(
            f"[{source}:v]trim=start={float(span.source_start):.9f}:end={float(span.source_end):.9f},setpts=(PTS-STARTPTS)*{ratio:.12f},fps={fps},scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1[v{index}]"
        )
        concat.append(f"[v{index}]")
        if want_audio:
            if has_audio[span.source_id]:
                speed = float((span.source_end - span.source_start) / span.duration)
                filters.append(
                    f"[{source}:a]atrim=start={float(span.source_start):.9f}:end={float(span.source_end):.9f},asetpts=PTS-STARTPTS,{_speed_audio_filter(speed)},aformat=sample_rates=48000:channel_layouts=stereo[a{index}]"
                )
            else:
                filters.append(
                    f"anullsrc=r=48000:cl=stereo,atrim=duration={float(span.duration):.9f}[a{index}]"
                )
            audio.append(f"[a{index}]")
    filters.append("".join(concat) + f"concat=n={len(concat)}:v=1:a=0[base]")
    if want_audio:
        filters.append("".join(audio) + f"concat=n={len(audio)}:v=0:a=1[sound]")
    current = "base"
    for index, layer in enumerate(graph.layers):
        asset = len(sources) + index
        duration = float(layer.output_end - layer.output_start)
        target_width = (
            width
            if layer.compose_mode != "picture_in_picture"
            else max(160, round(width * layer.scale))
        )
        scale = (
            f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}"
            if layer.compose_mode != "picture_in_picture"
            else f"scale={target_width}:-2"
        )
        fades = ""
        if layer.transition_in_sec:
            fades += f",fade=t=in:st=0:d={layer.transition_in_sec:.9f}:alpha=1"
        if layer.transition_out_sec:
            fades += f",fade=t=out:st={duration-layer.transition_out_sec:.9f}:d={layer.transition_out_sec:.9f}:alpha=1"
        filters.append(
            f"[{asset}:v]trim=start={float(layer.asset_start):.9f}:duration={duration:.9f},setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration={duration:.9f},fps={fps},{scale},format=rgba{fades},setpts=PTS+{float(layer.output_start):.9f}/TB[layer{index}]"
        )
        x, y = (
            ("0", "0")
            if layer.compose_mode != "picture_in_picture"
            else _pip_overlay_position_expr(layer.position, layer.margin)
        )
        next_name = f"composite{index}"
        filters.append(
            f"[{current}][layer{index}]overlay=x={x}:y={y}:eof_action=pass:enable='between(t,{float(layer.output_start):.9f},{float(layer.output_end):.9f})'[{next_name}]"
        )
        current = next_name
    script = root / f"visual_graph_{uuid.uuid4().hex}.ffscript"
    script.write_text(";\n".join(filters), encoding="utf-8")
    command.extend(["-filter_complex_script", str(script), "-map", f"[{current}]"])
    if any(has_audio.values()):
        command.extend(["-map", "[sound]", "-c:a", "aac", "-b:a", "192k"])
    if fidelity == "master":
        command.extend(
            ["-c:v", "prores_ks", "-profile:v", "4", "-pix_fmt", "yuv444p10le"]
        )
    else:
        command.extend(
            [
                "-c:v",
                "libx264",
                "-crf",
                "16",
                "-preset",
                "medium",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
            ]
        )
    command.extend(["-r", fps, "-t", str(float(graph.duration)), "-y", str(output)])
    _run_command(command, "Failed to render source-based visual graph")
    return str(output)
