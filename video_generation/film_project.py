from __future__ import annotations

import html
import json
import math
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import config
from engine import probe_video
from vex_visuals.evidence import file_digest, payload_digest
from video_generation.film_direction import FilmDirection, PALETTES
from video_generation.models import BeatGraph, VideoGenerationRequest
from video_generation.remotion_project import (
    TIMELINE_VERSION, _caption_cues, _stage_media, _write_json, stage_remotion_runtime,
)
from video_generation.renderer import CommandRecord, HyperframesVideoRuntime, VideoGenerationRuntimeError, _run_logged


def build_film_project(
    project_dir: Path, request: VideoGenerationRequest, graph: BeatGraph,
    direction: FilmDirection, runtime: HyperframesVideoRuntime,
    *, narration: Path | None, music: Path | None,
    reuse_project: Path | None = None,
) -> tuple[dict[str, str], list[CommandRecord]]:
    if len(graph.beats) != len(direction.shots) or any(
        beat.narration != shot.narration for beat, shot in zip(graph.beats, direction.shots)
    ):
        raise VideoGenerationRuntimeError("The film treatment no longer matches the narration beats.")
    root = project_dir / "remotion"
    root.mkdir()
    stage_remotion_runtime(root)
    fps = request.fps
    end_frame = max(len(graph.beats), math.ceil(graph.duration_sec * fps))
    cuts = [0]
    for index, beat in enumerate(graph.beats[1:], 1):
        cuts.append(min(end_frame - len(graph.beats) + index, max(cuts[-1] + 1, round(beat.start * fps))))
    cuts.append(end_frame)
    palette = direction.palette if isinstance(direction.palette, dict) else PALETTES[direction.palette]
    records: list[CommandRecord] = []
    scenes = []
    shot_entries = []
    status_path = project_dir / "film_production.json"
    report = {
        "version": "vex-hybrid-film-v2", "director": {"provider": direction.provider, "model": direction.model},
        "palette": direction.palette, "fps": fps, "duration_frames": end_frame, "shots": shot_entries,
    }
    reusable = {}
    if reuse_project is not None:
        previous = json.loads((reuse_project / "film_production.json").read_text(encoding="utf-8"))
        reusable = {row["id"]: row for row in previous["shots"] if row.get("status") == "staged"}
    for index, (beat, shot) in enumerate(zip(graph.beats, direction.shots, strict=True)):
        first, last = cuts[index:index + 2]
        frames = last - first
        scene = {
            "id": beat.beat_id, "start_frame": first, "duration_frames": frames,
            "program": None, "standfirst": shot.narration,
            "producer": shot.producer,
            "transition": "cut" if first == 0 else shot.transition,
            "transition_color": palette["accent"],
        }
        shot_root = project_dir / "shots" / beat.beat_id
        shot_root.mkdir(parents=True)
        payload = {
            **asdict(shot), "palette": palette, "duration": frames / fps,
            "fps": fps, "width": request.width, "height": request.height,
        }
        _write_json(shot_root / "shot.json", payload)
        row = {
            "id": beat.beat_id, "producer": scene["producer"], "start_frame": first,
            "duration_frames": frames, "intent": shot.intent,
            "source": (shot_root / "shot.json").relative_to(project_dir).as_posix(),
            "source_sha256": payload_digest(payload),
            "status": "planned",
        }
        shot_entries.append(row)
        _write_json(status_path, report)
        try:
            if shot.artwork and shot.producer == "remotion":
                scene["artwork"] = shot.artwork
                row["status"] = "inline"
            elif shot.kind == "editorial":
                scene["editorial"] = {"palette": palette, "labels": shot.labels, "action": shot.action}
                row["status"] = "inline"
            else:
                source_path = _stage_native_source(shot_root, payload, shot.kind)
                row["renderer_source"] = source_path.relative_to(project_dir).as_posix()
                # The Remotion composition applies the transition above the native clip.
                native_digest = payload_digest({key: value for key, value in payload.items() if key != "transition"})
                row["native_input_sha256"] = native_digest
                if request.render:
                    previous = reusable.get(beat.beat_id, {})
                    asset = _reuse_native_shot(reuse_project, previous, row, shot_root)
                    if asset is None:
                        asset, commands = _render_native(shot_root, source_path, runtime, request, shot.kind, frames)
                        records.extend(commands)
                        status = "staged"
                    else:
                        status = "reused"
                    name = _stage_media(root, asset, beat.beat_id)
                    scene["media"] = {"src": name, "kind": "video", "fit": "cover", "trim_start_frame": 0}
                    row.update(status=status, asset=asset.relative_to(project_dir).as_posix(), asset_sha256=file_digest(asset))
                else:
                    scene["pending_producer"] = scene["producer"]
        except (ValueError, OSError, RuntimeError) as exc:
            row.update(status="failed", error=str(exc))
            _write_json(status_path, report)
            raise VideoGenerationRuntimeError(f"{beat.beat_id} {scene['producer']} failed: {exc}. See {status_path}") from exc
        scenes.append(scene)
        _write_json(status_path, report)
    timeline = {
        "version": TIMELINE_VERSION, "width": request.width, "height": request.height,
        "fps": fps, "duration_frames": end_frame, "timing_source": graph.source,
        "scenes": scenes,
        "captions": _caption_cues(graph, fps, end_frame, cut_frames=tuple(cuts[1:-1])) if narration else [],
        "audio": {
            "narration": _stage_media(root, narration, "narration"),
            "music": _stage_media(root, music, "music") if request.music_volume > 0 else "",
            "music_volume": request.music_volume,
        },
    }
    _write_json(root / "input_props.json", {"timeline": timeline})
    _write_json(root / "render_request.json", {
        "render_mode": "video", "video_format": request.output_format,
        "video_quality": request.quality, "workers": request.workers,
    })
    return {
        "index_path": str(root / "entry.jsx"), "timeline_path": str(root / "input_props.json"),
        "render_request_path": str(root / "render_request.json"), "film_production_path": str(status_path),
    }, records


def _reuse_native_shot(
    previous_project: Path | None, previous: dict[str, Any], current: dict[str, Any], shot_root: Path,
) -> Path | None:
    digest_key = "native_input_sha256" if "native_input_sha256" in previous else "source_sha256"
    if previous_project is None or previous.get(digest_key) != current[digest_key] or not previous.get("asset"):
        return None
    asset = (previous_project / previous["asset"]).resolve()
    if not asset.is_relative_to(previous_project.resolve()) or not asset.is_file():
        return None
    if file_digest(asset) != previous.get("asset_sha256"):
        return None
    output = shot_root / "shot.mp4"
    shutil.copyfile(asset, output)
    return output


def _stage_native_source(root: Path, payload: dict[str, Any], kind: str) -> Path:
    if kind == "geometry":
        source = root / "scene.py"
        shutil.copyfile(Path(__file__).with_name("film_manim_runtime.py"), source)
    else:
        source = root / "index.html"
        source.write_text(_authored_html(payload) if payload.get("artwork") else _process_html(payload), encoding="utf-8")
    return source


def materialize_pending_shots(
    source: Path, job: Path, timeline: dict[str, Any], *, quality: str, workers: str,
) -> list[CommandRecord]:
    """Compile trusted adapter source from the saved direction on the existing frame clock."""
    if not any(scene.get("pending_producer") for scene in timeline["scenes"]):
        return []
    from video_generation.film_direction import load_film_direction

    direction = load_film_direction(source.parent / "film_direction.json")
    scenes = timeline["scenes"]
    if len(direction.shots) != len(scenes):
        raise ValueError("The saved direction does not match the timeline shot count.")
    pending = []
    for index, (scene, shot) in enumerate(zip(scenes, direction.shots, strict=True), 1):
        if scene.get("pending_producer"):
            if (
                scene["id"] != f"beat_{index:02d}" or scene.get("standfirst") != shot.narration
                or scene["pending_producer"] != shot.producer or shot.producer not in {"manim", "hyperframes"}
            ):
                raise ValueError(f"The saved direction does not match pending shot {scene['id']}.")
            pending.append((index, scene, shot))

    palette = direction.palette if isinstance(direction.palette, dict) else PALETTES[direction.palette]
    request = VideoGenerationRequest(
        prompt="", width=timeline["width"], height=timeline["height"], fps=timeline["fps"],
        quality=quality, workers=workers, renderer="hybrid",
    )
    runtime = HyperframesVideoRuntime()
    records = []
    for index, scene, shot in pending:
        root = job / "shots" / f"beat_{index:02d}"
        root.mkdir(parents=True)
        frames = scene["duration_frames"]
        payload = {
            **asdict(shot), "palette": palette, "duration": frames / request.fps,
            "fps": request.fps, "width": request.width, "height": request.height,
        }
        _write_json(root / "shot.json", payload)
        renderer_source = _stage_native_source(root, payload, shot.kind)
        asset, commands = _render_native(root, renderer_source, runtime, request, shot.kind, frames)
        records.extend(commands)
        name = _stage_media(job, asset, f"native_{index:02d}_{file_digest(asset)[:16]}")
        scene["media"] = {"src": name, "kind": "video", "fit": "cover", "trim_start_frame": 0}
        scene["producer"] = scene.pop("pending_producer")
    return records


def _render_native(
    root: Path, source: Path, runtime: HyperframesVideoRuntime, request: VideoGenerationRequest,
    kind: str, frames: int,
) -> tuple[Path, list[CommandRecord]]:
    native = root / "native.mp4"
    if kind == "geometry":
        record = _run_logged([sys.executable, str(source)], cwd=root, name="manim_shot",
                             log_path=root / "manim.log", timeout=300)
        if record.returncode or not native.is_file():
            raise VideoGenerationRuntimeError(f"Manim could not render this shot. Inspect {record.log_path}.")
        commands = [record]
    else:
        lint = runtime.lint(project_dir=root)
        _, record = runtime.render(
            project_dir=root, output_path=native, fps=30, quality=request.quality,
            output_format="mp4", workers=request.workers,
        )
        commands = [lint, record]
    media = probe_video(str(native))
    duration = frames / request.fps
    if not media["width"] or abs(media["duration_sec"] - duration) > max(.12, 2 / request.fps):
        raise VideoGenerationRuntimeError("Native output does not cover the scheduled shot interval.")
    output = root / "shot.mp4"
    normalize = _run_logged(
        [
            config.FFMPEG_PATH, "-v", "error", "-nostdin", "-y", "-i", str(native),
            "-vf", f"fps={request.fps},scale={request.width}:{request.height},tpad=stop_mode=clone:stop_duration=1",
            "-frames:v", str(frames), "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "15",
            "-pix_fmt", "yuv420p", str(output),
        ],
        cwd=root, name="normalize_shot", log_path=root / "normalize.log", timeout=300,
    )
    if normalize.returncode or not output.is_file():
        raise VideoGenerationRuntimeError(f"Could not conform the shot. Inspect {normalize.log_path}.")
    conformed = probe_video(str(output))
    if (
        (conformed["width"], conformed["height"]) != (request.width, request.height)
        or abs(conformed["fps"] - request.fps) > .001
        or abs(conformed["duration_sec"] - duration) > .5 / request.fps
    ):
        raise VideoGenerationRuntimeError("The conformed shot does not match the scheduled frame interval.")
    return output, [*commands, normalize]


def _authored_html(shot: dict[str, Any]) -> str:
    artwork = shot["artwork"]
    width, height, duration = shot["width"], shot["height"], shot["duration"]
    view = " ".join(map(str, artwork["view"]))
    layers = "".join(f'<g data-artwork-layer="{layer["id"]}">{layer["svg"]}</g>' for layer in artwork["layers"])
    timing = json.dumps([layer["keys"] for layer in artwork["layers"]])
    runtime_path = Path(__file__).resolve().parent.parent / "renderers" / "film_motion.mjs"
    motion = runtime_path.read_text(encoding="utf-8").replace("export function", "function", 1)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"></head>
    <body style="margin:0">
    <div id="root" data-composition-id="root" data-start="0" data-duration="{duration:.8f}"
      data-width="{width}" data-height="{height}" style="width:{width}px;height:{height}px;background:{artwork['background']}">
      <svg width="100%" height="100%" viewBox="0 0 {view}">{layers}</svg>
    </div>
    <script>{motion}
    const duration = {duration:.9f};
    const view = {json.dumps(artwork["view"])};
    const keys = {timing};
    const layers = [...document.querySelectorAll('[data-artwork-layer]')];
    function seek(time) {{
      const progress = Math.max(0, Math.min(1, time / Math.max(duration - 1 / 30, .0001)));
      layers.forEach((element, i) => {{
        const pose = filmLayerPose(keys[i], progress, view);
        element.setAttribute("transform", pose.transform);
        element.setAttribute("opacity", String(pose.opacity));
      }});
      return timeline;
    }}
    const timeline = {{duration: () => duration, time: seek, seek,
      progress: value => seek(value * duration), pause: () => timeline, play: () => timeline}};
    window.__timelines = window.__timelines || {{}};
    window.__timelines["root"] = timeline;
    seek(0);
    </script></body></html>"""


def _process_html(shot: dict[str, Any]) -> str:
    width, height, duration = shot["width"], shot["height"], shot["duration"]
    palette = shot["palette"]
    labels = "".join(f"<li><b>{html.escape(label)}</b></li>" for label in shot["labels"])
    size = min(width, height)
    direction = "column" if width / height < 1 else "row"
    return f"""<!doctype html><html lang="en"><head><meta charset="UTF-8"><style>
    * {{box-sizing:border-box}}
    body {{margin:0}}
    #root {{width:{width}px;height:{height}px;overflow:hidden;background:{palette['background']};
      color:{palette['text']};display:flex;justify-content:center;align-items:center;
      padding:0 8% {height * .12}px;font-family:Arial,sans-serif}}
    ol {{display:flex;flex-direction:{direction};gap:{size * .04}px;width:100%;margin:0;padding:0;list-style:none}}
    li {{flex:1;border-top:{max(2, size * .01)}px solid {palette['accent']};
      padding:{size * .045}px 0;text-wrap:balance;font-size:{size * .06}px;overflow-wrap:anywhere}}
    </style></head><body>
    <div id="root" data-composition-id="root" data-start="0" data-duration="{duration:.6f}"
      data-width="{width}" data-height="{height}"><ol>{labels}</ol></div>
    <script>
    const duration = {duration:.9f};
    const rows = [...document.querySelectorAll('li')];
    function seek(time) {{
      rows.forEach((row, i) => {{
        const progress = Math.max(0, Math.min(1, (time / duration - i * .21) / .23));
        const eased = 1 - (1 - progress) ** 3;
        row.style.opacity = eased;
        row.style.transform = 'translateY(' + ((1 - eased) * {size * .045}) + 'px)';
      }});
      return timeline;
    }}
    const timeline = {{duration: () => duration, time: seek, seek,
      progress: value => seek(value * duration), pause: () => timeline, play: () => timeline}};
    window.__timelines = window.__timelines || {{}};
    window.__timelines["root"] = timeline;
    seek(0);
    </script></body></html>"""
