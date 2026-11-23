"""Build a cached local player from the same composition used for delivery."""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from renderers.remotion_renderer import _find_remotion_node_root
import config
from state import ProjectState
from vex_runtime.hyperframes import resolve_node_executable
from vex_runtime.locking import exclusive_file_lock
from vex_runtime.paths import data_dir
from vex_runtime.production_edits import open_timeline
from vex_visuals.evidence import file_digest, payload_digest
from vex_visuals.fonts import font_runtime_source
from video_generation.production_timeline import audio_assets, cue_id, picture_assets
from video_generation.remotion_revisions import _project_asset
from video_generation.renderer import _run_logged


def build_studio_player() -> Path:
    node = resolve_node_executable()
    if not node:
        raise RuntimeError("Install the local Remotion runtime to enable live playback.")
    # The bundle is keyed to the Node executable, packaged source, and dependency
    # locks. Only probe the runtime on a cache miss.
    code = Path(__file__).resolve().parent.parent / "renderers"
    names = ("studio_player_runner.mjs", "studio_player.jsx", "remotion_entry.jsx",
             "film_motion.mjs", "remotion_scene_graph.jsx", "visual_motion.mjs")
    from vex_runtime.hyperframes import managed_renderer_runtime_dir

    candidates = [code.parent, managed_renderer_runtime_dir(node)]
    locks = {str(root): file_digest(root / "package-lock.json") for root in candidates
             if root is not None and (root / "package-lock.json").is_file()}
    fonts = font_runtime_source()
    fingerprint = payload_digest({"node": node, "locks": locks, "fonts": fonts,
                                  "code": {name: file_digest(code / name) for name in names}})
    root = (data_dir() / "renderers" / "studio-player").resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{fingerprint}.js"
    if target.is_file():
        return target
    with exclusive_file_lock(target.with_suffix(".lock"), timeout_sec=90):
        if target.is_file():
            return target
        runtime, reason = _find_remotion_node_root(node)
        if runtime is None:
            raise RuntimeError(reason)
        descriptor, name = tempfile.mkstemp(prefix="studio-player-", suffix=".js", dir=root)
        os.close(descriptor)
        temporary = Path(name)
        font_file = temporary.with_suffix(".fonts.mjs")
        try:
            font_file.write_text(fonts, encoding="utf-8")
            result = _run_logged([node, str(code / "studio_player_runner.mjs"), str(runtime), str(temporary), str(font_file)],
                                 cwd=runtime, name="studio_player", log_path=root / "build.log", timeout=90)
            if result.returncode or not temporary.stat().st_size:
                raise RuntimeError("Could not build the live preview. Update the Vex Remotion runtime.")
            temporary.replace(target)
        finally:
            font_file.unlink(missing_ok=True)
            temporary.unlink(missing_ok=True)
    return target


def live_media(state: ProjectState, name: str) -> Path:
    root, timeline = open_timeline(state)
    allowed = audio_assets(timeline) | picture_assets(timeline)
    if name not in allowed:
        raise FileNotFoundError("This asset is not in the current cut.")
    return _project_asset(root / "public", name)


def live_timeline(state: ProjectState, patch: dict | None = None) -> dict:
    if patch is None:
        _, timeline = open_timeline(state)
    else:
        from vex_runtime.production_edits import _patch

        operations = patch.get("operations")
        allowed = {"caption", "canvas", "music_gain", "voice_gain", "split", "reorder", "trim", "remove_range",
                   "audio_track_gain", "overlay_remove"}
        if not isinstance(operations, list) or any(
            not isinstance(item, dict) or not isinstance(item.get("op"), str) or item["op"] not in allowed
            for item in operations
        ):
            raise ValueError("This change needs a rendered preview.")
        _, _, timeline = _patch(state, patch)
    return {**timeline, "captions": [{**cue, "id": cue_id(cue)} for cue in timeline["captions"]]}


def waveform(state: ProjectState, name: str) -> Path:
    _, timeline = open_timeline(state)
    if name not in audio_assets(timeline):
        raise FileNotFoundError("This asset is not on an audio track.")
    source = live_media(state, name)
    root = Path(state.working_dir).resolve() / ".waveforms"
    if root.resolve() != root:
        raise ValueError("Waveforms must stay in the project workspace.")
    root.mkdir(exist_ok=True)
    stat = source.stat()
    key = payload_digest([str(source), stat.st_size, stat.st_mtime_ns])
    target = root / f"{key}.png"
    if target.resolve() != target:
        raise ValueError("The waveform escaped its project cache.")
    with exclusive_file_lock(root / f"{key}.lock", timeout_sec=35):
        if target.is_file():
            return target
        descriptor, name = tempfile.mkstemp(prefix="wave-", suffix=".png", dir=root)
        os.close(descriptor)
        temporary = Path(name)
        try:
            result = subprocess.run(
                [config.FFMPEG_PATH, "-nostdin", "-v", "error", "-y", "-i", str(source),
                 "-filter_complex", "[0:a:0]atrim=duration=600,showwavespic=s=460x36:colors=white[out]",
                 "-map", "[out]", "-frames:v", "1", "-threads", "1", str(temporary)],
                capture_output=True, timeout=30, check=False,
            )
            if result.returncode or not temporary.stat().st_size:
                raise FileNotFoundError("The audio waveform could not be prepared.")
            temporary.replace(target)
        except subprocess.TimeoutExpired as exc:
            raise FileNotFoundError("The audio waveform could not be prepared.") from exc
        finally:
            temporary.unlink(missing_ok=True)
    return target
