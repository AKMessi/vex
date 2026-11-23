"""Run optional renderer source inside a constrained local Docker container."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from engine import probe_video
from vex_visuals.evidence import file_digest


ENGINES = {"remotion": "scene.jsx", "manim": "scene.py", "hyperframes": "index.html"}
MAX_OUTPUT = 128 * 1024 * 1024
MAX_INPUT = 256 * 1024 * 1024
MAX_SOURCE = 96 * 1024
NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,99}")


def sandbox_image() -> str:
    image = os.getenv("VEX_SANDBOX_IMAGE", "").strip()
    if not re.fullmatch(r"sha256:[a-f0-9]{64}", image):
        raise ValueError("Set VEX_SANDBOX_IMAGE to the ID of a locally built Vex sandbox image.")
    return image


def _docker(command: list[str], *, timeout: int, stdout=None) -> None:
    try:
        result = subprocess.run(command, stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL if stdout is None else stdout,
                                stderr=subprocess.DEVNULL, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Sandbox {command[1]} could not finish within its deadline.") from exc
    if result.returncode:
        raise RuntimeError(f"Sandbox {command[1]} failed with exit status {result.returncode}.")


def check_sandbox() -> str:
    image = sandbox_image()
    docker = shutil.which("docker")
    if not docker:
        raise RuntimeError("Docker Engine is required for native-code film authoring.")
    _docker([docker, "image", "inspect", image], timeout=15)
    return image


def render_source(source: str, output: Path, *, engine: str, width: int, height: int,
                  fps: int, frames: int, assets: tuple[Path, ...] = ()) -> dict:
    if not isinstance(engine, str) or engine not in ENGINES:
        raise ValueError("Choose Remotion, Manim or HyperFrames.")
    if not isinstance(source, str) or not 0 < len(source.encode("utf-8")) <= MAX_SOURCE:
        raise ValueError("Renderer source must contain up to 96 KiB of UTF-8 text.")
    if (type(fps) is not int or not 1 <= fps <= 60 or type(frames) is not int or not 1 <= frames <= 20 * fps
            or any(type(edge) is not int or edge % 2 or not 320 <= edge <= 1920 for edge in (width, height))):
        raise ValueError("Sandbox shots require 1 to 60 fps, even dimensions from 320 to 1920 and up to 20 seconds.")
    image = sandbox_image()
    docker = shutil.which("docker")
    if not docker:
        raise RuntimeError("Docker Engine is required for arbitrary renderer source.")
    output = output.expanduser().absolute()
    if output.suffix.lower() != ".mp4" or output.exists() or output.is_symlink():
        raise ValueError("Choose a new MP4 file for the sandbox output.")
    if len(assets) > 16:
        raise ValueError("Sandbox jobs accept up to 16 local assets.")
    paths = [path.expanduser().resolve(strict=True) for path in assets]
    names = {path.name.casefold() for path in paths}
    if (len(names) != len(paths) or names & {name.casefold() for name in ENGINES.values()}
            or any(not path.is_file() or not NAME.fullmatch(path.name) for path in paths)
            or sum(path.stat().st_size for path in paths) > MAX_INPUT):
        raise ValueError("Sandbox assets need distinct simple filenames and a combined size of at most 256 MiB.")

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".sandbox-", suffix=".mp4", dir=output.parent)
    os.close(descriptor)
    temporary = Path(name)
    container = f"vex-shot-{uuid.uuid4().hex}"
    try:
        with tempfile.TemporaryDirectory(prefix="vex-sandbox-input-") as directory:
            inputs = Path(directory).resolve()
            if any(character in str(inputs) for character in (",", '"')):
                raise ValueError("Choose a temp directory whose path has no comma or quote for Docker.")
            (inputs / ENGINES[engine]).write_text(source, encoding="utf-8")
            for path in paths:
                shutil.copyfile(path, inputs / path.name)
            for path in inputs.iterdir():
                path.chmod(0o444)
            inputs.chmod(0o755)
            mount = f"type=bind,source={inputs},target=/input,readonly"
            try:
                _docker([docker, "run", "-d", "--pull=never", "--name", container,
                         "--network=none", "--ipc=private", "--read-only", "--cap-drop=ALL",
                         "--security-opt=no-new-privileges", "--user=65534:65534", "--pids-limit=128",
                         "--cpus=2", "--memory=2g", "--memory-swap=2g", "--log-driver=none",
                         "--tmpfs=/tmp:rw,nosuid,nodev,size=1280m,mode=1777",
                         "--tmpfs=/out:rw,nosuid,nodev,noexec,size=128m,mode=1777",
                         "--mount", mount, "--entrypoint", "/bin/sleep", image, "infinity"], timeout=30)
                driver = [docker, "exec", container, "python", "-I", "/opt/vex/sandbox/render.py"]
                try:
                    _docker([*driver, engine, str(width), str(height), str(fps), str(frames)], timeout=300)
                except RuntimeError as exc:
                    diagnostic = ""
                    with tempfile.TemporaryFile() as log:
                        try:
                            _docker([*driver, "extract-log"], timeout=10, stdout=log)
                            log.seek(0)
                            diagnostic = log.read(12000).decode("utf-8", errors="replace")
                        except RuntimeError:
                            pass
                    message = "".join(char for char in diagnostic if char.isprintable() or char == "\n").strip()
                    if message:
                        raise RuntimeError(f"{exc}\n{message}") from exc
                    raise
                with temporary.open("wb") as stream:
                    _docker([*driver, "extract"], timeout=30, stdout=stream)
            finally:
                try:
                    _docker([docker, "rm", "-f", container], timeout=15)
                except RuntimeError:
                    pass
        if not 0 < temporary.stat().st_size <= MAX_OUTPUT:
            raise ValueError("The sandbox did not return a bounded MP4.")
        metadata = probe_video(str(temporary), input_format="mov")
        if (metadata.get("width"), metadata.get("height"), metadata.get("video_frame_count")) != (width, height, frames):
            raise ValueError("The sandbox output did not match the requested frame clock.")
        if abs(metadata["fps"] - fps) > .001 or metadata.get("has_audio") or metadata.get("codec") != "h264":
            raise ValueError("The sandbox returned an unsupported media stream.")
        digest = file_digest(temporary)
        # Publish without replacing a file created by another process.
        os.link(temporary, output)
        return {"output": str(output), "sha256": digest, "engine": engine,
                "image": image, "duration_frames": frames, "fps": fps}
    finally:
        temporary.unlink(missing_ok=True)
