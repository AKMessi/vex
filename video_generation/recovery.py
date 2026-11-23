"""Keep file-producing production checkpoints independent of an attempt folder."""
from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path, PurePosixPath

from vex_visuals.evidence import file_digest


class _UncacheableStage(Exception):
    def __init__(self, result: dict):
        self.result = result


def _relative(root: Path, path: Path) -> str:
    path = path.resolve(strict=True)
    if not path.is_file() or not path.is_relative_to(root):
        raise ValueError("A production checkpoint output escaped its attempt folder.")
    return path.relative_to(root).as_posix()


def _cache_file(root: Path, source: Path) -> Path:
    identity = file_digest(source)
    target = root / identity
    if target.resolve() != target:
        raise ValueError("Production checkpoint media must stay in its cache.")
    if target.is_file() and file_digest(target) == identity:
        return target
    descriptor, name = tempfile.mkstemp(prefix=".stage-", dir=root)
    os.close(descriptor)
    temporary = Path(name)
    try:
        shutil.copyfile(source, temporary)
        if file_digest(temporary) != identity:
            raise ValueError("A completed production stage changed while it was being saved.")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def retain_media(checkpoint, source: Path) -> Path:
    cache = checkpoint.root / "production-media"
    if cache.resolve() != cache:
        raise ValueError("Production checkpoint media must stay in its cache.")
    cache.mkdir(exist_ok=True)
    return _cache_file(cache, source)


def stage_files(checkpoint, name: str, inputs: dict, root: Path,
                work: Callable[[], tuple[dict, Iterable[Path]]], *,
                reusable: Callable[[dict], bool] = lambda _: True) -> tuple[dict, bool]:
    if checkpoint is None:
        result, _ = work()
        return result, False
    root = root.resolve(strict=True)
    cache = checkpoint.root / "production-media"

    def save():
        result, outputs = work()
        if not reusable(result):
            raise _UncacheableStage(result)
        files = {_relative(root, path): retain_media(checkpoint, path).name for path in outputs}
        return {"result": result, "files": files}

    try:
        snapshot, reused = checkpoint.stage(name, inputs, save,
                                           output_paths=lambda value: [str(cache / token) for token in set(value["files"].values())])
    except _UncacheableStage as exc:
        return exc.result, False
    if reused:
        for relative, token in snapshot["files"].items():
            path = PurePosixPath(relative)
            target, source = root.joinpath(*path.parts), cache / token
            if (path.is_absolute() or ".." in path.parts or target.resolve() != target
                    or source.resolve().parent != cache):
                raise ValueError("A restored production stage would escape its workspace.")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    return snapshot["result"], reused


def existing_paths(root: Path, names: Iterable[str]) -> list[Path]:
    return [root / name for name in names if (root / name).is_file()]


def rebind_command(value: dict, directory: Path):
    from video_generation.renderer import CommandRecord

    log = Path(value["log_path"])
    local = directory / log.name
    return CommandRecord(**{**value, "cwd": str(directory), "log_path": str(local if local.is_file() else log)})
