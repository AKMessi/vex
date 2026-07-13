from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path

from vex_runtime import __version__


def _default_data_dir(
    *,
    platform: str | None = None,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> Path:
    platform = platform or sys.platform
    environ = os.environ if environ is None else environ
    home = Path.home() if home is None else home

    if platform == "win32":
        local_app_data = environ.get("LOCALAPPDATA", "").strip()
        base = Path(local_app_data) if local_app_data else home / "AppData" / "Local"
        return base / "AKMessi" / "vex"
    if platform == "darwin":
        return home / "Library" / "Application Support" / "vex"

    xdg_data_home = environ.get("XDG_DATA_HOME", "").strip()
    base = (
        Path(xdg_data_home).expanduser()
        if xdg_data_home
        else home / ".local" / "share"
    )
    return base / "vex"


def data_dir() -> Path:
    override = os.getenv("VEX_DATA_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return _default_data_dir()


def hyperframes_runtime_base_dir() -> Path:
    return data_dir() / "renderers" / "hyperframes" / f"vex-{__version__}"


def hyperframes_runtime_dir(runtime_key: str | None = None) -> Path:
    base = hyperframes_runtime_base_dir()
    if runtime_key is None:
        return base
    normalized = str(runtime_key).strip().lower()
    if not normalized or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789._-"
        for character in normalized
    ):
        raise ValueError(f"Invalid managed renderer runtime key: {runtime_key!r}")
    return base / normalized


def local_bin_name(name: str) -> str:
    return f"{name}.cmd" if os.name == "nt" else name


def managed_hyperframes_cli_path(runtime_key: str | None = None) -> Path:
    return (
        hyperframes_runtime_dir(runtime_key)
        / "node_modules"
        / ".bin"
        / local_bin_name("hyperframes")
    )
