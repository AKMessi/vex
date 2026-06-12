from __future__ import annotations

import os
from pathlib import Path

from platformdirs import user_data_path

from vex_runtime import __version__


def data_dir() -> Path:
    override = os.getenv("VEX_DATA_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return Path(user_data_path("vex", appauthor="AKMessi", ensure_exists=False))


def hyperframes_runtime_dir() -> Path:
    return data_dir() / "renderers" / "hyperframes" / f"vex-{__version__}"


def local_bin_name(name: str) -> str:
    return f"{name}.cmd" if os.name == "nt" else name


def managed_hyperframes_cli_path() -> Path:
    return hyperframes_runtime_dir() / "node_modules" / ".bin" / local_bin_name("hyperframes")

