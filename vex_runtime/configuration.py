from __future__ import annotations

import os
from importlib.resources import files
from pathlib import Path


class ConfigurationError(RuntimeError):
    pass


def write_config_template(destination: Path, *, force: bool = False) -> Path:
    destination = destination.expanduser().resolve()
    if destination.exists() and not force:
        raise ConfigurationError(
            f"Configuration already exists at {destination}. Use --force to replace it."
        )
    if destination.exists() and not destination.is_file():
        raise ConfigurationError(f"Configuration destination is not a file: {destination}")

    template = files("vex_runtime").joinpath("resources", "config", ".env.example")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        temporary.write_bytes(template.read_bytes())
        if os.name != "nt":
            temporary.chmod(0o600)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination

