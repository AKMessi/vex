from __future__ import annotations

import importlib
import importlib.metadata
import io
import sys
from typing import Any


PILLOW_DISTRIBUTION = "pillow"
PILLOW_REQUIREMENT = "pillow>=10.0.0"
IMAGEIO_DISTRIBUTION = "imageio"
IMAGEIO_REQUIREMENT = "imageio>=2.9.0"
PIPX_REPAIR_COMMAND = (
    "python -m pipx runpip vex-video install --force-reinstall --no-cache-dir "
    f'"{PILLOW_REQUIREMENT}" "{IMAGEIO_REQUIREMENT}"'
)


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _verify_image_roundtrip() -> dict[str, str]:
    image_module = importlib.import_module("PIL.Image")
    imageio_module = importlib.import_module("imageio.v3")

    image = image_module.new("RGB", (1, 1), color=(17, 29, 43))
    encoded = io.BytesIO()
    image.save(encoded, format="PNG")
    decoded = imageio_module.imread(encoded.getvalue(), extension=".png")
    if tuple(decoded.shape[:2]) != (1, 1):
        raise RuntimeError(
            f"ImageIO decoded an unexpected image shape: {decoded.shape!r}"
        )
    return {
        "pillow_module_path": str(getattr(image_module, "__file__", "") or ""),
        "imageio_module_path": str(
            getattr(imageio_module, "__file__", "") or ""
        ),
    }


def imaging_runtime_status() -> dict[str, Any]:
    versions = {
        "pillow_version": _distribution_version(PILLOW_DISTRIBUTION),
        "imageio_version": _distribution_version(IMAGEIO_DISTRIBUTION),
    }
    try:
        module_paths = _verify_image_roundtrip()
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        return {
            "available": False,
            "reason": (
                "The Pillow/ImageIO runtime is missing or incomplete in "
                f"{sys.executable}. Import and PNG round-trip verification failed "
                f"with {detail}. Close running Vex processes, then run "
                f"`{PIPX_REPAIR_COMMAND}`."
            ),
            "error": detail,
            "python_executable": sys.executable,
            "repair_command": PIPX_REPAIR_COMMAND,
            **versions,
        }
    return {
        "available": True,
        "reason": "",
        "error": "",
        "python_executable": sys.executable,
        "repair_command": PIPX_REPAIR_COMMAND,
        **versions,
        **module_paths,
    }


def require_imaging_runtime() -> dict[str, Any]:
    status = imaging_runtime_status()
    if not status["available"]:
        raise RuntimeError(str(status["reason"]))
    return status
