from __future__ import annotations

import pytest

from tools import auto_visuals
from vex_runtime import imaging


def test_imaging_runtime_verifies_real_png_roundtrip() -> None:
    status = imaging.imaging_runtime_status()

    assert status["available"] is True
    assert status["pillow_version"]
    assert status["imageio_version"]
    assert status["pillow_module_path"].endswith("Image.py")


def test_imaging_runtime_reports_partial_pillow_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import_module = imaging.importlib.import_module

    def import_module(name: str):
        if name == "PIL.Image":
            raise ImportError("cannot import name 'Image' from 'PIL'")
        return original_import_module(name)

    monkeypatch.setattr(imaging.importlib, "import_module", import_module)

    status = imaging.imaging_runtime_status()

    assert status["available"] is False
    assert "cannot import name 'Image' from 'PIL'" in status["reason"]
    assert "pipx runpip vex-video" in status["repair_command"]
    with pytest.raises(RuntimeError, match="Pillow/ImageIO runtime"):
        imaging.require_imaging_runtime()


def test_auto_visuals_fails_imaging_preflight_before_project_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        auto_visuals,
        "require_imaging_runtime",
        lambda: (_ for _ in ()).throw(RuntimeError("imaging preflight failed")),
    )

    state = object()
    result = auto_visuals.execute({"mode": "generated_only"}, state)

    assert result["success"] is False
    assert result["message"] == "imaging preflight failed"
    assert result["updated_state"] is state


def test_hyperframes_availability_includes_imaging_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from renderers import hyperframes_renderer

    monkeypatch.setattr(
        hyperframes_renderer,
        "imaging_runtime_status",
        lambda: {"available": False, "reason": "PIL.Image missing"},
    )

    status = hyperframes_renderer.HyperframesRenderer().availability()

    assert status.available is False
    assert status.reason == "PIL.Image missing"


def test_manim_availability_includes_imaging_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from renderers import manim_renderer

    monkeypatch.setattr(
        manim_renderer,
        "imaging_runtime_status",
        lambda: {"available": False, "reason": "PIL.Image missing"},
    )

    status = manim_renderer.ManimRenderer().availability()

    assert status.available is False
    assert status.reason == "PIL.Image missing"
