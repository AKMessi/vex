from __future__ import annotations

import tomllib
from pathlib import Path

import config
from vex_runtime import __version__


def test_pyproject_lists_all_root_packages() -> None:
    root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    configured_packages = set(pyproject["tool"]["setuptools"]["packages"])
    root_packages = {
        path.name
        for path in root.iterdir()
        if path.is_dir() and (path / "__init__.py").is_file()
    }

    assert root_packages <= configured_packages


def test_whisper_is_optional_not_default_dependency() -> None:
    root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))

    assert "openai-whisper>=20231117" not in pyproject["project"]["dependencies"]
    assert "openai-whisper>=20231117" in pyproject["project"]["optional-dependencies"]["transcription"]


def test_distribution_identity_and_version_have_single_authority() -> None:
    root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["name"] == "vex-video"
    assert "version" not in pyproject["project"]
    assert pyproject["project"]["dynamic"] == ["version"]
    assert pyproject["tool"]["setuptools"]["dynamic"]["version"] == {
        "attr": "vex_runtime.__version__"
    }
    assert config.VERSION == __version__ == "0.1.0rc1"


def test_manim_is_optional_not_default_dependency() -> None:
    root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))

    assert "manim>=0.20.0" not in pyproject["project"]["dependencies"]
    assert "manim>=0.20.0" in pyproject["project"]["optional-dependencies"]["manim"]
    assert "manim>=0.20.0" in pyproject["project"]["optional-dependencies"]["all"]
