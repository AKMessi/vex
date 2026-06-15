from __future__ import annotations

import json
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


def test_distribution_includes_long_form_visual_planner() -> None:
    root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    modules = set(pyproject["tool"]["setuptools"]["py-modules"])

    assert {"visual_opportunity", "visual_program"} <= modules
    assert (root / "tools" / "auto_visuals.py").is_file()


def test_distribution_includes_shorts_story_compiler() -> None:
    root = Path(__file__).resolve().parents[1]

    assert (root / "shorts" / "story_compiler.py").is_file()
    assert (root / "tools" / "auto_shorts.py").is_file()


def test_imaging_stack_is_a_direct_runtime_dependency() -> None:
    root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["dependencies"]

    assert "imageio>=2.9.0" in dependencies
    assert "pillow>=10.0.0" in dependencies
    assert (root / "vex_runtime" / "imaging.py").is_file()


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
    assert (
        pyproject["project"]["license"]
        == "LicenseRef-PolyForm-Noncommercial-1.0.0"
    )
    assert config.VERSION == __version__


def test_manim_is_optional_not_default_dependency() -> None:
    root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))

    assert "manim>=0.20.0" not in pyproject["project"]["dependencies"]
    assert "manim>=0.20.0" in pyproject["project"]["optional-dependencies"]["manim"]
    assert "manim>=0.20.0" in pyproject["project"]["optional-dependencies"]["all"]


def test_runtime_path_resolution_has_no_third_party_dependency() -> None:
    root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    runtime_paths = (root / "vex_runtime" / "paths.py").read_text(encoding="utf-8")

    assert not any(
        dependency.startswith("platformdirs")
        for dependency in pyproject["project"]["dependencies"]
    )
    assert "platformdirs" not in runtime_paths


def test_packaged_runtime_resources_match_repository_authorities() -> None:
    root = Path(__file__).resolve().parents[1]
    resources = root / "vex_runtime" / "resources"

    assert json.loads(
        (resources / "hyperframes" / "package.json").read_bytes()
    ) == json.loads((root / "package.json").read_bytes())
    assert json.loads(
        (resources / "hyperframes" / "package-lock.json").read_bytes()
    ) == json.loads((root / "package-lock.json").read_bytes())
    assert (resources / "config" / ".env.example").read_text(
        encoding="utf-8"
    ).splitlines() == (root / ".env.example").read_text(
        encoding="utf-8"
    ).splitlines()
