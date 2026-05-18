from __future__ import annotations

import tomllib
from pathlib import Path


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
