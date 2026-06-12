from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _release_checks():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "scripts" / "release_checks.py"
    spec = importlib.util.spec_from_file_location("release_checks", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_tag_must_match_canonical_version() -> None:
    module = _release_checks()

    module.validate_tag("v0.1.0rc1")
    with pytest.raises(module.ReleaseValidationError, match="mismatch"):
        module.validate_tag("v0.1.0")


def test_prerelease_detection_is_pep440_compatible_for_supported_tags() -> None:
    module = _release_checks()

    assert module.is_prerelease("0.1.0rc1") is True
    assert module.is_prerelease("0.1.0b2") is True
    assert module.is_prerelease("0.1.0") is False
