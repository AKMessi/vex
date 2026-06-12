from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
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


def test_cli_does_not_infer_release_tag_from_github_branch_environment(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["GITHUB_REF_NAME"] = "main"

    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "release_checks.py"),
            "--dist-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert "tag/version mismatch" not in result.stderr
    assert "Expected exactly one vex-video wheel" in result.stderr
