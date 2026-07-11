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

    module.validate_tag(f"v{module.__version__}")
    with pytest.raises(module.ReleaseValidationError, match="mismatch"):
        module.validate_tag("v999.0.0")


def test_release_artifacts_require_long_form_visual_runtime() -> None:
    module = _release_checks()

    assert {
        "shorts/story_compiler.py",
        "tools/auto_shorts.py",
        "visual_opportunity.py",
        "visual_program.py",
        "visual_skill_graph.py",
        "tools/video_generation.py",
        "video_generation/beat_tournament.py",
        "video_generation/director.py",
        "video_generation/pipeline.py",
        "video_generation/portfolio_judge.py",
        "video_generation/renderer.py",
        "video_generation/hyperframes_project.py",
        "video_generation/skill_graph.py",
        "tools/auto_visuals.py",
        "tools/song.py",
        "tools/song_director.py",
        "asset_registry.py",
        "content_cache.py",
        "evaluation_harness.py",
        "job_runner.py",
        "nle_interop.py",
        "plan_store.py",
        "plugin_api.py",
        "timeline.py",
        "tools/creative_optimizer.py",
        "vex_hyperframes/visual_world.py",
        "vex_hyperframes/visual_world_renderer.py",
        "vex_hyperframes/qa.py",
        "vex_runtime/imaging.py",
        "renderers/remotion_renderer.py",
        "renderers/remotion_entry.jsx",
        "renderers/remotion_runner.mjs",
        "vex_remotion/compiler.py",
        "vex_remotion/qa.py",
    } <= module.REQUIRED_WHEEL_FILES
    assert {"imageio", "kokoro-onnx", "pillow", "soundfile"} <= module.REQUIRED_RUNTIME_DEPENDENCIES


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
