from __future__ import annotations

from pathlib import Path

from renderers import resolve_renderer
from renderers.base import RendererStatus
from tools.auto_visuals import _max_render_workers
from vex_hyperframes import build_composition, retrieve_skill_slices, validate_composition_html
from vex_hyperframes.qa import HyperframesQualityReport
from vex_hyperframes.variants import build_variants


def test_hyperframes_composition_is_self_contained_and_valid() -> None:
    composition = build_composition(_spec(), width=1920, height=1080, fps=30)
    report = validate_composition_html(
        composition.html,
        expected_width=1920,
        expected_height=1080,
        expected_duration=3.0,
    )

    assert report.valid, report.errors
    assert report.composition_id == "vex-visual_001"
    assert report.clip_count >= 4
    assert "window.__timelines[\"vex-visual_001\"]" in composition.html
    assert "https://" not in composition.html
    assert "requestAnimationFrame" not in composition.html


def test_hyperframes_renderer_scores_premium_html_slides_above_manim() -> None:
    renderer, reason = resolve_renderer(
        _spec(),
        preferred="auto",
        allow_unavailable=True,
    )

    assert renderer.name == "hyperframes"
    assert "hyperframes scored" in reason


def test_hyperframes_skill_pack_includes_production_contract() -> None:
    skills = retrieve_skill_slices("signal_network")
    skill_ids = {skill.skill_id for skill in skills}

    assert "hyperframes-production-contract" in skill_ids
    assert any("window.__timelines" in " ".join(skill.rules) for skill in skills)


def test_auto_visuals_serializes_hyperframes_renders_to_avoid_npx_cache_contention() -> None:
    assert _max_render_workers({"renderer": "auto"}, 4, [_spec()]) == 1
    assert _max_render_workers({"renderer": "hyperframes"}, 4, [_spec()]) == 1
    assert _max_render_workers({"renderer": "manim", "max_render_workers": 3}, 4, [_spec()]) == 3


def test_hyperframes_command_uses_local_cli_without_runtime_install(monkeypatch, tmp_path: Path) -> None:
    import renderers.hyperframes_renderer as module

    cli_path = tmp_path / "node_modules" / ".bin" / module._local_bin_name("hyperframes")
    cli_path.parent.mkdir(parents=True)
    cli_path.write_text("cli", encoding="utf-8")
    monkeypatch.setattr(module.config, "HYPERFRAMES_CLI_PATH", str(cli_path))

    command = module._hyperframes_command("lint")

    assert command[0] == str(cli_path)
    assert command[-1] == "lint"
    assert "npx" not in command
    assert "--yes" not in command


def test_hyperframes_command_does_not_fall_back_to_global_path(monkeypatch, tmp_path: Path) -> None:
    import renderers.hyperframes_renderer as module

    fake_renderer = tmp_path / "repo" / "renderers" / "hyperframes_renderer.py"
    fake_renderer.parent.mkdir(parents=True)
    fake_renderer.write_text("# test", encoding="utf-8")
    fake_cwd = tmp_path / "cwd"
    fake_cwd.mkdir()
    monkeypatch.chdir(fake_cwd)
    monkeypatch.setattr(module, "__file__", str(fake_renderer))
    monkeypatch.setattr(module.config, "HYPERFRAMES_CLI_PATH", "hyperframes")
    monkeypatch.setattr(module.shutil, "which", lambda _name: str(tmp_path / "global" / "hyperframes"))

    assert module._hyperframes_cli_path() is None


def test_hyperframes_render_promotes_root_metadata_path(monkeypatch, tmp_path: Path) -> None:
    import renderers.hyperframes_renderer as module

    renderer = module.HyperframesRenderer()
    monkeypatch.setattr(renderer, "availability", lambda: RendererStatus(True, "ok"))
    monkeypatch.setattr(module.config, "HYPERFRAMES_VARIANT_COUNT", 1)
    monkeypatch.setattr(
        module,
        "probe_video",
        lambda _: {"width": 1920, "height": 1080, "duration_sec": 3.0},
    )

    def fake_render_variant(variant, *, job_dir, width, height, fps):
        variant_dir = job_dir / "variants" / variant.variant_id
        variant_dir.mkdir(parents=True, exist_ok=True)
        asset_path = variant_dir / "variant.mp4"
        asset_path.write_bytes(b"fake")
        index_path = variant_dir / "index.html"
        metadata_path = variant_dir / "hyperframes_metadata.json"
        index_path.write_text("<html></html>", encoding="utf-8")
        metadata_path.write_text("{}", encoding="utf-8")
        return {
            "variant_id": variant.variant_id,
            "variant_index": variant.variant_index,
            "asset_path": str(asset_path),
            "script_path": str(index_path),
            "artifact_paths": {"metadata_path": str(metadata_path)},
            "metadata": {"duration_sec": 3.0},
            "qa": {"score": 0.91, "passed": True},
        }

    monkeypatch.setattr(renderer, "_render_variant", fake_render_variant)

    asset = renderer.render(_spec(), tmp_path, width=1920, height=1080, fps=30)

    assert asset.artifact_paths["metadata_path"] == str(tmp_path / "visual_001" / "hyperframes_metadata.json")
    assert Path(asset.artifact_paths["variant_metadata_path"]).parts[-3:] == (
        "variants",
        "variant_01",
        "hyperframes_metadata.json",
    )


def test_hyperframes_variant_cli_runs_inside_variant_workspace(monkeypatch, tmp_path: Path) -> None:
    import renderers.hyperframes_renderer as module

    calls = []

    class FakeResult:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(command, *, cwd, capture_output, text, timeout):
        calls.append((list(command), cwd))
        if "render" in command:
            output_path = Path(command[command.index("--output") + 1])
            output_path.write_bytes(b"fake")
        return FakeResult()

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module, "_hyperframes_cli_path", lambda: "hyperframes")
    monkeypatch.setattr(module.config, "HYPERFRAMES_RENDER_QUALITY", "")
    monkeypatch.setattr(
        module,
        "probe_video",
        lambda _: {"width": 1920, "height": 1080, "duration_sec": 3.0},
    )
    monkeypatch.setattr(module, "extract_quality_frames", lambda *_, **__: [])
    monkeypatch.setattr(
        module,
        "analyze_hyperframes_quality",
        lambda **_: HyperframesQualityReport(passed=True, score=0.92),
    )

    variant = build_variants(_spec(), default_count=1)[0]
    module.HyperframesRenderer()._render_variant(
        variant,
        job_dir=tmp_path / "job",
        width=1920,
        height=1080,
        fps=30,
    )

    assert calls[0][0][-1] == "."
    assert calls[1][0][-1] == "."
    assert calls[0][1] == calls[1][1]


def _spec() -> dict:
    return {
        "visual_id": "visual_001",
        "template": "signal_network",
        "duration": 3.0,
        "headline": "Build First",
        "deck": "The learning loop gets shorter",
        "steps": ["Build", "Get stuck", "Target study", "Ship"],
        "visual_type_hint": "process",
        "composition_mode": "replace",
        "style_pack": "signal_lab",
        "importance": 0.9,
    }
