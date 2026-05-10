from __future__ import annotations

from renderers import resolve_renderer
from tools.auto_visuals import _max_render_workers
from vex_hyperframes import build_composition, retrieve_skill_slices, validate_composition_html


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


def test_hyperframes_windows_command_uses_node_for_npx_cmd(monkeypatch) -> None:
    import renderers.hyperframes_renderer as module

    def fake_which(name: str) -> str | None:
        if name == "npx":
            return r"C:\Program Files\nodejs\npx.CMD"
        if name == "node":
            return r"C:\Program Files\nodejs\node.exe"
        return None

    class FakeCliPath:
        def __init__(self, value: str) -> None:
            self.value = value

        def __truediv__(self, other: str) -> "FakeCliPath":
            return FakeCliPath(f"{self.value}\\{other}")

        def is_file(self) -> bool:
            return True

        def __str__(self) -> str:
            return self.value

    class FakeNpxPath:
        name = "npx.CMD"
        parent = FakeCliPath(r"C:\Program Files\nodejs")

    monkeypatch.setattr(module.shutil, "which", fake_which)
    monkeypatch.setattr(module, "Path", lambda value: FakeNpxPath())

    command = module._npx_command("lint")

    assert command[:3] == [
        r"C:\Program Files\nodejs\node.exe",
        "C:\\Program Files\\nodejs\\node_modules\\npm\\bin\\npx-cli.js",
        "--yes",
    ]
    assert command[-1] == "lint"


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
