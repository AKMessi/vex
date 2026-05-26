from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import config
from renderers import resolve_renderer
from renderers.base import RendererStatus, VisualRendererError
from renderers.blender_renderer import BlenderRenderer, _blender_command
from renderers.blender_spec import BlenderVisualSpec, SUPPORTED_BLENDER_TEMPLATES
from tools.auto_visuals import _normalize_manual_blender_specs


def test_blender_spec_supports_all_expected_templates(tmp_path: Path) -> None:
    expected = {
        "quote_focus",
        "keyword_stack",
        "metric_callout",
        "three_d_title",
        "floating_3d_label",
        "object_orbit",
        "logo_reveal",
        "screen_pointer_3d",
        "data_tunnel",
        "product_model_spin",
    }

    assert expected.issubset(SUPPORTED_BLENDER_TEMPLATES)
    assert expected.issubset(BlenderRenderer.supported_templates)

    spec = BlenderVisualSpec.from_raw(
        {
            "template": "floating_3d_label",
            "composition_mode": "overlay",
            "duration": 99,
            "text": "x" * 200,
            "accent_color": "blue",
        },
        render_root=tmp_path / "renders",
        width=1920,
        height=1080,
        fps=30,
    )

    assert spec.composition_mode == "overlay"
    assert spec.alpha is True
    assert spec.duration == 12.0
    assert len(spec.text) <= 120
    assert spec.accent_color == "#38BDF8"


def test_blender_spec_rejects_unsupported_template(tmp_path: Path) -> None:
    with pytest.raises(VisualRendererError, match="Unsupported Blender template"):
        BlenderVisualSpec.from_raw(
            {"template": "raw_python_scene"},
            render_root=tmp_path / "renders",
            width=1280,
            height=720,
            fps=30,
        )


def test_blender_spec_rejects_unsafe_asset_paths(tmp_path: Path) -> None:
    outside = tmp_path / "outside" / "model.glb"
    outside.parent.mkdir()
    outside.write_bytes(b"glb")

    with pytest.raises(VisualRendererError, match="must stay inside"):
        BlenderVisualSpec.from_raw(
            {
                "template": "product_model_spin",
                "asset_path": str(outside),
                "allowed_asset_roots": [str(tmp_path / "allowed")],
            },
            render_root=tmp_path / "allowed" / "project" / "renders",
            width=1280,
            height=720,
            fps=30,
        )


def test_blender_renderer_scores_3d_specs_and_resolves_when_allowed() -> None:
    renderer = BlenderRenderer()

    assert renderer.score_spec({"template": "three_d_title", "composition_mode": "replace"}) > renderer.score_spec(
        {"template": "quote_focus", "composition_mode": "replace"}
    )
    selected, reason = resolve_renderer(
        {"template": "three_d_title", "composition_mode": "replace"},
        allow_unavailable=True,
    )

    assert selected.name == "blender"
    assert "blender scored" in reason


def test_blender_command_uses_configured_path_without_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "BLENDER_PATH", "/tmp/custom blender", raising=False)

    assert _blender_command(Path("scene.py")) == ["/tmp/custom blender", "-b", "-P", "scene.py"]


def test_manual_blender_specs_lock_renderer_even_if_params_request_other_renderer() -> None:
    specs = _normalize_manual_blender_specs(
        [
            {
                "template": "three_d_title",
                "renderer_hint": "manim",
                "composition_mode": "replace",
                "headline": "Matrix",
                "start": 1,
                "end": 4,
            }
        ],
        clip_duration=10.0,
        width=1280,
        height=720,
        fps=30,
        force_fullscreen=False,
    )

    assert specs[0]["renderer_hint"] == "blender"
    assert specs[0]["require_generated_scene"] is True


def test_blender_overlay_render_metadata_without_requiring_blender(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    monkeypatch.setattr(config, "BLENDER_PATH", "blender-test", raising=False)
    monkeypatch.setattr(config, "FFMPEG_PATH", "ffmpeg-test", raising=False)
    monkeypatch.setattr(
        BlenderRenderer,
        "availability",
        lambda self: RendererStatus(True, ""),
    )

    def fake_run(command: list[str], capture_output: bool, text: bool) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[0] == "blender-test":
            script_path = Path(command[-1])
            frame_dir = script_path.parent / "frames"
            frame_dir.mkdir(exist_ok=True)
            (frame_dir / "frame_0001.png").write_bytes(b"png")
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[0] == "ffmpeg-test":
            Path(command[-1]).write_bytes(b"mov")
            return subprocess.CompletedProcess(command, 0, "", "")
        raise AssertionError(command)

    monkeypatch.setattr("renderers.blender_renderer.subprocess.run", fake_run)
    monkeypatch.setattr(
        "renderers.blender_renderer.probe_video",
        lambda path: {"width": 1280, "height": 720, "duration_sec": 3.0, "fps": 30.0},
    )

    asset = BlenderRenderer().render(
        {
            "visual_id": "label",
            "template": "floating_3d_label",
            "composition_mode": "overlay",
            "label": "Focus here",
        },
        render_root=tmp_path,
        width=1280,
        height=720,
        fps=30,
    )

    assert asset.asset_path.endswith(".mov")
    assert asset.metadata["renderer"] == "blender"
    assert asset.metadata["template"] == "floating_3d_label"
    assert asset.metadata["has_alpha"] is True
    assert asset.metadata["composition_mode"] == "overlay"
    assert Path(asset.metadata["script_path"]).is_file()
    assert calls[0][:3] == ["blender-test", "-b", "-P"]
    assert calls[1][0] == "ffmpeg-test"


def test_blender_replace_render_encodes_png_frames_without_requiring_blender(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    monkeypatch.setattr(config, "BLENDER_PATH", "blender-test", raising=False)
    monkeypatch.setattr(config, "FFMPEG_PATH", "ffmpeg-test", raising=False)
    monkeypatch.setattr(
        BlenderRenderer,
        "availability",
        lambda self: RendererStatus(True, ""),
    )

    def fake_run(command: list[str], capture_output: bool, text: bool) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[0] == "blender-test":
            script_path = Path(command[-1])
            frame_dir = script_path.parent / "frames"
            frame_dir.mkdir(exist_ok=True)
            (frame_dir / "frame_0001.png").write_bytes(b"png")
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[0] == "ffmpeg-test":
            Path(command[-1]).write_bytes(b"mp4")
            return subprocess.CompletedProcess(command, 0, "", "")
        raise AssertionError(command)

    monkeypatch.setattr("renderers.blender_renderer.subprocess.run", fake_run)
    monkeypatch.setattr(
        "renderers.blender_renderer.probe_video",
        lambda path: {"width": 640, "height": 360, "duration_sec": 1.0, "fps": 24.0},
    )

    asset = BlenderRenderer().render(
        {
            "visual_id": "title",
            "template": "three_d_title",
            "composition_mode": "replace",
            "headline": "Vex 3D",
            "duration": 1.0,
        },
        render_root=tmp_path,
        width=640,
        height=360,
        fps=24,
    )

    assert asset.asset_path.endswith(".mp4")
    assert asset.metadata["has_alpha"] is False
    assert "-c:v" in calls[1]
    assert "libx264" in calls[1]
