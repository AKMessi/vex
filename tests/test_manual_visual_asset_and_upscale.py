from __future__ import annotations

from pathlib import Path

from state import ProjectState, utc_now_iso
from tools import renderer_diagnostics, upscale, visual_asset


def test_manual_video_asset_inserts_exact_timing(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    state = _state(tmp_path)
    asset = tmp_path / "overlay.mp4"
    asset.write_bytes(b"asset")
    output = tmp_path / "manual_out.mp4"
    output.write_bytes(b"out")
    overlays: list[dict] = []

    monkeypatch.setattr(visual_asset, "probe_video", lambda _path: _metadata())
    monkeypatch.setattr(
        visual_asset,
        "apply_visual_overlays",
        lambda _input, _working, items: overlays.extend(items) or str(output),
    )

    result = visual_asset.execute(
        {
            "asset_path": str(asset),
            "start": "12.5",
            "end": "18",
            "composition_mode": "replace",
        },
        state,
    )

    assert result["success"] is True
    assert overlays[0]["start"] == 12.5
    assert overlays[0]["end"] == 18.0
    assert overlays[0]["asset_path"] == str(asset)
    assert overlays[0]["compose_mode"] == "replace"
    assert state.timeline[-1]["op"] == "add_visual_asset"


def test_manual_visual_asset_rejects_unsafe_external_path(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    state = _state(tmp_path / "project")
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"outside")
    monkeypatch.setattr(visual_asset, "apply_visual_overlays", lambda *_args, **_kwargs: "should-not-run")

    result = visual_asset.execute(
        {
            "asset_path": str(outside),
            "start": "1",
            "end": "2",
            "composition_mode": "replace",
        },
        state,
    )

    assert result["success"] is False
    assert "must stay inside" in result["message"]


def test_manual_html_asset_rejects_remote_urls(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    state = _state(tmp_path)
    html = tmp_path / "visual.html"
    html.write_text('<script src="https://example.com/app.js"></script>', encoding="utf-8")
    monkeypatch.setattr(visual_asset, "probe_video", lambda _path: _metadata())

    result = visual_asset.execute(
        {
            "asset_path": str(html),
            "start": "1",
            "end": "3",
            "composition_mode": "replace",
        },
        state,
    )

    assert result["success"] is False
    assert "remote URLs" in result["message"]


def test_upscale_video_uses_ffmpeg_scale_export(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    state = _state(tmp_path)
    exports: list[dict] = []

    monkeypatch.setattr(upscale, "estimate_output_size", lambda *_args: 1024)
    monkeypatch.setattr(upscale, "check_disk_space", lambda *_args: True)
    monkeypatch.setattr(
        upscale,
        "export",
        lambda _src, dst, preset: exports.append(dict(preset)) or dst,
    )

    result = upscale.execute(
        {"resolution": "1920x1080", "scale_mode": "fill"},
        state,
    )

    assert result["success"] is True
    assert exports[0]["resolution"] == "1920x1080"
    assert exports[0]["scale_mode"] == "fill"
    assert "not AI super-resolution" in result["message"]


def test_renderer_doctor_reports_dependency_status(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(renderer_diagnostics, "_hyperframes_cli_path", lambda: "/repo/node_modules/.bin/hyperframes")
    monkeypatch.setattr(renderer_diagnostics, "_node_major_version", lambda: 22)
    monkeypatch.setattr(renderer_diagnostics, "resolve_node_executable", lambda: "/bin/node")
    monkeypatch.setattr(renderer_diagnostics, "_node_platform_arch", lambda: ("win32", "x64", ""))
    monkeypatch.setattr(renderer_diagnostics, "_remotion_platform_blocker", lambda platform, arch: "")
    monkeypatch.setattr(renderer_diagnostics.shutil, "which", lambda name: f"/bin/{name}")
    monkeypatch.setattr(
        renderer_diagnostics,
        "imaging_runtime_status",
        lambda: {
            "available": True,
            "reason": "",
            "pillow_version": "12.2.0",
            "imageio_version": "2.37.3",
        },
    )
    monkeypatch.setattr(
        renderer_diagnostics,
        "_version",
        lambda command: {"available": True, "version": f"{command[0]} version"},
    )
    monkeypatch.setattr(renderer_diagnostics, "_run_node_package_probe", lambda: (True, ""))
    monkeypatch.setattr(
        renderer_diagnostics,
        "renderer_native_runtime_status",
        lambda **_: {"available": True, "reason": "", "versions": {"sharp": "0.34.5"}},
    )
    monkeypatch.setattr(renderer_diagnostics, "renderer_capabilities", lambda: [{"name": "hyperframes", "available": True}])

    report = renderer_diagnostics.renderer_doctor_report()

    assert report["hyperframes"]["available"] is True
    assert report["hyperframes"]["source"] == "configured_or_repository"
    assert report["hyperframes"]["reason"] == ""
    assert report["imaging"]["available"] is True
    assert report["imaging"]["pillow_version"] == "12.2.0"
    assert report["remotion"]["available"] is True
    assert report["ffmpeg"]["path"] == f"/bin/{renderer_diagnostics.config.FFMPEG_PATH}"
    assert report["renderer_capabilities"][0]["name"] == "hyperframes"


def _state(tmp_path: Path) -> ProjectState:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    now = utc_now_iso()
    return ProjectState(
        project_id="manual-asset-test",
        project_name="Manual Asset Test",
        created_at=now,
        updated_at=now,
        source_files=[str(source)],
        working_file=str(source),
        working_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        metadata=_metadata(),
        provider="test",
        model="test",
    )


def _metadata() -> dict[str, object]:
    return {
        "duration_sec": 30.0,
        "width": 1920,
        "height": 1080,
        "fps": 30.0,
        "has_audio": True,
    }
