from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from vex_runtime import hyperframes
from vex_runtime.configuration import ConfigurationError, write_config_template
from vex_runtime.paths import _default_data_dir, data_dir


def _write_fake_renderer_packages(stage_dir: Path) -> None:
    for package_name, version in hyperframes._locked_dependency_versions().items():
        package_dir = hyperframes._package_dir(
            stage_dir / "node_modules",
            package_name,
        )
        package_dir.mkdir(parents=True, exist_ok=True)
        (package_dir / "package.json").write_text(
            json.dumps({"name": package_name, "version": version}),
            encoding="utf-8",
        )


def test_config_template_is_created_without_overwriting(tmp_path: Path) -> None:
    destination = tmp_path / ".env"

    result = write_config_template(destination)

    assert result == destination
    assert "PROVIDER=gemini" in destination.read_text(encoding="utf-8")
    if os.name != "nt":
        assert destination.stat().st_mode & 0o777 == 0o600
    with pytest.raises(ConfigurationError, match="already exists"):
        write_config_template(destination)


def test_data_dir_override_is_authoritative(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "custom-vex-data"
    monkeypatch.setenv("VEX_DATA_DIR", str(destination))

    assert data_dir() == destination.resolve()


@pytest.mark.parametrize(
    ("platform", "environ", "home_suffix", "expected_suffix"),
    [
        (
            "linux",
            {"XDG_DATA_HOME": "/var/tmp/xdg-data"},
            "/home/editor",
            "/var/tmp/xdg-data/vex",
        ),
        (
            "linux",
            {},
            "/home/editor",
            "/home/editor/.local/share/vex",
        ),
        (
            "darwin",
            {},
            "/Users/editor",
            "/Users/editor/Library/Application Support/vex",
        ),
        (
            "win32",
            {"LOCALAPPDATA": "C:/Users/editor/AppData/Local"},
            "C:/Users/editor",
            "C:/Users/editor/AppData/Local/AKMessi/vex",
        ),
        (
            "win32",
            {},
            "C:/Users/editor",
            "C:/Users/editor/AppData/Local/AKMessi/vex",
        ),
    ],
)
def test_default_data_dir_uses_platform_standard_locations(
    platform: str,
    environ: dict[str, str],
    home_suffix: str,
    expected_suffix: str,
) -> None:
    result = _default_data_dir(
        platform=platform,
        environ=environ,
        home=Path(home_suffix),
    )

    assert result.as_posix() == expected_suffix


def test_managed_runtime_install_is_locked_verified_and_reused(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_dir = tmp_path / "runtime"
    calls: list[list[str]] = []

    monkeypatch.setattr(hyperframes, "hyperframes_runtime_dir", lambda: runtime_dir)
    monkeypatch.setattr(
        hyperframes,
        "managed_hyperframes_cli_path",
        lambda: (
            runtime_dir
            / "node_modules"
            / ".bin"
            / hyperframes.local_bin_name("hyperframes")
        ),
    )
    monkeypatch.setattr(
        hyperframes.shutil,
        "which",
        lambda name: f"/tools/{name}" if name in {"node", "npm"} else None,
    )
    monkeypatch.setattr(hyperframes, "node_major_version", lambda _path=None: 22)
    expected_version = hyperframes._locked_dependency_version()

    def fake_run(command, **kwargs):  # noqa: ANN001, ANN003
        calls.append(list(command))
        if command[0] == "/tools/npm" and command[1] == "ci":
            _write_fake_renderer_packages(Path(kwargs["cwd"]))
            cli = (
                Path(kwargs["cwd"])
                / "node_modules"
                / ".bin"
                / hyperframes.local_bin_name("hyperframes")
            )
            cli.parent.mkdir(parents=True)
            cli.write_text("cli", encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="installed", stderr="")
        if command[-1] == "--version" and "hyperframes" in command[0]:
            return SimpleNamespace(
                returncode=0,
                stdout=f"{expected_version}\n",
                stderr="",
            )
        if command == ["/tools/npm", "--version"]:
            return SimpleNamespace(returncode=0, stdout="10.9.0\n", stderr="")
        raise AssertionError(command)

    monkeypatch.setattr(hyperframes.subprocess, "run", fake_run)

    first = hyperframes.install_hyperframes_runtime()
    second = hyperframes.install_hyperframes_runtime()

    assert first["changed"] is True
    assert second["changed"] is False
    assert len([command for command in calls if command[1:2] == ["ci"]]) == 1
    marker = json.loads((runtime_dir / hyperframes.INSTALL_MARKER).read_text(encoding="utf-8"))
    assert marker["hyperframes_version"] == expected_version
    assert marker["remotion_version"] == hyperframes._locked_dependency_version("remotion")
    assert marker["renderer_versions"] == hyperframes._locked_dependency_versions()
    assert len(marker["package_lock_sha256"]) == 64


def test_managed_runtime_reinstalls_when_lock_digest_changes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_dir = tmp_path / "runtime"
    cli = (
        runtime_dir
        / "node_modules"
        / ".bin"
        / hyperframes.local_bin_name("hyperframes")
    )
    cli.parent.mkdir(parents=True)
    cli.write_text("old cli", encoding="utf-8")
    (runtime_dir / hyperframes.INSTALL_MARKER).write_text(
        json.dumps(
            {
                "hyperframes_version": "0.6.99",
                "package_lock_sha256": "old",
            }
        ),
        encoding="utf-8",
    )
    calls: list[list[str]] = []

    monkeypatch.setattr(hyperframes, "hyperframes_runtime_dir", lambda: runtime_dir)
    monkeypatch.setattr(hyperframes, "managed_hyperframes_cli_path", lambda: cli)
    monkeypatch.setattr(
        hyperframes.shutil,
        "which",
        lambda name: f"/tools/{name}" if name in {"node", "npm"} else None,
    )
    monkeypatch.setattr(hyperframes, "node_major_version", lambda _path=None: 22)
    expected_versions = hyperframes._locked_dependency_versions()
    expected_versions["hyperframes"] = "0.6.112"
    monkeypatch.setattr(
        hyperframes,
        "_locked_dependency_versions",
        lambda: expected_versions,
    )
    monkeypatch.setattr(hyperframes, "_lock_digest", lambda: "new-digest")

    def fake_run(command, **kwargs):  # noqa: ANN001, ANN003
        calls.append(list(command))
        if command[0] == "/tools/npm" and command[1] == "ci":
            _write_fake_renderer_packages(Path(kwargs["cwd"]))
            staged_cli = (
                Path(kwargs["cwd"])
                / "node_modules"
                / ".bin"
                / hyperframes.local_bin_name("hyperframes")
            )
            staged_cli.parent.mkdir(parents=True)
            staged_cli.write_text("new cli", encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="installed", stderr="")
        if command[-1] == "--version" and "hyperframes" in command[0]:
            return SimpleNamespace(returncode=0, stdout="0.6.112\n", stderr="")
        if command == ["/tools/npm", "--version"]:
            return SimpleNamespace(returncode=0, stdout="10.9.0\n", stderr="")
        raise AssertionError(command)

    monkeypatch.setattr(hyperframes.subprocess, "run", fake_run)

    result = hyperframes.install_hyperframes_runtime()

    assert result["changed"] is True
    assert any(command[1:2] == ["ci"] for command in calls)
    marker = json.loads((runtime_dir / hyperframes.INSTALL_MARKER).read_text(encoding="utf-8"))
    assert marker["hyperframes_version"] == "0.6.112"
    assert marker["remotion_version"] == expected_versions["remotion"]
    assert marker["package_lock_sha256"] == "new-digest"


def test_runtime_verification_rejects_missing_remotion_package(tmp_path: Path) -> None:
    _write_fake_renderer_packages(tmp_path)
    remotion_manifest = (
        hyperframes._package_dir(tmp_path / "node_modules", "remotion")
        / "package.json"
    )
    remotion_manifest.unlink()

    with pytest.raises(hyperframes.RuntimeInstallError, match="remotion is missing"):
        hyperframes._verify_staged_renderer_packages(tmp_path)


def test_runtime_install_rejects_old_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hyperframes.shutil, "which", lambda name: f"/tools/{name}")
    monkeypatch.setattr(hyperframes, "node_major_version", lambda _path=None: 20)

    with pytest.raises(hyperframes.RuntimeInstallError, match="too old"):
        hyperframes.install_hyperframes_runtime()
