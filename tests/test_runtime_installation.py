from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from vex_runtime import hyperframes
from vex_runtime.configuration import ConfigurationError, write_config_template
from vex_runtime.paths import _default_data_dir, data_dir


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
    assert len(marker["package_lock_sha256"]) == 64


def test_runtime_install_rejects_old_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hyperframes.shutil, "which", lambda name: f"/tools/{name}")
    monkeypatch.setattr(hyperframes, "node_major_version", lambda _path=None: 20)

    with pytest.raises(hyperframes.RuntimeInstallError, match="too old"):
        hyperframes.install_hyperframes_runtime()
