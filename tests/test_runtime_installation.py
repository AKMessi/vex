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


def _fake_node_identity(
    *,
    arch: str = "x64",
    major: int = 22,
    module_abi: str = "127",
) -> hyperframes.NodeRuntimeIdentity:
    return hyperframes.NodeRuntimeIdentity(
        executable="/tools/node",
        platform="win32",
        arch=arch,
        major=major,
        module_abi=module_abi,
    )


def _mock_managed_runtime_paths(
    monkeypatch: pytest.MonkeyPatch,
    runtime_base: Path,
) -> None:
    monkeypatch.setattr(
        hyperframes,
        "hyperframes_runtime_base_dir",
        lambda: runtime_base,
    )
    monkeypatch.setattr(
        hyperframes,
        "hyperframes_runtime_dir",
        lambda runtime_key=None: (
            runtime_base / runtime_key if runtime_key else runtime_base
        ),
    )
    monkeypatch.setattr(
        hyperframes,
        "managed_hyperframes_cli_path",
        lambda runtime_key=None: (
            (runtime_base / runtime_key if runtime_key else runtime_base)
            / "node_modules"
            / ".bin"
            / hyperframes.local_bin_name("hyperframes")
        ),
    )


def test_config_template_is_created_without_overwriting(tmp_path: Path) -> None:
    destination = tmp_path / ".env"

    result = write_config_template(destination)

    assert result == destination
    assert "PROVIDER=gemini" in destination.read_text(encoding="utf-8")
    assert "VEX_NODE_PATH=" in destination.read_text(encoding="utf-8")
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


def test_renderer_runtime_prefers_configured_node_and_sibling_npm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    node_dir = tmp_path / "node-x64"
    node_dir.mkdir()
    node_path = node_dir / "node.exe"
    npm_path = node_dir / "npm.cmd"
    node_path.write_bytes(b"node")
    npm_path.write_bytes(b"npm")
    monkeypatch.setenv(hyperframes.NODE_PATH_ENV, str(node_path))
    monkeypatch.delenv(hyperframes.NPM_PATH_ENV, raising=False)

    assert hyperframes.resolve_node_executable() == str(node_path.resolve())
    assert hyperframes.resolve_npm_executable() == str(npm_path)


def test_invalid_configured_node_does_not_fall_back_to_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(hyperframes.NODE_PATH_ENV, str(tmp_path / "missing-node.exe"))
    monkeypatch.setattr(hyperframes.shutil, "which", lambda _name: "/system/node")

    assert hyperframes.resolve_node_executable() is None


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


def test_managed_runtime_paths_are_isolated_by_node_architecture_and_abi(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("VEX_DATA_DIR", str(tmp_path))
    identities = iter(
        (
            _fake_node_identity(arch="arm64", module_abi="137"),
            _fake_node_identity(arch="x64", module_abi="137"),
            _fake_node_identity(arch="x64", module_abi="140"),
        )
    )
    monkeypatch.setattr(
        hyperframes,
        "node_runtime_identity",
        lambda _path=None: next(identities),
    )

    arm64 = hyperframes.managed_renderer_runtime_dir()
    x64 = hyperframes.managed_renderer_runtime_dir()
    next_abi = hyperframes.managed_renderer_runtime_dir()

    assert arm64 is not None and arm64.name == "win32-arm64-node22-abi137"
    assert x64 is not None and x64.name == "win32-x64-node22-abi137"
    assert next_abi is not None and next_abi.name == "win32-x64-node22-abi140"
    assert len({arm64, x64, next_abi}) == 3


def test_managed_runtime_install_is_locked_verified_and_reused(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_base = tmp_path / "runtime"
    identity = _fake_node_identity()
    runtime_dir = runtime_base / identity.runtime_key
    calls: list[list[str]] = []

    _mock_managed_runtime_paths(monkeypatch, runtime_base)
    monkeypatch.setattr(hyperframes, "node_runtime_identity", lambda _path=None: identity)
    monkeypatch.setattr(
        hyperframes.shutil,
        "which",
        lambda name: f"/tools/{name}" if name in {"node", "npm"} else None,
    )
    monkeypatch.setattr(
        hyperframes,
        "renderer_native_runtime_status",
        lambda **_: {
            "available": True,
            "reason": "",
            "versions": {"sharp": "0.34.5", "rspack": "loaded"},
            "remotion_ready": True,
        },
    )
    expected_version = hyperframes._locked_dependency_version()

    def fake_run(command, **kwargs):  # noqa: ANN001, ANN003
        calls.append(list(command))
        if command == ["/tools/node", "-p", "process.platform + ' ' + process.arch"]:
            return SimpleNamespace(returncode=0, stdout="win32 x64\n", stderr="")
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
        if command[-1] == "--version" and any("hyperframes" in part for part in command):
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
    assert marker["node_platform"] == "win32"
    assert marker["node_arch"] == "x64"
    assert marker["node_module_abi"] == "127"
    assert marker["runtime_key"] == "win32-x64-node22-abi127"
    assert marker["node_path"] == "/tools/node"
    assert len(marker["package_lock_sha256"]) == 64


def test_managed_runtime_reinstalls_when_lock_digest_changes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_base = tmp_path / "runtime"
    identity = _fake_node_identity()
    runtime_dir = runtime_base / identity.runtime_key
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

    _mock_managed_runtime_paths(monkeypatch, runtime_base)
    monkeypatch.setattr(hyperframes, "node_runtime_identity", lambda _path=None: identity)
    monkeypatch.setattr(
        hyperframes.shutil,
        "which",
        lambda name: f"/tools/{name}" if name in {"node", "npm"} else None,
    )
    monkeypatch.setattr(
        hyperframes,
        "renderer_native_runtime_status",
        lambda **_: {
            "available": True,
            "reason": "",
            "versions": {"sharp": "0.34.5", "rspack": "loaded"},
            "remotion_ready": True,
        },
    )
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
        if command == ["/tools/node", "-p", "process.platform + ' ' + process.arch"]:
            return SimpleNamespace(returncode=0, stdout="win32 x64\n", stderr="")
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
        if command[-1] == "--version" and any("hyperframes" in part for part in command):
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


def test_native_runtime_probe_reports_sharp_load_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cli = tmp_path / "node_modules" / ".bin" / hyperframes.local_bin_name("hyperframes")
    cli.parent.mkdir(parents=True)
    cli.write_text("cli", encoding="utf-8")
    monkeypatch.setattr(hyperframes, "resolve_node_executable", lambda: "/tools/node")
    monkeypatch.setattr(
        hyperframes,
        "node_runtime_identity",
        lambda _path=None: _fake_node_identity(arch="arm64"),
    )
    monkeypatch.setattr(
        hyperframes.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr='Could not load the "sharp" module using the win32-arm64 runtime',
        ),
    )

    status = hyperframes.renderer_native_runtime_status(cli_path=cli)

    assert status["available"] is False
    assert "sharp" in status["reason"]
    assert status["node_root"] == str(tmp_path)


def test_native_runtime_probe_loads_rspack_and_remotion_compositor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "node_modules").mkdir()
    commands: list[list[str]] = []
    identity = _fake_node_identity(major=24, module_abi="137")
    monkeypatch.setattr(
        hyperframes,
        "node_runtime_identity",
        lambda _path=None: identity,
    )

    def fake_run(command, **_kwargs):  # noqa: ANN001
        commands.append(list(command))
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "sharp": "0.34.5",
                    "vips": "8.17.3",
                    "rspack": "loaded",
                    "compositor": "@remotion/compositor-win32-x64-msvc",
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(hyperframes.subprocess, "run", fake_run)

    status = hyperframes.renderer_native_runtime_status(
        node_path="/tools/node",
        node_root=tmp_path,
        require_remotion=True,
    )

    assert status["available"] is True
    assert status["remotion_ready"] is True
    assert "require('@rspack/binding')" in commands[0][2]
    assert "@remotion/compositor-win32-x64-msvc" in commands[0][2]


def test_runtime_install_rejects_old_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hyperframes.shutil, "which", lambda name: f"/tools/{name}")
    monkeypatch.setattr(
        hyperframes,
        "node_runtime_identity",
        lambda _path=None: _fake_node_identity(major=20, module_abi="115"),
    )

    with pytest.raises(hyperframes.RuntimeInstallError, match="too old"):
        hyperframes.install_hyperframes_runtime()
