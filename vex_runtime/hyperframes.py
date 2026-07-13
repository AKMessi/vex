from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
from typing import Any, Iterator

from vex_runtime import __version__
from vex_runtime.paths import (
    hyperframes_runtime_dir,
    local_bin_name,
    managed_hyperframes_cli_path,
)

MINIMUM_NODE_MAJOR = 22
INSTALL_TIMEOUT_SEC = 1800
INSTALL_MARKER = ".vex-runtime.json"
REQUIRED_RENDERER_PACKAGES = (
    "hyperframes",
    "remotion",
    "@remotion/renderer",
    "@remotion/bundler",
    "@remotion/layout-utils",
    "react",
    "react-dom",
)
NODE_PATH_ENV = "VEX_NODE_PATH"
NPM_PATH_ENV = "VEX_NPM_PATH"


class RuntimeInstallError(RuntimeError):
    def __init__(self, message: str, *, log_path: Path | None = None) -> None:
        super().__init__(message)
        self.log_path = log_path


def _command_version(command: list[str], *, timeout: int = 20) -> tuple[int, str]:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeInstallError(f"Could not run {' '.join(command)}: {exc}") from exc
    output = (result.stdout or result.stderr or "").strip()
    return result.returncode, output


def _configured_executable(env_name: str) -> tuple[str | None, bool]:
    raw = str(os.getenv(env_name) or "").strip().strip('"')
    if not raw:
        return None, False
    candidate = Path(os.path.expandvars(raw)).expanduser().resolve(strict=False)
    return (str(candidate) if candidate.is_file() else None), True


def resolve_node_executable() -> str | None:
    configured, is_configured = _configured_executable(NODE_PATH_ENV)
    return configured if is_configured else shutil.which("node")


def resolve_npm_executable(*, node_path: str | None = None) -> str | None:
    configured, is_configured = _configured_executable(NPM_PATH_ENV)
    if is_configured:
        return configured

    selected_node, node_is_configured = _configured_executable(NODE_PATH_ENV)
    selected_node = node_path or selected_node
    if node_is_configured:
        if not selected_node:
            return None
        node_dir = Path(selected_node).parent
        candidates = (
            node_dir / "npm.cmd",
            node_dir / "npm",
            node_dir / "bin" / "npm",
        )
        return next((str(path) for path in candidates if path.is_file()), None)
    return shutil.which("npm")


def node_major_version(node_path: str | None = None) -> int | None:
    executable = node_path or resolve_node_executable()
    if not executable:
        return None
    try:
        return_code, output = _command_version([executable, "--version"])
    except RuntimeInstallError:
        return None
    if return_code != 0:
        return None
    match = re.search(r"v?(\d+)", output)
    return int(match.group(1)) if match else None


def node_platform_arch(node_path: str | None = None) -> tuple[str | None, str | None]:
    executable = node_path or resolve_node_executable()
    if not executable:
        return None, None
    try:
        return_code, output = _command_version(
            [executable, "-p", "process.platform + ' ' + process.arch"]
        )
    except RuntimeInstallError:
        return None, None
    if return_code != 0:
        return None, None
    parts = output.split()
    if len(parts) != 2:
        return None, None
    return parts[0], parts[1]


def _resource_bytes(name: str) -> bytes:
    resource = files("vex_runtime").joinpath("resources", "hyperframes", name)
    return resource.read_bytes()


def _locked_dependency_versions() -> dict[str, str]:
    manifest = json.loads(_resource_bytes("package.json"))
    dependencies = manifest.get("dependencies")
    if not isinstance(dependencies, dict):
        raise RuntimeInstallError("Packaged renderer manifest has no dependency map.")
    versions = {
        name: str(dependencies.get(name) or "").strip()
        for name in REQUIRED_RENDERER_PACKAGES
    }
    missing = [name for name, version in versions.items() if not version]
    if missing:
        raise RuntimeInstallError(
            "Packaged renderer manifest does not pin required dependencies: "
            + ", ".join(missing)
        )
    return versions


def _locked_dependency_version(name: str = "hyperframes") -> str:
    versions = _locked_dependency_versions()
    try:
        return versions[name]
    except KeyError as exc:
        raise RuntimeInstallError(f"Unknown managed renderer dependency: {name}") from exc


def _package_dir(node_modules: Path, package_name: str) -> Path:
    return node_modules.joinpath(*package_name.split("/"))


def _verify_staged_renderer_packages(stage_dir: Path) -> dict[str, str]:
    expected = _locked_dependency_versions()
    installed: dict[str, str] = {}
    for package_name, expected_version in expected.items():
        manifest_path = _package_dir(stage_dir / "node_modules", package_name) / "package.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeInstallError(
                f"npm completed but required renderer package {package_name} is missing or invalid."
            ) from exc
        installed_version = str(manifest.get("version") or "").strip()
        if installed_version != expected_version:
            raise RuntimeInstallError(
                f"Managed renderer package {package_name} version mismatch: "
                f"expected {expected_version}, received {installed_version or 'no version'}."
            )
        installed[package_name] = installed_version
    return installed


def resolve_hyperframes_cli_path(
    configured_cli: str = "hyperframes",
    *,
    repo_root: Path | None = None,
) -> Path | None:
    """Resolve only trusted HyperFrames CLI locations.

    Vex intentionally avoids falling back to a global `hyperframes` binary or a
    cwd-local `node_modules` path. The renderer must come from either the
    repository/runtime bundle or an explicit user-configured path.
    """
    configured = str(configured_cli or "hyperframes").strip() or "hyperframes"
    configured_path = Path(configured)
    if configured_path.is_absolute() or configured_path.parent != Path("."):
        return configured_path if configured_path.is_file() else None

    root = repo_root or Path(__file__).resolve().parent.parent
    binary_name = local_bin_name(configured)
    for candidate in (
        root / "node_modules" / ".bin" / binary_name,
        managed_hyperframes_cli_path(),
    ):
        if candidate.is_file():
            return candidate
    return None


def hyperframes_cli_command(cli_path: str | Path, *args: str) -> list[str]:
    cli = Path(cli_path).expanduser().resolve(strict=False)
    package_entrypoint = cli.parent.parent / "hyperframes" / "dist" / "cli.js"
    if package_entrypoint.is_file():
        node_path = resolve_node_executable()
        if not node_path:
            raise RuntimeInstallError(
                "Node.js is unavailable; check PATH or VEX_NODE_PATH."
            )
        return [node_path, str(package_entrypoint), *args]
    return [str(cli), *args]


def hyperframes_command(
    *args: str,
    configured_cli: str = "hyperframes",
    repo_root: Path | None = None,
) -> list[str]:
    cli_path = resolve_hyperframes_cli_path(
        configured_cli,
        repo_root=repo_root,
    )
    if cli_path is None:
        raise RuntimeInstallError(
            "HyperFrames CLI is unavailable. Run `vex renderers install hyperframes` "
            "or set HYPERFRAMES_CLI_PATH."
        )
    return hyperframes_cli_command(cli_path, *args)


def renderer_native_runtime_status(
    *,
    cli_path: str | Path | None = None,
    node_path: str | None = None,
) -> dict[str, Any]:
    selected_cli = Path(cli_path) if cli_path else resolve_hyperframes_cli_path()
    selected_node = node_path or resolve_node_executable()
    if selected_cli is None or not selected_cli.is_file():
        return {
            "available": False,
            "reason": "HyperFrames CLI is unavailable for native dependency probing.",
            "node_root": None,
        }
    if not selected_node:
        return {
            "available": False,
            "reason": "Node.js is unavailable for native dependency probing.",
            "node_root": None,
        }
    resolved_cli = selected_cli.expanduser().resolve(strict=False)
    node_modules = resolved_cli.parent.parent
    node_root = node_modules.parent
    probe = (
        "const sharp=require('sharp');"
        "if(!sharp.versions||!sharp.versions.sharp||!sharp.versions.vips){"
        "throw new Error('sharp native runtime did not expose version metadata');}"
        "process.stdout.write(JSON.stringify({sharp:sharp.versions.sharp,vips:sharp.versions.vips}));"
    )
    try:
        result = subprocess.run(
            [selected_node, "-e", probe],
            cwd=node_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "available": False,
            "reason": f"HyperFrames native dependency probe could not run: {exc}",
            "node_root": str(node_root),
        }
    detail = (result.stderr or result.stdout or "").strip()
    versions: dict[str, Any] = {}
    if result.returncode == 0:
        try:
            versions = json.loads(result.stdout)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {
                "available": False,
                "reason": "HyperFrames native dependency probe returned invalid version metadata.",
                "node_root": str(node_root),
                "versions": {},
            }
    return {
        "available": result.returncode == 0,
        "reason": (
            ""
            if result.returncode == 0
            else "HyperFrames native image runtime is unavailable: "
            + " ".join(detail.split())[:600]
        ),
        "node_root": str(node_root),
        "versions": versions,
    }


def _lock_digest() -> str:
    return hashlib.sha256(_resource_bytes("package-lock.json")).hexdigest()


def installed_runtime_status() -> dict[str, Any]:
    runtime_dir = hyperframes_runtime_dir()
    cli_path = managed_hyperframes_cli_path()
    marker_path = runtime_dir / INSTALL_MARKER
    metadata: dict[str, Any] = {}
    if marker_path.is_file():
        try:
            metadata = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            metadata = {}
    expected_versions = _locked_dependency_versions()
    expected_version = expected_versions["hyperframes"]
    expected_digest = _lock_digest()
    installed = cli_path.is_file() and marker_path.is_file()
    installed_versions = metadata.get("renderer_versions")
    matches_expected = (
        installed
        and metadata.get("hyperframes_version") == expected_version
        and metadata.get("remotion_version") == expected_versions["remotion"]
        and installed_versions == expected_versions
        and metadata.get("package_lock_sha256") == expected_digest
    )
    return {
        "installed": cli_path.is_file() and marker_path.is_file(),
        "matches_expected": matches_expected,
        "runtime_dir": str(runtime_dir),
        "cli_path": str(cli_path) if cli_path.is_file() else None,
        "marker_path": str(marker_path),
        "metadata": metadata,
        "expected_hyperframes_version": expected_version,
        "expected_remotion_version": expected_versions["remotion"],
        "expected_renderer_versions": expected_versions,
        "expected_package_lock_sha256": expected_digest,
        "vex_version": __version__,
    }


@contextmanager
def _installation_lock(parent: Path, *, timeout_sec: float = 30.0) -> Iterator[None]:
    parent.mkdir(parents=True, exist_ok=True)
    lock_dir = parent / ".install.lock"
    deadline = time.monotonic() + timeout_sec
    while True:
        try:
            lock_dir.mkdir()
            break
        except FileExistsError:
            try:
                stale = time.time() - lock_dir.stat().st_mtime > INSTALL_TIMEOUT_SEC
            except FileNotFoundError:
                continue
            if stale:
                shutil.rmtree(lock_dir, ignore_errors=True)
                continue
            if time.monotonic() >= deadline:
                raise RuntimeInstallError(
                    f"Another managed renderer installation is active at {lock_dir}."
                )
            time.sleep(0.2)
    try:
        yield
    finally:
        shutil.rmtree(lock_dir, ignore_errors=True)


def _persist_failed_install(stage_dir: Path, runtime_parent: Path) -> Path | None:
    if not stage_dir.exists():
        return None
    failed_dir = runtime_parent / "failed-installs" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{uuid.uuid4().hex[:8]}"
    )
    failed_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(stage_dir, failed_dir)
    except OSError:
        return stage_dir / "install.log"
    return failed_dir / "install.log"


def install_hyperframes_runtime(*, force: bool = False) -> dict[str, Any]:
    node_path = resolve_node_executable()
    npm_path = resolve_npm_executable(node_path=node_path)
    node_major = node_major_version(node_path)
    if not node_path or node_major is None:
        raise RuntimeInstallError(
            "Node.js is unavailable. Install Node.js 22+ or set VEX_NODE_PATH."
        )
    if node_major < MINIMUM_NODE_MAJOR:
        raise RuntimeInstallError(
            f"Node.js {node_major} is too old. Managed renderers require Node.js {MINIMUM_NODE_MAJOR}+."
        )
    if not npm_path:
        raise RuntimeInstallError(
            "npm is not available for the selected Node.js runtime. Install npm or set VEX_NPM_PATH."
        )

    node_platform, node_arch = node_platform_arch(node_path)
    if not node_platform or not node_arch:
        raise RuntimeInstallError("Could not determine the selected Node.js platform and architecture.")

    target_dir = hyperframes_runtime_dir()
    runtime_parent = target_dir.parent
    with _installation_lock(runtime_parent):
        existing = installed_runtime_status()
        existing_metadata = existing.get("metadata") or {}
        runtime_matches_node = (
            existing_metadata.get("node_platform") == node_platform
            and existing_metadata.get("node_arch") == node_arch
        )
        if (
            existing["installed"]
            and existing["matches_expected"]
            and runtime_matches_node
            and not force
        ):
            return {**existing, "changed": False}

        stage_dir = runtime_parent / f".install-{uuid.uuid4().hex}"
        backup_dir = runtime_parent / f".backup-{uuid.uuid4().hex}"
        log_path = stage_dir / "install.log"
        stage_dir.mkdir(parents=True, exist_ok=False)
        try:
            for name in ("package.json", "package-lock.json"):
                (stage_dir / name).write_bytes(_resource_bytes(name))

            command = [
                npm_path,
                "ci",
                "--no-audit",
                "--no-fund",
                "--loglevel=error",
            ]
            try:
                result = subprocess.run(
                    command,
                    cwd=stage_dir,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=INSTALL_TIMEOUT_SEC,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                log_path.write_text(f"$ {' '.join(command)}\n\n{exc}\n", encoding="utf-8")
                failed_log = _persist_failed_install(stage_dir, runtime_parent)
                raise RuntimeInstallError(
                    f"Renderer dependency installation did not complete: {exc}",
                    log_path=failed_log,
                ) from exc

            log_path.write_text(
                "\n".join(
                    [
                        "$ " + " ".join(command),
                        "",
                        f"exit_code={result.returncode}",
                        "",
                        "[stdout]",
                        result.stdout or "",
                        "",
                        "[stderr]",
                        result.stderr or "",
                    ]
                ),
                encoding="utf-8",
            )
            if result.returncode != 0:
                failed_log = _persist_failed_install(stage_dir, runtime_parent)
                raise RuntimeInstallError(
                    f"npm ci failed with exit code {result.returncode}.",
                    log_path=failed_log,
                )

            staged_cli = (
                stage_dir / "node_modules" / ".bin" / local_bin_name("hyperframes")
            )
            if not staged_cli.is_file():
                failed_log = _persist_failed_install(stage_dir, runtime_parent)
                raise RuntimeInstallError(
                    "npm completed but did not install the HyperFrames CLI.",
                    log_path=failed_log,
                )

            try:
                renderer_versions = _verify_staged_renderer_packages(stage_dir)
            except RuntimeInstallError as exc:
                failed_log = _persist_failed_install(stage_dir, runtime_parent)
                raise RuntimeInstallError(str(exc), log_path=failed_log) from exc

            native_status = renderer_native_runtime_status(
                cli_path=staged_cli,
                node_path=node_path,
            )
            if not native_status["available"]:
                failed_log = _persist_failed_install(stage_dir, runtime_parent)
                raise RuntimeInstallError(
                    str(native_status["reason"]),
                    log_path=failed_log,
                )

            return_code, installed_version = _command_version(
                hyperframes_cli_command(staged_cli, "--version")
            )
            expected_version = _locked_dependency_version()
            if return_code != 0 or installed_version.strip() != expected_version:
                failed_log = _persist_failed_install(stage_dir, runtime_parent)
                raise RuntimeInstallError(
                    "Installed HyperFrames version verification failed: "
                    f"expected {expected_version}, received {installed_version or 'no version'}.",
                    log_path=failed_log,
                )

            npm_return_code, npm_version = _command_version([npm_path, "--version"])
            metadata = {
                "schema_version": 3,
                "installed_at": datetime.now(timezone.utc).isoformat(),
                "vex_version": __version__,
                "hyperframes_version": expected_version,
                "remotion_version": renderer_versions["remotion"],
                "renderer_versions": renderer_versions,
                "native_runtime_versions": native_status.get("versions") or {},
                "node_major": node_major,
                "node_platform": node_platform,
                "node_arch": node_arch,
                "node_path": node_path,
                "npm_version": npm_version if npm_return_code == 0 else None,
                "package_lock_sha256": _lock_digest(),
            }
            (stage_dir / INSTALL_MARKER).write_text(
                json.dumps(metadata, indent=2, sort_keys=True),
                encoding="utf-8",
            )

            if target_dir.exists():
                os.replace(target_dir, backup_dir)
            try:
                os.replace(stage_dir, target_dir)
            except OSError:
                if backup_dir.exists() and not target_dir.exists():
                    os.replace(backup_dir, target_dir)
                raise
            shutil.rmtree(backup_dir, ignore_errors=True)
        except RuntimeInstallError:
            raise
        except Exception as exc:
            failed_log = _persist_failed_install(stage_dir, runtime_parent)
            raise RuntimeInstallError(
                f"Managed renderer runtime installation failed: {exc}",
                log_path=failed_log,
            ) from exc
        finally:
            shutil.rmtree(stage_dir, ignore_errors=True)

    status = installed_runtime_status()
    return {**status, "changed": True}
