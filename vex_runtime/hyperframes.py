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


def node_major_version(node_path: str | None = None) -> int | None:
    executable = node_path or shutil.which("node")
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


def _resource_bytes(name: str) -> bytes:
    resource = files("vex_runtime").joinpath("resources", "hyperframes", name)
    return resource.read_bytes()


def _locked_dependency_version() -> str:
    manifest = json.loads(_resource_bytes("package.json"))
    version = str((manifest.get("dependencies") or {}).get("hyperframes") or "").strip()
    if not version:
        raise RuntimeInstallError("Packaged HyperFrames manifest does not pin a dependency version.")
    return version


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
    return [str(cli_path), *args]


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
    expected_version = _locked_dependency_version()
    expected_digest = _lock_digest()
    installed = cli_path.is_file() and marker_path.is_file()
    matches_expected = (
        installed
        and metadata.get("hyperframes_version") == expected_version
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
                    f"Another HyperFrames installation is active at {lock_dir}."
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
    node_path = shutil.which("node")
    npm_path = shutil.which("npm")
    node_major = node_major_version(node_path)
    if not node_path or node_major is None:
        raise RuntimeInstallError("Node.js is not available. Install Node.js 22 or newer.")
    if node_major < MINIMUM_NODE_MAJOR:
        raise RuntimeInstallError(
            f"Node.js {node_major} is too old. HyperFrames requires Node.js {MINIMUM_NODE_MAJOR}+."
        )
    if not npm_path:
        raise RuntimeInstallError("npm is not available. Install npm before installing HyperFrames.")

    target_dir = hyperframes_runtime_dir()
    runtime_parent = target_dir.parent
    with _installation_lock(runtime_parent):
        existing = installed_runtime_status()
        if existing["installed"] and existing["matches_expected"] and not force:
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
                    f"HyperFrames dependency installation did not complete: {exc}",
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

            return_code, installed_version = _command_version(
                [str(staged_cli), "--version"]
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
                "schema_version": 1,
                "installed_at": datetime.now(timezone.utc).isoformat(),
                "vex_version": __version__,
                "hyperframes_version": expected_version,
                "node_major": node_major,
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
                f"HyperFrames runtime installation failed: {exc}",
                log_path=failed_log,
            ) from exc
        finally:
            shutil.rmtree(stage_dir, ignore_errors=True)

    status = installed_runtime_status()
    return {**status, "changed": True}
