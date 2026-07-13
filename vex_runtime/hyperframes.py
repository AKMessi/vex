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
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
from typing import Any, Iterator

from vex_runtime import __version__
from vex_runtime.paths import (
    hyperframes_runtime_base_dir,
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


@dataclass(frozen=True)
class NodeRuntimeIdentity:
    executable: str
    platform: str
    arch: str
    major: int
    module_abi: str
    libc: str | None = None

    @property
    def runtime_key(self) -> str:
        parts = [self.platform, self.arch]
        if self.libc:
            parts.append(self.libc)
        parts.extend((f"node{self.major}", f"abi{self.module_abi}"))
        return "-".join(parts).lower()

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_path": self.executable,
            "node_platform": self.platform,
            "node_arch": self.arch,
            "node_major": self.major,
            "node_module_abi": self.module_abi,
            "node_libc": self.libc,
            "runtime_key": self.runtime_key,
        }


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
    if node_path or node_is_configured:
        if not selected_node:
            return None
        node_dir = Path(selected_node).parent
        candidates = (
            node_dir / "npm.cmd",
            node_dir / "npm",
            node_dir / "bin" / "npm",
        )
        sibling_npm = next((str(path) for path in candidates if path.is_file()), None)
        if sibling_npm or node_is_configured:
            return sibling_npm
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


def node_runtime_identity(
    node_path: str | None = None,
) -> NodeRuntimeIdentity | None:
    executable = node_path or resolve_node_executable()
    if not executable:
        return None
    probe = (
        "const header=process.report?.getReport?.().header||{};"
        "const major=Number(process.versions.node.split('.')[0]);"
        "const libc=process.platform==='linux'"
        "?(header.glibcVersionRuntime?'glibc':'musl'):null;"
        "process.stdout.write(JSON.stringify({platform:process.platform,"
        "arch:process.arch,major,moduleAbi:process.versions.modules,libc}));"
    )
    try:
        return_code, output = _command_version([executable, "-e", probe])
        payload = json.loads(output) if return_code == 0 else {}
    except (RuntimeInstallError, TypeError, ValueError, json.JSONDecodeError):
        return None
    platform = str(payload.get("platform") or "").strip().lower()
    arch = str(payload.get("arch") or "").strip().lower()
    module_abi = str(payload.get("moduleAbi") or "").strip().lower()
    try:
        major = int(payload.get("major"))
    except (TypeError, ValueError):
        return None
    libc = str(payload.get("libc") or "").strip().lower() or None
    token_pattern = re.compile(r"^[a-z0-9_]+$")
    if (
        major <= 0
        or not token_pattern.fullmatch(platform)
        or not token_pattern.fullmatch(arch)
        or not token_pattern.fullmatch(module_abi)
        or (libc is not None and not token_pattern.fullmatch(libc))
    ):
        return None
    return NodeRuntimeIdentity(
        executable=str(Path(executable).expanduser().resolve(strict=False)),
        platform=platform,
        arch=arch,
        major=major,
        module_abi=module_abi,
        libc=libc,
    )


def managed_renderer_runtime_dir(node_path: str | None = None) -> Path | None:
    identity = node_runtime_identity(node_path)
    if identity is None:
        return None
    return hyperframes_runtime_dir(identity.runtime_key)


def managed_hyperframes_cli_path_for_node(
    node_path: str | None = None,
) -> Path | None:
    identity = node_runtime_identity(node_path)
    if identity is None:
        return None
    return managed_hyperframes_cli_path(identity.runtime_key)


def remotion_compositor_package(
    identity: NodeRuntimeIdentity,
) -> str | None:
    if identity.platform == "win32":
        return (
            "@remotion/compositor-win32-x64-msvc"
            if identity.arch == "x64"
            else None
        )
    if identity.platform == "darwin" and identity.arch in {"arm64", "x64"}:
        return f"@remotion/compositor-darwin-{identity.arch}"
    if identity.platform == "linux" and identity.arch in {"arm64", "x64"}:
        libc = "musl" if identity.libc == "musl" else "gnu"
        return f"@remotion/compositor-linux-{identity.arch}-{libc}"
    return None


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
    node_path: str | None = None,
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
    candidates: list[Path] = []
    managed_cli = managed_hyperframes_cli_path_for_node(node_path)
    if managed_cli is not None:
        candidates.append(managed_cli)
    candidates.append(root / "node_modules" / ".bin" / binary_name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def hyperframes_cli_command(
    cli_path: str | Path,
    *args: str,
    node_path: str | None = None,
) -> list[str]:
    cli = Path(cli_path).expanduser().resolve(strict=False)
    package_entrypoint = cli.parent.parent / "hyperframes" / "dist" / "cli.js"
    if package_entrypoint.is_file():
        selected_node = node_path or resolve_node_executable()
        if not selected_node:
            raise RuntimeInstallError(
                "Node.js is unavailable; check PATH or VEX_NODE_PATH."
            )
        return [selected_node, str(package_entrypoint), *args]
    return [str(cli), *args]


def hyperframes_command(
    *args: str,
    configured_cli: str = "hyperframes",
    repo_root: Path | None = None,
    node_path: str | None = None,
) -> list[str]:
    cli_path = resolve_hyperframes_cli_path(
        configured_cli,
        repo_root=repo_root,
        node_path=node_path,
    )
    if cli_path is None:
        raise RuntimeInstallError(
            "HyperFrames CLI is unavailable. Run `vex renderers install hyperframes` "
            "or set HYPERFRAMES_CLI_PATH."
        )
    return hyperframes_cli_command(cli_path, *args, node_path=node_path)


def renderer_native_runtime_status(
    *,
    cli_path: str | Path | None = None,
    node_path: str | None = None,
    node_root: str | Path | None = None,
    require_remotion: bool = False,
) -> dict[str, Any]:
    selected_node = node_path or resolve_node_executable()
    if not selected_node:
        return {
            "available": False,
            "reason": "Node.js is unavailable for native dependency probing.",
            "node_root": None,
        }
    identity = node_runtime_identity(selected_node)
    if identity is None:
        return {
            "available": False,
            "reason": "Could not identify the Node.js runtime for native dependency probing.",
            "node_root": str(node_root) if node_root else None,
        }
    selected_root: Path | None = None
    if node_root is not None:
        selected_root = Path(node_root).expanduser().resolve(strict=False)
    else:
        selected_cli = (
            Path(cli_path)
            if cli_path
            else resolve_hyperframes_cli_path(node_path=selected_node)
        )
        if selected_cli is None or not selected_cli.is_file():
            return {
                "available": False,
                "reason": "HyperFrames CLI is unavailable for native dependency probing.",
                "node_root": None,
                "runtime_identity": identity.to_dict(),
            }
        resolved_cli = selected_cli.expanduser().resolve(strict=False)
        selected_root = resolved_cli.parent.parent.parent
    if not (selected_root / "node_modules").is_dir():
        return {
            "available": False,
            "reason": f"{selected_root} does not contain node_modules.",
            "node_root": str(selected_root),
            "runtime_identity": identity.to_dict(),
        }
    compositor_package = remotion_compositor_package(identity)
    if require_remotion and compositor_package is None:
        return {
            "available": False,
            "reason": (
                "Remotion does not provide a native compositor for "
                f"{identity.platform}/{identity.arch}."
            ),
            "node_root": str(selected_root),
            "runtime_identity": identity.to_dict(),
        }
    remotion_packages = [
        "remotion",
        "@remotion/renderer",
        "@remotion/bundler",
        "@remotion/layout-utils",
        "react",
        "react-dom",
    ] if require_remotion else []
    probe = (
        f"const packages={json.dumps(remotion_packages)};"
        "for(const name of packages){require.resolve(name);}"
        "const sharp=require('sharp');"
        "if(!sharp.versions||!sharp.versions.sharp||!sharp.versions.vips){"
        "throw new Error('sharp native runtime did not expose version metadata');}"
        "require('@rspack/binding');"
        f"const compositor={json.dumps(compositor_package if require_remotion else None)};"
        "if(compositor){const runtime=require(compositor);"
        "if(!runtime||typeof runtime.dir!=='string'){"
        "throw new Error('Remotion compositor package did not expose its runtime directory');}}"
        "process.stdout.write(JSON.stringify({sharp:sharp.versions.sharp,"
        "vips:sharp.versions.vips,rspack:'loaded',compositor}));"
    )
    try:
        result = subprocess.run(
            [selected_node, "-e", probe],
            cwd=selected_root,
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
            "reason": f"Renderer native dependency probe could not run: {exc}",
            "node_root": str(selected_root),
            "runtime_identity": identity.to_dict(),
        }
    detail = (result.stderr or result.stdout or "").strip()
    versions: dict[str, Any] = {}
    if result.returncode == 0:
        try:
            versions = json.loads(result.stdout)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {
                "available": False,
                "reason": "Renderer native dependency probe returned invalid version metadata.",
                "node_root": str(selected_root),
                "versions": {},
                "runtime_identity": identity.to_dict(),
            }
    return {
        "available": result.returncode == 0,
        "reason": (
            ""
            if result.returncode == 0
            else "Renderer native runtime is unavailable for "
            + f"{identity.runtime_key}: "
            + " ".join(detail.split())[:600]
        ),
        "node_root": str(selected_root),
        "versions": versions,
        "runtime_identity": identity.to_dict(),
        "remotion_ready": bool(result.returncode == 0 and require_remotion),
    }


def _lock_digest() -> str:
    return hashlib.sha256(_resource_bytes("package-lock.json")).hexdigest()


def installed_runtime_status(
    *,
    node_path: str | None = None,
) -> dict[str, Any]:
    identity = node_runtime_identity(node_path)
    runtime_dir = (
        hyperframes_runtime_dir(identity.runtime_key)
        if identity is not None
        else hyperframes_runtime_base_dir()
    )
    cli_path = (
        managed_hyperframes_cli_path(identity.runtime_key)
        if identity is not None
        else runtime_dir / "node_modules" / ".bin" / local_bin_name("hyperframes")
    )
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
    identity_matches = bool(
        identity is not None
        and metadata.get("runtime_key") == identity.runtime_key
        and metadata.get("node_platform") == identity.platform
        and metadata.get("node_arch") == identity.arch
        and metadata.get("node_major") == identity.major
        and str(metadata.get("node_module_abi") or "") == identity.module_abi
        and (metadata.get("node_libc") or None) == identity.libc
    )
    matches_expected = (
        installed
        and identity_matches
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
        "runtime_identity": identity.to_dict() if identity is not None else None,
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


def install_hyperframes_runtime(
    *,
    force: bool = False,
    node_path: str | None = None,
    npm_path: str | None = None,
) -> dict[str, Any]:
    selected_node = node_path or resolve_node_executable()
    identity = node_runtime_identity(selected_node)
    if identity is None:
        raise RuntimeInstallError(
            "Node.js is unavailable or could not be identified. Install Node.js 22+ "
            "or set VEX_NODE_PATH."
        )
    selected_node = identity.executable
    selected_npm = npm_path or resolve_npm_executable(node_path=selected_node)
    if identity.major < MINIMUM_NODE_MAJOR:
        raise RuntimeInstallError(
            f"Node.js {identity.major} is too old. Managed renderers require Node.js {MINIMUM_NODE_MAJOR}+."
        )
    if not selected_npm:
        raise RuntimeInstallError(
            "npm is not available for the selected Node.js runtime. Install npm or set VEX_NPM_PATH."
        )

    target_dir = hyperframes_runtime_dir(identity.runtime_key)
    runtime_parent = hyperframes_runtime_base_dir()
    with _installation_lock(runtime_parent):
        existing = installed_runtime_status(node_path=selected_node)
        if (
            existing["installed"]
            and existing["matches_expected"]
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
                selected_npm,
                "ci",
                "--no-audit",
                "--no-fund",
                "--include=optional",
                f"--os={identity.platform}",
                f"--cpu={identity.arch}",
                "--loglevel=error",
            ]
            if identity.libc:
                command.append(f"--libc={identity.libc}")
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
                node_path=selected_node,
                require_remotion=remotion_compositor_package(identity) is not None,
            )
            if not native_status["available"]:
                failed_log = _persist_failed_install(stage_dir, runtime_parent)
                raise RuntimeInstallError(
                    str(native_status["reason"]),
                    log_path=failed_log,
                )

            return_code, installed_version = _command_version(
                hyperframes_cli_command(
                    staged_cli,
                    "--version",
                    node_path=selected_node,
                )
            )
            expected_version = _locked_dependency_version()
            if return_code != 0 or installed_version.strip() != expected_version:
                failed_log = _persist_failed_install(stage_dir, runtime_parent)
                raise RuntimeInstallError(
                    "Installed HyperFrames version verification failed: "
                    f"expected {expected_version}, received {installed_version or 'no version'}.",
                    log_path=failed_log,
                )

            npm_return_code, npm_version = _command_version([selected_npm, "--version"])
            metadata = {
                "schema_version": 4,
                "installed_at": datetime.now(timezone.utc).isoformat(),
                "vex_version": __version__,
                "hyperframes_version": expected_version,
                "remotion_version": renderer_versions["remotion"],
                "renderer_versions": renderer_versions,
                "native_runtime_versions": native_status.get("versions") or {},
                "renderer_capabilities": {
                    "hyperframes": True,
                    "remotion": bool(native_status.get("remotion_ready")),
                },
                **identity.to_dict(),
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

    status = installed_runtime_status(node_path=selected_node)
    return {**status, "changed": True}
