from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

from vex_runtime.paths import data_dir


WHISPER_DISTRIBUTION = "openai-whisper"
WHISPER_REQUIREMENT = "openai-whisper>=20231117"
WHISPER_INSTALL_TIMEOUT_SEC = 1800


class TranscriptionInstallError(RuntimeError):
    def __init__(self, message: str, *, log_path: Path | None = None) -> None:
        super().__init__(message)
        self.log_path = log_path


def _environment_is_isolated() -> bool:
    return Path(sys.prefix).resolve() != Path(sys.base_prefix).resolve()


def whisper_unavailable_message(error: BaseException | None = None) -> str:
    details = (
        f" Import failed: {type(error).__name__}: {error}."
        if error is not None and str(error).strip()
        else ""
    )
    isolation = (
        " Packages installed into another Python or your system Python are not "
        "visible inside this isolated Vex environment."
        if _environment_is_isolated()
        else ""
    )
    return (
        "Whisper is unavailable in the Python environment running Vex "
        f"({sys.executable}).{isolation}{details} "
        "Run `vex setup transcription` to install it here, or set "
        "`WHISPER_PYTHON_PATH` to another Python executable that can import Whisper."
    )


def load_whisper() -> ModuleType:
    try:
        return importlib.import_module("whisper")
    except Exception as exc:
        raise TranscriptionInstallError(whisper_unavailable_message(exc)) from exc


def _install_log_path() -> Path:
    path = data_dir() / "logs" / "transcription-install.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _worker_log_path() -> Path:
    path = data_dir() / "logs" / "transcription-worker.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_install_log(
    path: Path,
    command: list[str],
    result: subprocess.CompletedProcess[str],
) -> None:
    path.write_text(
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


def install_transcription_dependencies(
    *,
    force: bool = False,
    configured_python: str = "",
) -> dict[str, object]:
    if not force:
        try:
            load_whisper()
        except TranscriptionInstallError:
            external = discover_external_whisper_python(configured_python)
            if external is not None:
                return {
                    "changed": False,
                    "runtime": "external",
                    "python_executable": external["python_executable"],
                    "version": external["version"],
                    "log_path": None,
                }
        else:
            return {
                "changed": False,
                "runtime": "current",
                "python_executable": sys.executable,
                "version": importlib.metadata.version(WHISPER_DISTRIBUTION),
                "log_path": None,
            }

    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        WHISPER_REQUIREMENT,
    ]
    if force:
        command.insert(-1, "--force-reinstall")
    log_path = _install_log_path()
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=WHISPER_INSTALL_TIMEOUT_SEC,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TranscriptionInstallError(
            f"Whisper installation exceeded {WHISPER_INSTALL_TIMEOUT_SEC} seconds.",
            log_path=log_path,
        ) from exc
    except OSError as exc:
        raise TranscriptionInstallError(
            f"Could not start pip with {sys.executable}: {exc}",
            log_path=log_path,
        ) from exc
    _write_install_log(log_path, command, result)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip().splitlines()
        summary = detail[-1] if detail else f"pip exited with code {result.returncode}"
        raise TranscriptionInstallError(
            f"Whisper installation failed: {summary}",
            log_path=log_path,
        )

    importlib.invalidate_caches()
    try:
        load_whisper()
        version = importlib.metadata.version(WHISPER_DISTRIBUTION)
    except (TranscriptionInstallError, importlib.metadata.PackageNotFoundError) as exc:
        raise TranscriptionInstallError(
            f"Whisper installed but could not be verified in {sys.executable}: {exc}",
            log_path=log_path,
        ) from exc
    return {
        "changed": True,
        "runtime": "current",
        "python_executable": sys.executable,
        "version": version,
        "log_path": str(log_path),
    }


def _python_candidates(configured_python: str = "") -> list[str]:
    candidates: list[str] = []
    for candidate in (
        configured_python,
        shutil.which("python") or "",
        shutil.which("python3") or "",
    ):
        value = str(candidate or "").strip()
        if not value:
            continue
        try:
            resolved = str(Path(value).expanduser().resolve(strict=True))
        except OSError:
            continue
        if os.path.normcase(resolved) == os.path.normcase(
            str(Path(sys.executable).resolve(strict=False))
        ):
            continue
        if resolved not in candidates:
            candidates.append(resolved)
    return candidates


def discover_external_whisper_python(configured_python: str = "") -> dict[str, str] | None:
    probe = (
        "import importlib.metadata as m, importlib.util as u; "
        "assert u.find_spec('whisper') is not None; "
        f"print(m.version({WHISPER_DISTRIBUTION!r}))"
    )
    for candidate in _python_candidates(configured_python):
        try:
            result = subprocess.run(
                [candidate, "-c", probe],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0:
            version = (result.stdout or "").strip().splitlines()
            return {
                "python_executable": candidate,
                "version": version[-1] if version else "unknown",
            }
    return None


def _write_worker_log(
    path: Path,
    command: list[str],
    *,
    returncode: int | None,
    stdout: str,
    stderr: str,
) -> None:
    path.write_text(
        "\n".join(
            [
                "$ " + " ".join(command),
                "",
                f"exit_code={returncode if returncode is not None else 'timeout'}",
                "",
                "[stdout]",
                stdout,
                "",
                "[stderr]",
                stderr,
            ]
        ),
        encoding="utf-8",
    )


def transcribe_with_whisper(
    media_path: str | Path,
    *,
    model_name: str,
    configured_python: str = "",
    timeout_sec: int = 7200,
) -> dict[str, Any]:
    try:
        whisper = load_whisper()
    except TranscriptionInstallError as local_error:
        external = discover_external_whisper_python(configured_python)
        if external is None:
            raise TranscriptionInstallError(
                whisper_unavailable_message(local_error.__cause__ or local_error)
            ) from local_error
    else:
        model = whisper.load_model(model_name)
        try:
            return model.transcribe(
                str(media_path),
                word_timestamps=True,
                verbose=False,
            )
        except TypeError:
            return model.transcribe(str(media_path))

    jobs_root = data_dir() / "transcription_jobs"
    jobs_root.mkdir(parents=True, exist_ok=True)
    worker_path = Path(__file__).with_name("whisper_worker.py")
    log_path = _worker_log_path()
    with tempfile.TemporaryDirectory(prefix="whisper-", dir=jobs_root) as job_dir:
        result_path = Path(job_dir) / "result.json"
        command = [
            external["python_executable"],
            str(worker_path),
            "--input",
            str(Path(media_path).expanduser().resolve(strict=False)),
            "--model",
            model_name,
            "--output",
            str(result_path),
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=max(30, int(timeout_sec)),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            _write_worker_log(
                log_path,
                command,
                returncode=None,
                stdout=str(exc.stdout or ""),
                stderr=str(exc.stderr or ""),
            )
            raise TranscriptionInstallError(
                f"External Whisper transcription exceeded {timeout_sec} seconds.",
                log_path=log_path,
            ) from exc
        except OSError as exc:
            _write_worker_log(
                log_path,
                command,
                returncode=None,
                stdout="",
                stderr=str(exc),
            )
            raise TranscriptionInstallError(
                f"Could not start external Whisper transcription: {exc}",
                log_path=log_path,
            ) from exc
        _write_worker_log(
            log_path,
            command,
            returncode=result.returncode,
            stdout=result.stdout or "",
            stderr=result.stderr or "",
        )
        if result.returncode != 0 or not result_path.is_file():
            detail = (result.stderr or result.stdout or "").strip().splitlines()
            summary = detail[-1] if detail else "the worker produced no result"
            raise TranscriptionInstallError(
                f"External Whisper transcription failed: {summary}",
                log_path=log_path,
            )
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TranscriptionInstallError(
                f"External Whisper returned an invalid result: {exc}",
                log_path=log_path,
            ) from exc
        if not isinstance(payload, dict):
            raise TranscriptionInstallError(
                "External Whisper returned a non-object result.",
                log_path=log_path,
            )
        return payload
