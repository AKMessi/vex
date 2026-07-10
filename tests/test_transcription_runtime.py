from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from state import ProjectState, utc_now_iso
from tools import transcript
from vex_runtime import transcription


def test_missing_whisper_error_names_active_environment_and_setup_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def missing_module(_name: str) -> ModuleType:
        raise ModuleNotFoundError("No module named 'whisper'")

    monkeypatch.setattr(transcription.importlib, "import_module", missing_module)
    monkeypatch.setattr(transcription.sys, "executable", str(tmp_path / "venv" / "python"))
    monkeypatch.setattr(transcription.sys, "prefix", str(tmp_path / "venv"))
    monkeypatch.setattr(transcription.sys, "base_prefix", str(tmp_path / "python"))
    monkeypatch.setattr(
        transcription,
        "discover_external_whisper_python",
        lambda _configured="": None,
    )

    result = transcript.execute({}, _state(tmp_path))

    assert result["success"] is False
    assert str(tmp_path / "venv" / "python") in result["message"]
    assert "another Python" in result["message"]
    assert "`vex setup transcription`" in result["message"]


def test_external_whisper_discovery_uses_configured_interpreter_without_importing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    external_python = tmp_path / "system-python"
    external_python.write_text("python", encoding="utf-8")
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):  # noqa: ANN001
        commands.append(list(command))
        return SimpleNamespace(
            returncode=0,
            stdout="20250625\n",
            stderr="",
        )

    monkeypatch.setattr(transcription.subprocess, "run", fake_run)

    result = transcription.discover_external_whisper_python(str(external_python))

    assert result == {
        "python_executable": str(external_python.resolve()),
        "version": "20250625",
    }
    assert commands[0][0] == str(external_python.resolve())
    assert "find_spec('whisper')" in commands[0][2]


def test_transcription_uses_external_worker_when_pipx_environment_lacks_whisper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    media_path = tmp_path / "source.wav"
    media_path.write_bytes(b"audio")
    worker_log = tmp_path / "worker.log"

    monkeypatch.setattr(
        transcription,
        "load_whisper",
        lambda: (_ for _ in ()).throw(
            transcription.TranscriptionInstallError("missing")
        ),
    )
    monkeypatch.setattr(
        transcription,
        "discover_external_whisper_python",
        lambda _configured="": {
            "python_executable": str(tmp_path / "system-python"),
            "version": "20250625",
        },
    )
    monkeypatch.setattr(transcription, "data_dir", lambda: tmp_path)
    monkeypatch.setattr(transcription, "_worker_log_path", lambda: worker_log)

    def fake_run(command, **_kwargs):  # noqa: ANN001
        output_path = Path(command[command.index("--output") + 1])
        output_path.write_text(
            json.dumps(
                {
                    "text": "hello world",
                    "segments": [
                        {
                            "start": 0.0,
                            "end": 1.0,
                            "text": "hello world",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(transcription.subprocess, "run", fake_run)

    result = transcription.transcribe_with_whisper(
        media_path,
        model_name="tiny",
        configured_python=str(tmp_path / "system-python"),
        timeout_sec=60,
    )

    assert result["text"] == "hello world"
    assert result["segments"][0]["end"] == 1.0
    assert worker_log.is_file()


def test_transcription_installer_targets_running_python_and_verifies_import(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    running_python = tmp_path / "pipx" / "venvs" / "vex-video" / "python"
    whisper_module = ModuleType("whisper")
    import_attempts = 0

    def import_module(_name: str) -> ModuleType:
        nonlocal import_attempts
        import_attempts += 1
        if import_attempts == 1:
            raise ModuleNotFoundError("No module named 'whisper'")
        return whisper_module

    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):  # noqa: ANN001
        commands.append(list(command))
        return SimpleNamespace(returncode=0, stdout="installed", stderr="")

    monkeypatch.setattr(transcription.sys, "executable", str(running_python))
    monkeypatch.setattr(transcription.importlib, "import_module", import_module)
    monkeypatch.setattr(transcription.importlib, "invalidate_caches", lambda: None)
    monkeypatch.setattr(transcription, "_python_candidates", lambda _configured="": [])
    monkeypatch.setattr(
        transcription.importlib.metadata,
        "version",
        lambda _name: "20250625",
    )
    monkeypatch.setattr(transcription.subprocess, "run", fake_run)
    monkeypatch.setattr(
        transcription,
        "_install_log_path",
        lambda: tmp_path / "transcription-install.log",
    )

    result = transcription.install_transcription_dependencies()

    assert commands == [
        [
            str(running_python),
            "-m",
            "pip",
            "install",
            "--upgrade",
            transcription.WHISPER_REQUIREMENT,
        ]
    ]
    assert result["changed"] is True
    assert result["runtime"] == "current"
    assert result["version"] == "20250625"
    assert (tmp_path / "transcription-install.log").is_file()


def test_transcription_installer_enforces_install_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        transcription.importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(ModuleNotFoundError("missing")),
    )
    monkeypatch.setattr(transcription, "_python_candidates", lambda _configured="": [])
    monkeypatch.setattr(transcription, "_install_log_path", lambda: tmp_path / "install.log")

    def time_out(command, **kwargs):  # noqa: ANN001
        assert kwargs["timeout"] == transcription.WHISPER_INSTALL_TIMEOUT_SEC
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(transcription.subprocess, "run", time_out)

    with pytest.raises(transcription.TranscriptionInstallError, match="installation exceeded"):
        transcription.install_transcription_dependencies()


def test_external_whisper_launch_failure_uses_runtime_error_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    media_path = tmp_path / "source.wav"
    media_path.write_bytes(b"audio")
    worker_log = tmp_path / "worker.log"
    monkeypatch.setattr(
        transcription,
        "load_whisper",
        lambda: (_ for _ in ()).throw(transcription.TranscriptionInstallError("missing")),
    )
    monkeypatch.setattr(
        transcription,
        "discover_external_whisper_python",
        lambda _configured="": {
            "python_executable": str(tmp_path / "system-python"),
            "version": "20250625",
        },
    )
    monkeypatch.setattr(transcription, "data_dir", lambda: tmp_path)
    monkeypatch.setattr(transcription, "_worker_log_path", lambda: worker_log)
    monkeypatch.setattr(
        transcription.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("worker missing")),
    )

    with pytest.raises(
        transcription.TranscriptionInstallError,
        match="Could not start external Whisper transcription",
    ):
        transcription.transcribe_with_whisper(
            media_path,
            model_name="tiny",
            configured_python=str(tmp_path / "system-python"),
            timeout_sec=60,
        )

    assert "worker missing" in worker_log.read_text(encoding="utf-8")


def test_transcription_installer_skips_pip_when_whisper_is_importable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        transcription.importlib,
        "import_module",
        lambda _name: ModuleType("whisper"),
    )
    monkeypatch.setattr(
        transcription.importlib.metadata,
        "version",
        lambda _name: "20250625",
    )
    monkeypatch.setattr(
        transcription.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("pip should not run"),
    )

    result = transcription.install_transcription_dependencies()

    assert result["changed"] is False
    assert result["runtime"] == "current"
    assert result["version"] == "20250625"


def test_transcription_setup_reuses_discoverable_external_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        transcription,
        "load_whisper",
        lambda: (_ for _ in ()).throw(
            transcription.TranscriptionInstallError("missing")
        ),
    )
    monkeypatch.setattr(
        transcription,
        "discover_external_whisper_python",
        lambda _configured="": {
            "python_executable": "C:/Python314/python.exe",
            "version": "20250625",
        },
    )
    monkeypatch.setattr(
        transcription.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("pip should not run"),
    )

    result = transcription.install_transcription_dependencies()

    assert result == {
        "changed": False,
        "runtime": "external",
        "python_executable": "C:/Python314/python.exe",
        "version": "20250625",
        "log_path": None,
    }


def _state(tmp_path: Path) -> ProjectState:
    now = utc_now_iso()
    return ProjectState(
        project_id="transcription-test",
        project_name="Transcription Test",
        created_at=now,
        updated_at=now,
        source_files=[str(tmp_path / "source.mp4")],
        working_file=str(tmp_path / "working.mp4"),
        working_dir=str(tmp_path),
        output_dir=str(tmp_path / "output"),
        metadata={"duration_sec": 1.0, "width": 1280, "height": 720, "fps": 30.0},
        provider="test",
        model="test",
    )
