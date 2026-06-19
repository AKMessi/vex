from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import config
from engine import probe_video
from vex_runtime.hyperframes import (
    RuntimeInstallError,
    hyperframes_command,
    install_hyperframes_runtime,
    resolve_hyperframes_cli_path,
)


RENDER_ADAPTER_VERSION = "hyperframes-video-render-adapter-v1"


class VideoGenerationRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True)
class CommandRecord:
    name: str
    command: list[str]
    cwd: str
    returncode: int
    log_path: str
    stdout: str = ""
    stderr: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "command": self.command,
            "cwd": self.cwd,
            "returncode": self.returncode,
            "log_path": self.log_path,
            "stdout": self.stdout[-4000:],
            "stderr": self.stderr[-4000:],
        }


class HyperframesVideoRuntime:
    def __init__(self, *, auto_install: bool = True) -> None:
        self.auto_install = auto_install

    def ensure_available(self) -> str:
        cli = resolve_hyperframes_cli_path(config.HYPERFRAMES_CLI_PATH)
        if cli is not None:
            return str(cli)
        if not self.auto_install:
            raise VideoGenerationRuntimeError(
                "HyperFrames CLI is unavailable. Run `vex renderers install hyperframes` "
                "or set HYPERFRAMES_CLI_PATH."
            )
        try:
            install_hyperframes_runtime(force=False)
        except RuntimeInstallError as exc:
            detail = str(exc)
            if exc.log_path:
                detail += f" Install log: {exc.log_path}"
            raise VideoGenerationRuntimeError(detail) from exc
        cli = resolve_hyperframes_cli_path(config.HYPERFRAMES_CLI_PATH)
        if cli is None:
            raise VideoGenerationRuntimeError(
                "HyperFrames runtime installation completed, but the CLI could not be resolved."
            )
        return str(cli)

    def tts(
        self,
        *,
        text_path: Path,
        output_path: Path,
        voice: str,
        speed: float,
        language: str = "",
    ) -> CommandRecord:
        self.ensure_available()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = hyperframes_command(
            "tts",
            str(text_path),
            "--output",
            str(output_path),
            "--voice",
            voice,
            "--speed",
            f"{speed:.3f}",
            configured_cli=config.HYPERFRAMES_CLI_PATH,
        )
        if language:
            command.extend(["--lang", language])
        command.append("--json")
        record = _run_logged(
            command,
            cwd=text_path.parent,
            name="tts",
            log_path=text_path.parent / "logs" / "hyperframes_tts.log",
            timeout=1800,
        )
        if record.returncode != 0 or not output_path.is_file():
            raise VideoGenerationRuntimeError(
                "HyperFrames TTS failed: "
                + (record.stderr.strip() or record.stdout.strip() or f"see {record.log_path}")
            )
        return record

    def transcribe(
        self,
        *,
        media_path: Path,
        project_dir: Path,
        model: str = "small.en",
    ) -> tuple[Path | None, CommandRecord]:
        self.ensure_available()
        command = hyperframes_command(
            "transcribe",
            str(media_path),
            "--dir",
            str(project_dir),
            "--model",
            model,
            "--json",
            configured_cli=config.HYPERFRAMES_CLI_PATH,
        )
        record = _run_logged(
            command,
            cwd=project_dir,
            name="transcribe",
            log_path=project_dir / "logs" / "hyperframes_transcribe.log",
            timeout=max(int(getattr(config, "WHISPER_TRANSCRIBE_TIMEOUT_SEC", 7200)), 60),
        )
        if record.returncode != 0:
            raise VideoGenerationRuntimeError(
                "HyperFrames transcribe failed: "
                + (record.stderr.strip() or record.stdout.strip() or f"see {record.log_path}")
            )
        transcript_path = _find_transcript_path(project_dir, record.stdout)
        return transcript_path, record

    def lint(self, *, project_dir: Path) -> CommandRecord:
        self.ensure_available()
        command = hyperframes_command(
            "lint",
            "--json",
            ".",
            configured_cli=config.HYPERFRAMES_CLI_PATH,
        )
        record = _run_logged(
            command,
            cwd=project_dir,
            name="lint",
            log_path=project_dir / "logs" / "hyperframes_lint.log",
            timeout=int(getattr(config, "HYPERFRAMES_LINT_TIMEOUT_SEC", 90)),
        )
        if record.returncode != 0:
            raise VideoGenerationRuntimeError(
                "HyperFrames lint failed: "
                + (record.stderr.strip() or record.stdout.strip() or f"see {record.log_path}")
            )
        return record

    def render(
        self,
        *,
        project_dir: Path,
        output_path: Path,
        fps: int,
        quality: str,
        output_format: str,
        resolution: str = "",
        workers: str = "",
    ) -> tuple[dict[str, Any], CommandRecord]:
        self.ensure_available()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = hyperframes_command(
            "render",
            "--output",
            str(output_path),
            "--fps",
            str(int(fps)),
            "--quality",
            quality,
            "--format",
            output_format,
            configured_cli=config.HYPERFRAMES_CLI_PATH,
        )
        if resolution:
            command.extend(["--resolution", resolution])
        if workers:
            command.extend(["--workers", workers])
        command.append(".")
        timeout = int(getattr(config, "HYPERFRAMES_RENDER_TIMEOUT_SEC", 0) or 0)
        record = _run_logged(
            command,
            cwd=project_dir,
            name="render",
            log_path=project_dir / "logs" / "hyperframes_render.log",
            timeout=max(timeout, 30) if timeout > 0 else None,
        )
        if record.returncode != 0 or not output_path.is_file():
            raise VideoGenerationRuntimeError(
                "HyperFrames render failed: "
                + (record.stderr.strip() or record.stdout.strip() or f"see {record.log_path}")
            )
        metadata = probe_video(str(output_path))
        return metadata, record


def _run_logged(
    command: list[str],
    *,
    cwd: Path,
    name: str,
    log_path: Path,
    timeout: int | None,
) -> CommandRecord:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log_path.write_text(
            "\n".join(["$ " + " ".join(command), "", str(exc)]),
            encoding="utf-8",
        )
        raise VideoGenerationRuntimeError(f"Could not run HyperFrames {name}: {exc}") from exc
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
    return CommandRecord(
        name=name,
        command=list(command),
        cwd=str(cwd),
        returncode=int(result.returncode),
        log_path=str(log_path),
        stdout=result.stdout or "",
        stderr=result.stderr or "",
    )


def _find_transcript_path(project_dir: Path, stdout: str) -> Path | None:
    candidates = [
        project_dir / "transcript.json",
        project_dir / "transcripts" / "transcript.json",
        project_dir / "audio" / "transcript.json",
        project_dir / "narration.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    try:
        payload = json.loads(stdout)
    except (json.JSONDecodeError, TypeError):
        payload = None
    if isinstance(payload, dict):
        for key in ("transcriptPath", "transcript_path", "output", "path"):
            raw_path = payload.get(key)
            if not raw_path:
                continue
            candidate = Path(str(raw_path))
            if not candidate.is_absolute():
                candidate = project_dir / candidate
            if candidate.is_file():
                return candidate
        if payload.get("words") or payload.get("segments"):
            target = project_dir / "transcript.json"
            target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return target
    for candidate in project_dir.rglob("*.json"):
        if candidate.name.lower() in {"transcript.json", "narration.json"}:
            return candidate
    return None
