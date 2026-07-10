from __future__ import annotations

import subprocess
import sys
import time

import pytest

from vex_runtime.processes import run_streaming_stderr


def test_streaming_process_captures_stderr_and_reports_progress() -> None:
    lines: list[str] = []

    result = run_streaming_stderr(
        [sys.executable, "-c", "import sys; print('frame=1', file=sys.stderr)"],
        timeout_sec=5,
        on_stderr_line=lines.append,
    )

    assert result.returncode == 0
    assert result.stderr.strip() == "frame=1"
    assert [line.strip() for line in lines] == ["frame=1"]


def test_streaming_process_enforces_deadline_and_kills_child() -> None:
    started = time.monotonic()

    with pytest.raises(subprocess.TimeoutExpired) as exc_info:
        run_streaming_stderr(
            [
                sys.executable,
                "-c",
                "import sys,time; print('started', file=sys.stderr, flush=True); time.sleep(10)",
            ],
            timeout_sec=0.2,
        )

    assert time.monotonic() - started < 3.0
    assert "started" in str(exc_info.value.stderr)


def test_streaming_process_terminates_child_when_progress_handler_fails() -> None:
    process = _InterruptingProcess()

    with pytest.raises(RuntimeError, match="progress failed"):
        run_streaming_stderr(
            ["fake"],
            timeout_sec=5,
            on_stderr_line=lambda _line: (_ for _ in ()).throw(RuntimeError("progress failed")),
            popen_factory=lambda *_args, **_kwargs: process,
        )

    assert process.terminated is True


class _InterruptingProcess:
    def __init__(self) -> None:
        self.stderr = iter(["frame=1\n"])
        self.terminated = False

    def poll(self) -> None:
        return None

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def kill(self) -> None:
        self.terminated = True
