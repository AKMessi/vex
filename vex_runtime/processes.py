from __future__ import annotations

import subprocess
import threading
from dataclasses import dataclass
from typing import Callable, Sequence


@dataclass(frozen=True)
class StreamingProcessResult:
    returncode: int
    stderr: str


def run_streaming_stderr(
    command: Sequence[str],
    *,
    timeout_sec: float | None,
    on_stderr_line: Callable[[str], None] | None = None,
    stderr_line_limit: int | None = None,
    popen_factory: Callable[..., subprocess.Popen[str]] | None = None,
) -> StreamingProcessResult:
    """Run a process with streamed stderr, a hard deadline, and interruption cleanup."""

    factory = popen_factory or subprocess.Popen
    process = factory(
        list(command),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.stderr is None:
        _stop_process(process)
        raise RuntimeError("Failed to capture process stderr.")

    timed_out = threading.Event()

    def kill_after_deadline() -> None:
        timed_out.set()
        try:
            process.kill()
        except (OSError, AttributeError):
            pass

    timer: threading.Timer | None = None
    if timeout_sec is not None and float(timeout_sec) > 0:
        timer = threading.Timer(float(timeout_sec), kill_after_deadline)
        timer.daemon = True
        timer.start()

    stderr_lines: list[str] = []
    try:
        for line in process.stderr:
            stderr_lines.append(line)
            if stderr_line_limit is not None and len(stderr_lines) > stderr_line_limit:
                del stderr_lines[: len(stderr_lines) - stderr_line_limit]
            if on_stderr_line is not None:
                on_stderr_line(line)
        returncode = process.wait()
    except BaseException:
        _stop_process(process)
        raise
    finally:
        if timer is not None:
            timer.cancel()
        close_stderr = getattr(process.stderr, "close", None)
        if callable(close_stderr):
            close_stderr()

    stderr_text = "".join(stderr_lines)
    if timed_out.is_set():
        raise subprocess.TimeoutExpired(
            list(command),
            timeout_sec,
            stderr=stderr_text,
        )
    return StreamingProcessResult(returncode=returncode, stderr=stderr_text)


def _stop_process(process: subprocess.Popen[str]) -> None:
    poll = getattr(process, "poll", None)
    try:
        running = not callable(poll) or poll() is None
    except OSError:
        running = True
    if not running:
        return
    try:
        process.terminate()
    except (OSError, AttributeError):
        return
    try:
        process.wait(timeout=2)
        return
    except (OSError, subprocess.TimeoutExpired, TypeError):
        pass
    try:
        process.kill()
    except (OSError, AttributeError):
        return
    try:
        process.wait(timeout=2)
    except (OSError, subprocess.TimeoutExpired, TypeError):
        pass
