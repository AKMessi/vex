from __future__ import annotations

import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class FileLockTimeout(TimeoutError):
    pass


@contextmanager
def exclusive_file_lock(
    path: str | Path,
    *,
    timeout_sec: float = 5.0,
    stale_after_sec: float = 30.0,
) -> Iterator[None]:
    """Serialize filesystem work without requiring a third-party lock package."""

    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}:{uuid.uuid4().hex}"
    deadline = time.monotonic() + max(float(timeout_sec), 0.0)

    while True:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            try:
                age_sec = max(0.0, time.time() - lock_path.stat().st_mtime)
            except FileNotFoundError:
                continue
            except OSError:
                age_sec = 0.0
            if stale_after_sec > 0 and age_sec >= stale_after_sec:
                owner_pid = _lock_owner_pid(lock_path)
                if not process_is_running(owner_pid):
                    try:
                        lock_path.unlink()
                    except FileNotFoundError:
                        pass
                    except OSError:
                        pass
                    else:
                        continue
            if time.monotonic() >= deadline:
                raise FileLockTimeout(f"Timed out waiting for file lock: {lock_path}") from None
            time.sleep(0.05)
            continue

        try:
            with os.fdopen(descriptor, "w", encoding="ascii") as lock_file:
                lock_file.write(token)
                lock_file.flush()
                os.fsync(lock_file.fileno())
        except BaseException:
            lock_path.unlink(missing_ok=True)
            raise
        break

    try:
        yield
    finally:
        try:
            owner = lock_path.read_text(encoding="ascii")
        except (FileNotFoundError, OSError, UnicodeError):
            owner = ""
        if owner == token:
            lock_path.unlink(missing_ok=True)


def _lock_owner_pid(lock_path: Path) -> int:
    try:
        owner = lock_path.read_text(encoding="ascii").strip()
        return int(owner.partition(":")[0])
    except (FileNotFoundError, OSError, UnicodeError, ValueError):
        return 0


def process_is_running(pid: int) -> bool:
    process_id = int(pid)
    if process_id <= 0:
        return False
    if process_id == os.getpid():
        return True

    if os.name == "nt":
        return _windows_process_is_running(process_id)

    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _windows_process_is_running(pid: int) -> bool:
    import ctypes

    process_query_limited_information = 0x1000
    still_active = 259
    error_access_denied = 5

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    kernel32.GetExitCodeProcess.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int

    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        return ctypes.get_last_error() == error_access_denied
    try:
        exit_code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return True
        return exit_code.value == still_active
    finally:
        kernel32.CloseHandle(handle)
