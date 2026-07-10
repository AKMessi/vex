from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

import vex_runtime.locking as locking


def test_stale_lock_is_not_removed_while_owner_process_is_alive(tmp_path: Path) -> None:
    lock_path = tmp_path / "live.lock"
    lock_path.write_text(f"{os.getpid()}:existing", encoding="ascii")
    old = time.time() - 60
    os.utime(lock_path, (old, old))

    with pytest.raises(locking.FileLockTimeout):
        with locking.exclusive_file_lock(
            lock_path,
            timeout_sec=0.05,
            stale_after_sec=0.01,
        ):
            pass

    assert lock_path.exists()


def test_stale_lock_is_recovered_after_owner_process_stops(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "dead.lock"
    lock_path.write_text("999999:existing", encoding="ascii")
    old = time.time() - 60
    os.utime(lock_path, (old, old))
    monkeypatch.setattr(locking, "process_is_running", lambda _pid: False)

    with locking.exclusive_file_lock(
        lock_path,
        timeout_sec=0.2,
        stale_after_sec=0.01,
    ):
        assert lock_path.exists()

    assert not lock_path.exists()
