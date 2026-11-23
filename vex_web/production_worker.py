"""Own a Studio generation run for the lifetime of its worker process."""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import config
from vex_runtime.locking import exclusive_file_lock
from vex_web.production import ProductionStopped, produce_film, production_workspace


def worker_files(directory: Path, task_id: str) -> tuple[Path, Path, Path]:
    if not re.fullmatch(r"task__studio-production_[a-f0-9]{16}", task_id):
        raise ValueError("Invalid production run ID.")
    root = directory / "workers"
    root.mkdir(exist_ok=True)
    if root.resolve() != root:
        raise ValueError("The production worker directory must be local.")
    paths = tuple(root / f"{task_id}.{suffix}" for suffix in ("lock", "stop", "log"))
    if any(path.resolve() != path for path in paths):
        raise ValueError("The production worker files must stay inside the workspace.")
    return paths


def main(task_id: str, model: str) -> None:
    from vex_web.server import TaskManager, _web_task_from_payload

    config.reload_settings()
    directory = production_workspace()
    lock, stop, _ = worker_files(directory, task_id)
    manager = TaskManager(persist=True, max_workers=1)
    try:
        with exclusive_file_lock(lock, timeout_sec=20):
            row = manager._store.get(directory, task_id)
            if row is None or row["owner_pid"] != os.getpid() or row["payload"]["status"] != "queued":
                raise ValueError("The production run was not assigned to this worker.")
            task = _web_task_from_payload(row["payload"])
        manager._instance_id = task_id

        def progress(stage: str, completed: float) -> None:
            if stop.is_file():
                raise ProductionStopped()
            manager.production_progress(task, stage, completed)

        def work(active):
            progress("Starting production", .02)
            return produce_film(active.task_id, active.request, model, progress)

        manager._run(task, work)
    finally:
        manager.shutdown()
        stop.unlink(missing_ok=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
