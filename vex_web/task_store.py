"""Studio adapter over the project execution ledger."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vex_runtime.execution_store import ExecutionStore


class StudioTaskStore:
    def __init__(self) -> None:
        self._executions = ExecutionStore()

    def put(
        self,
        working_dir: str | Path,
        payload: dict[str, Any],
        *,
        owner_pid: int,
        owner_instance: str,
    ) -> None:
        self._executions.put(
            working_dir,
            execution_id=str(payload["task_id"]),
            project_id=str(payload["project_id"]),
            kind="studio",
            status=str(payload["status"]),
            payload=payload,
            owner_pid=owner_pid,
            owner_instance=owner_instance,
            stage=str(payload.get("stage") or payload["status"]),
            progress=float(payload.get("progress") or 0.0),
        )

    def get(self, working_dir: str | Path, task_id: str) -> dict[str, Any] | None:
        return self._executions.get(working_dir, task_id, kind="studio")

    def active(self, working_dir: str | Path) -> dict[str, Any] | None:
        return self._executions.active(working_dir, kind="studio")

    def latest(self, working_dir: str | Path) -> dict[str, Any] | None:
        return self._executions.latest(working_dir, kind="studio")

    def prune(self, working_dir: str | Path, *, retention_seconds: int, max_tasks: int) -> None:
        self._executions.prune(
            working_dir,
            kind="studio",
            retention_seconds=retention_seconds,
            max_records=max_tasks,
        )
