from __future__ import annotations

import json
import math
import os
import re
import tempfile
import uuid
import warnings
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vex_runtime.execution_store import ExecutionStore
from vex_runtime.locking import FileLockTimeout, exclusive_file_lock, process_is_running
from vex_runtime.project_catalog import catalog_path


JOB_SCHEMA_VERSION = 1
JOB_ID_RE = re.compile(r"^job_[A-Za-z0-9_-]{8,64}$")
RUNNABLE_STATUSES = {"queued", "failed"}
TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}
JOB_STAGE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,63}$")


ToolExecutor = Callable[[dict[str, Any], Any], dict[str, Any]]


class JobRunnerError(ValueError):
    pass


@dataclass
class JobRecord:
    job_id: str
    project_id: str
    tool_name: str
    params: dict[str, Any]
    status: str
    created_at: str
    updated_at: str
    attempts: int = 0
    started_at: str = ""
    finished_at: str = ""
    pid: int = 0
    message: str = ""
    error: str = ""
    result: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    stage: str = "queued"
    progress: float = 0.0
    schema_version: int = JOB_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def jobs_dir(working_dir: str | Path) -> Path:
    return Path(working_dir) / "jobs"


def job_path(working_dir: str | Path, job_id: str) -> Path:
    normalized = _normalize_job_id(job_id)
    return jobs_dir(working_dir) / f"{normalized}.json"


def create_tool_job(
    state: object,
    tool_name: str,
    params: Mapping[str, Any] | None = None,
    *,
    allowed_tools: set[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> JobRecord:
    normalized_tool = str(tool_name or "").strip()
    if not normalized_tool:
        raise JobRunnerError("Tool name is required.")
    if allowed_tools is not None and normalized_tool not in allowed_tools:
        raise JobRunnerError(f"Unknown job tool: {normalized_tool}")

    now = utc_now_iso()
    record = JobRecord(
        job_id=f"job_{uuid.uuid4().hex[:16]}",
        project_id=str(getattr(state, "project_id", "")),
        tool_name=normalized_tool,
        params=dict(params or {}),
        status="queued",
        created_at=now,
        updated_at=now,
        metadata=dict(metadata or {}),
    )
    write_job(getattr(state, "working_dir"), record)
    return record


def load_job(working_dir: str | Path, job_id: str) -> JobRecord:
    path = job_path(working_dir, job_id)
    if catalog_path(working_dir).is_file():
        row = ExecutionStore().get(working_dir, job_id, kind="tool")
        if row is not None:
            return _coerce_job(row["payload"])
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise JobRunnerError(f"Job not found: {job_id}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise JobRunnerError(f"Unable to read job: {job_id}") from exc
    return _coerce_job(payload)


def list_jobs(working_dir: str | Path, *, limit: int = 25) -> list[JobRecord]:
    records_by_id: dict[str, JobRecord] = {}
    if catalog_path(working_dir).is_file():
        for row in ExecutionStore().list(working_dir, kind="tool", limit=max(limit, 0)):
            record = _coerce_job(row["payload"])
            records_by_id[record.job_id] = record
    for path in jobs_dir(working_dir).glob("job_*.json"):
        try:
            record = _coerce_job(json.loads(path.read_text(encoding="utf-8")))
            records_by_id.setdefault(record.job_id, record)
        except (OSError, json.JSONDecodeError, JobRunnerError):
            continue
    records = list(records_by_id.values())
    records.sort(key=lambda record: record.updated_at, reverse=True)
    return records[: max(int(limit), 0)]


def write_job(working_dir: str | Path, record: JobRecord) -> Path:
    record.schema_version = JOB_SCHEMA_VERSION
    record.updated_at = record.updated_at or utc_now_iso()
    path = job_path(working_dir, record.job_id)
    catalog_backed = catalog_path(working_dir).is_file()
    if catalog_backed:
        ExecutionStore().put(
            working_dir,
            execution_id=record.job_id,
            project_id=record.project_id,
            kind="tool",
            status=record.status,
            payload=record.to_dict(),
            owner_pid=record.pid,
            stage=record.stage,
            progress=record.progress,
        )
    try:
        _atomic_write_json(path, record.to_dict())
    except OSError as exc:
        if not catalog_backed:
            raise
        warnings.warn(
            f"Job {record.job_id} was saved in the project catalog but its JSON export failed: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
    return path


def checkpoint_job(
    working_dir: str | Path,
    job_id: str,
    *,
    stage: str,
    progress: float,
    message: str = "",
) -> JobRecord:
    """Persist a cooperative checkpoint from the process running this job."""
    normalized_stage = str(stage or "").strip()
    if not JOB_STAGE_RE.fullmatch(normalized_stage):
        raise JobRunnerError("Job stage must be a short machine-readable name.")
    try:
        normalized_progress = float(progress)
    except (TypeError, ValueError) as exc:
        raise JobRunnerError("Job progress must be between 0 and 1.") from exc
    if not math.isfinite(normalized_progress) or not 0.0 <= normalized_progress <= 1.0:
        raise JobRunnerError("Job progress must be between 0 and 1.")
    path = job_path(working_dir, job_id)
    try:
        with exclusive_file_lock(_job_lock_path(path)):
            record = load_job(working_dir, job_id)
            if record.status != "running" or record.pid != os.getpid():
                raise JobRunnerError("Only the running job process may checkpoint this job.")
            if normalized_progress < record.progress:
                raise JobRunnerError("Job progress cannot move backward.")
            record.stage = normalized_stage
            record.progress = normalized_progress
            record.message = str(message or record.message)[:2_000]
            record.updated_at = utc_now_iso()
            write_job(working_dir, record)
            return record
    except FileLockTimeout as exc:
        raise JobRunnerError(f"Job {job_id} is being updated by another process.") from exc


def run_tool_job(
    state: object,
    job_id: str,
    executors: Mapping[str, ToolExecutor],
    *,
    force: bool = False,
) -> JobRecord:
    working_dir = getattr(state, "working_dir")
    state_project_id = str(getattr(state, "project_id", ""))
    record = _claim_job(working_dir, job_id, state_project_id=state_project_id, force=force)
    executor = executors.get(record.tool_name)
    if executor is None:
        record.status = "failed"
        record.finished_at = utc_now_iso()
        record.updated_at = record.finished_at
        record.pid = 0
        record.error = f"Unknown job tool: {record.tool_name}"
        record.message = record.error
        record.stage = "failed"
        write_job(working_dir, record)
        raise JobRunnerError(f"Unknown job tool: {record.tool_name}")

    try:
        result = executor(dict(record.params), state)
        success = bool(result.get("success")) if isinstance(result, Mapping) else False
        record.status = "succeeded" if success else "failed"
        record.result = _job_result_payload(result)
        record.message = str(record.result.get("message") or "")
        record.error = "" if success else record.message
        record.stage = "completed" if success else "failed"
        record.progress = 1.0 if success else record.progress
    except Exception as exc:  # noqa: BLE001
        record.status = "failed"
        record.error = str(exc)
        record.message = str(exc)
        record.result = {"success": False, "message": str(exc), "tool_name": record.tool_name}
        record.stage = "failed"
    finally:
        path = job_path(working_dir, job_id)
        with exclusive_file_lock(_job_lock_path(path)):
            latest = load_job(working_dir, job_id)
            if latest.status != "running" or latest.pid != os.getpid():
                raise JobRunnerError(f"Job {job_id} changed owner before completion.")
            record.progress = max(record.progress, latest.progress)
            record.finished_at = utc_now_iso()
            record.updated_at = record.finished_at
            record.pid = 0
            write_job(working_dir, record)
    return record


def _claim_job(
    working_dir: str | Path,
    job_id: str,
    *,
    state_project_id: str,
    force: bool,
) -> JobRecord:
    path = job_path(working_dir, job_id)
    lock_path = _job_lock_path(path)
    try:
        with exclusive_file_lock(lock_path):
            record = load_job(working_dir, job_id)
            if record.project_id and record.project_id != state_project_id:
                raise JobRunnerError(
                    f"Job {record.job_id} belongs to project {record.project_id}."
                )
            if not record.project_id:
                record.project_id = state_project_id
            if record.status == "running":
                if process_is_running(record.pid):
                    raise JobRunnerError(
                        f"Job {record.job_id} is already running in process {record.pid}."
                    )
                if not force:
                    raise JobRunnerError(
                        f"Job {record.job_id} was left running by a stopped process; "
                        "use --force to recover it."
                    )
            elif record.status not in RUNNABLE_STATUSES and not force:
                raise JobRunnerError(
                    f"Job {record.job_id} is {record.status}; use --force to run it again."
                )

            now = utc_now_iso()
            record.status = "running"
            record.started_at = now
            record.finished_at = ""
            record.updated_at = now
            record.attempts += 1
            record.pid = os.getpid()
            record.message = ""
            record.error = ""
            record.result = {}
            record.stage = "executing"
            record.progress = 0.0
            write_job(working_dir, record)
            return record
    except FileLockTimeout as exc:
        raise JobRunnerError(f"Job {job_id} is being claimed by another process.") from exc


def _coerce_job(payload: object) -> JobRecord:
    if not isinstance(payload, Mapping):
        raise JobRunnerError("Invalid job payload.")
    job_id = _normalize_job_id(str(payload.get("job_id") or ""))
    tool_name = str(payload.get("tool_name") or "").strip()
    if not tool_name:
        raise JobRunnerError("Job is missing a tool name.")
    status = str(payload.get("status") or "queued").strip().lower()
    if status not in {"queued", "running", "succeeded", "failed", "cancelled"}:
        status = "failed"
    return JobRecord(
        job_id=job_id,
        project_id=str(payload.get("project_id") or ""),
        tool_name=tool_name,
        params=dict(payload.get("params") or {}) if isinstance(payload.get("params"), Mapping) else {},
        status=status,
        created_at=str(payload.get("created_at") or utc_now_iso()),
        updated_at=str(payload.get("updated_at") or payload.get("created_at") or utc_now_iso()),
        attempts=_coerce_int(payload.get("attempts")),
        started_at=str(payload.get("started_at") or ""),
        finished_at=str(payload.get("finished_at") or ""),
        pid=_coerce_int(payload.get("pid")),
        message=str(payload.get("message") or ""),
        error=str(payload.get("error") or ""),
        result=dict(payload.get("result") or {}) if isinstance(payload.get("result"), Mapping) else {},
        metadata=dict(payload.get("metadata") or {}) if isinstance(payload.get("metadata"), Mapping) else {},
        stage=str(payload.get("stage") or status),
        progress=_coerce_progress(payload.get("progress")),
        schema_version=JOB_SCHEMA_VERSION,
    )


def _job_result_payload(raw: object) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {"success": False, "message": "Tool returned an invalid result."}
    payload = {
        key: value
        for key, value in raw.items()
        if key not in {"updated_state"}
    }
    return _json_safe(payload)


def _json_safe(value: object) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return str(value)


def _normalize_job_id(job_id: str) -> str:
    normalized = str(job_id or "").strip()
    if not JOB_ID_RE.fullmatch(normalized):
        raise JobRunnerError(f"Invalid job id: {job_id!r}")
    return normalized


def _job_lock_path(path: Path) -> Path:
    return path.with_name(f".{path.stem}.claim.lock")


def _coerce_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _coerce_progress(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not 0.0 <= parsed <= 1.0:
        return 0.0
    return parsed


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            json.dump(payload, temp_file, indent=2)
            temp_file.write("\n")
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)
