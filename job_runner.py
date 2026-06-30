from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


JOB_SCHEMA_VERSION = 1
JOB_ID_RE = re.compile(r"^job_[A-Za-z0-9_-]{8,64}$")
RUNNABLE_STATUSES = {"queued", "failed"}
TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}


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
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise JobRunnerError(f"Job not found: {job_id}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise JobRunnerError(f"Unable to read job: {job_id}") from exc
    return _coerce_job(payload)


def list_jobs(working_dir: str | Path, *, limit: int = 25) -> list[JobRecord]:
    records: list[JobRecord] = []
    for path in jobs_dir(working_dir).glob("job_*.json"):
        try:
            records.append(_coerce_job(json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, json.JSONDecodeError, JobRunnerError):
            continue
    records.sort(key=lambda record: record.updated_at, reverse=True)
    return records[: max(int(limit), 0)]


def write_job(working_dir: str | Path, record: JobRecord) -> Path:
    record.schema_version = JOB_SCHEMA_VERSION
    record.updated_at = record.updated_at or utc_now_iso()
    path = job_path(working_dir, record.job_id)
    _atomic_write_json(path, record.to_dict())
    return path


def run_tool_job(
    state: object,
    job_id: str,
    executors: Mapping[str, ToolExecutor],
    *,
    force: bool = False,
) -> JobRecord:
    record = load_job(getattr(state, "working_dir"), job_id)
    state_project_id = str(getattr(state, "project_id", ""))
    if record.project_id and record.project_id != state_project_id:
        raise JobRunnerError(f"Job {record.job_id} belongs to project {record.project_id}.")
    if record.status not in RUNNABLE_STATUSES and not force:
        raise JobRunnerError(f"Job {record.job_id} is {record.status}; use --force to run it again.")
    executor = executors.get(record.tool_name)
    if executor is None:
        raise JobRunnerError(f"Unknown job tool: {record.tool_name}")

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
    write_job(getattr(state, "working_dir"), record)

    try:
        result = executor(dict(record.params), state)
        success = bool(result.get("success")) if isinstance(result, Mapping) else False
        record.status = "succeeded" if success else "failed"
        record.result = _job_result_payload(result)
        record.message = str(record.result.get("message") or "")
        record.error = "" if success else record.message
    except Exception as exc:  # noqa: BLE001
        record.status = "failed"
        record.error = str(exc)
        record.message = str(exc)
        record.result = {"success": False, "message": str(exc), "tool_name": record.tool_name}
    finally:
        record.finished_at = utc_now_iso()
        record.updated_at = record.finished_at
        record.pid = 0
        write_job(getattr(state, "working_dir"), record)
    return record


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
    if value is None or isinstance(value, (str, int, float, bool)):
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


def _coerce_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


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
