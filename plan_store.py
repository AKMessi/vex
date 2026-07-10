from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from edit_plan import EditPlan, ToolStep
from vex_runtime.locking import FileLockTimeout, exclusive_file_lock, process_is_running


PLAN_STORE_SCHEMA_VERSION = 2
PLAN_ID_RE = re.compile(r"^plan_[A-Za-z0-9_-]{8,64}$")


class PlanStoreError(ValueError):
    pass


@dataclass
class PlanRecord:
    plan_id: str
    project_id: str
    instruction: str
    plan: dict[str, Any]
    status: str
    created_at: str
    updated_at: str
    applied_at: str = ""
    results: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""
    pid: int = 0
    schema_version: int = PLAN_STORE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def plans_dir(working_dir: str | Path) -> Path:
    return Path(working_dir) / "plans"


def plan_path(working_dir: str | Path, plan_id: str) -> Path:
    return plans_dir(working_dir) / f"{_normalize_plan_id(plan_id)}.json"


def create_plan_record(state: object, instruction: str, plan: EditPlan) -> PlanRecord:
    now = utc_now_iso()
    record = PlanRecord(
        plan_id=f"plan_{uuid.uuid4().hex[:16]}",
        project_id=str(getattr(state, "project_id", "")),
        instruction=str(instruction or ""),
        plan=plan.to_dict(),
        status="planned",
        created_at=now,
        updated_at=now,
    )
    write_plan_record(getattr(state, "working_dir"), record)
    return record


def load_plan_record(working_dir: str | Path, plan_id: str) -> PlanRecord:
    path = plan_path(working_dir, plan_id)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PlanStoreError(f"Plan not found: {plan_id}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise PlanStoreError(f"Unable to read plan: {plan_id}") from exc
    return _coerce_plan_record(payload)


def list_plan_records(working_dir: str | Path, *, limit: int = 25) -> list[PlanRecord]:
    records: list[PlanRecord] = []
    for path in plans_dir(working_dir).glob("plan_*.json"):
        try:
            records.append(_coerce_plan_record(json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, json.JSONDecodeError, PlanStoreError):
            continue
    records.sort(key=lambda record: record.updated_at, reverse=True)
    return records[: max(int(limit), 0)]


def write_plan_record(working_dir: str | Path, record: PlanRecord) -> Path:
    record.schema_version = PLAN_STORE_SCHEMA_VERSION
    record.updated_at = record.updated_at or utc_now_iso()
    path = plan_path(working_dir, record.plan_id)
    _atomic_write_json(path, record.to_dict())
    return path


def edit_plan_from_record(record: PlanRecord) -> EditPlan:
    payload = dict(record.plan or {})
    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise PlanStoreError(f"Plan {record.plan_id} does not contain executable steps.")
    steps: list[ToolStep] = []
    for raw_step in raw_steps:
        if not isinstance(raw_step, Mapping):
            raise PlanStoreError(f"Plan {record.plan_id} contains an invalid step.")
        tool = str(raw_step.get("tool") or "").strip()
        if not tool:
            raise PlanStoreError(f"Plan {record.plan_id} contains a step without a tool.")
        params = raw_step.get("params")
        steps.append(
            ToolStep(
                tool=tool,
                params=dict(params or {}) if isinstance(params, Mapping) else {},
                label=str(raw_step.get("label") or ""),
            )
        )
    return EditPlan(
        steps=steps,
        source=str(payload.get("source") or "stored_plan"),
        confidence=_coerce_float(payload.get("confidence"), default=1.0),
        reason=str(payload.get("reason") or ""),
        requires_llm=bool(payload.get("requires_llm", False)),
        can_run_async=bool(payload.get("can_run_async", False)),
        final_response_mode=str(payload.get("final_response_mode") or "tool_summary"),
    )


def mark_plan_record(
    working_dir: str | Path,
    record: PlanRecord,
    *,
    status: str,
    results: list[dict[str, Any]] | None = None,
    error: str = "",
) -> PlanRecord:
    now = utc_now_iso()
    normalized_status = str(status or "").strip().lower()
    if normalized_status not in {"planned", "applying", "applied", "failed"}:
        raise PlanStoreError(f"Invalid plan status: {status}")
    record.status = normalized_status
    record.updated_at = now
    record.error = str(error or "")
    if results is not None:
        record.results = [_json_safe(result) for result in results]
    if normalized_status in {"applied", "failed"}:
        record.applied_at = now
        record.pid = 0
    write_plan_record(working_dir, record)
    return record


def claim_plan_record(
    working_dir: str | Path,
    plan_id: str,
    *,
    state_project_id: str,
    force: bool = False,
) -> PlanRecord:
    path = plan_path(working_dir, plan_id)
    lock_path = path.with_name(f".{path.stem}.claim.lock")
    try:
        with exclusive_file_lock(lock_path):
            record = load_plan_record(working_dir, plan_id)
            if record.project_id and record.project_id != state_project_id:
                raise PlanStoreError(
                    f"Plan {record.plan_id} belongs to project {record.project_id}."
                )
            if record.status == "applying":
                if process_is_running(record.pid):
                    raise PlanStoreError(
                        f"Plan {record.plan_id} is already being applied in process {record.pid}."
                    )
                if not force:
                    raise PlanStoreError(
                        f"Plan {record.plan_id} was left applying by a stopped process; "
                        "use --force to recover it."
                    )
            elif record.status == "applied" and not force:
                raise PlanStoreError(
                    f"Plan {record.plan_id} is already applied; use --force to apply it again."
                )

            edit_plan_from_record(record)
            now = utc_now_iso()
            record.status = "applying"
            record.updated_at = now
            record.applied_at = ""
            record.results = []
            record.error = ""
            record.pid = os.getpid()
            write_plan_record(working_dir, record)
            return record
    except FileLockTimeout as exc:
        raise PlanStoreError(f"Plan {plan_id} is being claimed by another process.") from exc


def sanitize_plan_result(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): _json_safe(value)
        for key, value in dict(result or {}).items()
        if key not in {"updated_state"}
    }


def _coerce_plan_record(payload: object) -> PlanRecord:
    if not isinstance(payload, Mapping):
        raise PlanStoreError("Invalid plan payload.")
    plan_id = _normalize_plan_id(str(payload.get("plan_id") or ""))
    plan = payload.get("plan")
    if not isinstance(plan, Mapping):
        raise PlanStoreError(f"Plan {plan_id} is missing a plan payload.")
    status = str(payload.get("status") or "planned").strip().lower()
    if status not in {"planned", "applying", "applied", "failed"}:
        status = "failed"
    return PlanRecord(
        plan_id=plan_id,
        project_id=str(payload.get("project_id") or ""),
        instruction=str(payload.get("instruction") or ""),
        plan=dict(plan),
        status=status,
        created_at=str(payload.get("created_at") or utc_now_iso()),
        updated_at=str(payload.get("updated_at") or payload.get("created_at") or utc_now_iso()),
        applied_at=str(payload.get("applied_at") or ""),
        results=[
            dict(item)
            for item in payload.get("results") or []
            if isinstance(item, Mapping)
        ],
        error=str(payload.get("error") or ""),
        pid=_coerce_int(payload.get("pid")),
        schema_version=PLAN_STORE_SCHEMA_VERSION,
    )


def _normalize_plan_id(plan_id: str) -> str:
    normalized = str(plan_id or "").strip()
    if not PLAN_ID_RE.fullmatch(normalized):
        raise PlanStoreError(f"Invalid plan id: {plan_id!r}")
    return normalized


def _coerce_float(value: object, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


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
