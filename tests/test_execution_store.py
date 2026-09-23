from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

import config
import job_runner
from job_runner import JobRunnerError, checkpoint_job, create_tool_job, list_jobs, load_job, run_tool_job
from state import ProjectState, utc_now_iso
from vex_runtime.execution_store import ExecutionConflict, ExecutionStore
from vex_runtime.project_catalog import ProjectCatalogError, catalog_path
from vex_web.server import _project_detail
from vex_web.task_store import StudioTaskStore


def _project(tmp_path: Path) -> ProjectState:
    now = utc_now_iso()
    state = ProjectState(
        project_id="project-1",
        project_name="Test",
        created_at=now,
        updated_at=now,
        source_files=[str(tmp_path / "source.mp4")],
        working_file=str(tmp_path / "source.mp4"),
        working_dir=str(tmp_path),
        output_dir=str(tmp_path / "output"),
    )
    return state


def test_catalog_is_authoritative_for_cli_jobs(tmp_path: Path) -> None:
    state = _project(tmp_path)
    state.save()
    job = create_tool_job(state, "sample_tool", {"value": 2})
    assert ExecutionStore().get(tmp_path, job.job_id, kind="tool")["status"] == "queued"

    projection = tmp_path / "jobs" / f"{job.job_id}.json"
    projection.write_text('{"status":"succeeded"}', encoding="utf-8")
    assert load_job(tmp_path, job.job_id).status == "queued"
    projection.unlink()
    assert load_job(tmp_path, job.job_id).params == {"value": 2}
    assert [record.job_id for record in list_jobs(tmp_path)] == [job.job_id]
    assert _project_detail(state)["jobs"][0]["job_id"] == job.job_id


def test_legacy_cli_job_imports_on_first_write(tmp_path: Path) -> None:
    state = _project(tmp_path)
    job = create_tool_job(state, "sample_tool", {"value": 3})
    assert not catalog_path(tmp_path).exists()
    state.save()

    assert ExecutionStore().get(tmp_path, job.job_id, kind="tool") is None
    result = run_tool_job(
        state,
        job.job_id,
        {"sample_tool": lambda params, _state: {"success": True, "message": str(params["value"])}},
    )
    assert result.status == "succeeded"
    assert result.stage == "completed"
    assert result.progress == 1.0
    assert ExecutionStore().get(tmp_path, job.job_id, kind="tool")["status"] == "succeeded"


def test_execution_ledger_separates_studio_and_tool_records(tmp_path: Path) -> None:
    state = _project(tmp_path)
    state.save()
    job = create_tool_job(state, "sample_tool")
    studio_payload = {
        "task_id": "task_project-1_1234567890abcdef",
        "project_id": "project-1",
        "status": "running",
        "kind": "chat",
        "label": "Editing",
    }
    StudioTaskStore().put(tmp_path, studio_payload, owner_pid=999_999, owner_instance="old")

    store = ExecutionStore()
    assert store.active(tmp_path, kind="studio")["execution_id"] == studio_payload["task_id"]
    assert store.active(tmp_path, kind="tool")["execution_id"] == job.job_id
    assert {row["kind"] for row in (store.latest(tmp_path, kind="studio"), store.latest(tmp_path, kind="tool"))} == {"studio", "tool"}


def test_old_studio_table_is_imported_into_shared_ledger(tmp_path: Path) -> None:
    state = _project(tmp_path)
    state.save()
    task_id = "task_project-1_1234567890abcdef"
    payload = {"task_id": task_id, "project_id": "project-1", "status": "succeeded", "kind": "chat", "label": "Done"}
    with sqlite3.connect(catalog_path(tmp_path)) as connection:
        connection.execute(
            "CREATE TABLE studio_tasks (task_id TEXT PRIMARY KEY, project_id TEXT, status TEXT, "
            "updated_epoch REAL, owner_pid INTEGER, owner_instance TEXT, payload_json TEXT)"
        )
        connection.execute(
            "INSERT INTO studio_tasks VALUES (?, ?, ?, ?, ?, ?, ?)",
            (task_id, "project-1", "succeeded", 1.0, 0, "old", json.dumps(payload)),
        )

    loaded = StudioTaskStore().get(tmp_path, task_id)
    assert loaded is not None
    assert loaded["payload"]["label"] == "Done"
    assert ExecutionStore().get(tmp_path, task_id, kind="studio")["version"] == 1


def test_execution_id_cannot_change_owner_or_kind(tmp_path: Path) -> None:
    state = _project(tmp_path)
    state.save()
    store = ExecutionStore()
    store.put(
        tmp_path,
        execution_id="job_12345678",
        project_id="project-1",
        kind="tool",
        status="queued",
        payload={"job_id": "job_12345678", "project_id": "project-1", "status": "queued"},
    )
    with pytest.raises(ProjectCatalogError, match="another project or kind"):
        store.put(
            tmp_path,
            execution_id="job_12345678",
            project_id="project-2",
            kind="tool",
            status="queued",
            payload={"job_id": "job_12345678", "project_id": "project-2", "status": "queued"},
        )


def test_one_active_studio_task_per_project_is_enforced_by_database(tmp_path: Path) -> None:
    state = _project(tmp_path)
    state.save()
    store = StudioTaskStore()
    first = {
        "task_id": "task_project-1_1111111111111111",
        "project_id": "project-1",
        "status": "running",
    }
    second = {
        "task_id": "task_project-1_2222222222222222",
        "project_id": "project-1",
        "status": "queued",
    }
    store.put(tmp_path, first, owner_pid=1, owner_instance="first")
    with pytest.raises(ExecutionConflict, match="already active"):
        store.put(tmp_path, second, owner_pid=2, owner_instance="second")
    first["status"] = "succeeded"
    store.put(tmp_path, first, owner_pid=0, owner_instance="first")
    store.put(tmp_path, second, owner_pid=2, owner_instance="second")
    assert store.active(tmp_path)["payload"]["task_id"] == second["task_id"]


def test_cli_job_checkpoint_is_visible_to_studio_and_survives_completion(tmp_path: Path) -> None:
    state = _project(tmp_path)
    state.save()
    job = create_tool_job(state, "sample_tool")

    def executor(_params: dict, _state: ProjectState) -> dict:
        checkpoint_job(tmp_path, job.job_id, stage="rendering", progress=0.5, message="Halfway")
        active = _project_detail(state)["jobs"][0]
        assert active["stage"] == "rendering"
        assert active["progress"] == 0.5
        with pytest.raises(JobRunnerError, match="backward"):
            checkpoint_job(tmp_path, job.job_id, stage="rendering", progress=0.25)
        return {"success": True, "message": "Done"}

    completed = run_tool_job(state, job.job_id, {"sample_tool": executor})
    assert completed.stage == "completed"
    assert completed.progress == 1.0
    with pytest.raises(JobRunnerError, match="Only the running"):
        checkpoint_job(tmp_path, job.job_id, stage="late", progress=1.0)


def test_catalog_commit_survives_job_json_export_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = _project(tmp_path)
    state.save()

    def fail_export(_path: Path, _payload: dict) -> None:
        raise OSError("projection unavailable")

    monkeypatch.setattr(job_runner, "_atomic_write_json", fail_export)
    with pytest.warns(RuntimeWarning, match="saved in the project catalog"):
        job = create_tool_job(state, "sample_tool")
    assert load_job(tmp_path, job.job_id).status == "queued"


def test_catalog_commit_survives_project_json_export_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project_dir = tmp_path / "project-1"
    monkeypatch.setattr(config, "AGENT_PROJECTS_DIR", str(tmp_path))
    state = _project(project_dir)
    state.save()
    state.project_name = "After export failure"

    def fail_replace(_source: Path, _target: Path) -> None:
        raise OSError("projection unavailable")

    monkeypatch.setattr("state.os.replace", fail_replace)
    with pytest.warns(RuntimeWarning, match="saved in the catalog"):
        state.save()
    assert state.revision == 2
    assert ProjectState.load("project-1").project_name == "After export failure"
