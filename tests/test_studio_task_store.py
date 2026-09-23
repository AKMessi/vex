from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

import config
from state import ProjectState, utc_now_iso
from vex_web import server as web_server
from vex_web.server import TaskManager, WebTask
from vex_web.task_store import StudioTaskStore


def _project(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> ProjectState:
    monkeypatch.setattr(config, "AGENT_PROJECTS_DIR", str(tmp_path))
    directory = tmp_path / "project-1"
    now = utc_now_iso()
    state = ProjectState(
        project_id="project-1",
        project_name="Studio project",
        created_at=now,
        updated_at=now,
        source_files=[str(directory / "source.mp4")],
        working_file=str(directory / "source.mp4"),
        working_dir=str(directory),
        output_dir=str(directory / "output"),
    )
    state.save()
    return state


def test_studio_task_survives_manager_restart(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _project(monkeypatch, tmp_path)
    manager = TaskManager(max_workers=1, persist=True)
    try:
        task = manager.submit(
            "project-1", "chat", "Editing", lambda _task: {"success": True, "message": "Done"}
        )
        deadline = time.monotonic() + 3
        while manager.snapshot(task.task_id)["status"] != "succeeded" and time.monotonic() < deadline:
            time.sleep(0.01)
        assert manager.snapshot(task.task_id)["status"] == "succeeded"
    finally:
        manager.shutdown()

    restored = TaskManager(persist=True)
    try:
        snapshot = restored.snapshot(task.task_id)
        assert snapshot is not None
        assert snapshot["status"] == "succeeded"
        assert snapshot["message"] == "Done"
        assert restored.active_snapshot("project-1") is None
        assert restored.latest_snapshot("project-1")["task_id"] == task.task_id
    finally:
        restored.shutdown()


def test_studio_recovers_abandoned_task_as_interrupted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project = _project(monkeypatch, tmp_path)
    task = WebTask(
        task_id="task_project-1_1234567890abcdef",
        project_id="project-1",
        kind="chat",
        label="Editing",
        status="running",
    )
    store = StudioTaskStore()
    store.put(project.working_dir, task.to_dict(), owner_pid=999_999, owner_instance="old")
    monkeypatch.setattr(web_server, "process_is_running", lambda _pid: False)

    manager = TaskManager(persist=True)
    try:
        snapshot = manager.snapshot(task.task_id)
        assert snapshot is not None
        assert snapshot["status"] == "failed"
        assert "stopped" in snapshot["error"]
        assert manager.active_snapshot("project-1") is None
        assert manager.latest_snapshot("project-1")["status"] == "failed"
        assert store.get(project.working_dir, task.task_id)["payload"]["status"] == "failed"
    finally:
        manager.shutdown()


def test_studio_progress_is_persisted_while_work_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project = _project(monkeypatch, tmp_path)
    release = threading.Event()
    started = threading.Event()
    manager = TaskManager(max_workers=1, persist=True)

    def work(task: WebTask) -> dict:
        manager.append_event(task, {"kind": "tool", "title": "Rendering", "status": "running"})
        manager.append_stream(task, "First draft")
        started.set()
        assert release.wait(timeout=3)
        return {"success": True, "message": "Done"}

    try:
        task = manager.submit("project-1", "chat", "Editing", work)
        assert started.wait(timeout=2)
        stored = StudioTaskStore().get(project.working_dir, task.task_id)
        assert stored is not None
        assert stored["payload"]["status"] == "running"
        assert stored["payload"]["events"][0]["title"] == "Rendering"
        assert stored["payload"]["stream"] == "First draft"
    finally:
        release.set()
        manager.shutdown()
