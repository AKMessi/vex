from __future__ import annotations

from pathlib import Path

import pytest
from rich.console import Console

import main
from job_runner import JobRunnerError, create_tool_job, load_job, run_tool_job
from state import ProjectState, utc_now_iso


def test_create_tool_job_persists_queued_record(tmp_path: Path) -> None:
    state = _state(tmp_path)

    job = create_tool_job(
        state,
        "sample_tool",
        {"value": 3},
        allowed_tools={"sample_tool"},
        metadata={"contract": {"category": "test"}},
    )
    loaded = load_job(state.working_dir, job.job_id)

    assert job.job_id.startswith("job_")
    assert loaded.status == "queued"
    assert loaded.tool_name == "sample_tool"
    assert loaded.params == {"value": 3}
    assert loaded.metadata["contract"]["category"] == "test"


def test_run_tool_job_records_success_without_serializing_state(tmp_path: Path) -> None:
    state = _state(tmp_path)
    output_path = tmp_path / "out.mp4"
    job = create_tool_job(state, "sample_tool", {"value": 3}, allowed_tools={"sample_tool"})

    def executor(params: dict, project_state: ProjectState) -> dict:
        project_state.artifacts["job_value"] = params["value"]
        return {
            "success": True,
            "message": "done",
            "updated_state": project_state,
            "tool_name": "sample_tool",
            "output_path": output_path,
        }

    completed = run_tool_job(state, job.job_id, {"sample_tool": executor})
    loaded = load_job(state.working_dir, job.job_id)

    assert completed.status == "succeeded"
    assert completed.attempts == 1
    assert completed.result["output_path"] == str(output_path)
    assert "updated_state" not in completed.result
    assert loaded.status == "succeeded"
    assert state.artifacts["job_value"] == 3


def test_run_tool_job_records_tool_failure_result(tmp_path: Path) -> None:
    state = _state(tmp_path)
    job = create_tool_job(state, "sample_tool", allowed_tools={"sample_tool"})

    def executor(_params: dict, _state: ProjectState) -> dict:
        return {"success": False, "message": "tool rejected input", "tool_name": "sample_tool"}

    completed = run_tool_job(state, job.job_id, {"sample_tool": executor})

    assert completed.status == "failed"
    assert completed.error == "tool rejected input"
    assert completed.result["success"] is False


def test_run_tool_job_records_unhandled_exception(tmp_path: Path) -> None:
    state = _state(tmp_path)
    job = create_tool_job(state, "sample_tool", allowed_tools={"sample_tool"})

    def executor(_params: dict, _state: ProjectState) -> dict:
        raise RuntimeError("render process crashed")

    completed = run_tool_job(state, job.job_id, {"sample_tool": executor})

    assert completed.status == "failed"
    assert "render process crashed" in completed.error
    assert completed.result["tool_name"] == "sample_tool"


def test_run_tool_job_rejects_completed_job_without_force(tmp_path: Path) -> None:
    state = _state(tmp_path)
    job = create_tool_job(state, "sample_tool", allowed_tools={"sample_tool"})

    def executor(_params: dict, _state: ProjectState) -> dict:
        return {"success": True, "message": "done", "tool_name": "sample_tool"}

    run_tool_job(state, job.job_id, {"sample_tool": executor})

    with pytest.raises(JobRunnerError, match="use --force"):
        run_tool_job(state, job.job_id, {"sample_tool": executor})


def test_parse_job_params_requires_json_object() -> None:
    assert main.parse_job_params('{"max_visuals": 2}') == {"max_visuals": 2}
    with pytest.raises(Exception, match="JSON object"):
        main.parse_job_params("[1, 2]")


def test_render_jobs_table_lists_project_jobs(tmp_path: Path) -> None:
    state = _state(tmp_path)
    job = create_tool_job(state, "sample_tool", {"value": 3}, allowed_tools={"sample_tool"})
    console = Console(record=True, width=140)

    console.print(main.render_jobs_table(state))

    output = console.export_text()
    assert job.job_id in output
    assert "sample_tool" in output
    assert "queued" in output


def _state(tmp_path: Path) -> ProjectState:
    now = utc_now_iso()
    source_file = tmp_path / "source.mp4"
    source_file.write_bytes(b"placeholder")
    return ProjectState(
        project_id="test-project",
        project_name="Test Project",
        created_at=now,
        updated_at=now,
        source_files=[str(source_file)],
        working_file=str(tmp_path / "working.mp4"),
        working_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        metadata={"duration_sec": 120.0, "width": 1920, "height": 1080, "fps": 30.0},
        provider="test",
        model="test-model",
    )
