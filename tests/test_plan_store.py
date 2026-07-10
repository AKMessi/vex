from __future__ import annotations

from pathlib import Path

import pytest
from rich.console import Console

import main
from edit_plan import EditPlan, ToolStep
from plan_store import (
    PlanStoreError,
    create_plan_record,
    edit_plan_from_record,
    list_plan_records,
    load_plan_record,
    write_plan_record,
)
from state import ProjectState, utc_now_iso


def test_plan_store_round_trips_edit_plan(tmp_path: Path) -> None:
    state = _state(tmp_path)
    plan = EditPlan(
        steps=[ToolStep("get_video_info", {}, "inspect")],
        confidence=0.92,
        reason="metadata command",
    )

    record = create_plan_record(state, "show video metadata", plan)
    loaded = load_plan_record(state.working_dir, record.plan_id)
    restored = edit_plan_from_record(loaded)

    assert loaded.status == "planned"
    assert loaded.instruction == "show video metadata"
    assert restored.tool_names == ["get_video_info"]
    assert restored.confidence == 0.92
    assert list_plan_records(state.working_dir)[0].plan_id == record.plan_id


def test_direct_create_and_apply_plan_with_executor(tmp_path: Path) -> None:
    state = _state(tmp_path)
    record = main.direct_create_plan(state, "show video metadata")
    calls: list[dict] = []

    def fake_info(params: dict, project_state: ProjectState) -> dict:
        calls.append(dict(params))
        return {
            "success": True,
            "message": "Metadata ready.",
            "updated_state": project_state,
            "tool_name": "get_video_info",
        }

    applied = main.direct_apply_plan(
        state,
        record.plan_id,
        executors={"get_video_info": fake_info},
    )
    loaded = load_plan_record(state.working_dir, record.plan_id)

    assert calls == [{}]
    assert applied.status == "applied"
    assert loaded.status == "applied"
    assert loaded.results[0]["tool_name"] == "get_video_info"
    assert "updated_state" not in loaded.results[0]


def test_direct_apply_plan_rejects_already_applied_without_force(tmp_path: Path) -> None:
    state = _state(tmp_path)
    record = create_plan_record(
        state,
        "show video metadata",
        EditPlan(steps=[ToolStep("get_video_info", {}, "inspect")]),
    )

    def fake_info(_params: dict, project_state: ProjectState) -> dict:
        return {
            "success": True,
            "message": "Metadata ready.",
            "updated_state": project_state,
            "tool_name": "get_video_info",
        }

    main.direct_apply_plan(state, record.plan_id, executors={"get_video_info": fake_info})

    with pytest.raises(PlanStoreError, match="already applied"):
        main.direct_apply_plan(state, record.plan_id, executors={"get_video_info": fake_info})


def test_direct_apply_plan_rejects_force_while_worker_is_active(tmp_path: Path) -> None:
    state = _state(tmp_path)
    record = create_plan_record(
        state,
        "show video metadata",
        EditPlan(steps=[ToolStep("get_video_info", {}, "inspect")]),
    )
    record.status = "applying"
    record.pid = __import__("os").getpid()
    write_plan_record(state.working_dir, record)

    with pytest.raises(PlanStoreError, match="already being applied"):
        main.direct_apply_plan(
            state,
            record.plan_id,
            force=True,
            executors={"get_video_info": lambda _params, _state: {"success": True}},
        )


def test_direct_apply_plan_force_recovers_stale_application(
    monkeypatch,
    tmp_path: Path,
) -> None:  # noqa: ANN001
    state = _state(tmp_path)
    record = create_plan_record(
        state,
        "show video metadata",
        EditPlan(steps=[ToolStep("get_video_info", {}, "inspect")]),
    )
    record.status = "applying"
    record.pid = 999_999
    write_plan_record(state.working_dir, record)
    monkeypatch.setattr("plan_store.process_is_running", lambda _pid: False)

    applied = main.direct_apply_plan(
        state,
        record.plan_id,
        force=True,
        executors={
            "get_video_info": lambda _params, project_state: {
                "success": True,
                "message": "recovered",
                "updated_state": project_state,
            }
        },
    )

    assert applied.status == "applied"
    assert applied.pid == 0


def test_direct_apply_plan_marks_unknown_tool_failure(tmp_path: Path) -> None:
    state = _state(tmp_path)
    record = create_plan_record(
        state,
        "run missing tool",
        EditPlan(steps=[ToolStep("missing_tool", {}, "missing")]),
    )

    with pytest.raises(PlanStoreError, match="unknown tool"):
        main.direct_apply_plan(state, record.plan_id, executors={})

    loaded = load_plan_record(state.working_dir, record.plan_id)
    assert loaded.status == "failed"
    assert loaded.results[0]["tool_name"] == "missing_tool"


def test_direct_apply_plan_records_invalid_executor_result(tmp_path: Path) -> None:
    state = _state(tmp_path)
    record = create_plan_record(
        state,
        "show video metadata",
        EditPlan(steps=[ToolStep("get_video_info", {}, "inspect")]),
    )

    with pytest.raises(PlanStoreError, match="invalid result"):
        main.direct_apply_plan(
            state,
            record.plan_id,
            executors={"get_video_info": lambda _params, _state: None},
        )

    loaded = load_plan_record(state.working_dir, record.plan_id)
    assert loaded.status == "failed"
    assert loaded.pid == 0
    assert loaded.results[0]["success"] is False


def test_render_plans_table_lists_saved_plans(tmp_path: Path) -> None:
    state = _state(tmp_path)
    record = create_plan_record(
        state,
        "show video metadata",
        EditPlan(steps=[ToolStep("get_video_info", {}, "inspect")]),
    )
    console = Console(record=True, width=140)

    console.print(main.render_plans_table(state))

    output = console.export_text()
    assert record.plan_id in output
    assert "show video metadata" in output
    assert "planned" in output


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
