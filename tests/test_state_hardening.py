from __future__ import annotations

import json
from pathlib import Path

import config
from state import ProjectState, utc_now_iso
from timeline import PROJECT_STATE_SCHEMA_VERSION, TIMELINE_OPERATION_SCHEMA_VERSION


def test_project_state_save_writes_valid_json_without_temp_files(tmp_path: Path) -> None:
    state = _state(tmp_path, project_id="project-123")

    state.save()

    payload = json.loads(state.state_path.read_text(encoding="utf-8"))
    assert payload["project_id"] == "project-123"
    assert not list(tmp_path.glob(".project-123.*.tmp"))


def test_project_load_rejects_glob_and_path_lookup(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    monkeypatch.setattr(config, "AGENT_PROJECTS_DIR", str(tmp_path))
    _state(tmp_path / "project-123", project_id="project-123").save()

    for lookup in ("*", "../project-123", "project-123.json"):
        try:
            ProjectState.load(lookup)
        except FileNotFoundError as exc:
            assert "Invalid project id" in str(exc)
        else:
            raise AssertionError(f"Project lookup {lookup!r} should have been rejected.")


def test_project_load_supports_safe_partial_ids(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    monkeypatch.setattr(config, "AGENT_PROJECTS_DIR", str(tmp_path))
    _state(tmp_path / "abcdef123456", project_id="abcdef123456").save()

    loaded = ProjectState.load("abcdef")

    assert loaded.project_id == "abcdef123456"


def test_project_state_save_migrates_timeline_schema(tmp_path: Path) -> None:
    state = _state(tmp_path, project_id="project-123")
    state.timeline.append(
        {
            "op": "trim_clip",
            "params": {"start": 0.0, "end": 10.0},
            "timestamp": "2026-01-01T00:00:00+00:00",
            "description": "legacy trim",
        }
    )

    state.save()

    payload = json.loads(state.state_path.read_text(encoding="utf-8"))
    operation = payload["timeline"][0]
    assert payload["schema_version"] == PROJECT_STATE_SCHEMA_VERSION
    assert operation["schema_version"] == TIMELINE_OPERATION_SCHEMA_VERSION
    assert operation["op_id"].startswith("op_")
    assert operation["assets"] == []


def test_project_state_from_dict_migrates_legacy_payload(tmp_path: Path) -> None:
    now = utc_now_iso()
    payload = {
        "project_id": "project-123",
        "project_name": "Test Project",
        "created_at": now,
        "updated_at": now,
        "source_files": [str(tmp_path / "source.mp4")],
        "working_file": str(tmp_path / "working.mp4"),
        "working_dir": str(tmp_path),
        "output_dir": str(tmp_path / "out"),
        "timeline": [{"op": "trim_clip"}],
        "redo_stack": [{"op": "adjust_speed", "params": {"factor": 2}}],
    }

    state = ProjectState.from_dict(payload)

    assert state.schema_version == PROJECT_STATE_SCHEMA_VERSION
    assert state.timeline[0]["schema_version"] == TIMELINE_OPERATION_SCHEMA_VERSION
    assert state.timeline[0]["params"] == {}
    assert state.redo_stack[0]["op_id"].startswith("op_")


def _state(tmp_path: Path, *, project_id: str) -> ProjectState:
    now = utc_now_iso()
    return ProjectState(
        project_id=project_id,
        project_name="Test Project",
        created_at=now,
        updated_at=now,
        source_files=[str(tmp_path / "source.mp4")],
        working_file=str(tmp_path / "working.mp4"),
        working_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        metadata={"duration_sec": 120.0, "width": 1920, "height": 1080, "fps": 30.0},
        provider="test",
        model="test-model",
    )
