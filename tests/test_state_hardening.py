from __future__ import annotations

import json
from pathlib import Path

import config
from state import ProjectState, utc_now_iso


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
