from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

import config
from state import ProjectState, utc_now_iso
from vex_runtime.project_catalog import (
    ProjectCatalogError,
    ProjectRevisionConflict,
    catalog_path,
    list_revisions,
)


def _state(directory: Path, *, project_id: str = "project-123") -> ProjectState:
    now = utc_now_iso()
    return ProjectState(
        project_id=project_id,
        project_name="Initial",
        created_at=now,
        updated_at=now,
        source_files=[str(directory / "source.mp4")],
        working_file=str(directory / "source.mp4"),
        working_dir=str(directory),
        output_dir=str(directory / "output"),
    )


def test_project_catalog_is_authoritative_and_preserves_revisions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(config, "AGENT_PROJECTS_DIR", str(tmp_path))
    state = _state(tmp_path / "project-123")
    state.save()
    assert state.revision == 1
    state.project_name = "Second"
    state.save()
    assert state.revision == 2

    # A stale or missing compatibility export must not override the catalog.
    state.state_path.write_text('{"project_name":"tampered"}', encoding="utf-8")
    assert ProjectState.load(state.project_id).project_name == "Second"
    state.state_path.unlink()
    loaded = ProjectState.load(state.project_id)
    assert loaded.project_name == "Second"
    assert loaded.revision == 2
    assert [row["revision"] for row in list_revisions(state.working_dir)] == [2, 1]
    assert ProjectState.list_projects()[0]["project_name"] == "Second"


def test_project_catalog_rejects_stale_writes(tmp_path: Path) -> None:
    state = _state(tmp_path)
    state.save()
    stale = ProjectState.from_dict(json.loads(state.state_path.read_text(encoding="utf-8")))
    state.project_name = "Winner"
    state.save()
    stale.project_name = "Stale"

    with pytest.raises(ProjectRevisionConflict, match="Reload before editing"):
        stale.save()
    assert state.refresh_from_disk()
    assert state.project_name == "Winner"


def test_project_catalog_imports_legacy_json_before_next_write(tmp_path: Path) -> None:
    state = _state(tmp_path)
    state.state_path.parent.mkdir(parents=True, exist_ok=True)
    state.state_path.write_text(json.dumps(state.capture_snapshot()), encoding="utf-8")

    state.project_name = "Migrated"
    state.save()

    assert state.revision == 2
    assert catalog_path(tmp_path).is_file()
    with sqlite3.connect(catalog_path(tmp_path)) as connection:
        rows = connection.execute(
            "SELECT payload_json FROM project_revisions ORDER BY revision"
        ).fetchall()
    assert json.loads(rows[0][0])["project_name"] == "Initial"
    assert json.loads(rows[1][0])["project_name"] == "Migrated"


def test_project_catalog_does_not_fall_back_to_json_when_corrupt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(config, "AGENT_PROJECTS_DIR", str(tmp_path))
    state = _state(tmp_path / "project-123")
    state.save()
    with sqlite3.connect(catalog_path(state.working_dir)) as connection:
        connection.execute("UPDATE project_revisions SET payload_sha256 = 'invalid'")

    with pytest.raises(ProjectCatalogError, match="checksum mismatch"):
        ProjectState.load(state.project_id)
    with pytest.raises(ProjectCatalogError, match="checksum mismatch"):
        state.save()


def test_project_catalog_rejects_external_symlink(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(config, "AGENT_PROJECTS_DIR", str(tmp_path / "projects"))
    external = _state(tmp_path / "external" / "project-123")
    external.save()
    project_dir = tmp_path / "projects" / "project-123"
    project_dir.mkdir(parents=True)
    (project_dir / "project.sqlite3").symlink_to(catalog_path(external.working_dir))

    with pytest.raises(ProjectCatalogError, match="escapes"):
        _state(project_dir).save()
    assert not ProjectState.list_projects()


def test_snapshot_rollback_keeps_monotonic_revision(tmp_path: Path) -> None:
    state = _state(tmp_path)
    state.save()
    original = state.capture_snapshot()
    state.project_name = "Temporary"
    state.save()
    state.restore_snapshot(original)

    assert state.project_name == "Initial"
    assert state.revision == 3
    assert [row["revision"] for row in list_revisions(tmp_path)] == [3, 2, 1]
