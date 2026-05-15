from __future__ import annotations

import os
from pathlib import Path

import config
import main
from state import ProjectState, utc_now_iso


def test_auto_resume_ignores_only_saved_project_outside_launch_directory(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    projects_dir = tmp_path / "projects"
    launch_dir = tmp_path / "new-folder"
    old_source_dir = tmp_path / "old-folder"
    monkeypatch.setattr(config, "AGENT_PROJECTS_DIR", str(projects_dir))
    _state(projects_dir / "old-project", project_id="old-project", source_file=old_source_dir / "old.mp4").save()

    selected = main.select_auto_resume_project(ProjectState.list_projects(), launch_dir)

    assert selected is None


def test_auto_resume_selects_single_project_from_launch_directory(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    projects_dir = tmp_path / "projects"
    launch_dir = tmp_path / "footage"
    monkeypatch.setattr(config, "AGENT_PROJECTS_DIR", str(projects_dir))
    _state(projects_dir / "matching-project", project_id="matching-project", source_file=launch_dir / "raw_video.mp4").save()

    selected = main.select_auto_resume_project(ProjectState.list_projects(), launch_dir)

    assert selected is not None
    assert selected["project_id"] == "matching-project"


def test_load_command_reports_missing_video_path_instead_of_falling_through(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    monkeypatch.chdir(tmp_path)

    parsed = main.parse_load_source_command("load raw_video.mp4")

    assert parsed is not None
    assert parsed[0] == "missing_path"
    assert os.path.normcase(parsed[1]) == os.path.normcase(str(tmp_path / "raw_video.mp4"))


def test_load_command_resolves_existing_relative_video_path(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    video_path = tmp_path / "raw_video.mp4"
    video_path.write_bytes(b"placeholder")
    monkeypatch.chdir(tmp_path)

    parsed = main.parse_load_source_command("load raw_video.mp4")

    assert parsed is not None
    assert parsed[0] == "path"
    assert os.path.normcase(parsed[1]) == os.path.normcase(str(video_path))


def _state(tmp_path: Path, *, project_id: str, source_file: Path) -> ProjectState:
    now = utc_now_iso()
    return ProjectState(
        project_id=project_id,
        project_name="Test Project",
        created_at=now,
        updated_at=now,
        source_files=[str(source_file)],
        working_file=str(tmp_path / "working.mp4"),
        working_dir=str(tmp_path),
        output_dir=str(source_file.parent),
        metadata={"duration_sec": 120.0, "width": 1920, "height": 1080, "fps": 30.0},
        provider="test",
        model="test-model",
    )
