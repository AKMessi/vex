import shutil

import pytest

from vex_runtime.production_archive import relink_project_source


def test_project_can_relink_the_same_source_after_multiple_renames(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    original = tmp_path / "original.mp4"
    original.write_bytes(b"footage")
    shutil.copyfile(original, workspace / "source_original.mp4")

    class Project:
        working_dir = str(workspace)
        source_files = [str(original)]
        artifacts = {}

        def save(self):
            pass

    state = Project()
    first = original.rename(tmp_path / "first-move.mp4")
    assert relink_project_source(state, first) == first
    second = first.rename(tmp_path / "second-move.mp4")
    assert relink_project_source(state, second) == second
    assert state.source_files == [str(second)]
    wrong = tmp_path / "wrong.mp4"
    wrong.write_bytes(b"unrelated")
    with pytest.raises(ValueError, match="does not match"):
        relink_project_source(state, wrong)
    assert state.source_files == [str(second)]
