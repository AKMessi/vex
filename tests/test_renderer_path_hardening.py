from __future__ import annotations

from pathlib import Path

from renderers.base import safe_render_job_dir


def test_safe_render_job_dir_sanitizes_path_traversal(tmp_path: Path) -> None:
    render_root = tmp_path / "renders"

    job_dir = safe_render_job_dir(render_root, "../../outside")

    assert job_dir == (render_root / "outside").resolve()
    assert job_dir.parent == render_root.resolve()


def test_safe_render_job_dir_limits_untrusted_identifier_length(tmp_path: Path) -> None:
    render_root = tmp_path / "renders"

    job_dir = safe_render_job_dir(render_root, "a" * 200)

    assert job_dir.parent == render_root.resolve()
    assert len(job_dir.name) == 96
