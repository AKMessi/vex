from __future__ import annotations

from pathlib import Path

import tools.automation as automation


def test_bundle_directories_are_unique_within_the_same_second(
    monkeypatch,
    tmp_path: Path,
) -> None:  # noqa: ANN001
    monkeypatch.setattr(
        automation,
        "utc_now_iso",
        lambda: "2026-07-11T12:00:00+00:00",
    )

    first = automation.create_unique_bundle_dir(tmp_path, "project_auto_visuals")
    second = automation.create_unique_bundle_dir(tmp_path, "project_auto_visuals")

    assert first != second
    assert first.is_dir()
    assert second.is_dir()
    assert first.parent == tmp_path
    assert second.parent == tmp_path


def test_bundle_prefix_is_sanitized_to_stay_under_root(tmp_path: Path) -> None:
    bundle = automation.create_unique_bundle_dir(tmp_path, "../unsafe bundle")

    assert bundle.parent == tmp_path
    assert ".." not in bundle.name
