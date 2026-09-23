from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

import asset_registry
import content_cache
import tools.promotion as promotion_module
from asset_registry import load_asset_registry, record_project_asset
from content_cache import cache_project_file, load_cache_index
from state import ProjectState, utc_now_iso
from tools.promotion import promote_working_file
from vex_runtime import media_index
from vex_runtime.project_catalog import ProjectCatalogError, catalog_path, list_revisions
from vex_web.server import _project_detail


def _project(tmp_path: Path) -> tuple[ProjectState, Path]:
    now = utc_now_iso()
    source = tmp_path / "source.mp4"
    output = tmp_path / "output.mp4"
    source.write_bytes(b"original source")
    output.write_bytes(b"rendered output")
    state = ProjectState(
        project_id="project-1",
        project_name="Test",
        created_at=now,
        updated_at=now,
        source_files=[str(source)],
        working_file=str(source),
        working_dir=str(tmp_path),
        output_dir=str(tmp_path / "output"),
    )
    state.save()
    return state, output


def _promote(state: ProjectState, output: Path):  # noqa: ANN202
    return promote_working_file(
        state,
        output,
        operation={"op": "trim_clip", "params": {"start": 0.0, "end": 2.0}},
        metadata={"duration_sec": 2.0},
    )


def test_promotion_commits_revision_asset_and_cache_together(tmp_path: Path) -> None:
    state, output = _project(tmp_path)

    promotion = _promote(state, output)

    assert state.revision == 2
    assert len(list_revisions(tmp_path)) == 2
    assert load_asset_registry(tmp_path)["assets"][0]["asset_id"] == promotion.asset.asset_id
    assert load_cache_index(tmp_path)["entries"][0]["cache_key"] == promotion.cache_entry.cache_key
    assert _project_detail(state)["media_assets"][0]["asset_id"] == promotion.asset.asset_id
    with sqlite3.connect(catalog_path(tmp_path)) as connection:
        assert connection.execute("SELECT count(*) FROM project_assets").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM project_cache").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM project_revisions").fetchone()[0] == 2


def test_promotion_failure_rolls_back_all_catalog_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state, output = _project(tmp_path)
    before = state.capture_snapshot()
    original_insert = media_index.insert_media_record

    def fail_cache(connection, kind, record):  # noqa: ANN001, ANN202
        if kind == "cache":
            raise ProjectCatalogError("injected cache-index failure")
        return original_insert(connection, kind, record)

    monkeypatch.setattr(media_index, "insert_media_record", fail_cache)
    with pytest.raises(ProjectCatalogError, match="injected cache-index failure"):
        _promote(state, output)

    assert state.capture_snapshot() == before
    assert len(list_revisions(tmp_path)) == 1
    with sqlite3.connect(catalog_path(tmp_path)) as connection:
        assert connection.execute(
            "SELECT count(*) FROM sqlite_master WHERE name IN ('project_assets', 'project_cache')"
        ).fetchone()[0] == 0
    assert load_asset_registry(tmp_path)["assets"] == []
    assert load_cache_index(tmp_path)["entries"] == []


def test_promotion_imports_legacy_media_indexes_atomically(tmp_path: Path) -> None:
    state, output = _project(tmp_path)
    (tmp_path / "assets.json").write_text(
        json.dumps({"schema_version": 1, "assets": [{"asset_id": "asset_legacy", "kind": "video", "path": str(output), "created_at": "2026-01-01"}]}),
        encoding="utf-8",
    )
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "cache_index.json").write_text(
        json.dumps({"schema_version": 1, "entries": [{"cache_key": "sha256:legacy", "kind": "video", "created_at": "2026-01-01"}]}),
        encoding="utf-8",
    )

    promotion = _promote(state, output)

    assert {item["asset_id"] for item in load_asset_registry(tmp_path)["assets"]} == {
        "asset_legacy", promotion.asset.asset_id
    }
    assert {item["cache_key"] for item in load_cache_index(tmp_path)["entries"]} == {
        "sha256:legacy", promotion.cache_entry.cache_key
    }


def test_invalid_legacy_index_blocks_promotion_without_partial_commit(tmp_path: Path) -> None:
    state, output = _project(tmp_path)
    before = state.capture_snapshot()
    (tmp_path / "assets.json").write_text("not-json", encoding="utf-8")

    with pytest.raises(ProjectCatalogError, match="Unable to migrate legacy asset index"):
        _promote(state, output)

    assert state.capture_snapshot() == before
    assert len(list_revisions(tmp_path)) == 1


def test_external_legacy_index_symlink_blocks_promotion(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    state, output = _project(project_dir)
    external = tmp_path / "external-assets.json"
    external.write_text(json.dumps({"schema_version": 1, "assets": []}), encoding="utf-8")
    (project_dir / "assets.json").symlink_to(external)

    with pytest.raises(ProjectCatalogError, match="escapes"):
        _promote(state, output)
    assert state.revision == 1


def test_promotion_succeeds_if_json_projections_fail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state, output = _project(tmp_path)

    def fail_export(_path: Path, _payload: dict) -> None:
        raise OSError("projection unavailable")

    monkeypatch.setattr(asset_registry, "_atomic_write_json", fail_export)
    monkeypatch.setattr(content_cache, "_atomic_write_json", fail_export)
    with pytest.warns(RuntimeWarning, match="JSON export failed"):
        promotion = _promote(state, output)

    assert state.revision == 2
    assert load_asset_registry(tmp_path)["assets"][0]["asset_id"] == promotion.asset.asset_id
    assert load_cache_index(tmp_path)["entries"][0]["cache_key"] == promotion.cache_entry.cache_key


def test_standalone_media_records_use_catalog_when_available(tmp_path: Path) -> None:
    state, output = _project(tmp_path)
    asset = record_project_asset(state, output, kind="video")
    cached = cache_project_file(state, output, kind="video")
    (tmp_path / "assets.json").write_text("corrupt", encoding="utf-8")
    (tmp_path / "cache" / "cache_index.json").write_text("corrupt", encoding="utf-8")

    assert load_asset_registry(tmp_path)["assets"][0]["asset_id"] == asset.asset_id
    assert load_cache_index(tmp_path)["entries"][0]["cache_key"] == cached.cache_key


def test_corrupt_catalog_media_index_does_not_fall_back_to_json(tmp_path: Path) -> None:
    state, output = _project(tmp_path)
    _promote(state, output)
    with sqlite3.connect(catalog_path(tmp_path)) as connection:
        connection.execute("UPDATE project_assets SET payload_sha256='invalid'")

    with pytest.raises(ProjectCatalogError, match="checksum mismatch"):
        load_asset_registry(tmp_path)


def test_promotion_rejects_output_that_changes_during_preparation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state, output = _project(tmp_path)
    original_prepare = promotion_module.prepare_cache_file

    def change_output(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        output.write_bytes(b"changed after asset hash")
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(promotion_module, "prepare_cache_file", change_output)
    with pytest.raises(RuntimeError, match="changed while it was being promoted"):
        _promote(state, output)
    assert state.revision == 1
    assert len(list_revisions(tmp_path)) == 1
