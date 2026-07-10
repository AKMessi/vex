from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from asset_registry import load_asset_registry, latest_assets, record_asset
from content_cache import cache_file, find_cached_file, load_cache_index
from state import ProjectState, utc_now_iso
from timeline import TIMELINE_OPERATION_SCHEMA_VERSION
from tools import TOOL_CONTRACTS, TOOL_EXECUTORS
import tools.contracts as tool_contracts
from tools.contracts import ToolContract, execute_tool_contract, normalize_tool_result
from tools.promotion import promote_working_file


def test_tool_contracts_cover_public_executors() -> None:
    assert set(TOOL_CONTRACTS) == set(TOOL_EXECUTORS)
    assert TOOL_CONTRACTS["generate_video"].long_running
    assert not TOOL_CONTRACTS["renderers_doctor"].requires_project
    assert TOOL_CONTRACTS["trim_clip"].replayable


def test_tool_result_normalization_preserves_compatibility(tmp_path: Path) -> None:
    state = _state(tmp_path)
    contract = ToolContract(
        name="sample_tool",
        module_name="sample",
        function_name="execute",
        category="test",
        mutates_project=False,
    )

    result = normalize_tool_result({"success": 1}, contract=contract, state=state)
    invalid = normalize_tool_result(None, contract=contract, state=state)

    assert result["success"] is True
    assert result["message"] == ""
    assert result["updated_state"] is state
    assert result["tool_name"] == "sample_tool"
    assert result["contract"]["mutates_project"] is False
    assert invalid["success"] is False


def test_tool_result_normalization_does_not_treat_false_string_as_success(tmp_path: Path) -> None:
    state = _state(tmp_path)
    contract = ToolContract("sample", "sample", "execute", "test")

    result = normalize_tool_result({"success": "false"}, contract=contract, state=state)

    assert result["success"] is False


def test_mutating_tool_contract_rolls_back_failed_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = _state(tmp_path)
    original_file = state.working_file
    contract = ToolContract("sample", "sample", "execute", "test")

    def execute(_params: dict, project: ProjectState) -> dict[str, object]:
        project.working_file = str(tmp_path / "partial.mp4")
        project.artifacts["partial"] = True
        return {"success": False, "message": "rejected"}

    monkeypatch.setattr(
        tool_contracts,
        "import_module",
        lambda _name: SimpleNamespace(execute=execute),
    )

    result = execute_tool_contract(contract, {}, state)

    assert result["success"] is False
    assert state.working_file == original_file
    assert "partial" not in state.artifacts


def test_mutating_tool_contract_normalizes_exception_and_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = _state(tmp_path)
    contract = ToolContract("sample", "sample", "execute", "test")

    def execute(_params: dict, project: ProjectState) -> dict[str, object]:
        project.timeline.append({"op": "partial"})
        raise RuntimeError("executor crashed")

    monkeypatch.setattr(
        tool_contracts,
        "import_module",
        lambda _name: SimpleNamespace(execute=execute),
    )

    result = execute_tool_contract(contract, {}, state)

    assert result["success"] is False
    assert result["message"] == "executor crashed"
    assert state.timeline == []


def test_asset_registry_records_project_files(tmp_path: Path) -> None:
    asset_path = tmp_path / "clip.bin"
    asset_path.write_bytes(b"asset payload")

    record = record_asset(
        tmp_path,
        asset_path,
        kind="video",
        role="source",
        metadata={"duration_sec": 3.5},
    )

    registry = load_asset_registry(tmp_path)
    latest = latest_assets(tmp_path, kind="video", role="source")
    assert record.asset_id.startswith("asset_")
    assert record.checksum_sha256
    assert registry["assets"][0]["metadata"]["duration_sec"] == 3.5
    assert latest[0]["asset_id"] == record.asset_id


def test_content_cache_stores_files_by_checksum(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"cached video")

    entry = cache_file(tmp_path, source, kind="video", metadata={"role": "test"})
    found = find_cached_file(tmp_path, entry.cache_key)

    assert entry.cache_key.startswith("sha256:")
    assert Path(entry.cached_path).is_file()
    assert found is not None
    assert found.cached_path == entry.cached_path
    assert load_cache_index(tmp_path)["entries"][0]["metadata"]["role"] == "test"


def test_promote_working_file_updates_state_and_asset_registry(tmp_path: Path) -> None:
    state = _state(tmp_path)
    output_path = tmp_path / "trimmed.mp4"
    output_path.write_bytes(b"trimmed video")

    promotion = promote_working_file(
        state,
        output_path,
        operation={"op": "trim_clip", "params": {"start": 0.0, "end": 2.0}},
        metadata={"duration_sec": 2.0, "width": 1920, "height": 1080},
        asset_source="trim_clip",
    )

    saved = json.loads(state.state_path.read_text(encoding="utf-8"))
    assert state.working_file == str(output_path.resolve())
    assert state.timeline[-1]["schema_version"] == TIMELINE_OPERATION_SCHEMA_VERSION
    assert state.timeline[-1]["assets"] == [promotion.asset.asset_id]
    assert state.timeline[-1]["metadata"]["cache_key"] == promotion.cache_entry.cache_key
    assert Path(promotion.cache_entry.cached_path).is_file()
    assert saved["timeline"][-1]["op_id"] == promotion.operation["op_id"]
    assert load_asset_registry(tmp_path)["assets"][0]["asset_id"] == promotion.asset.asset_id


def test_promote_working_file_restores_memory_when_state_save_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = _state(tmp_path)
    output_path = tmp_path / "trimmed.mp4"
    output_path.write_bytes(b"trimmed video")
    old_working_file = state.working_file
    old_metadata = dict(state.metadata)
    old_timeline = list(state.timeline)

    def fail_save() -> None:
        raise RuntimeError("save failed")

    monkeypatch.setattr(state, "save", fail_save)

    with pytest.raises(RuntimeError, match="save failed"):
        promote_working_file(
            state,
            output_path,
            operation={"op": "trim_clip"},
            metadata={"duration_sec": 2.0},
        )

    assert state.working_file == old_working_file
    assert state.metadata == old_metadata
    assert state.timeline == old_timeline


def test_trim_tool_promotes_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import tools.trim as trim_tool

    state = _state(tmp_path)
    output_path = tmp_path / "engine-output.mp4"

    def fake_trim(_input_path: str, _working_dir: str, _start: float, _end: float | None) -> str:
        output_path.write_bytes(b"trimmed video")
        return str(output_path)

    monkeypatch.setattr(trim_tool, "trim", fake_trim)
    monkeypatch.setattr(
        trim_tool,
        "probe_video",
        lambda _path: {"duration_sec": 1.5, "width": 1280, "height": 720, "fps": 30.0},
    )

    result = trim_tool.execute({"start": "0", "end": "1"}, state)

    assert result["success"] is True
    assert result["asset_id"].startswith("asset_")
    assert result["operation_id"].startswith("op_")
    assert state.working_file == str(output_path.resolve())
    assert state.timeline[-1]["assets"] == [result["asset_id"]]


def _state(tmp_path: Path) -> ProjectState:
    now = utc_now_iso()
    return ProjectState(
        project_id="test-project",
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
