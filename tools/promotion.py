from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from asset_registry import AssetRecord, prepare_project_asset, sync_asset_registry_json
from content_cache import CacheEntry, prepare_cache_file, sync_cache_index_json
from state import ProjectState
from timeline import normalize_timeline_operation


@dataclass(frozen=True)
class PromotionResult:
    output_path: str
    operation: dict[str, Any]
    asset: AssetRecord
    cache_entry: CacheEntry


def promote_working_file(
    state: ProjectState,
    output_path: str | Path,
    *,
    operation: Mapping[str, Any],
    metadata: Mapping[str, Any],
    asset_kind: str = "video",
    asset_role: str = "timeline_result",
    asset_source: str = "",
    asset_metadata: Mapping[str, Any] | None = None,
) -> PromotionResult:
    resolved_output = Path(output_path).expanduser().resolve(strict=True)
    if not resolved_output.is_file():
        raise FileNotFoundError(f"Promoted output is not a file: {resolved_output}")

    previous_file = str(state.working_file or "")
    asset = prepare_project_asset(
        state,
        resolved_output,
        kind=asset_kind,
        role=asset_role,
        source=asset_source,
        metadata=dict(asset_metadata or {}),
        parents=[previous_file] if previous_file else [],
    )
    cache_entry = prepare_cache_file(
        state.working_dir,
        resolved_output,
        kind=asset_kind,
        metadata={"asset_id": asset.asset_id, **dict(asset_metadata or {})},
    )
    if (
        asset.checksum_sha256 != cache_entry.checksum_sha256
        or asset.size_bytes != cache_entry.size_bytes
    ):
        raise RuntimeError("Rendered output changed while it was being promoted; no project metadata was committed.")
    promoted_operation = normalize_timeline_operation(
        {
            **dict(operation),
            "result_file": str(resolved_output),
            "previous_file": previous_file,
            "assets": [asset.asset_id],
            "metadata": {
                **dict(operation.get("metadata") or {}),
                "cache_key": cache_entry.cache_key,
                "cached_path": cache_entry.cached_path,
            },
        }
    )

    snapshot = state.capture_snapshot()
    try:
        state.working_file = str(resolved_output)
        state.metadata = dict(metadata)
        state.timeline.append(promoted_operation)
        state.redo_stack.clear()
        state.save(asset_record=asset.to_dict(), cache_entry=cache_entry.to_dict())
    except Exception:
        state.restore_snapshot(snapshot, persist=False)
        raise

    # SQLite is authoritative. These JSON files are repairable projections.
    sync_asset_registry_json(state.working_dir)
    sync_cache_index_json(state.working_dir)

    return PromotionResult(
        output_path=str(resolved_output),
        operation=promoted_operation,
        asset=asset,
        cache_entry=cache_entry,
    )
