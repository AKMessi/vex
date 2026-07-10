from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from asset_registry import AssetRecord, record_project_asset
from content_cache import CacheEntry, cache_project_file
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
    asset = record_project_asset(
        state,
        resolved_output,
        kind=asset_kind,
        role=asset_role,
        source=asset_source,
        metadata=dict(asset_metadata or {}),
        parents=[previous_file] if previous_file else [],
    )
    cache_entry = cache_project_file(
        state,
        resolved_output,
        kind=asset_kind,
        metadata={"asset_id": asset.asset_id, **dict(asset_metadata or {})},
    )
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
        state.apply_operation(promoted_operation)
    except BaseException:
        state.restore_snapshot(snapshot)
        raise

    return PromotionResult(
        output_path=str(resolved_output),
        operation=promoted_operation,
        asset=asset,
        cache_entry=cache_entry,
    )
