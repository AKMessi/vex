from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from asset_registry import AssetRecord, record_project_asset
from state import ProjectState
from timeline import normalize_timeline_operation


@dataclass(frozen=True)
class PromotionResult:
    output_path: str
    operation: dict[str, Any]
    asset: AssetRecord


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
    promoted_operation = normalize_timeline_operation(
        {
            **dict(operation),
            "result_file": str(resolved_output),
            "previous_file": previous_file,
            "assets": [asset.asset_id],
        }
    )

    old_working_file = state.working_file
    old_metadata = dict(state.metadata or {})
    old_timeline = [dict(item) for item in state.timeline]
    old_redo_stack = [dict(item) for item in state.redo_stack]
    try:
        state.working_file = str(resolved_output)
        state.metadata = dict(metadata)
        state.apply_operation(promoted_operation)
    except Exception:
        state.working_file = old_working_file
        state.metadata = old_metadata
        state.timeline = old_timeline
        state.redo_stack = old_redo_stack
        raise

    return PromotionResult(
        output_path=str(resolved_output),
        operation=promoted_operation,
        asset=asset,
    )
