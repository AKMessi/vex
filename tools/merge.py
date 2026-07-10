from __future__ import annotations

import os
from datetime import datetime, timezone

from engine import VideoEngineError, merge, probe_video
from sources import VIDEO_EXTENSIONS
from state import ProjectState
from tools.path_security import UnsafeInputPathError, resolve_existing_project_file


def execute(params: dict, state: ProjectState) -> dict:
    try:
        paths = params["file_paths"]
        if not isinstance(paths, list) or not paths:
            raise ValueError("file_paths must be a non-empty list of video paths.")
        if any(not isinstance(path, str) or not path.strip() for path in paths):
            raise ValueError("Each merge input must be a non-empty video path.")
        resolved = [
            os.path.abspath(
                str(
                    resolve_existing_project_file(
                        str(path),
                        state,
                        allowed_suffixes=VIDEO_EXTENSIONS,
                    )
                )
            )
            for path in paths
        ]
    except (KeyError, TypeError, ValueError, UnsafeInputPathError) as exc:
        return {
            "success": False,
            "message": str(exc),
            "suggestion": None,
            "updated_state": state,
            "tool_name": "merge_clips",
        }
    all_paths = [state.working_file, *resolved]
    snapshot = state.capture_snapshot()
    try:
        metadata = [probe_video(path) for path in all_paths]
        mismatched = len({(item["width"], item["height"]) for item in metadata}) > 1
        output_path = merge(all_paths, state.working_dir)
        state.working_file = output_path
        state.metadata = probe_video(output_path)
        op = {
            "op": "merge_clips",
            "params": {"file_paths": ["__CURRENT__", *resolved]},
            "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "result_file": output_path,
            "description": f"Merged {len(all_paths)} clips",
        }
        state.apply_operation(op)
        suggestion = None
        if mismatched:
            suggestion = "[SUGGESTION]: The merged clips had mismatched resolutions, so auto-scaling was applied - reply 'yes' to apply or continue."
        return {
            "success": True,
            "message": f"Merged {len(all_paths)} clips successfully.",
            "suggestion": suggestion,
            "updated_state": state,
            "tool_name": "merge_clips",
        }
    except (KeyError, TypeError, ValueError, VideoEngineError, OSError) as exc:
        state.restore_snapshot(snapshot)
        return {
            "success": False,
            "message": str(exc),
            "suggestion": None,
            "updated_state": state,
            "tool_name": "merge_clips",
        }
    except BaseException:
        state.restore_snapshot(snapshot)
        raise
