from __future__ import annotations

import json
import os
from importlib import resources
from pathlib import Path

from engine import VideoEngineError, check_disk_space, estimate_output_size, export
from state import ProjectState

ENV_PRESETS_PATH = "VEX_EXPORT_PRESETS_PATH"


def _candidate_preset_paths() -> list[Path]:
    candidates: list[Path] = []
    override = str(os.getenv(ENV_PRESETS_PATH) or "").strip()
    if override:
        candidates.append(Path(override).expanduser())
    candidates.extend(
        [
            Path(__file__).resolve().parent.parent / "presets" / "export_presets.json",
            Path.cwd() / "presets" / "export_presets.json",
            Path.cwd() / "export_presets.json",
        ]
    )
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        normalized = str(path.resolve()) if path.exists() else str(path)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(path)
    return deduped


def load_presets() -> dict:
    searched: list[str] = []
    for path in _candidate_preset_paths():
        searched.append(str(path))
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    try:
        packaged = resources.files("presets").joinpath("export_presets.json")
        if packaged.is_file():
            searched.append("importlib.resources:presets/export_presets.json")
            return json.loads(packaged.read_text(encoding="utf-8"))
    except Exception:
        pass
    raise FileNotFoundError(
        "Export presets file not found. Looked in: " + ", ".join(searched)
    )


def execute(params: dict, state: ProjectState) -> dict:
    try:
        presets = load_presets()
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        return {
            "success": False,
            "message": str(exc),
            "suggestion": None,
            "updated_state": state,
            "tool_name": "export_video",
        }

    preset_name = params["preset_name"]
    if preset_name not in presets:
        return {
            "success": False,
            "message": f"Unknown export preset: {preset_name}",
            "suggestion": None,
            "updated_state": state,
            "tool_name": "export_video",
        }
    preset = dict(presets[preset_name])
    custom_settings = params.get("custom_settings") or {}
    preset.update({key: value for key, value in custom_settings.items() if value is not None})
    output_path = params.get("output_path")
    if output_path:
        output_path = os.path.abspath(output_path)
    else:
        suffix = preset.get("format") or "mp4"
        stem = "".join(ch for ch in state.project_name.replace(" ", "_") if ch.isalnum() or ch in {"_", "-"})
        output_path = os.path.join(state.output_dir, f"{stem}_{preset_name}.{suffix}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    estimate = estimate_output_size(state.working_file, preset)
    if not check_disk_space(output_path, estimate):
        return {
            "success": False,
            "message": f"Not enough disk space for export. Estimated size: {estimate / (1024 * 1024):.1f} MB.",
            "suggestion": None,
            "updated_state": state,
            "tool_name": "export_video",
        }
    try:
        saved = export(state.working_file, output_path, preset)
        return {
            "success": True,
            "message": f"Exported video to {saved}. Estimated size was {estimate / (1024 * 1024):.1f} MB.",
            "suggestion": None,
            "updated_state": state,
            "tool_name": "export_video",
        }
    except VideoEngineError as exc:
        return {
            "success": False,
            "message": str(exc),
            "suggestion": None,
            "updated_state": state,
            "tool_name": "export_video",
        }
