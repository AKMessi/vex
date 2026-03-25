from __future__ import annotations

import json
import os
import tempfile
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


def _safe_stem(project_name: str) -> str:
    return "".join(ch for ch in project_name.replace(" ", "_") if ch.isalnum() or ch in {"_", "-"}) or "export"


def _default_output_path(state: ProjectState, preset_name: str, preset: dict) -> str:
    suffix = preset.get("format") or "mp4"
    return os.path.join(state.output_dir, f"{_safe_stem(state.project_name)}_{preset_name}.{suffix}")


def _fallback_output_candidates(state: ProjectState, preset_name: str, preset: dict) -> list[Path]:
    suffix = preset.get("format") or "mp4"
    filename = f"{_safe_stem(state.project_name)}_{preset_name}.{suffix}"
    candidates = [
        Path(state.working_dir) / "exports",
        Path.home() / "Vex Exports",
        Path(tempfile.gettempdir()) / "vex-exports",
        Path.cwd() / "exports",
    ]
    deduped: list[Path] = []
    seen: set[str] = set()
    for directory in candidates:
        normalized = str(directory)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(directory / filename)
    return deduped


def _is_writable_destination(path: Path) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        probe = path.parent / f".{path.stem}.write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except Exception:
        return False


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
        output_path = _default_output_path(state, preset_name, preset)
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
        if not params.get("output_path") and "permission denied" in str(exc).lower():
            fallback_errors: list[str] = []
            for fallback_path in _fallback_output_candidates(state, preset_name, preset):
                if not _is_writable_destination(fallback_path):
                    fallback_errors.append(f"{fallback_path} (not writable)")
                    continue
                fallback_estimate = estimate_output_size(state.working_file, preset)
                if not check_disk_space(str(fallback_path), fallback_estimate):
                    fallback_errors.append(f"{fallback_path} (insufficient disk space)")
                    continue
                try:
                    saved = export(state.working_file, str(fallback_path), preset)
                    return {
                        "success": True,
                        "message": (
                            f"Exported video to {saved}. "
                            f"The original output directory was not writable, so Vex used a fallback export path."
                        ),
                        "suggestion": None,
                        "updated_state": state,
                        "tool_name": "export_video",
                    }
                except VideoEngineError as fallback_exc:
                    fallback_errors.append(f"{fallback_path} ({fallback_exc})")
            return {
                "success": False,
                "message": f"{exc} Fallback export attempts: {'; '.join(fallback_errors)}",
                "suggestion": None,
                "updated_state": state,
                "tool_name": "export_video",
            }
        return {
            "success": False,
            "message": str(exc),
            "suggestion": None,
            "updated_state": state,
            "tool_name": "export_video",
        }
