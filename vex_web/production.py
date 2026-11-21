"""Studio presentation of the shared video generation pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import config
from engine import VideoEngineError
from video_generation import generate_video, normalize_generation_request
from video_generation.models import MAX_BRIEF_CHARS
from vex_runtime.studio_projects import import_verified_film


PRODUCTION_SLOT = "_studio-production"


class ProductionStopped(Exception):
    """The user stopped production at a stage boundary."""


def production_workspace() -> Path:
    root = Path(config.AGENT_PROJECTS_DIR).expanduser().resolve()
    workspace = root / ".vex-studio" / "production"
    if workspace.resolve() != workspace:
        raise ValueError("The Studio production workspace must be local to the project library.")
    return workspace


def production_request(payload: dict[str, Any]) -> dict[str, Any]:
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > MAX_BRIEF_CHARS:
        raise ValueError(f"Write a video brief between 1 and {MAX_BRIEF_CHARS:,} characters.")
    options = {name: payload[name] for name in (
        "prompt", "title", "duration_sec", "aspect", "generate_audio", "renderer", "style",
    ) if name in payload}
    request = normalize_generation_request(options)
    return {
        "prompt": request.prompt, "title": request.title, "duration_sec": request.duration_sec,
        "aspect": request.aspect, "generate_audio": request.generate_audio,
        "renderer": request.renderer, "style": request.style if payload.get("style") else "",
    }


def produce_film(task_id: str, request: dict[str, Any], model: str, on_progress) -> dict[str, Any]:
    output_dir = _task_output_directory(task_id)
    result = generate_video({**request, "output_dir": str(output_dir)}, on_progress=on_progress)
    on_progress("Checking the production result", 0.96)
    public = {
        "title": result.title,
        "renderer": result.renderer, "duration_sec": result.duration_sec,
        "qa_passed": result.qa_passed, "issues": result.qa_issues[:8], "warnings": result.warnings[:6],
    }
    if result.output_path:
        output = Path(result.output_path).resolve(strict=True)
        public["render_file"] = output.relative_to(output_dir).as_posix()
        public["media_url"] = f"/api/tasks/{task_id}/media"
    if not result.qa_passed or "media_url" not in public:
        return {**public, "success": False, "message": "The film needs review before it can be opened for editing."}

    on_progress("Opening the film in Studio", 0.98)
    try:
        state = import_verified_film(
            output, Path(result.manifest_path), request["title"] or result.title, config.PROVIDER, model,
        )
    except (OSError, RuntimeError, ValueError, VideoEngineError):
        return {**public, "success": False, "message": "The export is ready, but Studio could not open an editable copy."}
    return {**public, "success": True, "project_id": state.project_id, "message": "Film ready to edit"}


def production_media(task_id: str, result: dict[str, Any]) -> Path:
    file = result.get("render_file")
    if not isinstance(file, str):
        raise FileNotFoundError("This run has no playable export.")
    directory = _task_output_directory(task_id)
    path = (directory / file).resolve(strict=True)
    if not path.is_relative_to(directory) or not path.is_file() or path.suffix.lower() not in {".mp4", ".mov", ".webm"}:
        raise FileNotFoundError("This run has no playable export.")
    return path


def _task_output_directory(task_id: str) -> Path:
    root = production_workspace() / "renders"
    directory = root / task_id
    if directory.parent != root or directory.resolve() != directory:
        raise ValueError("The production output must be inside its run directory.")
    return directory
