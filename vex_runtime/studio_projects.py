"""Create editable projects from imported footage or a verified film."""
from __future__ import annotations

import shutil
import uuid
from collections.abc import Callable
from pathlib import Path

import config
from engine import probe_video
from state import ProjectState, utc_now_iso
from vex_runtime.edit_graph import EditGraph


def create_project(
    video_path: str, name: str | None, provider_name: str, model_name: str,
    *, generated_manifest: Path | None = None, managed_source: bool = False,
    initialize: Callable[[ProjectState], None] | None = None,
) -> ProjectState:
    source = Path(video_path).resolve(strict=True)
    project_id = str(uuid.uuid4())
    working_dir = Path(config.AGENT_PROJECTS_DIR).expanduser().resolve() / project_id
    working_dir.mkdir(parents=True, exist_ok=False)
    working_file = working_dir / f"source_{source.name}"
    try:
        shutil.copy2(source, working_file)
        metadata = probe_video(str(working_file))
        graph = EditGraph.from_source(
            str(working_file),
            duration=metadata.get("duration_rational") or metadata.get("duration_sec"),
            fps=metadata.get("fps_ratio") or metadata.get("fps"),
        )
        output_dir = working_dir / "outputs" if managed_source else source.parent
        if managed_source:
            output_dir.mkdir()
        state = ProjectState(
            project_id=project_id,
            project_name=name or source.stem,
            created_at=utc_now_iso(),
            updated_at=utc_now_iso(),
            source_files=[str(working_file if managed_source else source)],
            working_file=str(working_file),
            working_dir=str(working_dir),
            output_dir=str(output_dir),
            metadata=metadata,
            edit_graph=graph.to_dict(),
            provider=provider_name,
            model=model_name,
            artifacts={"generated_source": {"manifest_path": str(generated_manifest)}} if generated_manifest else {},
        )
        if initialize:
            initialize(state)
        state.save()
        return state
    except Exception:
        if working_dir.resolve() == working_dir:
            shutil.rmtree(working_dir, ignore_errors=True)
        raise


def import_verified_film(output: Path, manifest: Path, title: str | None, provider: str, model: str) -> ProjectState:
    if output.suffix.lower() not in {".mp4", ".mov", ".webm"}:
        raise ValueError("Studio import requires MP4, MOV, or WebM.")
    return create_project(str(output), title, provider, model, generated_manifest=manifest.resolve(strict=True))
