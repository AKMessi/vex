"""Portable editable production cuts with content-checked media."""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import config
from state import ProjectState
from vex_runtime.production_edits import composition_source
from vex_runtime.studio_projects import import_verified_film
from vex_visuals.evidence import file_digest
from video_generation.models import VALID_QUALITIES
from video_generation.production_timeline import audio_assets, picture_assets
from video_generation.remotion_revisions import _project_asset, _read_object, _validate_timeline


VERSION = "vex-production-bundle-v1"
DELIVERIES = {"final.mp4", "final.mov", "final.webm"}


def _files(source: Path, output: str) -> dict[str, Path]:
    if output not in DELIVERIES:
        raise ValueError("Expected a verified video delivery.")
    files = {name: source / name for name in ("input_props.json", "render_request.json", output)}
    for path in files.values():
        if path.resolve(strict=True) != path or not path.is_file():
            raise ValueError("Production files must stay inside their cut directory.")
    settings = _read_object(files["render_request.json"])
    if (settings.get("render_mode") != "video" or settings.get("video_format") != Path(output).suffix[1:]
            or settings.get("video_quality") not in VALID_QUALITIES):
        raise ValueError("The export settings do not match the delivery.")
    timeline = _validate_timeline(_read_object(files["input_props.json"]))
    names = audio_assets(timeline) | picture_assets(timeline)
    for name in names:
        path = _project_asset(source / "public", name)
        if path != source / "public" / name:
            raise ValueError("Bundle media must use regular files inside public.")
        files[f"public/{name}"] = path
    return files


def _copy_files(files: dict[str, Path], target: Path, expected: dict[str, str] | None = None) -> dict[str, str]:
    if expected is not None and expected.keys() != files.keys():
        raise ValueError("The bundle inventory does not match its composition.")
    digests = {}
    for name, source in files.items():
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, path)
        digests[name] = file_digest(path)
        if expected is not None and digests[name] != expected[name]:
            raise ValueError(f"Bundle integrity check failed for {name}.")
    return digests


def export_bundle(state: ProjectState, destination: Path) -> Path:
    source, trims = composition_source(state)
    if trims:
        raise ValueError("Save a production revision to include timeline trims in the portable cut.")
    output = Path(state.working_file).suffix.lower()
    name = f"final{output}"
    files = _files(source, name)
    target = destination.expanduser().resolve()
    target.mkdir(parents=True, exist_ok=False)
    digests = _copy_files(files, target)
    (target / "bundle.json").write_text(json.dumps({
        "version": VERSION, "title": state.project_name, "output": name, "files": digests,
    }, indent=2), encoding="utf-8")
    return target


def open_bundle(path: Path, provider: str, model: str) -> ProjectState:
    source = path.expanduser().resolve(strict=True)
    bundle = source / "bundle.json"
    if bundle.resolve(strict=True) != bundle or bundle.stat().st_size > 1024 * 1024:
        raise ValueError("The bundle manifest is unavailable.")
    metadata = _read_object(bundle)
    if metadata.get("version") != VERSION or not isinstance(metadata.get("files"), dict):
        raise ValueError("Unsupported Vex production bundle.")
    output = metadata.get("output")
    if not isinstance(output, str) or output not in DELIVERIES:
        raise ValueError("The bundle has no supported delivery.")
    files = _files(source, output)
    root = Path(config.AGENT_PROJECTS_DIR).expanduser().resolve() / ".production_imports"
    if root.resolve() != root:
        raise ValueError("The project library import directory must be local.")
    root.mkdir(parents=True, exist_ok=True)
    target = Path(tempfile.mkdtemp(prefix="cut-", dir=root))
    try:
        _copy_files(files, target, metadata["files"])
        manifest = target / "revision.json"
        manifest.write_text(json.dumps({
            "version": "vex-remotion-revision-v1", "status": "verified", "output": output,
        }), encoding="utf-8")
        title = metadata.get("title")
        return import_verified_film(target / output, manifest, title[:120] if isinstance(title, str) else None,
                                    provider, model)
    except Exception:
        # The import has no project reference until the copy and validation succeed.
        shutil.rmtree(target)
        raise
