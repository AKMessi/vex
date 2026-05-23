from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Callable

import config
from effects.qa import validate_effect_output
from smartie import (
    SmartieBundle,
    SmartieBundleError,
    load_smartie_bundle,
    plan_smartie_attention_effects,
    validate_smartie_effect_plan,
)
from state import ProjectState, utc_now_iso


ProbeVideo = Callable[[str], dict[str, Any]]


class SmartieImportError(RuntimeError):
    pass


def import_smartie_bundle(
    bundle_path: str | Path,
    *,
    project: str | None = None,
    render: bool = False,
    provider_name: str | None = None,
    model_name: str | None = None,
    probe_video_fn: ProbeVideo | None = None,
) -> tuple[ProjectState, dict[str, Any]]:
    bundle = load_smartie_bundle(bundle_path)
    state = _prepare_project(
        bundle,
        project=project,
        provider_name=provider_name or config.PROVIDER,
        model_name=model_name or _default_model_name(),
        probe_video_fn=probe_video_fn,
    )

    metadata = state.metadata or _metadata_from_bundle(bundle)
    duration = float(metadata.get("duration_sec") or 0.0)
    width = int(metadata.get("width") or 0)
    height = int(metadata.get("height") or 0)
    fps = float(metadata.get("fps") or 30.0) or 30.0

    plan = plan_smartie_attention_effects(
        bundle,
        duration=duration,
        width=width,
        height=height,
        fps=fps,
    )
    plan_validation = validate_smartie_effect_plan(plan, duration=duration)
    if not plan_validation["ok"]:
        raise SmartieImportError("Smartie attention plan failed validation: " + "; ".join(plan_validation["errors"]))

    timestamp_label = utc_now_iso().replace(":", "-").replace("+00:00", "Z")
    import_root = Path(state.working_dir) / "smartie_imports"
    import_dir = import_root / timestamp_label
    import_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = import_dir / "manifest.json"
    plan_path = import_dir / "effect_plan.json"
    validation_path = import_dir / "validation.json"
    filtergraph_path = import_dir / "filtergraph.txt"

    import_manifest: dict[str, Any] = {
        "created_at": utc_now_iso(),
        "project_id": state.project_id,
        "project_name": state.project_name,
        "bundle": bundle.to_summary_dict(),
        "source_video": str(bundle.source_video),
        "working_file": state.working_file,
        "metadata": metadata,
        "effect_plan": plan.to_dict(),
        "validation": plan_validation,
        "rendered": False,
    }
    plan_path.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
    validation_path.write_text(json.dumps(plan_validation, indent=2), encoding="utf-8")

    render_validation: dict[str, Any] | None = None
    output_path: str | None = None
    if render and plan.effects:
        from engine import apply_timed_effects, probe_video

        source_metadata = dict(metadata)
        input_working_file = state.working_file
        output_path = apply_timed_effects(
            input_working_file,
            state.working_dir,
            plan.to_dict(),
            filtergraph_path=str(filtergraph_path),
        )
        output_metadata = probe_video(output_path)
        render_validation = validate_effect_output(source_metadata, output_metadata, plan)
        state.working_file = output_path
        state.metadata = output_metadata
        import_manifest["rendered"] = True
        import_manifest["input_working_file"] = input_working_file
        import_manifest["output_path"] = output_path
        import_manifest["render_validation"] = render_validation
        import_manifest["filtergraph_path"] = str(filtergraph_path)
        state.apply_operation(
            {
                "op": "add_smartie_effects",
                "params": {
                    "manifest_path": str(manifest_path),
                    "effect_plan": plan.to_dict(),
                    "source": "smartie_attention",
                },
                "timestamp": utc_now_iso(),
                "result_file": output_path,
                "description": f"Added {len(plan.effects)} Smartie attention zoom effects",
            }
        )

    manifest_path.write_text(json.dumps(import_manifest, indent=2), encoding="utf-8")
    latest_artifact = {
        "created_at": import_manifest["created_at"],
        "bundle_dir": str(bundle.root),
        "manifest_path": str(manifest_path),
        "plan_path": str(plan_path),
        "validation_path": str(validation_path),
        "effect_count": len(plan.effects),
        "rendered": bool(render and plan.effects),
        "output_path": output_path,
        "validation": plan_validation,
        "render_validation": render_validation,
    }
    state.artifacts["latest_smartie_import"] = latest_artifact
    history = list(state.artifacts.get("smartie_import_history") or [])
    history.append(latest_artifact)
    state.artifacts["smartie_import_history"] = history[-10:]
    state.save()

    result = {
        "project_id": state.project_id,
        "project_name": state.project_name,
        "working_dir": state.working_dir,
        "manifest_path": str(manifest_path),
        "plan_path": str(plan_path),
        "validation": plan_validation,
        "effect_count": len(plan.effects),
        "rendered": bool(render and plan.effects),
        "output_path": output_path,
    }
    return state, result


def _prepare_project(
    bundle: SmartieBundle,
    *,
    project: str | None,
    provider_name: str,
    model_name: str,
    probe_video_fn: ProbeVideo | None,
) -> ProjectState:
    existing = _load_existing_project(project)
    project_name = _project_name(project, bundle)
    if existing is not None:
        working_dir = Path(existing.working_dir)
        state = existing
    else:
        project_id = str(uuid.uuid4())
        working_dir = _project_working_dir(project, project_id)
        output_dir = working_dir / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        now = utc_now_iso()
        state = ProjectState(
            project_id=project_id,
            project_name=project_name,
            created_at=now,
            updated_at=now,
            source_files=[],
            working_file="",
            working_dir=str(working_dir),
            output_dir=str(output_dir),
            timeline=[],
            redo_stack=[],
            session_log=[],
            metadata={},
            artifacts={},
            provider=provider_name,
            model=model_name,
        )

    working_dir.mkdir(parents=True, exist_ok=True)
    copied_source = _copy_source_video(bundle.source_video, working_dir)
    state.source_files = [str(bundle.source_video)]
    state.working_file = str(copied_source)
    state.metadata = _probe_or_declared_metadata(bundle, str(copied_source), probe_video_fn)
    state.provider = provider_name
    state.model = model_name
    state.artifacts.setdefault("smartie", {})
    state.artifacts["smartie"] = {
        "bundle_dir": str(bundle.root),
        "source_video": str(bundle.source_video),
        "manifest_path": str(bundle.manifest.path),
        "attention_timeline_path": str(bundle.attention_timeline_path),
        "recording_smartie_path": str(bundle.smartie_metadata_path) if bundle.smartie_metadata_path else None,
        "preview_thumbnails_dir": str(bundle.preview_thumbnails_dir) if bundle.preview_thumbnails_dir else None,
        "attention_event_count": len(bundle.attention_points),
        "declared_metadata": bundle.manifest.declared_metadata,
    }
    state.save()
    return state


def _load_existing_project(project: str | None) -> ProjectState | None:
    if not project:
        return None
    try:
        return ProjectState.load(project)
    except (FileNotFoundError, ValueError):
        pass
    project_path = Path(project).expanduser()
    if _looks_like_path(project) and project_path.is_dir():
        for state_file in sorted(project_path.glob("*.json")):
            try:
                payload = json.loads(state_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            coerced = ProjectState._coerce_project_payload(payload)
            if coerced is not None:
                return ProjectState.from_dict(coerced)
    return None


def _project_working_dir(project: str | None, project_id: str) -> Path:
    if project and _looks_like_path(project):
        return Path(project).expanduser().resolve()
    return Path(config.AGENT_PROJECTS_DIR) / project_id


def _project_name(project: str | None, bundle: SmartieBundle) -> str:
    if project and not _looks_like_path(project):
        return str(project).strip()
    title = _manifest_title(bundle.manifest.raw)
    if title:
        return title
    return bundle.root.name or Path(bundle.source_video).stem


def _copy_source_video(source: Path, working_dir: Path) -> Path:
    suffix = source.suffix or ".webm"
    stem = _safe_stem(source.stem or "recording")
    target = working_dir / f"source_{stem}{suffix}"
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    return target


def _probe_or_declared_metadata(
    bundle: SmartieBundle,
    working_file: str,
    probe_video_fn: ProbeVideo | None,
) -> dict[str, Any]:
    metadata = _metadata_from_bundle(bundle)
    probe = probe_video_fn or _probe_video
    try:
        probed = probe(working_file)
    except Exception as exc:
        if _metadata_is_usable(metadata):
            metadata["probe_warning"] = str(exc)
            return metadata
        raise SmartieImportError(f"Could not probe Smartie source video metadata: {exc}") from exc
    merged = dict(metadata)
    merged.update(probed)
    if not _metadata_is_usable(merged):
        raise SmartieImportError("Smartie source video metadata is missing duration, fps, or resolution.")
    return merged


def _metadata_from_bundle(bundle: SmartieBundle) -> dict[str, Any]:
    metadata = dict(bundle.manifest.declared_metadata)
    metadata.setdefault("has_audio", True)
    if bundle.source_video.is_file():
        metadata["size_bytes"] = bundle.source_video.stat().st_size
    return metadata


def _metadata_is_usable(metadata: dict[str, Any]) -> bool:
    try:
        return (
            float(metadata.get("duration_sec") or 0.0) > 0.0
            and float(metadata.get("fps") or 0.0) > 0.0
            and int(metadata.get("width") or 0) > 0
            and int(metadata.get("height") or 0) > 0
        )
    except (TypeError, ValueError):
        return False


def _probe_video(path: str) -> dict[str, Any]:
    from engine import probe_video

    return probe_video(path)


def _default_model_name() -> str:
    provider = config.normalize_provider_name(config.PROVIDER)
    if provider == "gemini":
        return config.GEMINI_MODEL
    if provider == "claude":
        return config.CLAUDE_MODEL
    return config.local_llm_model(provider)


def _manifest_title(payload: dict[str, Any]) -> str:
    for key in ("title", "name", "recording_name", "recordingName", "project_name", "projectName"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _safe_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return stem or "recording"


def _looks_like_path(value: str) -> bool:
    return (
        os.sep in value
        or (os.altsep is not None and os.altsep in value)
        or "\\" in value
        or value.startswith(".")
        or value.startswith("~")
        or Path(value).is_absolute()
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import a Smartie recording bundle into Vex.")
    parser.add_argument("bundle_path", help="Path to the Smartie bundle directory.")
    parser.add_argument("--project", help="Existing Vex project id, project name, or project directory.")
    parser.add_argument("--render", action="store_true", help="Render Smartie attention zoom effects immediately.")
    args = parser.parse_args(argv)
    try:
        config.configure_runtime_logging()
        config.validate_config(require_provider=False)
        _state, result = import_smartie_bundle(
            args.bundle_path,
            project=args.project,
            render=args.render,
        )
    except (SmartieBundleError, SmartieImportError, OSError, ValueError) as exc:
        print(f"Smartie import failed: {exc}")
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
