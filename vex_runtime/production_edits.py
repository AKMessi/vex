"""Project transactions for editable films, shared by Studio, CLI and the agent."""
from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from engine import probe_video
from state import ProjectState
from tools.promotion import promote_working_file
from vex_visuals.evidence import file_digest, payload_digest
from video_generation.production_timeline import apply_timeline_edits, cue_id
from video_generation.remotion_revisions import _read_object, _validate_timeline, revise_remotion_project


OPERATION = "edit_production"
REVISION_ID = re.compile(r"remotion-[a-z0-9_]{8}")


def composition_source(state: ProjectState) -> tuple[Path, list[dict]]:
    trims = []
    for operation in reversed(state.timeline):
        name = operation["op"]
        params = operation["params"]
        if name == OPERATION:
            manifest = _revision_folder(state, params.get("cut")) / "revision.json"
            break
        if name != "trim_clip":
            raise ValueError("This cut contains a baked edit. Undo that edit to work on the film's source shots.")
        trims.append({"op": "trim", "start": params["start"], "end": params.get("end")})
    else:
        artifact = state.artifacts.get("generated_source")
        path = artifact.get("manifest_path") if isinstance(artifact, dict) else None
        if not path:
            raise ValueError("This project has no saved film composition.")
        manifest = Path(path).resolve(strict=True)
    payload = _read_object(manifest)
    if payload.get("version") == "vex-remotion-revision-v1" and payload.get("status") == "verified":
        composition = manifest.parent
        output = composition / payload["output"]
    elif payload.get("version") == "generated-video-manifest-v1" and payload.get("qa", {}).get("passed"):
        timeline = Path(payload["artifacts"]["timeline_path"]).resolve(strict=True)
        if timeline.name != "input_props.json" or not timeline.is_relative_to(manifest.parent):
            raise ValueError("The production timeline must be inside its generated project.")
        composition = timeline.parent
        output = Path(payload["artifacts"]["output_path"])
    else:
        raise ValueError("This project has no verified Remotion delivery.")
    output = output.resolve(strict=True)
    if output.parent != composition or output.name not in {"final.mp4", "final.mov", "final.webm"}:
        raise ValueError("The verified delivery must be inside its generated composition.")
    return composition, list(reversed(trims))


def open_timeline(state: ProjectState) -> tuple[Path, dict]:
    source, trims = composition_source(state)
    timeline = apply_timeline_edits(_validate_timeline(_read_object(source / "input_props.json")), trims)
    return source, timeline


def inspect_production(state: ProjectState) -> dict:
    _, timeline = open_timeline(state)
    fps = timeline["fps"]
    return {
        "revision": state.revision,
        "width": timeline["width"], "height": timeline["height"], "fps": fps,
        "duration_frames": timeline["duration_frames"],
        "shots": [{
            "id": shot["id"], "number": index, "producer": shot.get("producer", "remotion"),
            "start_sec": shot["start_frame"] / fps,
            "end_sec": (shot["start_frame"] + shot["duration_frames"]) / fps,
            "start_frame": shot["start_frame"], "duration_frames": shot["duration_frames"],
            "narration": shot.get("standfirst", ""),
            "editable_artwork": not shot.get("media"),
        } for index, shot in enumerate(timeline["scenes"], 1)],
        "captions": [{"id": cue_id(cue), "text": cue["text"], "start_frame": cue["start_frame"],
                      "duration_frames": cue["duration_frames"]} for cue in timeline["captions"]],
        "tracks": {"narration": bool(timeline["audio"].get("narration") or timeline["audio"].get("voice_clips")),
                   "music": bool(timeline["audio"].get("music")), "music_gain": timeline["audio"].get("music_volume", 0)},
        "direction": state.artifacts.get("production_direction", {"notes": "", "locked_shots": []}),
    }


def _revision_folder(state: ProjectState, cut: object) -> Path:
    if not isinstance(cut, str) or not REVISION_ID.fullmatch(cut):
        raise ValueError("Invalid production cut ID.")
    root = Path(state.working_dir).resolve() / "production_cuts"
    path = root / cut
    if path.resolve(strict=True) != path or not path.is_dir():
        raise ValueError("The production cut is unavailable.")
    return path


def _patch(state: ProjectState, patch: dict) -> tuple[Path, list[dict], dict]:
    if type(patch.get("revision")) is not int or patch["revision"] != state.revision:
        raise ValueError("The project changed. Inspect the current cut and create a new proposal.")
    operations = patch.get("operations")
    if not isinstance(operations, list) or not 1 <= len(operations) <= 20:
        raise ValueError("Provide 1 to 20 scoped production operations.")
    source, trims = composition_source(state)
    path = source / "input_props.json"
    original = _validate_timeline(_read_object(path))
    edited = apply_timeline_edits(original, [*trims, *operations])
    current = apply_timeline_edits(original, trims)
    if edited == current:
        raise ValueError("This proposal does not change the cut.")
    _verify_locks(state, current, edited)
    return source, [*trims, *operations], edited


def save_direction(state: ProjectState, request: dict) -> dict:
    inventory = inspect_production(state)
    if type(request.get("revision")) is not int or request["revision"] != state.revision:
        raise ValueError("The project changed. Refresh before saving its direction.")
    notes, locked = request.get("notes", ""), request.get("locked_shots", [])
    choices = {shot["id"] for shot in inventory["shots"]}
    if not isinstance(notes, str) or len(notes) > 3000:
        raise ValueError("Keep the project direction under 3,000 characters.")
    if not isinstance(locked, list) or any(not isinstance(item, str) or item not in choices for item in locked):
        raise ValueError("Lock only shots that are in the current cut.")
    state.artifacts["production_direction"] = {"notes": notes.strip(), "locked_shots": sorted(set(locked))}
    state.save()
    return {"success": True, "message": "Project direction saved.", "revision": state.revision}


def _verify_locks(state: ProjectState, before: dict, after: dict) -> None:
    locked = set(state.artifacts.get("production_direction", {}).get("locked_shots", []))
    if not locked:
        return
    if any(before[key] != after[key] for key in ("width", "height", "audio")):
        raise ValueError("Unlock the protected shots before changing the canvas or audio.")
    current = {shot["id"]: shot for shot in after["scenes"]}
    for shot in before["scenes"]:
        if shot["id"] not in locked:
            continue
        start, end = shot["start_frame"], shot["start_frame"] + shot["duration_frames"]

        def cues(timeline):
            return [(row["start_frame"], row["duration_frames"], row["text"]) for row in timeline["captions"]
                    if row["start_frame"] < end and row["start_frame"] + row["duration_frames"] > start]

        if current.get(shot["id"]) != shot or cues(before) != cues(after):
            raise ValueError(f"Unlock the protected shot {shot['id']} before changing it.")


def _root(state: ProjectState) -> Path:
    root = Path(state.working_dir).resolve() / "production_cuts"
    if root.resolve() != root:
        raise ValueError("Production cuts must stay in the project workspace.")
    return root


def _preview_frames(timeline: dict, operations: list[dict]) -> tuple[int, ...]:
    frames = set()
    for operation in operations:
        if operation["op"] == "caption":
            cue = next((row for row in timeline["captions"] if cue_id(row) == operation["id"]), None)
            if cue:
                frames.add(cue["start_frame"] + cue["duration_frames"] // 2)
        if operation["op"] in {"artwork", "narration", "manim", "hyperframes"}:
            shot = next((row for row in timeline["scenes"] if row["id"] == operation["id"]), None)
            if shot:
                frames.update(shot["start_frame"] + int((shot["duration_frames"] - 1) * part) for part in (0.2, 0.55, 0.9))
    if not frames:
        frames.update(shot["start_frame"] + shot["duration_frames"] // 2 for shot in timeline["scenes"][:12])
    return tuple(sorted(frames)[:12])


def preview_production(state: ProjectState, patch: dict) -> dict:
    source, edits, timeline = _patch(state, patch)
    path = source / "input_props.json"
    source_digest = file_digest(path)
    frames = _preview_frames(timeline, patch["operations"])
    _, manifest, receipt = revise_remotion_project(source, replacements={}, edits=edits, preview_frames=frames, output_dir=_root(state))
    if file_digest(path) != source_digest:
        raise ValueError("The source composition changed during preview.")
    draft = {"patch": patch, "source_sha256": source_digest, "receipt_sha256": payload_digest(receipt)}
    (manifest.parent / "proposal.json").write_text(json.dumps(draft, allow_nan=False), encoding="utf-8")
    report = _read_object(manifest)
    speech = [name for name in report["asset_sha256"] if name.startswith("speech_")]
    return {"cut": manifest.parent.name, "revision": state.revision, "frames": receipt["frames"], "speech": speech}


def propose_production(state: ProjectState, request: dict) -> dict:
    """Ask the project's model for a small set of operations against a selection."""
    from broll_intelligence import extract_json_object
    from providers import resolve_reasoning_model
    from video_generation.film_direction import ARTWORK_CONTRACT, _request_film_direction
    from vex_runtime.visual_run import visual_run

    selection = request.get("shot")
    prompt = request.get("instruction")
    if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 2000:
        raise ValueError("Describe the edit in 1 to 2,000 characters.")
    inventory = inspect_production(state)
    if type(request.get("revision")) is not int or request["revision"] != inventory["revision"]:
        raise ValueError("The project changed. Refresh it before proposing an edit.")
    shot = next((row for row in inventory["shots"] if row["id"] == selection), None)
    if shot is None:
        raise ValueError("Select the shot you want to direct.")
    if selection in inventory["direction"]["locked_shots"]:
        raise ValueError("Unlock the protected shot before proposing changes.")
    provider, model = resolve_reasoning_model(state.provider, state.model or None)
    if not provider:
        raise ValueError("Configure a text model for natural-language shot edits.")
    source, trims = composition_source(state)
    timeline = apply_timeline_edits(_validate_timeline(_read_object(source / "input_props.json")), trims)
    scene = next(row for row in timeline["scenes"] if row["id"] == selection)
    start, end = shot["start_frame"], shot["start_frame"] + shot["duration_frames"]
    captions = [cue for cue in inventory["captions"] if start <= cue["start_frame"]
                and cue["start_frame"] + cue["duration_frames"] <= end]
    context = json.dumps({"instruction": prompt, "shot": shot, "artwork": scene.get("artwork"),
                          "captions": captions,
                          "project_direction": inventory["direction"]["notes"],
                          "canvas": [inventory["width"], inventory["height"]]}, ensure_ascii=False)
    system = (
        'Return only JSON {"operations":[...]}. Edit only the selected shot. '
        'Operations are {"op":"caption","id":"cue id","text":"new text"} or '
        '{"op":"artwork","id":"shot id","artwork":{...}} for browser-authored shots, '
        '{"op":"narration","id":"shot id","text":"a new spoken line shorter than the shot"}, '
        'or {"op":"manim","id":"shot id","geometry":{"action":"curve_morph","intent":"viewer objective",'
        '"marks":[{"before":[0,0],"after":[0,0]}]}} for a requested mathematical diagram. '
        'Curve morph needs 3 to 12 corresponding normalized points in drawing order. '
        'Use {"op":"hyperframes","id":"shot id","artwork":{...}} to author a dedicated graphics shot. '
        'Preserve existing artwork properties the user did not ask to change. '
        'Return an empty array if the request cannot be represented. ' + ARTWORK_CONTRACT
    )
    with visual_run(state.working_dir, max_model_calls=1, max_tokens=len(system + context) + 8192, timeout_sec=180):
        response = _request_film_direction(provider, model, system, context, thinking=False, token_limit=8192)
    result = json.loads(extract_json_object(response))
    operations = result.get("operations") if isinstance(result, dict) else None
    ids = {cue["id"] for cue in captions}
    if not isinstance(operations, list) or not operations:
        raise ValueError("The model did not propose a supported change for the selection.")
    for operation in operations:
        if not isinstance(operation, dict) or not (
            operation.get("op") == "caption" and isinstance(operation.get("id"), str) and operation["id"] in ids
            or operation.get("op") in {"artwork", "narration", "manim", "hyperframes"} and operation.get("id") == selection
        ):
            raise ValueError("The proposal exceeded the selected shot.")
    return {"revision": state.revision, "operations": operations}


def preview_speech(state: ProjectState, cut: str, name: str) -> Path:
    folder = _revision_folder(state, cut)
    if not re.fullmatch(r"speech_[a-f0-9]{64}\.wav", name):
        raise ValueError("Choose a recorded narration take.")
    report = _read_object(folder / "revision.json")
    path = folder / "public" / name
    if (report.get("status") != "previewed" or path.resolve(strict=True) != path
            or file_digest(path) != report["asset_sha256"].get(name)):
        raise ValueError("The preview take is unavailable.")
    return path


def review_production(state: ProjectState, cut: str) -> dict:
    """Request a second opinion on the proof frames without changing the draft."""
    from providers import resolve_reasoning_model
    from providers.multimodal import request_visual_json
    from vex_runtime.visual_run import visual_run
    from video_generation.film_direction import _request_film_direction
    from video_generation.film_model import uses_deepseek_json

    folder = _revision_folder(state, cut)
    report = _read_object(folder / "revision.json")
    draft = _read_object(folder / "proposal.json")
    source, _, _ = _patch(state, draft["patch"])
    receipt = report["receipt"]
    if (report.get("status") != "previewed" or payload_digest(receipt) != draft["receipt_sha256"]
            or file_digest(source / "input_props.json") != draft["source_sha256"]):
        raise ValueError("Review requires a verified preview of the current cut.")
    frames = tuple(_verify_preview_frame(folder, receipt, row["frame"]) for row in receipt["frames"][:4])
    provider, model = resolve_reasoning_model(state.provider, state.model or None)
    if not provider or (provider not in {"gemini", "claude", "groq"} and not uses_deepseek_json(provider, model)):
        raise ValueError("Choose a configured vision model to review the proposed frames.")
    prompt = (
        'Review the preview frames in chronological order. Return {"observations":["..."]} with '
        'up to three short, concrete notes on readability, visible clarity, or possible problems. '
        'Distinguish what is visible from what a still cannot establish about motion or sound.'
    )
    with visual_run(state.working_dir, max_model_calls=1, max_tokens=32_768, timeout_sec=180):
        if uses_deepseek_json(provider, model):
            result = json.loads(_request_film_direction(
                provider, model, "Review the actual video preview. Return JSON only.", prompt,
                thinking=False, token_limit=1024, images=frames,
            ))
        else:
            result = request_visual_json(provider, model, prompt, list(frames))
    notes = result.get("observations") if isinstance(result, dict) else None
    if not isinstance(notes, list) or any(not isinstance(note, str) for note in notes):
        raise ValueError("The model did not return an actionable preview review.")
    review = {"provider": provider, "model": model, "observations": [note[:400] for note in notes[:3]],
              "frames": [row["frame"] for row in receipt["frames"][:4]], "preview_sha256": draft["receipt_sha256"]}
    (folder / "review.json").write_text(json.dumps(review, ensure_ascii=False), encoding="utf-8")
    speech = [name for name in report["asset_sha256"] if name.startswith("speech_")]
    return {"success": True, "message": "Preview review ready.", "cut": cut, "revision": state.revision,
            "frames": receipt["frames"], "speech": speech, "review": review}


def preview_image(state: ProjectState, cut: str, frame: int) -> Path:
    folder = _revision_folder(state, cut)
    report = _read_object(folder / "revision.json")
    if report.get("status") != "previewed":
        raise ValueError("This cut has no preview.")
    return _verify_preview_frame(folder, report["receipt"], frame)


def _verify_preview_frame(folder: Path, receipt: dict, frame: int) -> Path:
    evidence = next((item for item in receipt["frames"] if item["frame"] == frame), None)
    path = folder / "remotion_preview_frames" / f"frame_{frame:08d}.png"
    if not evidence or path.resolve(strict=True) != path or file_digest(path) != evidence["sha256"]:
        raise ValueError("The requested preview frame is unavailable.")
    return path


def commit_production(state: ProjectState, cut: str, *, on_progress=None) -> dict:
    folder = _revision_folder(state, cut)
    draft = _read_object(folder / "proposal.json")
    report = _read_object(folder / "revision.json")
    if report.get("status") != "previewed" or payload_digest(report["receipt"]) != draft["receipt_sha256"]:
        raise ValueError("The proposal has no matching preview evidence.")
    for row in report["receipt"]["frames"]:
        _verify_preview_frame(folder, report["receipt"], row["frame"])
    source, edits, _ = _patch(state, draft["patch"])
    if file_digest(source / "input_props.json") != draft["source_sha256"]:
        raise ValueError("The saved composition changed. Preview the edit again.")
    return deliver_production(state, source, edits, {}, expected=report, on_progress=on_progress)


def deliver_production(state: ProjectState, source: Path, edits: list[dict], replacements: dict[str, Path],
                       *, expected: dict | None = None, on_progress=None) -> dict:
    if on_progress:
        on_progress("Rendering the revised cut", 0.2)
    output, manifest, receipt = revise_remotion_project(source, replacements=replacements, edits=edits, output_dir=_root(state))
    if not receipt["passed"]:
        raise ValueError("The revised cut failed delivery checks.")
    if expected:
        delivery = _read_object(manifest)
        verified_inputs = ("source_timeline_sha256", "timeline_sha256", "asset_sha256", "runtime_sha256", "bundle_fingerprint")
        if any(expected[key] != delivery[key] for key in verified_inputs):
            raise ValueError("The render inputs changed since preview. Review a new preview.")
    audio = None
    if any(edit["op"] in {"trim", "remove_range", "reorder", "narration", "music_gain"} for edit in edits):
        from subprocess import SubprocessError
        from video_generation.production_audio import measure_audio

        try:
            audio = measure_audio(output)
        except (OSError, ValueError, SubprocessError):
            audio = {"error": "Audio measurement unavailable"}
        (manifest.parent / "audio_measurement.json").write_text(json.dumps(audio), encoding="utf-8")
    if on_progress:
        on_progress("Saving the verified cut", 0.94)
    _save_cut(state, output, manifest.parent)
    return {"success": True, "message": "Revised cut saved.", "revision": state.revision,
            **receipt["rendering"], **({"audio": audio} if audio is not None else {})}


def _save_cut(state: ProjectState, output: Path, folder: Path) -> None:
    promote_working_file(
        state, output, operation={"op": OPERATION, "params": {"cut": folder.name},
                                  "description": "Saved an editable production cut"},
        metadata=probe_video(str(output)), asset_source=OPERATION,
    )


def attach_footage(state: ProjectState) -> dict:
    """Give the current footage a production timeline and retain its source clock."""
    from video_generation.remotion_delivery import check_remotion_delivery
    from video_generation.remotion_project import TIMELINE_VERSION, stage_remotion_runtime
    from video_generation.remotion_revisions import _write

    try:
        composition_source(state)
    except ValueError:
        pass
    else:
        raise ValueError("This project already has an editable production cut.")
    source = Path(state.working_file).resolve(strict=True)
    metadata = probe_video(str(source))
    fps = metadata["fps"]
    frames = metadata.get("video_frame_count")
    if (source.suffix.lower() != ".mp4" or metadata["codec"] != "h264" or not frames or not fps
            or not fps.is_integer() or abs((metadata.get("video_duration_sec") or 0) * fps - frames) > .1):
        raise ValueError("Starting an editable cut from footage requires an H.264 MP4 with an integer frame clock.")
    timeline = {
        "version": TIMELINE_VERSION, "width": metadata["width"], "height": metadata["height"],
        "fps": int(fps), "duration_frames": frames, "captions": [],
        "audio": {"narration": "footage.mp4" if metadata["has_audio"] else "", "music": "", "music_volume": 0},
        "scenes": [{"id": "footage", "start_frame": 0, "duration_frames": frames, "producer": "local_media",
                    "media": {"src": "footage.mp4", "kind": "video", "fit": "contain", "trim_start_frame": 0}}],
    }
    _validate_timeline({"timeline": timeline})
    root = _root(state)
    root.mkdir(exist_ok=True)
    job = Path(tempfile.mkdtemp(prefix="remotion-", dir=root))
    stage_remotion_runtime(job)
    (job / "public").mkdir()
    output = job / "final.mp4"
    shutil.copyfile(source, output)
    shutil.copyfile(output, job / "public" / "footage.mp4")
    props = job / "input_props.json"
    _write(props, {"timeline": timeline})
    _write(job / "render_request.json", {"render_mode": "video", "video_format": "mp4", "video_quality": "standard"})
    receipt = check_remotion_delivery(props, output, metadata, reusable=False)
    if not receipt["passed"]:
        raise ValueError(f"Footage did not pass delivery checks: {', '.join(receipt['issues'])}")
    _write(job / "revision.json", {"version": "vex-remotion-revision-v1", "status": "verified",
                                   "output": output.name, "receipt": receipt})
    _save_cut(state, output, job)
    return {"success": True, "message": "Opened footage as an editable production cut.", "revision": state.revision}
