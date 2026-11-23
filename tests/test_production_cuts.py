from __future__ import annotations

import json

import pytest

import config
from vex_runtime import studio_projects
from vex_runtime.production_bundle import export_bundle, open_bundle
from vex_runtime.production_edits import inspect_production, preview_production
from video_generation.production_timeline import apply_timeline_edits, cue_id
from video_generation.remotion_incremental import _shot_ranges


def _timeline() -> dict:
    return {
        "version": "vex-remotion-video-v1", "width": 640, "height": 360, "fps": 24,
        "duration_frames": 96,
        "scenes": [
            {"id": name, "start_frame": i * 48, "duration_frames": 48, "standfirst": name}
            for i, name in enumerate(("first", "second"))
        ],
        "captions": [
            {"start_frame": 12, "duration_frames": 12, "text": "First"},
            {"start_frame": 52, "duration_frames": 18, "text": "Second"},
        ],
        "audio": {"narration": "voice.wav", "music": "", "music_volume": 0},
    }


def test_trim_split_reorder_preserve_original_motion_and_narration_coordinates():
    base = _timeline()
    first_id = cue_id(base["captions"][0])
    updated = apply_timeline_edits(base, [
        {"op": "trim", "start": "1/2", "end": "7/2"},
        {"op": "reorder", "shots": ["second", "first"]},
        {"op": "split", "id": "first", "at": "2"},
        {"op": "caption", "id": first_id, "text": "Keep this wording"},
    ])
    assert base["duration_frames"] == 96
    assert updated["duration_frames"] == 72
    assert [(row["start_frame"], row["duration_frames"], row["source_offset_frames"]) for row in updated["scenes"]] == [
        (0, 36, 0), (36, 12, 12), (48, 24, 24),
    ]
    assert [(clip["start_frame"], clip["trim_start_frame"], clip["duration_frames"])
            for clip in updated["audio"]["voice_clips"]] == [(0, 48, 36), (36, 12, 12), (48, 24, 24)]
    assert [(cue["start_frame"], cue["text"]) for cue in updated["captions"]] == [(4, "Second"), (36, "Keep this wording")]


def test_caption_change_invalidates_only_visible_frames():
    base = _timeline()
    updated = apply_timeline_edits(base, [{"op": "caption", "id": cue_id(base["captions"][1]), "text": "Edited"}])
    ranges, shots = _shot_ranges(base, updated)
    assert ranges == ((52, 70),)
    assert shots == {"second"}
    assert updated["audio"] == base["audio"]


def test_portable_bundle_checks_media_and_stale_patches(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AGENT_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setattr(studio_projects, "probe_video", lambda _: {
        "duration_sec": 4, "fps": 24, "width": 640, "height": 360,
    })
    source = tmp_path / "original"
    (source / "public").mkdir(parents=True)
    (source / "public/voice.wav").write_bytes(b"original sound")
    output = source / "final.mp4"
    output.write_bytes(b"original film")
    (source / "input_props.json").write_text(json.dumps({"timeline": _timeline()}), encoding="utf-8")
    (source / "render_request.json").write_text(json.dumps({
        "render_mode": "video", "video_format": "mp4", "video_quality": "standard",
    }), encoding="utf-8")
    manifest = source / "revision.json"
    manifest.write_text(json.dumps({
        "version": "vex-remotion-revision-v1", "status": "verified", "output": output.name,
    }), encoding="utf-8")
    state = studio_projects.import_verified_film(output, manifest, "Optics", "offline", "")
    patch = {"revision": state.revision, "operations": [{"op": "reorder", "shots": ["second", "first"]}]}
    state.save()
    with pytest.raises(ValueError, match="project changed"):
        preview_production(state, patch)

    bundle = export_bundle(state, tmp_path / "bundle")
    restored = open_bundle(bundle, "offline", "")
    assert inspect_production(restored)["shots"] == inspect_production(state)["shots"]
    (bundle / "public/voice.wav").write_bytes(b"altered")
    with pytest.raises(ValueError, match="integrity"):
        open_bundle(bundle, "offline", "")


def test_production_retry_restores_completed_direction_across_output_folders(tmp_path, monkeypatch):
    import providers
    from video_generation import pipeline

    calls = []
    original = pipeline._direct_generation

    def direct(*args):
        calls.append(True)
        return original(*args)

    class StopAfterDirection(Exception):
        pass

    monkeypatch.setattr(pipeline, "_direct_generation", direct)
    monkeypatch.setattr(config, "PROVIDER", "gemini")
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    monkeypatch.setenv("PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setattr(providers, "resolve_reasoning_model", lambda *args, **kwargs: ("", ""))
    events = []

    def stop(stage, progress):
        events.append(stage)
        if progress >= .40:
            raise StopAfterDirection()

    for name in ("first-run", "retry"):
        with pytest.raises(StopAfterDirection):
            pipeline.generate_video({
                "prompt": "Show how a lens focuses light", "renderer": "remotion",
                "generate_audio": False, "render": False, "output_dir": str(tmp_path / name),
            }, on_progress=stop, checkpoint_dir=tmp_path / "checkpoints")

    assert len(calls) == 1
    assert "Restored the saved film direction" in events


def test_external_render_job_launches_from_the_shared_browser_runtime(tmp_path, monkeypatch):
    from video_generation import remotion_project

    runtime = tmp_path / "managed-runtime"
    job = tmp_path / "external-library" / "project" / "revision"
    runtime.mkdir()
    job.mkdir(parents=True)
    monkeypatch.setattr(remotion_project, "resolve_node_executable", lambda: "node")
    monkeypatch.setattr(remotion_project, "_find_remotion_node_root", lambda _node: (runtime, ""))
    launches = []
    monkeypatch.setattr(remotion_project, "_run_logged", lambda argv, **options: launches.append((argv, options)))

    remotion_project.run_remotion_job(job, workers="1", auto_install_runtime=False, name="external_render")

    [(argv, launch)] = launches
    assert launch["cwd"] == runtime.resolve()
    assert argv[-1] == str(job.resolve())
    assert launch["log_path"] == job / "external_render.log"
