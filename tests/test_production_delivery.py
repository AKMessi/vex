import json
import os
from pathlib import Path
import shutil
import subprocess
from zipfile import ZipFile

import pytest

import config
from engine import probe_video
from vex_runtime import production_delivery as delivery
from vex_runtime.production_edits import subtitle_patch
from vex_runtime.studio_projects import import_verified_film
from vex_visuals.evidence import file_digest, payload_digest
from video_generation.production_captions import parse_captions, serialize_captions
from video_generation.production_timeline import apply_timeline_edits


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _run(*args):
    return subprocess.check_output([config.FFMPEG_PATH, "-hide_banner", "-loglevel", "error", *map(str, args)])


@pytest.fixture
def saved_cut(tmp_path, monkeypatch):
    if not shutil.which(config.FFMPEG_PATH):
        pytest.skip("requires FFmpeg")
    library = tmp_path / "projects"
    monkeypatch.setattr(config, "AGENT_PROJECTS_DIR", str(library))
    source = tmp_path / "master"
    source.mkdir()
    master = source / "final.mp4"
    _run("-f", "lavfi", "-i", "testsrc2=s=320x320:r=24:d=2",
         "-f", "lavfi", "-i", "sine=f=440:r=48000:d=2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", master)
    timeline = {
        "version": "vex-remotion-video-v1", "width": 320, "height": 320, "fps": 24, "duration_frames": 48,
        "scenes": [{"id": "one", "producer": "local_media", "start_frame": 0, "duration_frames": 48,
                    "media": {"src": "raw.mp4", "kind": "video", "fit": "contain", "trim_start_frame": 0}}],
        "captions": [{"start_frame": 6, "duration_frames": 18, "text": "First & second\nNew line"},
                     {"start_frame": 26, "duration_frames": 18, "text": "Last cue"}],
        "audio": {"narration": "voice.wav"},
        "markers": [{"start_frame": 23, "text": "Chapter 2"}, {"start_frame": 32, "text": "Outside the clip"}],
    }
    public = source / "public"
    public.mkdir()
    shutil.copyfile(master, public / "raw.mp4")
    _run("-i", master, "-vn", public / "voice.wav")
    props = {"timeline": timeline}
    _write(source / "input_props.json", props)
    _write(source / "render_request.json", {"render_mode": "video", "video_format": "mp4", "video_quality": "standard"})
    manifest = source / "revision.json"
    _write(manifest, {"version": "vex-remotion-revision-v1", "status": "verified", "output": "final.mp4",
                      "receipt": {"output_sha256": file_digest(master), "timeline_sha256": payload_digest(props)},
                      "asset_sha256": {p.name: file_digest(p) for p in public.iterdir()}})
    return import_verified_film(master, manifest, "Delivery example", "offline", ""), master, timeline


def test_subtitle_roundtrip_keeps_text_and_frame_boundaries():
    cues = [{"start_frame": 1, "duration_frames": 11, "text": "A & B\nC < D → -->"},
            {"start_frame": 12, "duration_frames": 1, "text": "Привет"}]
    for kind in ("srt", "vtt"):
        subtitle = serialize_captions(cues, 24, kind=kind)
        assert "--&gt;" in subtitle
        assert parse_captions(subtitle, kind, fps=24, duration_frames=13) == cues
    vtt = "\ufeffWEBVTT\r\n\r\nNOTE Ignore this\r\n\r\nscene-a\r\n00:00.000 --> 00:01.000 align:center\r\n<v Narrator><b>Hello</b>"
    assert parse_captions(vtt, "vtt", fps=24, duration_frames=24) == [
        {"start_frame": 0, "duration_frames": 24, "text": "Hello"},
    ]


@pytest.mark.parametrize("text", [
    "1\n00:00:01,000 --> 00:00:02,000\nOut of bounds",
    "1\n00:00:00,000 --> 00:00:00,002\nSubframe",
    "1\n00:00:00,000 --> 00:00:00,900\nA\n\n2\n00:00:00,500 --> 00:00:00,800\nB",
    "1\n00:00:00,000 --> 00:00:01,000\n<b></b>",
])
def test_invalid_subtitle_track_rejects_the_whole_import(text):
    with pytest.raises(ValueError, match="cue"):
        parse_captions(text, "srt", fps=24, duration_frames=24)


def test_full_delivery_reuses_the_master_and_verifies_packages(tmp_path, saved_cut, monkeypatch):
    state, master, _ = saved_cut
    destination = tmp_path / "delivery.zip"
    monkeypatch.setattr(delivery, "_run_command", lambda *_: pytest.fail("A compatible master must not be re-encoded"))
    report = delivery.publish_delivery(state, destination, {"revision": state.revision})
    assert report["videos"] == [{"file": "cut.mp4", "method": "copied", "audio": True}]
    assert report["files"]["cut.mp4"]["sha256"] == file_digest(master)
    assert delivery.read_delivery(destination, verify=True) == report
    with pytest.raises(FileExistsError):
        delivery.publish_delivery(state, destination, {"revision": state.revision})
    spoiled = tmp_path / "spoiled.zip"
    with ZipFile(destination) as original, ZipFile(spoiled, "w") as changed:
        for name in original.namelist():
            data = original.read(name)
            changed.writestr(name, b"X" + data[1:] if name == "captions.srt" else data)
    with pytest.raises(ValueError, match="checksum"):
        delivery.read_delivery(spoiled, verify=True)


def test_clip_uses_the_saved_trim_offset_and_rebases_all_sidecars(tmp_path, saved_cut):
    state, master, _ = saved_cut
    state.timeline = [{"op": "trim_clip", "params": {"start": "1/4", "end": "7/4"}}]
    output = tmp_path / "range.zip"
    report = delivery.publish_delivery(state, output, {
        "revision": state.revision, "start": "3/8", "end": "1",
    })
    assert (report["start_frame"], report["end_frame"]) == (9, 24)
    assert report["caption_count"] == 2
    with ZipFile(output) as bundle:
        clip = Path(bundle.extract("cut.mp4", tmp_path))
        assert "00:00:00,000 --> 00:00:00,375" in bundle.read("captions.srt").decode()
        assert "00:00:00.458 --> 00:00:00.625" in bundle.read("captions.vtt").decode()
        assert bundle.read("chapters.vtt").decode() == "WEBVTT\n\n1\n00:00:00.333 --> 00:00:00.625\nChapter 2\n"
    assert probe_video(str(clip))["video_frame_count"] == 15
    assert probe_video(str(clip))["has_audio"]
    # Content starts on source frame 15 even though no keyframe was requested there.
    original_frame = _run("-i", master, "-vf", r"select=eq(n\,15)", "-frames:v", 1, "-pix_fmt", "gray", "-f", "rawvideo", "-")
    clip_frame = _run("-i", clip, "-frames:v", 1, "-pix_fmt", "gray", "-f", "rawvideo", "-")
    assert len(clip_frame) == len(original_frame) == 320 * 320
    assert sum(abs(a - b) for a, b in zip(original_frame, clip_frame, strict=True)) / len(clip_frame) < 2


def test_delivery_rejects_changed_source_or_stale_revision(tmp_path, saved_cut):
    state, master, timeline = saved_cut
    output = tmp_path / "never.zip"
    with pytest.raises(ValueError, match="cut changed"):
        delivery.publish_delivery(state, output, {"revision": state.revision + 1})
    with pytest.raises(ValueError, match="end time is exclusive"):
        delivery.publish_delivery(state, output, {"revision": state.revision, "start": "2"})
    with pytest.raises(ValueError, match="receipt"):
        _write(master.parent / "input_props.json", {"timeline": {**timeline, "captions": []}})
        delivery.publish_delivery(state, output, {"revision": state.revision})
    assert not output.exists()


def test_subtitle_import_builds_an_ordinary_uncommitted_proposal(saved_cut):
    state, _, timeline = saved_cut
    srt = "1\n00:00:00,125 --> 00:00:01,000\nNew caption"
    patch = subtitle_patch(state, state.revision, srt, "srt")
    after = apply_timeline_edits(timeline, patch["operations"])
    assert after["captions"] == [{"start_frame": 3, "duration_frames": 21, "text": "New caption"}]
    assert after["scenes"] == timeline["scenes"]
    assert after["audio"] == timeline["audio"]
    assert timeline["captions"][0]["text"].startswith("First")


@pytest.mark.skipif(os.getenv("VEX_RUN_RENDER_TESTS") != "1", reason="requires installed Remotion")
def test_clean_delivery_removes_caption_pixels_and_reuses_original_mix(tmp_path, saved_cut):
    from video_generation.remotion_revisions import revise_remotion_project

    state, master, _ = saved_cut
    captioned, manifest, evidence = revise_remotion_project(master.parent, replacements={}, workers="2")
    assert evidence["passed"]
    state = import_verified_film(captioned, manifest, "Captioned film", "offline", "")
    package = tmp_path / "dual.zip"
    report = delivery.publish_delivery(state, package, {"revision": state.revision, "captions": "both"})
    assert report["caption_count"] == 2
    with ZipFile(package) as bundle:
        assert set(bundle.namelist()) == {
            "cut-captioned.mp4", "cut-clean.mp4", "chapters.vtt", "captions.srt", "captions.vtt", "delivery.json",
        }
        burned = Path(bundle.extract("cut-captioned.mp4", tmp_path))
        clean = Path(bundle.extract("cut-clean.mp4", tmp_path))
    assert _run("-i", burned, "-map", "0:a", "-f", "s16le", "-") == _run("-i", clean, "-map", "0:a", "-f", "s16le", "-")
    def frame(path, number):
        return _run("-i", path, "-vf", f"select=eq(n\\,{number})", "-frames:v", 1, "-pix_fmt", "gray", "-f", "rawvideo", "-")

    assert file_digest(burned) == file_digest(captioned)
    original, alternate = frame(burned, 12), frame(clean, 12)
    assert sum(abs(a - b) > 20 for a, b in zip(original, alternate, strict=True)) > 200
