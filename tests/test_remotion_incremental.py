from __future__ import annotations

import json
from pathlib import Path

import pytest

from vex_visuals.evidence import file_digest, payload_digest
from video_generation import remotion_incremental as incremental


def _timeline() -> dict:
    return {
        "width": 640, "height": 360, "fps": 24, "duration_frames": 96,
        "captions": [],
        "audio": {"narration": "voice.wav", "music": "", "music_volume": 0},
        "scenes": [
            {"id": f"shot-{i}", "start_frame": i * 24, "duration_frames": 24, "standfirst": str(i)}
            for i in range(4)
        ],
    }


def _stage(path: Path, timeline: dict) -> None:
    path.mkdir()
    (path / "public").mkdir()
    (path / "public/voice.wav").write_bytes(b"voice")
    (path / "input_props.json").write_text(json.dumps({"timeline": timeline}), encoding="utf-8")
    (path / "render_request.json").write_text(
        '{"render_mode":"video","video_format":"mp4","video_quality":"standard"}', encoding="utf-8",
    )
    (path / "remotion_result.json").write_text(
        json.dumps({"bundle_fingerprint": "a" * 64}), encoding="utf-8",
    )
    for name in incremental.RUNTIME_FILES:
        (path / name).write_bytes(name.encode())


def test_reuse_coalesces_adjacent_shots_and_keeps_the_frame_clock():
    before = _timeline()
    after = _timeline()
    after["scenes"][0]["standfirst"] = "First change"
    after["scenes"][1]["standfirst"] = "Second change"
    after["scenes"][3]["standfirst"] = "Third change"

    ranges, changed = incremental._shot_ranges(before, after)

    assert ranges == ((0, 48), (72, 96))
    assert changed == {"shot-0", "shot-1", "shot-3"}
    graph = incremental._splice_filter(ranges, 96, 24)
    assert "[0:v:0]trim=start_frame=48:end_frame=72" in graph
    assert graph.endswith("concat=n=3:v=1:a=0,settb=1/24,setpts=N[out]")


def test_reuse_refuses_a_caption_edit_or_a_changed_shot_boundary():
    before = _timeline()
    after = _timeline()
    after["captions"] = [{"text": "Changed"}]
    with pytest.raises(ValueError, match="captions or audio"):
        incremental._shot_ranges(before, after)
    after["captions"] = []
    after["scenes"][1]["duration_frames"] = 25
    with pytest.raises(ValueError, match="shot boundaries"):
        incremental._shot_ranges(before, after)


@pytest.mark.parametrize(("tamper", "error"), [
    ("source-asset", "source media"),
    ("revision-asset", "audio track"),
    ("source-movie", "no longer matches"),
    ("runtime", "rendering runtime"),
    ("reuse-depth", "repeated revisions"),
])
def test_receipt_requires_matching_inputs_and_a_fresh_picture(tmp_path, monkeypatch, tamper, error):
    original, revised = tmp_path / "source", tmp_path / "revision"
    before, after = _timeline(), _timeline()
    after["scenes"][1]["standfirst"] = "New visual"
    _stage(original, before)
    _stage(revised, after)
    output = original / "final.mp4"
    output.write_bytes(b"verified")
    metadata = {"codec": "h264", "pix_fmt": "yuv420p", "width": 640, "height": 360,
                "fps": 24, "video_duration_sec": 4, "video_frame_count": 96}
    monkeypatch.setattr(incremental, "probe_video", lambda _path: metadata)
    receipt = {"passed": True, "timeline_sha256": payload_digest({"timeline": before}),
               "output_sha256": file_digest(output)}
    incremental.record_delivery(original / "input_props.json", output, receipt)

    assert incremental._plan_reuse(original, revised).ranges == ((24, 48),)

    files = {
        "source-asset": original / "public/voice.wav", "revision-asset": revised / "public/voice.wav",
        "source-movie": output, "runtime": revised / "entry.jsx",
    }
    if tamper == "reuse-depth":
        incremental.record_delivery(original / "input_props.json", output, receipt, reuse_passes=2)
    else:
        files[tamper].write_bytes(b"modified")
    with pytest.raises(ValueError, match=error):
        incremental._plan_reuse(original, revised)
