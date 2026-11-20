from __future__ import annotations

import json

from PIL import Image

from video_generation import remotion_delivery


def test_remotion_uses_video_stream_clock_when_aac_pads_container(tmp_path, monkeypatch):
    props = tmp_path / "input_props.json"
    props.write_text(json.dumps({"timeline": {
        "width": 960, "height": 540, "fps": 24, "duration_frames": 442,
        "audio": {"narration": "narration.wav", "music": ""},
        "scenes": [{"id": "beat_01", "start_frame": 0, "duration_frames": 442,
                    "media": {"kind": "video"}}],
    }}), encoding="utf-8")
    output = tmp_path / "final.mp4"
    output.write_bytes(b"fixture")

    def extract(_output, samples, **kwargs):
        for path, _ in samples:
            path.parent.mkdir(exist_ok=True)
            Image.new("RGB", (24, 24)).save(path)
        return [path for path, _ in samples], []

    monkeypatch.setattr(remotion_delivery, "extract_native_frames", extract)
    metadata = {
        "width": 960, "height": 540, "codec": "h264", "fps": 24, "has_audio": True,
        "duration_sec": 18.474667, "video_duration_sec": 18.416667,
    }
    assert remotion_delivery.check_remotion_delivery(props, output, metadata)["passed"]
    mismatched = {**metadata, "video_duration_sec": 18.125}
    result = remotion_delivery.check_remotion_delivery(props, output, mismatched)
    assert "rendered_frame_clock_duration_mismatch" in result["issues"]
