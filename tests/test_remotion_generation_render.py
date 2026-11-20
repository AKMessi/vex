from __future__ import annotations

import json
import math
import os
import struct
import wave
from pathlib import Path

import pytest

from engine import probe_video
from vex_visuals.evidence import file_digest
from video_generation import generate_video
from video_generation.renderer import CommandRecord, HyperframesVideoRuntime


@pytest.mark.skipif(os.getenv("VEX_RUN_RENDER_TESTS") != "1", reason="requires installed Remotion and FFmpeg")
@pytest.mark.parametrize("resolution,narrated", [("960x540", False), ("540x960", True)])
def test_full_remotion_render(tmp_path, monkeypatch, resolution, narrated):
    def synthesize(self, *, output_path, **kwargs):
        output_path.parent.mkdir(exist_ok=True)
        with wave.open(str(output_path), "wb") as audio:
            audio.setparams((1, 2, 8000, 0, "NONE", ""))
            audio.writeframes(b"".join(
                struct.pack("<h", round(math.sin(2 * math.pi * 440 * frame / 8000) * 4000))
                for frame in range(8000 * 6)
            ))
        return CommandRecord("tts_fixture", [], str(output_path.parent), 0, "")

    monkeypatch.setattr(HyperframesVideoRuntime, "tts", synthesize)
    result = generate_video({
        "script": (
            "Sparse attention changes dense token links into a focused reasoning path. "
            "The query selects relevant memory blocks and ignores irrelevant tokens."
        ),
        "renderer": "remotion",
        "resolution": resolution,
        "fps": 12,
        "workers": "2",
        "duration_sec": 6,
        "generate_audio": narrated,
        "transcribe_audio": False,
        "auto_install_runtime": False,
        "output_dir": str(tmp_path),
    })
    assert result.qa_passed, Path(result.qa_path).read_text(encoding="utf-8")
    qa = json.loads(Path(result.qa_path).read_text(encoding="utf-8"))
    timeline = json.loads(Path(result.timeline_path).read_text(encoding="utf-8"))["timeline"]
    receipt = qa["evidence"]["visual_quality"]
    output = Path(result.output_path)
    assert receipt["output_sha256"] == file_digest(output)
    assert len(receipt["typography_fallbacks"]) < len(timeline["scenes"])
    assert all(shot["frames"] for shot in receipt["shots"])
    assert receipt["shots"][-1]["frames"][-1]["frame"] == timeline["duration_frames"] - 1
    assert probe_video(str(output))["has_audio"] is narrated
    assert bool(timeline["captions"]) is narrated
    assert not list(output.parent.glob(".render-bundle-*"))
