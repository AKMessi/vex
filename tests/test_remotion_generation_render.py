from __future__ import annotations

import json
import math
import os
import struct
import subprocess
import wave
from pathlib import Path

import pytest
from PIL import Image

import config
from engine import probe_video
from vex_visuals.evidence import file_digest
from video_generation import generate_video
from video_generation.renderer import CommandRecord, HyperframesVideoRuntime
from video_generation.remotion_revisions import revise_remotion_project


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
        "quality": "standard" if narrated else "high",
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
    if narrated:
        return

    report = json.loads((output.parent / "shot_direction.json").read_text(encoding="utf-8"))
    assert report["preflight"]["status"] == "complete", report["preflight"]
    assert report["preflight"]["candidate_count"] >= 2
    assert any(
        issue.startswith("empty_bound_panel:")
        for shot in report["shots"]
        for candidate in shot["candidates"]
        for issue in candidate["issues"]
    )
    selected = [shot for shot in report["shots"] if shot.get("selection_basis") == "rendered_preflight"]
    assert selected
    for shot in selected:
        candidate = next(row for row in shot["rendered_preflight"] if row["program_id"] == shot["selected_program_id"])
        assert candidate["accepted"]
        assert len(candidate["frames"]) == 4
        assert all(
            not row["issues"] for row in shot["candidates"] if row.get("program_id") == shot["selected_program_id"]
        )
        for frame in candidate["frames"]:
            assert frame["sha256"] == file_digest(output.parent / frame["path"])

    source_digest = file_digest(output)
    clip = tmp_path / "clip.mp4"
    subprocess.run([
        config.FFMPEG_PATH, "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=blue:size=320x240:rate=12:duration=6",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip),
    ], check=True)
    still = tmp_path / "still.png"
    Image.new("RGB", (320, 240), "green").save(still)
    source_project = Path(result.project_dir)
    with pytest.raises(ValueError, match="Unknown shot IDs"):
        revise_remotion_project(source_project, replacements={"missing": still})
    revised, report, revised_receipt = revise_remotion_project(
        source_project, replacements={"beat_01": clip, "beat_02": still}, workers="2",
    )
    assert revised_receipt["passed"], report.read_text(encoding="utf-8")
    assert file_digest(output) == source_digest
    assert file_digest(revised) != source_digest
    assert revised_receipt["expected_duration_sec"] == receipt["expected_duration_sec"]
    for shot, channel in zip(revised_receipt["shots"], (2, 1), strict=True):
        assert shot["program_id"] is None
        assert shot["media"]["kind"] in {"image", "video"}
        with Image.open(revised.parent / shot["frames"][-1]["path"]) as frame:
            rgb = frame.convert("RGB").getpixel((frame.width // 2, frame.height // 2))
            assert rgb[channel] > max(component for index, component in enumerate(rgb) if index != channel) + 80
