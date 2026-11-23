import shutil
import subprocess
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest
from PIL import Image

import config
from vex_runtime.edit_graph import EditGraph, VisualLayer
from vex_runtime.production_archive import export_history, open_history
from vex_runtime.production_edits import attach_footage, open_timeline
from vex_runtime.studio_projects import create_project
from video_generation.production_timeline import apply_timeline_edits


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="Requires FFmpeg fixtures")
def test_footage_retains_source_ranges_and_editable_overlays_in_an_archive(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "AGENT_PROJECTS_DIR", str(tmp_path / "projects"))
    source = tmp_path / "picture.mp4"
    overlay = tmp_path / "overlay.png"
    Image.new("RGBA", (80, 60), (35, 161, 100, 255)).save(overlay)
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=640x360:r=24:d=3",
                    "-f", "lavfi", "-i", "sine=f=330:d=3", "-c:v", "libx264", "-c:a", "aac", "-shortest", str(source)],
                   timeout=30, check=True)
    state = create_project(str(source), "Editable source", "test", "test")
    graph = EditGraph.from_source(state.working_file, duration=3, fps=24).cut(1, 2)
    graph = replace(graph, layers=(VisualLayer("brand", str(overlay), Fraction(1, 2), Fraction(3, 2),
                                              compose_mode="picture_in_picture"),))
    # A saved project can already contain source layers and an encoded working cut.
    baked = Path(state.working_dir) / "edited.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-i", state.working_file, "-t", "2", "-c", "copy", str(baked)],
                   timeout=30, check=True)
    state.working_file = str(baked)
    state.edit_graph = graph.to_dict()
    state.save()
    attach_footage(state)
    _, timeline = open_timeline(state)
    assert [scene["media"]["trim_start_frame"] for scene in timeline["scenes"]] == [0, 48]
    assert timeline["overlays"][0]["id"] == "brand"
    assert timeline["audio"]["tracks"][0]["clips"][1]["trim_start_frame"] == 48
    trimmed = apply_timeline_edits(timeline, [{"op": "trim", "start": "3/4", "end": "7/4"}])
    assert [(row["start_frame"], row["duration_frames"]) for row in trimmed["overlays"]] == [(0, 6), (6, 12)]
    restored = open_history(export_history(state, tmp_path / "archive"), "test", "test")
    root, reopened = open_timeline(restored)
    assert reopened == timeline
    assert (root / "public" / timeline["overlays"][0]["media"]["src"]).read_bytes() == overlay.read_bytes()
