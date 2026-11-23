import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest

import config
from engine import probe_video
from nle_interop import export_nle_bundle
from state import ProjectState
from vex_runtime.edit_graph import EditGraph
from vex_runtime.otio_interop import import_otio
from vex_runtime.studio_projects import create_project


otio = pytest.importorskip("opentimelineio")


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="Requires FFmpeg fixtures")
def test_otio_round_trip_keeps_cut_windows_and_linked_audio(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "AGENT_PROJECTS_DIR", str(tmp_path / "projects"))
    source = tmp_path / "source with spaces.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=320x180:r=24:d=3",
        "-f", "lavfi", "-i", "sine=f=330:d=3", "-c:v", "libx264", "-c:a", "aac", "-shortest", str(source),
    ], check=True, timeout=30)
    state = create_project(str(source), "Editorial handoff", "test", "test")
    state.edit_graph = EditGraph.from_source(state.working_file, duration=3, fps=24).cut(1, 2).to_dict()
    exported = export_nle_bundle(state, tmp_path, formats={"otio"})
    path = Path(exported.files["otio"])
    timeline = otio.adapters.read_from_file(str(path))
    assert len(timeline.tracks) == 2
    assert [clip.source_range.start_time.value for clip in timeline.tracks[0]] == [0, 48]
    assert [clip.source_range.duration.value for clip in timeline.tracks[1]] == [24, 24]
    restored = import_otio(path, None, None, "test", "test")
    graph = EditGraph.from_mapping(ProjectState.load(restored.project_id).edit_graph)
    assert graph.duration == 2
    assert graph.spans[1].source_start == Fraction(2)
    assert Path(graph.sources[graph.spans[0].source_id].media_path).parent.name == "media"
    assert probe_video(restored.working_file)["video_frame_count"] == 48
    assert probe_video(restored.working_file)["has_audio"]


@pytest.mark.parametrize("reference", ["https://example.com/external.mp4", "../external.mp4"])
def test_otio_rejects_media_outside_local_scope(reference, tmp_path):
    path = tmp_path / "cut.otio"
    timeline = otio.schema.Timeline()
    video = otio.schema.Track(kind=otio.schema.TrackKind.Video)
    video.append(otio.schema.Clip(media_reference=otio.schema.ExternalReference(target_url=reference),
                                  source_range=otio.opentime.TimeRange(otio.opentime.RationalTime(0, 24),
                                                                      otio.opentime.RationalTime(24, 24))))
    timeline.tracks.append(video)
    otio.adapters.write_to_file(timeline, str(path))
    with pytest.raises(ValueError, match="local media|outside the selected media"):
        import_otio(path, None, None, "test", "test")
