import os
import subprocess
from pathlib import Path

import pytest
from PIL import Image

import config
from engine import probe_video
from nle_interop import export_nle_bundle
from vex_runtime.otio_interop import import_otio
from vex_runtime.production_edits import open_timeline


otio = pytest.importorskip("opentimelineio")


@pytest.mark.skipif(os.getenv("VEX_RUN_RENDER_TESTS") != "1", reason="Requires the local Remotion runtime")
def test_multitrack_dissolve_and_independent_audio_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AGENT_PROJECTS_DIR", str(tmp_path / "projects"))
    source, sound = tmp_path / "picture.mp4", tmp_path / "sound.wav"
    overlay = tmp_path / "overlay.png"
    Image.new("RGBA", (32, 32), (37, 233, 124, 125)).save(overlay)
    commands = [
        ["-f", "lavfi", "-i", "testsrc2=s=640x360:r=24:d=3", "-c:v", "libx264", str(source)],
        ["-f", "lavfi", "-i", "sine=f=420:d=2", "-c:a", "pcm_s16le", str(sound)],
    ]
    for command in commands:
        subprocess.run([config.FFMPEG_PATH, "-v", "error", "-nostdin", *command], check=True, timeout=30)
    time = lambda frames: otio.opentime.RationalTime(frames, 24)

    def clip(name, start, length):
        return otio.schema.Clip(source_range=otio.opentime.TimeRange(time(start), time(length)),
                                media_reference=otio.schema.ExternalReference(target_url=name))

    doc = otio.schema.Timeline(name="Multitrack production")
    doc.global_start_time = time(0)
    base = otio.schema.Track(kind=otio.schema.TrackKind.Video)
    base.extend([clip(source.name, 0, 24), otio.schema.Transition(
        transition_type=otio.schema.TransitionTypes.SMPTE_Dissolve, in_offset=time(4), out_offset=time(4)),
        clip(source.name, 36, 24)])
    picture = otio.schema.Track(kind=otio.schema.TrackKind.Video)
    picture.extend([otio.schema.Gap(duration=time(8)), clip(overlay.name, 0, 12)])
    dialogue = otio.schema.Track(name="Dialogue", kind=otio.schema.TrackKind.Audio)
    dialogue.append(clip(sound.name, 0, 48))
    ambience = otio.schema.Track(name="Ambience", kind=otio.schema.TrackKind.Audio)
    ambience.extend([otio.schema.Gap(duration=time(5)), clip(sound.name, 10, 34)])
    doc.tracks.extend([base, picture, dialogue, ambience])
    doc.tracks.markers.append(otio.schema.Marker(name="Review the blend", marked_range=otio.opentime.TimeRange(time(24), time(0))))
    native = tmp_path / "original.otio"
    otio.adapters.write_to_file(doc, str(native))
    first = import_otio(native, None, None, "test", "test")
    _, timeline = open_timeline(first)
    assert [row["name"] for row in timeline["audio"]["tracks"]] == ["Dialogue", "Ambience"]
    assert len(timeline["overlays"]) == 3
    assert timeline["markers"] == [{"text": "Review the blend", "start_frame": 24}]
    assert probe_video(first.working_file)["has_audio"]
    exported = export_nle_bundle(first, tmp_path / "handoff", formats={"otio"})
    local = Path(exported.files["otio"])
    restored = import_otio(local, None, None, "test", "test")
    _, reopened = open_timeline(restored)
    assert len(reopened["audio"]["tracks"]) == len(timeline["audio"]["tracks"])
    assert reopened["markers"] == timeline["markers"]
    assert probe_video(restored.working_file)["video_frame_count"] == 48


@pytest.mark.skipif(os.getenv("VEX_RUN_RENDER_TESTS") != "1", reason="Requires the local Remotion runtime")
def test_studio_handoff_keeps_artwork_and_caption_as_separate_edits(tmp_path, monkeypatch):
    from vex_runtime.editorial_sources import deliver_editorial
    from vex_runtime.studio_projects import import_verified_film
    from video_generation.remotion_project import TIMELINE_VERSION

    monkeypatch.setattr(config, "AGENT_PROJECTS_DIR", str(tmp_path / "projects"))
    composition = tmp_path / "generated"
    (composition / "public").mkdir(parents=True)
    timeline = {"version": TIMELINE_VERSION, "fps": 24, "width": 640, "height": 360, "duration_frames": 24,
                "audio": {"narration": "", "music": "", "music_volume": 0},
                "captions": [{"start_frame": 8, "duration_frames": 12, "text": "A caption with its own timing"}],
                "scenes": [{"id": "opening_gap", "producer": "local_media", "gap": True, "start_frame": 0, "duration_frames": 2},
                           {"id": "dot_motion", "producer": "remotion", "start_frame": 2, "duration_frames": 22,
                            "artwork": {"background": "#112233", "view": [640, 360], "layers": [
                                {"id": "dot", "svg": '<circle cx="200" cy="160" r="80" fill="#FFCC55"/>',
                                 "keys": [[0, 0, 0, 1, 0, 1], [1, 160, 0, 1, 0, 1]]}]}}]}
    output = deliver_editorial(composition, timeline)
    state = import_verified_film(output, composition / "revision.json", "Clean overlays", "test", "test")
    result = export_nle_bundle(state, tmp_path / "handoff", formats={"otio"})
    restored = import_otio(Path(result.files["otio"]), None, None, "test", "test")
    _, reopened = open_timeline(restored)
    assert reopened["scenes"][0]["gap"] is True
    assert reopened["scenes"][1]["artwork"] == timeline["scenes"][1]["artwork"]
    assert reopened["captions"] == timeline["captions"]
