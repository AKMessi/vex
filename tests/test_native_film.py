import json
import os
from pathlib import Path

import pytest

from engine import probe_video
from video_generation import film_direction, generate_video, native_author
from video_generation.remotion_revisions import inspect_remotion_project


def _source(context):
    width, height, fps, frames = (context[key] for key in ("width", "height", "fps", "duration_frames"))
    duration = frames / fps
    if context["engine"] == "manim":
        return ('from manim import *\nclass VexScene(Scene):\n def construct(self):\n'
                f'  dot = Dot(color=YELLOW).scale(4)\n  self.add(dot)\n  self.play(dot.animate.shift(RIGHT * 3), run_time={duration!r})\n')
    if context["engine"] == "hyperframes":
        return f'''<!doctype html><html><body style="margin:0"><div id="root" data-composition-id="root" data-start="0"
 data-duration="{duration:.12f}" data-width="{width}" data-height="{height}" style="width:{width}px;height:{height}px;background:#184881">
 <div id="dot" style="position:absolute;top:30%;width:80px;height:80px;border-radius:50%;background:#fc4"></div></div>
 <script>const duration={duration};function seek(t){{document.querySelector('#dot').style.left=(15+60*t/duration)+'%';return timeline;}}
 const timeline={{duration:()=>duration,seek,time:seek,progress:p=>seek(p*duration),pause:()=>timeline,play:()=>timeline}};
 window.__timelines={{root:timeline}};seek(0);</script></body></html>'''
    return f'''import React from 'react';
import {{Composition, registerRoot, useCurrentFrame}} from 'remotion';
const Shot = () => <div style={{{{background:'#194f44',height:'100%',padding:40}}}}><div style={{{{width:80,height:80,
  borderRadius:40,background:'#fc4',marginLeft:5*useCurrentFrame()}}}} /></div>;
registerRoot(() => <Composition id="VexScene" component={{Shot}} width={{{width}}} height={{{height}}}
  fps={{{fps}}} durationInFrames={{{frames}}} />);
'''


def _treatment(tmp_path):
    file = tmp_path / "direction.json"
    artwork = {"background": "#112233", "view": [480, 320],
               "layers": [{"id": "dot", "svg": '<circle cx="100" cy="100" r="30" fill="#ffcc44"/>',
                           "keys": [[0, 0, 0, 1, 0, 1], [1, 80, 0, 1, 0, 1]]}]}
    file.write_text(json.dumps({"provider": "fixture", "model": "fixture", "palette": "signal", "shots": [
        {"renderer": "manim", "kind": "geometry", "action": "rotate", "intent": "Show a dot travel", "labels": [],
         "narration": ""},
        {"renderer": "hyperframes", "intent": "Continue the dot's movement", "narration": "", "artwork": artwork},
        {"renderer": "remotion", "intent": "Resolve the final moving dot", "narration": "", "artwork": artwork},
    ]}), encoding="utf-8")
    return file


def _author_fixture(monkeypatch):
    monkeypatch.setattr(native_author, "resolve_reasoning_model", lambda: ("fixture", "fixture"))
    calls = []

    def direct(provider, model, system, prompt, **_):
        context = json.loads(prompt)
        calls.append(context)
        return json.dumps({"source": _source(context)})

    monkeypatch.setattr(film_direction, "_request_film_direction", direct)
    return calls


def _film(tmp_path, *, render):
    return generate_video({"prompt": "Three linked scenes with a moving dot", "renderer": "hybrid",
                           "film_direction_path": str(_treatment(tmp_path)), "native_code": True, "duration_sec": 6,
                           "generate_audio": False, "resolution": "480x320", "fps": 24, "workers": "2",
                           "output_dir": str(tmp_path), "render": render}, checkpoint_dir=tmp_path / "checkpoints")


def test_initial_production_authors_native_source_at_the_shot_clock(tmp_path, monkeypatch):
    calls = _author_fixture(monkeypatch)
    output = _film(tmp_path, render=False)
    assert [row["engine"] for row in calls] == ["manim", "hyperframes", "remotion"]
    assert sum(row["duration_frames"] for row in calls) == 6 * 24
    assert all(row["pending"] for row in inspect_remotion_project(Path(output.project_dir))["shots"])
    # A retry in a new output folder reuses the authored source without more model calls.
    second = _film(tmp_path, render=False)
    assert len(calls) == 3
    assert Path(output.timeline_path).read_bytes() == Path(second.timeline_path).read_bytes()


@pytest.mark.skipif(os.getenv("VEX_SANDBOX_LIVE") != "1", reason="Requires Linux Docker and Remotion")
def test_three_model_authored_native_scenes_deliver_one_film(tmp_path, monkeypatch):
    calls = _author_fixture(monkeypatch)
    output = _film(tmp_path, render=True)
    assert output.qa_passed, output.qa_issues
    metadata = probe_video(output.output_path)
    assert metadata["video_frame_count"] == 6 * 24
    assert metadata["codec"] == "h264"
    production = json.loads((Path(output.project_dir) / "film_production.json").read_text(encoding="utf-8"))
    assert [row["producer"] for row in production["shots"]] == ["manim", "hyperframes", "remotion"]
    assert len(calls) == 3 and all(row["authoring"] == "native_code" and row["status"] == "staged" for row in production["shots"])
