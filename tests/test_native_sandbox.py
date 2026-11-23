import os
from pathlib import Path

import pytest

from vex_runtime import native_sandbox


@pytest.fixture
def sandbox(monkeypatch):
    monkeypatch.setenv("VEX_SANDBOX_IMAGE", "sha256:" + "a" * 64)
    monkeypatch.setattr(native_sandbox.shutil, "which", lambda _: "docker")


def test_sandbox_only_exposes_explicit_inputs_and_publishes_verified_clip(sandbox, monkeypatch, tmp_path):
    asset = tmp_path / "art.png"
    asset.write_bytes(b"picture")
    calls = []

    def docker(args, *, timeout, stdout=None):
        calls.append(args)
        if args[1] == "run":
            options = set(args)
            assert {"--network=none", "--read-only", "--cap-drop=ALL", "--user=65534:65534",
                    "--security-opt=no-new-privileges", "--pids-limit=128", "--memory=2g"} <= options
            assert "--env" not in options
            mount = args[args.index("--mount") + 1]
            assert mount.endswith(",target=/input,readonly")
            directory = Path(mount.split(",")[1].removeprefix("source="))
            assert directory != tmp_path
            assert {item.name for item in directory.iterdir()} == {"scene.py", "art.png"}
        if stdout is not None:
            stdout.write(b"sandbox video")

    monkeypatch.setattr(native_sandbox, "_docker", docker)
    def probe(_, *, input_format):
        assert input_format == "mov"
        return dict(width=640, height=360, fps=24, video_frame_count=24, has_audio=False, codec="h264")

    monkeypatch.setattr(native_sandbox, "probe_video", probe)
    output = tmp_path / "new.mp4"
    result = native_sandbox.render_source("source", output, engine="manim", width=640, height=360,
                                          fps=24, frames=24, assets=(asset,))
    assert [argv[1] for argv in calls] == ["run", "exec", "exec", "rm"]
    assert result["sha256"] == native_sandbox.file_digest(output)
    assert output.read_bytes() == b"sandbox video"


def test_sandbox_removes_failed_container_and_does_not_publish(sandbox, monkeypatch, tmp_path):
    calls = []

    def docker(args, **_):
        calls.append(args[1])
        if args[1] == "exec":
            raise RuntimeError("render failed")

    monkeypatch.setattr(native_sandbox, "_docker", docker)
    with pytest.raises(RuntimeError, match="render failed"):
        native_sandbox.render_source("source", tmp_path / "new.mp4", engine="remotion", width=640,
                                     height=360, fps=24, frames=24)
    assert calls == ["run", "exec", "exec", "rm"]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(os.getenv("VEX_SANDBOX_LIVE") != "1", reason="Requires a locally built Docker image")
@pytest.mark.parametrize(("engine", "source"), [
    ("manim", "from manim import *\nclass VexScene(Scene):\n def construct(self):\n  self.add(Circle())\n  self.wait(1)"),
    ("remotion", """import React from 'react';
import {Composition, registerRoot, useCurrentFrame} from 'remotion';
const Moving = () => <div style={{marginLeft: useCurrentFrame(), background: 'blue'}}>Vex</div>;
registerRoot(() => <Composition id='VexScene' component={Moving} width={640} height={360}
  fps={24} durationInFrames={24}/>);
"""),
    ("hyperframes", """<!doctype html><html><head><meta charset='utf-8'></head><body>
<div id='root' data-composition-id='root' data-start='0' data-duration='1' data-width='640' data-height='360'
 style='background:blue;width:640px;height:360px'>Vex</div></body></html>"""),
])
def test_live_sandbox_render(engine, source, tmp_path):
    output = tmp_path / "scene.mp4"
    result = native_sandbox.render_source(source, output, engine=engine, width=640, height=360, fps=24, frames=24)
    assert result["duration_frames"] == 24
    assert output.stat().st_size > 1024
