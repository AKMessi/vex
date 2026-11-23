import os
from pathlib import Path

import pytest

from engine import probe_video
from vex_runtime.visual_run import VisualRun
from video_generation import pipeline
from video_generation.recovery import stage_files
from vex_visuals.evidence import file_digest


def test_checkpoint_restores_verified_files_and_rebuilds_corruption(tmp_path):
    checkpoint = VisualRun(tmp_path / "checkpoints")
    attempts = [tmp_path / f"attempt-{index}" for index in range(3)]
    for attempt in attempts:
        attempt.mkdir()
    executions = []

    def work(attempt):
        executions.append(attempt)
        output = attempt / "output.mp4"
        output.write_bytes(b"expected media")
        return {"name": output.name}, [output]

    for index, attempt in enumerate(attempts):
        result, reused = stage_files(checkpoint, "fixture", {"clock": 24}, attempt, lambda: work(attempt))
        assert (attempt / result["name"]).read_bytes() == b"expected media"
        assert reused is (index == 1)
        if index == 0:
            (attempt / "output.mp4").unlink()
        if index == 1:
            cached = next((checkpoint.root / "production-media").iterdir())
            cached.write_bytes(b"tampered")
    assert executions == [attempts[0], attempts[2]]


@pytest.mark.skipif(os.getenv("VEX_RUN_RENDER_TESTS") != "1", reason="Requires the local Remotion runtime")
def test_retry_reuses_real_render_after_worker_stops_before_review(monkeypatch, tmp_path):
    import config
    import providers

    monkeypatch.setattr(config, "PROVIDER", "gemini")
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    monkeypatch.setenv("PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setattr(providers, "resolve_reasoning_model", lambda *args, **kwargs: ("", ""))

    class WorkerStopped(Exception):
        pass

    first_delivery = []
    first_output = tmp_path / "first"

    def stop(stage, _progress):
        if stage == "Checking the exported frames":
            first_delivery.append(file_digest(next(first_output.rglob("final.mp4"))))
            raise WorkerStopped()

    request = {"prompt": "Show how a lens focuses light", "renderer": "remotion", "resolution": "640x360",
               "duration_sec": 6, "generate_audio": False, "fps": 24, "workers": "2"}
    checkpoint = tmp_path / "checkpoints"
    with pytest.raises(WorkerStopped):
        pipeline.generate_video({**request, "output_dir": str(first_output)}, on_progress=stop, checkpoint_dir=checkpoint)
    first_output.rename(tmp_path / "moved-first-attempt")

    def render_again(*args, **kwargs):
        raise AssertionError("A successful export must be restored without running the renderer again")

    monkeypatch.setattr(pipeline, "render_remotion_project", render_again)
    events = []
    result = pipeline.generate_video({**request, "output_dir": str(tmp_path / "retry")},
                                     on_progress=lambda label, _: events.append(label), checkpoint_dir=checkpoint)
    assert "Restored the rendered film" in events
    assert file_digest(Path(result.output_path)) == first_delivery[0]
    assert probe_video(result.output_path)["video_frame_count"] == 144
