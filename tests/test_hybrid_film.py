from __future__ import annotations

import json
import os
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

import broll_intelligence
import config
from video_generation import generate_video
from video_generation.film_direction import parse_film_direction
from video_generation.models import BeatGraph, TimedWord
from video_generation.remotion_project import _caption_cues
from video_generation.renderer import CommandRecord, HyperframesVideoRuntime


def treatment():
    return {
        "palette": "signal", "shots": [
            {"narration": "A scattered field contracts into a tight cluster.",
             "focus": "tight cluster", "kind": "geometry", "labels": ["scattered field", "tight cluster"],
             "action": "compress", "intent": "Notice the same dots converge."},
            {"narration": "Capture the cluster, filter the signal, then release a clean pulse.",
             "focus": "clean pulse", "kind": "process", "labels": ["Capture the cluster", "filter the signal", "release a clean pulse"],
             "action": "flow", "intent": "See the three production stages reveal in order."},
            {"narration": "The mass of noise becomes a readable signal.",
             "focus": "readable signal", "kind": "editorial", "labels": ["mass of noise", "readable signal"],
             "action": "contrast", "intent": "Contrast the starting state with the payoff."},
        ]
    }


def configure_fixture(monkeypatch, model="film-director-fixture"):
    monkeypatch.setattr(config, "PROVIDER", "openai_compatible")
    monkeypatch.setattr(config, "OPENAI_COMPAT_MODEL", model)
    calls = []

    def direct(provider, model, system, prompt, **kwargs):
        calls.append((provider, model, json.loads(prompt)))
        if "For a repair" in system:
            return json.dumps({"patches": [{"id": "beat_03", "changes": {"action": "reveal"}}]})
        return json.dumps(treatment())

    monkeypatch.setattr(broll_intelligence, "call_reasoning_model", direct)
    return calls


def test_direction_rejects_unsupported_or_ungrounded_shots():
    direction = treatment()
    direction["shots"][0]["labels"][0] = "12 million nodes"
    with pytest.raises(ValueError, match="quote its own narration"):
        parse_film_direction(direction, "fixture", "fixture")
    direction = treatment()
    direction["shots"][0]["marks"] = [{"before": [0, 0], "after": [0, 0]}] * 3
    with pytest.raises(ValueError, match="visible motion"):
        parse_film_direction(direction, "fixture", "fixture")
    direction = treatment()
    direction["shots"][0]["action"] = "shell command"
    with pytest.raises(ValueError, match="unsupported action"):
        parse_film_direction(direction, "fixture", "fixture")


def test_hybrid_project_compiles_a_provider_treatment_without_rendering(tmp_path, monkeypatch):
    calls = configure_fixture(monkeypatch)
    result = generate_video({
        "prompt": "Show a noisy signal becoming a readable signal", "renderer": "hybrid",
        "duration_sec": 12, "generate_audio": False, "render": False, "output_dir": str(tmp_path),
    })
    assert result.qa_passed, result.qa_issues
    assert calls[0][:2] == ("openai_compatible", "film-director-fixture")
    project = Path(result.project_dir)
    timeline = json.loads(Path(result.timeline_path).read_text())["timeline"]
    assert [shot["producer"] for shot in timeline["scenes"]] == ["manim", "hyperframes", "remotion"]
    assert sum(shot["duration_frames"] for shot in timeline["scenes"]) == timeline["duration_frames"]
    assert (project / "shots/beat_01/scene.py").is_file()
    assert (project / "shots/beat_02/index.html").is_file()


def test_caption_groups_split_at_native_shot_cuts():
    graph = BeatGraph("fixture", 2, "estimated_words", [], [
        TimedWord("sharp", .7, .95),
        TimedWord("Focus", 1.0, 1.2),
        TimedWord("now", 1.2, 1.5),
    ])
    cues = _caption_cues(graph, 24, 48, cut_frames=(24,))
    assert [cue["text"] for cue in cues] == ["sharp", "Focus now"]
    assert cues[0]["start_frame"] + cues[0]["duration_frames"] <= 24
    assert cues[1]["start_frame"] == 24


def test_lens_action_requires_optical_source_evidence():
    direction = treatment()
    direction["shots"][0]["action"] = "focus_rays"
    with pytest.raises(ValueError, match="spoken lens"):
        parse_film_direction(direction, "fixture", "fixture")


@pytest.mark.skipif(os.getenv("VEX_RUN_HYBRID_RENDER") != "1", reason="requires Manim, HyperFrames, Remotion and FFmpeg")
def test_hybrid_render_composes_real_native_shots_and_sound(tmp_path, monkeypatch):
    from video_generation import film_review

    configure_fixture(monkeypatch, "deepseek-flash")
    reviews = []

    class Reviewer:
        def __init__(self, provider):
            assert provider == "openai_compatible"

        def chat(self, messages, tools, system):
            parts = messages[0]["content"]
            assert sum(part["type"] == "image_url" for part in parts) == 9
            assert parts[-1]["image_url"]["url"].startswith("data:image/png;base64,")
            reviews.append(parts[0]["text"])
            shots = [
                {"id": f"beat_{i:02d}", "verdict": "revise" if i == 3 and len(reviews) == 1 else "pass",
                 "finding": "Make the payoff read in a direct reveal."}
                for i in range(1, 4)
            ]
            return SimpleNamespace(text=json.dumps({"shots": shots}))

        def close(self):
            pass

    monkeypatch.setattr(film_review, "OpenAICompatibleProvider", Reviewer)

    def tts(self, *, output_path, **kwargs):
        output_path.parent.mkdir(exist_ok=True)
        with wave.open(str(output_path), "wb") as stream:
            stream.setparams((1, 2, 8000, 0, "NONE", ""))
            stream.writeframes(bytes(8000 * 2 * 9))
        return CommandRecord("fixture_tts", [], str(output_path.parent), 0, "")

    monkeypatch.setattr(HyperframesVideoRuntime, "tts", tts)
    result = generate_video({
        "prompt": "Show a noisy signal becoming a readable signal", "renderer": "hybrid",
        "duration_sec": 12, "resolution": "480x320", "fps": 12, "workers": "2",
        "transcribe_audio": False, "auto_install_runtime": False, "output_dir": str(tmp_path),
    })
    assert result.qa_passed, Path(result.qa_path).read_text()
    assert result.has_audio
    project = Path(result.project_dir)
    report = json.loads((project / "film_production.json").read_text())
    assert [shot["status"] for shot in report["shots"]] == ["staged", "staged", "inline"]
    for shot in report["shots"][:2]:
        assert (project / shot["asset"]).stat().st_size > 1000
    evidence = json.loads(Path(result.qa_path).read_text())["evidence"]
    qa = evidence["visual_quality"]
    assert [shot["producer"] for shot in qa["shots"]] == ["manim", "hyperframes", "remotion"]
    assert all(len(shot["frames"]) == 3 for shot in qa["shots"])
    assert len(reviews) == 2
    assert evidence["film_review"]["passed"]
    assert json.loads((project / "film_repair.json").read_text())["accepted"]
    revised = json.loads((project / "film_repair/film_production.json").read_text())
    assert [shot["status"] for shot in revised["shots"]] == ["reused", "reused", "inline"]
    assert Path(result.output_path).is_relative_to(project / "film_repair")
