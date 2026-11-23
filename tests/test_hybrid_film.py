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
from video_generation.film_artwork import validate_artwork
from video_generation.film_direction import direct_film, parse_film_direction
from video_generation.models import BeatGraph, ScriptPlan, TimedWord, VideoGenerationRequest
from video_generation.remotion_project import _caption_cues
from video_generation.remotion_revisions import inspect_remotion_project, revise_remotion_project
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


def configure_fixture(monkeypatch, model="film-director-fixture", repair=None):
    monkeypatch.setattr(config, "PROVIDER", "openai_compatible")
    monkeypatch.setattr(config, "OPENAI_COMPAT_MODEL", model)
    monkeypatch.setattr(config, "OPENAI_COMPAT_BASE_URL", "http://localhost:11434/v1")
    calls = []

    def direct(provider, model, system, prompt, **kwargs):
        calls.append((provider, model, json.loads(prompt)))
        result = treatment()
        result["palette"] = {"background": "#112233", "text": "#fff", "accent": "#aaf"}
        result["shots"][0]["renderer"] = "manim"
        result["shots"][0].pop("labels")
        result["shots"][0]["transition"] = "accent_wipe"
        for index, renderer in [(1, "hyperframes"), (2, "remotion")]:
            result["shots"][index]["renderer"] = renderer
            result["shots"][index]["artwork"] = {
                "background": "#112233", "view": [480, 320],
                "layers": [{"id": "signal", "svg": '<path d="M70 140 L180 95 L260 120 L380 75" stroke="#aaf" stroke-width="12" fill="none"/>',
                            "keys": [[0, -80, 0, 1, 0, .5], [1, 20, 0, 1, 0, 1]]}],
            }
        if "For a repair" in system:
            return json.dumps({"patches": repair(result)})
        return json.dumps(result)

    monkeypatch.setattr(broll_intelligence, "call_reasoning_model", direct)
    return calls


def test_narration_retry_keeps_valid_artwork_and_uses_actual_limits(monkeypatch):
    from dataclasses import replace

    from video_generation import film_direction

    fixture = configure_fixture(monkeypatch)
    authored = json.loads(broll_intelligence.call_reasoning_model("fixture", "fixture", "", "{}"))
    authored["shots"][1]["narration"] = " ".join(["Echo"] * 70) + "."
    expected = parse_film_direction(authored, "openai_compatible", "film-director-fixture", authored=True)
    repaired = [shot["narration"] for shot in treatment()["shots"]]
    budgets = []

    def author(provider, model, system, prompt, **kwargs):
        budgets.append(kwargs["token_limit"])
        assert prompt["canvas"] == [1920, 1080]
        assert (prompt["minimum_words"], prompt["maximum_words"]) == (23, 53)
        if len(budgets) == 1:
            return json.dumps(authored)
        assert system == film_direction.NARRATION_REPAIR_PROMPT
        assert "words" in prompt["validation_feedback"]
        assert len(prompt["shots"]) == 3
        assert not any("artwork" in shot for shot in prompt["shots"])
        return json.dumps({"narration": repaired})

    monkeypatch.setattr(
        film_direction, "_request_film_direction",
        lambda provider, model, system, prompt, **kwargs: author(provider, model, system, json.loads(prompt), **kwargs),
    )
    initial = ScriptPlan("Signal", "", "", "fixture", "Show a signal")
    _, direction = direct_film(VideoGenerationRequest(prompt=initial.prompt, duration_sec=16), initial)
    assert len(fixture) == 1
    assert budgets == [None, 2048]
    assert direction == replace(expected, shots=[
        replace(shot, narration=line) for shot, line in zip(expected.shots, repaired, strict=True)
    ])


def test_comparison_samples_a_short_animated_edit_and_neighboring_control():
    from video_generation.film_comparison import _sample_frames

    primary = {"id": "primary", "start_frame": 40, "duration_frames": 101}
    neighbor = {"id": "neighbor", "start_frame": 141, "duration_frames": 24}
    pulse = {
        **primary, "artwork": {"layers": [{"id": "pulse", "keys": [
            [.06, 0, 0, 1, 0, 0], [.14, 0, 0, 1, 0, 1], [.16, 0, 0, 1, 0, 0],
        ]}]},
    }
    assert _sample_frames([primary, neighbor], [pulse, neighbor], {"primary"}) == {
        "primary": [40, 54, 95, 140], "neighbor": [141],
    }


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


@pytest.mark.skipif(os.getenv("VEX_RUN_HYBRID_RENDER") != "1", reason="requires Manim, HyperFrames, Remotion and FFmpeg")
def test_saved_plan_can_be_prepared_previewed_and_delivered_without_model(monkeypatch, tmp_path):
    from PIL import Image
    from typer.testing import CliRunner

    from main import app
    from state import ProjectState
    from vex_visuals.evidence import file_digest

    calls = configure_fixture(monkeypatch)
    result = generate_video({
        "prompt": "Show a signal acquiring structure", "renderer": "hybrid",
        "duration_sec": 12, "resolution": "480x320", "fps": 12, "workers": "2",
        "generate_audio": False, "render": False, "output_dir": str(tmp_path),
    })
    source = Path(result.project_dir)
    source_props = Path(result.timeline_path).read_bytes()
    assert [shot["pending"] for shot in inspect_remotion_project(source)["shots"]] == [True, True, False]
    prepared, preparation_report, receipt = revise_remotion_project(
        source, replacements={}, stage_only=True,
    )
    assert receipt["status"] == "prepared"
    preparation = json.loads(preparation_report.read_text(encoding="utf-8"))
    assert len(preparation["asset_sha256"]) == 2
    assert preparation["native_commands"]
    assert not any(shot["pending"] for shot in inspect_remotion_project(prepared)["shots"])
    assert not list(prepared.glob("final.*"))
    assert Path(result.timeline_path).read_bytes() == source_props
    final_frame = json.loads(source_props)["timeline"]["duration_frames"] - 1

    previews, _, receipt = revise_remotion_project(
        prepared, replacements={}, preview_frames=(0, final_frame),
    )
    assert previews.parent.parent == prepared.parent
    assert [item["frame"] for item in receipt["frames"]] == [0, final_frame]
    assert not list(previews.parent.glob("final.*"))
    for item in receipt["frames"]:
        with Image.open(previews.parent / item["path"]) as image:
            assert image.size == (480, 320)
    assert json.loads((previews.parent / "render_request.json").read_text(encoding="utf-8"))["render_mode"] == "video"

    studio = tmp_path / "studio"
    deliveries = tmp_path / "deliveries"
    monkeypatch.setenv("AGENT_PROJECTS_DIR", str(studio))
    monkeypatch.setattr(config, "AGENT_PROJECTS_DIR", str(studio))
    cli = CliRunner().invoke(app, [
        "render-video-project", str(previews.parent), "--quality", "draft", "--format", "mov",
        "--output-dir", str(deliveries), "--add-to-studio",
    ])
    assert cli.exit_code == 0, cli.output
    assert "Added to Studio" in cli.output
    report = next(deliveries.glob("*/revision.json"))
    delivery = json.loads(report.read_text(encoding="utf-8"))
    assert delivery["receipt"]["passed"]
    assert delivery["receipt"]["shots"][0]["frames"][-1]["detail_fraction"] >= .002
    assert delivery["output"] == "final.mov"
    assert not delivery["native_commands"]
    state = ProjectState.load(next(studio.iterdir()).name)
    assert file_digest(Path(state.working_file)) == file_digest(report.parent / "final.mov")
    assert Path(state.artifacts["generated_source"]["manifest_path"]) == report.resolve()
    assert len(calls) == 1


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
    assert timeline["scenes"][0]["transition"] == "cut"
    assert sum(shot["duration_frames"] for shot in timeline["scenes"]) == timeline["duration_frames"]
    assert (project / "shots/beat_01/scene.py").is_file()
    assert "M70 140" in (project / "shots/beat_02/index.html").read_text(encoding="utf-8")
    assert timeline["scenes"][2]["artwork"]["layers"][0]["id"] == "signal"
    replay = generate_video({
        "prompt": "Replay this film", "renderer": "hybrid", "duration_sec": 12,
        "generate_audio": False, "render": False, "output_dir": str(tmp_path),
        "film_direction_path": str(project / "film_direction.json"),
    })
    assert len(calls) == 1
    assert replay.qa_passed
    assert json.loads(Path(replay.timeline_path).read_text())["timeline"] == timeline


@pytest.mark.parametrize(("weights", "frames"), [([1], [240]), ([1, 3], [60, 180])])
def test_automatic_hybrid_preserves_silent_shot_count_and_pacing(tmp_path, monkeypatch, weights, frames):
    monkeypatch.setattr(config, "PROVIDER", "ollama")
    monkeypatch.setattr(config, "OLLAMA_MODEL", "fixture-model")

    def direct(provider, model, system, prompt, **kwargs):
        assert (provider, model) == ("ollama", "fixture-model")
        brief = json.loads(prompt)
        assert brief["narration_requested"] is False
        assert brief["style"] == "Soft tungsten light, no labels"
        return json.dumps({
            "palette": {"background": "#112233", "text": "#fff", "accent": "#aaf"},
            "shots": [{
                "renderer": "remotion",
                "narration": "",
                "intent": "The distant light holds in frame.",
                "duration_weight": weight,
                "artwork": {"background": "#112233", "layers": [
                    {"id": "light", "svg": '<circle cx="500" cy="270" r="32" fill="#aaf"/>'},
                ]},
            } for weight in weights],
        })

    monkeypatch.setattr(broll_intelligence, "call_reasoning_model", direct)
    result = generate_video({
        "prompt": "Hold on a distant light.", "duration_sec": 10, "fps": 24,
        "style": "Soft tungsten light, no labels",
        "generate_audio": False, "render": False, "output_dir": str(tmp_path),
    })
    assert result.renderer == "hybrid"
    assert result.qa_passed, result.qa_issues
    timeline = json.loads(Path(result.timeline_path).read_text())["timeline"]
    assert [shot["duration_frames"] for shot in timeline["scenes"]] == frames
    assert not timeline["captions"]
    assert not Path(result.project_dir, "VIDEO_SKILL_GRAPH.json").exists()
    project = Path(result.project_dir)
    with pytest.raises(ValueError, match="within the timeline"):
        revise_remotion_project(project, replacements={}, preview_frames=(timeline["duration_frames"],))
    staged, _, receipt = revise_remotion_project(project, replacements={}, stage_only=True)
    assert receipt["passed"]
    assert json.loads((staged / "input_props.json").read_text())["timeline"] == timeline


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


def test_authored_shot_can_use_its_own_layout_color_and_sentence_grouping():
    shot = {
        "renderer": "remotion", "intent": "Show the focusing plane moving.",
        "narration": "The lens moves upstream. The focal plane shifts toward it.",
        "artwork": {"background": "#123456", "view": [960, 540], "layers": [
            {"id": "plane", "svg": '<line x1="600" y1="60" x2="600" y2="440" stroke="#fff"/>',
             "keys": [[0, 0, 0, 1, 0, 1], [1, -120, 0, 1, 0, 1]]},
        ]},
    }
    direction = parse_film_direction({
        "palette": {"background": "#123456", "text": "#fff", "accent": "#ffaa22"}, "shots": [shot],
    }, "fixture", "fixture", authored=True)
    assert direction.shots[0].artwork["layers"][0]["keys"][1][1] == -120
    assert direction.shots[0].narration == shot["narration"]


@pytest.mark.parametrize("svg", [
    '<script>alert("unsafe")</script>',
    '<rect onload="alert(1)" width="80" height="40"/>',
    '<rect fill="url(https://example.com/a)" width="80" height="40"/>',
    '<image href="https://example.com/a.png"/>',
    '<foreignObject><body/></foreignObject>',
    '<rect filter="url(#missing)" width="80" height="40"/>',
])
def test_model_markup_cannot_execute_or_load_external_content(svg):
    with pytest.raises(ValueError):
        validate_artwork({"background": "#123", "layers": [{"id": "layer", "svg": svg}]})


@pytest.mark.parametrize("attachment", [
    {"parent": "child"}, {"parent": "missing"}, {"pivot": [float("nan"), 100]}, {"pivot": [5]},
])
def test_artwork_rejects_invalid_layer_attachments(attachment):
    with pytest.raises(ValueError, match="parent|pivot"):
        validate_artwork({
            "background": "#123",
            "layers": [{"id": "child", "svg": '<circle cx="100" cy="100" r="50"/>', **attachment}],
        })


def test_film_repair_updates_only_named_layer_fields(monkeypatch):
    from video_generation import film_direction

    base = parse_film_direction({
        "palette": {"background": "#112233", "text": "#fff", "accent": "#aaf"},
        "shots": [{
            "renderer": "remotion", "narration": "", "intent": "Show the parts moving together.",
            "artwork": {"background": "#112233", "layers": [
                {"id": "part", "svg": '<rect width="100" height="40"/>'},
                {"id": "light", "svg": '<circle cx="40" cy="50" r="10"/>'},
            ]},
        }],
    }, "fixture", "fixture", authored=True)
    changes = {"parent": "part", "pivot": [40, 50]}
    monkeypatch.setattr(film_direction, "_request_film_direction", lambda *args, **kwargs: json.dumps({
        "patches": [{"id": "beat_01", "changes": {"layers": {"light": changes}}}],
    }))
    repaired = film_direction.revise_film_direction(base, {"beat_01": "Light drifts away."}, ["beat_01"])
    original, updated = base.shots[0].artwork["layers"], repaired.shots[0].artwork["layers"]
    assert updated == [original[0], {**original[1], **changes}]
    assert "parent" not in original[1]
    changes.clear()
    changes["id"] = "renamed"
    with pytest.raises(ValueError, match="A layer update may change"):
        film_direction.revise_film_direction(base, {"beat_01": "Light drifts away."}, ["beat_01"])


def test_film_repair_can_remove_a_wipe_without_changing_the_scene(monkeypatch):
    from dataclasses import replace

    from video_generation import film_direction

    configure_fixture(monkeypatch)
    payload = json.loads(broll_intelligence.call_reasoning_model("fixture", "fixture", "", "{}"))
    payload["shots"][1]["transition"] = "accent_wipe"
    base = parse_film_direction(payload, "fixture", "fixture", authored=True)
    monkeypatch.setattr(film_direction, "_request_film_direction", lambda *args, **kwargs: json.dumps({
        "patches": [{"id": "beat_02", "changes": {"transition": None}}],
    }))

    result = film_direction.revise_film_direction(
        base, {"beat_02": "Remove the opening wipe."}, ["beat_01", "beat_02", "beat_03"],
    )

    assert result.shots == [base.shots[0], replace(base.shots[1], transition="cut"), base.shots[2]]


@pytest.mark.skipif(os.getenv("VEX_RUN_HYBRID_RENDER") != "1", reason="requires HyperFrames, Remotion and FFmpeg")
def test_attached_layer_keeps_its_center_while_scaling_in_both_browsers(tmp_path):
    import numpy as np
    from PIL import Image

    artwork = {
        "background": "#0b1320", "view": [480, 320],
        "layers": [
            {"id": "sensor", "svg": '<rect x="136" y="50" width="8" height="220" fill="#ff3030"/>',
             "keys": [[0, 0, 0, 1, 0, 1], [1, 160, 0, 1, 0, 1]]},
            {"id": "blur", "parent": "sensor", "pivot": [140, 160],
             "svg": '<circle cx="140" cy="160" r="36" fill="#3080ff"/>',
             "keys": [[0, 0, 0, .3, 0, 1], [1, 0, 0, 1, 0, 1]]},
        ],
    }
    direction = tmp_path / "direction.json"
    direction.write_text(json.dumps({
        "provider": "fixture", "model": "fixture",
        "palette": {"background": "#0b1320", "text": "#fff", "accent": "#3080ff"},
        "shots": [
            {"renderer": engine, "intent": "The blur grows while staying attached to the moving sensor.",
             "narration": "", "artwork": artwork}
            for engine in ("hyperframes", "remotion")
        ],
    }), encoding="utf-8")
    result = generate_video({
        "prompt": "Show an attached blur", "renderer": "hybrid", "duration_sec": 6,
        "resolution": "480x320", "fps": 30, "workers": "2", "generate_audio": False,
        "auto_install_runtime": False, "output_dir": str(tmp_path), "film_direction_path": str(direction),
    })
    assert result.qa_passed, result.qa_issues
    qa = json.loads(Path(result.qa_path).read_text(encoding="utf-8"))["evidence"]["visual_quality"]
    measurements = []
    for shot in qa["shots"]:
        samples = []
        for frame in shot["frames"]:
            with Image.open(Path(result.output_path).parent / frame["path"]) as image:
                pixels = np.asarray(image.convert("RGB"), dtype=np.int16)
            red, green, blue = pixels.transpose(2, 0, 1)
            _, sensor = np.where((red > 140) & (red - green > 80))
            _, blur = np.where((blue > 140) & (blue - red > 80))
            assert abs(sensor.mean() - blur.mean()) <= 2
            samples.append((blur.mean(), len(blur)))
        assert samples[-1][0] - samples[0][0] > 140
        assert samples[-1][1] > 5 * samples[0][1]
        measurements.append(samples)
    assert all(abs(a[0] - b[0]) <= 4 for a, b in zip(*measurements, strict=True))


@pytest.mark.skipif(os.getenv("VEX_RUN_HYBRID_RENDER") != "1", reason="requires Manim, HyperFrames, Remotion and FFmpeg")
@pytest.mark.parametrize("visible_repair", [True, False], ids=["visible", "invisible"])
def test_hybrid_render_composes_real_native_shots_and_sound(tmp_path, monkeypatch, visible_repair):
    import numpy as np
    from PIL import Image

    from video_generation import film_review

    target = "beat_02" if visible_repair else "beat_03"

    def repair(direction):
        if visible_repair:
            return [{"id": target, "changes": {"transition": "accent_wipe"}}]
        artwork = direction["shots"][2]["artwork"]
        artwork["layers"].append({
            "id": "hidden", "svg": '<circle cx="240" cy="120" r="42" fill="#fff" opacity="0"/>',
        })
        return [{"id": target, "changes": {"artwork": artwork}}]

    configure_fixture(monkeypatch, "deepseek-flash", repair)
    reviews = []

    class Reviewer:
        def __init__(self, provider):
            assert provider == "openai_compatible"

        def chat(self, messages, tools, system):
            parts = messages[0]["content"]
            assert sum(part["type"] == "image_url" for part in parts) == 9
            assert "Make the payoff read in a direct reveal." not in parts[0]["text"]
            assert parts[-1]["image_url"]["url"].startswith("data:image/png;base64,")
            reviews.append(parts[0]["text"])
            shots = [
                {"id": f"beat_{i:02d}", "verdict": "revise" if f"beat_{i:02d}" == target and len(reviews) == 1 else "pass",
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
    assert result.qa_passed == visible_repair, Path(result.qa_path).read_text()
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
    for shot in qa["shots"][1:]:
        positions = []
        for frame in shot["frames"][1:]:
            with Image.open(Path(result.output_path).parent / frame["path"]) as image:
                pixels = np.asarray(image.convert("RGB"), dtype=np.int16)[:int(image.height * .8)]
            _, x = np.where((pixels[:, :, 2] > 130) & (pixels[:, :, 2] - pixels[:, :, 0] > 45))
            assert len(x) > 100
            positions.append(x.mean())
        assert positions[-1] - positions[0] > 25
    assert all("visual_intent" in request for request in reviews)
    assert len(reviews) == (2 if visible_repair else 1)
    assert evidence["film_review"]["passed"] == visible_repair
    repair_report = json.loads((project / "film_repair.json").read_text())
    assert repair_report["accepted"] == visible_repair
    comparison = repair_report["comparison"]
    assert comparison["passed"] == visible_repair
    for shot in comparison["shots"]:
        assert any(frame["visible_change"] for frame in shot["frames"]) == (visible_repair and shot["targeted"])
    if visible_repair:
        wiped = next(shot for shot in comparison["shots"] if shot["targeted"])
        assert all(frame["changed_pixel_fraction"] < 0.20 for frame in wiped["frames"])
        rendering = repair_report["rendering"]
        assert rendering["mode"] == "shot_ranges", rendering
        assert rendering["reused_frames"] > 0 and rendering["audio_copied"] is True
        assert comparison["shots"][-1]["id"] == "beat_03"
        assert comparison["shots"][-1]["targeted"] is False
    else:
        assert comparison["issues"] == ["no_visible_change:beat_03"]
        assert not (project / "film_repair/remotion/final.mp4").exists()
    assert Path(inspect_remotion_project(project)["composition"]) == Path(result.output_path).parent
    revised = json.loads((project / "film_repair/film_production.json").read_text())
    assert [shot["status"] for shot in revised["shots"]] == ["reused", "reused", "inline"]
    assert Path(result.output_path).is_relative_to(project / "film_repair") == visible_repair
