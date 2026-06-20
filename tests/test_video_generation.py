from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import main
import prompts
from intent_compiler import compile_intent
from video_generation import generate_video, normalize_generation_request
from video_generation.beat_graph import (
    build_initial_beat_graph,
    parse_transcript_words,
    retime_beat_graph,
)
from video_generation.cinematographer import build_cinematic_plan, inline_cinematic_composition
from video_generation.hyperframes_project import build_index_html
from video_generation.models import Beat, BeatGraph, ScriptPlan, TimedWord
from video_generation.motion import build_motion_plan
from video_generation.renderer import HyperframesVideoRuntime
from video_generation.script_planner import build_script_plan


def test_generation_request_normalizes_audio_first_defaults(tmp_path: Path) -> None:
    request = normalize_generation_request(
        {
            "prompt": "Generate a vertical video about attention routing",
            "aspect": "9:16",
            "format": "gif",
            "duration_sec": 3,
            "render_resolution": "4k",
            "output_dir": str(tmp_path),
        }
    )

    assert request.aspect == "portrait"
    assert (request.width, request.height) == (1080, 1920)
    assert request.output_format == "mp4"
    assert request.render_resolution == "portrait-4k"
    assert request.duration_sec == 6.0
    assert request.generate_audio is True
    assert request.transcribe_audio is True


def test_beat_graph_uses_hyperframes_word_timing() -> None:
    request = normalize_generation_request(
        {
            "prompt": "routing",
            "script": "First, route the signal. Then compress the memory into four blocks.",
            "render": False,
            "generate_audio": False,
        }
    )
    plan = build_script_plan(request)
    fallback = build_initial_beat_graph(plan, target_duration_sec=10.0, voice_speed=1.0)
    transcript_words = parse_transcript_words(
        {
            "segments": [
                {
                    "words": [
                        {"word": "First", "start": 0.0, "end": 0.4},
                        {"word": "route", "start": 0.4, "end": 0.8},
                        {"word": "signal", "start": 0.8, "end": 1.3},
                        {"word": "Then", "start": 1.8, "end": 2.1},
                        {"word": "compress", "start": 2.1, "end": 2.7},
                        {"word": "memory", "start": 2.7, "end": 3.2},
                        {"word": "blocks", "start": 3.2, "end": 3.8},
                    ]
                }
            ]
        }
    )

    retimed = retime_beat_graph(
        plan,
        transcript_words=transcript_words,
        duration_sec=4.0,
        fallback=fallback,
    )

    assert retimed.source == "hyperframes_transcript"
    assert retimed.beats[0].start == 0.0
    assert retimed.beats[-1].end == 4.0
    assert retimed.words[-1].text == "blocks"


def test_project_html_wires_audio_captions_and_timeline() -> None:
    request = normalize_generation_request(
        {
            "prompt": "make a video about cache compression",
            "script": "Cache compression turns many tokens into fewer blocks. The payoff is lower memory pressure.",
            "generate_audio": True,
            "render": False,
        }
    )
    plan = build_script_plan(request)
    beat_graph = build_initial_beat_graph(plan, target_duration_sec=12.0, voice_speed=1.0)

    html = build_index_html(
        request=request,
        plan=plan,
        beat_graph=beat_graph,
        audio_path=Path("project/audio/narration.wav"),
    )

    assert 'data-composition-id="root"' in html
    assert 'src="audio/narration.wav"' in html
    assert 'class="clip caption ' in html
    assert 'data-layout-allow-occlusion="caption-overlay"' in html
    assert 'data-layout-allow-overflow="transition-streak"' in html
    assert 'data-track-index="170"' in html
    assert 'window.__timelines["root"]' in html
    assert "vex-motion-plan" in html
    assert "requestAnimationFrame" not in html


def test_project_html_avoids_adjacent_clip_float_overlap() -> None:
    request = normalize_generation_request(
        {
            "prompt": "prove adjacent timing stays lint clean",
            "generate_audio": False,
            "render": False,
        }
    )
    plan = ScriptPlan(
        title="Timing Proof",
        narration="First beat. Second beat.",
        design_direction="timing proof",
        source="test",
        prompt=request.prompt,
    )
    beat_graph = BeatGraph(
        version="test",
        duration_sec=18.4,
        source="test",
        beats=[
            Beat(
                beat_id="beat_01",
                index=1,
                start=9.436,
                end=13.918,
                title="First",
                narration="First beat.",
                caption="First beat.",
                scene_type="concept",
            ),
            Beat(
                beat_id="beat_02",
                index=2,
                start=13.918,
                end=18.4,
                title="Second",
                narration="Second beat.",
                caption="Second beat.",
                scene_type="proof",
            ),
        ],
    )

    html = build_index_html(request=request, plan=plan, beat_graph=beat_graph)

    assert 'data-start="9.436" data-duration="4.481"' in html
    assert 'data-start="13.918" data-duration="4.482"' in html


def test_cinematographer_compiles_sparse_attention_into_visual_worlds() -> None:
    request = normalize_generation_request(
        {
            "prompt": "show how sparse attention turns token links into a focused reasoning path",
            "duration_sec": 12,
            "render": False,
            "generate_audio": False,
        }
    )
    plan = build_script_plan(request)
    beat_graph = build_initial_beat_graph(plan, target_duration_sec=12.0, voice_speed=1.0)
    cinematic = build_cinematic_plan(request=request, plan=plan, beat_graph=beat_graph)

    assert cinematic.accepted_count == len(beat_graph.beats)
    assert all(item.scene_type != "none" for item in cinematic.beat_compositions)
    assert all(item.template.startswith("semantic_") for item in cinematic.beat_compositions)
    assert all("visual-world-canvas" in item.composition_html for item in cinematic.beat_compositions)

    inline_html = inline_cinematic_composition(
        cinematic.beat_compositions[0],
        start=1.5,
        duration=3.0,
        track_index=10,
    )
    assert 'class="clip beat-composition inline-composition' in inline_html
    assert 'data-cinematic-composition-id="vex-' in inline_html
    assert 'data-layout-allow-overflow="camera-safe-area"' in inline_html
    assert "row-gap:24px" in inline_html
    assert 'data-composition-id="vex-' not in inline_html
    assert 'document.getElementById("root")' not in inline_html
    assert "window.__timelines" not in inline_html
    assert 'document.querySelectorAll("[data-anim]")' not in inline_html
    assert inline_html.count('class="clip ') == 1


def test_cinematographer_uses_clause_grounded_generated_video_labels() -> None:
    script = (
        "\ufeffFirst, Vex listens to the idea. "
        "It breaks the narration into beats, chooses where a visual actually helps, "
        "builds semantic scenes, compiles native HyperFrames motion, adds voice, "
        "renders frames, and runs QA. "
        "The proof is simple: token chaos becomes a focused reasoning path."
    )
    request = normalize_generation_request(
        {
            "prompt": (
                "Create the best possible X proof video for Vex. Make it feel "
                "like a premium product demo, not a slide deck."
            ),
            "script": script,
            "duration_sec": 18,
            "render": False,
            "generate_audio": False,
        }
    )
    plan = build_script_plan(request)
    beat_graph = build_initial_beat_graph(plan, target_duration_sec=18.0, voice_speed=1.0)

    cinematic = build_cinematic_plan(request=request, plan=plan, beat_graph=beat_graph)

    visible_labels: list[str] = []
    for item in cinematic.beat_compositions:
        if not item.compiler_passed:
            continue
        visible_labels.append(str(item.spec.get("headline") or ""))
        visible_labels.extend(
            str(label)
            for label in (item.spec.get("qa_contract") or {}).get("required_labels") or []
        )
    normalized_labels = [label.lower() for label in visible_labels]
    visible_copy = " ".join(normalized_labels)
    assert cinematic.accepted_count == len(beat_graph.beats)
    assert "helps builds" not in normalized_labels
    assert all("path. create" not in label for label in normalized_labels)
    assert all("\ufeff" not in label for label in visible_labels)
    assert "simple" not in normalized_labels
    assert "visible input" not in normalized_labels
    assert "route useful signal" not in normalized_labels
    assert "focused reasoning path" in visible_copy
    assert "builds semantic scenes" in visible_copy


def test_project_only_pipeline_writes_manifest_without_runtime(tmp_path: Path) -> None:
    result = generate_video(
        {
            "prompt": "generate a video about retrieval augmented generation",
            "script": "Retrieval adds evidence before generation. The answer becomes easier to verify.",
            "output_dir": str(tmp_path),
            "render": False,
            "generate_audio": False,
            "duration_sec": 9,
        }
    )

    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))

    assert result.rendered is False
    assert Path(result.index_path).is_file()
    assert Path(result.beat_graph_path).is_file()
    assert manifest["request"]["render"] is False
    assert manifest["qa"]["passed"] is True
    assert manifest["beat_graph"]["beats"]
    assert Path(manifest["artifacts"]["cinematography_path"]).is_file()
    assert Path(manifest["artifacts"]["motion_plan_path"]).is_file()
    assert Path(manifest["artifacts"]["motion_cues_path"]).is_file()
    assert manifest["qa"]["evidence"]["cinematic_plan"]["accepted_count"] >= 1
    assert manifest["qa"]["evidence"]["motion_plan"]["native_composition_count"] >= 1
    assert manifest["qa"]["evidence"]["motion_plan"]["audio_cue_count"] >= 1
    composition_files = list((Path(result.project_dir) / "compositions").glob("*.html"))
    assert composition_files
    index_html = Path(result.index_path).read_text(encoding="utf-8")
    assert "data-external-composition-src" in index_html
    assert 'data-native-hyperframes-motion="true"' in index_html
    assert 'data-vex-native-composition="true"' in index_html
    assert "seekNestedComposition" in index_html
    assert "nativeScenes = beatCompositions.map" in index_html
    assert "applyNativeSceneController" in index_html
    assert 'querySelectorAll("[data-anim]")' in index_html
    assert "--vex-camera-x" in index_html
    assert "--route-progress" in index_html
    assert "--vex-inner-roll" in index_html
    assert "requestAnimationFrame" not in index_html
    composition_html = composition_files[0].read_text(encoding="utf-8")
    assert composition_html.lstrip().startswith("<template")
    assert 'id="root"' not in composition_html
    assert 'data-vex-native-composition="true"' in composition_html
    assert 'data-duration="' in composition_html
    assert "__vexNativeMotionPatched" in composition_html


def test_motion_plan_filters_low_value_audio_cue_words() -> None:
    request = normalize_generation_request(
        {
            "prompt": "generate a cinematic proof video for Vex",
            "script": "First, the proof is simple: token chaos becomes a focused reasoning path.",
            "render": False,
            "generate_audio": False,
        }
    )
    plan = ScriptPlan(
        title="Cue Filtering",
        narration=request.script,
        design_direction="cinematic proof",
        source="test",
        prompt=request.prompt,
    )
    beat_graph = BeatGraph(
        version="test",
        duration_sec=6.0,
        source="test",
        beats=[
            Beat(
                beat_id="beat_01",
                index=1,
                start=0.0,
                end=6.0,
                title="Cue Filtering",
                narration=request.script,
                caption=request.script,
                scene_type="proof",
                keywords=["token", "chaos", "focused", "reasoning"],
                visual_metaphor="semantic camera cue test",
            )
        ],
        words=[
            TimedWord("First", 0.0, 0.2),
            TimedWord("proof", 0.3, 0.65),
            TimedWord("simple", 0.7, 1.1),
            TimedWord("a", 1.15, 1.2),
            TimedWord("token", 1.2, 1.55),
            TimedWord("chaos", 1.6, 2.0),
            TimedWord("becomes", 2.1, 2.45),
            TimedWord("focused", 2.5, 2.95),
            TimedWord("reasoning", 3.0, 3.55),
            TimedWord("path", 3.65, 4.0),
        ],
    )

    motion = build_motion_plan(request=request, plan=plan, beat_graph=beat_graph)
    labels = [
        cue.label.lower()
        for profile in motion.beat_profiles
        for cue in profile.audio_cues
    ]

    assert "first" not in labels
    assert "simple" not in labels
    assert "a" not in labels
    assert "becomes" not in labels
    assert "reasoning" in labels
    assert "focused" in labels


def test_generate_video_uses_vex_whisper_fallback_for_transcription(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import video_generation.pipeline as module

    def fake_tts(self, *, text_path, output_path, voice, speed, language=""):  # noqa: ANN001, ANN003
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"audio")
        return module.CommandRecord(
            name="tts",
            command=["hyperframes", "tts"],
            cwd=str(text_path.parent),
            returncode=0,
            log_path=str(tmp_path / "tts.log"),
        )

    def fake_transcribe(self, *, media_path, project_dir, model="small.en"):  # noqa: ANN001, ANN003
        raise module.VideoGenerationRuntimeError("cli whisper failed")

    monkeypatch.setattr(module.HyperframesVideoRuntime, "tts", fake_tts)
    monkeypatch.setattr(module.HyperframesVideoRuntime, "transcribe", fake_transcribe)
    monkeypatch.setattr(
        module,
        "probe_video",
        lambda _path: {"duration_sec": 2.0, "width": 0, "height": 0, "has_audio": True},
    )
    monkeypatch.setattr(
        module,
        "transcribe_with_whisper",
        lambda *_args, **_kwargs: {
            "text": "Hello world",
            "segments": [{"start": 0.0, "end": 2.0, "text": "Hello world"}],
        },
    )

    result = generate_video(
        {
            "prompt": "generate a video about fallback transcription",
            "script": "Hello world.",
            "output_dir": str(tmp_path),
            "render": False,
            "generate_audio": True,
            "transcribe_audio": True,
        }
    )

    transcript = json.loads(Path(result.transcript_path).read_text(encoding="utf-8"))

    assert result.transcript_path.endswith("transcript.vex-whisper.json")
    assert transcript["words"][0]["word"] == "Hello"
    assert any("used Vex Whisper fallback" in item for item in result.warnings)


def test_hyperframes_runtime_builds_trusted_commands(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    import video_generation.renderer as module

    cli = tmp_path / "node_modules" / ".bin" / "hyperframes.cmd"
    cli.parent.mkdir(parents=True)
    cli.write_text("cli", encoding="utf-8")
    monkeypatch.setattr(module.config, "HYPERFRAMES_CLI_PATH", str(cli))
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):  # noqa: ANN001, ANN003
        calls.append(list(command))
        if "render" in command:
            output = Path(command[command.index("--output") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"mp4")
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        module,
        "probe_video",
        lambda _: {"duration_sec": 5.0, "width": 1920, "height": 1080, "has_audio": True},
    )

    runtime = HyperframesVideoRuntime(auto_install=False)
    metadata, _record = runtime.render(
        project_dir=tmp_path,
        output_path=tmp_path / "renders" / "final.mp4",
        fps=30,
        quality="standard",
        output_format="mp4",
        resolution="landscape-4k",
        workers="1",
    )

    assert metadata["has_audio"] is True
    assert calls[0][0] == str(cli)
    assert calls[0][-1] == "."
    assert calls[0][calls[0].index("--workers") + 1] == "1"
    assert calls[0][calls[0].index("--resolution") + 1] == "landscape-4k"


def test_generate_video_is_registered_in_cli_schema_and_intent() -> None:
    assert "generate_video" in main.TOOL_EXECUTORS
    assert any(schema["name"] == "generate_video" for schema in prompts.TOOL_SCHEMAS)

    plan = compile_intent("generate a portrait hyperframes video about sparse attention in 20 seconds", None)

    assert plan is not None
    assert plan.steps[0].tool == "generate_video"
    assert plan.steps[0].params["aspect"] == "portrait"
    assert plan.steps[0].params["duration_sec"] == 20
