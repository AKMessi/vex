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
from video_generation.hyperframes_project import build_index_html
from video_generation.renderer import HyperframesVideoRuntime
from video_generation.script_planner import build_script_plan


def test_generation_request_normalizes_audio_first_defaults(tmp_path: Path) -> None:
    request = normalize_generation_request(
        {
            "prompt": "Generate a vertical video about attention routing",
            "aspect": "9:16",
            "format": "gif",
            "duration_sec": 3,
            "output_dir": str(tmp_path),
        }
    )

    assert request.aspect == "portrait"
    assert (request.width, request.height) == (1080, 1920)
    assert request.output_format == "mp4"
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
    assert 'class="clip caption"' in html
    assert 'window.__timelines["root"]' in html
    assert "requestAnimationFrame" not in html


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
        workers="1",
    )

    assert metadata["has_audio"] is True
    assert calls[0][0] == str(cli)
    assert calls[0][-1] == "."
    assert calls[0][calls[0].index("--workers") + 1] == "1"


def test_generate_video_is_registered_in_cli_schema_and_intent() -> None:
    assert "generate_video" in main.TOOL_EXECUTORS
    assert any(schema["name"] == "generate_video" for schema in prompts.TOOL_SCHEMAS)

    plan = compile_intent("generate a portrait hyperframes video about sparse attention in 20 seconds", None)

    assert plan is not None
    assert plan.steps[0].tool == "generate_video"
    assert plan.steps[0].params["aspect"] == "portrait"
    assert plan.steps[0].params["duration_sec"] == 20
