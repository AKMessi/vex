from __future__ import annotations

from pathlib import Path

from rich.console import Console

import main
from agent_trace import TraceEvent, render_trace_table
from tools.creative_registry import record_creative_run
from state import ProjectState, utc_now_iso


def test_trace_table_renders_actor_status_and_duration() -> None:
    event = TraceEvent(
        step=1,
        kind="tool",
        title="trim_clip completed",
        detail="Saved edited clip",
        status="success",
        metadata={"duration_sec": 1.4},
    )
    console = Console(record=True, width=120)

    console.print(render_trace_table([event]))

    output = console.export_text()
    assert "OK" in output
    assert "tool" in output
    assert "trim_clip completed" in output
    assert "1.4s" in output


def test_live_log_buffer_replaces_progress_lines() -> None:
    buffer = main.LiveLogBuffer(max_lines=4)

    buffer.write("10/100\r")
    buffer.write("20/100\r")
    buffer.write("finished\n")

    assert buffer.snapshot() == ["20/100", "finished"]


def test_project_dashboard_surfaces_artifacts_and_timeline(tmp_path: Path) -> None:
    state = _state(tmp_path)
    state.timeline.append(
        {
            "op": "trim_clip",
            "description": "Removed first 10 seconds",
            "params": {},
            "timestamp": utc_now_iso(),
        }
    )
    state.artifacts["latest_transcript"] = {"segment_count": 8, "word_count": 320}
    state.artifacts["latest_auto_visuals"] = {"count": 3, "renderer": "hyperframes", "style_pack": "product_ui"}
    record_creative_run(
        working_dir=state.working_dir,
        feature="auto_visuals",
        manifest_path=str(tmp_path / "manifest.json"),
        quality_score=0.82,
        summary={"count": 3, "renderer": "hyperframes", "style_pack": "product_ui"},
    )
    console = Console(record=True, width=140)

    console.print(main.render_project_dashboard(state))

    output = console.export_text()
    assert "Project: Demo Project" in output
    assert "Transcript" in output
    assert "Auto visuals" in output
    assert "Recent Timeline" in output
    assert "Creative Runs" in output
    assert "trim_clip" in output


def test_creative_runs_table_renders_registry_records(tmp_path: Path) -> None:
    state = _state(tmp_path)
    record_creative_run(
        working_dir=state.working_dir,
        feature="auto_shorts",
        manifest_path=str(tmp_path / "shorts_manifest.json"),
        quality_score=0.77,
        summary={"count": 2, "target_platform": "youtube_shorts", "candidate_count": 24},
    )
    console = Console(record=True, width=140)

    console.print(main.render_creative_runs_table(state))

    output = console.export_text()
    assert "auto_shorts" in output
    assert "2 shorts" in output
    assert "0.77" in output


def test_repl_prompt_escapes_project_name(tmp_path: Path) -> None:
    state = _state(tmp_path)
    state.project_name = "Clip [final]"

    assert r"Clip \[final]" in main.repl_prompt(state)


def _state(tmp_path: Path) -> ProjectState:
    now = utc_now_iso()
    source_file = tmp_path / "source.mp4"
    source_file.write_bytes(b"placeholder")
    return ProjectState(
        project_id="demo-project-id",
        project_name="Demo Project",
        created_at=now,
        updated_at=now,
        source_files=[str(source_file)],
        working_file=str(tmp_path / "working.mp4"),
        working_dir=str(tmp_path),
        output_dir=str(tmp_path / "outputs"),
        metadata={"duration_sec": 95.0, "width": 1920, "height": 1080, "fps": 30.0, "size_bytes": 1024},
        provider="gemini",
        model="gemini-test",
    )
