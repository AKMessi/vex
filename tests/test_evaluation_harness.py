from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

import main
from evaluation_harness import (
    EvaluationCase,
    evaluate_intent_case,
    run_intent_evaluation,
    write_evaluation_report,
)
from state import ProjectState, utc_now_iso


def test_builtin_intent_evaluation_passes(tmp_path: Path) -> None:
    report = run_intent_evaluation(state=_state(tmp_path))

    assert report.passed is True
    assert report.failed_count == 0
    assert report.score == 1.0
    assert {case.case_id for case in report.cases} >= {"trim_export_chain", "generate_video"}


def test_evaluate_intent_case_reports_tool_mismatch(tmp_path: Path) -> None:
    result = evaluate_intent_case(
        EvaluationCase(
            case_id="bad_expectation",
            instruction="show video metadata",
            expected_tools=["trim_clip"],
        ),
        state=_state(tmp_path),
    )

    assert result.passed is False
    assert "expected tools" in result.issues[0]


def test_write_evaluation_report_outputs_json(tmp_path: Path) -> None:
    report = run_intent_evaluation(state=_state(tmp_path))
    output_path = tmp_path / "eval.json"

    write_evaluation_report(report, output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["passed"] is True


def test_render_evaluation_report_shows_score(tmp_path: Path) -> None:
    report = run_intent_evaluation(state=_state(tmp_path))
    console = Console(record=True, width=140)

    console.print(main.render_evaluation_report(report))

    output = console.export_text()
    assert "Score: 5/5" in output
    assert "trim_export_chain" in output


def _state(tmp_path: Path) -> ProjectState:
    now = utc_now_iso()
    source_file = tmp_path / "source.mp4"
    source_file.write_bytes(b"source")
    return ProjectState(
        project_id="test-project",
        project_name="Test Project",
        created_at=now,
        updated_at=now,
        source_files=[str(source_file)],
        working_file=str(tmp_path / "working.mp4"),
        working_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        metadata={"duration_sec": 120.0, "width": 1920, "height": 1080, "fps": 30.0},
        provider="test",
        model="test-model",
    )
