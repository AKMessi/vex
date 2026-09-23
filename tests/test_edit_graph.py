from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from vex_runtime.edit_graph import ClipSpan, EditGraph, EditGraphError, SourceNode
from state import ProjectState, utc_now_iso
from tools.promotion import promote_working_file
from vex_web.server import _public_edit_graph
import main


def test_graph_uses_rational_time_and_round_trips(tmp_path: Path) -> None:
    graph = EditGraph.from_source(
        tmp_path / "source.mp4", duration="3003/1001", fps="30000/1001"
    )

    assert graph.fps == Fraction(30000, 1001)
    assert graph.duration == Fraction(3)
    assert EditGraph.from_mapping(graph.to_dict()) == graph
    assert graph.map_output_time("1/3") == ("src_0", Fraction(1, 3))


def test_project_import_starts_with_retained_source_graph(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    source = tmp_path / "input.mp4"
    source.write_bytes(b"source")
    monkeypatch.setattr(main.config, "AGENT_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setattr(
        main, "probe_video",
        lambda _path: {"duration_sec": 2.5, "duration_rational": "5/2", "fps": 29.97, "fps_ratio": "30000/1001"},
    )

    state = main.create_project(str(source), None, "test", "test-model")

    graph = EditGraph.from_mapping(state.edit_graph)
    assert graph.duration == Fraction(5, 2)
    assert graph.fps == Fraction(30000, 1001)
    assert graph.sources["src_0"].media_path == state.working_file
    assert Path(state.working_file).read_bytes() == b"source"


def test_trim_and_ripple_cut_preserve_source_time_mapping(tmp_path: Path) -> None:
    graph = EditGraph.from_source(tmp_path / "source.mp4", duration=20, fps=30)
    graph = graph.cut(5, 10)
    assert graph.duration == 15
    assert len(graph.spans) == 2
    assert graph.map_output_time(5) == ("src_0", Fraction(10))

    graph = graph.trim(3, 12)
    assert graph.duration == 9
    assert graph.spans[0].source_start == 3
    assert graph.spans[0].source_end == 5
    assert graph.spans[1].source_start == 10
    assert graph.spans[1].source_end == 17
    assert graph.map_output_time(2) == ("src_0", Fraction(10))
    assert graph.map_output_time(graph.duration) == ("src_0", Fraction(17))


def test_fractional_trim_does_not_accumulate_float_drift(tmp_path: Path) -> None:
    graph = EditGraph.from_source(tmp_path / "source.mp4", duration=10, fps="30000/1001")
    for _ in range(10):
        graph = graph.trim("1/10")
    assert graph.spans[0].source_start == 1
    assert graph.duration == 9


@pytest.mark.parametrize(
    "spans",
    [
        (ClipSpan("src_0", Fraction(0), Fraction(1), Fraction(1), Fraction(2)),),
        (
            ClipSpan("src_0", Fraction(0), Fraction(1), Fraction(0), Fraction(1)),
            ClipSpan("src_0", Fraction(1), Fraction(2), Fraction(2), Fraction(3)),
        ),
        (ClipSpan("src_0", Fraction(0), Fraction(11), Fraction(0), Fraction(11)),),
    ],
)
def test_graph_rejects_gaps_and_out_of_source_ranges(tmp_path: Path, spans: tuple[ClipSpan, ...]) -> None:
    source = SourceNode("src_0", str(tmp_path / "source.mp4"), Fraction(10))
    with pytest.raises(EditGraphError):
        EditGraph(sources={"src_0": source}, spans=spans, fps=Fraction(30))


def test_graph_rejects_unknown_future_schema(tmp_path: Path) -> None:
    payload = EditGraph.from_source(tmp_path / "source.mp4", duration=10, fps=30).to_dict()
    payload["schema_version"] = 2
    with pytest.raises(EditGraphError, match="Unsupported"):
        EditGraph.from_mapping(payload)


def test_graph_rejects_mismatched_declared_duration(tmp_path: Path) -> None:
    payload = EditGraph.from_source(tmp_path / "source.mp4", duration=10, fps=30).to_dict()
    payload["duration"] = "11/1"
    with pytest.raises(EditGraphError, match="duration"):
        EditGraph.from_mapping(payload)


def test_cut_cannot_remove_entire_graph(tmp_path: Path) -> None:
    graph = EditGraph.from_source(tmp_path / "source.mp4", duration=10, fps=30)
    with pytest.raises(EditGraphError, match="entire"):
        graph.cut(0, 10)


def test_promoted_trims_compose_source_mapping_and_studio_summary(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    now = utc_now_iso()
    state = ProjectState(
        project_id="graph-test",
        project_name="Graph test",
        created_at=now,
        updated_at=now,
        source_files=[str(source)],
        working_file=str(source),
        working_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        metadata={"duration_sec": 10, "fps": 30},
        edit_graph=EditGraph.from_source(source, duration=10, fps=30).to_dict(),
    )
    state.save()
    first = tmp_path / "first.mp4"
    first.write_bytes(b"first")
    promote_working_file(
        state, first,
        operation={"op": "trim_clip", "params": {"start": 2, "end": 8}},
        metadata={"duration_sec": 6, "fps": 30},
    )
    second = tmp_path / "second.mp4"
    second.write_bytes(b"second")
    promote_working_file(
        state, second,
        operation={"op": "trim_clip", "params": {"start": 1, "end": 4}},
        metadata={"duration_sec": 3, "fps": 30},
    )

    graph = EditGraph.from_mapping(state.edit_graph)
    summary = _public_edit_graph(state)
    assert graph.provenance == "source"
    assert graph.spans[0].source_start == 3
    assert graph.spans[0].source_end == 6
    assert summary["spans"][0]["source_start_sec"] == 3.0
    assert "media_path" not in str(summary)
    assert EditGraph.from_mapping(ProjectState.from_dict(state.capture_snapshot()).edit_graph) == graph


def test_unsupported_promotion_is_labeled_rendered_anchor(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "graded.mp4"
    source.write_bytes(b"source")
    output.write_bytes(b"graded")
    now = utc_now_iso()
    state = ProjectState(
        project_id="anchor-test", project_name="Anchor test", created_at=now, updated_at=now,
        source_files=[str(source)], working_file=str(source), working_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"), metadata={"duration_sec": 10, "fps": 30},
        edit_graph=EditGraph.from_source(source, duration=10, fps=30).to_dict(),
    )
    state.save()
    promote_working_file(
        state, output, operation={"op": "auto_color_grade", "params": {}},
        metadata={"duration_sec": 10, "fps": 30},
    )
    graph = EditGraph.from_mapping(state.edit_graph)
    assert graph.provenance == "rendered_anchor"
    assert graph.sources["src_0"].media_path == str(output)


def test_legacy_apply_operation_cannot_leave_stale_source_graph(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "effect.mp4"
    source.write_bytes(b"source")
    output.write_bytes(b"effect")
    now = utc_now_iso()
    state = ProjectState(
        project_id="legacy-test", project_name="Legacy", created_at=now, updated_at=now,
        source_files=[str(source)], working_file=str(source), working_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"), metadata={"duration_sec": 10, "fps": 30},
        edit_graph=EditGraph.from_source(source, duration=10, fps=30).to_dict(),
    )
    state.save()
    state.working_file = str(output)
    state.apply_operation({"op": "add_text_overlay", "result_file": str(output), "params": {}})
    graph = EditGraph.from_mapping(state.edit_graph)
    assert graph.provenance == "rendered_anchor"
    assert graph.sources["src_0"].media_path == str(output)
