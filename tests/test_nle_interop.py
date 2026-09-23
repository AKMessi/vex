from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

import main
from nle_interop import build_edl, build_fcpxml, build_nle_timeline_payload, export_nle_bundle
from state import ProjectState, utc_now_iso
from vex_runtime.edit_graph import EditGraph


def test_nle_timeline_payload_preserves_operations_and_markers(tmp_path: Path) -> None:
    state = _state(tmp_path)
    state.timeline.append(
        {
            "op": "trim_clip",
            "params": {"start": "5", "end": "20"},
            "description": "Trimmed intro",
            "timestamp": utc_now_iso(),
        }
    )

    payload = build_nle_timeline_payload(state)

    assert payload["project"]["project_id"] == "test-project"
    assert payload["media"]["working_file"] == state.working_file
    assert payload["operations"][0]["op"] == "trim_clip"
    assert payload["markers"][0]["start_sec"] == 5.0
    assert payload["markers"][0]["label"] == "1. Trimmed intro"


def test_fcpxml_and_edl_exports_are_parseable(tmp_path: Path) -> None:
    state = _state(tmp_path)
    state.timeline.append(
        {
            "op": "add_text_overlay",
            "params": {"overlays": [{"start": 7.5, "end": 9.0}]},
            "description": "Added title",
            "timestamp": utc_now_iso(),
        }
    )
    payload = build_nle_timeline_payload(state)

    fcpxml = build_fcpxml(state, payload)
    edl = build_edl(state, payload)
    root = ET.fromstring(fcpxml)

    assert root.tag == "fcpxml"
    assert root.find(".//asset-clip") is not None
    assert root.find(".//marker").attrib["value"] == "1. Added title"
    assert "TITLE: Test Project" in edl
    assert "* VEX_OP 001: add_text_overlay - Added title" in edl


def test_export_nle_bundle_writes_requested_formats(tmp_path: Path) -> None:
    state = _state(tmp_path)
    output_dir = tmp_path / "exports"

    result = export_nle_bundle(state, output_dir, formats={"json", "edl"})

    assert set(result.files) == {"json", "edl"}
    assert Path(result.files["json"]).is_file()
    assert Path(result.files["edl"]).is_file()
    assert json.loads(Path(result.files["json"]).read_text(encoding="utf-8"))["schema_version"] == 2


def test_graph_backed_export_contains_real_source_cuts(tmp_path: Path) -> None:
    state = _state(tmp_path)
    graph = EditGraph.from_source(state.source_files[0], duration=20, fps="30000/1001").cut(5, 10)
    state.edit_graph = graph.to_dict()
    state.metadata["duration_sec"] = 15

    payload = build_nle_timeline_payload(state)
    root = ET.fromstring(build_fcpxml(state, payload))
    clips = root.findall(".//spine/asset-clip")
    edl = build_edl(state, payload)

    assert payload["handoff_mode"] == "source"
    assert root.find("./resources/format").attrib["frameDuration"] == "1001/30000s"
    assert len(clips) == 2
    assert [clip.attrib["offset"] for clip in clips] == ["0/1s", "5/1s"]
    assert [clip.attrib["start"] for clip in clips] == ["0/1s", "10/1s"]
    assert [clip.attrib["duration"] for clip in clips] == ["5/1s", "10/1s"]
    assert len([line for line in edl.splitlines() if line.startswith(("001  ", "002  "))]) == 2
    assert "* SOURCE FILE: " + state.source_files[0] in edl


def test_graph_export_rejects_media_outside_project_roots(tmp_path: Path) -> None:
    state = _state(tmp_path)
    state.edit_graph = EditGraph.from_source(
        tmp_path.parent / "outside.mp4", duration=10, fps=30
    ).to_dict()
    with pytest.raises(ValueError, match="outside project media roots"):
        build_nle_timeline_payload(state)


def test_parse_nle_formats_validates_values() -> None:
    assert main.parse_nle_formats("json,fcpxml") == {"json", "fcpxml"}
    assert main.parse_nle_formats("all") == {"json", "fcpxml", "edl"}
    with pytest.raises(Exception, match="format"):
        main.parse_nle_formats("xml")


def _state(tmp_path: Path) -> ProjectState:
    now = utc_now_iso()
    source_file = tmp_path / "source.mp4"
    working_file = tmp_path / "working.mp4"
    source_file.write_bytes(b"source")
    working_file.write_bytes(b"working")
    return ProjectState(
        project_id="test-project",
        project_name="Test Project",
        created_at=now,
        updated_at=now,
        source_files=[str(source_file)],
        working_file=str(working_file),
        working_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        metadata={"duration_sec": 120.0, "width": 1920, "height": 1080, "fps": 30.0},
        provider="test",
        model="test-model",
    )
