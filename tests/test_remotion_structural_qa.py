from __future__ import annotations

import copy

from tests.test_remotion_semantic_pipeline import _grounded_process_spec
from vex_remotion.compiler import compile_remotion_scene_program
from vex_remotion.structural_qa import evaluate_remotion_structure
from vex_visuals.scene_graph import sign_scene_graph


def _program() -> dict:
    result = compile_remotion_scene_program(
        _grounded_process_spec(),
        width=1280,
        height=720,
        fps=30,
    )
    assert result.program is not None
    return result.program.to_dict()


def test_structural_qa_accepts_compiled_scene_graph() -> None:
    report = evaluate_remotion_structure(_program())

    assert report.passed, report.issues
    assert report.score >= 0.95
    assert report.metrics["missing_object_bindings"] == []
    assert report.metrics["missing_relation_bindings"] == []
    assert report.metrics["motion_safe_area_intrusion_count"] == 0
    assert report.resolved_layout


def test_structural_qa_rejects_missing_required_semantic_binding() -> None:
    program = _program()
    graph = copy.deepcopy(program["scene_graph"])
    bound = next(
        node
        for node in graph["nodes"]
        if node["binding"].get("kind") == "object"
    )
    missing_id = str(bound["binding"]["id"])
    bound["binding"]["id"] = "different_object"
    program["scene_graph"] = sign_scene_graph(graph)

    report = evaluate_remotion_structure(program)

    assert not report.passed
    assert (
        f"remotion_structural_qa_missing_semantic_binding:object:{missing_id}"
        in report.issues
    )


def test_structural_qa_rejects_text_that_cannot_fit_at_font_floor() -> None:
    program = _program()
    graph = copy.deepcopy(program["scene_graph"])
    title = next(node for node in graph["nodes"] if node["role"] == "title")
    title["content"]["text"] = "Unbreakable" * 20
    title["layout"]["width"] = 0.05
    title["layout"]["height"] = 0.03
    program["scene_graph"] = sign_scene_graph(graph)

    report = evaluate_remotion_structure(program)

    assert not report.passed
    assert (
        f"remotion_structural_qa_clipped_text:{title['node_id']}"
        in report.issues
    )


def test_structural_qa_rejects_visible_motion_outside_safe_area() -> None:
    program = _program()
    graph = copy.deepcopy(program["scene_graph"])
    target = next(
        node
        for node in graph["nodes"]
        if node["primitive"] in {"graph_node", "semantic_token"}
    )
    template = next(
        track
        for track in graph["motion_graph"]["tracks"]
        if track["target_id"] == target["node_id"]
    )
    template["property"] = "translate_x"
    template["keyframes"] = [
        {"t": 0.0, "value": 0.0, "easing": "linear"},
        {"t": 1.0, "value": 0.9, "easing": "linear"},
    ]
    program["scene_graph"] = sign_scene_graph(graph)

    report = evaluate_remotion_structure(program)

    assert not report.passed
    assert any(
        item.startswith(
            f"remotion_structural_qa_motion_exits_safe_area:{target['node_id']}@"
        )
        for item in report.issues
    )
