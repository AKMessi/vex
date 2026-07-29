from __future__ import annotations

import copy

from tests.test_remotion_semantic_pipeline import _grounded_process_spec
from vex_remotion.compiler import compile_remotion_scene_program
from vex_visuals.scene_graph import (
    CAPABILITY_REGISTRY,
    SCENE_GRAPH_VERSION,
    scene_graph_signature,
    sign_scene_graph,
    validate_scene_graph,
)


def _compiled_graph() -> dict:
    result = compile_remotion_scene_program(
        _grounded_process_spec(),
        width=1920,
        height=1080,
        fps=30,
    )
    assert result.passed, result.errors
    assert result.program is not None
    return result.program.scene_graph


def test_scene_graph_is_signed_deterministic_and_capability_bounded() -> None:
    first = _compiled_graph()
    second = _compiled_graph()

    assert first["version"] == SCENE_GRAPH_VERSION
    assert first["signature"] == second["signature"]
    assert first["signature"] == scene_graph_signature(first)
    assert first["source_program_signature"]
    assert first["evidence_signature"]
    assert first["nodes"]
    assert first["relations"]
    assert first["motion_graph"]["phases"][-1]["role"] == "hold"
    assert set(first["telemetry_contract"]["required_probes"]) >= {
        "geometry",
        "motion_state",
        "relation_paths",
        "semantic_bindings",
        "text_bounds",
    }

    validation = validate_scene_graph(first)
    assert validation.passed, validation.errors
    assert validation.backend_counts["svg"] >= 1
    assert validation.capability_cost > 0
    assert {
        item["primitive"] for item in first["capability_manifest"]
    } == {
        item["primitive"] for item in first["nodes"]
    }


def test_scene_graph_rejects_signature_and_capability_tampering() -> None:
    graph = _compiled_graph()
    tampered = copy.deepcopy(graph)
    tampered["nodes"][0]["backend"] = "three"

    validation = validate_scene_graph(tampered)

    assert not validation.passed
    assert "scene_graph_signature_mismatch" in validation.errors
    assert any(
        error.startswith("unsupported_scene_backend:")
        for error in validation.errors
    )

    resigned = sign_scene_graph(tampered)
    validation = validate_scene_graph(resigned)
    assert not validation.passed
    assert any(
        error.startswith("unsupported_scene_backend:")
        for error in validation.errors
    )


def test_scene_graph_rejects_parent_cycles_remote_assets_and_unknown_motion() -> None:
    graph = _compiled_graph()
    graph = copy.deepcopy(graph)
    first, second = graph["nodes"][:2]
    first["parent_id"] = second["node_id"]
    second["parent_id"] = first["node_id"]
    first["content"]["asset"] = {"uri": "https://example.com/untrusted.png"}
    graph["motion_graph"]["tracks"][0]["property"] = "execute_javascript"
    graph = sign_scene_graph(graph)

    validation = validate_scene_graph(graph)

    assert not validation.passed
    assert any(error.startswith("scene_node_parent_cycle:") for error in validation.errors)
    assert any(error.startswith("remote_scene_asset_forbidden:") for error in validation.errors)
    assert any(
        error.startswith("unsupported_scene_motion_property:")
        for error in validation.errors
    )


def test_capability_manifest_is_closed_and_declares_fallbacks() -> None:
    graph = _compiled_graph()

    assert CAPABILITY_REGISTRY
    for capability in CAPABILITY_REGISTRY.values():
        assert capability.backend
        assert capability.fallback_backend
        assert capability.motion_properties
        assert capability.cost > 0

    tampered = copy.deepcopy(graph)
    tampered["capability_manifest"][0]["cost"] = 0.1
    tampered = sign_scene_graph(tampered)

    validation = validate_scene_graph(tampered)
    assert not validation.passed
    assert any(
        error.startswith("scene_graph_capability_manifest_tampered:")
        for error in validation.errors
    )
