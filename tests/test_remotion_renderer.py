from __future__ import annotations

from pathlib import Path

from renderers import resolve_renderer
from renderers.remotion_renderer import (
    RemotionRenderer,
    _build_input_props,
    _candidate_node_roots,
    _probe_node_packages_at,
)
from tools.auto_visuals import _compile_hyperframes_specs, _rendered_visual_quality_for_spec
from renderers.base import RenderedAsset


def _spec() -> dict:
    return {
        "visual_id": "visual_001",
        "card_id": "card_001",
        "template": "signal_network",
        "renderer_hint": "remotion",
        "visual_intent_type": "mechanism",
        "visual_type_hint": "process",
        "composition_mode": "replace",
        "headline": "Agents Coordinate",
        "deck": "Planner, tools, and memory move in sequence",
        "supporting_lines": ["Planner selects", "Tool acts", "Memory updates"],
        "steps": ["Plan", "Act", "Observe"],
        "keywords": ["planner", "tools", "memory"],
        "duration": 3.2,
        "importance": 0.8,
        "auto_visuals_director": {
            "director_score": 72,
            "copy_alignment": 0.62,
        },
    }


def test_remotion_renderer_resolves_as_strict_backend() -> None:
    renderer, reason = resolve_renderer(
        _spec(),
        preferred="remotion",
        allow_unavailable=True,
    )

    assert renderer.name == "remotion"
    assert "remotion was explicitly preferred" in reason


def test_remotion_input_props_preserve_structured_visual_data() -> None:
    props = _build_input_props(
        {
            **_spec(),
            "sentence_text": "Accuracy reaches 42% as the planner routes the job.",
            "context_text": (
                "Accuracy reaches 42% as the planner routes the job. "
                "Planner selects, the tool acts, and memory updates."
            ),
            "semantic_frame": {
                "steps": ["Planner selects", "the tool acts", "memory updates"],
                "viewer_takeaway": "Accuracy reaches 42%",
            },
            "metric_facts": [{"value": "42%", "label": "accuracy"}],
            "visual_beats": [{"text": "Planner routes the job"}],
        },
        width=1920,
        height=1080,
        fps=30,
    )

    program = props["program"]

    assert program["width"] == 1920
    assert program["height"] == 1080
    assert program["scene_family"] == "metric"
    assert any(node["value"] == "42%" for node in program["nodes"])
    assert program["quality_contract"]["required_labels"]
    assert program["quality_contract"]["min_motion_area"] == 0.018
    assert program["creative_direction"]["signature"]
    assert program["creative_direction"]["medium_family"] == "data_sculpture"


def test_hyperframes_compiler_bypasses_remotion_specs() -> None:
    plan, report = _compile_hyperframes_specs([_spec()])

    assert {key: plan[0][key] for key in _spec()} == _spec()
    assert plan[0]["video_design_bible"]["signature"]
    assert plan[0]["creative_direction_history"] == []
    assert report["compiled_count"] == 0
    assert report["accepted_count"] == 1
    assert report["estimated_render_count"] == 1


def test_remotion_rendered_quality_uses_renderer_metadata() -> None:
    asset = RenderedAsset(
        asset_path="visual.mp4",
        width=1920,
        height=1080,
        duration_sec=3.2,
        renderer="remotion",
        job_dir=".",
        script_path="entry.jsx",
        metadata={"quality_score": 0.72, "quality_passed": True},
    )

    qa = _rendered_visual_quality_for_spec(_spec(), asset)

    assert qa.renderer == "remotion"
    assert qa.passed
    assert qa.score >= 0.6


def test_remotion_renderer_scores_dom_explainer_specs() -> None:
    renderer = RemotionRenderer()

    assert renderer.supports(_spec())
    assert renderer.score_spec(_spec()) > 1.0


def test_remotion_prefers_managed_runtime_over_checkout(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    managed = tmp_path / "managed"
    monkeypatch.setattr(
        "renderers.remotion_renderer.managed_renderer_runtime_dir",
        lambda _node_path=None: managed,
    )

    assert _candidate_node_roots()[0] == managed.resolve()


def test_remotion_package_probe_loads_native_rspack_binding(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    (tmp_path / "node_modules").mkdir()
    probes: list[dict] = []
    monkeypatch.setattr(
        "renderers.remotion_renderer.resolve_node_executable",
        lambda: "/runtime/node",
    )

    def fake_probe(**kwargs):  # noqa: ANN001
        probes.append(kwargs)
        return {"available": True, "reason": ""}

    monkeypatch.setattr(
        "renderers.remotion_renderer.renderer_native_runtime_status",
        fake_probe,
    )

    assert _probe_node_packages_at(tmp_path) == (True, "")
    assert probes == [
        {
            "node_path": "/runtime/node",
            "node_root": tmp_path,
            "require_remotion": True,
        }
    ]


def test_remotion_react_entry_is_frame_driven_and_uses_measured_text() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "renderers"
        / "remotion_entry.jsx"
    ).read_text(encoding="utf-8")

    assert "fitText" in source
    assert "useCurrentFrame" in source
    assert "calculateMetadata" in source
    assert "data-vex-required-label" in source
    assert "DirectionBackdrop" in source
    assert "KineticTypeScene" in source
    assert "RelationConnector" in source
    assert "OpenVisualScene" in source
    assert "openTrackValue" in source
    assert "data-vex-open-visual-program" in source
    assert "data-vex-required-edge" in source
    assert "transition:" not in source
