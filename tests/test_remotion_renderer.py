from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from renderers import resolve_renderer
from renderers.remotion_renderer import (
    RemotionRenderer,
    _build_input_props,
    _candidate_preflight,
    _candidate_node_roots,
    _probe_node_packages_at,
    _report_signature,
    _render_fidelity,
)
from tests.test_remotion_semantic_pipeline import _grounded_process_spec
from tools.auto_visuals import _compile_hyperframes_specs, _rendered_visual_quality_for_spec
from renderers.base import RenderedAsset
from vex_remotion.compiler import compile_remotion_scene_program


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
    assert "SceneGraphScene" in source
    assert "SceneGraphLayer" in source
    assert "openTrackValue" in source
    assert "data-vex-open-visual-program" in source
    assert "data-vex-required-edge" in source
    assert "transition:" not in source
    assert "semanticFontFloor" in source
    assert "!program.scene_graph?.nodes?.length" in source


def test_remotion_scene_graph_runtime_has_specialized_renderers_and_solver() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "renderers"
        / "remotion_scene_graph.jsx"
    ).read_text(encoding="utf-8")

    assert "solveSceneGraphLayout" in source
    assert "RoutedRelations" in source
    assert "DataChart" in source
    assert "KineticText" in source
    assert "MetricMark" in source
    assert "MaskedMedia" in source
    assert "VectorPathNode" in source
    assert "data-vex-scene-graph-signature" in source
    assert "data-vex-relation-path" in source
    assert "no invented scale" in source
    assert "transition:" not in source


def test_remotion_runner_uses_lossless_intermediate_frames_and_software_gl() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "renderers"
        / "remotion_runner.mjs"
    ).read_text(encoding="utf-8")

    assert "imageFormat: 'png'" in source
    assert "VEX_REMOTION_GL" in source
    assert "const chromiumOptions = {gl: openGlRenderer}" in source
    assert "bundleFingerprint" in source
    assert "VEX_REMOTION_BUNDLE_CACHE_DIR" in source
    assert "openBrowser" in source
    assert "puppeteerInstance: browser" in source
    assert "renderStill" in source
    assert "renderMode === 'preview'" in source


def test_remotion_render_fidelity_fails_closed_to_final() -> None:
    assert _render_fidelity({}) == "final"
    assert _render_fidelity({"remotion_render_fidelity": "preview"}) == "preview"
    assert _render_fidelity({"remotion_render_fidelity": "stills"}) == "final"
    assert _render_fidelity({"remotion_render_fidelity": "../../escape"}) == "final"


def test_rendered_candidate_preflight_selects_and_signs_actual_frame_winner(
    monkeypatch,
    tmp_path: Path,
) -> None:  # noqa: ANN001
    source = _grounded_process_spec()
    compiled = compile_remotion_scene_program(
        source,
        width=1280,
        height=720,
        fps=30,
    )
    assert compiled.program is not None
    candidates = compiled.program.open_visual_program_candidates
    spec = {
        **source,
        "open_visual_program": candidates[0],
        "open_visual_program_candidates": candidates,
    }

    def fake_run(command, **_kwargs):  # noqa: ANN001
        job_dir = Path(command[-1])
        batch = json.loads(
            (job_dir / "candidate_input_props.json").read_text(encoding="utf-8")
        )
        rendered = []
        for item in batch:
            frame_dir = job_dir / "frames" / item["candidate_id"]
            frame_dir.mkdir(parents=True)
            frame_paths = []
            for index in range(3):
                frame_path = frame_dir / f"frame_{index}.png"
                frame_path.write_bytes(b"frame")
                frame_paths.append(str(frame_path))
            rendered.append(
                {
                    "candidate_id": item["candidate_id"],
                    "frame_paths": frame_paths,
                }
            )
        (job_dir / "remotion_result.json").write_text(
            json.dumps(
                {
                    "candidate_stills": rendered,
                    "bundle_fingerprint": "a" * 64,
                    "bundle_cache_hit": True,
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    scores = iter([0.61, 0.93, 0.72, 0.68, 0.81, 0.76])

    class FakeAesthetic:
        passed = True
        issues: list[str] = []
        warnings: list[str] = []

        def __init__(self, score: float) -> None:
            self.score = score

        def to_dict(self) -> dict:
            return {"passed": True, "score": self.score}

    monkeypatch.setattr(
        "renderers.remotion_renderer.subprocess.run",
        fake_run,
    )
    monkeypatch.setattr(
        "renderers.remotion_renderer.evaluate_frame_aesthetics",
        lambda *_args: FakeAesthetic(next(scores)),
    )

    selected_spec, report = _candidate_preflight(
        spec,
        job_dir=tmp_path / "job",
        node_path="node",
        node_root=tmp_path,
        width=1280,
        height=720,
        fps=30,
    )

    assert report["passed"]
    assert report["selection_mode"] == "rendered_contact_sheet"
    assert report["requested_candidate_count"] == len(candidates)
    assert report["rendered_candidate_count"] == len(candidates)
    assert report["selected_program_id"] == candidates[1]["program_id"]
    assert selected_spec["open_visual_program"]["program_id"] == candidates[1]["program_id"]
    assert selected_spec["open_visual_program_candidates"] == [candidates[1]]
    unsigned = {key: value for key, value in report.items() if key != "signature"}
    assert report["signature"] == _report_signature(unsigned)


def test_rendered_candidate_preflight_rejects_substituted_programs(
    monkeypatch,
    tmp_path: Path,
) -> None:  # noqa: ANN001
    source = _grounded_process_spec()
    compiled = compile_remotion_scene_program(
        source,
        width=1280,
        height=720,
        fps=30,
    )
    assert compiled.program is not None
    candidates = compiled.program.open_visual_program_candidates
    tampered_identity = {
        **source,
        "visual_id": "different_visual_identity",
        "open_visual_program": candidates[0],
        "open_visual_program_candidates": candidates,
    }
    monkeypatch.setattr(
        "renderers.remotion_renderer.subprocess.run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("substituted candidates must not reach the renderer")
        ),
    )

    selected_spec, report = _candidate_preflight(
        tampered_identity,
        job_dir=tmp_path / "job",
        node_path="node",
        node_root=tmp_path,
        width=1280,
        height=720,
        fps=30,
    )

    assert selected_spec == tampered_identity
    assert not report["enabled"]
    assert report["reason"] == "insufficient_valid_candidates"
    assert len(report["rejected"]) == len(candidates)
    assert all(
        "candidate_program_was_rejected_or_substituted" in item["errors"]
        for item in report["rejected"]
    )
