from __future__ import annotations

import json
import re
from pathlib import Path

from renderers import RendererMatch, resolve_renderer
from renderers.base import RenderedAsset, RendererStatus
from tools.auto_visuals import (
    _contextual_visual_budget,
    _filter_renderer_capabilities,
    _max_render_workers,
    _render_generated_visual,
    RenderedVisualQA,
)
from vex_hyperframes import build_composition, retrieve_skill_slices, validate_composition_html
from vex_hyperframes.qa import HyperframesQualityReport
from vex_hyperframes.variants import build_variants


NEW_HYPERFRAMES_TEMPLATES = (
    "concept_map",
    "problem_solution",
    "myth_buster",
    "checklist_reveal",
    "risk_radar",
    "opportunity_map",
    "scorecard",
    "pipeline_xray",
    "decision_tree",
    "momentum_wave",
    "focus_ring",
    "timeline_filmstrip",
    "quote_breakdown",
    "market_map",
    "mechanism_blueprint",
    "data_pulse",
)


def test_hyperframes_composition_is_self_contained_and_valid() -> None:
    composition = build_composition(_spec(), width=1920, height=1080, fps=30)
    report = validate_composition_html(
        composition.html,
        expected_width=1920,
        expected_height=1080,
        expected_duration=3.0,
    )

    assert report.valid, report.errors
    assert report.composition_id == "vex-visual_001"
    assert report.clip_count >= 4
    assert "window.__timelines[\"vex-visual_001\"]" in composition.html
    assert "https://" not in composition.html
    assert "requestAnimationFrame" not in composition.html


def test_hyperframes_new_template_pack_builds_valid_compositions() -> None:
    renderer = resolve_renderer(_spec(), preferred="hyperframes", allow_unavailable=True)[0]

    for template in NEW_HYPERFRAMES_TEMPLATES:
        spec = {
            **_spec(),
            "visual_id": f"visual_{template}",
            "template": template,
            "headline": template.replace("_", " ").title(),
            "supporting_lines": ["Input signal", "Context shift", "Output proof"],
            "steps": ["Start", "Mechanism", "Decision", "Payoff"],
            "keywords": ["signal", "context", "payoff"],
        }
        composition = build_composition(spec, width=1280, height=720, fps=30)
        report = validate_composition_html(
            composition.html,
            expected_width=1280,
            expected_height=720,
            expected_duration=3.0,
        )

        assert renderer.supports(spec)
        assert report.valid, (template, report.errors)
        assert composition.metadata["template"] == template


def test_hyperframes_metric_stage_uses_source_metrics_without_synthetic_percentages() -> None:
    composition = build_composition(
        {
            **_spec(),
            "template": "data_journey",
            "headline": "10% KV cache",
            "emphasis_text": "10%",
            "supporting_lines": ["DeepSeek V4 Pro requires less cache"],
            "metric_facts": [{"value": "10%", "label": "DeepSeek V4 Pro requires 10% KV cache"}],
        },
        width=1280,
        height=720,
        fps=30,
    )

    stat_values = re.findall(
        r'<div class="stat-card"[^>]*>.*?<span>([^<]+)</span>',
        composition.html,
        flags=re.DOTALL,
    )
    assert "10%" in stat_values
    assert not any(re.fullmatch(r"\d+%", value) and value != "10%" for value in stat_values)


def test_hyperframes_renderer_scores_premium_html_slides_above_manim() -> None:
    renderer, reason = resolve_renderer(
        _spec(),
        preferred="auto",
        allow_unavailable=True,
    )

    assert renderer.name == "hyperframes"
    assert "hyperframes scored" in reason


def test_hyperframes_renderer_scores_new_premium_templates() -> None:
    for template in NEW_HYPERFRAMES_TEMPLATES:
        renderer, reason = resolve_renderer(
            {**_spec(), "template": template},
            preferred="auto",
            allow_unavailable=True,
        )

        assert renderer.name == "hyperframes"
        assert "hyperframes scored" in reason


def test_hyperframes_automatic_route_rejects_legacy_template() -> None:
    import renderers.hyperframes_renderer as module

    renderer = module.HyperframesRenderer()

    assert renderer.score_spec(
        {
            **_spec(),
            "template": "concept_map",
            "hyperframes_automatic_semantic_route": True,
        }
    ) == -1.0


def test_hyperframes_skill_pack_includes_production_contract() -> None:
    skills = retrieve_skill_slices("signal_network")
    skill_ids = {skill.skill_id for skill in skills}

    assert "hyperframes-production-contract" in skill_ids
    assert any("window.__timelines" in " ".join(skill.rules) for skill in skills)


def test_auto_visuals_serializes_hyperframes_renders_to_avoid_npx_cache_contention() -> None:
    assert _max_render_workers({"renderer": "auto"}, 4, [_spec()]) == 1
    assert _max_render_workers({"renderer": "hyperframes"}, 4, [_spec()]) == 1
    assert _max_render_workers({"renderer": "both"}, 4, [_spec()]) == 1
    assert _max_render_workers({"renderer": "manim", "max_render_workers": 3}, 4, [_spec()]) == 3


def test_auto_visuals_filters_renderer_capabilities_for_strict_modes() -> None:
    capabilities = [
        {"name": "hyperframes", "available": True},
        {"name": "manim", "available": True},
        {"name": "ffmpeg", "available": True},
        {"name": "blender", "available": True},
    ]

    assert [item["name"] for item in _filter_renderer_capabilities(capabilities, "hyperframes")] == ["hyperframes"]
    assert [item["name"] for item in _filter_renderer_capabilities(capabilities, "manim")] == ["manim"]
    assert [item["name"] for item in _filter_renderer_capabilities(capabilities, "both")] == ["hyperframes", "manim"]


def test_hyperframes_preference_does_not_fall_back_to_manim(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    import tools.auto_visuals as module

    calls = []

    class FakeHyperframes:
        name = "hyperframes"

        def render(self, spec, *, render_root, width, height, fps):  # noqa: ANN001
            return RenderedAsset(
                asset_path=str(tmp_path / "visual.mp4"),
                width=width,
                height=height,
                duration_sec=float(spec.get("duration") or 3.0),
                renderer=self.name,
                job_dir=str(render_root),
                script_path="",
            )

    def fake_resolve_renderer(spec, *, preferred, exclude):  # noqa: ANN001
        calls.append((preferred, set(exclude)))
        assert "manim" in exclude
        return FakeHyperframes(), "hyperframes was explicitly preferred."

    monkeypatch.setattr(module, "resolve_renderer", fake_resolve_renderer)

    asset, _ = _render_generated_visual(
        {**_spec(), "renderer_hint": "manim"},
        preferred_renderer="hyperframes",
        allowed_renderers={"hyperframes"},
        render_root=tmp_path,
        width=1920,
        height=1080,
        fps=30,
    )

    assert asset.renderer == "hyperframes"
    assert calls[0][0] == "hyperframes"


def test_auto_visuals_quality_tournament_promotes_best_render_qa(
    monkeypatch,
    tmp_path: Path,
) -> None:  # noqa: ANN001
    import tools.auto_visuals as module

    class FakeRenderer:
        def __init__(self, name: str) -> None:
            self.name = name

        def render(self, spec, *, render_root, width, height, fps):  # noqa: ANN001
            return RenderedAsset(
                asset_path=str(render_root / f"{self.name}.mp4"),
                width=width,
                height=height,
                duration_sec=3.0,
                renderer=self.name,
                job_dir=str(render_root),
                script_path="",
            )

    hyperframes = FakeRenderer("hyperframes")
    ffmpeg = FakeRenderer("ffmpeg")
    monkeypatch.setattr(
        module,
        "rank_renderers",
        lambda *args, **kwargs: [
            RendererMatch(hyperframes, 1.2, "hyperframes ranked first"),
            RendererMatch(ffmpeg, 0.7, "ffmpeg ranked second"),
        ],
    )
    monkeypatch.setattr(
        module,
        "_rendered_visual_quality_for_spec",
        lambda spec, asset: RenderedVisualQA(
            visual_id=str(spec.get("visual_id")),
            renderer=asset.renderer,
            score=0.93 if asset.renderer == "ffmpeg" else 0.64,
            passed=asset.renderer == "ffmpeg",
            issues=[] if asset.renderer == "ffmpeg" else ["semantic_qa_failed"],
            warnings=[],
            repair_action="keep" if asset.renderer == "ffmpeg" else "drop",
        ),
    )

    asset, reason = _render_generated_visual(
        {
            **_spec(),
            "renderer_hint": "hyperframes",
            "visual_intent_type": "mechanism",
        },
        preferred_renderer="auto",
        render_root=tmp_path,
        width=1920,
        height=1080,
        fps=30,
        renderer_strategy="quality_tournament",
        tournament_size=2,
    )

    assert asset.renderer == "ffmpeg"
    assert "promoted ffmpeg" in reason
    report = asset.metadata["renderer_tournament"]
    assert report["rendered_count"] == 2
    assert report["selected_qa_passed"] is True
    assert [item["renderer"] for item in report["attempts"]] == [
        "hyperframes",
        "ffmpeg",
    ]


def test_auto_visuals_quality_tournament_continues_after_renderer_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:  # noqa: ANN001
    import tools.auto_visuals as module

    class FailingRenderer:
        name = "hyperframes"

        def render(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise module.VisualRendererError("proof render failed")

    class PassingRenderer:
        name = "ffmpeg"

        def render(self, spec, *, render_root, width, height, fps):  # noqa: ANN001
            return RenderedAsset(
                asset_path=str(render_root / "ffmpeg.mp4"),
                width=width,
                height=height,
                duration_sec=3.0,
                renderer=self.name,
                job_dir=str(render_root),
                script_path="",
            )

    monkeypatch.setattr(
        module,
        "rank_renderers",
        lambda *args, **kwargs: [
            RendererMatch(FailingRenderer(), 1.2, "first"),
            RendererMatch(PassingRenderer(), 0.7, "second"),
        ],
    )
    monkeypatch.setattr(
        module,
        "_rendered_visual_quality_for_spec",
        lambda spec, asset: RenderedVisualQA(
            visual_id=str(spec.get("visual_id")),
            renderer=asset.renderer,
            score=0.82,
            passed=True,
            issues=[],
            warnings=[],
            repair_action="keep",
        ),
    )

    asset, _ = _render_generated_visual(
        {
            **_spec(),
            "renderer_hint": "hyperframes",
            "visual_intent_type": "mechanism",
        },
        preferred_renderer="auto",
        render_root=tmp_path,
        width=1920,
        height=1080,
        fps=30,
        renderer_strategy="quality_tournament",
        tournament_size=1,
    )

    assert asset.renderer == "ffmpeg"
    attempts = asset.metadata["renderer_tournament"]["attempts"]
    assert attempts[0]["rendered"] is False
    assert attempts[1]["rendered"] is True


def test_auto_visuals_default_budget_scales_with_contextual_opportunities() -> None:
    cards = [
        {
            "visualizability": 0.66,
            "intuition_payoff": 0.68,
            "numeric_hits": 1 if index % 3 == 0 else 0,
            "process_cues": 0.36,
            "contrast_cues": 0.12,
        }
        for index in range(11)
    ]

    budget = _contextual_visual_budget(
        cards,
        clip_duration=92.0,
        renderer_name="hyperframes",
        mode="generated_only",
    )

    assert budget >= 10
    assert budget <= 16


def test_hyperframes_command_uses_local_cli_without_runtime_install(monkeypatch, tmp_path: Path) -> None:
    import renderers.hyperframes_renderer as module

    cli_path = tmp_path / "node_modules" / ".bin" / module._local_bin_name("hyperframes")
    cli_path.parent.mkdir(parents=True)
    cli_path.write_text("cli", encoding="utf-8")
    monkeypatch.setattr(module.config, "HYPERFRAMES_CLI_PATH", str(cli_path))

    command = module._hyperframes_command("lint")

    assert command[0] == str(cli_path)
    assert command[-1] == "lint"
    assert "npx" not in command
    assert "--yes" not in command


def test_hyperframes_command_does_not_fall_back_to_global_or_cwd_path(monkeypatch, tmp_path: Path) -> None:
    import renderers.hyperframes_renderer as module

    fake_renderer = tmp_path / "repo" / "renderers" / "hyperframes_renderer.py"
    fake_renderer.parent.mkdir(parents=True)
    fake_renderer.write_text("# test", encoding="utf-8")
    fake_cwd = tmp_path / "cwd"
    fake_cwd.mkdir()
    monkeypatch.chdir(fake_cwd)
    monkeypatch.setattr(module, "__file__", str(fake_renderer))
    monkeypatch.setattr(module.config, "HYPERFRAMES_CLI_PATH", "hyperframes")
    monkeypatch.setattr(module.shutil, "which", lambda _name: str(tmp_path / "global" / "hyperframes"))
    monkeypatch.setattr(
        module,
        "managed_hyperframes_cli_path",
        lambda: tmp_path / "managed" / "hyperframes",
    )
    cwd_cli = fake_cwd / "node_modules" / ".bin" / module._local_bin_name("hyperframes")
    cwd_cli.parent.mkdir(parents=True)
    cwd_cli.write_text("untrusted", encoding="utf-8")

    assert module._hyperframes_cli_path() is None


def test_hyperframes_render_promotes_root_metadata_path(monkeypatch, tmp_path: Path) -> None:
    import renderers.hyperframes_renderer as module

    renderer = module.HyperframesRenderer()
    monkeypatch.setattr(renderer, "availability", lambda: RendererStatus(True, "ok"))
    monkeypatch.setattr(module.config, "HYPERFRAMES_VARIANT_COUNT", 1)
    monkeypatch.setattr(
        module,
        "probe_video",
        lambda _: {"width": 1920, "height": 1080, "duration_sec": 3.0},
    )

    def fake_render_variant(variant, *, job_dir, width, height, fps):
        variant_dir = job_dir / "variants" / variant.variant_id
        variant_dir.mkdir(parents=True, exist_ok=True)
        asset_path = variant_dir / "variant.mp4"
        asset_path.write_bytes(b"fake")
        index_path = variant_dir / "index.html"
        metadata_path = variant_dir / "hyperframes_metadata.json"
        index_path.write_text("<html></html>", encoding="utf-8")
        metadata_path.write_text("{}", encoding="utf-8")
        return {
            "variant_id": variant.variant_id,
            "variant_index": variant.variant_index,
            "asset_path": str(asset_path),
            "script_path": str(index_path),
            "artifact_paths": {"metadata_path": str(metadata_path)},
            "metadata": {"duration_sec": 3.0},
            "qa": {"score": 0.91, "passed": True},
        }

    monkeypatch.setattr(renderer, "_render_variant", fake_render_variant)

    asset = renderer.render(_spec(), tmp_path, width=1920, height=1080, fps=30)

    assert asset.artifact_paths["metadata_path"] == str(tmp_path / "visual_001" / "hyperframes_metadata.json")
    assert Path(asset.artifact_paths["variant_metadata_path"]).parts[-3:] == (
        "variants",
        "variant_01",
        "hyperframes_metadata.json",
    )


def test_hyperframes_variant_cli_runs_inside_variant_workspace(monkeypatch, tmp_path: Path) -> None:
    import renderers.hyperframes_renderer as module

    calls = []

    class FakeResult:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(command, *, cwd, capture_output, text, timeout):
        calls.append((list(command), cwd, timeout))
        if "render" in command:
            output_path = Path(command[command.index("--output") + 1])
            output_path.write_bytes(b"fake")
        return FakeResult()

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module, "_hyperframes_cli_path", lambda: "hyperframes")
    monkeypatch.setattr(module.config, "HYPERFRAMES_RENDER_QUALITY", "")
    monkeypatch.setattr(
        module,
        "probe_video",
        lambda _: {"width": 1920, "height": 1080, "duration_sec": 3.0},
    )
    monkeypatch.setattr(module, "extract_quality_frames", lambda *_, **__: [])
    monkeypatch.setattr(
        module,
        "analyze_hyperframes_quality",
        lambda **_: HyperframesQualityReport(passed=True, score=0.92),
    )

    variant = build_variants(_spec(), default_count=1)[0]
    module.HyperframesRenderer()._render_variant(
        variant,
        job_dir=tmp_path / "job",
        width=1920,
        height=1080,
        fps=30,
    )

    assert calls[0][0][-1] == "."
    assert calls[1][0][-1] == "."
    assert calls[0][1] == calls[1][1]
    assert calls[1][2] is None


def test_hyperframes_render_runs_monotonic_cegis_and_final_judge(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import renderers.hyperframes_renderer as module
    from vex_hyperframes.compiler import compile_hyperframes_plan
    from vex_hyperframes.final_judge import FinalIndependentVerdict

    plan = compile_hyperframes_plan(
        {
            "visual_id": "cegis_route",
            "sentence_text": (
                "The request is classified, checked against policy, then sent "
                "to a human."
            ),
            "context_text": "The handoff prevents unsupported answers.",
            "semantic_frame": {
                "steps": [
                    "Classify request",
                    "Check policy",
                    "Send to human",
                ],
                "result": "Prevent unsupported answers",
            },
            "required_labels": [
                "Classify request",
                "Check policy",
                "Send to human",
                "Prevent unsupported answers",
            ],
            "duration": 4.0,
            "composition_mode": "replace",
        }
    )
    spec = {
        **plan.renderer_spec,
        "visual_proof_programs": plan.renderer_spec[
            "visual_proof_programs"
        ][:1],
    }
    renderer = module.HyperframesRenderer()
    monkeypatch.setattr(renderer, "availability", lambda: RendererStatus(True, "ok"))
    monkeypatch.setattr(
        module,
        "probe_video",
        lambda _: {"width": 1920, "height": 1080, "duration_sec": 4.0},
    )

    def fake_render_variant(variant, *, job_dir, width, height, fps):
        variant_dir = job_dir / "variants" / variant.variant_id
        variant_dir.mkdir(parents=True, exist_ok=True)
        asset_path = variant_dir / "variant.mp4"
        asset_path.write_bytes(b"fake")
        relation = variant.spec["scene_program_v2"]["relations"][0]
        repaired = "_repair_" in variant.variant_id
        hard = [] if repaired else [
            {
                "counterexample_id": "blind_01_missing_relation",
                "critic": "blind",
                "issue_type": "missing_relation",
                "severity": "hard_failure",
                "summary": "Route is unclear.",
                "expected": "Readable route.",
                "observed": "No route.",
                "confidence": 0.96,
                "frame_id": "",
                "timestamp_sec": None,
                "element_ids": [],
                "relation_ids": [relation["relation_id"]],
                "evidence_ids": relation["evidence_ids"],
                "regions": [],
                "allowed_repairs": ["strengthen_relation"],
            }
        ]
        score = 0.91 if repaired else 0.62
        return {
            "variant_id": variant.variant_id,
            "variant_index": variant.variant_index,
            "asset_path": str(asset_path),
            "script_path": str(variant_dir / "index.html"),
            "job_dir": str(variant_dir),
            "artifact_paths": {"qa_frame_paths": []},
            "metadata": {
                "duration_sec": 4.0,
                "scene_program_v2": variant.spec["scene_program_v2"],
                "visual_explanation_ir": variant.spec[
                    "visual_explanation_ir"
                ],
                "visual_claim_graph": variant.spec["visual_claim_graph"],
                "hyperframes_production_contract": variant.spec[
                    "hyperframes_production_contract"
                ],
                "stage": {
                    "object_coverage": 1.0,
                    "relation_coverage": 1.0,
                },
                "semantic_qa": {
                    "score": score,
                    "hard_failures": [],
                    "object_coverage": 1.0,
                },
                "visual_critics": {
                    "passed": repaired,
                    "score": score,
                    "hard_failure_count": len(hard),
                    "counterexamples": hard,
                },
            },
            "qa": {"score": score, "passed": repaired},
        }

    monkeypatch.setattr(renderer, "_render_variant", fake_render_variant)
    monkeypatch.setattr(
        module,
        "judge_final_candidate",
        lambda *_, **__: FinalIndependentVerdict(
            version="hyperframes-independent-final-judge-v1",
            available=False,
            passed=True,
            score=0.91,
            thesis="",
            local_gate_passed=True,
        ),
    )

    asset = renderer.render(
        spec,
        tmp_path,
        width=1920,
        height=1080,
        fps=30,
    )

    assert asset.metadata["selected_variant_id"].endswith("_repair_01")
    assert Path(asset.artifact_paths["repair_history_path"]).is_file()
    assert Path(
        asset.artifact_paths["final_independent_verdict_path"]
    ).is_file()
    history = json.loads(
        Path(asset.artifact_paths["repair_history_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert history["rounds"][0]["accepted"] is True
    assert history["rounds"][0]["reason"] == "hard_failures_reduced"


def _spec() -> dict:
    return {
        "visual_id": "visual_001",
        "template": "signal_network",
        "duration": 3.0,
        "headline": "Build First",
        "deck": "The learning loop gets shorter",
        "steps": ["Build", "Get stuck", "Target study", "Ship"],
        "visual_type_hint": "process",
        "composition_mode": "replace",
        "style_pack": "signal_lab",
        "importance": 0.9,
    }
