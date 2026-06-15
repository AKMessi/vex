from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace

from tools.auto_visuals import (
    _compile_hyperframes_specs,
    _compile_hyperframes_specs_with_reserves,
    _prepare_visual_spec,
    _prior_auto_visual_failure_card_ids,
)
from vex_hyperframes.composer import build_composition


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def test_auto_visuals_compiles_hyperframes_into_grounded_semantic_scene() -> None:
    compiled, report = _compile_hyperframes_specs([_interface_spec()])

    assert report["input_count"] == 1
    assert report["compiled_count"] == 1
    assert report["rejected_count"] == 0
    assert compiled[0]["template"] == "semantic_interface"
    assert compiled[0]["hyperframes_production_contract"]["semantic_signature"]
    assert compiled[0]["qa_contract"]["required_labels"]
    assert len(compiled[0]["visual_proof_programs"]) == 4
    assert report["proof_candidate_count"] == 4
    assert report["estimated_render_count"] == 4
    assert report["compiled"][0]["proof_tournament_signature"]
    assert report["compiled"][0]["claim_graph_signature"]
    assert report["compiled"][0]["blind_inverse_decoder"]["enabled"] is True
    assert compiled[0]["semantic_continuity"] == {
        "continuity_group": "editor-workflow",
        "concept_ids": ["failed-shot-recovery"],
        "concept_color": "#EAB308",
        "motif": "inspection-rail",
        "episode_id": "episode-1",
    }
    assert compiled[0]["theme"]["accent"] == "#EAB308"
    assert compiled[0]["hyperframes_automatic_semantic_route"] is True
    assert compiled[0]["hyperframes_legacy_template_policy"] == "manual_only"
    assert compiled[0]["scene_program_v2"]["version"].endswith("-v2")
    repair = compiled[0]["hyperframes_compiler"][
        "counterexample_guided_repair"
    ]
    assert repair["enabled"] is True
    assert repair["max_rounds"] >= 1


def test_auto_visuals_rejects_generic_hyperframes_filler_before_render() -> None:
    compiled, report = _compile_hyperframes_specs(
        [
            {
                "visual_id": "visual_filler",
                "card_id": "card_filler",
                "renderer_hint": "hyperframes",
                "template": "keyword_stack",
                "sentence_text": "This idea changes everything.",
                "context_text": "The source provides no process, comparison, metric, or relationship.",
                "headline": "Big idea",
                "duration": 3.0,
            }
        ]
    )

    assert compiled == []
    assert report["rejected_count"] == 1
    assert report["rejected"][0]["render_policy"] == "reject"
    assert "no_supported_explanatory_structure" in (
        report["rejected"][0]["issues"]
        + report["rejected"][0]["rejection_reasons"]
    )


def test_semantic_compiler_substitutes_executable_reserve() -> None:
    primary = {
        "visual_id": "visual_primary",
        "card_id": "card_primary",
        "renderer_hint": "hyperframes",
        "template": "keyword_stack",
        "sentence_text": "This changes everything.",
        "context_text": "No source-grounded structure is present.",
        "headline": "Big idea",
        "duration": 3.0,
        "start": 2.0,
        "end": 5.0,
        "semantic_episode_id": "episode-primary",
    }
    reserve = {
        **_interface_spec(),
        "visual_id": "reserve_001",
        "card_id": "card_reserve",
        "start": 8.0,
        "end": 12.0,
        "semantic_episode_id": "episode-reserve",
        "source_card_ids": ["subtitle_008", "subtitle_009"],
        "opportunity_contract": {"score": 0.91},
    }

    selected, remaining, report = _compile_hyperframes_specs_with_reserves(
        [primary],
        [reserve],
        target_count=1,
    )

    assert [item["card_id"] for item in selected] == ["card_reserve"]
    assert selected[0]["source_card_ids"] == ["subtitle_008", "subtitle_009"]
    assert selected[0]["opportunity_recovery"]["stage"] == "semantic_compile"
    assert remaining == []
    assert report["reserve_substitutions"][0]["card_id"] == "card_reserve"


def test_prior_failure_memory_tracks_source_subtitle_cards(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "failed_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "plan": [
                    {
                        "card_id": "opportunity_001",
                        "source_card_ids": ["subtitle_004", "subtitle_005"],
                    }
                ],
                "hyperframes_compiler": {
                    "rejected": [
                        {
                            "card_id": "opportunity_002",
                            "source_card_ids": ["subtitle_011"],
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "creative_runs.json").write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "feature": "auto_visuals",
                        "manifest_path": str(manifest_path),
                        "summary": {"status": "failed_qa"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    failed_ids = _prior_auto_visual_failure_card_ids(
        SimpleNamespace(working_dir=str(tmp_path))
    )

    assert failed_ids == {
        "opportunity_001",
        "opportunity_002",
        "subtitle_004",
        "subtitle_005",
        "subtitle_011",
    }


def test_prepare_visual_spec_extracts_real_frame_for_grounded_interface(
    tmp_path: Path,
    monkeypatch,
) -> None:
    working_file = tmp_path / "source.mp4"
    working_file.write_bytes(b"video")
    bundle_dir = tmp_path / "bundle"
    state = SimpleNamespace(
        working_file=str(working_file),
        working_dir=str(tmp_path),
        output_dir=str(tmp_path / "output"),
        source_files=[str(working_file)],
    )

    def fake_extract(
        video_path: str,
        output_path: Path,
        *,
        time_sec: float,
    ) -> bool:
        assert video_path == str(working_file)
        assert time_sec == 3.5
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(PNG_1X1)
        return True

    monkeypatch.setattr(
        "tools.auto_visuals._extract_source_grounding_frame",
        fake_extract,
    )
    spec = _interface_spec()
    spec["visual_explanation_ir"] = {
        "scene_type": "grounded_interface_walkthrough"
    }
    spec["auto_visuals_director"] = {
        "source_frame_analysis": {
            "source_type": "screen_or_slide",
            "time_sec": 3.5,
        }
    }

    prepared = _prepare_visual_spec(
        spec,
        style_pack="auto",
        provider_name="gemini",
        model_name="test-model",
        state=state,
        bundle_dir=bundle_dir,
    )

    grounding = prepared["source_asset_grounding"]
    assert grounding["kind"] == "source_video_frame"
    assert Path(grounding["asset_path"]).read_bytes() == PNG_1X1
    assert str(bundle_dir.resolve()) in prepared["allowed_asset_roots"]


def test_semantic_interface_embeds_only_approved_source_images(
    tmp_path: Path,
) -> None:
    approved_root = tmp_path / "approved"
    approved_root.mkdir()
    source_image = approved_root / "interface.png"
    source_image.write_bytes(PNG_1X1)
    compiled, _ = _compile_hyperframes_specs([_interface_spec()])
    renderer_spec = {
        **compiled[0],
        "allowed_asset_roots": [str(approved_root)],
        "source_asset_grounding": {
            "kind": "source_video_frame",
            "asset_path": str(source_image),
            "source_type": "screen_or_slide",
            "time_sec": 3.5,
        },
    }

    composition = build_composition(renderer_spec, width=1280, height=720, fps=30)

    assert "data:image/png;base64," in composition.html
    assert str(source_image) not in composition.html
    assert composition.metadata["stage"]["source_asset_grounded"] is True
    assert composition.metadata["semantic_continuity"]["motif"] == "inspection-rail"

    renderer_spec["allowed_asset_roots"] = [str(tmp_path / "denied")]
    denied = build_composition(renderer_spec, width=1280, height=720, fps=30)
    assert "data:image/png;base64," not in denied.html
    assert denied.metadata["stage"]["source_asset_grounded"] is False


def _interface_spec() -> dict[str, object]:
    return {
        "visual_id": "visual_interface",
        "card_id": "card_interface",
        "renderer_hint": "hyperframes",
        "template": "interface_cascade",
        "visual_type_hint": "product_ui",
        "sentence_text": (
            "The editor highlights the failed shot, opens its render log, "
            "and lets the user retry only that shot."
        ),
        "context_text": "The source recording contains the actual editor interface.",
        "semantic_frame": {
            "screen": "Editor interface",
            "focus": "Failed shot",
            "action": "Open render log",
            "result": "Retry that shot",
        },
        "required_labels": ["Failed shot", "Render log", "Retry that shot"],
        "duration": 4.0,
        "composition_mode": "replace",
        "continuity_group": "editor-workflow",
        "concept_ids": ["failed-shot-recovery"],
        "episode_id": "episode-1",
        "episode_context": {
            "motif": "inspection-rail",
            "concepts": [
                {
                    "concept_id": "failed-shot-recovery",
                    "color": "#EAB308",
                }
            ],
        },
    }
