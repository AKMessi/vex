from __future__ import annotations

import copy
import json

import config
import pytest
import tools.auto_visuals as auto_visuals
from tools.auto_visuals import (
    _ModelPlanningBudget,
    _ModelPlanningBudgetExhausted,
    _compile_open_visual_specs,
)
from vex_hyperframes.composer import build_composition
from vex_hyperframes.open_visual_runtime import compile_open_visual_stage
from vex_hyperframes.variants import build_variants, select_best_variant
from vex_visuals.generative_authoring import author_open_visual_programs
from vex_visuals.open_visual_program import (
    apply_open_visual_patch,
    build_open_visual_program_candidates,
    open_visual_program_signature,
    select_open_visual_program,
    sign_open_visual_program,
    validate_open_visual_program,
)


def _compression_ir() -> dict:
    evidence = (
        "You don't read all million pages per thought. Every four tokens become "
        "one compressed KV entry. Then an indexer scores each compressed block "
        "and picks only the top blocks."
    )
    return {
        "visual_id": "visual_001",
        "scene_type": "guided_process",
        "thesis": "Compressed Sparse Attention",
        "takeaway": "The indexer scores each compressed block",
        "render_policy": "render",
        "evidence": [
            {
                "evidence_id": "evidence_transcript",
                "source_type": "transcript",
                "text": evidence,
                "confidence": 1.0,
            }
        ],
        "facts": [
            {
                "fact_id": "fact_problem",
                "fact_type": "problem",
                "label": "You don't read all million pages per thought",
                "evidence_ids": ["evidence_transcript"],
                "grounding": "transcript_exact",
            },
            {
                "fact_id": "fact_compress",
                "fact_type": "mechanism",
                "label": "Every four tokens become one compressed KV entry",
                "evidence_ids": ["evidence_transcript"],
                "grounding": "transcript_exact",
            },
            {
                "fact_id": "fact_index",
                "fact_type": "result",
                "label": "The indexer scores each compressed block",
                "evidence_ids": ["evidence_transcript"],
                "grounding": "transcript_exact",
            },
        ],
        "objects": [
            {
                "object_id": "object_problem",
                "role": "problem",
                "label": "You don't read all million pages per thought",
                "fact_ids": ["fact_problem"],
            },
            {
                "object_id": "object_compress",
                "role": "mechanism",
                "label": "Every four tokens become one compressed KV entry",
                "fact_ids": ["fact_compress"],
            },
            {
                "object_id": "object_index",
                "role": "result",
                "label": "The indexer scores each compressed block",
                "fact_ids": ["fact_index"],
            },
        ],
        "relations": [
            {
                "relation_id": "relation_compress",
                "source_id": "object_problem",
                "target_id": "object_compress",
                "relation_type": "precedes",
                "evidence_ids": ["evidence_transcript"],
            },
            {
                "relation_id": "relation_index",
                "source_id": "object_compress",
                "target_id": "object_index",
                "relation_type": "precedes",
                "evidence_ids": ["evidence_transcript"],
            },
        ],
        "required_labels": [
            "You don't read all million pages per thought",
            "Every four tokens become one compressed KV entry",
            "The indexer scores each compressed block",
        ],
        "forbidden_content": ["invented metrics"],
        "metadata": {
            "display_title": "Compressed Sparse Attention",
            "display_title_evidence": "Compressed Sparse Attention explains how compression works.",
        },
    }


def _candidates() -> list[dict]:
    return build_open_visual_program_candidates(
        _compression_ir(),
        visual_id="visual_001",
        width=1920,
        height=1080,
        duration_sec=4.8,
        fps=60,
        candidate_count=3,
    )


def test_compression_program_is_deterministic_grounded_and_selected() -> None:
    first = _candidates()
    second = _candidates()

    assert len(first) == 3
    assert [item["signature"] for item in first] == [
        item["signature"] for item in second
    ]
    assert all(
        validate_open_visual_program(item, ir=_compression_ir()).passed
        for item in first
    )
    tournament = select_open_visual_program(first, ir=_compression_ir())
    selected = next(
        item
        for item in first
        if item["program_id"] == tournament.selected_program_id
    )

    assert selected["concept"]["medium"] == "spatial_metaphor"
    assert {item["property"] for item in selected["tracks"]} >= {
        "progress",
        "scale",
        "translate_x",
        "translate_y",
    }
    labels = {
        str(item.get("element_id")): str(item.get("text") or "")
        for item in selected["elements"]
    }
    assert labels["source_label"] == "You don't read all million pages per thought"
    assert labels["compressed_output"] == "Every four tokens become one compressed KV entry"
    assert labels["indexer_result"] == "The indexer scores each compressed block"


def test_open_visual_runtime_owns_canvas_and_repairs_palette_contrast() -> None:
    program = copy.deepcopy(_candidates()[0])
    program["palette"].update(
        {
            "background": "#101418",
            "surface": "#F4F0E8",
            "ink": "#111111",
            "muted": "#252A30",
        }
    )
    program["canvas"].update({"width": 1280, "height": 720})
    program = sign_open_visual_program(program)

    compiled = compile_open_visual_stage(program, ir=_compression_ir())

    assert "background:#101418" in compiled.html
    assert "--ovp-canvas-text:#F4F0E8" in compiled.html
    assert 'data-vex-required-label="You don&#x27;t read all million pages per thought"' in compiled.html
    assert 'data-vex-required-label="Every four tokens become one compressed KV entry"' in compiled.html
    assert "font-size:clamp(22px" in compiled.html


def test_open_visual_composition_overrides_legacy_stage_insets() -> None:
    program = _candidates()[0]
    composition = build_composition(
        {
            "visual_id": "visual_001",
            "template": "signal_network",
            "duration": 4.8,
            "visual_explanation_ir": _compression_ir(),
            "open_visual_program": program,
        },
        width=1920,
        height=1080,
        fps=60,
    )

    assert ".stage.open-visual-stage { inset: 0; overflow: hidden; }" in composition.html
    assert ".open-visual-stage .ovp-stage { position: absolute; inset: 0; }" in composition.html


def test_signature_tampering_and_invented_copy_are_rejected() -> None:
    program = _candidates()[0]
    tampered = copy.deepcopy(program)
    tampered["concept"]["takeaway"] = "Unsupported 99 percent improvement"

    validation = validate_open_visual_program(tampered, ir=_compression_ir())

    assert not validation.passed
    assert "open_visual_program_signature_mismatch" in validation.errors

    invented = copy.deepcopy(program)
    invented["elements"][0]["text"] = "Guaranteed 99 percent speedup"
    invented = sign_open_visual_program(invented)
    validation = validate_open_visual_program(invented, ir=_compression_ir())
    assert not validation.passed
    assert any(error.startswith("ungrounded_element_copy:") for error in validation.errors)


def test_temporal_proof_rejects_hidden_context_and_implicit_relations() -> None:
    program = _candidates()[0]
    validation = validate_open_visual_program(program, ir=_compression_ir())
    relation_ids = {item["relation_id"] for item in program["relations"]}
    relation_track_targets = {
        item["target_id"]
        for item in program["tracks"]
        if item["property"] == "progress"
    }

    assert validation.temporal_proof_score >= 0.9
    assert relation_ids <= relation_track_targets

    hidden = copy.deepcopy(program)
    title_track = next(item for item in hidden["tracks"] if item["track_id"] == "title_opacity")
    title_track["keyframes"][0]["value"] = 0.0
    hidden = sign_open_visual_program(hidden)
    hidden_validation = validate_open_visual_program(hidden, ir=_compression_ir())
    assert "temporal_proof_initial_context_hidden:title" in hidden_validation.errors

    disconnected = copy.deepcopy(program)
    relation_id = disconnected["relations"][0]["relation_id"]
    disconnected["tracks"] = [
        item for item in disconnected["tracks"] if item["target_id"] != relation_id
    ]
    disconnected = sign_open_visual_program(disconnected)
    disconnected_validation = validate_open_visual_program(
        disconnected,
        ir=_compression_ir(),
    )
    assert (
        f"temporal_proof_relation_has_no_reveal_track:{relation_id}"
        in disconnected_validation.errors
    )


def test_resource_abuse_and_unknown_semantic_bindings_are_rejected() -> None:
    program = copy.deepcopy(_candidates()[0])
    program["elements"][0]["repeat"] = 10_000
    program["elements"][1]["binding"] = {
        "kind": "object",
        "id": "invented_object",
    }
    program = sign_open_visual_program(program)

    validation = validate_open_visual_program(program, ir=_compression_ir())

    assert not validation.passed
    assert any(error.startswith("invalid_element_repeat:") for error in validation.errors)
    assert any(error.startswith("invalid_semantic_binding:") for error in validation.errors)


def test_decorative_elements_do_not_require_evidence_bindings() -> None:
    program = copy.deepcopy(_candidates()[0])
    program["elements"].append(
        {
            "element_id": "decorative_registration_mark",
            "type": "shape",
            "role": "registration_mark",
            "text": "",
            "decorative": True,
            "layout": {
                "x": 0.94,
                "y": 0.06,
                "width": 0.02,
                "height": 0.02,
                "anchor": "top_left",
            },
            "style": {"fill": "accent"},
            "repeat": 1,
        }
    )
    program = sign_open_visual_program(program)

    validation = validate_open_visual_program(program, ir=_compression_ir())

    assert validation.passed


def test_typed_patch_is_resigned_and_revalidated() -> None:
    program = _candidates()[0]
    result = apply_open_visual_patch(
        program,
        [
            {
                "op": "move",
                "target_id": "compressed_output",
                "x": 0.6,
                "y": 0.39,
            },
            {
                "op": "set_style",
                "target_id": "compressed_output",
                "style": {"stroke_width": 4, "position": "fixed"},
            },
        ],
        ir=_compression_ir(),
    )

    assert result.passed
    assert result.program["signature"] == open_visual_program_signature(result.program)
    output = next(
        item
        for item in result.program["elements"]
        if item["element_id"] == "compressed_output"
    )
    assert output["layout"]["x"] == 0.6
    assert "position" not in output["style"]


def test_model_authoring_repairs_invalid_first_response() -> None:
    valid = copy.deepcopy(_candidates()[0])
    valid.pop("signature")
    valid["program_id"] = "model-authored-compression"
    valid["concept"]["composition"] = "model authored compression field"
    responses = iter(
        [
            json.dumps({"programs": [{"program_id": "broken", "elements": []}]}),
            json.dumps({"programs": [valid]}),
        ]
    )

    result = author_open_visual_programs(
        {
            "visual_id": "visual_001",
            "duration": 4.8,
            "generation_provider": "gemini",
            "generation_model": "test-model",
        },
        ir=_compression_ir(),
        width=1920,
        height=1080,
        fps=60,
        reasoning_call=lambda *_args: next(responses),
        candidate_count=3,
        max_model_attempts=2,
    )

    assert result.passed
    assert result.model_attempts == 2
    assert result.model_program_count == 1
    assert result.rejected_model_programs


def test_hyperframes_runtime_executes_open_program_and_maps_candidates() -> None:
    candidates = _candidates()
    spec = {
        "visual_id": "visual_001",
        "template": "signal_network",
        "duration": 4.8,
        "visual_explanation_ir": _compression_ir(),
        "open_visual_program": candidates[0],
        "open_visual_program_candidates": candidates,
        "visual_proof_programs": [
            {"proof_program_id": "proof_01"},
            {"proof_program_id": "proof_02"},
            {"proof_program_id": "proof_03"},
        ],
    }

    variants = build_variants(spec)
    composition = build_composition(
        variants[0].spec,
        width=1920,
        height=1080,
        fps=60,
    )

    assert [
        item.spec["open_visual_program"]["program_id"] for item in variants
    ] == [item["program_id"] for item in candidates]
    assert composition.metadata["stage"]["generation_mode"] == "open_visual_program"
    assert 'data-open-visual-program="visual_001-ovp-01"' in composition.html


def test_hyperframes_selection_rewards_semantic_fitness() -> None:
    records = [
        {
            "variant_id": "pretty_but_generic",
            "asset_path": "generic.mp4",
            "eligible_for_selection": True,
            "qa": {"passed": True, "score": 0.95},
            "metadata": {
                "stage": {"semantic_fitness": 0.63},
                "vision_qa": {"available": False},
            },
        },
        {
            "variant_id": "meaningful_mechanism",
            "asset_path": "mechanism.mp4",
            "eligible_for_selection": True,
            "qa": {"passed": True, "score": 0.82},
            "metadata": {
                "stage": {"semantic_fitness": 1.0},
                "vision_qa": {"available": False},
            },
        },
    ]

    selected = select_best_variant(records)

    assert selected is not None
    assert selected["variant_id"] == "meaningful_mechanism"


def test_auto_visuals_pipeline_attaches_open_program_without_model(
    monkeypatch,
) -> None:  # noqa: ANN001
    monkeypatch.setattr(config, "OPEN_VISUAL_PROGRAM_ENABLED", True)
    monkeypatch.setattr(config, "OPEN_VISUAL_PROGRAM_LLM_AUTHORING", False)
    monkeypatch.setattr(config, "OPEN_VISUAL_PROGRAM_CANDIDATES", 3)
    monkeypatch.setattr(config, "OPEN_VISUAL_PROGRAM_AUTHORING_ATTEMPTS", 1)
    monkeypatch.setattr(config, "OPEN_VISUAL_PROGRAM_MIN_SCORE", 0.78)
    plan, reserves, report = _compile_open_visual_specs(
        [
            {
                "visual_id": "visual_001",
                "renderer_hint": "remotion",
                "duration": 4.8,
                "visual_explanation_ir": _compression_ir(),
            }
        ],
        [],
        provider_name="gemini",
        model_name="test-model",
        width=1920,
        height=1080,
        fps=60,
    )

    assert not reserves
    assert report["compiled_count"] == 1
    assert report["model_authored_count"] == 0
    assert plan[0]["open_visual_program"]["concept"]["medium"] == "spatial_metaphor"
    assert len(plan[0]["open_visual_program_candidates"]) == 3


def test_model_planning_budget_enforces_call_limit_and_timeout_override(
    monkeypatch,
) -> None:  # noqa: ANN001
    calls: list[dict[str, object]] = []
    events: list[dict[str, object]] = []

    def fake_reasoning(*_args, **kwargs) -> str:  # noqa: ANN003
        calls.append(dict(kwargs))
        return "{}"

    monkeypatch.setattr(auto_visuals, "call_reasoning_model", fake_reasoning)
    budget = _ModelPlanningBudget(
        max_calls=1,
        call_timeout_sec=17,
        total_timeout_sec=60,
        on_update=events.append,
    )
    reasoning = budget.caller("test_stage")

    assert reasoning("gemini", "test-model", "system", "user") == "{}"
    with pytest.raises(_ModelPlanningBudgetExhausted):
        reasoning("gemini", "test-model", "system", "user")

    assert calls == [{"max_attempts": 1, "timeout_sec": pytest.approx(17)}]
    assert [event["event"] for event in events] == [
        "started",
        "completed",
        "skipped",
    ]
    assert budget.snapshot()["exhausted_reason"] == "model_call_budget_exhausted"


def test_open_visual_authoring_models_only_high_value_primaries(
    monkeypatch,
) -> None:  # noqa: ANN001
    calls: list[dict[str, object]] = []
    progress: list[dict[str, object]] = []

    def fake_reasoning(*_args, **kwargs) -> str:  # noqa: ANN003
        calls.append(dict(kwargs))
        return "{}"

    monkeypatch.setattr(auto_visuals, "call_reasoning_model", fake_reasoning)
    monkeypatch.setattr(config, "OPEN_VISUAL_PROGRAM_ENABLED", True)
    monkeypatch.setattr(config, "OPEN_VISUAL_PROGRAM_LLM_AUTHORING", True)
    monkeypatch.setattr(config, "OPEN_VISUAL_PROGRAM_CANDIDATES", 3)
    monkeypatch.setattr(config, "OPEN_VISUAL_PROGRAM_AUTHORING_ATTEMPTS", 1)
    monkeypatch.setattr(config, "OPEN_VISUAL_PROGRAM_MIN_SCORE", 0.78)
    monkeypatch.setattr(config, "AUTO_VISUALS_MODEL_PRIMARY_AUTHORING_LIMIT", 1)
    plan, reserves, report = _compile_open_visual_specs(
        [
            {
                "visual_id": "visual_low",
                "renderer_hint": "remotion",
                "duration": 4.8,
                "confidence": 0.5,
                "visual_explanation_ir": _compression_ir(),
            },
            {
                "visual_id": "visual_high",
                "renderer_hint": "remotion",
                "duration": 4.8,
                "confidence": 0.95,
                "visual_explanation_ir": _compression_ir(),
            },
        ],
        [
            {
                "visual_id": "reserve_001",
                "renderer_hint": "remotion",
                "duration": 4.8,
                "confidence": 1.0,
                "visual_explanation_ir": _compression_ir(),
            }
        ],
        provider_name="gemini",
        model_name="test-model",
        width=1920,
        height=1080,
        fps=60,
        reasoning_budget=_ModelPlanningBudget(
            max_calls=6,
            call_timeout_sec=20,
            total_timeout_sec=90,
        ),
        progress_callback=progress.append,
    )

    assert len(plan) == 2
    assert len(reserves) == 1
    assert len(calls) == 2
    assert report["primary_model_authoring_selected_count"] == 1
    assert report["reserve_model_authoring_count"] == 0
    assert report["model_planning"]["calls_started"] == 2
    assert [item["role"] for item in progress] == [
        "primary",
        "primary",
        "reserve",
    ]
    assert [item["model_authoring"] for item in progress] == [
        False,
        True,
        False,
    ]
