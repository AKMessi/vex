from __future__ import annotations

import json
from pathlib import Path

from vex_hyperframes.compiler import compile_hyperframes_plan
from vex_hyperframes.visual_world import (
    build_video_design_bible,
    validate_visual_world_program,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "hyperframes_semantic_cases.json"


def test_compiler_builds_distinct_signed_visual_worlds_for_proof_candidates() -> None:
    plan = compile_hyperframes_plan(_spec(_case("causal_passive_learning")))
    programs = plan.renderer_spec["visual_proof_programs"]

    assert plan.passed is True, plan.issues
    assert len(programs) == 4
    assert len(
        {item["visual_world_program"]["medium_family"] for item in programs}
    ) == 4
    assert all(
        validate_visual_world_program(
            item["visual_world_program"],
            scene_program=item["scene_program_v2"],
        ).passed
        for item in programs
    )


def test_visual_world_avoids_recent_medium_and_background() -> None:
    history = [
        {
            "medium_family": "spatial_metaphor",
            "background_mode": "spatial_horizon",
        },
        {
            "medium_family": "data_sculpture",
            "background_mode": "radial_data_field",
        },
    ]
    plan = compile_hyperframes_plan(
        {
            **_spec(_case("data_threshold_latency")),
            "visual_world_history": history,
        }
    )
    world = plan.renderer_spec["visual_world_program"]

    assert world["medium_family"] not in {"spatial_metaphor", "data_sculpture"}
    assert world["background_mode"] not in {
        "spatial_horizon",
        "radial_data_field",
    }


def test_directed_visual_brief_biases_primary_medium_without_flattening_tournament() -> None:
    plan = compile_hyperframes_plan(
        {
            **_spec(_case("quote_exact_language")),
            "directed_visual_brief": {
                "version": "directed-hyperframes-visual-v1",
                "idea": "Make the quote feel like luminous data particles.",
                "grounding_policy": "transcript_evidence_only",
                "preferred_medium_family": "data_sculpture",
            },
        }
    )
    worlds = [
        item["visual_world_program"]
        for item in plan.renderer_spec["visual_proof_programs"]
    ]
    directions = [
        item["creative_direction_program"]
        for item in plan.renderer_spec["visual_proof_programs"]
    ]

    assert plan.passed is True, plan.issues
    assert worlds[0]["medium_family"] == "data_sculpture"
    assert all(item["signature"] for item in directions)
    assert [item["medium_family"] for item in directions] == [
        item["medium_family"] for item in worlds
    ]
    assert len({item["medium_family"] for item in worlds}) > 1


def test_visual_world_signature_rejects_cosmetic_tampering() -> None:
    plan = compile_hyperframes_plan(_spec(_case("process_support_handoff")))
    proof = plan.renderer_spec["visual_proof_programs"][0]
    world = dict(proof["visual_world_program"])
    world["medium_family"] = (
        "product_interface"
        if world["medium_family"] != "product_interface"
        else "kinetic_typography"
    )

    validation = validate_visual_world_program(
        world,
        scene_program=proof["scene_program_v2"],
    )

    assert validation.passed is False
    assert "visual_world_signature_mismatch" in validation.errors


def test_non_interface_worlds_forbid_card_dominated_compositions() -> None:
    plan = compile_hyperframes_plan(_spec(_case("quote_exact_language")))
    worlds = [
        item["visual_world_program"]
        for item in plan.renderer_spec["visual_proof_programs"]
    ]

    assert any(item["card_policy"] == "forbidden" for item in worlds)
    assert all(
        float(item["fingerprint"]["panel_ratio_target"]) <= 0.08
        for item in worlds
        if item["card_policy"] == "forbidden"
    )


def test_video_design_bible_is_deterministic_and_palette_diverse() -> None:
    specs = [
        {
            "visual_id": "visual_001",
            "episode_id": "episode_01",
            "concept_ids": ["compression"],
        },
        {
            "visual_id": "visual_002",
            "episode_id": "episode_02",
            "concept_ids": ["routing"],
        },
    ]

    first = build_video_design_bible(specs)
    second = build_video_design_bible(specs)

    assert first == second
    assert len(first.palette_sequence) >= 3
    assert len(
        {item["background"] for item in first.palette_sequence}
    ) == len(first.palette_sequence)
    assert "same_medium_in_previous_two_visuals" in first.forbidden_repetition


def _case(case_id: str) -> dict:
    return next(
        item
        for item in json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        if item["case_id"] == case_id
    )


def _spec(case: dict) -> dict:
    semantic_frame = dict(case.get("semantic_frame") or {})
    required = list(case.get("required_labels") or [])
    if case.get("expected_scene_type") in {"guided_process", "architecture_flow"}:
        semantic_frame["steps"] = required
    if case.get("expected_scene_type") == "grounded_interface_walkthrough":
        semantic_frame["action"] = semantic_frame.get("action") or required[-2]
        semantic_frame["result"] = semantic_frame.get("result") or required[-1]
    return {
        "visual_id": case["case_id"],
        "sentence_text": case["transcript"],
        "context_text": case["context"],
        "semantic_frame": semantic_frame,
        "metric_facts": case.get("metric_facts") or [],
        "required_labels": required,
        "duration": 4.0,
        "composition_mode": "replace",
    }
