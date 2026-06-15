from __future__ import annotations

from tools.creative_optimizer import optimize_creative_set


def test_optimizer_selects_quality_over_input_order() -> None:
    selected, report = optimize_creative_set(
        [
            _candidate("early", start=1.0, score=58.0),
            _candidate("strong", start=8.0, score=94.0),
        ],
        budget=1,
    )

    assert [item["visual_id"] for item in selected] == ["strong"]
    assert report["rejected"][0]["reason"] == "set_budget_outcompeted"


def test_optimizer_resolves_overlap_in_favor_of_stronger_render() -> None:
    selected, report = optimize_creative_set(
        [
            _candidate("early", start=1.0, score=72.0, rendered_score=0.61),
            _candidate("strong", start=2.0, score=86.0, rendered_score=0.94),
        ],
        budget=2,
        phase="rendered",
    )

    assert [item["visual_id"] for item in selected] == ["strong"]
    assert report["rejected"][0]["reason"] == "timing_conflict_with_stronger_set"
    assert report["rejected"][0]["conflicting_visual_ids"] == ["strong"]


def test_optimizer_rewards_semantic_coverage_over_redundant_candidates() -> None:
    selected, report = optimize_creative_set(
        [
            _candidate(
                "pipeline_a",
                start=1.0,
                score=91.0,
                beat_ids=["beat_pipeline"],
                concept_ids=["concept_pipeline"],
            ),
            _candidate(
                "pipeline_b",
                start=8.0,
                score=90.0,
                beat_ids=["beat_pipeline"],
                concept_ids=["concept_pipeline"],
            ),
            _candidate(
                "feedback",
                start=15.0,
                score=84.0,
                beat_ids=["beat_feedback"],
                concept_ids=["concept_feedback"],
                intent_type="contrast",
                template="problem_solution",
            ),
        ],
        budget=2,
    )

    assert {item["visual_id"] for item in selected} == {"pipeline_a", "feedback"}
    assert report["metrics"]["beat_coverage"] == 1.0
    assert report["metrics"]["concept_coverage"] == 1.0


def test_optimizer_is_deterministic_for_equal_candidates() -> None:
    candidates = [
        _candidate("later", start=8.0, score=80.0),
        _candidate("earlier", start=1.0, score=80.0),
    ]

    first, first_report = optimize_creative_set(candidates, budget=1)
    second, second_report = optimize_creative_set(candidates, budget=1)

    assert [item["visual_id"] for item in first] == ["earlier"]
    assert first == second
    assert first_report == second_report


def test_optimizer_handles_duplicate_planner_ids_without_dropping_candidates() -> None:
    selected, report = optimize_creative_set(
        [
            _candidate("visual_001", start=1.0, score=78.0),
            _candidate("visual_001", start=8.0, score=88.0),
        ],
        budget=2,
    )

    assert len(selected) == 2
    assert [item["candidate_id"] for item in report["selected"]] == [
        "visual_001",
        "visual_001#2",
    ]


def test_optimizer_rewards_distinct_visual_worlds_over_cosmetic_repetition() -> None:
    selected, report = optimize_creative_set(
        [
            _candidate(
                "same_world_a",
                start=1.0,
                score=91.0,
                medium_family="data_sculpture",
                background_mode="radial_data_field",
            ),
            _candidate(
                "same_world_b",
                start=8.0,
                score=90.0,
                medium_family="data_sculpture",
                background_mode="radial_data_field",
            ),
            _candidate(
                "different_world",
                start=15.0,
                score=85.0,
                medium_family="editorial_collage",
                background_mode="paper_registration",
            ),
        ],
        budget=2,
    )

    assert {item["visual_id"] for item in selected} == {
        "same_world_a",
        "different_world",
    }
    assert report["metrics"]["unique_media"] == 2
    assert report["metrics"]["unique_backgrounds"] == 2


def _candidate(
    visual_id: str,
    *,
    start: float,
    score: float,
    rendered_score: float = 0.8,
    beat_ids: list[str] | None = None,
    concept_ids: list[str] | None = None,
    intent_type: str = "mechanism",
    template: str = "mechanism_blueprint",
    medium_family: str = "",
    background_mode: str = "",
) -> dict[str, object]:
    return {
        "visual_id": visual_id,
        "start": start,
        "end": start + 3.0,
        "template": template,
        "renderer_hint": "hyperframes",
        "visual_intent_type": intent_type,
        "confidence": 0.9,
        "concept_ids": concept_ids or [],
        "creative_graph_signals": {
            "graph_visual_opportunity": 0.8,
            "graph_retention_score": 0.76,
            "graph_topic_alignment": 0.82,
            "graph_beat_ids": beat_ids or [],
        },
        "auto_visuals_director": {
            "director_score": score,
            "copy_alignment": 0.84,
        },
        "rendered_visual_qa": {
            "passed": True,
            "score": rendered_score,
        },
        "visual_world_program": (
            {
                "medium_family": medium_family,
                "canvas_system": "paper_canvas",
                "background_mode": background_mode,
                "motion_choreography": "assemble_and_mask",
                "fingerprint": {
                    "signature": f"{medium_family}:{background_mode}",
                    "panel_ratio_target": 0.04,
                },
            }
            if medium_family
            else {}
        ),
    }
