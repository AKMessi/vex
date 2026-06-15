from __future__ import annotations

from visual_opportunity import (
    VISUAL_OPPORTUNITY_PLAN_VERSION,
    build_semantic_episodes,
    build_visual_opportunity_plan,
)


def test_semantic_episodes_use_discourse_boundaries_not_fixed_time_buckets() -> None:
    cards = [
        _card(1, 0.0, "Attention is quadratic.", priority=70.0),
        _card(
            2,
            3.0,
            "Every token compares with every other token.",
            priority=78.0,
            process=0.4,
        ),
        _card(3, 8.0, "Now let's look at compressed sparse attention.", priority=62.0),
        _card(
            4,
            11.0,
            "First it compresses four tokens into one entry.",
            priority=88.0,
            process=0.8,
        ),
        _card(
            5,
            15.0,
            "Then the indexer selects the top tokens.",
            priority=90.0,
            process=0.9,
        ),
    ]

    episodes = build_semantic_episodes(cards, clip_duration=22.0)

    assert len(episodes) == 2
    assert episodes[1].boundary_reason == "discourse_transition"
    assert episodes[1].card_ids == [
        "visual_card_003",
        "visual_card_004",
        "visual_card_005",
    ]


def test_long_form_planner_selects_complete_partition_mechanism() -> None:
    cards = [
        _card(
            1,
            0.0,
            "Not from the marketing view, but the actual architectural view.",
            priority=76.0,
            contrast=0.4,
        ),
        _card(
            2,
            4.0,
            "Because there is a specific thing in the architecture.",
            priority=74.0,
        ),
        _card(
            3,
            14.0,
            "Now let's inspect the compression mechanism.",
            priority=66.0,
        ),
        _card(
            4,
            18.0,
            "There are 32 original tokens in context.",
            priority=91.0,
            numeric=1,
            metric_facts=[{"value": "32", "label": "32 original tokens"}],
        ),
        _card(
            5,
            22.0,
            "After compression, 4 is to 1 and 8 compressed blocks remain.",
            priority=96.0,
            numeric=2,
            process=0.8,
            metric_facts=[
                {"value": "4", "label": "Four tokens per compressed block"},
                {"value": "8", "label": "8 compressed blocks remain"},
            ],
        ),
        _card(
            6,
            26.0,
            "Each compressed block summarizes four original tokens.",
            priority=92.0,
            process=0.7,
        ),
    ]

    plan = build_visual_opportunity_plan(
        cards,
        clip_duration=34.0,
        requested_count=4,
    )

    assert plan.version == VISUAL_OPPORTUNITY_PLAN_VERSION
    assert plan.selected
    selected = plan.selected[0]
    assert selected.scene_type == "set_partition"
    assert selected.preflight["passed"] is True
    assert selected.source_card_ids[:2] == [
        "visual_card_004",
        "visual_card_005",
    ]
    assert "32 original tokens" in selected.card["sentence_text"]
    assert "8 compressed blocks" in selected.card["sentence_text"]
    assert all(
        "marketing view" not in item.card.get("sentence_text", "").lower()
        for item in plan.selected
    )


def test_planner_covers_strong_early_middle_and_late_semantic_episodes() -> None:
    cards = [
        _card(
            1,
            5.0,
            "First the request enters the API.",
            priority=91.0,
            process=0.8,
        ),
        _card(
            2,
            8.0,
            "Then authentication passes it to the planner.",
            priority=94.0,
            process=0.9,
        ),
        _card(
            3,
            11.0,
            "Finally the renderer returns the response.",
            priority=92.0,
            process=0.9,
        ),
        _card(
            4,
            150.0,
            "Now the support agent classifies the request.",
            priority=90.0,
            process=0.8,
        ),
        _card(
            5,
            153.0,
            "Then it checks policy before routing uncertainty.",
            priority=95.0,
            process=0.9,
        ),
        _card(
            6,
            156.0,
            "Finally uncertain cases reach a human.",
            priority=93.0,
            process=0.9,
        ),
        _card(
            7,
            310.0,
            "Now the training pipeline creates separate domain experts.",
            priority=92.0,
            process=0.8,
        ),
        _card(
            8,
            313.0,
            "Then the router selects the right expert for each task.",
            priority=97.0,
            process=0.9,
        ),
        _card(
            9,
            316.0,
            "Finally the selected expert generates the response.",
            priority=94.0,
            process=0.9,
        ),
    ]

    plan = build_visual_opportunity_plan(
        cards,
        clip_duration=330.0,
        requested_count=8,
    )

    starts = [item.start for item in plan.selected]
    assert any(start < 30 for start in starts)
    assert any(120 < start < 220 for start in starts)
    assert any(start > 280 for start in starts)
    assert len({item.episode_id for item in plan.selected}) == 3


def test_planner_keeps_reserves_and_honors_failure_memory() -> None:
    cards = [
        _card(
            1,
            5.0,
            "First the request enters the API.",
            priority=92.0,
            process=0.8,
        ),
        _card(
            2,
            8.0,
            "Then authentication passes it to the planner.",
            priority=96.0,
            process=0.9,
        ),
        _card(
            3,
            11.0,
            "Finally the renderer returns the response.",
            priority=94.0,
            process=0.9,
        ),
        _card(
            4,
            70.0,
            "Now the agent classifies the support request.",
            priority=90.0,
            process=0.8,
        ),
        _card(
            5,
            73.0,
            "Then it checks policy and routes uncertainty.",
            priority=95.0,
            process=0.9,
        ),
        _card(
            6,
            76.0,
            "Finally a human handles the uncertain case.",
            priority=93.0,
            process=0.9,
        ),
    ]
    first = build_visual_opportunity_plan(
        cards,
        clip_duration=90.0,
        requested_count=4,
    )
    assert first.selected
    blocked = first.selected[0]

    second = build_visual_opportunity_plan(
        cards,
        clip_duration=90.0,
        requested_count=4,
        blocked_card_ids={
            blocked.opportunity_id,
            *blocked.source_card_ids,
        },
    )

    assert all(
        item.opportunity_id != blocked.opportunity_id
        for item in second.selected + second.reserves
    )
    assert any(
        item.reason == "blocked_by_prior_failure_or_usage"
        for item in second.rejected
    )


def _card(
    index: int,
    start: float,
    text: str,
    *,
    priority: float,
    numeric: int = 0,
    process: float = 0.0,
    contrast: float = 0.0,
    metric_facts: list[dict[str, str]] | None = None,
) -> dict:
    return {
        "card_id": f"visual_card_{index:03d}",
        "start": start,
        "end": start + 2.8,
        "sentence_text": text,
        "source_sentence_text": text,
        "context_text": text,
        "previous_text": "",
        "next_text": "",
        "keywords": [],
        "metric_facts": metric_facts or [],
        "visual_type_hint": "process" if process else "data_graphic" if numeric else "cutaway",
        "word_count": len(text.split()),
        "words_per_second": 3.0,
        "pause_before": 0.3,
        "pause_after": 0.3,
        "nearest_scene_cut": None,
        "scene_distance": 2.0,
        "sentence_numeric_hits": numeric,
        "numeric_hits": numeric,
        "sentence_process_cues": process,
        "process_cues": process,
        "sentence_contrast_cues": contrast,
        "contrast_cues": contrast,
        "generic_penalty": 0.05,
        "concrete_hits": 2.0,
        "proper_nouns": 1,
        "replace_safety": 0.8,
        "visualizability": min(priority / 100.0, 0.98),
        "semantic_frame": {},
        "intuition_mode": "process_route" if process else "metric_proof" if numeric else "concept_emphasis",
        "intuition_role": "core_mechanism",
        "intuition_payoff": min(priority / 100.0, 0.98),
        "novelty_key": f"card-{index}",
        "suggested_composition": "replace",
        "style_pack": "signal_lab",
        "suggested_renderer": "hyperframes",
        "priority": priority,
        "source_frame_analysis": {
            "visual_need": 0.8,
            "source_richness": 0.2,
        },
        "creative_graph_signals": {
            "graph_visual_opportunity": 0.85,
        },
    }
