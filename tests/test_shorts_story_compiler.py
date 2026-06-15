from __future__ import annotations

import json

from shorts import (
    build_semantic_units,
    build_story_chapters,
    compile_story_proposal,
)
import tools.auto_shorts as auto_shorts


def test_semantic_units_preserve_complete_whisper_thoughts() -> None:
    segments = [
        {
            "start": 171.58,
            "end": 176.82,
            "text": "There was a single scalar controlling two things that were not relevant to each other.",
        },
        {
            "start": 176.82,
            "end": 181.66,
            "text": "Deciding how much to erase old associations and how much to write new values.",
        },
        {
            "start": 181.66,
            "end": 185.66,
            "text": "Is it even relatable? I might want to add up more.",
        },
    ]

    units = build_semantic_units(segments)

    assert [unit["text"] for unit in units] == [
        segment["text"] for segment in segments
    ]
    assert all(unit["complete_start"] for unit in units)
    assert all(unit["complete_end"] for unit in units)


def test_story_compiler_rejects_context_dependent_disconnected_stitch() -> None:
    units = build_semantic_units(
        [
            {
                "start": 0.0,
                "end": 8.0,
                "text": "The pricing mistake is discounting before customers understand the result.",
            },
            {
                "start": 8.1,
                "end": 16.0,
                "text": "The fix is proving the outcome before discussing the price.",
            },
            {
                "start": 60.0,
                "end": 68.0,
                "text": "And this changes the matrix update from the previous derivation.",
            },
        ]
    )

    compiled = compile_story_proposal(
        {
            "title": "Broken stitch",
            "confidence": 92,
            "source_ranges": [
                {"role": "hook", "unit_ids": ["unit_0001"]},
                {"role": "payoff", "unit_ids": ["unit_0003"]},
            ],
        },
        units,
        candidate_id="story_001",
        min_duration_sec=12.0,
        max_duration_sec=30.0,
    )

    assert compiled["passed"] is False
    assert any("context-dependent" in error for error in compiled["errors"])
    assert any("causal topic" in error for error in compiled["errors"])


def test_story_compiler_accepts_complete_contiguous_arc() -> None:
    units = build_semantic_units(
        [
            {
                "start": 0.0,
                "end": 8.0,
                "text": "The pricing mistake is discounting before customers understand the outcome.",
            },
            {
                "start": 8.1,
                "end": 16.0,
                "text": "Customers compare the result they want, not the percentage discount.",
            },
            {
                "start": 16.1,
                "end": 24.0,
                "text": "The fix is proving the outcome first and discussing price second.",
            },
        ]
    )

    compiled = compile_story_proposal(
        {
            "title": "Sell the outcome first",
            "hook": "Discounting early is the pricing mistake.",
            "reason": "Complete problem, explanation, and payoff.",
            "confidence": 90,
            "unit_ids": ["unit_0001", "unit_0002", "unit_0003"],
        },
        units,
        candidate_id="story_001",
        min_duration_sec=20.0,
        max_duration_sec=30.0,
    )

    assert compiled["passed"] is True
    assert compiled["candidate"]["composition_mode"] == "single_window"
    assert compiled["candidate"]["source_ranges"][0]["unit_ids"] == [
        "unit_0001",
        "unit_0002",
        "unit_0003",
    ]


def test_story_compiler_rejects_unknown_or_implicit_discontiguous_units() -> None:
    units = build_semantic_units(
        [
            {"start": 0.0, "end": 8.0, "text": "A complete opening explains the problem."},
            {"start": 8.1, "end": 16.0, "text": "The explanation continues with the mechanism."},
            {"start": 16.1, "end": 24.0, "text": "A complete ending provides the final answer."},
        ]
    )

    compiled = compile_story_proposal(
        {
            "unit_ids": ["unit_0001", "unit_0003", "unit_9999"],
            "confidence": 90,
        },
        units,
        candidate_id="story_001",
        min_duration_sec=12.0,
        max_duration_sec=30.0,
    )

    assert compiled["passed"] is False
    assert any("unknown semantic units" in error for error in compiled["errors"])
    assert any("explicit source_ranges" in error for error in compiled["errors"])


def test_hierarchical_planner_compiles_model_unit_ids(monkeypatch) -> None:  # noqa: ANN001
    units = build_semantic_units(
        [
            {
                "start": 0.0,
                "end": 8.0,
                "text": "The biggest agent mistake is automating before mapping the workflow.",
            },
            {
                "start": 8.1,
                "end": 16.0,
                "text": "That creates brittle handoffs because nobody owns the fallback.",
            },
            {
                "start": 16.1,
                "end": 24.0,
                "text": "The fix is defining approvals and escalation before building the agent.",
            },
        ]
    )
    monkeypatch.setattr(
        auto_shorts,
        "_call_reasoning_model",
        lambda *_args, **_kwargs: json.dumps(
            [
                {
                    "title": "Map the workflow first",
                    "hook": "The biggest agent mistake happens before coding.",
                    "reason": "A complete mistake, consequence, and fix.",
                    "confidence": 94,
                    "keywords": ["agents", "workflow", "fallback"],
                    "unit_ids": ["unit_0001", "unit_0002", "unit_0003"],
                }
            ]
        ),
    )

    candidates, provenance = auto_shorts._plan_story_candidates_with_llm(
        provider_name="gemini",
        model_name="test",
        semantic_units=units,
        count=3,
        min_duration_sec=20.0,
        max_duration_sec=30.0,
        target_platform="youtube_shorts",
        video_context={
            "thesis_excerpt": "Reliable AI agents start with workflow design.",
            "core_keywords": ["agents", "workflow"],
            "main_keywords": ["agents", "workflow", "fallback"],
        },
    )

    assert len(build_story_chapters(units)) == 1
    assert candidates[0]["candidate_origin"] == "hierarchical_story_planner"
    assert candidates[0]["story_plan"]["critic"]["passed"] is True
    assert provenance["status"] == "completed"
    assert provenance["accepted_candidate_ids"] == ["story_001"]


def test_hierarchical_planner_bounds_model_calls_for_very_long_videos(monkeypatch) -> None:  # noqa: ANN001
    units = build_semantic_units(
        [
            {
                "start": float(index * 5),
                "end": float(index * 5 + 4),
                "text": f"Complete explanation unit {index} describes the system clearly.",
            }
            for index in range(1000)
        ]
    )
    calls = 0

    def empty_planner(*_args, **_kwargs):  # noqa: ANN002, ANN003
        nonlocal calls
        calls += 1
        return "[]"

    monkeypatch.setattr(auto_shorts, "_call_reasoning_model", empty_planner)

    candidates, provenance = auto_shorts._plan_story_candidates_with_llm(
        provider_name="gemini",
        model_name="test",
        semantic_units=units,
        count=6,
        min_duration_sec=20.0,
        max_duration_sec=45.0,
        target_platform="youtube_shorts",
        video_context={},
    )

    assert candidates == []
    assert calls <= 24
    assert provenance["chapter_count"] <= 24


def test_quality_gate_normalizes_string_reason_as_one_reason() -> None:
    normalized = auto_shorts._normalize_short_quality_gate(
        {
            "passed": False,
            "score": 22,
            "reasons": "The short begins mid-thought.",
            "rejection_reason": "abrupt fragment",
        },
        {
            "passed": False,
            "score": 20,
            "abruptness": 20,
            "standalone": 20,
            "payoff": 20,
            "topic_fit": 50,
            "stitch_continuity": 20,
            "reasons": ["fallback"],
            "rejection_reason": "fallback",
        },
        "test-model",
    )

    assert normalized["reasons"] == ["The short begins mid-thought."]


def test_candidate_tournament_prompt_is_blind_to_internal_scores() -> None:
    candidate = {
        "candidate_id": "story_001",
        "start": 10.0,
        "end": 34.0,
        "duration": 24.0,
        "composition_mode": "story_compilation",
        "source_ranges": [
            {"index": 1, "start": 10.0, "end": 20.0, "role": "hook"},
            {"index": 2, "start": 40.0, "end": 54.0, "role": "payoff"},
        ],
        "excerpt": "The assembled transcript contains the complete viewer-facing story.",
        "heuristic_score": 99.0,
        "score_breakdown": {"hook_strength": 99.0, "payoff": 99.0},
        "selection_reasons": ["internal role labels say this is excellent"],
    }

    formatted = auto_shorts._format_candidates_for_llm([candidate])

    assert "Final assembled transcript" in formatted
    assert "heuristic" not in formatted
    assert "hook_strength" not in formatted
    assert "payoff" not in formatted
    assert "internal role labels" not in formatted
