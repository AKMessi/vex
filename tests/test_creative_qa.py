from __future__ import annotations

from creative_intelligence import build_video_understanding_graph, candidate_graph_signals
from creative_qa import (
    evaluate_color_grade_quality,
    evaluate_short_candidate_quality,
    evaluate_visual_plan_quality,
)


def _graph():
    segments = [
        {"start": 0.0, "end": 3.0, "text": "Wait, this workflow fails for one hidden reason."},
        {"start": 3.2, "end": 8.4, "text": "The system has no measurement pipeline, so quality never improves."},
        {"start": 8.8, "end": 16.0, "text": "The proof is that every weekly test exposes the exact bottleneck."},
        {"start": 16.2, "end": 23.0, "text": "So the fix is a loop: plan, measure, improve, and automate."},
    ]
    return build_video_understanding_graph(
        transcript_text=" ".join(str(segment["text"]) for segment in segments),
        segments=segments,
        metadata={"duration_sec": 23.0, "width": 1920, "height": 1080},
    )


def test_short_candidate_quality_report_records_publishability_signals() -> None:
    graph = _graph()
    candidate = {
        "candidate_id": "cand_01",
        "start": 0.0,
        "end": 23.0,
        "duration": 23.0,
        "excerpt": graph.transcript_excerpt,
        "score_breakdown": {
            "story_completeness": 74,
            "standalone_clarity": 70,
            "hook_strength": 72,
            "payoff": 68,
        },
        "creative_graph_signals": candidate_graph_signals(
            graph,
            start=0.0,
            end=23.0,
            text=graph.transcript_excerpt,
            target_platform="youtube_shorts",
        ),
    }

    report = evaluate_short_candidate_quality(candidate, graph, target_platform="youtube_shorts")

    assert report.score > 0.4
    assert "graph_retention" in report.metrics
    assert report.evidence["graph_beat_ids"]


def test_short_candidate_quality_flags_weak_fragments() -> None:
    graph = _graph()
    candidate = {
        "candidate_id": "cand_bad",
        "start": 3.2,
        "end": 8.4,
        "duration": 5.2,
        "excerpt": "The system has no measurement pipeline.",
        "score_breakdown": {
            "story_completeness": 20,
            "standalone_clarity": 30,
            "hook_strength": 10,
            "payoff": 10,
        },
    }

    report = evaluate_short_candidate_quality(candidate, graph, target_platform="youtube_shorts")

    assert report.passed is False
    assert "short_candidate_too_short_for_standalone_publish" in report.issues


def test_visual_plan_quality_scores_semantic_coverage_and_spacing() -> None:
    graph = _graph()
    first_signals = candidate_graph_signals(
        graph,
        start=3.2,
        end=8.4,
        text="measurement pipeline quality",
    )
    second_signals = candidate_graph_signals(
        graph,
        start=16.2,
        end=23.0,
        text="plan measure improve automate loop",
    )
    plan = [
        {
            "visual_id": "visual_001",
            "card_id": "card_001",
            "start": 3.2,
            "end": 6.8,
            "renderer_hint": "hyperframes",
            "creative_graph_signals": first_signals,
        },
        {
            "visual_id": "visual_002",
            "card_id": "card_002",
            "start": 16.2,
            "end": 20.0,
            "renderer_hint": "hyperframes",
            "creative_graph_signals": second_signals,
        },
    ]

    report = evaluate_visual_plan_quality(plan, graph, max_visuals=4)

    assert report.score > 0.3
    assert report.metrics["spacing_score"] == 1.0
    assert report.evidence["visual_ids"] == ["visual_001", "visual_002"]


def test_color_grade_quality_combines_validation_and_warning_penalty() -> None:
    plan = {
        "resolved_look": "natural",
        "filter_graph": "eq=contrast=1.02",
        "adjustments": {
            "average_selected_score": 0.91,
            "real_preview_fraction": 1.0,
            "correction_strength": 0.24,
        },
        "manifest": {"mode": "shot_aware", "shot_count": 2},
        "validation": {
            "passed": True,
            "score": 0.94,
            "warnings": [],
            "shot_validation": {"passed": True, "score": 0.92},
        },
        "warnings": [],
    }

    report = evaluate_color_grade_quality(plan)

    assert report.passed is True
    assert report.score > 0.75
    assert report.metrics["real_preview_fraction"] == 1.0
