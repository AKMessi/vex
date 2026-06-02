from __future__ import annotations

from tools.creative_intelligence import (
    annotate_visual_cards_with_graph,
    build_color_grade_quality_contract,
    build_video_understanding_graph,
    candidate_graph_signals,
    graph_to_video_context,
)
from tools.auto_shorts import _apply_creative_graph_to_candidates


def _segments() -> list[dict[str, float | str]]:
    return [
        {
            "start": 0.0,
            "end": 3.0,
            "text": "Wait, everyone is making the same AI agent mistake.",
        },
        {
            "start": 3.2,
            "end": 8.0,
            "text": "The problem is they build tools before they define the system and feedback loop.",
        },
        {
            "start": 8.4,
            "end": 15.0,
            "text": "The proof is simple: teams with a clear evaluation pipeline improve quality every week.",
        },
        {
            "start": 15.4,
            "end": 21.0,
            "text": "So the framework is plan, test, measure, and only then automate the workflow.",
        },
    ]


def test_video_understanding_graph_exports_existing_video_context_shape() -> None:
    graph = build_video_understanding_graph(
        transcript_text=" ".join(str(segment["text"]) for segment in _segments()),
        segments=_segments(),
        metadata={"duration_sec": 21.0, "width": 1920, "height": 1080, "fps": 30.0},
        scene_cuts=[3.1, 8.2, 15.2],
        source_context={"feature": "test"},
    )

    context = graph_to_video_context(graph)

    assert graph.version == "video-understanding-graph-v1"
    assert graph.retention_moments
    assert context["keyword_weights"]
    assert context["quality_contract"]["quality_tier"] == "world_class_local"
    assert "framework" in context["main_keywords"]


def test_candidate_graph_signals_reward_complete_retention_arc() -> None:
    graph = build_video_understanding_graph(
        transcript_text=" ".join(str(segment["text"]) for segment in _segments()),
        segments=_segments(),
        metadata={"duration_sec": 21.0},
    )

    complete = candidate_graph_signals(
        graph,
        start=0.0,
        end=21.0,
        text=" ".join(str(segment["text"]) for segment in _segments()),
        target_platform="youtube_shorts",
    )
    fragment = candidate_graph_signals(
        graph,
        start=3.2,
        end=8.0,
        text=str(_segments()[1]["text"]),
        target_platform="youtube_shorts",
    )

    assert complete["graph_retention_score"] > fragment["graph_retention_score"]
    assert complete["graph_payoff_energy"] > 0.0
    assert complete["graph_visual_opportunity"] > 0.0


def test_auto_shorts_candidates_receive_creative_graph_scores() -> None:
    graph = build_video_understanding_graph(
        transcript_text=" ".join(str(segment["text"]) for segment in _segments()),
        segments=_segments(),
        metadata={"duration_sec": 21.0},
    )
    candidates = [
        {
            "candidate_id": "cand_01",
            "start": 0.0,
            "end": 21.0,
            "duration": 21.0,
            "excerpt": " ".join(str(segment["text"]) for segment in _segments()),
            "heuristic_score": 50.0,
            "score_breakdown": {},
            "selection_reasons": [],
        }
    ]

    _apply_creative_graph_to_candidates(candidates, graph, target_platform="youtube_shorts")

    candidate = candidates[0]
    assert candidate["creative_graph_signals"]["graph_beat_ids"]
    assert candidate["score_breakdown"]["creative_graph_retention_score"] > 0.0
    assert "creative_graph_retention_score" in candidate["score_breakdown"]


def test_visual_cards_are_prioritized_by_graph_opportunity() -> None:
    graph = build_video_understanding_graph(
        transcript_text=" ".join(str(segment["text"]) for segment in _segments()),
        segments=_segments(),
        metadata={"duration_sec": 21.0},
    )
    cards = [
        {
            "card_id": "card_01",
            "start": 8.4,
            "end": 15.0,
            "sentence_text": "teams with a clear evaluation pipeline improve quality every week",
            "context_text": "proof and workflow",
            "priority": 20.0,
        }
    ]

    annotated = annotate_visual_cards_with_graph(cards, graph)

    assert annotated[0]["priority"] > cards[0]["priority"]
    assert annotated[0]["creative_graph_signals"]["graph_visual_opportunity"] > 0.0


def test_color_grade_quality_contract_tightens_for_strong_stylized_looks() -> None:
    contract = build_color_grade_quality_contract(
        look="cinematic",
        intensity=1.25,
        metadata={"duration_sec": 60.0, "width": 3840, "height": 2160, "fps": 29.97},
    )

    assert contract["contract_id"] == "color-grade-program-v2-local"
    assert contract["metrics"]["avoid_over_saturation"] >= 0.96
    assert "shot_to_shot_grade_continuity" in contract["soft_gates"]
