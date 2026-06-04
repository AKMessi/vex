from __future__ import annotations

from typing import Any

import config
import broll_intelligence
from broll_intelligence import StockBrollCandidate
from tools.creative_intelligence import build_video_understanding_graph
from tools import pexels_broll


def _candidate(provider: str, title: str, tags: list[str]) -> dict[str, Any]:
    return StockBrollCandidate(
        provider=provider,
        provider_display_name=provider.title(),
        provider_id=f"{provider}-1",
        title=title,
        description=" ".join(tags),
        tags=tags,
        duration=8.0,
        source_url=f"https://example.com/{provider}/{title.replace(' ', '-')}",
        download_url=f"https://cdn.example.com/{provider}.mp4",
        preview_url=f"https://cdn.example.com/{provider}.jpg",
        creator_name="Creator",
        creator_url="https://example.com/creator",
        license_name=f"{provider.title()} License",
        license_url="https://example.com/license",
        attribution_required=True,
        width=1920,
        height=1080,
        quality="hd",
    ).to_dict()


def test_stock_provider_selection_uses_configured_keys(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(config, "PEXELS_API_KEY", "pexels-key")
    monkeypatch.setattr(config, "PIXABAY_API_KEY", "")
    monkeypatch.setattr(config, "COVERR_API_KEY", "coverr-key")
    monkeypatch.setattr(config, "AUTO_BROLL_PROVIDERS", "auto")

    assert broll_intelligence.configured_stock_provider_names() == ["pexels", "coverr"]
    assert broll_intelligence.configured_stock_provider_names("coverr,pixabay") == ["coverr"]
    assert broll_intelligence.missing_stock_provider_keys("coverr,pixabay") == ["PIXABAY_API_KEY"]


def test_pixabay_search_normalizes_video_candidate(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(config, "PIXABAY_API_KEY", "pixabay-key")
    seen: dict[str, str] = {}

    def fake_pixabay_get_json(url: str) -> tuple[dict, dict[str, str]]:
        seen["url"] = url
        return (
            {
                "hits": [
                    {
                        "id": 125,
                        "pageURL": "https://pixabay.com/videos/id-125/",
                        "tags": "neural network, data, technology",
                        "duration": 12,
                        "videos": {
                            "medium": {
                                "url": "https://cdn.pixabay.com/video/125_medium.mp4",
                                "width": 1920,
                                "height": 1080,
                                "size": 3_562_083,
                                "thumbnail": "https://cdn.pixabay.com/video/125_medium.jpg",
                            }
                        },
                        "user_id": 1281706,
                        "user": "Coverr-Free-Footage",
                    }
                ]
            },
            {"remaining": "99"},
        )

    monkeypatch.setattr(broll_intelligence, "pixabay_get_json", fake_pixabay_get_json)

    candidates, headers = broll_intelligence.search_stock_provider(
        "pixabay",
        "neural network",
        "landscape",
        1920,
        1080,
        per_page=6,
    )

    assert headers == {"remaining": "99"}
    assert "safesearch=true" in seen["url"]
    assert candidates[0]["provider"] == "pixabay"
    assert candidates[0]["download_url"] == "https://cdn.pixabay.com/video/125_medium.mp4"
    assert candidates[0]["creator_url"].startswith("https://pixabay.com/users/")
    assert candidates[0]["file_info"]["quality"] == "medium"


def test_coverr_search_normalizes_video_candidate(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(config, "COVERR_API_KEY", "coverr-key")

    def fake_coverr_get_json(_url: str) -> tuple[dict, dict[str, str]]:
        return (
            {
                "hits": [
                    {
                        "id": "S1YbPl1NfI",
                        "title": "Developer Working On A Product Dashboard",
                        "description": "A focused developer works with software on screen",
                        "is_vertical": False,
                        "tags": ["developer", "software", "dashboard"],
                        "duration": 11.625,
                        "max_width": 2048,
                        "max_height": 1152,
                        "thumbnail": "https://storage.coverr.co/t/example",
                        "urls": {
                            "mp4_download": "https://storage.coverr.co/videos/example/download?token=abc",
                        },
                    }
                ]
            },
            {"remaining": "49"},
        )

    monkeypatch.setattr(broll_intelligence, "coverr_get_json", fake_coverr_get_json)

    candidates, headers = broll_intelligence.search_stock_provider(
        "coverr",
        "software dashboard",
        "landscape",
        1920,
        1080,
        per_page=6,
    )

    assert headers == {"remaining": "49"}
    assert candidates[0]["provider"] == "coverr"
    assert candidates[0]["source_url"] == "https://coverr.co/videos/S1YbPl1NfI"
    assert candidates[0]["download_url"].startswith("https://storage.coverr.co/videos/")
    assert candidates[0]["file_info"]["quality"] == "curated"


def test_collect_search_candidates_ranks_across_configured_providers(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(config, "PIXABAY_API_KEY", "pixabay-key")
    monkeypatch.setattr(config, "COVERR_API_KEY", "coverr-key")
    calls: list[tuple[str, str]] = []

    def fake_search_stock_provider(
        provider_name: str,
        query: str,
        _target_orientation: str,
        _target_width: int,
        _target_height: int,
        per_page: int = 6,
    ) -> tuple[list[dict], dict[str, str]]:
        calls.append((provider_name, query))
        if provider_name == "pixabay":
            return [_candidate("pixabay", "Neural Network Data Flow", ["neural", "network", "data"])], {"remaining": "10"}
        return [_candidate("coverr", "Team Meeting", ["business", "people", "office"])], {"remaining": "40"}

    monkeypatch.setattr(broll_intelligence, "search_stock_provider", fake_search_stock_provider)
    plan_item = {
        "start": 1.0,
        "end": 3.0,
        "subtitle_text": "The neural network routes data through attention layers",
        "context_text": "Attention layers move data through the model architecture.",
        "primary_query": "neural network data flow",
        "backup_queries": ["AI data architecture"],
        "must_include": ["neural", "data"],
        "avoid": ["office"],
        "visual_type": "abstract_motion",
    }

    candidates, headers = broll_intelligence.collect_search_candidates(
        plan_item,
        "landscape",
        1920,
        1080,
        provider_names=["pixabay", "coverr"],
    )

    assert {provider for provider, _query in calls} == {"pixabay", "coverr"}
    assert headers["pixabay"]["remaining"] == "10"
    assert candidates[0]["provider"] == "pixabay"
    assert candidates[0]["score"] > candidates[-1]["score"]


def test_collect_search_candidates_keeps_legacy_pexels_search_fn() -> None:
    def fake_search_fn(_query: str, orientation: str, per_page: int) -> tuple[list[dict], dict[str, str]]:
        assert orientation == "landscape"
        assert per_page == 6
        return (
            [
                {
                    "id": 42,
                    "url": "https://www.pexels.com/video/software-dashboard-42/",
                    "duration": 9,
                    "image": "https://images.pexels.com/photos/example.jpeg",
                    "user": {"name": "Pexels Creator", "url": "https://www.pexels.com/@creator"},
                    "video_files": [
                        {
                            "id": 7,
                            "file_type": "video/mp4",
                            "width": 1920,
                            "height": 1080,
                            "quality": "hd",
                            "fps": 30,
                            "link": "https://videos.pexels.com/video-files/example.mp4",
                        }
                    ],
                }
            ],
            {"remaining": "199"},
        )

    plan_item = {
        "start": 0.0,
        "end": 2.0,
        "subtitle_text": "The software dashboard reveals the issue",
        "context_text": "The dashboard makes the workflow visible.",
        "primary_query": "software dashboard",
        "backup_queries": [],
        "must_include": ["dashboard"],
        "avoid": [],
        "visual_type": "product_ui",
    }

    candidates, headers = broll_intelligence.collect_search_candidates(
        plan_item,
        "landscape",
        1920,
        1080,
        search_fn=fake_search_fn,
    )

    assert headers["pexels"]["remaining"] == "199"
    assert candidates[0]["provider"] == "pexels"
    assert candidates[0]["provider_id"] == "42"
    assert candidates[0]["file_info"]["link"].endswith(".mp4")


def test_auto_broll_reports_missing_all_provider_keys(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(config, "PEXELS_API_KEY", "")
    monkeypatch.setattr(config, "PIXABAY_API_KEY", "")
    monkeypatch.setattr(config, "COVERR_API_KEY", "")
    monkeypatch.setattr(config, "AUTO_BROLL_PROVIDERS", "auto")

    result = pexels_broll.execute({}, object())  # type: ignore[arg-type]

    assert result["success"] is False
    assert "PEXELS_API_KEY" in result["message"]
    assert "PIXABAY_API_KEY" in result["message"]
    assert "COVERR_API_KEY" in result["message"]


def test_broll_director_builds_typed_graph_aware_intents(monkeypatch) -> None:  # noqa: ANN001
    segments = [
        {"start": 0.0, "end": 2.0, "text": "The neural network routes data through attention layers."},
        {"start": 2.4, "end": 5.2, "text": "The proof is the dashboard shows quality improving every week."},
        {"start": 5.8, "end": 8.0, "text": "So the process is measure, fix, and automate the workflow."},
    ]
    graph = build_video_understanding_graph(
        transcript_text=" ".join(str(segment["text"]) for segment in segments),
        segments=segments,
        metadata={"duration_sec": 8.0, "width": 1920, "height": 1080},
    )
    cards = broll_intelligence.build_context_cards(segments, 8.0)

    def fake_llm_plan(**_kwargs: Any) -> list[dict]:
        return [
            {
                "card_id": cards[0]["card_id"],
                "start": cards[0]["start"],
                "end": max(float(cards[0]["end"]), float(cards[0]["start"]) + 1.2),
                "subtitle_text": cards[0]["subtitle_text"],
                "context_text": cards[0]["context_text"],
                "keywords": cards[0]["keywords"],
                "visual_type": "abstract_motion",
                "primary_query": "neural network data flow",
                "backup_queries": ["AI attention architecture"],
                "must_include": ["neural", "data"],
                "avoid": ["random office"],
                "direction": "Use a data-flow cutaway.",
                "rationale": "This beat describes an invisible mechanism.",
                "confidence": 0.9,
            }
        ]

    monkeypatch.setattr(broll_intelligence, "analyze_broll_plan_with_llm", fake_llm_plan)

    director_plan, plan_items = broll_intelligence.build_broll_director_plan(
        cards=cards,
        clip_duration=8.0,
        max_overlays=2,
        min_overlay_sec=1.0,
        max_overlay_sec=3.0,
        orientation="landscape",
        provider_name="gemini",
        model_name="test",
        graph=graph,
    )

    assert director_plan.version == "broll-director-v2"
    assert director_plan.policy["uses_creative_graph"] is True
    assert plan_items
    assert plan_items[0]["broll_intent"]["intent_type"] in {"abstract_concept", "data_evidence"}
    assert plan_items[0]["provider_queries"]["pixabay"][0] == "neural network data flow"
    assert plan_items[0]["creative_graph_signals"]["graph_visual_opportunity"] > 0.0


def test_query_variants_prefers_director_provider_queries() -> None:
    plan_item = {
        "primary_query": "generic technology",
        "backup_queries": ["office work"],
        "visual_type": "abstract_concept",
        "provider_queries": {
            "pixabay": ["neural network data tunnel", "abstract AI data"],
            "coverr": ["technology"],
        },
    }

    variants = broll_intelligence.query_variants(plan_item, "pixabay")

    assert variants[:2] == ["neural network data tunnel", "abstract AI data"]
    assert "generic technology" in variants


def test_candidate_visual_verification_rejects_avoid_terms() -> None:
    plan_item = {
        "start": 1.0,
        "end": 3.0,
        "subtitle_text": "The neural network routes data through attention layers",
        "context_text": "A model architecture explanation",
        "primary_query": "neural network data flow",
        "must_include": ["neural", "data"],
        "avoid": ["office"],
    }
    candidate = _candidate("coverr", "Generic Office Meeting", ["office", "people", "business"])

    verification = broll_intelligence.verify_stock_candidate_for_intent(
        plan_item,
        candidate,
        "landscape",
    )

    assert verification.passed is False
    assert "avoid_terms_present" in verification.issues


def test_final_broll_qa_rejects_llm_flagged_abrupt_insert(monkeypatch) -> None:  # noqa: ANN001
    overlays = [
        {
            "card_id": "card_001",
            "start": 1.0,
            "end": 3.0,
            "subtitle_text": "The dashboard proves quality improved.",
            "stock_provider_display_name": "Pixabay",
            "stock_source_url": "https://pixabay.com/videos/id-1/",
            "query_used": "analytics dashboard",
            "selection_reason": "Strong semantic match.",
            "candidate_score": 54.0,
            "visual_verification": {"score": 0.86, "issues": [], "warnings": []},
            "broll_intent": {"intent_type": "data_evidence"},
        },
        {
            "card_id": "card_002",
            "start": 4.2,
            "end": 6.0,
            "subtitle_text": "Then we automate the workflow.",
            "stock_provider_display_name": "Coverr",
            "stock_source_url": "https://coverr.co/videos/abc",
            "query_used": "business work",
            "selection_reason": "Weak fallback.",
            "candidate_score": 45.0,
            "visual_verification": {"score": 0.62, "issues": [], "warnings": ["metadata_reads_generic"]},
            "broll_intent": {"intent_type": "process"},
        },
    ]

    def fake_reasoning(*_args: Any, **_kwargs: Any) -> str:
        return (
            '{"overall_score": 0.77, "decisions": ['
            '{"card_id": "card_001", "decision": "keep", "reason": "Direct evidence match."},'
            '{"card_id": "card_002", "decision": "reject", "reason": "Too generic and abrupt."}'
            "]}"
        )

    monkeypatch.setattr(broll_intelligence, "call_reasoning_model", fake_reasoning)

    approved, report = broll_intelligence.evaluate_broll_final_plan_with_llm(
        provider_name="gemini",
        model_name="test",
        overlays=overlays,
        clip_duration=8.0,
        transcript_excerpt="The dashboard proves quality improved. Then we automate the workflow.",
        director_plan={"policy": {"min_gap_sec": 0.8}},
    )

    assert [item["card_id"] for item in approved] == ["card_001"]
    assert report["mode"] == "deterministic_plus_llm"
    assert report["rejected_count"] == 1
