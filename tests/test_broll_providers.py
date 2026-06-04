from __future__ import annotations

from typing import Any

import config
import broll_intelligence
from broll_intelligence import StockBrollCandidate
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
