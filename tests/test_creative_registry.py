from __future__ import annotations

from tools.creative_registry import (
    REGISTRY_VERSION,
    latest_creative_runs,
    load_creative_policy,
    load_creative_registry,
    record_creative_run,
)


def test_record_creative_run_creates_local_registry(tmp_path) -> None:
    result = record_creative_run(
        working_dir=tmp_path,
        feature="auto_shorts",
        manifest_path=str(tmp_path / "manifest.json"),
        output_path=str(tmp_path / "short.mp4"),
        graph_version="video-understanding-graph-v1",
        quality_score=0.81,
        summary={"count": 1},
        artifacts={"bundle_dir": str(tmp_path)},
    )

    registry = load_creative_registry(tmp_path)

    assert result["registered"] is True
    assert registry["version"] == REGISTRY_VERSION
    assert len(registry["runs"]) == 1
    assert registry["runs"][0]["feature"] == "auto_shorts"
    assert registry["runs"][0]["quality_score"] == 0.81


def test_latest_creative_runs_filters_by_feature(tmp_path) -> None:
    record_creative_run(
        working_dir=tmp_path,
        feature="auto_shorts",
        quality_score=0.7,
    )
    record_creative_run(
        working_dir=tmp_path,
        feature="auto_visuals",
        quality_score=0.8,
    )
    record_creative_run(
        working_dir=tmp_path,
        feature="auto_shorts",
        quality_score=0.9,
    )

    latest_shorts = latest_creative_runs(tmp_path, feature="auto_shorts", limit=2)

    assert [item["feature"] for item in latest_shorts] == ["auto_shorts", "auto_shorts"]
    assert latest_shorts[0]["quality_score"] == 0.9
    assert latest_shorts[1]["quality_score"] == 0.7


def test_creative_policy_learns_bounded_renderer_quality_priors(tmp_path) -> None:
    for _ in range(4):
        record_creative_run(
            working_dir=tmp_path,
            feature="auto_visuals",
            summary={
                "outcome_signals": [
                    {
                        "renderer": "hyperframes",
                        "intent_type": "mechanism",
                        "template": "mechanism_blueprint",
                        "qa_score": 0.42,
                        "qa_passed": False,
                    },
                    {
                        "renderer": "ffmpeg",
                        "intent_type": "mechanism",
                        "template": "mechanism_blueprint",
                        "qa_score": 0.91,
                        "qa_passed": True,
                    },
                ]
            },
        )

    policy = load_creative_policy(tmp_path, feature="auto_visuals")

    assert policy.outcome_count == 8
    assert policy.renderer_adjustments["hyperframes"] < 0.0
    assert policy.renderer_adjustments["ffmpeg"] > 0.0
    assert -0.04 <= policy.adjustment_for(
        renderer="hyperframes",
        intent_type="mechanism",
        template="mechanism_blueprint",
    ) < 0.0
    assert 0.0 < policy.adjustment_for(
        renderer="ffmpeg",
        intent_type="mechanism",
        template="mechanism_blueprint",
    ) <= 0.04


def test_creative_policy_requires_repeated_outcomes(tmp_path) -> None:
    record_creative_run(
        working_dir=tmp_path,
        feature="auto_visuals",
        summary={
            "outcome_signals": [
                {
                    "renderer": "hyperframes",
                    "intent_type": "mechanism",
                    "template": "mechanism_blueprint",
                    "qa_score": 0.2,
                    "qa_passed": False,
                }
            ]
        },
    )

    policy = load_creative_policy(tmp_path, feature="auto_visuals")

    assert policy.outcome_count == 1
    assert policy.renderer_adjustments == {}
    assert policy.adjustment_for(renderer="hyperframes") == 0.0
