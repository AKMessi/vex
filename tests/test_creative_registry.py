from __future__ import annotations

from tools.creative_registry import (
    REGISTRY_VERSION,
    latest_creative_runs,
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
