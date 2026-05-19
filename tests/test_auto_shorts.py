from __future__ import annotations

import json
from pathlib import Path

import tools.auto_shorts as auto_shorts
from state import ProjectState, utc_now_iso


def test_auto_shorts_ranks_hook_payoff_window_above_generic_transcript() -> None:
    segments = [
        {"start": 0.0, "end": 6.0, "text": "Today I want to talk about the updates from this week."},
        {"start": 6.1, "end": 13.0, "text": "The team met and we reviewed the roadmap and the normal process."},
        {"start": 13.2, "end": 22.0, "text": "There are several things happening and we will continue improving them."},
        {
            "start": 30.0,
            "end": 37.0,
            "text": "Wait, the biggest mistake founders make is chasing AI agents before the workflow is clear.",
        },
        {
            "start": 37.2,
            "end": 44.0,
            "text": "Because 80 percent of the value comes from removing the broken step first.",
        },
        {
            "start": 44.3,
            "end": 53.0,
            "text": "That is the simple system that makes the result feel automatic.",
        },
    ]

    candidates = auto_shorts._build_candidates(segments, 18.0, 30.0, limit=8, target_platform="youtube_shorts")

    assert candidates
    top = candidates[0]
    assert top["start"] >= 30.0
    assert top["score_breakdown"]["hook_strength"] >= 70
    assert top["score_breakdown"]["payoff"] >= 60
    assert "concrete numbers increase credibility" in top["selection_reasons"]


def test_fallback_selections_prefer_topic_diversity() -> None:
    candidates = [
        _candidate("cand_01", 0.0, 25.0, 92.0, ["ai", "agents", "workflow"]),
        _candidate("cand_02", 40.0, 65.0, 90.0, ["ai", "agents", "workflow"]),
        _candidate("cand_03", 80.0, 105.0, 84.0, ["pricing", "founders", "revenue"]),
    ]

    selections = auto_shorts._fallback_selections(candidates, count=2)

    assert [selection["candidate_id"] for selection in selections] == ["cand_01", "cand_03"]


def test_llm_selection_backfills_missing_diverse_picks(monkeypatch) -> None:  # noqa: ANN001
    candidates = [
        _candidate("cand_01", 0.0, 25.0, 92.0, ["ai", "agents", "workflow"]),
        _candidate("cand_02", 40.0, 65.0, 90.0, ["ai", "agents", "workflow"]),
        _candidate("cand_03", 80.0, 105.0, 84.0, ["pricing", "founders", "revenue"]),
    ]
    monkeypatch.setattr(
        auto_shorts,
        "_call_reasoning_model",
        lambda *_args, **_kwargs: json.dumps(
            [
                {
                    "candidate_id": "cand_02",
                    "score": 96,
                    "title": "AI Workflow Mistake",
                    "hook": "The AI agent mistake founders keep making",
                    "reason": "Strong hook.",
                    "keywords": ["ai", "agents"],
                }
            ]
        ),
    )

    selections = auto_shorts._select_shorts_with_llm(
        provider_name="gemini",
        model_name="test",
        candidates=candidates,
        transcript_text="test transcript",
        count=2,
        min_duration_sec=20.0,
        max_duration_sec=45.0,
        target_platform="youtube_shorts",
    )

    assert [selection["candidate_id"] for selection in selections] == ["cand_02", "cand_03"]


def test_execute_passes_subtitle_style_and_records_candidate_breakdown(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    state = _state(tmp_path)
    transcript_text = (
        "Wait, the biggest mistake founders make is chasing AI agents before the workflow is clear. "
        "Because 80 percent of the value comes from removing the broken step first."
    )
    (tmp_path / "transcript.txt").write_text(transcript_text, encoding="utf-8")
    (tmp_path / "transcript.srt").write_text(
        "\n".join(
            [
                "1",
                "00:00:00,000 --> 00:00:10,000",
                "Wait, the biggest mistake founders make is chasing AI agents before the workflow is clear.",
                "",
                "2",
                "00:00:10,200 --> 00:00:22,000",
                "Because 80 percent of the value comes from removing the broken step first.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    source_clip = tmp_path / "source_clip.mp4"
    source_clip.write_bytes(b"raw")
    vertical_clip = tmp_path / "vertical.mp4"
    vertical_clip.write_bytes(b"vertical")
    render_calls: list[dict[str, object]] = []

    monkeypatch.setattr(auto_shorts, "_select_shorts_with_llm", lambda **kwargs: auto_shorts._fallback_selections(kwargs["candidates"], 1))
    monkeypatch.setattr(auto_shorts, "_analyze_viral_score_with_llm", lambda **_kwargs: {"viral_score": {"overall": 88, "hook_strength": 90, "payoff": 86, "novelty": 80, "clarity": 84, "shareability": 82}, "viral_explanation": ["Strong hook."]})
    monkeypatch.setattr(auto_shorts, "_analyze_b_roll_with_llm", lambda **_kwargs: [])
    monkeypatch.setattr(auto_shorts, "_analyze_punch_in_with_llm", lambda **_kwargs: [])
    monkeypatch.setattr(auto_shorts, "trim", lambda *_args, **_kwargs: str(source_clip))
    monkeypatch.setattr(auto_shorts, "apply_center_punch_ins", lambda input_path, *_args, **_kwargs: input_path)
    monkeypatch.setattr(auto_shorts, "render_vertical_short", lambda input_path, working_dir, **kwargs: render_calls.append({"input_path": input_path, "working_dir": working_dir, **kwargs}) or str(vertical_clip))
    monkeypatch.setattr(auto_shorts, "probe_video", lambda _path: {"width": 1080, "height": 1920})

    result = auto_shorts.execute(
        {
            "count": 1,
            "min_duration_sec": 12,
            "max_duration_sec": 30,
            "include_compilation": False,
            "subtitle_style": "glass",
        },
        state,
    )

    assert result["success"] is True
    assert render_calls[0]["subtitle_style"] == "glass"
    manifest_path = Path(state.artifacts["latest_auto_shorts"]["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["subtitle_style"] == "glass"
    assert manifest["shorts"][0]["score_breakdown"]["hook_strength"] >= 70


def _candidate(candidate_id: str, start: float, end: float, score: float, keywords: list[str]) -> dict:
    return {
        "candidate_id": candidate_id,
        "start": start,
        "end": end,
        "duration": end - start,
        "excerpt": " ".join(keywords),
        "heuristic_score": score,
        "score_breakdown": {
            "hook_strength": score,
            "payoff": score,
            "novelty": score,
            "clarity": score,
            "shareability": score,
            "words_per_sec": 3.0,
        },
        "selection_reasons": ["test candidate"],
        "keywords": keywords,
    }


def _state(tmp_path: Path) -> ProjectState:
    now = utc_now_iso()
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    return ProjectState(
        project_id="auto-shorts-test",
        project_name="Auto Shorts Test",
        created_at=now,
        updated_at=now,
        source_files=[str(source)],
        working_file=str(source),
        working_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        metadata={"duration_sec": 30.0, "width": 1920, "height": 1080, "fps": 30.0},
        provider="test",
        model="test-model",
    )
