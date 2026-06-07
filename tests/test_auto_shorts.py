from __future__ import annotations

import json
from pathlib import Path

from shorts import build_shorts_program, validate_short_edit_plan, validate_shorts_program
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


def test_video_context_penalizes_spicy_but_off_topic_windows() -> None:
    segments = [
        {
            "start": 0.0,
            "end": 6.0,
            "text": "This video is about building reliable AI agents for customer support workflows.",
        },
        {
            "start": 6.2,
            "end": 13.0,
            "text": "The core problem is teams automate before mapping escalation and handoff steps.",
        },
        {
            "start": 13.2,
            "end": 21.0,
            "text": "A practical support agent needs clear inputs, approvals, and fallback rules.",
        },
        {
            "start": 40.0,
            "end": 48.0,
            "text": "Wait, this insane Bitcoin mistake made a million dollars overnight and nobody saw the secret coming.",
        },
        {
            "start": 60.0,
            "end": 68.0,
            "text": "The best AI agent starts with the support workflow because clean handoffs prevent broken automation.",
        },
        {
            "start": 68.2,
            "end": 76.0,
            "text": "That is the system that makes the customer experience reliable instead of chaotic.",
        },
    ]
    transcript = " ".join(str(segment["text"]) for segment in segments)
    context = auto_shorts._build_video_context(transcript, segments)

    candidates = auto_shorts._build_candidates(
        segments,
        8.0,
        18.0,
        limit=10,
        target_platform="youtube_shorts",
        video_context=context,
    )

    bitcoin_candidate = next(candidate for candidate in candidates if "Bitcoin" in candidate["excerpt"])
    top = candidates[0]
    assert "support workflow" in top["excerpt"]
    assert top["score_breakdown"]["context_score"] > bitcoin_candidate["score_breakdown"]["context_score"]
    assert bitcoin_candidate["score_breakdown"]["misleading_clip_penalty"] > top["score_breakdown"]["misleading_clip_penalty"]


def test_context_scoring_penalizes_abrupt_context_dependent_starts() -> None:
    transcript = (
        "This video explains pricing strategy for early products. "
        "The pricing mistake is discounting before the customer understands the outcome."
    )
    context = auto_shorts._build_video_context(
        transcript,
        [
            {"start": 0.0, "end": 8.0, "text": "This video explains pricing strategy for early products."},
            {
                "start": 8.2,
                "end": 18.0,
                "text": "The pricing mistake is discounting before the customer understands the outcome.",
            },
        ],
    )

    abrupt_score, abrupt_breakdown, _abrupt_reasons = auto_shorts._score_transcript_window(
        "And that is why it breaks before they understand the outcome.",
        8.0,
        video_context=context,
    )
    standalone_score, standalone_breakdown, _standalone_reasons = auto_shorts._score_transcript_window(
        "The pricing mistake is discounting before the customer understands the outcome.",
        8.0,
        video_context=context,
    )

    assert standalone_score > abrupt_score
    assert abrupt_breakdown["abrupt_start_penalty"] > standalone_breakdown["abrupt_start_penalty"]
    assert standalone_breakdown["standalone_clarity"] > abrupt_breakdown["standalone_clarity"]


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


def test_shorts_director_builds_typed_program_and_portfolio() -> None:
    segments = [
        {"start": 0.0, "end": 7.0, "text": "Wait, the pricing mistake is discounting before value is clear."},
        {"start": 7.2, "end": 15.0, "text": "Because customers buy the outcome, not the discount."},
        {"start": 24.0, "end": 32.0, "text": "The system is to prove the result first, then talk about price."},
        {"start": 48.0, "end": 58.0, "text": "Finally, the takeaway is simple: sell the outcome before the number."},
    ]
    transcript = " ".join(str(segment["text"]) for segment in segments)
    context = auto_shorts._build_video_context(transcript, segments)
    candidates = auto_shorts._build_candidates(
        segments,
        12.0,
        30.0,
        limit=8,
        target_platform="youtube_shorts",
        video_context=context,
    )

    program = build_shorts_program(
        transcript_text=transcript,
        segments=segments,
        candidates=candidates,
        selections=[],
        requested_count=2,
        target_platform="youtube_shorts",
        min_duration_sec=12.0,
        max_duration_sec=30.0,
        video_context=context,
    )
    validation = validate_shorts_program(program)

    assert validation["passed"] is True
    assert program.moments
    assert program.candidates
    assert program.portfolio.selected_candidate_ids
    edit_plan = program.edit_plans[program.portfolio.selected_candidate_ids[0]]
    assert edit_plan.punch_in_policy["max_moments"] >= 1
    assert edit_plan.source_ranges
    assert edit_plan.operations
    assert validate_short_edit_plan(edit_plan)["passed"] is True
    assert program.to_dict()["version"] == "shorts-director-v3"


def test_director_scores_are_attached_to_candidates() -> None:
    segments = [
        {"start": 0.0, "end": 8.0, "text": "Wait, this is the biggest AI workflow mistake."},
        {"start": 8.2, "end": 18.0, "text": "Because teams automate before the handoff is clear."},
        {"start": 18.2, "end": 28.0, "text": "The fix is mapping approvals before building the agent."},
    ]
    transcript = " ".join(str(segment["text"]) for segment in segments)
    context = auto_shorts._build_video_context(transcript, segments)
    candidates = auto_shorts._build_candidates(
        segments,
        12.0,
        30.0,
        limit=8,
        target_platform="youtube_shorts",
        video_context=context,
    )
    program = build_shorts_program(
        transcript_text=transcript,
        segments=segments,
        candidates=candidates,
        selections=[],
        requested_count=1,
        target_platform="youtube_shorts",
        min_duration_sec=12.0,
        max_duration_sec=30.0,
        video_context=context,
    )

    auto_shorts._apply_shorts_program_to_candidates(candidates, program)

    assert candidates[0]["director_plan"]["program_score"] >= 1
    assert "director_score" in candidates[0]["score_breakdown"]
    assert candidates[0]["score_breakdown"]["primary_role"] in {"hook", "proof", "tension", "payoff", "quote", "setup", "support"}


def test_remix_candidates_stitch_separate_story_beats() -> None:
    segments = [
        {"start": 0.0, "end": 5.0, "text": "Wait, the biggest pricing mistake is discounting before value is clear."},
        {"start": 24.0, "end": 31.0, "text": "Because 80 percent of the trust comes from proving the outcome first."},
        {"start": 62.0, "end": 69.0, "text": "The fix is a simple system: prove the result, then talk about price."},
        {"start": 90.0, "end": 96.0, "text": "That takeaway makes the buying decision feel obvious instead of risky."},
    ]
    transcript = " ".join(str(segment["text"]) for segment in segments)
    context = auto_shorts._build_video_context(transcript, segments)

    remixes = auto_shorts._build_remix_candidates(
        [],
        segments,
        min_duration_sec=14.0,
        max_duration_sec=24.0,
        target_platform="youtube_shorts",
        video_context=context,
        limit=4,
    )

    assert remixes
    remix = remixes[0]
    assert remix["composition_mode"] == "remix"
    assert len(remix["source_ranges"]) >= 2
    assert 14.0 <= remix["duration"] <= 24.0
    assert "director edit graph" in remix["selection_reasons"][0]
    assert remix["edit_plan_seed"]["strategy"] == "graph_beam_search"
    assert remix["edit_plan_seed"]["operations"]
    assert all(source_range.get("reason") for source_range in remix["source_ranges"])


def test_edit_plan_validation_rejects_unsafe_source_maps() -> None:
    invalid_plan = {
        "candidate_id": "bad_01",
        "target_duration_sec": 8.0,
        "source_ranges": [
            {"index": 1, "start": 5.0, "end": 4.0, "duration": -1.0, "role": "hook", "speed": 1.0},
            {"index": 1, "start": 7.0, "end": 9.0, "duration": 2.0, "role": "unsupported", "speed": 4.0},
        ],
        "operations": [
            {"type": "raw_ffmpeg", "source_range_index": 9, "start_sec": 0.0, "end_sec": 1.0},
        ],
        "remix_policy": {"max_source_ranges": 4},
    }

    validation = validate_short_edit_plan(invalid_plan)

    assert validation["passed"] is False
    assert any("unsupported type" in error for error in validation["errors"])
    assert any("unsupported role" in error for error in validation["errors"])
    assert any("duplicates index" in error for error in validation["errors"])


def test_preflight_rejects_abrupt_stitched_edit_plan() -> None:
    candidate = {
        "candidate_id": "remix_bad",
        "start": 5.0,
        "end": 20.0,
        "duration": 15.0,
        "excerpt": "And this depends on what came before. Then another fragment continues.",
        "source_ranges": [
            {"index": 1, "start": 5.0, "end": 10.0, "duration": 5.0, "role": "support"},
            {"index": 2, "start": 30.0, "end": 40.0, "duration": 10.0, "role": "support"},
        ],
    }
    edit_plan = {
        "candidate_id": "remix_bad",
        "target_duration_sec": 15.0,
        "quality_floor": 58.0,
        "source_ranges": candidate["source_ranges"],
        "operations": [{"type": "caption_emphasis", "start_sec": 0.0, "end_sec": 15.0}],
        "remix_policy": {"max_source_ranges": 4},
    }
    preflight = auto_shorts._preflight_short_edit_plan(
        candidate=candidate,
        selection={"candidate_id": "remix_bad", "title": "Bad Remix", "hook": "And this"},
        clip_segments=[
            {"start": 0.0, "end": 5.0, "text": "And this depends on what came before."},
            {"start": 5.0, "end": 15.0, "text": "Then another fragment continues."},
        ],
        edit_plan=edit_plan,
        target_platform="youtube_shorts",
        video_context={"core_keywords": ["pricing"], "main_keywords": ["pricing"], "thesis_excerpt": "Pricing strategy."},
    )

    assert preflight["passed"] is False
    assert any("opens without hook" in error for error in preflight["errors"])


def test_stitched_transcript_segments_are_remapped_to_output_timeline() -> None:
    transcript_segments = [
        {"start": 0.0, "end": 2.0, "text": "Hook line."},
        {"start": 10.0, "end": 12.0, "text": "Payoff line."},
    ]
    stitched = auto_shorts._clip_transcript_segments_for_ranges(
        transcript_segments,
        [
            {"index": 1, "start": 0.0, "end": 2.0, "role": "hook"},
            {"index": 2, "start": 10.0, "end": 12.0, "role": "payoff"},
        ],
    )

    assert stitched == [
        {
            "start": 0.0,
            "end": 2.0,
            "text": "Hook line.",
            "source_start": 0.0,
            "source_end": 2.0,
            "source_range_index": 1,
            "source_role": "hook",
        },
        {
            "start": 2.0,
            "end": 4.0,
            "text": "Payoff line.",
            "source_start": 10.0,
            "source_end": 12.0,
            "source_range_index": 2,
            "source_role": "payoff",
        },
    ]


def test_fallback_quality_gate_rejects_abrupt_fragments() -> None:
    gate = auto_shorts._fallback_short_quality_gate(
        {
            "candidate_id": "remix_01",
            "start": 4.0,
            "end": 12.0,
            "duration": 8.0,
            "excerpt": "And this is why it happened with that thing from earlier.",
            "source_ranges": [{"index": 1, "start": 4.0, "end": 12.0, "role": "support"}],
        },
        {"title": "Abrupt Fragment", "hook": "And this is why"},
        [{"start": 0.0, "end": 8.0, "text": "And this is why it happened with that thing from earlier."}],
        {
            "duration": 8.0,
            "source_ranges": [{"index": 1, "start": 4.0, "end": 12.0, "role": "support"}],
            "target_platform": "youtube_shorts",
        },
        {"core_keywords": ["pricing", "value"], "main_keywords": ["pricing", "value"], "thesis_excerpt": "Pricing strategy."},
    )

    assert gate["passed"] is False
    assert gate["verdict"] == "rejected"
    assert gate["rejection_reason"]


def test_short_edit_plan_normalization_coerces_string_levels() -> None:
    plan = {
        "candidate_id": "cand_01",
        "target_duration_sec": "8",
        "quality_floor": "high",
        "source_ranges": [
            {"index": "1", "start": "0", "end": "8", "duration": "8", "role": "hook", "speed": "high"},
        ],
        "operations": [
            {
                "type": "punch_in",
                "source_range_index": "1",
                "start_sec": "0.5",
                "end_sec": "1.5",
                "params": {"zoom": "high", "intensity": "high"},
            },
            {
                "type": "auto_visual",
                "source_range_index": "1",
                "start_sec": "2",
                "end_sec": "3",
                "params": {"strength": "high", "preferred_types": ["ui_card"]},
            },
        ],
        "remix_policy": {"max_source_ranges": "high"},
        "punch_in_policy": {"enabled": True, "max_moments": "high", "min_gap_sec": "medium", "max_zoom": "high"},
        "visual_insert_policy": {"enabled": True, "max_inserts": "high"},
    }

    normalized = auto_shorts._normalize_short_edit_plan_fields(plan, {"duration": 8.0})

    assert normalized["quality_floor"] == 64.0
    assert normalized["source_ranges"][0]["speed"] == 1.15
    assert normalized["punch_in_policy"]["max_moments"] == 8
    assert normalized["punch_in_policy"]["max_zoom"] == 1.18
    assert normalized["visual_insert_policy"]["max_inserts"] == 8
    assert normalized["operations"][0]["params"]["zoom"] == 1.18
    assert normalized["operations"][0]["params"]["intensity"] == 0.8
    assert normalized["operations"][1]["params"]["strength"] == 0.8
    assert validate_short_edit_plan(normalized)["passed"] is True


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
    monkeypatch.setattr(auto_shorts, "_quality_gate_with_llm", lambda **_kwargs: {"passed": True, "score": 90, "verdict": "approved", "abruptness": 92, "standalone": 90, "payoff": 88, "topic_fit": 86, "stitch_continuity": 100, "rejection_reason": "", "reasons": ["clean short"], "model_used": "test"})
    monkeypatch.setattr(auto_shorts, "trim", lambda *_args, **_kwargs: str(source_clip))
    monkeypatch.setattr(auto_shorts, "merge", lambda *_args, **_kwargs: str(source_clip))
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
    assert manifest["video_context"]["main_keywords"]
    assert manifest["shorts_program"]["version"] == "shorts-director-v3"
    assert manifest["program_validation"]["passed"] is True
    assert manifest["shorts"][0]["director_plan"]["program_score"] >= 1
    assert manifest["shorts"][0]["edit_plan"]["framing_mode"] in {"center_stage_safe", "stitched_center_stage_safe"}
    assert manifest["shorts"][0]["edit_plan"]["source_ranges"]
    assert manifest["shorts"][0]["edit_plan"]["operations"]
    assert manifest["shorts"][0]["preflight"]["passed"] is True
    assert manifest["shorts"][0]["render_validation"]["passed"] is True
    assert manifest["shorts"][0]["quality_gate"]["passed"] is True
    assert manifest["shorts"][0]["source_ranges"]
    assert manifest["shorts"][0]["score_breakdown"]["hook_strength"] >= 70
    assert "context_score" in manifest["shorts"][0]["score_breakdown"]
    assert manifest["rendered_count"] == 1
    assert manifest["accepted_count"] == 1
    assert manifest["rejected_count"] == 0
    assert manifest["rejected_shorts"] == []
    assert Path(manifest["drafts_dir"]).is_dir()
    assert Path(manifest["accepted_dir"]).is_dir()
    assert Path(manifest["rejected_dir"]).is_dir()
    assert Path(manifest["shorts"][0]["draft_video_path"]).parent.parent.name == "drafts"
    assert Path(manifest["shorts"][0]["vertical_video_path"]).parent.name == "accepted"
    assert Path(manifest["shorts"][0]["vertical_video_path"]).exists()
    assert state.artifacts["latest_auto_shorts"]["accepted_count"] == 1
    assert state.artifacts["latest_auto_shorts"]["rejected_count"] == 0


def test_execute_reports_rendered_rejected_shorts_separately(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
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

    monkeypatch.setattr(auto_shorts, "_select_shorts_with_llm", lambda **kwargs: auto_shorts._fallback_selections(kwargs["candidates"], 1))
    monkeypatch.setattr(auto_shorts, "_analyze_viral_score_with_llm", lambda **_kwargs: {"viral_score": {"overall": 88, "hook_strength": 90, "payoff": 86, "novelty": 80, "clarity": 84, "shareability": 82}, "viral_explanation": ["Strong hook."]})
    monkeypatch.setattr(auto_shorts, "_analyze_b_roll_with_llm", lambda **_kwargs: [])
    monkeypatch.setattr(auto_shorts, "_analyze_punch_in_with_llm", lambda **_kwargs: [])
    monkeypatch.setattr(
        auto_shorts,
        "_quality_gate_with_llm",
        lambda **_kwargs: {
            "passed": False,
            "score": 42,
            "verdict": "rejected",
            "abruptness": 40,
            "standalone": 35,
            "payoff": 45,
            "topic_fit": 60,
            "stitch_continuity": 55,
            "rejection_reason": "clip needs prior context",
            "reasons": ["context dependent"],
            "model_used": "test",
        },
    )
    monkeypatch.setattr(auto_shorts, "trim", lambda *_args, **_kwargs: str(source_clip))
    monkeypatch.setattr(auto_shorts, "merge", lambda *_args, **_kwargs: str(source_clip))
    monkeypatch.setattr(auto_shorts, "apply_center_punch_ins", lambda input_path, *_args, **_kwargs: input_path)
    monkeypatch.setattr(auto_shorts, "render_vertical_short", lambda *_args, **_kwargs: str(vertical_clip))
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

    assert result["success"] is False
    assert "Rendered 1 draft(s), accepted 0, rejected 1" in result["message"]
    assert "Rejected files:" in result["message"]
    assert "Drafts:" in result["message"]
    manifest_path = Path(state.artifacts["latest_auto_shorts"]["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["shorts"] == []
    assert manifest["rendered_count"] == 1
    assert manifest["accepted_count"] == 0
    assert manifest["rejected_count"] == 1
    assert Path(manifest["drafts_dir"]).is_dir()
    assert Path(manifest["rejected_dir"]).is_dir()
    rejected = manifest["rejected_shorts"][0]
    assert rejected["stage"] == "post_render_qa"
    assert rejected["quality_gate"]["rejection_reason"] == "clip needs prior context"
    assert "quality gate: clip needs prior context" in rejected["rejection_reasons"]
    assert Path(rejected["rejected_video_path"]).parent.name == "rejected"
    assert Path(rejected["rejected_video_path"]).exists()
    assert Path(rejected["rejected_record_path"]).exists()


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
