from __future__ import annotations

from tests.test_visual_communication_contract import _ir
from vex_visuals.generative_authoring import compile_open_visual_program_for_spec
from vex_visuals.portfolio import (
    evaluate_visual_portfolio,
    extract_visual_portfolio_identity,
    same_creative_grammar,
)


def _compiled(visual_id: str, candidate_index: int = 0) -> dict:
    spec, result = compile_open_visual_program_for_spec(
        {"visual_id": visual_id, "renderer_hint": "remotion", "duration": 4.8},
        ir=_ir(),
        width=1920,
        height=1080,
        fps=60,
        enable_model_authoring=False,
        candidate_count=4,
    )
    assert result.passed
    spec["open_visual_program"] = dict(
        spec["open_visual_program_candidates"][candidate_index]
    )
    spec["renderer"] = "remotion"
    return spec


def test_portfolio_identity_recovers_selected_concept_lane_and_motion() -> None:
    identity = extract_visual_portfolio_identity(_compiled("visual_001"))

    assert identity.concept_id
    assert identity.lane != "unknown"
    assert identity.medium != "unknown"
    assert identity.motion_grammar != "unknown"
    assert identity.identity_signature


def test_portfolio_report_rewards_distinct_concept_lanes() -> None:
    diverse = [_compiled(f"visual_{index:03d}", index - 1) for index in range(1, 4)]
    repeated = [_compiled(f"visual_{index:03d}", 0) for index in range(1, 4)]

    diverse_report = evaluate_visual_portfolio(diverse)
    repeated_report = evaluate_visual_portfolio(repeated)

    assert diverse_report.visual_count == 3
    assert diverse_report.lane_entropy > repeated_report.lane_entropy
    assert diverse_report.motion_entropy > repeated_report.motion_entropy
    assert diverse_report.score > repeated_report.score
    assert repeated_report.consecutive_repetitions


def test_same_creative_grammar_requires_full_lane_medium_motion_and_composition_match() -> None:
    first = extract_visual_portfolio_identity(_compiled("visual_001", 0))
    repeated = extract_visual_portfolio_identity(_compiled("visual_002", 0))
    alternate = extract_visual_portfolio_identity(_compiled("visual_003", 1))

    assert same_creative_grammar(first, repeated)
    assert not same_creative_grammar(first, alternate)
