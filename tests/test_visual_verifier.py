from __future__ import annotations

from pathlib import Path

from tests.test_visual_communication_contract import _ir
from vex_visuals.communication_contract import build_communication_contract
from vex_visuals.verifier import (
    VisualCandidateEvidence,
    VisualQualityState,
    evaluate_verifier_payload,
    reset_verifier_circuits,
    run_pairwise_visual_tournament,
    run_visual_verifier,
)


def _payload(contract) -> dict:  # noqa: ANN001
    answers = {}
    for question in contract.questions:
        if question.proposition_id in {"proposition_01", "proposition_02"}:
            answers[question.question_id] = "Four tokens compress into one KV summary"
        elif question.proposition_id == "proposition_03":
            answers[question.question_id] = "The indexer ranks compressed blocks and selects the strongest"
        elif question.proposition_id == "relation_01":
            answers[question.question_id] = "The token group transforms into a compact entry"
        else:
            answers[question.question_id] = "The compact entry enables the indexer to pick top blocks"
    return {
        "thesis": "Compressed attention condenses token groups before ranking blocks",
        "answers": answers,
        "sequence": [
            "four tokens compress into a KV summary",
            "the compact block is ranked by an indexer",
        ],
        "unsupported_claims": [],
        "design": {
            "hierarchy": 0.9,
            "composition": 0.88,
            "typography": 0.86,
            "density": 0.82,
            "polish": 0.9,
            "originality": 0.84,
            "issues": [],
        },
        "temporal": {
            "causal_readability": 0.92,
            "meaningful_motion": 0.9,
            "smoothness": 0.87,
            "sequence_legibility": 0.9,
            "settling": 0.88,
            "issues": [],
        },
        "technical_defects": [],
    }


def _frame(tmp_path: Path, name: str = "frame.png") -> Path:
    path = tmp_path / name
    path.write_bytes(b"valid-test-frame")
    return path


def test_multidimensional_verifier_marks_semantic_design_and_motion_pass_as_verified() -> None:
    contract = build_communication_contract(_ir())
    report = evaluate_verifier_payload(_payload(contract), contract, provider="test", model="vision")

    assert report.state == VisualQualityState.VERIFIED
    assert report.publishable
    assert report.verified
    assert report.semantic.passed
    assert report.design.passed
    assert report.temporal.passed


def test_unsupported_claim_is_a_hard_semantic_rejection() -> None:
    contract = build_communication_contract(_ir())
    payload = _payload(contract)
    payload["unsupported_claims"] = ["Compression saves exactly 90 percent of compute"]

    report = evaluate_verifier_payload(payload, contract)

    assert report.state == VisualQualityState.REJECTED
    assert not report.publishable
    assert "viewer_found_unsupported_claims" in report.issues


def test_provider_outage_is_explicitly_unverified_or_degraded_never_verified(tmp_path: Path) -> None:
    reset_verifier_circuits()
    contract = build_communication_contract(_ir())

    def unavailable(*_args):  # noqa: ANN002, ANN202
        raise TimeoutError("vision provider overloaded")

    strict = run_visual_verifier(
        [_frame(tmp_path)],
        contract,
        provider_models=[("test", "vision")],
        request=unavailable,
        strict=True,
        local_gate_passed=True,
        local_score=0.95,
    )
    reset_verifier_circuits()
    balanced = run_visual_verifier(
        [_frame(tmp_path)],
        contract,
        provider_models=[("test", "vision")],
        request=unavailable,
        strict=False,
        local_gate_passed=True,
        local_score=0.95,
    )

    assert strict.state == VisualQualityState.UNVERIFIED
    assert not strict.publishable
    assert balanced.state == VisualQualityState.DEGRADED
    assert balanced.publishable
    assert not balanced.verified


def test_verifier_cache_is_content_addressed_and_avoids_duplicate_calls(tmp_path: Path) -> None:
    reset_verifier_circuits()
    contract = build_communication_contract(_ir())
    calls = 0

    def request(*_args):  # noqa: ANN002, ANN202
        nonlocal calls
        calls += 1
        return _payload(contract)

    kwargs = {
        "provider_models": [("test", "vision")],
        "request": request,
        "cache_dir": tmp_path / "cache",
    }
    first = run_visual_verifier([_frame(tmp_path)], contract, **kwargs)
    second = run_visual_verifier([_frame(tmp_path)], contract, **kwargs)

    assert first.verified
    assert second.verified
    assert second.cache_hit
    assert calls == 1


def test_bidirectional_pairwise_judge_corrects_for_candidate_order(tmp_path: Path) -> None:
    reset_verifier_circuits()
    contract = build_communication_contract(_ir())
    verified = evaluate_verifier_payload(_payload(contract), contract)
    first = VisualCandidateEvidence("first", [_frame(tmp_path, "a.png")], verified, 0.82)
    second = VisualCandidateEvidence("second", [_frame(tmp_path, "b.png")], verified, 0.8)
    calls = 0

    def prefer_second(_provider, _model, _prompt, _frames):  # noqa: ANN001, ANN202
        nonlocal calls
        calls += 1
        # Candidate second is B in the first ordering and A in the reversed ordering.
        return {
            "winner": "B" if calls == 1 else "A",
            "confidence": 0.91,
            "reasons": ["Cleaner proof-bearing transformation"],
        }

    tournament = run_pairwise_visual_tournament(
        [first, second],
        contract,
        provider_models=[("test", "vision")],
        request=prefer_second,
    )

    assert tournament.selection_mode == "bidirectional_pairwise"
    assert tournament.selected_candidate_id == "second"
    assert tournament.comparisons[0].order_consistent


def test_position_inconsistent_pairwise_result_falls_back_to_verified_scores(tmp_path: Path) -> None:
    reset_verifier_circuits()
    contract = build_communication_contract(_ir())
    high_payload = _payload(contract)
    low_payload = _payload(contract)
    low_payload["design"]["polish"] = 0.7
    first = VisualCandidateEvidence(
        "first",
        [_frame(tmp_path, "a.png")],
        evaluate_verifier_payload(high_payload, contract),
        0.88,
    )
    second = VisualCandidateEvidence(
        "second",
        [_frame(tmp_path, "b.png")],
        evaluate_verifier_payload(low_payload, contract),
        0.75,
    )

    def always_first(_provider, _model, _prompt, _frames):  # noqa: ANN001, ANN202
        return {"winner": "A", "confidence": 0.8, "reasons": []}

    tournament = run_pairwise_visual_tournament(
        [first, second],
        contract,
        provider_models=[("test", "vision")],
        request=always_first,
    )

    assert tournament.selection_mode == "verified_score_fallback"
    assert tournament.selected_candidate_id == "first"
    assert not tournament.comparisons[0].order_consistent
