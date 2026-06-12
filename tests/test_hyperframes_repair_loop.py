from __future__ import annotations

from vex_hyperframes.final_judge import judge_final_candidate
from vex_hyperframes.repair_loop import assess_monotonic_improvement
from vex_hyperframes.variants import select_best_variant


def test_monotonic_repair_accepts_reduced_hard_failures() -> None:
    before = _record(
        "before",
        quality=0.72,
        semantic=0.68,
        critic=0.6,
        hard=["missing_relation", "ambiguous_thesis"],
    )
    after = _record(
        "after",
        quality=0.75,
        semantic=0.76,
        critic=0.78,
        hard=["ambiguous_thesis"],
    )

    decision = assess_monotonic_improvement(
        before,
        after,
        min_score_delta=0.025,
    )

    assert decision.accepted is True
    assert decision.reason == "hard_failures_reduced"
    assert decision.hard_failure_delta == 1


def test_monotonic_repair_rejects_quality_regression() -> None:
    before = _record(
        "before",
        quality=0.9,
        semantic=0.8,
        critic=0.78,
        hard=["missing_relation"],
    )
    after = _record(
        "after",
        quality=0.7,
        semantic=0.84,
        critic=0.86,
        hard=[],
    )

    decision = assess_monotonic_improvement(
        before,
        after,
        min_score_delta=0.025,
    )

    assert decision.accepted is False
    assert decision.reason == "render_quality_regressed"


def test_selection_excludes_non_monotonic_repair() -> None:
    selected = select_best_variant(
        [
            {
                "variant_id": "rejected_repair",
                "asset_path": "repair.mp4",
                "eligible_for_selection": False,
                "qa": {"passed": True, "score": 0.98},
            },
            {
                "variant_id": "accepted",
                "asset_path": "accepted.mp4",
                "qa": {"passed": True, "score": 0.84},
            },
        ]
    )

    assert selected is not None
    assert selected["variant_id"] == "accepted"


def test_final_judge_uses_artifact_gate_without_vision_key(
    monkeypatch,
) -> None:
    import config

    monkeypatch.setattr(config, "GEMINI_API_KEY", None)
    verdict = judge_final_candidate(
        [],
        production_contract={"takeaway": "A request reaches review."},
        scene_program={"program_signature": "signed"},
        quality_report={"passed": True, "score": 0.91},
        critic_bundle={
            "passed": True,
            "score": 0.88,
            "hard_failure_count": 0,
        },
        qa_mode="hybrid",
    )

    assert verdict.available is False
    assert verdict.local_gate_passed is True
    assert verdict.passed is True
    assert verdict.score == 0.88


def test_final_judge_strict_mode_rejects_missing_vision(
    monkeypatch,
) -> None:
    import config

    monkeypatch.setattr(config, "GEMINI_API_KEY", None)
    verdict = judge_final_candidate(
        [],
        production_contract={},
        scene_program={"program_signature": "signed"},
        quality_report={"passed": True, "score": 0.9},
        critic_bundle={
            "passed": True,
            "score": 0.9,
            "hard_failure_count": 0,
        },
        qa_mode="vision",
    )

    assert verdict.passed is False
    assert "independent_vision_judge_unavailable" in verdict.issues


def _record(
    variant_id: str,
    *,
    quality: float,
    semantic: float,
    critic: float,
    hard: list[str],
) -> dict:
    return {
        "variant_id": variant_id,
        "qa": {"score": quality, "passed": False},
        "metadata": {
            "stage": {
                "object_coverage": 1.0,
                "relation_coverage": 1.0,
            },
            "semantic_qa": {
                "score": semantic,
                "hard_failures": [],
                "object_coverage": 1.0,
            },
            "visual_critics": {
                "score": critic,
                "counterexamples": [
                    {
                        "issue_type": issue,
                        "severity": "hard_failure",
                        "element_ids": [],
                        "relation_ids": [],
                    }
                    for issue in hard
                ],
            },
        },
    }
