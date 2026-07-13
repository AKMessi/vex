from __future__ import annotations

import json

from tests.test_visual_communication_contract import _ir
from vex_visuals.communication_contract import build_communication_contract
from vex_visuals.concept_search import (
    CONCEPT_LANES,
    apply_concept_to_program,
    author_visual_concepts,
    build_visual_concept_candidates,
    build_visual_reference_board,
    normalize_authored_visual_concepts,
    score_visual_concepts,
    validate_visual_concept,
)
from vex_visuals.open_visual_program import build_open_visual_program_candidates, validate_open_visual_program
from vex_visuals.generative_authoring import compile_open_visual_program_for_spec


def test_concept_swarm_covers_every_proposition_with_distinct_visual_languages() -> None:
    contract = build_communication_contract(_ir())
    concepts = build_visual_concept_candidates({}, contract.to_dict(), candidate_count=6)

    assert [item.lane for item in concepts] == list(CONCEPT_LANES)
    assert len({item.medium for item in concepts}) == 6
    for concept in concepts:
        assert not validate_visual_concept(concept, contract.to_dict())
        assert len(concept.semantic_encodings) == len(contract.propositions)
        assert concept.focal_beat
        assert concept.motion_grammar not in {"fade", "slide", "generic"}


def test_reference_board_progresses_from_premise_to_stable_proof_hold() -> None:
    contract = build_communication_contract(_ir())
    concept = build_visual_concept_candidates({}, contract.to_dict(), candidate_count=1)[0]
    board = build_visual_reference_board(concept, contract)

    assert [item.fraction for item in board.frames] == [0.08, 0.38, 0.7, 0.94]
    assert board.frames[-1].hold
    assert set(board.frames[-1].visible_proposition_ids) == {
        item.proposition_id for item in contract.propositions
    }
    assert board.signature


def test_model_concepts_are_grounded_and_invalid_proposition_references_are_rejected() -> None:
    contract = build_communication_contract(_ir())
    base = build_visual_concept_candidates({}, contract.to_dict(), candidate_count=1)[0].to_dict()
    base.pop("signature")
    valid = {**base, "concept_id": "authored-valid", "lane": "physical_transformation"}
    invalid = json.loads(json.dumps(valid))
    invalid["concept_id"] = "authored-invalid"
    invalid["semantic_encodings"][0]["proposition_id"] = "invented_proposition"

    accepted, rejected = normalize_authored_visual_concepts(
        {"concepts": [valid, invalid]},
        contract.to_dict(),
        visual_id="visual_001",
    )

    assert [item.concept_id for item in accepted] == ["authored-valid"]
    assert rejected[0]["concept_id"] == "authored-invalid"
    assert "visual_concept_invented_proposition" in rejected[0]["issues"]


def test_concept_search_uses_model_candidate_and_survives_provider_failure() -> None:
    contract = build_communication_contract(_ir())
    model_concept = build_visual_concept_candidates({}, contract.to_dict(), candidate_count=1)[0].to_dict()
    model_concept.pop("signature")
    model_concept["concept_id"] = "model-concept"
    model_concept["metaphor"] = "Signals fold into a compact lens that reveals ranked blocks."

    def model_call(*_args: str) -> str:
        return json.dumps({"concepts": [model_concept]})

    result = author_visual_concepts(
        {"generation_provider": "gemini", "generation_model": "test-model"},
        contract,
        reasoning_call=model_call,
    )

    assert result.model_concept_count == 1
    assert any(item.concept_id == "model-concept" for item in result.concepts)
    assert result.selected_concept_id

    def unavailable(*_args: str) -> str:
        raise TimeoutError("provider unavailable")

    fallback = author_visual_concepts(
        {"generation_provider": "gemini", "generation_model": "test-model"},
        contract,
        reasoning_call=unavailable,
    )
    assert fallback.selected_concept_id
    assert fallback.model_concept_count == 0
    assert "model_concept_authoring_unavailable:TimeoutError" in fallback.warnings


def test_concept_history_penalizes_repeated_lane_without_breaking_eligibility() -> None:
    contract = build_communication_contract(_ir())
    concepts = build_visual_concept_candidates({}, contract.to_dict(), candidate_count=2)
    scores = score_visual_concepts(
        concepts,
        contract.to_dict(),
        history=[{"lane": concepts[0].lane, "visual_nouns": concepts[0].visual_nouns}],
    )
    score_by_id = {item.concept_id: item for item in scores}

    assert score_by_id[concepts[0].concept_id].novelty == 0.0
    assert score_by_id[concepts[1].concept_id].novelty == 1.0
    assert all(item.eligible for item in scores)


def test_selected_concept_and_reference_board_remain_valid_open_program_metadata() -> None:
    contract = build_communication_contract(_ir())
    concept = build_visual_concept_candidates({}, contract.to_dict(), candidate_count=1)[0]
    board = build_visual_reference_board(concept, contract)
    program = build_open_visual_program_candidates(
        _ir(),
        visual_id="visual_001",
        width=1920,
        height=1080,
        duration_sec=4.8,
        fps=60,
        candidate_count=1,
    )[0]

    directed = apply_concept_to_program(program, concept, board)
    validation = validate_open_visual_program(directed, ir=_ir())

    assert validation.passed
    assert directed["concept"]["medium"] == concept.medium
    assert directed["quality_contract"]["visual_reference_board_signature"] == board.signature


def test_open_program_compilation_carries_visual_director_contracts_into_execution() -> None:
    compiled, result = compile_open_visual_program_for_spec(
        {
            "visual_id": "visual_001",
            "duration": 4.8,
            "generation_provider": "gemini",
            "generation_model": "test-model",
        },
        ir=_ir(),
        width=1920,
        height=1080,
        fps=60,
        enable_model_authoring=False,
        candidate_count=4,
    )

    assert result.passed
    assert compiled["visual_communication_contract"]["signature"]
    assert len(compiled["visual_concept_search"]["concepts"]) == 6
    assert compiled["open_visual_authoring"]["selected_concept_id"]
    assert all(
        item["quality_contract"].get("visual_reference_board_signature")
        for item in compiled["open_visual_program_candidates"]
    )
