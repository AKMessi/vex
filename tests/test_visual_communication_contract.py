from __future__ import annotations

from vex_visuals.communication_contract import (
    build_communication_contract,
    communication_contract_signature,
    evaluate_viewer_answers,
    semantic_text_score,
    validate_communication_contract,
)


def _ir() -> dict:
    return {
        "visual_id": "visual_001",
        "scene_type": "guided_process",
        "thesis": "Compressed Sparse Attention",
        "takeaway": "An indexer ranks compressed blocks and selects only the best ones",
        "evidence": [
            {
                "evidence_id": "evidence_transcript",
                "source_type": "transcript",
                "text": (
                    "Every four tokens become one compressed KV entry. "
                    "Then an indexer scores every compressed block and picks the top blocks."
                ),
            }
        ],
        "facts": [
            {
                "fact_id": "fact_compress",
                "fact_type": "mechanism",
                "label": "Every four tokens become one compressed KV entry",
                "subject": "four tokens",
                "predicate": "become",
                "object": "one compressed KV entry",
                "evidence_ids": ["evidence_transcript"],
                "confidence": 0.98,
            },
            {
                "fact_id": "fact_select",
                "fact_type": "result",
                "label": "The indexer scores each compressed block and picks the top blocks",
                "evidence_ids": ["evidence_transcript"],
                "confidence": 0.96,
            },
        ],
        "objects": [
            {
                "object_id": "tokens",
                "role": "input",
                "label": "4 tokens",
                "meaning": "Four tokens",
                "fact_ids": ["fact_compress"],
                "emphasis": 0.75,
            },
            {
                "object_id": "entry",
                "role": "mechanism",
                "label": "1 compressed KV entry",
                "meaning": "One compressed representation",
                "fact_ids": ["fact_compress"],
                "emphasis": 0.85,
            },
            {
                "object_id": "indexer",
                "role": "result",
                "label": "Indexer selects top blocks",
                "meaning": "The indexer ranks compressed blocks and selects the strongest",
                "fact_ids": ["fact_select"],
                "emphasis": 0.95,
            },
        ],
        "relations": [
            {
                "relation_id": "compression",
                "source_id": "tokens",
                "relation_type": "transforms_into",
                "target_id": "entry",
                "evidence_ids": ["evidence_transcript"],
                "required": True,
            },
            {
                "relation_id": "selection",
                "source_id": "entry",
                "relation_type": "enables",
                "target_id": "indexer",
                "evidence_ids": ["evidence_transcript"],
                "required": True,
            },
        ],
        "beats": [
            {
                "beat_id": "compress",
                "subject_id": "tokens",
                "action": "compresses_into",
                "target_id": "entry",
                "start_fraction": 0.1,
            },
            {
                "beat_id": "select",
                "subject_id": "entry",
                "action": "is_ranked_by",
                "target_id": "indexer",
                "start_fraction": 0.55,
            },
        ],
        "required_labels": ["4 tokens", "1 compressed KV entry"],
        "forbidden_content": ["invented metrics"],
    }


def test_contract_is_signed_dependency_aware_and_source_bound() -> None:
    contract = build_communication_contract(_ir())

    assert not validate_communication_contract(contract, source_ir=_ir())
    assert contract.signature == communication_contract_signature(contract)
    assert len(contract.propositions) == 5
    assert [item.proposition for item in contract.propositions[:3]] == [
        "Four tokens",
        "One compressed representation",
        "The indexer ranks compressed blocks and selects the strongest",
    ]
    assert len({item.question for item in contract.questions}) == len(contract.questions)
    relation = next(item for item in contract.propositions if item.proposition_id == "relation_01")
    assert relation.dependency_ids == ["proposition_01", "proposition_02"]
    assert "4tokens" in contract.required_terms


def test_semantic_match_accepts_concise_paraphrases_without_exact_transcript_copy() -> None:
    assert semantic_text_score(
        "Four tokens compress into one KV summary",
        "Every four tokens become one compressed KV entry",
    ) >= 0.56
    assert semantic_text_score(
        "The indexer ranks blocks and selects the strongest",
        "The indexer scores each compressed block and picks the top blocks",
    ) >= 0.56


def test_viewer_evaluation_passes_semantic_answers_and_temporal_proof() -> None:
    contract = build_communication_contract(_ir())
    answers = {
        question.question_id: (
            "Four tokens compress into one KV summary"
            if question.proposition_id in {"proposition_01", "proposition_02"}
            else "The indexer ranks compressed blocks and selects the strongest"
            if question.proposition_id == "proposition_03"
            else "The token group turns into a compact entry"
            if question.proposition_id == "relation_01"
            else "The compact entry enables the indexer to pick top blocks"
        )
        for question in contract.questions
    }

    result = evaluate_viewer_answers(
        contract,
        answers,
        decoded_sequence=[
            "four tokens condense into a KV summary",
            "the compact block is ranked by an indexer",
        ],
    )

    assert result.passed
    assert result.score >= contract.minimum_semantic_score
    assert result.proposition_coverage == 1.0


def test_viewer_evaluation_hard_fails_numeric_drift_and_unsupported_claims() -> None:
    contract = build_communication_contract(_ir())
    answers = {
        question.question_id: "Eight tokens become two entries, saving 90 percent"
        for question in contract.questions
    }

    result = evaluate_viewer_answers(
        contract,
        answers,
        unsupported_claims=["The method saves 90 percent of compute"],
    )

    assert not result.passed
    assert "viewer_found_unsupported_claims" in result.issues
    assert any(
        "viewer_answer_numeric_mismatch" in item.issues
        for item in result.results
        if item.proposition_id in {"proposition_01", "proposition_02"}
    )


def test_viewer_evaluation_uses_all_blind_decoder_evidence_without_losing_numeric_rigor() -> None:
    contract = build_communication_contract(_ir())
    answers = {
        question.question_id: "Compressed Sparse Attention"
        for question in contract.questions
    }

    result = evaluate_viewer_answers(
        contract,
        answers,
        decoded_thesis=(
            "Four input tokens compress into a single KV entry before an indexer "
            "selects the top blocks."
        ),
        decoded_sequence=[
            "Four tokens move toward a compression gate.",
            "The four tokens resolve into one compressed KV entry.",
            "The compressed entry enables an indexer to select top blocks.",
        ],
    )

    assert result.passed
    assert result.proposition_coverage == 1.0
    assert result.temporal_score >= 0.64


def test_tampered_contract_is_rejected() -> None:
    contract = build_communication_contract(_ir()).to_dict()
    contract["takeaway"] = "Invented takeaway"

    assert "communication_contract_signature_mismatch" in validate_communication_contract(contract)
