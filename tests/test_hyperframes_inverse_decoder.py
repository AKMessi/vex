from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np

from vex_hyperframes.compiler import compile_hyperframes_plan
from vex_hyperframes.inverse_decoder import (
    BlindFrameDecode,
    DecodedRelation,
    blind_decode_prompt,
    build_counterfactual_frames,
    evaluate_inverse_decode,
    parse_blind_decode,
)


def test_blind_decoder_prompt_contains_no_intended_answer() -> None:
    prompt = blind_decode_prompt(4)

    assert "Classify request" not in prompt
    assert "production contract" not in prompt.lower()
    assert "not given the intended" in prompt
    assert "unsupported_visual_claims" in prompt


def test_inverse_decoder_accepts_decodable_claim_and_sensitive_counterfactuals() -> None:
    plan = compile_hyperframes_plan(_process_spec())
    contract = plan.production_contract.to_dict()
    graph = contract["visual_claim_graph"]
    labels = {
        item["node_id"]: item["label"]
        for item in graph["nodes"]
    }
    decoded = BlindFrameDecode(
        thesis=contract["takeaway"],
        objects=list(labels.values()),
        relations=[
            DecodedRelation(
                source=labels[item["source_id"]],
                relation_type=item["relation_type"],
                target=labels[item["target_id"]],
                confidence=0.96,
            )
            for item in graph["relations"]
        ],
        sequence=[labels[item] for item in graph["sequence_node_ids"]],
        confidence=0.95,
    )
    ablated = BlindFrameDecode(
        thesis="Several labels are visible but their relationship is unclear.",
        objects=list(labels.values()),
        relations=[],
        sequence=[],
        confidence=0.18,
        ambiguities=["No clear connector remains."],
    )
    scrambled = BlindFrameDecode(
        thesis="The order is ambiguous.",
        objects=list(labels.values()),
        relations=[],
        sequence=list(reversed(decoded.sequence)),
        confidence=0.24,
        ambiguities=["The process order conflicts across frames."],
    )

    report = evaluate_inverse_decode(
        decoded,
        production_contract=contract,
        relation_ablation_decode=ablated,
        temporal_scramble_decode=scrambled,
    )

    assert report.passed is True
    assert report.relation_coverage >= 0.95
    assert report.sequence_score == 1.0
    assert report.counterfactual.passed is True


def test_inverse_decoder_rejects_label_only_visual_without_relations() -> None:
    plan = compile_hyperframes_plan(_process_spec())
    contract = plan.production_contract.to_dict()
    labels = [
        item["label"]
        for item in contract["visual_claim_graph"]["nodes"]
    ]
    decoded = BlindFrameDecode(
        thesis=contract["takeaway"],
        objects=labels,
        relations=[],
        sequence=labels,
        confidence=0.88,
    )

    report = evaluate_inverse_decode(
        decoded,
        production_contract=contract,
        require_counterfactuals=False,
    )

    assert report.passed is False
    assert report.object_coverage == 1.0
    assert report.relation_coverage == 0.0
    assert "blind_decoder_could_not_recover_required_relations" in report.issues


def test_counterfactual_builder_ablates_proof_region_and_scrambles_time(
    tmp_path: Path,
) -> None:
    paths: list[Path] = []
    for index, value in enumerate((32, 96, 160, 224), start=1):
        frame = np.full((80, 120, 3), value, dtype=np.uint8)
        frame[32:48, 8:112] = 255 - value
        path = tmp_path / f"frame_{index:02d}.png"
        iio.imwrite(path, frame)
        paths.append(path)

    ablated, scrambled = build_counterfactual_frames(
        paths,
        tmp_path / "counterfactuals",
        encoding_family="linear_trace",
    )

    assert len(ablated) == 4
    assert not np.array_equal(iio.imread(paths[0]), iio.imread(ablated[0]))
    assert scrambled[0] == paths[-1]
    assert scrambled[-1] == paths[0]


def test_blind_decode_parser_rejects_relations_outside_fixed_ontology() -> None:
    decoded = parse_blind_decode(
        {
            "thesis": "A request reaches review.",
            "objects": ["Request", "Review"],
            "relations": [
                {
                    "source": "Request",
                    "relation_type": "teleports_magically",
                    "target": "Review",
                    "confidence": 0.9,
                },
                {
                    "source": "Request",
                    "relation_type": "routes_to",
                    "target": "Review",
                    "confidence": 0.8,
                },
            ],
            "sequence": ["Request", "Review"],
            "confidence": 0.82,
        }
    )

    assert len(decoded.relations) == 1
    assert decoded.relations[0].relation_type == "routes_to"


def test_inverse_decoder_matches_grounded_inflection_variants() -> None:
    decoded = BlindFrameDecode(
        thesis="The agent checks policy.",
        objects=["checks policy"],
        relations=[],
        sequence=["checks policy"],
        confidence=0.9,
    )
    report = evaluate_inverse_decode(
        decoded,
        production_contract={
            "thesis": "Check policy",
            "takeaway": "Check policy",
            "visual_claim_graph": {
                "nodes": [
                    {"node_id": "object_01", "label": "Check policy"}
                ],
                "relations": [],
                "sequence_node_ids": ["object_01"],
            },
        },
        require_counterfactuals=False,
        min_score=0.6,
    )

    assert report.passed is True
    assert report.object_coverage == 1.0


def _process_spec() -> dict:
    return {
        "visual_id": "inverse_decoder_process",
        "sentence_text": (
            "The request is classified, checked against policy, then sent to a human."
        ),
        "context_text": "The handoff prevents unsupported answers.",
        "semantic_frame": {
            "steps": ["Classify request", "Check policy", "Send to human"],
            "result": "Prevent unsupported answers",
        },
        "required_labels": [
            "Classify request",
            "Check policy",
            "Send to human",
            "Prevent unsupported answers",
        ],
        "duration": 4.0,
        "composition_mode": "replace",
    }
