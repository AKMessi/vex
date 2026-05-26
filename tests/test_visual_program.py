from __future__ import annotations

from pathlib import Path

from engine import _normalize_visual_overlays
from vex_hyperframes.composer import build_composition
from vex_hyperframes.validator import validate_composition_html
from visual_program import apply_visual_program_to_specs, build_visual_narrative_program


def test_visual_program_builds_episode_context_and_transitions() -> None:
    program = build_visual_narrative_program(
        [_card("visual_card_001", 4.0, mode="causal_chain"), _card("visual_card_002", 8.0, mode="process_route")],
        clip_duration=18.0,
        max_visuals=5,
        scene_cuts=[3.9, 12.0],
        prefer_premium=True,
    )

    payload = program.to_dict()

    assert payload["program_id"] == "visual_program_v1"
    assert payload["style_bible"]["primary_style_pack"] == "signal_lab"
    assert payload["episodes"]
    assert payload["episodes"][0]["transition_in"]["kind"] in {"scene_match_cut", "soft_luma_fade", "audio_bridge_fade", "micro_dissolve"}
    assert payload["concept_memory"][0]["concept_id"] == "concept_01"


def test_visual_program_enriches_specs_for_hyperframes_context() -> None:
    program = build_visual_narrative_program(
        [_card("visual_card_001", 4.0, mode="causal_chain")],
        clip_duration=14.0,
        max_visuals=4,
        scene_cuts=[],
        prefer_premium=True,
    )

    enriched = apply_visual_program_to_specs(
        [
            {
                "visual_id": "visual_001",
                "card_id": "visual_card_001",
                "template": "ribbon_quote",
                "renderer_hint": "auto",
                "style_pack": "editorial_clean",
                "composition_mode": "replace",
                "steps": [],
            }
        ],
        program,
        style_pack="auto",
    )

    spec = enriched[0]
    assert spec["template"] == "causal_chain"
    assert spec["renderer_hint"] == "hyperframes"
    assert spec["episode_id"] == "episode_001"
    assert spec["program_context"]["program_id"] == "visual_program_v1"
    assert spec["transition_out"]["direction"] == "out"


def test_hyperframes_new_templates_validate_and_carry_program_metadata() -> None:
    for template in [
        "causal_chain",
        "flywheel_loop",
        "decision_matrix",
        "anatomy_cutaway",
        "stack_ranking",
        "contrast_ladder",
        "proof_sequence",
        "narrative_arc",
    ]:
        composition = build_composition(
            {
                "visual_id": f"visual_{template}",
                "template": template,
                "headline": "Feedback Loop",
                "deck": "Context makes the mechanism visible",
                "steps": ["Cause", "Mechanism", "Effect", "Payoff"],
                "supporting_lines": ["Signal", "Criteria", "Result"],
                "left_detail": "Scattered effort",
                "right_detail": "Focused feedback",
                "duration": 3.0,
                "visual_type_hint": "process",
                "composition_mode": "replace",
                "program_context": {"program_id": "visual_program_v1"},
                "episode_context": {"continuity_group": "chapter_01:concept_01"},
                "transition_in": {"kind": "soft_luma_fade", "duration_sec": 0.24},
                "transition_out": {"kind": "micro_dissolve", "duration_sec": 0.14},
            },
            width=1280,
            height=720,
            fps=30,
        )
        report = validate_composition_html(
            composition.html,
            expected_width=1280,
            expected_height=720,
            expected_duration=3.0,
        )

        assert report.valid, (template, report.errors)
        assert composition.metadata["template"] == template
        assert composition.metadata["program_context"]["program_id"] == "visual_program_v1"


def test_replace_overlay_preserves_transition_handles(tmp_path: Path) -> None:
    asset_path = tmp_path / "visual.mp4"
    asset_path.write_bytes(b"placeholder")

    normalized = _normalize_visual_overlays(
        [
            {
                "start": 1.0,
                "end": 4.0,
                "asset_path": str(asset_path),
                "compose_mode": "replace",
                "transition_in": {"kind": "soft_luma_fade", "duration_sec": 0.24},
                "transition_out": {"kind": "micro_dissolve", "duration_sec": 0.18},
            }
        ],
        duration=8.0,
        width=1920,
        height=1080,
    )

    assert normalized[0]["transition_in_sec"] == 0.24
    assert normalized[0]["transition_out_sec"] == 0.18


def test_alpha_overlay_mode_stays_full_frame(tmp_path: Path) -> None:
    asset_path = tmp_path / "visual.mov"
    asset_path.write_bytes(b"placeholder")

    normalized = _normalize_visual_overlays(
        [
            {
                "start": 1.0,
                "end": 4.0,
                "asset_path": str(asset_path),
                "compose_mode": "overlay",
                "has_alpha": True,
                "position": "center_right",
                "scale": 0.4,
            }
        ],
        duration=8.0,
        width=1920,
        height=1080,
    )

    assert normalized[0]["compose_mode"] == "overlay"
    assert normalized[0]["position"] == "center"
    assert normalized[0]["scale"] == 1.0


def _card(card_id: str, start: float, *, mode: str) -> dict:
    role = "core_mechanism" if mode in {"causal_chain", "process_route"} else "concrete_proof"
    return {
        "card_id": card_id,
        "start": start,
        "end": start + 2.4,
        "sentence_text": "Because the loop repeats, each attempt creates clearer feedback.",
        "context_text": "The system moves from scattered effort to a focused feedback loop.",
        "previous_text": "Start with the real attempt.",
        "next_text": "Then target only the blocker.",
        "keywords": ["feedback loop", "attempt", "blocker"],
        "visual_type_hint": "process",
        "word_count": 12,
        "words_per_second": 3.0,
        "pause_before": 0.45,
        "pause_after": 0.38,
        "nearest_scene_cut": None,
        "scene_distance": 0.2 if card_id.endswith("001") else 2.4,
        "sentence_numeric_hits": 0,
        "numeric_hits": 0,
        "sentence_process_cues": 0.8,
        "process_cues": 0.8,
        "sentence_contrast_cues": 0.35,
        "contrast_cues": 0.35,
        "generic_penalty": 0.08,
        "concrete_hits": 2.0,
        "proper_nouns": 0,
        "replace_safety": 0.74,
        "visualizability": 0.78,
        "semantic_frame": {
            "intuition_mode": mode,
            "intuition_role": role,
            "intuition_payoff": 0.86,
            "before_state": "Scattered effort",
            "after_state": "Focused feedback loop",
            "cause": "The loop repeats",
            "effect": "Feedback gets clearer",
            "mental_model": "Show the cause and effect as a repeatable feedback mechanism.",
            "viewer_takeaway": "Focused feedback loop",
        },
        "intuition_mode": mode,
        "intuition_role": role,
        "intuition_payoff": 0.86,
        "novelty_key": card_id,
        "suggested_composition": "replace",
        "style_pack": "signal_lab",
        "suggested_renderer": "hyperframes",
        "priority": 92.0 if card_id.endswith("001") else 84.0,
    }
