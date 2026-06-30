from __future__ import annotations

from tools.auto_visuals import _compile_hyperframes_specs
from visual_intelligence import fallback_visual_plan
from visual_skill_graph import (
    apply_visual_skill_graph,
    route_visual_skill,
    skill_graph_prompt_block,
)


def test_skill_graph_routes_metric_opportunity_to_metric_story() -> None:
    card = _metric_card()

    decision = route_visual_skill(
        card,
        available_renderers=[{"name": "hyperframes", "available": True}],
        prefer_premium=True,
    )

    assert decision.passed is True
    assert decision.skill_id == "metric-story"
    assert decision.scene_type in {"metric_delta", "metric_proof"}
    assert decision.renderer_hint == "hyperframes"
    assert decision.preferred_template in {"data_journey", "proof_sequence", "data_pulse"}
    assert decision.slot_values["metric_facts"][0]["value"] == "10%"
    assert any(
        item["skill_id"] == "hyperframes-metric-story"
        for item in decision.skill_slices
    )


def test_skill_graph_rejects_vague_visual_without_executable_scene() -> None:
    decision = route_visual_skill(_vague_card())

    assert decision.passed is False
    assert decision.skill_id == ""
    assert "no_skill_for_scene_type" in decision.reasons
    assert decision.preflight["scene_type"] == "none"


def test_skill_graph_seed_controls_deterministic_fallback_plan() -> None:
    enriched, report = apply_visual_skill_graph(
        [_decision_card()],
        available_renderers=[{"name": "hyperframes", "available": True}],
        prefer_premium=True,
        force_fullscreen=True,
    )

    assert report["accepted_count"] == 1
    assert report["skill_counts"] == {"decision-gate": 1}
    plan = fallback_visual_plan(
        enriched,
        clip_duration=12.0,
        max_visuals=1,
        min_visual_sec=2.4,
        max_visual_sec=5.0,
        scene_cuts=[],
        available_renderers=[{"name": "hyperframes", "available": True}],
        prefer_premium=True,
    )

    assert len(plan) == 1
    assert plan[0]["template"] == "decision_tree"
    assert plan[0]["renderer_hint"] == "hyperframes"
    assert plan[0]["composition_mode"] == "replace"
    assert plan[0]["auto_visual_skill"]["skill_id"] == "decision-gate"


def test_skill_graph_prompt_block_is_a_hard_contract_for_average_models() -> None:
    _, report = apply_visual_skill_graph(
        [_decision_card()],
        available_renderers=[{"name": "hyperframes", "available": True}],
        prefer_premium=True,
    )

    prompt = skill_graph_prompt_block(report)

    assert "Treat the selected skill as the visual architecture" in prompt
    assert "decision-gate" in prompt
    assert "required_labels=" in prompt


def test_semantic_compiler_preserves_auto_visual_skill_contract() -> None:
    enriched, _ = apply_visual_skill_graph(
        [_interface_card()],
        available_renderers=[{"name": "hyperframes", "available": True}],
        prefer_premium=True,
    )
    plan = fallback_visual_plan(
        enriched,
        clip_duration=10.0,
        max_visuals=1,
        min_visual_sec=2.4,
        max_visual_sec=5.0,
        scene_cuts=[],
        available_renderers=[{"name": "hyperframes", "available": True}],
        prefer_premium=True,
    )

    compiled, report = _compile_hyperframes_specs(plan)

    assert report["compiled_count"] == 1
    assert compiled[0]["auto_visual_skill"]["skill_id"] == "grounded-interface"
    assert compiled[0]["hyperframes_compiler"]["scene_type"] == "grounded_interface_walkthrough"


def _base_card(**overrides: object) -> dict[str, object]:
    card = {
        "card_id": "card_001",
        "start": 1.0,
        "end": 4.0,
        "sentence_text": "",
        "source_sentence_text": "",
        "context_text": "",
        "planning_context_text": "",
        "previous_text": "",
        "next_text": "",
        "semantic_frame": {},
        "metric_facts": [],
        "keywords": ["system", "proof"],
        "visual_type_hint": "process",
        "style_pack": "signal_lab",
        "suggested_renderer": "hyperframes",
        "suggested_composition": "replace",
        "priority": 88.0,
        "visualizability": 0.9,
        "generic_penalty": 0.05,
        "numeric_hits": 0,
        "sentence_numeric_hits": 0,
        "process_cues": 0.7,
        "sentence_process_cues": 0.7,
        "contrast_cues": 0.1,
        "sentence_contrast_cues": 0.1,
        "pause_before": 0.3,
        "pause_after": 0.3,
        "scene_distance": 1.0,
        "replace_safety": 0.8,
        "word_count": 12,
        "words_per_second": 2.1,
        "opportunity_contract": {"score": 0.9},
    }
    card.update(overrides)
    return card


def _metric_card() -> dict[str, object]:
    return _base_card(
        card_id="metric_cache",
        sentence_text="DeepSeek V4 Pro needs only 10 percent of the KV cache used before.",
        context_text="The memory reduction lets longer contexts fit on the same hardware.",
        planning_context_text="DeepSeek V4 Pro needs only 10 percent of the KV cache used before. The memory reduction lets longer contexts fit on the same hardware.",
        semantic_frame={
            "viewer_takeaway": "KV cache drops to 10 percent",
            "before_state": "Previous KV cache",
            "after_state": "10 percent KV cache",
            "mechanism": "Memory reduction",
            "result": "Longer context on the same hardware",
        },
        metric_facts=[{"value": "10%", "label": "KV cache requirement"}],
        keywords=["KV cache", "10 percent", "memory"],
        visual_type_hint="data_graphic",
        numeric_hits=1,
        sentence_numeric_hits=1,
    )


def _decision_card() -> dict[str, object]:
    return _base_card(
        card_id="decision_quality",
        sentence_text="If transcript confidence is low, request review; otherwise continue to rendering.",
        context_text="The branch prevents uncertain captions from reaching export.",
        planning_context_text="If transcript confidence is low, request review; otherwise continue to rendering. The branch prevents uncertain captions from reaching export.",
        semantic_frame={
            "decision": "Transcript confidence",
            "low_branch": "Request review",
            "high_branch": "Continue rendering",
            "constraint": "Protect caption accuracy",
        },
        keywords=["transcript confidence", "review", "rendering"],
        process_cues=0.6,
        sentence_process_cues=0.6,
        contrast_cues=0.4,
        sentence_contrast_cues=0.4,
    )


def _interface_card() -> dict[str, object]:
    return _base_card(
        card_id="interface_retry",
        sentence_text=(
            "The editor highlights the failed shot, opens its render log, "
            "and lets the user retry only that shot."
        ),
        context_text="The source recording contains the actual editor interface.",
        planning_context_text=(
            "The editor highlights the failed shot, opens its render log, "
            "and lets the user retry only that shot. The source recording contains the actual editor interface."
        ),
        semantic_frame={
            "screen": "Editor interface",
            "focus": "Failed shot",
            "action": "Open render log",
            "result": "Retry that shot",
        },
        keywords=["failed shot", "render log", "retry"],
        visual_type_hint="product_ui",
    )


def _vague_card() -> dict[str, object]:
    return _base_card(
        card_id="vague",
        sentence_text="This powerful idea can change everything if you believe in it.",
        context_text="No concrete mechanism, entities, evidence, or observable change is provided.",
        planning_context_text="This powerful idea can change everything if you believe in it.",
        semantic_frame={},
        keywords=["idea", "change"],
        visual_type_hint="abstract_motion",
        priority=40.0,
        visualizability=0.2,
        generic_penalty=0.9,
        process_cues=0.0,
        sentence_process_cues=0.0,
        contrast_cues=0.0,
        sentence_contrast_cues=0.0,
        opportunity_contract={"score": 0.1},
    )
