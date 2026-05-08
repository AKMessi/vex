from __future__ import annotations

from vex_manim.blueprint import build_scene_blueprints
from vex_manim.briefs import build_scene_brief
from vex_manim.director import (
    _execution_plan_system_prompt,
    _execution_plan_user_prompt,
    _system_prompt,
    _user_prompt,
    build_deterministic_execution_plan,
)
from vex_manim.skill_pack import retrieve_skill_slices


def test_manim_skill_retrieval_always_includes_production_guardrails() -> None:
    brief = _brief()
    skills = retrieve_skill_slices(brief, limit=1, preferred_features=())
    skill_ids = [skill.skill_id for skill in skills]

    assert skill_ids[:3] == [
        "manim-ce-production-contract",
        "visual-logic-fidelity",
        "runtime-safety-and-api-footguns",
    ]
    assert len(skills) == 4
    prompt_block = "\n\n".join(skill.to_prompt_block() for skill in skills)
    assert "Manim Community Edition" in prompt_block
    assert "black frame" in prompt_block
    assert "ValueTracker" in prompt_block
    assert "ax.c2p" in prompt_block
    assert "Transform mutates" in prompt_block


def test_manim_codegen_prompt_contains_common_failure_mode_contracts() -> None:
    brief = _brief()
    blueprint = build_scene_blueprints(brief, limit=1)[0]
    plan = build_deterministic_execution_plan(brief, blueprint)
    skills = retrieve_skill_slices(brief, limit=2, preferred_features=blueprint.suggested_features)

    system_prompt = _system_prompt()
    user_prompt = _user_prompt(
        brief,
        examples=[],
        skills=skills,
        blueprint=blueprint,
        execution_plan=plan,
        alternative_blueprints=[],
        storyboard_context="- Frame 1: show passive watching\n- Frame 2: expose project blocker\n- Frame 3: resolve with targeted study",
    )

    assert "Manim Community Edition" in system_prompt
    assert "manimlib" in system_prompt
    assert "construct(self)" in system_prompt
    assert "Use Text for ordinary copy" in system_prompt
    assert "ValueTracker with always_redraw" in system_prompt
    assert "ax.c2p" in system_prompt
    assert "Transform mutates" in system_prompt
    assert "visual events is a failed scene" in system_prompt

    assert "misspelled construct method renders a black frame" in user_prompt
    assert "Do not use manimlib" in user_prompt
    assert "skipping the causal visual logic" in user_prompt
    assert "Use Text for normal labels" in user_prompt
    assert "tracker.get_value() snapshots" in user_prompt
    assert "ax.c2p" in user_prompt
    assert "Transform changes the source into the target" in user_prompt


def test_execution_plan_prompt_requires_api_and_visual_logic_guardrails() -> None:
    brief = _brief()
    blueprint = build_scene_blueprints(brief, limit=1)[0]

    system_prompt = _execution_plan_system_prompt()
    user_prompt = _execution_plan_user_prompt(brief, blueprint)

    assert "Manim Community Edition" in system_prompt
    assert "legacy ManimGL API mixing" in system_prompt
    assert "visual-logic drift" in system_prompt
    assert "visibly changes and why" in user_prompt
    assert "correct construct(self)" in user_prompt
    assert "Axes coordinate conversion" in user_prompt
    assert "transform semantics" in user_prompt


def _brief():
    return build_scene_brief(
        {
            "visual_id": "skill_contract_smoke",
            "template": "spotlight_compare",
            "headline": "Build First Study Later",
            "deck": "Inverted learning loop",
            "sentence_text": "You do not learn hard things by watching tutorials for ten hours.",
            "context_text": "Pick a small project, get stuck, then study exactly what blocks you.",
            "left_detail": "Tutorial binge",
            "right_detail": "Build then study",
            "supporting_lines": ["Get stuck", "Targeted study", "Build first"],
            "duration": 5.0,
            "semantic_frame": {
                "intuition_mode": "misconception_flip",
                "mental_model": "Active building exposes the exact gaps passive watching hides.",
                "viewer_takeaway": "Build first, study the blocker.",
                "before_state": "Tutorial binge",
                "after_state": "Build then study",
                "cause": "Passive watching hides gaps",
                "effect": "Getting stuck reveals the next lesson",
            },
        },
        width=1920,
        height=1080,
        fps=30,
        latex_available=False,
    )
