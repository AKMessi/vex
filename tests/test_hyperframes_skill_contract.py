from __future__ import annotations

from pathlib import Path

from vex_hyperframes.skill_pack import retrieve_skill_slices


SKILL_ROOT = Path(__file__).parents[1] / "skills" / "vex-hyperframes-director"


def test_hyperframes_skill_retrieval_always_includes_hard_guardrails() -> None:
    skills = retrieve_skill_slices(
        "semantic_interface",
        scene_type="grounded_interface_walkthrough",
        blueprint_id="interface_state_trace",
        limit=1,
    )
    skill_ids = [skill.skill_id for skill in skills]

    assert skill_ids[:4] == [
        "hyperframes-production-contract",
        "hyperframes-evidence-fidelity",
        "hyperframes-seekable-motion",
        "hyperframes-semantic-qa",
    ]
    assert skill_ids[4] == "hyperframes-grounded-interface"
    prompt = "\n\n".join(skill.to_prompt_block() for skill in skills)
    assert "Synthetic percentages" in prompt
    assert "window.__timelines" in prompt
    assert "least-bad variant" in prompt


def test_hyperframes_skill_retrieval_targets_semantic_scene_family() -> None:
    metric = retrieve_skill_slices(
        "semantic_metric",
        scene_type="metric_intervention",
        blueprint_id="metric_intervention_trace",
        limit=1,
    )
    architecture = retrieve_skill_slices(
        "semantic_architecture",
        scene_type="architecture_flow",
        blueprint_id="architecture_service_lifecycle",
        limit=1,
    )

    assert metric[-1].skill_id == "hyperframes-metric-story"
    assert architecture[-1].skill_id == "hyperframes-architecture-flow"


def test_repository_hyperframes_skill_has_progressive_disclosure_resources() -> None:
    skill_text = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
    agent_text = (SKILL_ROOT / "agents" / "openai.yaml").read_text(encoding="utf-8")
    references = sorted((SKILL_ROOT / "references").glob("*.md"))

    assert skill_text.startswith("---\nname: vex-hyperframes-director\n")
    assert "description:" in skill_text
    assert "[TODO" not in skill_text
    assert "Build explanations, not themed slides." in skill_text
    assert len(references) == 5
    assert all(path.name in skill_text for path in references)
    assert "$vex-hyperframes-director" in agent_text
