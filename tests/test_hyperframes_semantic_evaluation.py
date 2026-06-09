from __future__ import annotations

from pathlib import Path

from vex_hyperframes.evaluation import (
    evaluate_semantic_output,
    load_semantic_fixtures,
    visible_text_from_html,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "hyperframes_semantic_cases.json"


def test_semantic_fixture_corpus_covers_core_scene_families_and_rejection() -> None:
    fixtures = load_semantic_fixtures(FIXTURE_PATH)

    scene_types = {fixture.expected_scene_type for fixture in fixtures}
    assert len(fixtures) >= 12
    assert {
        "metric_delta",
        "causal_intervention",
        "guided_process",
        "matched_state_transform",
        "grounded_interface_walkthrough",
        "architecture_flow",
        "decision_branch",
        "narrative_progression",
        "none",
    }.issubset(scene_types)
    assert sum(1 for fixture in fixtures if fixture.expected_action == "reject") >= 2


def test_semantic_evaluator_accepts_grounded_explanation() -> None:
    fixture = _fixture("causal_passive_learning")
    rendered = """
    <html>
      <head><style>.hidden { content: "Input"; }</style></head>
      <body>
        <section>Tutorial watching hides gaps</section>
        <section>No retrieval pressure</section>
        <section>Build a project</section>
        <section>Targeted study</section>
        <script>const invisible = "82%";</script>
      </body>
    </html>
    """

    report = evaluate_semantic_output(fixture, html=rendered, selected=True)

    assert report.passed is True
    assert report.required_coverage == 1.0
    assert report.provenance_coverage >= 0.95
    assert report.invented_numeric_facts == []
    assert visible_text_from_html(rendered).startswith("Tutorial watching")


def test_semantic_evaluator_rejects_fabricated_metrics_and_placeholder_copy() -> None:
    fixture = _fixture("interface_real_states")
    rendered = """
    <main>
      <h1>Editor interface</h1>
      <div>Input captured <span>82%</span></div>
      <div>Model scores context <span>87%</span></div>
      <div>Action rendered <span>92%</span></div>
    </main>
    """

    report = evaluate_semantic_output(fixture, html=rendered, selected=True)

    assert report.passed is False
    assert set(report.invented_numeric_facts) == {"82%", "87%", "92%"}
    assert "render_contains_invented_numeric_facts" in report.issues
    assert "render_contains_forbidden_copy" in report.issues
    assert "render_contains_generic_placeholder_copy" in report.issues


def test_semantic_evaluator_accepts_deliberate_no_visual_decision() -> None:
    fixture = _fixture("vague_motivation_reject")

    report = evaluate_semantic_output(fixture, selected=False)

    assert report.passed is True
    assert report.action_matched is True
    assert report.score == 1.0


def _fixture(case_id: str):
    return next(item for item in load_semantic_fixtures(FIXTURE_PATH) if item.case_id == case_id)
