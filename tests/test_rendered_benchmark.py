import copy
import json

import pytest
import config
from vex_visuals.benchmark import (
    corpus,
    compile_case,
    compare_reports,
    calibrate_report,
)
from vex_visuals.open_visual_program import validate_open_visual_program
from visual_explanation import validate_visual_explanation_ir
from vex_runtime.renderer_policy import renderer_policy, apply_renderer_policy


def test_fixed_rendered_corpus_is_grounded_and_covers_all_aspects():
    cases = corpus()["cases"]
    assert {"landscape", "portrait", "square"} <= {
        tag for case in cases for tag in case["tags"]
    }
    for case in cases:
        spec = compile_case(case)
        assert validate_visual_explanation_ir(spec["visual_explanation_ir"]).passed
        assert validate_open_visual_program(
            spec["open_visual_program"], ir=spec["visual_explanation_ir"]
        ).passed
        assert spec["open_visual_program_candidates"] == [spec["open_visual_program"]]


def _report():
    return {
        "corpus_digest": "same",
        "vision_requested": False,
        "cases": [
            {
                "key": "scene:remotion",
                "canvas": [1280, 720],
                "technical_passed": True,
                "semantic_state": "unverified",
                "local_score": 0.8,
                "elapsed_sec": 3,
            }
        ],
    }


def test_comparison_rejects_missing_cases_and_exposes_regressions():
    before = _report()
    after = copy.deepcopy(before)
    after["cases"][0]["technical_passed"] = False
    assert compare_reports(before, after)["technical_regressions"] == 1
    after["cases"] = []
    with pytest.raises(ValueError):
        compare_reports(before, after)


def test_human_calibration_never_invents_ratings_or_acceptance():
    report = _report()
    assert calibrate_report(report, [])["mean_visual_preference"] is None
    ratings = [
        {
            "key": "scene:remotion",
            "reviewer": "reviewer-a",
            "comprehension": 2,
            "visual_preference": 4,
        }
    ]
    assert calibrate_report(report, ratings)["false_acceptance_rate"] is None
    report["cases"][0]["semantic_state"] = "verified"
    assert calibrate_report(report, ratings)["false_acceptance_rate"] == 1
    with pytest.raises(ValueError):
        calibrate_report(report, ratings * 2)


def test_worker_policy_propagates_overrides_without_serializing_secrets(monkeypatch):
    for role in ["AUTHORING", "REPAIR", "VERIFIER", "PAIRWISE"]:
        monkeypatch.setenv(f"VISUAL_{role}_REASONING_EFFORT", "none")
    monkeypatch.setattr(config, "GROQ_API_KEY", "sentinel-secret")
    monkeypatch.setattr(config, "HYPERFRAMES_VARIANT_COUNT", 1)
    policy = renderer_policy()
    assert "sentinel-secret" not in json.dumps(policy)
    monkeypatch.setattr(config, "HYPERFRAMES_VARIANT_COUNT", 4)
    apply_renderer_policy(policy)
    assert config.HYPERFRAMES_VARIANT_COUNT == 1
    policy["credentials_enabled"]["GROQ_API_KEY"] = False
    apply_renderer_policy(policy)
    assert not config.GROQ_API_KEY
    with pytest.raises(ValueError):
        apply_renderer_policy({"settings": {"GROQ_API_KEY": "forbidden"}})


def test_role_reasoning_settings_are_independent_and_validated(monkeypatch):
    monkeypatch.setenv("VISUAL_AUTHORING_REASONING_EFFORT", "low")
    monkeypatch.setenv("VISUAL_VERIFIER_REASONING_EFFORT", "none")
    assert config.visual_reasoning_effort("authoring") == "low"
    assert config.visual_reasoning_effort("verifier") == "none"
    monkeypatch.setenv("VISUAL_REPAIR_REASONING_EFFORT", "invented")
    with pytest.raises(ValueError):
        config.visual_reasoning_effort("repair")


def test_object_questions_bind_quantities_to_the_object_instead_of_entire_fact():
    from vex_visuals.communication_contract import build_communication_contract
    from tests.test_visual_communication_contract import _ir

    ir = _ir()
    ir["objects"][0]["meaning"] = ir["facts"][0]["label"]
    contract = build_communication_contract(ir)
    assert contract.propositions[0].exact_numbers == ["4tokens"]


def test_authoring_stage_resumes_completed_compilation_without_repeating_work(
    tmp_path, monkeypatch
):
    import vex_visuals.generative_authoring as authoring
    from vex_runtime.visual_run import visual_run
    from tests.test_visual_communication_contract import _ir

    spec = {"visual_id": "resume-scene", "duration": 4.8}
    options = {
        "ir": _ir(),
        "width": 1280,
        "height": 720,
        "fps": 24,
        "enable_model_authoring": False,
        "candidate_count": 1,
    }
    with visual_run(tmp_path):
        first, result = authoring.compile_open_visual_program_for_spec(spec, **options)
        assert result.passed and not first["authoring_stage_cache_hit"]
    monkeypatch.setattr(
        authoring,
        "_compile_open_visual_program_for_spec",
        lambda *_args, **_kwargs: pytest.fail("Completed authoring was repeated"),
    )
    with visual_run(tmp_path):
        resumed, recovered = authoring.compile_open_visual_program_for_spec(
            spec, **options
        )
        assert resumed["authoring_stage_cache_hit"]
        assert recovered.to_dict() == result.to_dict()


def test_observed_layout_density_and_text_contrast_do_not_confuse_sparse_ink():
    from vex_visuals.telemetry import (
        evaluate_browser_telemetry,
        observed_layout_occupancy,
    )

    sample = {
        "fonts_ready": True,
        "viewport": {"width": 1000, "height": 1000},
        "nodes": [
            {
                "element_id": "title",
                "visible": True,
                "text": "Required title",
                "font_size": 40,
                "text_color": "rgb(255, 255, 255)",
                "background_color": "rgb(244, 240, 232)",
                "bounds": {"x": 100, "y": 100, "width": 800, "height": 300},
            }
        ],
    }
    assert (
        "browser_low_contrast_text:title"
        in evaluate_browser_telemetry([sample])["issues"]
    )
    sample["nodes"][0]["text_color"] = "rgb(17, 17, 17)"
    assert evaluate_browser_telemetry([sample])["passed"]
    assert observed_layout_occupancy([sample]) == pytest.approx(0.24)
    # A nested rectangle cannot double-count occupied space.
    sample["nodes"].append(dict(sample["nodes"][0]))
    assert observed_layout_occupancy([sample]) == pytest.approx(0.24)
