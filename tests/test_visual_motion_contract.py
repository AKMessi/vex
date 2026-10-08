import json
import subprocess
from pathlib import Path
import pytest
from tests.test_visual_director import _spec
from tests.test_visual_communication_contract import _ir
from vex_visuals.motion_state import evaluate_track
from vex_visuals.open_visual_program import sign_open_visual_program, validate_open_visual_program
from vex_hyperframes.open_visual_runtime import compile_open_visual_stage
from vex_runtime.hyperframes import resolve_node_executable


def test_motion_keeps_intermediate_keys_and_absolute_initial_offset():
    track = {"property": "translate_x", "keyframes": [{"t": 0, "value": .1, "easing": "linear"}, {"t": .5, "value": .7, "easing": "linear"}, {"t": 1, "value": .1, "easing": "linear"}]}
    assert evaluate_track([track], "translate_x", 0) == .1
    assert evaluate_track([track], "translate_x", .5) == .7
    assert evaluate_track([track], "translate_x", 1) == .1


def test_python_and_javascript_motion_evaluators_conform():
    node = resolve_node_executable()
    if not node:
        pytest.skip("Node is required for cross-runtime motion conformance")
    module = (Path(__file__).parents[1] / "renderers/visual_motion.mjs").as_uri()
    for easing in ["linear", "ease_in", "ease_out", "ease_in_out", "spring_gentle", "spring_snappy"]:
        tracks = [{"property": "scale", "keyframes": [{"t": 0, "value": .6, "easing": easing}, {"t": .5, "value": 1.2, "easing": easing}, {"t": 1, "value": 1, "easing": easing}]}]
        code = f"import {{evaluateTrack}} from {json.dumps(module)}; console.log(JSON.stringify([0,.1,.25,.5,.75,1].map(t=>evaluateTrack({json.dumps(tracks)},'scale',t,1))));"
        values = json.loads(subprocess.check_output([node, "--input-type=module", "-e", code], text=True))
        assert values == pytest.approx([evaluate_track(tracks, "scale", time, 1) for time in [0, .1, .25, .5, .75, 1]])


def test_chart_data_requires_individual_fact_grounding():
    program = _spec()["open_visual_program"]
    element = program["elements"][1]
    element["type"] = "chart"
    element["data"] = [{"label": "tokens", "value": 4, "fact_id": "fact_compress"}]
    program = sign_open_visual_program(program)
    # Numeric source may use words; an explicit numeric field is also supported.
    ir = _ir()
    ir["facts"][0]["value"] = "4"
    from vex_visuals.generative_authoring import compile_open_visual_program_for_spec
    fresh, _ = compile_open_visual_program_for_spec({"visual_id": "visual_001", "duration": 4.8}, ir=ir, width=1920, height=1080, fps=30, enable_model_authoring=False)
    program = fresh["open_visual_program"]
    program["elements"][1]["type"] = "chart"
    program["elements"][1]["data"] = [{"label": "tokens", "value": 4, "fact_id": "fact_compress"}]
    program = sign_open_visual_program(program)
    assert validate_open_visual_program(program, ir=ir).passed
    html = compile_open_visual_stage(program, ir=ir).html
    assert 'data-vex-chart-value="4.0"' in html
    program["elements"][1]["data"][0]["value"] = 99
    assert "chart_value_not_grounded:" in " ".join(validate_open_visual_program(sign_open_visual_program(program), ir=ir).errors)
