from pathlib import Path
from types import SimpleNamespace

from tests.test_visual_communication_contract import _ir
from tests.test_visual_verifier import _payload
from vex_visuals.communication_contract import build_communication_contract, semantic_text_score
from vex_visuals.evidence import build_verification_receipt, select_evidence_frames, validate_verification_receipt
from vex_visuals.verifier import evaluate_verifier_payload, run_visual_verifier
from tests.test_visual_director import _spec, _asset, _LocalQA
from vex_visuals.director import direct_rendered_visual
from tools.auto_visuals import _merge_visual_director_quality, RenderedVisualQA


def test_negated_claims_cannot_be_verified():
    contract = build_communication_contract(_ir())
    payload = _payload(contract)
    payload["answers"] = {question.question_id: "It is false that " + question.expected_answers[0] for question in contract.questions}
    payload["thesis"] = "It is false that " + contract.thesis
    payload["sequence"] = ["It is false that " + value for value in contract.temporal_sequence]
    report = evaluate_verifier_payload(payload, contract)
    assert not report.publishable
    assert "viewer_contradicted_required_claim" in report.issues


def test_relationship_direction_is_not_bag_of_words():
    assert semantic_text_score("The planner selects the tool", "The tool selects the planner") == 0
    assert semantic_text_score("The planner selects the tool", "The planner picks the tool") > 0.8


def test_zero_frames_never_publish_in_balanced_mode():
    report = run_visual_verifier([], build_communication_contract(_ir()), strict=False, local_gate_passed=True, local_score=1)
    assert not report.publishable


def test_evidence_frame_budget_preserves_final_hold(tmp_path):
    frames = []
    for index in range(12):
        path = tmp_path / f"frame_{index:02d}.png"
        path.write_bytes(bytes([index]))
        frames.append(path)
    selected = select_evidence_frames(frames, limit=4)
    assert len(selected) == 4
    assert selected[0] == frames[0]
    assert selected[-2:] == frames[-2:]


def test_receipt_invalidates_when_final_video_changes(tmp_path):
    path = tmp_path / "final.mp4"
    frame = tmp_path / "frame.png"
    path.write_bytes(b"video")
    frame.write_bytes(b"frame")
    receipt = build_verification_receipt(SimpleNamespace(asset_path=str(path), renderer="remotion", metadata={}), {}, [frame], {"passed": True})
    assert validate_verification_receipt(receipt, path)
    path.write_bytes(b"different-final-video")
    assert not validate_verification_receipt(receipt, path)


def test_failed_final_local_evidence_cannot_inherit_preview_approval(tmp_path):
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"evidence")
    contract = build_communication_contract(_ir())
    outcome = direct_rendered_visual(_spec(), _asset(), "initial", ir=_ir(), contract=contract.to_dict(), render_candidate=lambda *_: None, evaluate_local_quality=lambda *_: _LocalQA(True, .9, []), extract_candidate_frames=lambda *_: [frame], strict=True, max_repair_rounds=0, provider_models=[("test", "vision")], vision_request=lambda *_: _payload(contract))
    qa = RenderedVisualQA("visual_001", "remotion", .1, False, ["remotion_render_final_frame_is_visually_empty"], [], "drop", {})
    merged = _merge_visual_director_quality(qa, outcome)
    assert not merged.passed
    assert "remotion_render_final_frame_is_visually_empty" in merged.issues
