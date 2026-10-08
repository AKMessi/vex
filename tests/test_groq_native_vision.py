import json
from pathlib import Path
from PIL import Image
import config
from providers.groq_provider import GroqProvider
from providers.multimodal import pack_vision_images, groq_completion
from tests.test_visual_director import _spec
from tests.test_visual_communication_contract import _ir
from tests.test_visual_verifier import _payload
from vex_visuals.communication_contract import build_communication_contract
from vex_visuals.verifier import evaluate_verifier_payload
from vex_visuals.vision_repair import propose_frame_repairs
from vex_visuals.repair import apply_visual_repair


def _frames(tmp_path, count=8):
    paths = []
    for index in range(count):
        path = tmp_path / f"frame_{index}.png"
        Image.new("RGB", (320, 180), (index * 20, 30, 70)).save(path)
        paths.append(path)
    return paths


def test_native_vision_packs_all_frames_within_three_image_limit(tmp_path):
    images = pack_vision_images(_frames(tmp_path))
    assert len(images) == 3
    assert all(data.startswith(b"\x89PNG") for data in images)
    from io import BytesIO
    assert sum(Image.open(BytesIO(data)).height for data in images) == 8 * (180 + 28)


def test_groq_provider_uses_native_model_and_tool_protocol(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "test-key")
    provider = GroqProvider()
    try:
        payload = provider._build_payload([{"role": "user", "content": "edit"}], [{"name": "inspect", "description": "inspect", "parameters": {"type": "object"}}], "system", stream=False)
        assert payload["model"] == "qwen/qwen3.8-27b"
        assert payload["reasoning_format"] == "hidden"
        assert payload["parallel_tool_calls"] is False
        assert not provider._base_url.startswith("http://localhost")
    finally:
        provider.close()


def test_frame_repair_receives_pixels_and_applies_only_validated_scene_changes(tmp_path):
    spec = _spec()
    contract = build_communication_contract(_ir())
    payload = _payload(contract)
    payload["design"]["typography"] = .1
    report = evaluate_verifier_payload(payload, contract, provider="groq", model=config.GROQ_MODEL)
    frames = _frames(tmp_path, 2)
    target = spec["open_visual_program"]["elements"][0]["element_id"]
    def request(provider, model, prompt, supplied_frames):
        assert supplied_frames == frames
        assert "CURRENT PROGRAM" in prompt
        return {"counterexamples": [{"frame_index": 1, "target_id": target, "observed": "heavy title", "requirement": "readability"}], "operations": [{"op": "set_style", "target_id": target, "style": {"font_weight": 850}}]}
    plan, diagnostics = propose_frame_repairs(spec, report, frames, round_index=1, request=request)
    assert plan is not None and diagnostics["available"]
    application = apply_visual_repair(spec, plan, ir=_ir())
    assert application.passed
    assert application.spec["open_visual_program"]["elements"][0]["style"]["font_weight"] == 850


def test_native_repair_cannot_execute_code(tmp_path):
    contract = build_communication_contract(_ir())
    report = evaluate_verifier_payload(_payload(contract), contract, provider="groq", model=config.GROQ_MODEL)
    plan, diagnostics = propose_frame_repairs(_spec(), report, _frames(tmp_path, 1), round_index=1, request=lambda *_: {"counterexamples": [], "operations": [{"op": "exec", "target_id": "title", "code": "unsafe"}]})
    assert plan is None
    assert not diagnostics["available"]
