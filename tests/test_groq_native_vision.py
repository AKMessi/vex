import json
from pathlib import Path
from PIL import Image
import config
import httpx
import pytest
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
        payload = provider._build_payload(
            [{"role": "user", "content": "edit"}],
            [
                {
                    "name": "inspect",
                    "description": "inspect",
                    "parameters": {"type": "object"},
                }
            ],
            "system",
            stream=False,
        )
        assert payload["model"] == "qwen/qwen3.8-27b"
        assert payload["reasoning_format"] == "hidden"
        assert payload["parallel_tool_calls"] is False
        assert not provider._base_url.startswith("http://localhost")
    finally:
        provider.close()


def test_frame_repair_receives_pixels_and_applies_only_validated_scene_changes(
    tmp_path,
):
    spec = _spec()
    contract = build_communication_contract(_ir())
    payload = _payload(contract)
    payload["design"]["typography"] = 0.1
    report = evaluate_verifier_payload(
        payload, contract, provider="groq", model=config.GROQ_MODEL
    )
    frames = _frames(tmp_path, 2)
    target = spec["open_visual_program"]["elements"][0]["element_id"]

    def request(provider, model, prompt, supplied_frames):
        assert supplied_frames == frames
        assert "CURRENT PROGRAM" in prompt
        return {
            "counterexamples": [
                {
                    "frame_index": 1,
                    "target_id": target,
                    "observed": "heavy title",
                    "requirement": "readability",
                }
            ],
            "operations": [
                {"op": "set_style", "target_id": target, "style": {"font_weight": 850}}
            ],
        }

    plan, diagnostics = propose_frame_repairs(
        spec, report, frames, round_index=1, request=request
    )
    assert plan is not None and diagnostics["available"]
    application = apply_visual_repair(spec, plan, ir=_ir())
    assert application.passed
    assert (
        application.spec["open_visual_program"]["elements"][0]["style"]["font_weight"]
        == 850
    )


def test_native_repair_cannot_execute_code(tmp_path):
    contract = build_communication_contract(_ir())
    report = evaluate_verifier_payload(
        _payload(contract), contract, provider="groq", model=config.GROQ_MODEL
    )
    plan, diagnostics = propose_frame_repairs(
        _spec(),
        report,
        _frames(tmp_path, 1),
        round_index=1,
        request=lambda *_: {
            "counterexamples": [],
            "operations": [{"op": "exec", "target_id": "title", "code": "unsafe"}],
        },
    )
    assert plan is None
    assert not diagnostics["available"]


def test_native_frame_patch_cannot_apply_to_a_changed_source_program(tmp_path):
    from vex_visuals.open_visual_program import sign_open_visual_program
    spec=_spec()
    contract=build_communication_contract(_ir())
    report=evaluate_verifier_payload(_payload(contract),contract,provider="groq",model=config.GROQ_MODEL)
    target=spec["open_visual_program"]["elements"][0]["element_id"]
    plan,_=propose_frame_repairs(spec,report,_frames(tmp_path,1),round_index=1,request=lambda *_:{"counterexamples":[],"operations":[{"op":"set_style","target_id":target,"style":{"font_weight":850}}]})
    changed=dict(spec)
    changed["open_visual_program"]=sign_open_visual_program({**spec["open_visual_program"],"seed":spec["open_visual_program"]["seed"]+1})
    stale=apply_visual_repair(changed,plan,ir=_ir())
    assert not stale.changed
    assert any(item["reason"]=="native_frame_patch_source_changed" for item in stale.rejected_operations)


def test_groq_transport_reduces_output_to_account_quota_and_preserves_schema(
    monkeypatch,
):
    import providers.multimodal as multimodal

    monkeypatch.setattr(config, "GROQ_API_KEY", "test-key")
    calls = []

    def respond(request):
        payload = json.loads(request.content)
        calls.append(payload)
        if len(calls) == 1:
            return httpx.Response(
                429,
                json={
                    "error": {
                        "message": "Request too large for output tokens per minute. Limit 1000"
                    }
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                "usage": {"total_tokens": 20},
            },
        )

    original = httpx.Client
    monkeypatch.setattr(
        multimodal.httpx,
        "Client",
        lambda **kwargs: original(transport=httpx.MockTransport(respond), **kwargs),
    )
    schema = {"type": "object", "properties": {}, "additionalProperties": False}
    result = groq_completion("system", "prompt", max_tokens=8192, schema=schema)
    assert result["text"] == "{}"
    assert [call["max_completion_tokens"] for call in calls] == [8192, 872]
    assert calls[-1]["response_format"]["json_schema"]["schema"] == schema


def test_groq_transport_rejects_truncated_native_output(monkeypatch):
    import providers.multimodal as multimodal

    monkeypatch.setattr(config, "GROQ_API_KEY", "test-key")
    original = httpx.Client
    monkeypatch.setattr(
        multimodal.httpx,
        "Client",
        lambda **kwargs: original(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200,
                    json={
                        "choices": [
                            {
                                "message": {"content": "{broken"},
                                "finish_reason": "length",
                            }
                        ]
                    },
                )
            ),
            **kwargs,
        ),
    )
    with pytest.raises(ValueError, match="truncated"):
        groq_completion("system", "prompt")


def test_input_quota_retry_retains_all_frames_and_reduces_only_resolution(tmp_path,monkeypatch):
    import base64
    from io import BytesIO
    import providers.multimodal as multimodal
    monkeypatch.setattr(config,"GROQ_API_KEY","test-key")
    frames=[]
    for index in range(6):
        path=tmp_path/f"large_{index}.png"
        Image.new("RGB",(1280,720),(index*20,30,70)).save(path)
        frames.append(path)
    calls=[]
    def respond(request):
        calls.append(json.loads(request.content))
        if len(calls)==1:
            return httpx.Response(413,json={"error":{"message":"Request too large on input tokens per minute: Limit 7000, Requested 12000"}})
        return httpx.Response(200,json={"choices":[{"message":{"content":"{}"},"finish_reason":"stop"}]})
    original=httpx.Client
    monkeypatch.setattr(multimodal.httpx,"Client",lambda **kwargs:original(transport=httpx.MockTransport(respond),**kwargs))
    assert groq_completion("system","prompt",frames=frames)["text"]=="{}"
    dimensions=[]
    for call in calls:
        content=call["messages"][1]["content"]
        assert len(content)==4
        sheet=Image.open(BytesIO(base64.b64decode(content[1]["image_url"]["url"].split(",",1)[1])))
        dimensions.append(sheet.size)
    assert dimensions[1][0]<dimensions[0][0]
    assert dimensions[1][1]<dimensions[0][1]
