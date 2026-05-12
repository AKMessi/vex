from __future__ import annotations

import json
from typing import Any

import config
from providers import get_provider
from providers.base import ToolCall
from providers.openai_compatible_provider import OpenAICompatibleProvider


def test_get_provider_supports_local_aliases(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(config, "OPENAI_COMPAT_MODEL", "local-model")
    monkeypatch.setattr(config, "OLLAMA_MODEL", "ollama-model")
    monkeypatch.setattr(config, "LM_STUDIO_MODEL", "studio-model")
    monkeypatch.setattr(config, "LLAMA_CPP_MODEL", "llama-model")

    assert get_provider("openai_compatible").model_name == "local-model"
    assert get_provider("ollama").model_name == "ollama-model"
    assert get_provider("lm-studio").model_name == "studio-model"
    assert get_provider("llama.cpp").model_name == "llama-model"


def test_openai_compatible_provider_parses_native_tool_calls(monkeypatch) -> None:  # noqa: ANN001
    provider = OpenAICompatibleProvider("openai_compatible")
    provider._client = _FakeClient(
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "trim_clip",
                                    "arguments": json.dumps({"start": "0", "end": "30"}),
                                },
                            }
                        ],
                    }
                }
            ]
        }
    )

    response = provider.chat(
        [{"role": "user", "content": "trim first 30 seconds"}],
        [_tool_schema("trim_clip")],
        "system",
    )

    assert response.text == ""
    assert response.tool_calls == [ToolCall(id="call_1", name="trim_clip", params={"start": "0", "end": "30"})]
    assert provider._client.last_payload["tools"][0]["type"] == "function"


def test_openai_compatible_provider_accepts_json_text_tool_fallback() -> None:
    provider = OpenAICompatibleProvider("openai_compatible")
    provider._client = _FakeClient(
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "tool": "plan_encode",
                                "params": {"raw_request": "convert to mp4"},
                            }
                        )
                    }
                }
            ]
        }
    )

    response = provider.chat(
        [{"role": "user", "content": "convert to mp4"}],
        [_tool_schema("plan_encode")],
        "system",
    )

    assert response.text == ""
    assert response.tool_calls == [
        ToolCall(id="local_text_tool_1", name="plan_encode", params={"raw_request": "convert to mp4"})
    ]


def test_openai_compatible_provider_formats_tool_results() -> None:
    provider = OpenAICompatibleProvider("ollama")

    message = provider.format_tool_result(
        "call_1",
        {
            "tool_name": "trim_clip",
            "success": True,
            "message": "Trimmed.",
            "updated_state": object(),
            "suggestion": "ignored",
        },
    )

    payload = json.loads(message["content"])
    assert message["role"] == "tool"
    assert message["tool_call_id"] == "call_1"
    assert payload["tool_name"] == "trim_clip"
    assert payload["success"] is True
    assert "updated_state" not in payload
    assert "suggestion" not in payload


def test_validate_config_allows_local_provider_without_cloud_api_keys(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5-coder:14b")
    monkeypatch.setattr(config.shutil, "which", lambda _path: "ffmpeg")

    try:
        config.validate_config()
        assert config.PROVIDER == "ollama"
    finally:
        monkeypatch.delenv("PROVIDER", raising=False)
        monkeypatch.delenv("OLLAMA_MODEL", raising=False)
        config.reload_settings()


def _tool_schema(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "description": f"{name} tool",
        "parameters": {
            "type": "object",
            "properties": {
                "start": {"type": "string"},
                "end": {"type": "string"},
                "raw_request": {"type": "string"},
            },
            "required": [],
        },
    }


class _FakeClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.last_payload: dict[str, Any] = {}

    def post(self, _path: str, *, json: dict[str, Any]) -> "_FakeResponse":  # noqa: A002
        self.last_payload = json
        return _FakeResponse(self.payload)


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.status_code = 200
        self.text = ""

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload
