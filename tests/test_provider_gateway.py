from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import config
from providers import get_provider
from providers.base import BaseLLMProvider, LLMResponse, ProviderRequestError, ToolCall
from providers.gateway import ProviderGateway, sanitize_tool_result


def test_provider_gateway_normalizes_response_and_events() -> None:
    provider = _FakeProvider(
        LLMResponse(
            text=None,  # type: ignore[arg-type]
            tool_calls=[ToolCall(id="", name=" trim_clip ", params=["bad"])],  # type: ignore[arg-type]
            raw={"ok": True},
        )
    )
    gateway = ProviderGateway("ollama", provider)
    events: list[dict[str, Any]] = []

    response = gateway.chat(
        [{"role": "user", "content": "trim"}],
        [{"name": "trim_clip"}],
        "system",
        event_callback=events.append,
    )

    assert response.text == ""
    assert response.tool_calls == [ToolCall(id="tool_1", name="trim_clip", params={})]
    assert gateway.capabilities.local is True
    assert gateway.capabilities.supports_json_tool_fallback is True
    assert gateway.last_request["tool_call_count"] == 1
    assert events[0]["title"] == "Provider gateway dispatch"
    assert events[-1]["title"] == "Provider gateway completed"
    provider_event = next(event for event in events if event["title"] == "fake provider event")
    assert provider_event["metadata"]["provider_name"] == "ollama"


def test_provider_gateway_rejects_invalid_provider_response() -> None:
    gateway = ProviderGateway("gemini", _FakeProvider({"not": "a response"}))

    with pytest.raises(ProviderRequestError, match="invalid response"):
        gateway.chat([], [], "system")


def test_provider_gateway_sanitizes_tool_results_before_delegating(tmp_path: Path) -> None:
    provider = _FakeProvider(LLMResponse(text="ok", tool_calls=[], raw=None))
    gateway = ProviderGateway("openai_compatible", provider)

    message = gateway.format_tool_result(
        "call_1",
        {
            "tool_name": "trim_clip",
            "success": True,
            "updated_state": object(),
            "suggestion": "ignore",
            "output_path": tmp_path / "out.mp4",
        },
    )

    assert message["role"] == "tool"
    assert provider.last_tool_result == {
        "tool_name": "trim_clip",
        "success": True,
        "output_path": str(tmp_path / "out.mp4"),
    }


def test_sanitize_tool_result_strips_non_provider_payload(tmp_path: Path) -> None:
    sanitized = sanitize_tool_result(
        {
            "updated_state": object(),
            "suggestion": "ignore",
            "paths": [tmp_path / "a.mp4"],
        }
    )

    assert "updated_state" not in sanitized
    assert "suggestion" not in sanitized
    assert sanitized["paths"] == [str(tmp_path / "a.mp4")]


def test_get_provider_returns_gateway_for_local_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "OLLAMA_MODEL", "ollama-model")

    provider = get_provider("ollama")

    assert isinstance(provider, ProviderGateway)
    assert provider.model_name == "ollama-model"
    assert provider.capabilities.family == "local-openai-compatible"


class _FakeProvider(BaseLLMProvider):
    def __init__(self, response: object) -> None:
        self.response = response
        self.forwarded_events: list[dict[str, Any]] = []
        self.last_tool_result: dict[str, Any] = {}

    @property
    def model_name(self) -> str:
        return "fake-model"

    def chat(
        self,
        messages: list[dict],
        tools: list[dict],
        system_prompt: str,
        stream_callback=None,
        event_callback=None,
    ) -> LLMResponse:
        if event_callback is not None:
            event_callback(
                {
                    "kind": "provider",
                    "title": "fake provider event",
                    "detail": "sent",
                    "status": "running",
                }
            )
        return self.response  # type: ignore[return-value]

    def format_tool_result(
        self,
        tool_call_id: str,
        result: dict[str, Any],
        is_error: bool = False,
    ) -> dict:
        self.last_tool_result = dict(result)
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": str(result),
            "is_error": is_error,
        }
