from __future__ import annotations

import time
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import config
from providers.base import BaseLLMProvider, LLMResponse, ProviderRequestError, ToolCall, emit_event_safely


@dataclass(frozen=True)
class ProviderCapabilities:
    provider_name: str
    model_name: str
    supports_tools: bool
    supports_streaming: bool
    supports_json_tool_fallback: bool
    local: bool
    family: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProviderGateway(BaseLLMProvider):
    def __init__(self, provider_name: str, provider: BaseLLMProvider) -> None:
        self.provider_name = config.normalize_provider_name(provider_name)
        self.provider = provider
        self.capabilities = provider_capabilities(self.provider_name, provider.model_name)
        self.last_request: dict[str, Any] = {}

    @property
    def model_name(self) -> str:
        return self.provider.model_name

    def chat(
        self,
        messages: list[dict],
        tools: list[dict],
        system_prompt: str,
        stream_callback=None,
        event_callback=None,
    ) -> LLMResponse:
        if not isinstance(messages, list):
            raise ProviderRequestError("Provider gateway expected messages to be a list.")
        if not isinstance(tools, list):
            raise ProviderRequestError("Provider gateway expected tools to be a list.")

        request_id = f"llm_{uuid.uuid4().hex[:12]}"
        started = time.monotonic()
        self._emit_gateway_event(
            event_callback,
            request_id=request_id,
            title="Provider gateway dispatch",
            detail=f"{self.provider_name} / {self.model_name}",
            status="running",
            metadata={
                "message_count": len(messages),
                "tool_count": len(tools),
                "capabilities": self.capabilities.to_dict(),
            },
        )
        try:
            response = self.provider.chat(
                messages,
                tools,
                system_prompt,
                stream_callback=stream_callback,
                event_callback=self._wrap_event_callback(event_callback, request_id),
            )
            normalized = self._normalize_response(response)
        except ProviderRequestError:
            self._emit_gateway_event(
                event_callback,
                request_id=request_id,
                title="Provider gateway failed",
                detail=f"{self.provider_name} returned a provider error.",
                status="error",
            )
            raise
        except Exception as exc:  # noqa: BLE001
            detail = f"{self.provider_name} failed before returning a valid response: {exc}"
            self._emit_gateway_event(
                event_callback,
                request_id=request_id,
                title="Provider gateway failed",
                detail=detail,
                status="error",
            )
            raise ProviderRequestError(detail) from exc

        elapsed = max(time.monotonic() - started, 0.0)
        self.last_request = {
            "request_id": request_id,
            "provider_name": self.provider_name,
            "model_name": self.model_name,
            "duration_sec": elapsed,
            "tool_call_count": len(normalized.tool_calls),
            "text_length": len(normalized.text),
        }
        self._emit_gateway_event(
            event_callback,
            request_id=request_id,
            title="Provider gateway completed",
            detail=(
                f"{len(normalized.tool_calls)} tool call"
                f"{'' if len(normalized.tool_calls) == 1 else 's'}, "
                f"{len(normalized.text)} text chars."
            ),
            status="success",
            metadata={"duration_sec": elapsed},
        )
        return normalized

    def format_tool_result(
        self,
        tool_call_id: str,
        result: dict[str, Any],
        is_error: bool = False,
    ) -> dict:
        message = self.provider.format_tool_result(
            tool_call_id,
            sanitize_tool_result(result),
            is_error=is_error,
        )
        if not isinstance(message, dict) or message.get("role") != "tool":
            raise ProviderRequestError("Provider returned an invalid tool-result message.")
        return message

    def _normalize_response(self, response: object) -> LLMResponse:
        if not isinstance(response, LLMResponse):
            raise ProviderRequestError("Provider returned an invalid response object.")
        tool_calls: list[ToolCall] = []
        used_ids: set[str] = set()
        for index, item in enumerate(response.tool_calls or [], start=1):
            normalized = _normalize_tool_call(item, index=index, used_ids=used_ids)
            used_ids.add(normalized.id)
            tool_calls.append(normalized)
        return LLMResponse(
            text=str(response.text or ""),
            tool_calls=tool_calls,
            raw=response.raw,
        )

    def _wrap_event_callback(self, event_callback, request_id: str):
        if event_callback is None:
            return None

        def emit(event: object) -> None:
            if not isinstance(event, dict):
                emit_event_safely(event_callback, event)
                return
            payload = dict(event)
            metadata = dict(payload.get("metadata") or {})
            metadata.update(
                {
                    "request_id": request_id,
                    "provider_name": self.provider_name,
                    "model_name": self.model_name,
                }
            )
            payload["metadata"] = metadata
            emit_event_safely(event_callback, payload)

        return emit

    def _emit_gateway_event(
        self,
        event_callback,
        *,
        request_id: str,
        title: str,
        detail: str,
        status: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if event_callback is None:
            return
        payload_metadata = {
            "request_id": request_id,
            "provider_name": self.provider_name,
            "model_name": self.model_name,
        }
        payload_metadata.update(metadata or {})
        emit_event_safely(
            event_callback,
            {
                "kind": "provider",
                "title": title,
                "detail": detail,
                "status": status,
                "metadata": payload_metadata,
            },
        )


def provider_capabilities(provider_name: str, model_name: str) -> ProviderCapabilities:
    normalized = config.normalize_provider_name(provider_name)
    local = normalized in config.LOCAL_LLM_PROVIDERS
    family = "local-openai-compatible" if local else normalized
    return ProviderCapabilities(
        provider_name=normalized,
        model_name=str(model_name or ""),
        supports_tools=True,
        supports_streaming=True,
        supports_json_tool_fallback=local,
        local=local,
        family=family,
    )


def sanitize_tool_result(result: Mapping[str, Any]) -> dict[str, Any]:
    strip_keys = {"updated_state", "suggestion"}
    return {
        str(key): _json_safe(value)
        for key, value in dict(result or {}).items()
        if key not in strip_keys
    }


def _normalize_tool_call(item: object, *, index: int, used_ids: set[str]) -> ToolCall:
    if not isinstance(item, ToolCall):
        raise ProviderRequestError("Provider returned an invalid tool call.")
    name = str(item.name or "").strip()
    if not name:
        raise ProviderRequestError("Provider returned a tool call without a name.")
    params = item.params if isinstance(item.params, dict) else {}
    base_id = str(item.id or f"tool_{index}").strip() or f"tool_{index}"
    call_id = base_id
    suffix = 2
    while call_id in used_ids:
        call_id = f"{base_id}_{suffix}"
        suffix += 1
    return ToolCall(
        id=call_id,
        name=name,
        params=dict(params),
    )


def _json_safe(value: object) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return str(value)
