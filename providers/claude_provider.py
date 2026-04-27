from __future__ import annotations

import json
import time
from typing import Any

import httpx
from anthropic import APIConnectionError, APIStatusError, Anthropic, InternalServerError, RateLimitError

import config
from providers.base import BaseLLMProvider, LLMResponse, ProviderRequestError, ToolCall


class ClaudeProvider(BaseLLMProvider):
    def __init__(self) -> None:
        self.client = Anthropic(
            api_key=config.ANTHROPIC_API_KEY,
            timeout=config.ANTHROPIC_TIMEOUT_SEC,
        )
        self._model_name = config.CLAUDE_MODEL

    @property
    def model_name(self) -> str:
        return self._model_name

    def _translate_tools(self, tools: list[dict]) -> list[dict]:
        return [
            {
                "name": schema["name"],
                "description": schema["description"],
                "input_schema": schema["parameters"],
            }
            for schema in tools
        ]

    def _translate_messages(self, messages: list[dict]) -> list[dict]:
        native_messages: list[dict[str, Any]] = []
        for message in messages:
            role = message["role"]
            if role in {"user", "assistant"} and "content" in message:
                native_messages.append({"role": role, "content": message["content"]})
            elif role == "assistant" and "tool_calls" in message:
                native_messages.append(
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": call["id"],
                                "name": call["name"],
                                "input": call.get("params", {}),
                            }
                            for call in message["tool_calls"]
                        ],
                    }
                )
            elif role == "tool":
                payload = json.loads(message["content"])
                native_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": message["tool_call_id"],
                                "content": json.dumps(payload),
                                "is_error": payload.get("is_error", False),
                            }
                        ],
                    }
                )
        return native_messages

    def _emit_provider_event(
        self,
        event_callback,
        *,
        title: str,
        detail: str,
        status: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if event_callback is None:
            return
        payload: dict[str, Any] = {
            "kind": "provider",
            "title": title,
            "detail": detail,
            "status": status,
        }
        if metadata:
            payload["metadata"] = metadata
        event_callback(payload)

    def _status_code_for_error(self, exc: Exception) -> int | None:
        status_code = getattr(exc, "status_code", None)
        if isinstance(status_code, int):
            return status_code
        response = getattr(exc, "response", None)
        response_status = getattr(response, "status_code", None)
        if isinstance(response_status, int):
            return response_status
        return None

    def _is_retryable_error(self, exc: Exception) -> bool:
        if isinstance(exc, ProviderRequestError):
            return False
        if isinstance(exc, (InternalServerError, APIConnectionError, RateLimitError, httpx.HTTPError, TimeoutError, OSError)):
            return True
        if isinstance(exc, APIStatusError):
            status_code = self._status_code_for_error(exc)
            if status_code in {408, 409, 425, 429}:
                return True
            if status_code is not None and status_code >= 500:
                return True
        message = str(exc).lower()
        retry_hints = (
            "internal error",
            "temporar",
            "timeout",
            "timed out",
            "connection reset",
            "service unavailable",
            "overloaded",
            "rate limit",
            "retry",
        )
        return any(hint in message for hint in retry_hints)

    def _summarize_exception(self, exc: Exception) -> str:
        status_code = self._status_code_for_error(exc)
        message = " ".join(str(exc).split()).strip()
        if status_code is not None:
            if message:
                return f"{status_code} {message}"
            return str(status_code)
        return message or exc.__class__.__name__

    def _raise_provider_error(
        self,
        exc: Exception,
        *,
        event_callback,
        stage_label: str,
        attempts: int,
        retryable: bool,
    ) -> None:
        if retryable:
            detail = (
                f"Claude hit a temporary error while {stage_label.lower()} after {attempts} attempt"
                f"{'' if attempts == 1 else 's'}. Please retry the command."
            )
        else:
            summary = self._summarize_exception(exc)
            detail = (
                f"Claude failed while {stage_label.lower()}"
                f"{': ' + summary if summary else '.'}"
            )
        self._emit_provider_event(
            event_callback,
            title="Claude request failed",
            detail=detail,
            status="error",
        )
        raise ProviderRequestError(detail) from exc

    def _with_retry(self, operation, *, event_callback, stage_label: str):
        max_attempts = max(1, int(config.LLM_REQUEST_MAX_RETRIES))
        for attempt in range(1, max_attempts + 1):
            try:
                return operation()
            except ProviderRequestError:
                raise
            except Exception as exc:  # noqa: BLE001
                retryable = self._is_retryable_error(exc)
                if retryable and attempt < max_attempts:
                    delay = float(config.LLM_RETRY_BASE_DELAY_SEC) * (2 ** (attempt - 1))
                    self._emit_provider_event(
                        event_callback,
                        title="Claude temporary error",
                        detail=(
                            f"{self._summarize_exception(exc)}. "
                            f"Retrying in {delay:.1f}s ({attempt + 1}/{max_attempts})."
                        ),
                        status="running",
                        metadata={"attempt": attempt + 1, "max_attempts": max_attempts},
                    )
                    time.sleep(delay)
                    continue
                self._raise_provider_error(
                    exc,
                    event_callback=event_callback,
                    stage_label=stage_label,
                    attempts=attempt,
                    retryable=retryable,
                )

    def chat(
        self,
        messages: list[dict],
        tools: list[dict],
        system_prompt: str,
        stream_callback=None,
        event_callback=None,
    ) -> LLMResponse:
        native_messages = self._translate_messages(messages)
        translated_tools = self._translate_tools(tools)
        self._emit_provider_event(
            event_callback,
            title="Sending request to Claude",
            detail=f"Model: {self._model_name}",
            status="running",
        )
        if stream_callback is not None:
            def run_stream() -> LLMResponse:
                text_chunks: list[str] = []
                tool_calls: list[ToolCall] = []
                announced_text = False
                try:
                    with self.client.messages.stream(
                        model=self._model_name,
                        system=system_prompt,
                        max_tokens=4096,
                        tools=translated_tools,
                        messages=native_messages,
                    ) as stream:
                        for event in stream:
                            if event.type == "content_block_delta" and getattr(event.delta, "text", None):
                                if not announced_text:
                                    self._emit_provider_event(
                                        event_callback,
                                        title="Streaming assistant response",
                                        detail="Receiving model output.",
                                        status="running",
                                    )
                                    announced_text = True
                                text_chunks.append(event.delta.text)
                                stream_callback(event.delta.text)
                        final_message = stream.get_final_message()
                except Exception as exc:  # noqa: BLE001
                    if text_chunks:
                        raise ProviderRequestError(
                            "Claude interrupted the response stream after partial output. Please retry the command."
                        ) from exc
                    raise
                for block in final_message.content:
                    if block.type == "tool_use":
                        tool_calls.append(ToolCall(id=block.id, name=block.name, params=block.input))
                if tool_calls:
                    self._emit_provider_event(
                        event_callback,
                        title="Model requested tools",
                        detail=", ".join(call.name for call in tool_calls[:4]),
                        status="info",
                    )
                else:
                    self._emit_provider_event(
                        event_callback,
                        title="Model finished response",
                        detail="No tool calls were returned.",
                        status="success",
                    )
                return LLMResponse(text="".join(text_chunks), tool_calls=tool_calls, raw=final_message)

            return self._with_retry(
                run_stream,
                event_callback=event_callback,
                stage_label="streaming the Claude response",
            )

        def run_once() -> LLMResponse:
            text_chunks: list[str] = []
            tool_calls: list[ToolCall] = []
            response = self.client.messages.create(
                model=self._model_name,
                system=system_prompt,
                max_tokens=4096,
                tools=translated_tools,
                messages=native_messages,
            )
            for block in response.content:
                if block.type == "text":
                    text_chunks.append(block.text)
                elif block.type == "tool_use":
                    tool_calls.append(ToolCall(id=block.id, name=block.name, params=block.input))
            if tool_calls:
                self._emit_provider_event(
                    event_callback,
                    title="Model requested tools",
                    detail=", ".join(call.name for call in tool_calls[:4]),
                    status="info",
                )
            else:
                self._emit_provider_event(
                    event_callback,
                    title="Model returned text response",
                    detail="Ready to finalize the turn.",
                    status="success",
                )
            return LLMResponse(text="".join(text_chunks), tool_calls=tool_calls, raw=response)

        return self._with_retry(
            run_once,
            event_callback=event_callback,
            stage_label="requesting Claude output",
        )

    def format_tool_result(
        self,
        tool_call_id: str,
        result: dict[str, Any],
        is_error: bool = False,
    ) -> dict:
        strip_keys = {"updated_state", "suggestion"}
        payload = {
            "tool_call_id": tool_call_id,
            "tool_name": result.get("tool_name", "tool_result"),
            "is_error": is_error,
            **{key: value for key, value in result.items() if key not in strip_keys},
        }
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps(payload),
        }
