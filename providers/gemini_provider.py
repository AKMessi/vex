from __future__ import annotations

import json
import time
from typing import Any

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

import config
from providers.base import BaseLLMProvider, LLMResponse, ProviderRequestError, ToolCall


class GeminiProvider(BaseLLMProvider):
    def __init__(self) -> None:
        self._client = genai.Client(
            api_key=config.GEMINI_API_KEY,
            http_options=config.google_genai_http_options(),
        )
        self._model_name = config.GEMINI_MODEL
        self._tool_call_parts: dict[str, types.Part] = {}

    @property
    def model_name(self) -> str:
        return self._model_name

    def _build_tools(self, tools: list[dict]) -> list[types.Tool]:
        declarations: list[types.FunctionDeclaration] = []
        for schema in tools:
            declarations.append(
                types.FunctionDeclaration(
                    name=schema["name"],
                    description=schema["description"],
                    parameters_json_schema=self._sanitize_schema(schema["parameters"]),
                )
            )
        return [types.Tool(function_declarations=declarations)]

    def _sanitize_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        allowed_keys = {
            "type",
            "format",
            "description",
            "nullable",
            "enum",
            "items",
            "properties",
            "required",
        }
        sanitized: dict[str, Any] = {}
        for key, value in schema.items():
            if key not in allowed_keys:
                continue
            if key == "properties" and isinstance(value, dict):
                sanitized[key] = {
                    prop_name: self._sanitize_schema(prop_schema)
                    for prop_name, prop_schema in value.items()
                    if isinstance(prop_schema, dict)
                }
            elif key == "items" and isinstance(value, dict):
                sanitized[key] = self._sanitize_schema(value)
            else:
                sanitized[key] = value
        return sanitized

    def _extract_tool_calls(self, response: Any) -> list[ToolCall]:
        extracted: list[ToolCall] = []
        content = None
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            content = getattr(candidates[0], "content", None)
        elif hasattr(response, "content"):
            content = getattr(response, "content", None)

        if content is not None:
            for index, part in enumerate(getattr(content, "parts", []) or [], start=1):
                function_call = getattr(part, "function_call", None)
                if function_call is None:
                    continue
                call_id = getattr(function_call, "id", None) or f"gemini_{index}"
                self._tool_call_parts[call_id] = part
                extracted.append(
                    ToolCall(
                        id=call_id,
                        name=function_call.name,
                        params=dict(function_call.args),
                    )
                )
            if extracted:
                return extracted

        for index, function_call in enumerate(getattr(response, "function_calls", []) or [], start=1):
            call_id = getattr(function_call, "id", None) or f"gemini_{index}"
            extracted.append(
                ToolCall(
                    id=call_id,
                    name=function_call.name,
                    params=dict(function_call.args),
                )
            )
        return extracted

    def _neutral_to_native(self, messages: list[dict]) -> list[types.Content]:
        native_messages: list[types.Content] = []
        for message in messages:
            role = message["role"]
            if role in {"user", "assistant"} and "content" in message:
                native_messages.append(
                    types.Content(
                        role="model" if role == "assistant" else "user",
                        parts=[types.Part.from_text(text=message["content"])],
                    )
                )
            elif role == "assistant" and "tool_calls" in message:
                native_messages.append(
                    types.Content(
                        role="model",
                        parts=[
                            self._tool_call_parts.get(call["id"])
                            or types.Part.from_function_call(
                                name=call["name"],
                                args=call.get("params", {}),
                            )
                            for call in message["tool_calls"]
                        ]
                    )
                )
            elif role == "tool":
                payload = json.loads(message["content"])
                native_messages.append(
                    types.Content(
                        role="tool",
                        parts=[
                            types.Part.from_function_response(
                                name=payload.get("tool_name", "tool_result"),
                                response=payload,
                            )
                        ],
                    )
                )
        return native_messages

    def _build_config(self, tools: list[dict], system_prompt: str) -> types.GenerateContentConfig:
        return config.build_gemini_generation_config(
            system_prompt,
            model_name=self._model_name,
            tools=self._build_tools(tools),
        )

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
        if isinstance(exc, (genai_errors.ServerError, httpx.HTTPError, TimeoutError, OSError)):
            return True
        if isinstance(exc, (genai_errors.ClientError, genai_errors.APIError)):
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
                f"Gemini hit a temporary error while {stage_label.lower()} after {attempts} attempt"
                f"{'' if attempts == 1 else 's'}. Please retry the command."
            )
        else:
            summary = self._summarize_exception(exc)
            detail = (
                f"Gemini failed while {stage_label.lower()}"
                f"{': ' + summary if summary else '.'}"
            )
        self._emit_provider_event(
            event_callback,
            title="Gemini request failed",
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
                        title="Gemini temporary error",
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
        native_messages = self._neutral_to_native(messages)
        config_obj = self._build_config(tools, system_prompt)
        self._emit_provider_event(
            event_callback,
            title="Sending request to Gemini",
            detail=f"Model: {self._model_name}",
            status="running",
        )
        if stream_callback is not None:
            def run_stream() -> LLMResponse:
                text_chunks: list[str] = []
                raw_response = []
                all_tool_calls: list[ToolCall] = []
                announced_text = False
                try:
                    for chunk in self._client.models.generate_content_stream(
                        model=self._model_name,
                        contents=native_messages,
                        config=config_obj,
                    ):
                        raw_response.append(chunk)
                        if getattr(chunk, "text", None):
                            if not announced_text:
                                self._emit_provider_event(
                                    event_callback,
                                    title="Streaming assistant response",
                                    detail="Receiving model output.",
                                    status="running",
                                )
                                announced_text = True
                            text_chunks.append(chunk.text)
                            stream_callback(chunk.text)
                        all_tool_calls.extend(self._extract_tool_calls(chunk))
                except Exception as exc:  # noqa: BLE001
                    if text_chunks:
                        raise ProviderRequestError(
                            "Gemini interrupted the response stream after partial output. Please retry the command."
                        ) from exc
                    raise
                if all_tool_calls:
                    self._emit_provider_event(
                        event_callback,
                        title="Model requested tools",
                        detail=", ".join(call.name for call in all_tool_calls[:4]),
                        status="info",
                    )
                else:
                    self._emit_provider_event(
                        event_callback,
                        title="Model finished response",
                        detail="No tool calls were returned.",
                        status="success",
                    )
                return LLMResponse(text="".join(text_chunks), tool_calls=all_tool_calls, raw=raw_response)

            return self._with_retry(
                run_stream,
                event_callback=event_callback,
                stage_label="streaming the Gemini response",
            )

        def run_once() -> LLMResponse:
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=native_messages,
                config=config_obj,
            )
            text = getattr(response, "text", "") or ""
            tool_calls = self._extract_tool_calls(response)
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
            return LLMResponse(text=text, tool_calls=tool_calls, raw=response)

        return self._with_retry(
            run_once,
            event_callback=event_callback,
            stage_label="requesting Gemini output",
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
