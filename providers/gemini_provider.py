from __future__ import annotations

import json
from typing import Any

from google import genai
from google.genai import types

import config
from providers.base import BaseLLMProvider, LLMResponse, ToolCall


class GeminiProvider(BaseLLMProvider):
    def __init__(self) -> None:
        self._client = genai.Client(api_key=config.GEMINI_API_KEY)
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
        return types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=self._build_tools(tools),
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )

    def chat(
        self,
        messages: list[dict],
        tools: list[dict],
        system_prompt: str,
        stream_callback=None,
    ) -> LLMResponse:
        native_messages = self._neutral_to_native(messages)
        config_obj = self._build_config(tools, system_prompt)
        if stream_callback is not None:
            text_chunks: list[str] = []
            raw_response = []
            all_tool_calls: list[ToolCall] = []
            for chunk in self._client.models.generate_content_stream(
                model=self._model_name,
                contents=native_messages,
                config=config_obj,
            ):
                raw_response.append(chunk)
                if getattr(chunk, "text", None):
                    text_chunks.append(chunk.text)
                    stream_callback(chunk.text)
                all_tool_calls.extend(self._extract_tool_calls(chunk))
            return LLMResponse(text="".join(text_chunks), tool_calls=all_tool_calls, raw=raw_response)

        response = self._client.models.generate_content(
            model=self._model_name,
            contents=native_messages,
            config=config_obj,
        )
        text = getattr(response, "text", "") or ""
        tool_calls = self._extract_tool_calls(response)
        return LLMResponse(text=text, tool_calls=tool_calls, raw=response)

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
