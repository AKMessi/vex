from __future__ import annotations

import json
from typing import Any

from anthropic import Anthropic

import config
from providers.base import BaseLLMProvider, LLMResponse, ToolCall


class ClaudeProvider(BaseLLMProvider):
    def __init__(self) -> None:
        self.client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
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

    def chat(
        self,
        messages: list[dict],
        tools: list[dict],
        system_prompt: str,
        stream_callback=None,
    ) -> LLMResponse:
        native_messages = self._translate_messages(messages)
        translated_tools = self._translate_tools(tools)
        text_chunks: list[str] = []
        tool_calls: list[ToolCall] = []
        if stream_callback is not None:
            with self.client.messages.stream(
                model=self._model_name,
                system=system_prompt,
                max_tokens=4096,
                tools=translated_tools,
                messages=native_messages,
            ) as stream:
                for event in stream:
                    if event.type == "content_block_delta" and getattr(event.delta, "text", None):
                        text_chunks.append(event.delta.text)
                        stream_callback(event.delta.text)
                final_message = stream.get_final_message()
            for block in final_message.content:
                if block.type == "tool_use":
                    tool_calls.append(ToolCall(id=block.id, name=block.name, params=block.input))
            return LLMResponse(text="".join(text_chunks), tool_calls=tool_calls, raw=final_message)

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
        return LLMResponse(text="".join(text_chunks), tool_calls=tool_calls, raw=response)

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
