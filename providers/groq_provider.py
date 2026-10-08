"""Groq's native tool/streaming API using the existing HTTP transport."""
from __future__ import annotations

import httpx
import config
from providers.base import ProviderRequestError
from providers.openai_compatible_provider import OpenAICompatibleProvider


class GroqProvider(OpenAICompatibleProvider):
    def __init__(self) -> None:
        if not config.GROQ_API_KEY:
            raise ProviderRequestError("GROQ_API_KEY is not configured")
        self._provider_name = "groq"
        self._display_name = "Groq"
        self._base_url = "https://api.groq.com/openai/v1"
        self._api_key = config.GROQ_API_KEY
        self._model_name = config.GROQ_MODEL
        self._timeout_sec = config.GROQ_TIMEOUT_SEC
        self._max_tokens = config.GROQ_MAX_TOKENS
        self._temperature = 0.7
        self._client = httpx.Client(base_url=self._base_url, timeout=self._timeout_sec, headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"})

    def _build_payload(self, messages, tools, system_prompt, *, stream):
        payload = super()._build_payload(messages, tools, system_prompt, stream=stream)
        payload["reasoning_effort"] = config.GROQ_REASONING_EFFORT
        payload["reasoning_format"] = "hidden"
        if payload.get("tools"):
            payload["parallel_tool_calls"] = False
        return payload

    def close(self) -> None:
        self._client.close()
