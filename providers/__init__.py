from __future__ import annotations

import config
from providers.base import BaseLLMProvider
from providers.gateway import ProviderGateway


def get_provider(name: str) -> BaseLLMProvider:
    normalized = config.normalize_provider_name(name)
    if normalized == "gemini":
        from providers.gemini_provider import GeminiProvider

        return ProviderGateway(normalized, GeminiProvider())
    if normalized == "claude":
        from providers.claude_provider import ClaudeProvider

        return ProviderGateway(normalized, ClaudeProvider())
    if normalized in config.LOCAL_LLM_PROVIDERS:
        from providers.openai_compatible_provider import OpenAICompatibleProvider

        return ProviderGateway(normalized, OpenAICompatibleProvider(normalized))
    raise ValueError(
        f"Unknown provider {name!r}. "
        "Valid options: gemini, claude, openai_compatible, ollama, lmstudio, llama_cpp."
    )
