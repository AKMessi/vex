from __future__ import annotations

import config
from providers.base import BaseLLMProvider
from providers.gateway import ProviderGateway


def resolve_reasoning_model(provider: str | None = None, model: str | None = None) -> tuple[str, str]:
    """Resolve a configured text model without probing a service."""
    name = config.normalize_provider_name(provider or config.PROVIDER)
    cloud = {
        "gemini": (config.GEMINI_API_KEY, config.GEMINI_MODEL),
        "claude": (config.ANTHROPIC_API_KEY, config.CLAUDE_MODEL),
        "groq": (config.GROQ_API_KEY, config.GROQ_MODEL),
    }
    if name in cloud:
        key, default_model = cloud[name]
        return (name, model or default_model) if key else ("", "")
    if name in config.LOCAL_LLM_PROVIDERS:
        selected = model or config.local_llm_model(name)
        return (name, selected) if selected else ("", "")
    return "", ""


def get_provider(name: str) -> BaseLLMProvider:
    normalized = config.normalize_provider_name(name)
    if normalized == "groq":
        from providers.groq_provider import GroqProvider
        return ProviderGateway(normalized, GroqProvider())
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
        "Valid options: gemini, claude, groq, openai_compatible, ollama, lmstudio, llama_cpp."
    )
