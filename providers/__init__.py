from __future__ import annotations

from providers.base import BaseLLMProvider


def get_provider(name: str) -> BaseLLMProvider:
    normalized = (name or "gemini").strip().lower()
    if normalized == "gemini":
        from providers.gemini_provider import GeminiProvider

        return GeminiProvider()
    if normalized == "claude":
        from providers.claude_provider import ClaudeProvider

        return ClaudeProvider()
    raise ValueError(f"Unknown provider {name!r}. Valid options: gemini, claude.")
