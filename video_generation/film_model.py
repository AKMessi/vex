"""Hosted structured-output transport for film direction and review."""
from __future__ import annotations

from urllib.parse import urlsplit

import config
from providers.openai_compatible_provider import OpenAICompatibleProvider
from vex_runtime.visual_run import model_budget


def uses_deepseek_json(provider: str, model: str) -> bool:
    return (
        provider == "openai_compatible" and model in {"deepseek-flash", "deepseek-v4-pro"}
        and urlsplit(config.local_llm_base_url(provider)).hostname == "api.deepseek.com"
    )


def deepseek_film_json(
    messages: list[dict], system: str, prompt: str, *,
    max_tokens: int, thinking: bool, image_count: int = 0, model: str | None = None,
) -> str:
    client = OpenAICompatibleProvider("openai_compatible", model=model)
    try:
        with model_budget(system + prompt, max_tokens, image_count, provider="openai_compatible",
                          model=model or client.model_name) as usage:
            response = client.complete_deepseek_json(messages, system, max_tokens=max_tokens, thinking=thinking)
            usage.update(response.raw.get("usage") or {})
        if response.raw["choices"][0].get("finish_reason") == "length":
            raise ValueError(f"DeepSeek reached the {max_tokens}-token output limit.")
        if not response.text.strip():
            raise ValueError("DeepSeek returned an empty JSON response.")
        return response.text
    finally:
        client.close()
