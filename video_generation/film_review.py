"""Pixel review through Vex's OpenAI-compatible transport for DeepSeek Flash."""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from providers.openai_compatible_provider import OpenAICompatibleProvider
from vex_runtime.visual_run import model_budget
from video_generation.film_direction import FilmDirection
from video_generation.remotion_gemini import review_delivery


DEEPSEEK_VISION_MODELS = frozenset({
    "deepseek-flash", "deepseek-v4-flash", "deepseek-v4-flash-vision-exp", "deepseek-v4.1-flash",
})


def supports_film_vision(direction: FilmDirection) -> bool:
    return direction.provider == "openai_compatible" and direction.model.lower() in DEEPSEEK_VISION_MODELS


def review_film(
    output: Path, delivery: dict[str, Any], direction: FilmDirection, shot_sources: dict[str, str],
) -> dict[str, Any]:
    from broll_intelligence import extract_json_object

    client = OpenAICompatibleProvider(direction.provider)

    def request(provider, model, prompt, frames, *, schema):
        content = [{"type": "text", "text": prompt + "\nReturn JSON with this structure:\n" + json.dumps(schema)}]
        for frame in frames:
            encoded = base64.b64encode(frame.read_bytes()).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}})
        with model_budget(prompt, 4096, len(frames)):
            response = client.chat(
                [{"role": "user", "content": content}], [],
                "Review actual film frames as an independent motion-design editor. Respond with a JSON object.",
            )
        return json.loads(extract_json_object(response.text))

    try:
        return review_delivery(
            output, delivery, shot_sources, provider=direction.provider,
            model=direction.model, visual_request=request,
        )
    finally:
        client.close()
