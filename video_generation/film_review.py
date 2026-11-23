"""Pixel review through Vex's OpenAI-compatible transport for DeepSeek Flash."""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from providers.openai_compatible_provider import OpenAICompatibleProvider
from vex_runtime.visual_run import model_budget
from video_generation.film_direction import FilmDirection
from video_generation.film_model import deepseek_film_json, uses_deepseek_json
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

    hosted = uses_deepseek_json(direction.provider, direction.model)
    client = None if hosted else OpenAICompatibleProvider(direction.provider)

    def request(provider, model, prompt, frames, *, schema):
        content = [{"type": "text", "text": prompt + "\nReturn JSON with this structure:\n" + json.dumps(schema)}]
        for frame in frames:
            encoded = base64.b64encode(frame.read_bytes()).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}})
        system = "Review actual film frames as an independent motion-design editor. Respond with a JSON object."
        messages = [{"role": "user", "content": content}]
        if hosted:
            response = deepseek_film_json(messages, system, content[0]["text"], max_tokens=4096,
                                          thinking=False, image_count=len(frames), model=model)
        else:
            with model_budget(prompt, 4096, len(frames)):
                response = client.chat(messages, [], system).text
        return json.loads(extract_json_object(response))

    try:
        return review_delivery(
            output, delivery, shot_sources, provider=direction.provider,
            model=direction.model, visual_request=request,
            shot_intents=dict(zip(shot_sources, (shot.intent for shot in direction.shots), strict=True)),
        )
    finally:
        if client is not None:
            client.close()
