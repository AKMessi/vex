"""Native image+text requests, with bounded image packing and JSON validation."""
from __future__ import annotations

import base64
import io
import json
import math
from pathlib import Path
from typing import Any

import httpx
from PIL import Image, ImageDraw
import config


def pack_vision_images(frame_paths: list[Path], *, max_images: int = 3) -> list[bytes]:
    if not frame_paths:
        return []
    if len(frame_paths) > 24:
        raise ValueError("Vision evidence exceeds the 24-frame budget")
    groups = min(len(frame_paths), max(1, int(max_images)))
    result = []
    for group in range(groups):
        start = group * len(frame_paths) // groups
        end = (group + 1) * len(frame_paths) // groups
        frames = []
        for index, path in enumerate(frame_paths[start:end], start):
            with Image.open(path) as source:
                frame = source.convert("RGB")
                frame.thumbnail((960, 720), Image.Resampling.LANCZOS)
                frames.append((index, frame.copy()))
        width = max(frame.width for _, frame in frames)
        height = sum(frame.height + 28 for _, frame in frames)
        sheet = Image.new("RGB", (width, height), "#111111")
        draw = ImageDraw.Draw(sheet)
        y = 0
        for index, frame in frames:
            draw.text((8, y + 5), f"QA FRAME {index + 1:02d} / {len(frame_paths):02d} (chronological)", fill="white")
            sheet.paste(frame, (0, y + 28))
            y += frame.height + 28
        stream = io.BytesIO()
        sheet.save(stream, format="PNG")
        result.append(stream.getvalue())
    if sum(map(len, result)) > 18 * 1024 * 1024:
        raise ValueError("Vision evidence exceeds the 18 MB request budget")
    return result


def groq_completion(system: str, prompt: str, *, model: str = "", frames: list[Path] | None = None, json_output: bool = True, timeout_sec: float | None = None, max_tokens: int | None = None, reasoning_effort: str | None = None) -> dict[str, Any]:
    if not config.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not configured")
    content: Any = prompt
    if frames:
        images = pack_vision_images(frames)
        content = [{"type": "text", "text": prompt + "\nImages contain numbered QA frames. Read them top to bottom, image by image; preserve chronological order."}]
        content.extend({"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(data).decode("ascii")}} for data in images)
    payload = {
        "model": model or config.GROQ_MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
        "max_completion_tokens": min(int(max_tokens or config.GROQ_MAX_TOKENS), 16384),
        "temperature": 0.2,
        "reasoning_effort": reasoning_effort or config.GROQ_REASONING_EFFORT,
        "reasoning_format": "hidden",
    }
    if json_output:
        payload["response_format"] = {"type": "json_object"}
    from vex_runtime.visual_run import model_budget
    with model_budget(system+prompt,payload["max_completion_tokens"],len(frames or [])) as usage:
        with httpx.Client(timeout=timeout_sec or config.GROQ_TIMEOUT_SEC) as client:
            response = client.post("https://api.groq.com/openai/v1/chat/completions", headers={"Authorization": "Bearer " + config.GROQ_API_KEY}, json=payload)
            response.raise_for_status()
            result = response.json()
            usage.update(result.get("usage") or {})
    choices = result.get("choices") or []
    if not choices or choices[0].get("finish_reason") == "length":
        raise ValueError("Groq returned missing or truncated output")
    text = str((choices[0].get("message") or {}).get("content") or "")
    return {"text": text, "usage": dict(result.get("usage") or {}), "model": str(result.get("model") or payload["model"])}


def request_visual_json(provider: str, model: str, prompt: str, frames: list[Path]) -> dict[str, Any]:
    if provider == "groq":
        return _request_visual_json(provider,model,prompt,frames)
    from vex_runtime.visual_run import model_budget
    with model_budget(prompt,8192,len(frames)) as usage:
        value = _request_visual_json(provider,model,prompt,frames)
        usage.update(value.pop("_usage",{}))
        return value


def _request_visual_json(provider: str, model: str, prompt: str, frames: list[Path]) -> dict[str, Any]:
    system = "You are an independent visual evaluator. Inspect actual pixels; return one JSON object only."
    usage = {}
    if provider == "groq":
        result = groq_completion(system, prompt, model=model, frames=frames)
        text = result["text"]
    elif provider == "gemini":
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=config.GEMINI_API_KEY, http_options=config.google_genai_http_options(retry_attempts=1))
        try:
            response = client.models.generate_content(model=model, contents=[types.Part.from_text(text=prompt), *[types.Part.from_bytes(data=path.read_bytes(), mime_type="image/png") for path in frames]], config=config.build_gemini_generation_config(system, model_name=model))
            text = getattr(response, "text", "") or ""
            total = getattr(getattr(response,"usage_metadata",None),"total_token_count",None)
            if total is not None:
                usage["total_tokens"] = total
        finally:
            client.close()
    elif provider == "claude":
        from anthropic import Anthropic
        with Anthropic(api_key=config.ANTHROPIC_API_KEY, timeout=config.ANTHROPIC_TIMEOUT_SEC, max_retries=0) as client:
            content = [{"type": "text", "text": prompt}, *[{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(path.read_bytes()).decode("ascii")}} for path in frames]]
            response = client.messages.create(model=model, max_tokens=8192, system=system, messages=[{"role": "user", "content": content}])
            text = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
            usage["total_tokens"] = response.usage.input_tokens + response.usage.output_tokens
    else:
        raise ValueError(f"Unsupported native vision provider: {provider}")
    start, end = text.find("{"), text.rfind("}")
    value = json.loads(text[start:end + 1]) if start >= 0 and end > start else None
    if not isinstance(value, dict):
        raise ValueError("Native vision model did not return a JSON object")
    if provider != "groq":
        value["_usage"] = usage
    return value
