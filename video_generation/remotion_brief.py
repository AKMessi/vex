from __future__ import annotations

import json
import re
from dataclasses import replace

from providers import resolve_reasoning_model
from video_generation.models import ScriptPlan, VideoGenerationRequest
from video_generation.script_planner import WORDS_PER_MINUTE, split_script_sentences


BRIEF_VERSION = "remotion-brief-v1"
SYSTEM_PROMPT = """You write narration for a video director and motion-design compiler.
Return one JSON object with a beats array. Each beat has narration and focus strings.
Write exactly one short spoken sentence per beat. Each focus is a short contiguous
quote from its narration, naming the action or object the scene should make visible.
Follow the creative brief, format, audience and requested length. Open with a
specific premise, reveal the mechanism or story through concrete objects and actions,
and land the requested takeaway or CTA. Make successive beats complementary.
Use natural speech. Avoid instructions to the editor, stage directions, and generic
claims about filtering noise or tightening loops. Avoid invented dates, statistics,
measurements, testimonials and citations. Use digit values only if present in the brief.
For fiction, follow the brief and write the story as narration."""


def author_remotion_script(
    request: VideoGenerationRequest, offline_plan: ScriptPlan,
) -> tuple[ScriptPlan, list[str]]:
    if request.script:
        return offline_plan, []
    if request.gemini_production:
        from video_generation.remotion_gemini import director_model
        provider, model = "gemini", director_model()
    else:
        provider, model = resolve_reasoning_model()
    if not provider:
        return offline_plan, ["remotion_brief_used_offline_script"]

    from broll_intelligence import call_reasoning_model, extract_json_object

    word_target = max(10, round(request.duration_sec * request.voice_speed * WORDS_PER_MINUTE / 60))
    brief = {
        "brief": request.prompt,
        "title": request.title,
        "audience": request.audience,
        "cta": request.cta,
        "aspect": request.aspect,
        "style": request.style,
        "language": request.language,
        "target_words": word_target,
        "max_shots": 24,
    }
    try:
        response = call_reasoning_model(
            provider, model, SYSTEM_PROMPT, json.dumps(brief, ensure_ascii=False),
            max_attempts=1, timeout_sec=45,
        )
    except Exception:
        return offline_plan, ["remotion_brief_provider_unavailable_used_offline_script"]
    try:
        narration, focus = _validate_treatment(
            json.loads(extract_json_object(response)), request, word_target,
        )
    except (AttributeError, TypeError, ValueError):
        return offline_plan, ["remotion_brief_treatment_rejected_used_offline_script"]
    return replace(
        offline_plan,
        narration=narration,
        source=BRIEF_VERSION,
        shot_focus=focus,
        authoring_provider=f"{provider}/{model}",
    ), []


def _validate_treatment(
    value: object, request: VideoGenerationRequest, target_words: int,
) -> tuple[str, list[str]]:
    beats = value.get("beats") if isinstance(value, dict) else None
    if not isinstance(beats, list) or not 1 <= len(beats) <= 24:
        raise ValueError("Treatment requires a bounded list of beats.")
    narration: list[str] = []
    focus: list[str] = []
    for beat in beats:
        if not isinstance(beat, dict) or not all(isinstance(beat.get(key), str) for key in ("narration", "focus")):
            raise ValueError("Treatment beats require narration and focus.")
        sentence, focal_text = (" ".join(beat[key].split()) for key in ("narration", "focus"))
        if not 8 <= len(sentence) <= 260 or not re.search(r"[.!?]$", sentence) or len(split_script_sentences(sentence)) != 1:
            raise ValueError("Each treatment beat must be one spoken sentence.")
        if not 3 <= len(focal_text) <= 64 or focal_text.casefold() not in sentence.casefold():
            raise ValueError("Shot focus must be a short quote from its spoken sentence.")
        narration.append(sentence)
        focus.append(focal_text)
    script = " ".join(narration)
    if split_script_sentences(script) != narration or len(set(narration)) != len(narration):
        raise ValueError("Treatment beats must be distinct and match the spoken sentences.")
    word_count = len(script.split())
    if not max(5, target_words * 0.6) <= word_count <= max(16, target_words * 1.25):
        raise ValueError("Treatment falls outside the target narration budget.")
    numbers = re.compile(r"\d+(?:[,.]\d+)*")
    source = " ".join([request.title, request.prompt, request.cta])
    if set(numbers.findall(script)) - set(numbers.findall(source)):
        raise ValueError("Treatment contains a digit value missing from the brief.")
    return script, focus
