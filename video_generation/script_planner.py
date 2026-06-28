from __future__ import annotations

import re

from video_generation.models import ScriptPlan, VideoGenerationRequest


SCRIPT_PLANNER_VERSION = "audio-first-script-planner-v1"
WORDS_PER_MINUTE = 155.0

_STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "and",
    "are",
    "because",
    "but",
    "can",
    "for",
    "from",
    "how",
    "into",
    "that",
    "the",
    "this",
    "through",
    "when",
    "where",
    "with",
    "you",
    "your",
}


def build_script_plan(request: VideoGenerationRequest) -> ScriptPlan:
    narration = _normalize_script(request.script) if request.script else _script_from_prompt(request)
    title_source = _strip_instruction_phrases(request.prompt) if request.prompt else narration
    title = request.title or _title_from_text(title_source)
    design_direction = _design_direction(request)
    return ScriptPlan(
        title=title,
        narration=narration,
        design_direction=design_direction,
        source="user_script" if request.script else SCRIPT_PLANNER_VERSION,
        prompt=request.prompt,
        audience=request.audience,
        cta=request.cta,
    )


def estimated_script_duration(script: str, *, voice_speed: float = 1.0) -> float:
    words = len(_words(script))
    if words <= 0:
        return 0.0
    speed = max(float(voice_speed or 1.0), 0.2)
    return round((words / (WORDS_PER_MINUTE * speed)) * 60.0 + 0.65, 3)


def split_script_sentences(script: str) -> list[str]:
    pieces = [
        re.sub(r"\s+", " ", item).strip(" \t\r\n-")
        for item in re.split(r"(?<=[.!?])\s+|\n+", script)
    ]
    sentences = [piece for piece in pieces if len(_words(piece)) >= 2]
    if sentences:
        return sentences
    cleaned = _normalize_script(script)
    return [cleaned] if cleaned else []


def keyword_candidates(text: str, *, limit: int = 6) -> list[str]:
    seen: set[str] = set()
    keywords: list[str] = []
    for token in _words(text):
        key = token.lower().strip("'")
        if len(key) < 4 or key in _STOPWORDS or key in seen:
            continue
        seen.add(key)
        keywords.append(key)
        if len(keywords) >= limit:
            break
    return keywords


def _script_from_prompt(request: VideoGenerationRequest) -> str:
    topic = _strip_instruction_phrases(request.prompt)
    audience = f" for {request.audience}" if request.audience else ""
    cta = request.cta.strip()
    target_sentences = max(3, min(7, int(round(request.duration_sec / 7.0))))
    lower_topic = topic.lower()
    if "sparse attention" in lower_topic or (
        "attention" in lower_topic and "token" in lower_topic
    ):
        core = [
            "Dense attention starts with a wall of token links, where every token can look at everything.",
            "Sparse attention applies a mask that drops irrelevant links and keeps useful routes active.",
            "The result is a focused reasoning path, so compute follows signal instead of noise.",
        ]
    elif "video generation" in lower_topic or "hyperframes" in lower_topic:
        core = [
            f"{topic} starts with a concrete script, not a pile of loose visual ideas.",
            "The planner turns each beat into a signed visual contract with source-backed labels, relations, and motion.",
            "HyperFrames renders those contracts as distinct scenes, then frame QA rejects weak layouts before export.",
        ]
    else:
        core = [
            f"{topic} starts with a visible input{audience} and a messy bottleneck.",
            "The key mechanism filters noise, preserves the useful signal, and routes attention through fewer steps.",
            "The result is a clearer path the viewer can inspect without holding every detail in memory.",
            "A strong visual should show the before state, the intervention, and the resolved outcome in one readable sequence.",
            "Once the structure is visible, the takeaway becomes practical: improve the bottleneck, then tighten the loop.",
        ]
    if cta:
        core.append(cta if re.search(r"[.!?]$", cta) else f"{cta}.")
    selected = core[:target_sentences]
    if cta and cta not in selected[-1]:
        selected[-1] = cta if re.search(r"[.!?]$", cta) else f"{cta}."
    return _normalize_script(" ".join(selected))


def _design_direction(request: VideoGenerationRequest) -> str:
    style = request.style.replace("_", " ")
    if request.aspect == "portrait":
        frame = "vertical short-form frame with large captions and compressed visual hierarchy"
    elif request.aspect == "square":
        frame = "square social frame with centered kinetic typography"
    else:
        frame = "landscape explainer frame with cinematic breathing room"
    return (
        f"{style}; {frame}; audio timing is the source of truth; each beat should "
        "use a different visual system instead of repeating boxes and lines."
    )


def _title_from_text(text: str) -> str:
    keywords = keyword_candidates(text, limit=5)
    if not keywords:
        return "Generated Video"
    title = " ".join(word.capitalize() for word in keywords[:4])
    return title[:90].strip() or "Generated Video"


def _strip_instruction_phrases(prompt: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(prompt or "")).strip(" .")
    cleaned = re.sub(
        r"^(?:make|create|generate|build|produce|show|explain|visualize|illustrate|demonstrate)\s+"
        r"(?:(?:a|an)\s+)?(?:(?:video|hyperframes\s+video)\s+)?(?:about|on|for|how)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip(" .")
    if not cleaned:
        return "This idea"
    return cleaned[0].upper() + cleaned[1:]


def _normalize_script(script: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(script or "").replace("\ufeff", "")).strip()
    if cleaned and not re.search(r"[.!?]$", cleaned):
        cleaned += "."
    return cleaned


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9'%-]*", str(text or ""))
