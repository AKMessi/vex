from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from video_generation.models import Beat, BeatGraph, ScriptPlan, VideoGenerationRequest
from video_generation.script_planner import keyword_candidates


DIRECTOR_CREW_VERSION = "hyperframes-director-crew-v1"
SCRIPT_DIRECTOR_VERSION = "script-director-v1"

_GENERIC_SCRIPT_PATTERNS = {
    "visible input",
    "messy bottleneck",
    "useful signal",
    "fewer steps",
    "holding every detail",
    "route attention",
}

_GENERIC_VISUAL_PATTERNS = {
    "abstract concept field",
    "boxes and lines",
    "generic filler",
    "decorative motion",
    "fake metric",
    "unearned chart",
}

_LOW_VALUE_REQUIRED_OBJECTS = {
    "after",
    "again",
    "before",
    "better",
    "clear",
    "concept",
    "context",
    "create",
    "each",
    "easier",
    "explain",
    "explains",
    "final",
    "first",
    "generate",
    "generated",
    "helps",
    "idea",
    "instead",
    "make",
    "measured",
    "naming",
    "number",
    "once",
    "outcome",
    "payoff",
    "proof",
    "practical",
    "problem",
    "promise",
    "protected",
    "result",
    "scene",
    "show",
    "shows",
    "should",
    "state",
    "takeaway",
    "then",
    "through",
    "video",
    "viewer",
    "visible",
    "weak",
}

_LOW_VALUE_REQUIRED_OBJECT_STEMS = (
    "add",
    "attach",
    "becom",
    "build",
    "choos",
    "compil",
    "fad",
    "generat",
    "mak",
    "nam",
    "render",
    "show",
    "start",
    "turn",
)


@dataclass(frozen=True)
class ProductionBrief:
    title: str
    topic: str
    promise: str
    audience: str
    style: str
    hook: str
    story_arc: list[str]
    visual_language: list[str]
    forbidden_patterns: list[str]
    quality_bars: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BeatContract:
    beat_id: str
    index: int
    objective: str
    viewer_question: str
    visual_job: str
    required_objects: list[str]
    required_relation: str
    motion_intent: str
    transition_intent: str
    anti_filler_rule: str
    evidence_text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DirectorPackage:
    version: str
    script_director_version: str
    script_rewrite_applied: bool
    brief: ProductionBrief
    beat_contracts: list[BeatContract]
    portfolio_policy: dict[str, Any] = field(default_factory=dict)
    crew_roles: list[str] = field(default_factory=list)

    def beat_contract(self, beat_id: str) -> BeatContract | None:
        for contract in self.beat_contracts:
            if contract.beat_id == beat_id:
                return contract
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "script_director_version": self.script_director_version,
            "script_rewrite_applied": self.script_rewrite_applied,
            "brief": self.brief.to_dict(),
            "beat_contracts": [contract.to_dict() for contract in self.beat_contracts],
            "portfolio_policy": dict(self.portfolio_policy),
            "crew_roles": list(self.crew_roles),
        }


def direct_script_plan(
    request: VideoGenerationRequest,
    plan: ScriptPlan,
) -> ScriptPlan:
    """Repair prompt-only generated scripts before timing and visual planning."""

    if request.script:
        return plan
    if not _script_needs_director_rewrite(plan):
        return plan
    topic = _topic_from_request(request, plan)
    keywords = _keywords_for(topic, request.prompt, limit=6)
    if _is_sparse_attention(topic):
        narration = (
            "Dense attention starts as a wall of token links, where every token can inspect every other token. "
            "Sparse attention draws a mask over that wall and removes links that add noise. "
            "The surviving routes form a focused reasoning path, so compute follows evidence instead of noise. "
            "The final frame should make the compression visible: fewer active routes, clearer reasoning, and no lost context."
        )
    elif _is_retrieval_augmented_generation(topic):
        narration = (
            "Retrieval augmented generation starts with a question beside a pile of disconnected context. "
            "The retriever searches trusted sources and pulls back evidence that matches the question. "
            "The generator reads that evidence beside the prompt, so the answer is grounded instead of guessed. "
            "The final proof is visible: citations attach to the answer, weak context fades out, and verification becomes straightforward."
        )
    elif _is_vex_or_hyperframes(topic):
        narration = (
            "A weak video generator starts with loose text and hopes motion will make it feel real. "
            "Vex starts with a production contract: script, beat timing, visual evidence, motion intent, and QA gates. "
            "Each beat becomes a HyperFrames world with objects that explain the claim, not decoration around it. "
            "Then variants compete, weak scenes are rejected, and the final cut keeps only visuals that help the viewer understand."
        )
    else:
        lead = _readable_topic(topic)
        primary = keywords[0] if keywords else "the idea"
        secondary = keywords[1] if len(keywords) > 1 else "the bottleneck"
        third = keywords[2] if len(keywords) > 2 else "the useful signal"
        narration = (
            f"{lead} becomes understandable when the viewer sees {primary} collide with {secondary}. "
            f"The video should expose the hidden mechanism, then show exactly what changes when {third} is protected. "
            f"Instead of naming the idea, each scene turns one claim into visible evidence: before state, intervention, and payoff. "
            f"The final takeaway is practical: once the structure is visible, the viewer knows what to improve and why it matters."
        )
    if request.cta:
        cta = request.cta.strip()
        narration += " " + (cta if re.search(r"[.!?]$", cta) else f"{cta}.")
    return ScriptPlan(
        title=plan.title,
        narration=_normalize_sentence_spacing(narration),
        design_direction=(
            f"{plan.design_direction}; director-led story arc; every beat must show "
            "a cause, contrast, mechanism, or payoff with distinct HyperFrames motion."
        ),
        source=SCRIPT_DIRECTOR_VERSION,
        prompt=plan.prompt,
        audience=plan.audience,
        cta=plan.cta,
    )


def build_director_package(
    *,
    request: VideoGenerationRequest,
    plan: ScriptPlan,
    beat_graph: BeatGraph,
    script_rewrite_applied: bool = False,
) -> DirectorPackage:
    topic = _topic_from_request(request, plan)
    keywords = _keywords_for(topic, plan.narration, request.prompt, limit=8)
    brief = ProductionBrief(
        title=plan.title,
        topic=_readable_topic(topic),
        promise=_promise_for(topic, keywords),
        audience=request.audience or "curious technical viewer",
        style=request.style,
        hook=_hook_for(topic, keywords),
        story_arc=[_arc_label(beat) for beat in beat_graph.beats],
        visual_language=_visual_language_for(request, keywords),
        forbidden_patterns=sorted(_GENERIC_VISUAL_PATTERNS),
        quality_bars=[
            "every beat has a viewer question and a visual answer",
            "motion must reveal state change, not decorate static text",
            "adjacent beats must use distinguishable visual-world fingerprints",
            "captions support the scene but never carry the whole explanation",
            "final render must pass layout, motion, semantic, and portfolio gates",
        ],
    )
    contracts = [
        _contract_for_beat(
            beat,
            topic=topic,
            keywords=keywords,
            is_last=beat.index >= len(beat_graph.beats),
        )
        for beat in beat_graph.beats
    ]
    return DirectorPackage(
        version=DIRECTOR_CREW_VERSION,
        script_director_version=SCRIPT_DIRECTOR_VERSION,
        script_rewrite_applied=script_rewrite_applied,
        brief=brief,
        beat_contracts=contracts,
        portfolio_policy={
            "min_semantic_coverage": 0.82,
            "min_native_motion_coverage": 0.82,
            "max_repeated_medium_run": 2,
            "require_distinct_world_signatures": True,
            "require_tournament_records": True,
            "reject_generic_script_patterns": sorted(_GENERIC_SCRIPT_PATTERNS),
            "prefer_full_hyperframes_surface": [
                "semantic_visual_worlds",
                "proof_program_variants",
                "seekable_motion",
                "audio_locked_cues",
                "transition_overlays",
                "portfolio_diversity",
            ],
        },
        crew_roles=[
            "showrunner",
            "script_director",
            "storyboard_director",
            "per_beat_visual_agent",
            "motion_designer",
            "semantic_qa_judge",
            "portfolio_judge",
        ],
    )


def director_context_for_beat(
    package: DirectorPackage | None,
    beat_id: str,
) -> dict[str, Any]:
    if package is None:
        return {}
    contract = package.beat_contract(beat_id)
    if contract is None:
        return {}
    return {
        "production_brief": package.brief.to_dict(),
        "beat_contract": contract.to_dict(),
        "portfolio_policy": dict(package.portfolio_policy),
    }


def generic_script_patterns_found(text: str) -> list[str]:
    normalized = _normalize_key(text)
    return sorted(pattern for pattern in _GENERIC_SCRIPT_PATTERNS if pattern in normalized)


def _script_needs_director_rewrite(plan: ScriptPlan) -> bool:
    normalized = _normalize_key(plan.narration)
    if any(pattern in normalized for pattern in _GENERIC_SCRIPT_PATTERNS):
        return True
    sentences = [item for item in re.split(r"(?<=[.!?])\s+", plan.narration) if item.strip()]
    if len(sentences) < 3:
        return True
    keywords = keyword_candidates(plan.prompt, limit=4)
    if not keywords:
        return False
    narration_key = _normalize_key(plan.narration)
    covered = sum(1 for keyword in keywords if keyword in narration_key)
    return covered < max(1, min(2, len(keywords)))


def _contract_for_beat(
    beat: Beat,
    *,
    topic: str,
    keywords: list[str],
    is_last: bool,
) -> BeatContract:
    objects = _required_objects_for(beat, topic=topic, keywords=keywords)
    relation = _relation_for(beat, objects)
    return BeatContract(
        beat_id=beat.beat_id,
        index=beat.index,
        objective=_objective_for(beat, is_last=is_last),
        viewer_question=_viewer_question_for(beat),
        visual_job=_visual_job_for(beat),
        required_objects=objects,
        required_relation=relation,
        motion_intent=_motion_intent_for(beat),
        transition_intent=_transition_intent_for(beat, is_last=is_last),
        anti_filler_rule=(
            "Reject this beat if the scene could be swapped with a generic "
            "title card without losing meaning."
        ),
        evidence_text=beat.narration,
    )


def _required_objects_for(beat: Beat, *, topic: str, keywords: list[str]) -> list[str]:
    source_text = " ".join([beat.narration, beat.title, " ".join(beat.keywords)])
    source_candidates = [
        item
        for item in keyword_candidates(source_text, limit=12)
        if _is_source_grounded_object(item, source_text)
    ]
    if len(source_candidates) < 3:
        topic_candidates = [
            item
            for item in keyword_candidates(" ".join([topic, " ".join(keywords)]), limit=8)
            if _is_source_grounded_object(item, source_text)
        ]
        source_candidates.extend(topic_candidates)
    return _unique_readable(source_candidates, limit=5)


def _relation_for(beat: Beat, objects: list[str]) -> str:
    if beat.scene_type == "contrast":
        return "show how the after state resolves the before state"
    if beat.scene_type == "process":
        return "trace the path from input through mechanism to output"
    if beat.scene_type == "metric":
        return "make the number visibly affect the surrounding system"
    if beat.scene_type == "proof":
        return "connect claim to evidence before revealing the payoff"
    if len(objects) >= 2:
        return f"make {objects[0]} visibly change {objects[1]}"
    return "turn the narration into an inspectable visual state change"


def _objective_for(beat: Beat, *, is_last: bool) -> str:
    if is_last:
        return "land the final takeaway with visible proof"
    if beat.index == 1:
        return "earn attention by making the problem concrete"
    return {
        "metric": "make the magnitude legible",
        "contrast": "make the before-after delta impossible to miss",
        "process": "make the mechanism traceable",
        "proof": "make evidence resolve into a claim",
    }.get(beat.scene_type, "make the concept inspectable")


def _viewer_question_for(beat: Beat) -> str:
    return {
        "metric": "How big is the change?",
        "contrast": "What changed and why is the new state better?",
        "process": "What moves where, and what controls the path?",
        "proof": "What evidence makes the claim believable?",
        "hook": "Why should I care in the first three seconds?",
    }.get(beat.scene_type, "What is the concrete object I should understand?")


def _visual_job_for(beat: Beat) -> str:
    return {
        "metric": "metric proof scene",
        "contrast": "state transformation scene",
        "process": "mechanism walkthrough scene",
        "proof": "claim-evidence-payoff scene",
        "hook": "kinetic promise scene",
    }.get(beat.scene_type, "concept-to-mechanism scene")


def _motion_intent_for(beat: Beat) -> str:
    return {
        "metric": "number drives bars, particles, or compression pressure",
        "contrast": "camera crosses the hinge between before and after",
        "process": "route draws progressively with node handoffs",
        "proof": "evidence layers lock into one resolved state",
        "hook": "fast reveal, snap focus, then controlled hold",
    }.get(beat.scene_type, "parallax depth reveals hidden structure")


def _transition_intent_for(beat: Beat, *, is_last: bool) -> str:
    if is_last:
        return "resolve to a clean hold for the final thought"
    if beat.scene_type in {"hook", "contrast"}:
        return "energetic cut that preserves the viewer's mental model"
    return "semantic morph into the next beat's first object"


def _promise_for(topic: str, keywords: list[str]) -> str:
    subject = _readable_topic(topic)
    if keywords:
        return f"Make {subject} understandable through {', '.join(keywords[:3])}."
    return f"Make {subject} understandable through visible state changes."


def _hook_for(topic: str, keywords: list[str]) -> str:
    if len(keywords) >= 2:
        return f"Show {keywords[0]} colliding with {keywords[1]} before explaining it."
    return f"Make {_readable_topic(topic)} visible before naming it."


def _arc_label(beat: Beat) -> str:
    return f"{beat.index:02d}:{beat.scene_type}:{beat.title}"


def _visual_language_for(request: VideoGenerationRequest, keywords: list[str]) -> list[str]:
    language = [
        request.style.replace("_", " "),
        "semantic visual worlds",
        "audio-locked motion cues",
        "proof-program variants",
    ]
    if any(item in {"token", "attention", "graph", "route", "memory"} for item in keywords):
        language.append("data sculpture and routed systems")
    if request.aspect == "portrait":
        language.append("large-caption vertical composition")
    elif request.aspect == "square":
        language.append("center-weighted social composition")
    else:
        language.append("cinematic landscape composition")
    return _unique_readable(language, limit=8)


def _topic_from_request(request: VideoGenerationRequest, plan: ScriptPlan) -> str:
    raw = request.prompt or plan.prompt or plan.title or plan.narration
    cleaned = re.sub(
        r"^(?:make|create|generate|build|produce|show|explain|visualize|illustrate|demonstrate)\s+"
        r"(?:(?:a|an)\s+)?(?:(?:video|hyperframes\s+video)\s+)?(?:about|on|for|how)?\s*",
        "",
        str(raw or ""),
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", cleaned).strip(" .") or "this idea"


def _keywords_for(*values: str, limit: int) -> list[str]:
    return _unique_readable(
        [
            item
            for value in values
            for item in keyword_candidates(value, limit=limit)
        ],
        limit=limit,
    )


def _is_sparse_attention(topic: str) -> bool:
    normalized = _normalize_key(topic)
    return "sparse attention" in normalized or (
        "attention" in normalized and "token" in normalized
    )


def _is_retrieval_augmented_generation(topic: str) -> bool:
    normalized = _normalize_key(topic)
    return (
        "retrieval augmented generation" in normalized
        or "rag" in normalized.split()
        or ("retrieval" in normalized and "generation" in normalized)
    )


def _is_vex_or_hyperframes(topic: str) -> bool:
    normalized = _normalize_key(topic)
    return "vex" in normalized or "hyperframes" in normalized or "video generation" in normalized


def _readable_topic(topic: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(topic or "")).strip(" .")
    if not cleaned:
        return "this idea"
    return cleaned[0].upper() + cleaned[1:]


def _normalize_sentence_spacing(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9%+./-]+", " ", str(value or "").lower()).strip()


def _unique_readable(values: list[str], *, limit: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = re.sub(r"\s+", " ", str(value or "")).strip(" ,.;:-").lower()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
        if len(result) >= limit:
            break
    return result


def _is_source_grounded_object(label: str, source_text: str) -> bool:
    normalized = _normalize_key(label)
    if not normalized or normalized in _LOW_VALUE_REQUIRED_OBJECTS:
        return False
    if any(normalized.startswith(stem) for stem in _LOW_VALUE_REQUIRED_OBJECT_STEMS):
        return False
    if len(normalized) < 4 and not any(ch.isdigit() for ch in normalized):
        return False
    source_key = _normalize_key(source_text)
    if normalized in source_key:
        return True
    tokens = [token for token in normalized.split() if token]
    return bool(tokens) and all(token in source_key for token in tokens)
