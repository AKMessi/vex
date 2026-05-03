from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from vex_manim.blueprint import SceneBlueprint
from vex_manim.briefs import SceneBrief


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "but",
    "by",
    "for",
    "from",
    "get",
    "gets",
    "got",
    "had",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "so",
    "that",
    "the",
    "their",
    "then",
    "this",
    "to",
    "was",
    "we",
    "when",
    "with",
    "you",
    "your",
}


@dataclass
class VisualObject:
    object_id: str
    role: str
    meaning: str
    representation: str
    copy: list[str] = field(default_factory=list)
    motion: str = ""
    constraints: list[str] = field(default_factory=list)
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StoryboardFrame:
    frame_id: str
    time_window: str
    focus_object: str
    visible_objects: list[str]
    camera: str
    caption: str
    required_change: str
    negative_space: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VisualExplanationIR:
    visual_id: str
    scene_type: str
    claim: str
    viewer_question: str
    misconception: str
    correct_model: str
    proof_signal: str
    visual_goal: str
    narrative_arc: list[str]
    objects: list[VisualObject]
    required_motion: list[str]
    copy_budget_words: int
    forbidden_patterns: list[str]
    evidence: dict[str, Any]
    source_context: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "visual_id": self.visual_id,
            "scene_type": self.scene_type,
            "claim": self.claim,
            "viewer_question": self.viewer_question,
            "misconception": self.misconception,
            "correct_model": self.correct_model,
            "proof_signal": self.proof_signal,
            "visual_goal": self.visual_goal,
            "narrative_arc": list(self.narrative_arc),
            "objects": [item.to_dict() for item in self.objects],
            "required_motion": list(self.required_motion),
            "copy_budget_words": int(self.copy_budget_words),
            "forbidden_patterns": list(self.forbidden_patterns),
            "evidence": dict(self.evidence),
            "source_context": dict(self.source_context),
        }


@dataclass
class StoryboardCritique:
    blueprint_id: str
    score: float
    passed: bool
    fatal_issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "blueprint_id": self.blueprint_id,
            "score": round(float(self.score), 3),
            "passed": bool(self.passed),
            "fatal_issues": list(self.fatal_issues),
            "warnings": list(self.warnings),
            "strengths": list(self.strengths),
        }


def _clean(text: Any, *, max_chars: int = 140) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip(" -,\n\t")
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip(" ,.;:-")
    return cleaned


def _tokens(text: str) -> list[str]:
    return [
        token.lower()
        for token in re.findall(r"[A-Za-z0-9%+.-]+(?:'[A-Za-z0-9%+.-]+)?", text or "")
        if token and token.lower() not in _STOPWORDS
    ]


def _phrase(text: Any, *, max_words: int = 5, max_chars: int = 44) -> str:
    cleaned = _clean(text, max_chars=max_chars * 2)
    if not cleaned:
        return ""
    tokens = re.findall(r"[A-Za-z0-9%+.-]+(?:'[A-Za-z0-9%+.-]+)?", cleaned)
    if not tokens:
        return cleaned[:max_chars].rstrip()
    kept: list[str] = []
    for token in tokens:
        if not kept and token.lower() in _STOPWORDS:
            continue
        kept.append(token)
        if len(kept) >= max_words:
            break
    candidate = " ".join(kept).strip() or " ".join(tokens[:max_words]).strip()
    if len(candidate) > max_chars:
        candidate = candidate[:max_chars].rstrip(" ,.;:-")
    return candidate


def _first_phrase(*values: Any, max_words: int = 6, max_chars: int = 56) -> str:
    for value in values:
        phrase = _phrase(value, max_words=max_words, max_chars=max_chars)
        if phrase:
            return phrase
    return ""


def _unique(items: list[str], *, limit: int = 8) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        cleaned = _clean(item, max_chars=90)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
        if len(result) >= limit:
            break
    return result


def _extract_numbers(*texts: str) -> list[str]:
    hits: list[str] = []
    for text in texts:
        hits.extend(re.findall(r"\b\d+(?:\.\d+)?\s*(?:x|%|k|m|hr|hrs|hour|hours|min|mins|minutes|sec|seconds|pages?|steps?)?\b", text or "", flags=re.IGNORECASE))
    return _unique([_clean(hit, max_chars=24) for hit in hits], limit=4)


def _keywords(*texts: str, limit: int = 6) -> list[str]:
    counts: dict[str, int] = {}
    for token in _tokens(" ".join(texts)):
        if len(token) < 4 and not any(char.isdigit() for char in token):
            continue
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))
    return [item[0] for item in ranked[:limit]]


def _scene_type(brief: SceneBrief, blueprint: SceneBlueprint | None) -> str:
    family = str((blueprint.scene_family if blueprint else brief.scene_family) or brief.scene_family)
    mode = str(brief.intuition_mode or "").lower()
    archetype = str(getattr(blueprint, "archetype", "") or "").lower()
    if family == "comparison_morph" or mode == "misconception_flip" or "morph" in archetype:
        return "before_after_morph"
    if family == "system_map" or mode == "causal_chain" or "network" in archetype:
        return "signal_flow_system"
    if family == "timeline_journey" or mode == "process_route" or "route" in archetype:
        return "guided_process_route"
    if family in {"metric_story", "dashboard_build"} or mode == "metric_proof" or "metric" in archetype:
        return "metric_proof"
    if family in {"kinetic_quote", "kinetic_stack"}:
        return "concept_emphasis"
    if family == "interface_focus":
        return "interface_causality"
    return "concept_map"


def _supporting_terms(brief: SceneBrief) -> list[str]:
    copy_bank = dict(brief.copy_bank or {})
    raw_terms: list[str] = []
    raw_terms.extend(str(item) for item in list(copy_bank.get("steps") or []))
    raw_terms.extend(str(item) for item in list(copy_bank.get("supporting_lines") or []))
    raw_terms.extend(str(item) for item in list(copy_bank.get("keywords") or []))
    raw_terms.extend(
        [
            str(copy_bank.get("left_detail") or ""),
            str(copy_bank.get("right_detail") or ""),
            brief.before_state,
            brief.after_state,
            brief.cause,
            brief.effect,
            brief.viewer_takeaway,
        ]
    )
    return _unique([_phrase(item, max_words=4, max_chars=34) for item in raw_terms], limit=6)


def _object(
    object_id: str,
    *,
    role: str,
    meaning: str,
    representation: str,
    copy: list[str] | None = None,
    motion: str,
    source: str,
    constraints: list[str] | None = None,
) -> VisualObject:
    return VisualObject(
        object_id=object_id,
        role=role,
        meaning=_clean(meaning, max_chars=120),
        representation=_clean(representation, max_chars=120),
        copy=_unique([_phrase(item, max_words=5, max_chars=42) for item in list(copy or [])], limit=3),
        motion=_clean(motion, max_chars=110),
        constraints=list(constraints or []),
        source=_clean(source, max_chars=80),
    )


def build_visual_explanation_ir(
    spec: dict[str, Any],
    brief: SceneBrief,
    blueprint: SceneBlueprint | None = None,
) -> VisualExplanationIR:
    scene_type = _scene_type(brief, blueprint)
    source_context = {
        "spoken_anchor": _clean(brief.spoken_anchor, max_chars=220),
        "story_window": _clean(brief.story_window or brief.context, max_chars=320),
        "objective": _clean(brief.objective, max_chars=220),
    }
    terms = _supporting_terms(brief)
    numbers = _extract_numbers(brief.headline, brief.deck, brief.spoken_anchor, brief.context, brief.story_window)
    keywords = _keywords(brief.headline, brief.deck, brief.spoken_anchor, brief.context, brief.story_window)
    claim = _first_phrase(brief.viewer_takeaway, brief.headline, brief.mental_model, brief.objective, max_words=7, max_chars=64) or "Key idea"
    misconception = _first_phrase(brief.before_state, spec.get("left_detail"), brief.spoken_anchor, max_words=5, max_chars=46)
    correct_model = _first_phrase(brief.after_state, brief.viewer_takeaway, spec.get("right_detail"), brief.mental_model, max_words=6, max_chars=56)
    proof_signal = _first_phrase(brief.effect, brief.cause, brief.deck, *(numbers or terms), max_words=5, max_chars=48)
    viewer_question = _clean(
        f"Why does {claim.lower()} matter?" if claim else "What mental model should change?",
        max_chars=90,
    )
    visual_goal = _clean(
        brief.mental_model
        or brief.viewer_takeaway
        or f"Make {claim.lower()} visible through motion, not transcript text.",
        max_chars=180,
    )
    evidence = {
        "numbers": numbers,
        "keywords": keywords,
        "terms": terms,
        "scene_family": brief.scene_family,
        "intuition_mode": brief.intuition_mode,
        "visual_metaphor": brief.visual_metaphor,
    }
    objects: list[VisualObject] = [
        _object(
            "hero_claim",
            role="title_anchor",
            meaning="The compact claim the viewer should remember.",
            representation="Editorial title locked to a safe corner, never a paragraph.",
            copy=[claim],
            motion="Stagger in, then yield focus to the visual mechanism.",
            source="viewer_takeaway/headline",
            constraints=["max 1 short line plus optional deck"],
        )
    ]
    if scene_type == "before_after_morph":
        objects.extend(
            [
                _object(
                    "wrong_model",
                    role="before_state",
                    meaning="The tempting but weak mental model.",
                    representation="Dim structure, brittle stack, or shrinking side of a split.",
                    copy=[misconception or "Passive input"],
                    motion="Enter first, then compress or fade as the better model appears.",
                    source="before_state",
                    constraints=["must not dominate final frame"],
                ),
                _object(
                    "better_model",
                    role="after_state",
                    meaning="The corrected model the viewer should adopt.",
                    representation="Brighter system, route, or active structure replacing the wrong model.",
                    copy=[correct_model or claim],
                    motion="Use TransformMatchingShapes, ReplacementTransform, or matched movement from the wrong model.",
                    source="after_state/viewer_takeaway",
                ),
                _object(
                    "transform_bridge",
                    role="causal_bridge",
                    meaning="Shows why the shift happens.",
                    representation="Traveling pulse, hinge, wipe, or connecting path between states.",
                    copy=[proof_signal],
                    motion="Pulse crosses from wrong model to better model while camera follows.",
                    source="cause/effect",
                ),
            ]
        )
        required_motion = ["state collapse", "matched transform", "payoff reveal"]
        narrative_arc = ["Show the weak model.", "Make it visibly fail or compress.", "Replace it with the useful model."]
    elif scene_type == "signal_flow_system":
        labels = terms + [claim, correct_model, proof_signal]
        objects.extend(
            [
                _object(
                    "source_node",
                    role="input",
                    meaning="Where the action or idea begins.",
                    representation="Signal node with compact label.",
                    copy=[labels[0] if labels else "Input"],
                    motion="Light up as the route starts.",
                    source="context",
                ),
                _object(
                    "mechanism_node",
                    role="mechanism",
                    meaning="The hidden cause or leverage point.",
                    representation="Larger hub, bottleneck, or orbiting core.",
                    copy=[labels[1] if len(labels) > 1 else correct_model or "Mechanism"],
                    motion="Orbit, pulse, or redraw while the signal passes through.",
                    source="cause/mental_model",
                ),
                _object(
                    "outcome_node",
                    role="outcome",
                    meaning="The payoff or consequence.",
                    representation="Destination node or expanding output field.",
                    copy=[labels[2] if len(labels) > 2 else proof_signal or "Outcome"],
                    motion="Receives the pulse, expands, then settles.",
                    source="effect/viewer_takeaway",
                ),
                _object(
                    "signal_path",
                    role="motion_spine",
                    meaning="Makes causality legible.",
                    representation="Curved path with traveling glow and trace.",
                    copy=[],
                    motion="MoveAlongPath with glow trail; no disconnected cards.",
                    source="blueprint",
                ),
            ]
        )
        required_motion = ["signal travel", "hub activation", "outcome expansion"]
        narrative_arc = ["Start with the input.", "Route through the hidden mechanism.", "Land on the outcome."]
    elif scene_type == "guided_process_route":
        labels = (terms + ["Start", "Friction", "Correction", "Payoff"])[:4]
        objects.extend(
            [
                _object(
                    f"route_step_{index + 1}",
                    role="process_step",
                    meaning=f"Process checkpoint {index + 1}.",
                    representation="Node on a curved route, not an isolated card.",
                    copy=[label],
                    motion="Reveal on the route as the traveler reaches it.",
                    source="steps/supporting_lines",
                )
                for index, label in enumerate(labels)
            ]
        )
        objects.append(
            _object(
                "traveler",
                role="attention_driver",
                meaning="Guides viewer attention through the process.",
                representation="Glow dot, cursor, or bead moving along the route.",
                copy=[],
                motion="MoveAlongPath from first step to final step with camera follow.",
                source="blueprint",
            )
        )
        required_motion = ["route draw", "traveler movement", "step-by-step reveal"]
        narrative_arc = ["Draw the route.", "Move through friction or decision points.", "Arrive at the model/payoff."]
    elif scene_type == "metric_proof":
        metric = numbers[0] if numbers else _first_phrase(brief.headline, brief.deck, max_words=3, max_chars=24) or "Signal"
        objects.extend(
            [
                _object(
                    "metric_badge",
                    role="metric",
                    meaning="The measurable proof behind the claim.",
                    representation="Tracked number, badge, or gauge.",
                    copy=[metric],
                    motion="Count, pulse, or lock onto chart point.",
                    source="numeric evidence/headline",
                ),
                _object(
                    "evidence_curve",
                    role="chart",
                    meaning="Shows the trend or threshold visually.",
                    representation="Axes/path/curve with highlighted point.",
                    copy=[],
                    motion="Draw curve, then move marker to the proof point.",
                    source="blueprint",
                ),
                _object(
                    "threshold_marker",
                    role="contrast",
                    meaning="Makes the before/after threshold obvious.",
                    representation="Line, wall, gate, or threshold band.",
                    copy=[proof_signal or correct_model],
                    motion="Marker locks in as the metric crosses it.",
                    source="effect/deck",
                ),
            ]
        )
        required_motion = ["metric count", "curve/path draw", "threshold highlight"]
        narrative_arc = ["State the measurable claim.", "Draw the evidence geometry.", "Highlight the proof point."]
    elif scene_type == "interface_causality":
        labels = terms + [claim, correct_model]
        objects.extend(
            [
                _object(
                    "primary_surface",
                    role="interface",
                    meaning="The concrete surface where the idea becomes actionable.",
                    representation="One polished UI surface or dashboard, not many boxes.",
                    copy=[labels[0] if labels else claim],
                    motion="Surface assembles from meaningful modules.",
                    source="visual_type/product_ui",
                ),
                _object(
                    "focus_module",
                    role="focus",
                    meaning="The exact control, step, or lever that matters.",
                    representation="Highlighted module with focus beam and connector.",
                    copy=[labels[1] if len(labels) > 1 else correct_model],
                    motion="Camera punch-in and beam sweep.",
                    source="viewer_takeaway",
                ),
                _object(
                    "feedback_signal",
                    role="result",
                    meaning="Shows the outcome of using the lever.",
                    representation="Signal, progress route, or status trace.",
                    copy=[proof_signal],
                    motion="Signal leaves focus module and resolves into outcome.",
                    source="effect",
                ),
            ]
        )
        required_motion = ["surface assembly", "focus punch-in", "feedback signal"]
        narrative_arc = ["Assemble the actionable surface.", "Focus the key lever.", "Show the result."]
    else:
        labels = terms + [correct_model, proof_signal, claim]
        objects.extend(
            [
                _object(
                    "concept_core",
                    role="core_idea",
                    meaning="The central mental model.",
                    representation="Kinetic phrase or compact symbolic core.",
                    copy=[claim],
                    motion="Write or morph into place.",
                    source="headline/viewer_takeaway",
                ),
                _object(
                    "support_orbits",
                    role="supporting_evidence",
                    meaning="Supporting ideas orbit or connect to the core.",
                    representation="Small labels on orbit/route/beam, not stacked paragraphs.",
                    copy=labels[:3],
                    motion="Orbit, stagger, or connect into the core.",
                    source="supporting_lines",
                ),
                _object(
                    "payoff_marker",
                    role="payoff",
                    meaning="The final viewer takeaway.",
                    representation="Accent marker, underline, or glow lock.",
                    copy=[correct_model or proof_signal],
                    motion="Locks in on final beat.",
                    source="after_state/effect",
                ),
            ]
        )
        required_motion = ["kinetic reveal", "support convergence", "payoff lock"]
        narrative_arc = ["Introduce the idea.", "Connect the supporting logic.", "Land the takeaway."]

    forbidden = _unique(
        [
            *brief.must_avoid,
            "static box-only layout",
            "subtitle duplicated as a paragraph",
            "random decorative chart with no connection to the narration",
            "more than one large paragraph on screen",
            "labels crossing shapes or sitting outside safe bounds",
            "unmotivated duplicate visual repeated twice",
        ],
        limit=12,
    )
    return VisualExplanationIR(
        visual_id=brief.visual_id,
        scene_type=scene_type,
        claim=claim,
        viewer_question=viewer_question,
        misconception=misconception,
        correct_model=correct_model,
        proof_signal=proof_signal,
        visual_goal=visual_goal,
        narrative_arc=narrative_arc,
        objects=objects,
        required_motion=required_motion,
        copy_budget_words=max(8, int(brief.text_budget_words or 16)),
        forbidden_patterns=forbidden,
        evidence=evidence,
        source_context=source_context,
    )


def build_storyboard_frames(
    ir: VisualExplanationIR,
    brief: SceneBrief,
    blueprint: SceneBlueprint,
) -> list[StoryboardFrame]:
    object_ids = [item.object_id for item in ir.objects]
    semantic_ids = [item.object_id for item in ir.objects if item.role not in {"title_anchor", "background"}]
    first_focus = semantic_ids[0] if semantic_ids else object_ids[0]
    middle_focus = semantic_ids[min(1, len(semantic_ids) - 1)] if semantic_ids else first_focus
    final_focus = semantic_ids[-1] if semantic_ids else first_focus
    camera = _clean(blueprint.camera_plan or brief.camera_style or "wide then guided focus", max_chars=100)
    return [
        StoryboardFrame(
            frame_id="frame_01_establish",
            time_window="0-25%",
            focus_object=first_focus,
            visible_objects=_unique(["hero_claim", first_focus], limit=4),
            camera="wide safe-frame composition; no cropped text",
            caption=_clean(ir.narrative_arc[0] if ir.narrative_arc else "Establish the claim.", max_chars=100),
            required_change="Viewer sees the problem or claim before any explanation text expands.",
            negative_space="Leave one third of the frame open for motion to enter.",
        ),
        StoryboardFrame(
            frame_id="frame_02_mechanism",
            time_window="25-70%",
            focus_object=middle_focus,
            visible_objects=_unique(["hero_claim", first_focus, middle_focus, "signal_path", "traveler", "evidence_curve"], limit=6),
            camera=camera,
            caption=_clean(ir.narrative_arc[1] if len(ir.narrative_arc) > 1 else "Reveal the mechanism.", max_chars=100),
            required_change=", ".join(ir.required_motion[:2]) or "Motion must explain the hidden relationship.",
            negative_space="Keep labels outside the moving spine; do not pile text over the diagram.",
        ),
        StoryboardFrame(
            frame_id="frame_03_payoff",
            time_window="70-100%",
            focus_object=final_focus,
            visible_objects=_unique(["hero_claim", middle_focus, final_focus, "payoff_marker", "threshold_marker"], limit=6),
            camera="settled final focus with the payoff object dominant",
            caption=_clean(ir.narrative_arc[2] if len(ir.narrative_arc) > 2 else "Land the viewer takeaway.", max_chars=100),
            required_change=", ".join(ir.required_motion[-2:]) or "Final state must visibly differ from the opening state.",
            negative_space="Final frame must be screenshot-clean: no overlapping labels, no cropped words.",
        ),
    ]


def _word_count(lines: list[str]) -> int:
    return sum(len(re.findall(r"[A-Za-z0-9%+.-]+", line or "")) for line in lines)


def _has_motion_keyword(text: str, keywords: set[str]) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def critique_storyboard(
    ir: VisualExplanationIR,
    frames: list[StoryboardFrame],
    brief: SceneBrief,
    blueprint: SceneBlueprint,
) -> StoryboardCritique:
    score = 0.36
    fatal: list[str] = []
    warnings: list[str] = []
    strengths: list[str] = []
    object_text = " ".join(
        [
            ir.scene_type,
            ir.visual_goal,
            " ".join(ir.required_motion),
            blueprint.focal_system,
            blueprint.archetype,
            blueprint.layout_thesis,
        ]
    ).lower()
    object_roles = {item.role for item in ir.objects}
    all_copy = [line for item in ir.objects for line in item.copy]
    copy_words = _word_count(all_copy)
    if ir.claim and ir.visual_goal:
        score += 0.1
        strengths.append("has explicit claim and visual goal")
    else:
        fatal.append("missing claim or visual goal")
    if len(ir.objects) >= 4:
        score += 0.08
        strengths.append("has enough visual objects")
    else:
        fatal.append("too few visual objects")
    if len(ir.required_motion) >= 3:
        score += 0.1
        strengths.append("has a three-part motion contract")
    else:
        fatal.append("missing concrete motion contract")
    if len(frames) == 3 and len({frame.focus_object for frame in frames}) >= 2:
        score += 0.1
        strengths.append("storyboard changes focus across beats")
    else:
        warnings.append("storyboard focus does not change enough")
    if copy_words <= ir.copy_budget_words + 4:
        score += 0.08
        strengths.append("copy fits budget")
    elif copy_words > ir.copy_budget_words * 2:
        fatal.append(f"copy budget blown: {copy_words} words for budget {ir.copy_budget_words}")
    else:
        warnings.append(f"copy is slightly high: {copy_words} words")
    scene_type_requirements = {
        "before_after_morph": ({"before_state", "after_state", "causal_bridge"}, {"morph", "transform", "collapse", "replace"}),
        "signal_flow_system": ({"input", "mechanism", "outcome", "motion_spine"}, {"signal", "path", "flow", "route"}),
        "guided_process_route": ({"process_step", "attention_driver"}, {"route", "path", "traveler", "step"}),
        "metric_proof": ({"metric", "chart", "contrast"}, {"metric", "curve", "chart", "threshold", "count"}),
        "interface_causality": ({"interface", "focus", "result"}, {"surface", "focus", "signal", "feedback"}),
    }
    required_roles, required_motion_words = scene_type_requirements.get(
        ir.scene_type,
        ({"core_idea", "supporting_evidence", "payoff"}, {"reveal", "connect", "converge", "lock"}),
    )
    if object_roles.intersection(required_roles):
        score += 0.08
    else:
        fatal.append(f"storyboard objects do not match {ir.scene_type}")
    if _has_motion_keyword(object_text, required_motion_words):
        score += 0.08
        strengths.append("motion words match scene type")
    else:
        warnings.append("motion contract may not match scene type strongly enough")
    if brief.viewer_takeaway or brief.mental_model or brief.after_state:
        semantic_terms = set(_tokens(" ".join([brief.viewer_takeaway, brief.mental_model, brief.after_state, brief.cause, brief.effect])))
        ir_terms = set(_tokens(" ".join([ir.claim, ir.correct_model, ir.visual_goal, ir.proof_signal])))
        overlap = len(semantic_terms.intersection(ir_terms))
        if overlap >= 1:
            score += 0.08
            strengths.append("uses semantic frame, not just subtitle text")
        else:
            warnings.append("semantic frame overlap is weak")
    generic_hits = 0
    for item in ir.objects:
        text = f"{item.role} {item.representation} {' '.join(item.copy)}".lower()
        if any(pattern in text for pattern in {"generic card", "plain box", "text box", "paragraph"}):
            generic_hits += 1
    if generic_hits == 0:
        score += 0.06
    elif generic_hits >= 2:
        fatal.append("too many generic text/card objects")
    else:
        warnings.append("one object still smells generic")
    if any(frame.negative_space for frame in frames):
        score += 0.04
    if blueprint.dynamic_devices:
        score += min(0.06, len(blueprint.dynamic_devices) * 0.02)
    score = max(0.0, min(1.0, score))
    passed = score >= 0.66 and not fatal
    return StoryboardCritique(
        blueprint_id=blueprint.blueprint_id,
        score=score,
        passed=passed,
        fatal_issues=fatal,
        warnings=warnings[:6],
        strengths=strengths[:6],
    )


def storyboard_prompt_block(
    ir: VisualExplanationIR,
    frames: list[StoryboardFrame],
    critique: StoryboardCritique | None = None,
) -> str:
    object_lines = [
        (
            f"- {item.object_id}: role={item.role}; meaning={item.meaning}; "
            f"representation={item.representation}; copy={item.copy or ['(none)']}; motion={item.motion}"
        )
        for item in ir.objects
    ]
    frame_lines = [
        (
            f"- {frame.frame_id} ({frame.time_window}): focus={frame.focus_object}; "
            f"visible={', '.join(frame.visible_objects)}; camera={frame.camera}; "
            f"change={frame.required_change}; layout={frame.negative_space}"
        )
        for frame in frames
    ]
    critique_lines: list[str] = []
    if critique is not None:
        critique_lines = [
            f"Critic score: {critique.score:.3f}; passed={critique.passed}",
        ]
        if critique.fatal_issues:
            critique_lines.append("Fatal issues to avoid: " + "; ".join(critique.fatal_issues))
        if critique.warnings:
            critique_lines.append("Warnings to respect: " + "; ".join(critique.warnings))
    return "\n".join(
        [
            "Visual Explanation IR:",
            f"- Scene type: {ir.scene_type}",
            f"- Claim: {ir.claim}",
            f"- Viewer question: {ir.viewer_question}",
            f"- Misconception/before: {ir.misconception or '(none)'}",
            f"- Correct model/after: {ir.correct_model or '(derive from claim)'}",
            f"- Proof signal: {ir.proof_signal or '(visual proof, no extra prose)'}",
            f"- Visual goal: {ir.visual_goal}",
            f"- Required motion: {', '.join(ir.required_motion)}",
            f"- Copy budget: {ir.copy_budget_words} on-screen words",
            "Visual objects:",
            *object_lines,
            "Storyboard contract:",
            *frame_lines,
            "Forbidden patterns:",
            *[f"- {item}" for item in ir.forbidden_patterns[:8]],
            *critique_lines,
            "Non-negotiable: implement these frames as the scene contract. If a visual detail does not serve this IR, remove it.",
        ]
    )
