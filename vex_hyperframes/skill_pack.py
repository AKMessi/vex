from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class HyperframesSkillSlice:
    skill_id: str
    title: str
    applies_to_templates: tuple[str, ...]
    rules: tuple[str, ...]
    avoid: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_CORE_SLICES: tuple[HyperframesSkillSlice, ...] = (
    HyperframesSkillSlice(
        skill_id="hyperframes-production-contract",
        title="Hyperframes Production Contract",
        applies_to_templates=(),
        rules=(
            "The composition root must define data-composition-id, data-start, data-width, data-height, and data-duration.",
            "Every visible timed element must include class=\"clip\", data-start, data-duration, and data-track-index.",
            "Track indexes must not overlap for timed visible elements.",
            "Register one seekable timeline under window.__timelines[compositionId] before rendering.",
            "Keep the composition self-contained so local batch renders do not depend on CDN availability.",
        ),
        avoid=(
            "Deprecated data-layer or data-end attributes.",
            "Wall-clock animation loops, requestAnimationFrame-only motion, or setInterval-driven state.",
            "Remote fonts, remote scripts, or remote media in generated compositions.",
        ),
    ),
    HyperframesSkillSlice(
        skill_id="hyperframes-motion-language",
        title="Seekable Motion Language",
        applies_to_templates=(),
        rules=(
            "Drive animation from the requested render time, not elapsed browser time.",
            "Use entrance, emphasis, and resolve phases so a short visual still reads as a complete thought.",
            "Prefer transforms, opacity, stroke reveal, and CSS variables over expensive filters.",
            "Keep all copy inside safe margins and make every text block overflow-wrap defensively.",
        ),
        avoid=(
            "Large animated blur stacks or backdrop filters.",
            "Layouts that rely on viewport reflow during rendering.",
            "Tiny body copy that cannot survive compression in picture-in-picture mode.",
        ),
    ),
    HyperframesSkillSlice(
        skill_id="hyperframes-explainer-slides",
        title="Explainer Slide Archetypes",
        applies_to_templates=(
            "data_journey",
            "signal_network",
            "kinetic_route",
            "spotlight_compare",
            "interface_cascade",
            "ribbon_quote",
            "causal_chain",
            "flywheel_loop",
            "decision_matrix",
            "anatomy_cutaway",
            "stack_ranking",
            "contrast_ladder",
            "proof_sequence",
            "narrative_arc",
            "concept_map",
            "problem_solution",
            "myth_buster",
            "checklist_reveal",
            "risk_radar",
            "opportunity_map",
            "scorecard",
            "pipeline_xray",
            "decision_tree",
            "momentum_wave",
            "focus_ring",
            "timeline_filmstrip",
            "quote_breakdown",
            "market_map",
            "mechanism_blueprint",
            "data_pulse",
            "metric_callout",
            "keyword_stack",
            "timeline_steps",
            "comparison_split",
            "quote_focus",
            "system_flow",
            "stat_grid",
        ),
        rules=(
            "Use data_journey and stat_grid for numeric proof or dashboard-style claims.",
            "Use proof_sequence when multiple facts should accumulate into a single argument.",
            "Use signal_network, system_flow, kinetic_route, causal_chain, flywheel_loop, and timeline_steps for causal movement, loops, and process beats.",
            "Use spotlight_compare, contrast_ladder, decision_matrix, and comparison_split for before/after, tradeoff, or misconception flips.",
            "Use anatomy_cutaway when a system, product, or idea needs layered explanation.",
            "Use stack_ranking for ordered priorities, tiers, or weighted factors.",
            "Use narrative_arc when the spoken beat has setup, tension, and payoff.",
            "Use concept_map, opportunity_map, and market_map when the narration names a landscape, category space, or connected concept set.",
            "Use problem_solution and myth_buster when the beat needs a clear false-belief or pain-to-fix pivot.",
            "Use checklist_reveal, scorecard, risk_radar, and decision_tree for practical criteria, quality gates, risks, and if/then routing.",
            "Use pipeline_xray and mechanism_blueprint for hidden workflows, stacks, or under-the-hood explanations.",
            "Use momentum_wave, focus_ring, timeline_filmstrip, quote_breakdown, and data_pulse for trend, attention, story, quote deconstruction, and feedback-signal beats.",
            "Use interface_cascade for product or UI walkthroughs.",
            "Use ribbon_quote, quote_focus, and keyword_stack for distilled abstract claims.",
        ),
        avoid=(
            "Generic boxes with no motion relationship to the narration.",
            "Repeating the full spoken sentence as slide text.",
            "More than four simultaneous conceptual objects in a short insert.",
        ),
    ),
)


def retrieve_skill_slices(template: str | None = None, *, limit: int = 4) -> list[HyperframesSkillSlice]:
    normalized = str(template or "").strip().lower()
    ranked: list[tuple[float, HyperframesSkillSlice]] = []
    for index, skill in enumerate(_CORE_SLICES):
        score = 1.0 - index * 0.01
        if normalized and normalized in skill.applies_to_templates:
            score += 0.45
        if not skill.applies_to_templates:
            score += 0.2
        ranked.append((score, skill))
    return [skill for _, skill in sorted(ranked, key=lambda item: item[0], reverse=True)[:limit]]
