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
    scene_types: tuple[str, ...] = ()
    blueprint_tags: tuple[str, ...] = ()
    mandatory: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_prompt_block(self) -> str:
        lines = [f"Skill: {self.skill_id} - {self.title}", "Rules:"]
        lines.extend(f"- {item}" for item in self.rules)
        if self.avoid:
            lines.append("Avoid:")
            lines.extend(f"- {item}" for item in self.avoid)
        return "\n".join(lines)


_CORE_SLICES: tuple[HyperframesSkillSlice, ...] = (
    HyperframesSkillSlice(
        skill_id="hyperframes-production-contract",
        title="HyperFrames Production Contract",
        applies_to_templates=(),
        scene_types=(),
        rules=(
            "Require Visual Explanation IR, a selected semantic blueprint, a storyboard review, and a signed production contract before semantic rendering.",
            "The composition root must define data-composition-id, data-start, data-width, data-height, and data-duration.",
            "Every visible timed element must include class=\"clip\", data-start, data-duration, and data-track-index.",
            "Register one seekable timeline under window.__timelines[compositionId] before rendering.",
            "Keep the composition self-contained and deterministic so local renders never depend on network timing.",
        ),
        avoid=(
            "Rendering a rejected or structurally invalid Visual Explanation IR.",
            "Deprecated data-layer or data-end attributes.",
            "Remote fonts, scripts, media, URLs, or mutable runtime dependencies.",
        ),
        mandatory=True,
    ),
    HyperframesSkillSlice(
        skill_id="hyperframes-evidence-fidelity",
        title="Evidence Fidelity And Copy Provenance",
        applies_to_templates=(),
        scene_types=(),
        rules=(
            "Render only labels, metrics, entities, states, and quotes that resolve to evidence spans or explicitly grounded semantic facts.",
            "Treat every number as untrusted until its normalized value and unit occur in source evidence.",
            "Keep exact quotes exact; use semantic emphasis without paraphrasing quotation copy.",
            "Reject weak ideas instead of filling missing roles with generic copy.",
        ),
        avoid=(
            "Synthetic percentages, thresholds, scores, progress values, interface states, risks, or entities.",
            "Generic labels such as Input, Output, Core Idea, Mechanism, Result, or Signal when the source does not name them concretely.",
            "Repeating the transcript as a paragraph instead of compressing grounded meaning into labels.",
        ),
        mandatory=True,
    ),
    HyperframesSkillSlice(
        skill_id="hyperframes-visual-claim-graph",
        title="Signed Visual Claim Graph",
        applies_to_templates=(),
        scene_types=(),
        rules=(
            "Compile grounded objects into directed, typed relations and blind proof questions before choosing a visual program.",
            "Bind node, relation, sequence, and question payloads into the production-contract signature.",
            "Require every proof-bearing relation to resolve to grounded object IDs and evidence-backed labels.",
        ),
        avoid=(
            "Treating a list of correct labels as an explanation.",
            "Adding arrows or branches whose endpoints do not exist in the signed claim graph.",
            "Changing relation meaning after the production contract is signed.",
        ),
        mandatory=True,
    ),
    HyperframesSkillSlice(
        skill_id="hyperframes-structural-proof-search",
        title="Structural Visual Proof Search",
        applies_to_templates=(),
        scene_types=(),
        rules=(
            "Treat curated blueprints as search priors, not final templates.",
            "Generate multiple structurally distinct proof programs with separate blueprint contracts, encoding families, and deterministic IDs.",
            "Render every proof candidate independently before applying bounded repair.",
            "Promote the candidate with the strongest decoded relationships, not the prettiest surface treatment.",
        ),
        avoid=(
            "Cosmetic variants that keep the same explanatory structure.",
            "Mutating one proof candidate because a different candidate failed.",
            "Promoting a candidate that loses the tournament's signed contract or relation IDs.",
        ),
        mandatory=True,
    ),
    HyperframesSkillSlice(
        skill_id="hyperframes-seekable-motion",
        title="Seekable Semantic Motion",
        applies_to_templates=(),
        scene_types=(),
        rules=(
            "Drive all animation from requested composition time, never wall-clock elapsed time.",
            "Give every motion beat a semantic purpose: establish state, expose mechanism, apply intervention, follow route, compare evidence, or resolve outcome.",
            "Keep object identity stable across beats so changes are inspectable.",
            "Use transforms, opacity, stroke reveal, path progress, and CSS variables that remain deterministic at arbitrary seek times.",
        ),
        avoid=(
            "requestAnimationFrame-only motion, setInterval, or uncontrolled CSS animation clocks.",
            "Decorative movement that does not change the viewer's understanding.",
            "Destroying one object and introducing an unrelated replacement when a state should transform.",
        ),
        mandatory=True,
    ),
    HyperframesSkillSlice(
        skill_id="hyperframes-blind-inverse-decoder",
        title="Blind Inverse Decoding And Counterfactual QA",
        applies_to_templates=(),
        scene_types=(),
        rules=(
            "Decode the rendered frames without providing the intended thesis, transcript, storyboard, labels, or expected relations.",
            "Grade the blind decode against the signed claim graph only after decoding is complete.",
            "Ablate proof-bearing regions and scramble chronological frames; comprehension must degrade when the visual grammar is causally necessary.",
            "Persist decoded claims, relation coverage, counterfactual deltas, and exact missing relation IDs.",
        ),
        avoid=(
            "Giving the vision model the expected answer and asking it to confirm correctness.",
            "Passing label-only scenes whose required relations cannot be recovered.",
            "Using aesthetic quality as a substitute for independent semantic decoding.",
        ),
        mandatory=True,
    ),
    HyperframesSkillSlice(
        skill_id="hyperframes-semantic-qa",
        title="Semantic Screenshot And Motion QA",
        applies_to_templates=(),
        scene_types=(),
        rules=(
            "Require the paused resolved frame to communicate the source-backed relationship without transcript assistance.",
            "Verify required labels, objects, and motion beats against the signed contract before accepting visual aesthetics.",
            "Check multiple time samples so entrance-only motion cannot hide a static or incoherent middle.",
            "Reject a polished render when semantic coverage, object continuity, or evidence fidelity fails.",
        ),
        avoid=(
            "Selecting the least-bad variant when every variant fails a hard semantic check.",
            "Using contrast, occupancy, or visual polish as a substitute for explanatory correctness.",
            "Reporting a generic QA failure without the missing label, object, beat, or evidence reason.",
        ),
        mandatory=True,
    ),
    HyperframesSkillSlice(
        skill_id="hyperframes-metric-story",
        title="Measured Change And Metric Proof",
        applies_to_templates=("semantic_metric",),
        scene_types=("metric_delta", "metric_intervention", "metric_proof"),
        blueprint_tags=("metric", "axis", "threshold", "proof"),
        rules=(
            "Attach the hero metric to evidence geometry, registered states, or an intervention trace.",
            "Keep source-backed before and after values visible long enough to compare.",
            "Show an intervention on the causal path between measured states when one is named.",
            "Use axes and thresholds only when their scale or threshold is grounded.",
        ),
        avoid=(
            "Random bars, decorative sparklines, fake gauges, and unlabeled chart scales.",
            "A giant number floating without evidence.",
            "Implying causality from a metric delta when no intervention is supported.",
        ),
    ),
    HyperframesSkillSlice(
        skill_id="hyperframes-causal-spine",
        title="Causal Mechanism And Intervention",
        applies_to_templates=("semantic_causal",),
        scene_types=("causal_intervention",),
        blueprint_tags=("cause", "mechanism", "intervention", "counterfactual"),
        rules=(
            "Keep problem, mechanism, intervention, and result on one persistent causal spine.",
            "Make the intervention visibly alter signal behavior or route.",
            "Use a counterfactual split only when both outcomes are supported.",
        ),
        avoid=(
            "Static cause/effect cards with a decorative arrow.",
            "Skipping the mechanism and jumping directly from problem to result.",
            "Inventing an untreated outcome to complete a symmetrical layout.",
        ),
    ),
    HyperframesSkillSlice(
        skill_id="hyperframes-route-choreography",
        title="Processes, Handoffs, And Guided Routes",
        applies_to_templates=("semantic_route",),
        scene_types=("guided_process",),
        blueprint_tags=("route", "handoff", "process"),
        rules=(
            "Arrange source-ordered steps along one route and retain completed-state memory.",
            "Use one persistent traveler or token so ownership and sequence remain understandable.",
            "Make a handoff visible through ownership zones, token treatment, or route transition.",
        ),
        avoid=(
            "Disconnected numbered cards.",
            "Revealing all steps simultaneously.",
            "Replacing the token at a handoff and losing process identity.",
        ),
    ),
    HyperframesSkillSlice(
        skill_id="hyperframes-architecture-flow",
        title="Architecture And Service Lifecycle",
        applies_to_templates=("semantic_architecture",),
        scene_types=("architecture_flow",),
        blueprint_tags=("api", "service", "pipeline", "layer"),
        rules=(
            "Show explicit service boundaries and one request token moving in source order.",
            "Keep ownership labels short and tied to the service that owns the stage.",
            "Show the return path when the source names an output or response.",
        ),
        avoid=(
            "Network wallpaper with unlabeled boxes.",
            "Bidirectional arrows that do not encode request versus response.",
            "Adding databases, queues, workers, or infrastructure not present in evidence.",
        ),
    ),
    HyperframesSkillSlice(
        skill_id="hyperframes-matched-transform",
        title="Matched State Transformation",
        applies_to_templates=("semantic_transform",),
        scene_types=("matched_state_transform",),
        blueprint_tags=("before", "after", "constraint", "morph"),
        rules=(
            "Register equivalent before and after objects spatially so the change can be inspected.",
            "Keep preserved constraints fixed while changed elements transform around them.",
            "Use difference highlighting after the morph rather than duplicating prose.",
        ),
        avoid=(
            "Static versus cards with unrelated internal layouts.",
            "Hiding the invariant or quality gate that the source says remains.",
            "Using glow alone to imply improvement.",
        ),
    ),
    HyperframesSkillSlice(
        skill_id="hyperframes-grounded-interface",
        title="Grounded Interface Walkthrough",
        applies_to_templates=("semantic_interface",),
        scene_types=("grounded_interface_walkthrough",),
        blueprint_tags=("interface", "screen", "action", "result"),
        rules=(
            "Render only named interface states, actions, controls, and feedback.",
            "Keep the action target and resulting state in one stable interface context.",
            "Use a focus beam, cursor trace, or feedback path to explain causality.",
            "Prefer real captured UI assets when the source recording contains the named state.",
        ),
        avoid=(
            "Imaginary dashboards, controls, percentages, progress bars, logs, or notifications.",
            "Generic UI rows such as Input Captured or Action Rendered.",
            "A carousel of unrelated mock screens.",
        ),
    ),
    HyperframesSkillSlice(
        skill_id="hyperframes-decision-gate",
        title="Decision Gates And Guardrails",
        applies_to_templates=("semantic_decision",),
        scene_types=("decision_branch",),
        blueprint_tags=("decision", "branch", "gate", "guardrail"),
        rules=(
            "Center the named condition and keep both grounded outcomes inspectable.",
            "Activate only the selected route at a given beat.",
            "Show a protected downstream constraint when evidence names one.",
        ),
        avoid=(
            "Generic yes/no branches.",
            "Both branches appearing active simultaneously.",
            "Traffic-light decoration without a named condition.",
        ),
    ),
    HyperframesSkillSlice(
        skill_id="hyperframes-narrative-continuity",
        title="Narrative Progression And Recovery",
        applies_to_templates=("semantic_narrative",),
        scene_types=("narrative_progression",),
        blueprint_tags=("setup", "turn", "payoff", "recovery"),
        rules=(
            "Keep the same subject visible across setup, turning point, and payoff.",
            "Make the turning point alter trajectory or state rather than add another label.",
            "Hold the payoff long enough to compare it with the initial state.",
        ),
        avoid=(
            "A generic filmstrip with unrelated frames.",
            "Equal visual weight for every beat when one decisive turn drives the story.",
            "Resetting the spatial frame between beats.",
        ),
    ),
    HyperframesSkillSlice(
        skill_id="hyperframes-exact-quote",
        title="Evidence-Backed Quote Direction",
        applies_to_templates=("semantic_quote",),
        scene_types=("evidence_backed_quote",),
        blueprint_tags=("quote", "phrase", "exact"),
        rules=(
            "Preserve exact source language and use one decisive phrase emphasis.",
            "Keep the full quote recoverable after any clause isolation.",
            "Use restrained motion so typography remains readable and authoritative.",
        ),
        avoid=(
            "Paraphrasing quoted language.",
            "Word clouds or keywords not present in the quote.",
            "Over-animating every word independently.",
        ),
    ),
)


def retrieve_skill_slices(
    template: str | None = None,
    *,
    scene_type: str | None = None,
    blueprint_id: str | None = None,
    limit: int = 6,
) -> list[HyperframesSkillSlice]:
    normalized_template = str(template or "").strip().lower()
    normalized_scene = str(scene_type or "").strip().lower()
    normalized_blueprint = str(blueprint_id or "").strip().lower()
    mandatory = [skill for skill in _CORE_SLICES if skill.mandatory]
    ranked: list[tuple[float, HyperframesSkillSlice]] = []
    for index, skill in enumerate(_CORE_SLICES):
        if skill.mandatory:
            continue
        score = 0.2 - index * 0.001
        if normalized_template and normalized_template in skill.applies_to_templates:
            score += 0.75
        if normalized_scene and normalized_scene in skill.scene_types:
            score += 0.9
        score += min(
            sum(1 for tag in skill.blueprint_tags if tag in normalized_blueprint) * 0.12,
            0.36,
        )
        ranked.append((score, skill))
    selected = list(mandatory)
    remaining = max(int(limit), 0)
    selected.extend(
        skill
        for score, skill in sorted(ranked, key=lambda item: item[0], reverse=True)
        if score > 0.2
    )
    if len(selected) < len(mandatory) + remaining:
        selected.extend(
            skill
            for _, skill in sorted(ranked, key=lambda item: item[0], reverse=True)
            if skill not in selected
        )
    return selected[: len(mandatory) + remaining]


__all__ = [
    "HyperframesSkillSlice",
    "retrieve_skill_slices",
]
