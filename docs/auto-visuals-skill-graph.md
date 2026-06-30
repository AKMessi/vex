# Auto Visuals Skill Graph

Auto Visuals now routes every selected transcript opportunity through an
executable skill graph before language-model planning.

The skill graph is the reliability layer between opportunity selection and
rendering:

1. transcript cards become executable visual opportunities
2. each opportunity is compiled through the HyperFrames semantic preflight
3. scene type selects a first-class visual skill
4. the skill fills a bounded slot schema from source evidence
5. deterministic plan seeds constrain fallback and model planning
6. the HyperFrames compiler builds claim graphs, proof programs, visual worlds,
   renderer specs, QA contracts, and repair metadata
7. render QA and final timeline QA decide promotion

## Why It Exists

Average language models are unreliable when asked to invent the entire visual
architecture: renderer, template, timing, copy, proof structure, and QA contract
all at once.

The skill graph changes the model job from architecture design to bounded slot
fill. The model can polish a headline or choose between allowed phrasings, but
it cannot replace a `decision-gate` with a generic quote card or invent a metric
for a `metric-story`.

## Skill Contract

Every skill defines:

- supported scene types
- preferred templates
- renderer and composition policy
- required and optional slots
- blueprint tags
- proof encodings
- visual-world medium priors
- QA floor
- reject rules
- anti-patterns

The current production skills cover:

- metric stories and measured proof
- causal mechanism spines
- process route choreography
- architecture and service lifecycle flows
- matched state transforms
- grounded interface walkthroughs
- decision gates and guardrails
- narrative progression and recovery
- token partition/compression scenes
- exact quote direction

## Runtime Artifacts

The automatic Auto Visuals manifest records:

- `auto_visual_skill_graph`
- per-card `auto_visual_skill`
- selected skill, scene type, template, renderer, slots, skill slices, blueprint
  priors, proof encodings, visual-world mediums, QA floor, reject rules, and
  deterministic plan seed

Accepted overlays also carry `auto_visual_skill`, so a rendered visual can be
audited back to its source skill contract.

## Failure Policy

The skill graph rejects before rendering when:

- semantic preflight fails
- no skill supports the compiled scene type
- required evidence slots are missing
- the route score falls below the production floor

Rejected opportunities remain visible in the run report and can be replaced by
reserve opportunities later in the existing semantic compile and render recovery
paths.

## Implementation

The runtime lives in `visual_skill_graph.py`.

Automatic planning is wired in `tools/auto_visuals.py` after visual opportunity
selection and renderer capability checks, before `analyze_visual_plan_with_llm`.

Deterministic fallback and model prompts consume the skill graph through
`visual_intelligence.py`.
