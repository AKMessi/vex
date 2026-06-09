# Contracts

## Pipeline Boundary

Use this order:

1. Transcript/imported evidence
2. `VisualExplanationIR`
3. Storyboard panels and review
4. Curated blueprint selection
5. Signed production contract
6. Renderer specification
7. Deterministic composition
8. Semantic QA
9. Visual and motion QA

Do not generate HTML when a preceding contract rejects the visual.

## Visual Explanation IR

Require:

- `scene_type`
- `render_policy`
- evidence spans
- grounded facts
- typed objects
- timed semantic beats
- required labels
- forbidden content
- rejection reasons

Numbers require normalized value and unit provenance. Exact quotes remain exact. Semantic labels with no meaningful evidence overlap are unverified.

## Storyboard

Each panel must define:

- time window
- focus object
- visible object set
- visible semantic change
- semantic purpose
- camera instruction
- resolved-frame requirement

Reject storyboards that omit required objects, focus unknown objects, or contain invalid timing.

## Production Contract

Require:

- selected blueprint ID
- required labels and object IDs
- required semantic motion
- required visual devices
- screenshot test
- forbidden content
- quality floor
- semantic signature

The signature binds evidence, facts, objects, beats, scene type, and blueprint. Preserve it in renderer metadata and QA reports.

## Rejection Policy

Reject when:

- evidence is missing
- a requested metric is unverified
- required labels lack provenance
- the scene has no supported explanatory structure
- blueprint roles are missing
- semantic motion cannot be expressed
- the final frame cannot pass the screenshot test
