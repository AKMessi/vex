# Contracts

## Pipeline Boundary

Use this order:

1. Transcript/imported evidence
2. `VisualExplanationIR`
3. Signed visual claim graph
4. Storyboard panels and review
5. Ranked blueprint priors
6. Structural proof-program tournament
7. Per-program signed production contract
8. Deterministic composition
9. Blind inverse decoding and counterfactual QA
10. Visual and motion QA

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

## Visual Claim Graph

Require:

- grounded nodes with stable IDs
- directed relation types
- required relation IDs
- sequence node IDs
- blind proof questions
- graph signature

A render with all required labels can still fail when the relations are not independently decodable.

## Proof Tournament

Every candidate requires:

- deterministic proof-program ID
- blueprint ID and stage family
- structural encoding family
- relation mode
- structural prior
- its own production contract and semantic signature

Candidates compete untouched. Bounded repair begins only after no original candidate passes.

## Rejection Policy

Reject when:

- evidence is missing
- a requested metric is unverified
- required labels lack provenance
- the scene has no supported explanatory structure
- blueprint roles are missing
- semantic motion cannot be expressed
- the final frame cannot pass the screenshot test
