# HyperFrames Visual Proof Search

## Objective

Auto Visuals must produce an explanation that a viewer can independently decode.
Correct words, polished motion, and a plausible template are not sufficient.

The new architecture treats visual generation as search over executable proof
programs:

1. compile source evidence into a signed visual claim graph
2. generate structurally different candidate programs
3. render each candidate independently
4. blindly decode the rendered frames without exposing the intended answer
5. run counterfactual perturbations
6. promote the candidate with the strongest independently recovered proof

## Architecture

### Visual Claim Graph

`vex_hyperframes/claim_graph.py` converts grounded IR objects into:

- stable nodes
- typed directed relations
- required relation IDs
- sequence node IDs
- blind proof questions
- a tamper-detecting graph signature

The graph is embedded in the production contract. Changing a relation, endpoint, or
sequence invalidates the signature.

### Blueprint Priors

`vex_hyperframes/blueprints.py` now ranks every blueprint whose grounded role
prerequisites pass. A blueprint is a search prior, not the final visual.

Automatic rendering never selects legacy decorative templates. Legacy names remain
available for manual compatibility, while Auto Visuals compiles to `semantic_*`
stages.

### Structural Proof Tournament

`vex_hyperframes/proof_program.py` combines ranked blueprints with scene-specific
strategies and five executable encoding families:

- `linear_trace`
- `split_register`
- `layered_flow`
- `focal_gate`
- `radial_evidence`

Each candidate has:

- deterministic program ID
- blueprint and stage family
- strategy and relation mode
- structural prior
- required relation IDs
- a separate signed production contract

The default tournament contains four candidates and supports up to eight.

### Executable Structural Encodings

`vex_hyperframes/composer.py` stamps the proof program into HTML and applies
encoding-specific layout behavior. Candidate differences affect spatial reasoning,
grouping, hierarchy, routing, gates, and evidence convergence. They are not palette
or typography variations.

### Blind Inverse Decoder

`vex_hyperframes/inverse_decoder.py` and `vex_hyperframes/vision_qa.py` separate
decoding from grading.

The model receives only chronological frames and a fixed relation ontology. It does
not receive:

- transcript
- intended thesis
- expected labels
- storyboard
- blueprint
- claim graph
- production contract

The decoder returns:

- inferred thesis
- visible objects
- typed directed relations
- communicated sequence
- confidence
- ambiguities
- unsupported visual claims

Only after this response is complete does deterministic code compare it with the
signed claim graph.

### Counterfactual QA

Two additional blind decodes test whether the visual grammar is necessary:

- relation ablation masks the proof-bearing region selected by the encoding family
- temporal scramble reorders sampled frames

If the same claim remains equally understandable after removing relation geometry or
destroying time order, the visual is likely label-driven rather than explanatory.

### Tournament Promotion

`vex_hyperframes/variants.py` promotes passing candidates by:

1. inverse-decoder score
2. required-relation coverage
3. sequence recovery
4. counterfactual sensitivity
5. conventional visual-quality score

A polished candidate with weak decoded proof loses to a clearer explanatory program.

### Repair Isolation

`renderers/hyperframes_renderer.py` renders every original proof candidate without
cross-candidate mutation. Bounded repair begins only when the complete untouched
tournament has no passing candidate.

Repair outputs receive separate IDs and artifact directories. The original failed
candidate remains inspectable.

## Runtime Transparency

Auto Visuals records:

- claim-graph signature
- tournament signature
- proof candidate count
- program IDs, blueprints, strategies, and encodings
- estimated render count
- blind-decoder threshold
- counterfactual configuration
- selected proof program
- decoded relation coverage
- counterfactual score
- all failed candidates and exact reasons

Counterfactual frames and decodes are persisted inside each variant QA bundle.

## Configuration

```env
HYPERFRAMES_PROOF_CANDIDATE_COUNT=4
HYPERFRAMES_ENABLE_VISION_QA=true
HYPERFRAMES_ENABLE_COUNTERFACTUAL_QA=true
HYPERFRAMES_BLIND_DECODER_MIN_SCORE=0.68
HYPERFRAMES_MIN_QUALITY_SCORE=0.78
```

`HYPERFRAMES_PROOF_CANDIDATE_COUNT` is capped at eight. Increasing it improves search
coverage but increases render and provider cost.

## Improvement To Vex

The previous system could verify that correct labels appeared and that motion existed,
yet still accept a visual whose actual relationship was vague. Visual Proof Search
changes the acceptance question from:

> Does this render resemble the requested concept?

to:

> Can an evaluator recover the signed claim from the render without being told the
> answer, and does that understanding depend on the authored visual structure?

This directly targets irrelevant diagrams, decorative arrows, disconnected cards,
generic template filler, and polished scenes that repeat transcript words without
explaining them.

## Deliberate Limits

- Blind decoding requires a compatible vision model and provider credentials.
- Counterfactual masks are deterministic encoding-aware approximations, not full
  object segmentation.
- More candidates and three blind decodes per candidate increase runtime and API cost.
- Unsupported or vague source material is still rejected rather than visually
  completed with invented content.

## Release Verification

- full repository suite: `304 passed`
- focused HyperFrames and Auto Visuals suite: `77 passed`
- Python compile checks: passed
- npm production dependency audit: zero known vulnerabilities
- real local render: `1280x720`, `4.0s`, four sampled QA frames
- promoted program: `progressive_stack` with `layered_flow`
- local semantic score: `0.9279`
- final quality score: `1.0`
