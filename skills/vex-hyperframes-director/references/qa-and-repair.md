# QA And Repair

## QA Order

1. Contract integrity
2. Evidence and copy provenance
3. Required-label coverage
4. Required-object coverage
5. Semantic beat coverage
6. Resolved-frame screenshot test
7. Animation continuity across sampled times
8. Layout, contrast, occupancy, edges, text density, and motion amount
9. Blind inverse decoding
10. Relation-ablation and temporal-scramble counterfactuals
11. Layout, contrast, occupancy, edges, text density, and motion amount

Do not promote a scene that fails steps 1 through 10.

## Time Samples

Inspect at least:

- establishment
- mechanism or intervention
- resolution
- final hold

Compare adjacent samples to confirm meaningful state change, not only pixel noise.

## Blind Decoder

The decoder receives chronological frames and a fixed relation ontology only. It must
not receive the transcript, intended thesis, expected labels, storyboard, blueprint,
claim graph, or production contract.

After decoding, compare:

- inferred thesis against the signed thesis/takeaway
- decoded objects against claim-graph nodes
- decoded directed relations against required relation IDs
- decoded sequence against signed sequence node IDs

## Counterfactuals

- Relation ablation masks the proof-bearing region selected by the structural encoding.
- Temporal scramble reorders sampled frames.
- A strong visual should lose relation or sequence confidence under the relevant perturbation.
- Persist both perturbed frames and decoded outputs.

## Repair Classes

Allow bounded repair for:

- missing required label already present in facts
- text overflow
- weak contrast
- clipped object
- insufficient final hold
- timing mismatch
- under-emphasized required object

Run bounded repair only after every untouched proof candidate has completed QA.

Do not repair by inventing content or changing evidence.

## Reroute Or Reject

Reroute to:

- Manim for formulas, geometry, symbolic graphs, or mathematical object continuity
- Blender for actual 3D spatial/model requirements
- source footage or local assets when the real interface/product is available

Reject when no renderer can satisfy the evidence-backed contract.

## Diagnostics

Persist:

- IR and validation
- storyboard and review
- blueprint selection
- production contract
- source spec
- generated HTML
- HTML validation
- lint and render logs
- sampled frames
- semantic QA
- visual QA
- repair attempts
- final rejection or promotion reason
