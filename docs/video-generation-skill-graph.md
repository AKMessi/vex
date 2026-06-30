# Video Generation Skill Graph

`generate_video` now uses a deterministic Video Generation Skill Graph before
semantic cinematography, native motion, portfolio judging, and final QA.

The public tool contract stays simple: callers can provide a prompt, optional
script, aspect, duration, quality, voice, and render options. Vex owns the hard
production architecture internally.

## Pipeline Placement

```text
request
  -> script planner
  -> script director
  -> audio-first beat graph
  -> director crew contracts
  -> video generation skill graph
  -> semantic cinematographer
  -> native motion plan
  -> portfolio judge
  -> HyperFrames project writer
  -> render and generated-video QA
```

The graph is built after the director crew so it can consume source-grounded
beat contracts, and before the cinematographer so it can constrain all visual
candidate generation.

## Contracts

The graph writes `VIDEO_SKILL_GRAPH.json` with:

- selected production skill, such as architecture demo, metric proof,
  product walkthrough, decision story, process trace, narrative proof,
  technical explainer, or quote manifesto
- per-beat arc role
- per-beat visual skill and target scene type
- renderer route
- required slots and missing slots
- source-grounded labels and metric facts
- semantic frame seed
- HyperFrames skill slices
- proof encoding and visual-world medium priors
- QA floor, motion technique, camera move, transition intent, and effect stack
- continuity ledger for recurring subjects, renderer routes, and world memory

## Runtime Use

The semantic cinematographer uses the assignment as the first candidate for each
beat, then keeps the older heuristic candidates as bounded fallbacks. This
preserves resilience while giving the best path a clear architecture.

The per-beat tournament rewards candidates that match the assignment's scene
type, visual-world medium, proof encoding, label coverage, and carried skill
contract.

The native motion compiler consumes the same assignment for technique, camera
move, transitions, energy, and effect stack. Motion therefore follows the visual
skill rather than generic beat type alone.

The portfolio judge and generated-video QA treat the graph as a release
condition. Missing graphs, failed graphs, incomplete assignments, or skill
coverage below the production floor fail the generated video instead of being
hidden as metadata.

## Artifact Surface

Generated projects now include:

- `VIDEO_SKILL_GRAPH.json`
- skill graph summary in `manifest.json`
- skill graph evidence in `generated_video_qa.json`
- beat skill notes in `STORYBOARD.md`
- production skill and continuity summary in `DESIGN.md`
- skill metadata on fallback HTML scenes
- skill assignment metadata on compiled HyperFrames compositions

## Design Rules

- Do not expand the public LLM-facing tool schema for ordinary use.
- Do not invent metrics, services, UI states, branches, or outcomes to satisfy a
  skill.
- Prefer HyperFrames for semantic explainer worlds; reroute future specialist
  renderers only when the signed contract demands them.
- Keep legacy heuristic candidates as fallbacks, but score and promote
  skill-contract matches first.
- Treat blueprints as search priors, not final visuals.
- Keep every beat frame-addressable, seekable, and inspectable.
