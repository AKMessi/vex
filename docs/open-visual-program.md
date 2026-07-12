# Open Visual Program

Vex Auto Visuals use Open Visual Program v1 as the shared executable scene
contract for HyperFrames and Remotion. The model controls composition, metaphor,
geometry, visual primitives, and choreography. Vex controls evidence, validation,
resource use, deterministic execution, and publication QA.

## Pipeline

1. Opportunity planning produces a signed `VisualExplanationIR` containing
   transcript evidence, facts, objects, relations, required labels, and forbidden
   content.
2. Generative authoring asks the configured reasoning model for distinct complete
   scene graphs. Templates are examples and fallback macros, not the output format.
3. Every candidate is normalized, signed, and checked against the packaged
   `open_visual_program.schema.json` authority.
4. Semantic validation rejects unknown evidence bindings, invented copy, missing
   objects or relations, static semantic elements, unsafe asset references,
   invalid geometry, excess overlap, and resource-budget violations.
5. Deterministic candidates are always generated. Invalid or unavailable model
   output therefore lowers creative variety without making Auto Visuals fail.
6. The tournament scores grounding, object and relation coverage, semantic motion,
   mechanism fitness, novelty, layout safety, and resource cost.
7. HyperFrames assigns distinct open programs to proof variants and renders them
   through its existing visual critics and final independent gate. Semantic
   fitness contributes to winner selection.
8. Remotion embeds the selected open program and tournament in its signed scene
   artifact and interprets the graph with frame-driven React components.

## Trust Boundary

Every non-decorative visual element has a binding to an exact `fact_id`,
`object_id`, or `relation_id`. Visible copy must be supported by transcript or
explicit title evidence. A valid JSON shape is not sufficient: the runtime also
recomputes the program signature and evidence signature before execution.

Remote assets, executable code, dynamic imports, network requests, filesystem
access, wall-clock animation, and unseeded randomness are not part of the Open
Visual Program language. This gives the model design control without giving
generated content process authority.

## Scene Language

Open Visual Program supports normalized layout, semantic bindings, constraints,
relations, repeated primitives, and motion tracks. Current element primitives are:

- text, token, shape, group, path, connector, icon, metric, and chart
- particle, mask, and evidence-bound local image

Motion tracks support opacity, translation, scale, rotation, progress, emphasis,
and blur. Each track carries a semantic intent and ordered keyframes. Both
renderers derive animation only from timeline progress, FPS, program inputs, and
the deterministic seed.

## Typed Repair

`apply_open_visual_patch()` accepts only bounded operations:

- move or resize an existing element
- replace evidence-grounded text
- update an allowlisted style field
- update an existing motion track
- remove decorative elements
- update the concept description

The patched program is re-signed and must pass the complete schema, evidence,
geometry, motion, and resource validation suite. Unknown targets, arbitrary CSS,
and unsupported operations are rejected.

## Configuration

| Variable | Default | Purpose |
|---|---:|---|
| `OPEN_VISUAL_PROGRAM_ENABLED` | `true` | Enables shared open-program compilation |
| `OPEN_VISUAL_PROGRAM_LLM_AUTHORING` | `true` | Allows evidence-bound model authoring |
| `OPEN_VISUAL_PROGRAM_CANDIDATES` | `3` | Candidate budget, capped at four |
| `OPEN_VISUAL_PROGRAM_AUTHORING_ATTEMPTS` | `2` | Initial model attempt plus one repair |
| `OPEN_VISUAL_PROGRAM_MIN_SCORE` | `0.78` | Static acceptance floor before rendering |

## Artifacts

Auto Visuals manifests include the selected program ID, authoring mode, candidate
count, semantic-fitness score, tournament signature, novelty fingerprint, and
model rejection diagnostics. Renderer job directories preserve the complete
program and the renderer-specific signed scene artifact. HyperFrames additionally
stores one `open_visual_program.json` per rendered proof candidate.

## Compatibility

Legacy Remotion scene families, HyperFrames Visual World programs, Scene Program
V2, bespoke scenes, and templates remain available. They are compatibility
fallbacks when Open Visual Program is disabled or cannot be compiled. Existing
manual visual commands are not forced through model authoring.
