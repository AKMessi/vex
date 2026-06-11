# Feedback-Driven Automation and HyperFrames Architecture Report

## Status

Implemented, tested, committed, and pushed to `main`.

This release changes Vex from a collection of quality-biased automatic commands into
an explainable automation system with explicit user control, inspectable progress,
safe manual overrides, and a semantic compiler for generated HyperFrames visuals.

The HyperFrames change is architectural, not a larger template catalog. Automatic
generation no longer treats transcript text as copy for an attractive motion slide.
It must first prove that the source contains a supported explanation, compile that
explanation into typed objects and motion, and carry a signed contract through render
and QA.

## Commit Ledger

| Commit | Scope |
| --- | --- |
| `0f3dce5` | Coverage policies, configurable automation caps, density planning, SRT transcript handling, progress status |
| `4a1d505` | Exact-time manual visual insertion, FFmpeg scaling, renderer diagnostics |
| `fc74830` | Typed Auto Shorts normalization and accepted/rejected/draft output separation |
| `deec1d4` | HyperFrames semantic golden corpus and evaluation harness |
| `a1efe25` | Shared evidence-backed Visual Explanation IR |
| `39d0bf1` | Storyboard review, curated semantic blueprints, signed production contract, compiler |
| `681bbc3` | Deterministic grounded `semantic_*` renderer stages |
| `356120a` | Runtime and repository HyperFrames director skill packs |
| `e53d895` | Safe typed bespoke scene authoring |
| `d061851` | Semantic QA, optional vision QA, bounded repair, strict variant promotion |
| `e020701` | Mandatory Auto Visuals compiler gate, continuity, real source-frame grounding |

## User-Visible Outcome

Vex now behaves differently in five important ways:

1. Generic automation remains quality-first and may return fewer results.
2. Explicit counts increase coverage pressure and are recorded in the run manifest.
3. Exact-time user assets bypass transcript scoring but not path, timing, render, or
   media validation.
4. Long jobs expose an incremental `run_status.json` with phase history and counts.
5. HyperFrames visuals must explain source-backed content. Generic filler, invented
   metrics, fake interfaces, and unsupported relationships are rejected before
   rendering.

## Before and After

### Previous HyperFrames path

```text
transcript beat
  -> LLM chooses template and copy
  -> renderer creates visually plausible HTML
  -> image-level quality score
  -> least-bad candidate may be selected
```

This path could make polished but irrelevant visuals because template fit and surface
quality were stronger constraints than explanatory truth.

### Current HyperFrames path

```text
transcript / imported SRT / semantic frame
  -> evidence spans and grounded facts
  -> VisualExplanationIR
  -> storyboard and storyboard review
  -> role-constrained blueprint selection
  -> signed production contract
  -> deterministic semantic renderer spec
  -> safe composition or typed bespoke program
  -> semantic QA at four times
  -> local visual/motion QA
  -> optional vision critique
  -> bounded repair or rejection
  -> final timeline QA and composite
```

Every boundary emits machine-readable evidence. HTML generation is downstream of the
semantic acceptance decision, not the place where meaning is invented.

## Shared Automation Control Plane

### Coverage policy

`tools/automation.py` defines one policy vocabulary for Auto B-roll and Auto Visuals:

- `quality_only`: default for generic commands; keep only strong candidates.
- `target_count`: make a stronger effort to reach a requested count using the best
  valid candidates available.
- `exact_count`: strongest count request, while preserving hard safety, render, and
  QA rejection.

Explicit natural-language counts compile to `target_count`. Manifests and messages
record:

- `requested_count`
- `selected_count`
- `rejected_count`
- `rejection_reasons`

This removes the previous ambiguity where a user asked for a count but could not tell
whether the planner ignored it, ran out of candidates, or rejected renders.

### Capacity and density

- Auto B-roll has a configurable production cap through
  `AUTO_BROLL_MAX_OVERLAYS`, default `24`.
- Auto Visuals has a configurable cap through
  `AUTO_VISUALS_MAX_VISUALS`, default `32`.
- Auto Visuals density accepts `sparse`, `balanced`, `dense`, and
  `chapter_coverage`.
- Long clips default toward chapter-aware planning rather than a fixed short-video
  count.

Caps are still hard resource boundaries. Density changes planning coverage, not
semantic or renderer acceptance.

### Progress and planning preview

Auto B-roll and Auto Visuals create `run_status.json` at the beginning of the run and
update it incrementally by phase.

Auto Visuals phases:

```text
transcript_load
candidate_scoring
planning
director
semantic_compile
render
qa
final_composite
complete
```

The initial payload includes planned count, estimated render count, renderer
preference, expected slow steps, output bundle path, transcript source, and timed
segment count. The file retains the last 80 phase events.

## Transcript and No-Audio Behavior

`tools/transcript_utils.py` treats a valid project-local SRT as a timed transcript
source even if the video has no audio stream.

Automation reports:

- transcript source
- usable timed segment count
- selected and rejected counts
- concrete rejection reasons

The screen-recording/tutorial path also samples source frames and classifies them as
screen/slide, busy detail, rich camera, talking head/simple, or flat/static. This
prevents low-emotion tutorial narration from being treated as low visual value and
helps Vex avoid replacing an already information-dense screen unnecessarily.

## Auto B-Roll Architecture

Auto B-roll now:

1. Loads an existing timed transcript before attempting transcription.
2. Builds a planning preview and run status.
3. Uses explicit coverage policy and requested count.
4. Scores candidate cards and searches configured providers.
5. Preserves the hard provider, timing, path, and final-composite gates.
6. Writes count and rejection accounting into the manifest and user response.

The quality-first default is unchanged. The difference is that user intent and
rejections are now observable and count pressure is explicit.

## Auto Visuals Architecture

Auto Visuals retains the existing video-level narrative program:

- chapters
- concept memory
- continuity groups
- visual episodes and beats
- transition contracts
- creative graph signals

It now adds a semantic compiler boundary after Auto Visuals Director and before any
HyperFrames render.

Non-HyperFrames renderers retain their specialist paths. HyperFrames candidates must
compile successfully or they are dropped with exact compiler issues.

## Visual Explanation IR

`visual_explanation.py` is the source-of-truth contract shared by semantic planning,
storyboarding, rendering, and QA.

The IR contains:

- evidence spans with source type and timing
- grounded facts with provenance and confidence
- typed explanation objects with semantic roles
- timed explanation beats
- viewer question, thesis, and takeaway
- required visible labels
- forbidden content
- render or reject policy
- explicit rejection reasons

Numeric claims receive an additional provenance check. A number, percentage, duration,
count, multiplier, or unit that cannot be found in source evidence is not renderable.

The compiler also rejects:

- unsupported explanation structures
- too few grounded objects
- insufficient semantic motion
- generic placeholder labels
- requested labels without source provenance
- visuals with no grounded facts

This is the primary defense against polished nonsense.

## Semantic Scene Taxonomy

The compiler supports 11 explanation types:

- `architecture_flow`
- `causal_intervention`
- `decision_branch`
- `evidence_backed_quote`
- `grounded_interface_walkthrough`
- `guided_process`
- `matched_state_transform`
- `metric_delta`
- `metric_intervention`
- `metric_proof`
- `narrative_progression`

Ambiguous content compiles to `none` and is rejected. The renderer does not silently
turn it into a quote card or generic concept map.

## Curated Blueprint Library

`vex_hyperframes/blueprints.py` contains 23 curated blueprints. Every scene type has
at least two options, and causal explanations have a third direct-trace option for
sources that state cause and effect without naming a separate mechanism.

Each blueprint declares:

- required semantic roles
- minimum object count
- layout thesis
- motion spine
- dynamic visual devices
- anti-patterns
- source-language selection tags
- priority

Selection is prerequisite-driven. A blueprint cannot be selected because its style
sounds appropriate when its semantic roles are missing.

Legacy template names remain for manual compatibility. Automatic semantic work emits
only `semantic_*` stage families.

## Storyboard and Production Contract

Before composition, `vex_hyperframes/storyboard.py` turns explanation beats into timed
panels. Review fails when:

- fewer than two semantic beats exist
- required objects are omitted
- a panel focuses an unknown object
- timing is invalid
- the resolved state is not held

`vex_hyperframes/production_contract.py` then signs the accepted plan with:

- scene type and blueprint ID
- thesis, viewer question, and takeaway
- required labels and object IDs
- required semantic motion and devices
- final-frame screenshot test
- forbidden content
- scene-specific quality floor
- SHA-256 semantic signature

The signature binds scene type, blueprint, facts, objects, and beats. It is preserved
through renderer metadata and QA so the final asset can be traced to the explanation
that authorized it.

## Deterministic Semantic Rendering

`vex_hyperframes/composer.py` implements grounded semantic stage families for metrics,
causality, process routes, architecture, state transforms, interfaces, decisions,
narrative progression, and exact quotes.

These stages:

- render labels from typed explanation objects
- use motion to express the contracted relationship
- hold a readable resolved frame
- expose deterministic seek controls through
  `window.__timelines[compositionId]`
- preserve semantic metadata
- avoid synthetic progress values, statuses, and unsupported UI copy

An unknown template raises an error. There is no automatic quote-card fallback.

## Real Source-Frame Grounding

When source analysis identifies a screen recording or busy interface and the scene
compiles as `grounded_interface_walkthrough`, Auto Visuals extracts a real PNG frame
from the working video.

The composer:

1. Resolves the image against explicit allowed roots.
2. Requires a supported local image suffix.
3. Enforces a 20 MB size limit.
4. Embeds the image as a self-contained data URI.
5. Uses source-backed action/result labels as an annotation rail.

The generated HTML receives no unrestricted filesystem path and no remote URL. When
the asset fails validation, the stage uses the deterministic grounded interface
layout without claiming that a real source frame was used.

## Cross-Scene Continuity

The visual narrative program already carried recurring concepts and episode context.
The compiler now turns that memory into renderer-level continuity metadata:

- continuity group
- concept IDs
- primary concept color
- motif
- episode ID

The concept color is applied to the semantic scene theme and all continuity metadata
is preserved in the final composition. This gives repeated concepts a stable visual
identity without allowing style continuity to override evidence.

## HyperFrames Director Skill Pack

There are two skill layers:

1. `vex_hyperframes/skill_pack.py` provides runtime guardrails and scene-specific
   guidance to the composer.
2. `skills/vex-hyperframes-director/` provides a repository skill with workflow,
   contracts, blueprint guidance, motion/layout rules, QA/repair policy, examples,
   and agent metadata.

The skill's non-negotiable rules include:

- build explanations, not themed slides
- never invent metrics, entities, interface states, risks, or outcomes
- never hide a failed semantic request behind a generic fallback
- never promote visual polish over grounding or contract failure
- preserve deterministic timeline seekability
- keep generated compositions local-only
- add golden fixtures for every new explanation pattern

This is deliberately different from a style prompt pack. It codifies architectural
decisions and acceptance criteria.

## Safe Bespoke Authoring

Some grounded scenes need a composition beyond the curated stage implementations.
`vex_hyperframes/authoring.py` provides a bounded typed JSON scene language for:

- grounded text
- object placement
- connectors
- emphasis
- deterministic motion
- final-state holds

The authoring compiler validates copy, object IDs, motion targets, timing, and source
grounding before producing HTML. `vex_hyperframes/safety.py` rejects remote URLs,
executable hooks, and unsafe HTML patterns.

This provides custom composition without accepting arbitrary LLM-authored JavaScript,
Node code, shell commands, or filesystem access.

## Semantic, Motion, and Vision QA

Renderer QA runs in layers:

1. Production-contract integrity
2. Required-label coverage
3. Required-object coverage
4. Storyboard and semantic-beat coverage
5. Resolved-frame screenshot requirement
6. Four-time animation inspection
7. Local contrast, occupancy, dead-space, edge, text-density, and motion scoring
8. Optional provider-backed vision critique

The four semantic samples represent establishment, mechanism/intervention, resolution,
and final hold. Adjacent frames must show meaningful semantic state change, not only
pixel noise.

`HYPERFRAMES_QA_MODE` controls whether missing vision critique is informational or
strict. Local semantic checks remain available without provider credentials.

## Bounded Repair and Promotion

Repair is allowed only for diagnosable presentation failures:

- grounded copy placement
- missing grounded object coverage
- insufficient semantic motion
- insufficient final hold

The repair variant uses the same IR, blueprint, production contract, and semantic
signature. It cannot add facts or change the explanation.

Renderer rerouting is recommended when another engine owns the problem:

- Manim for formulas, geometry, and symbolic graphs
- FFmpeg/source assets for real interface imagery
- Blender for actual 3D spatial or model requirements

Variant selection now requires a passing candidate. A failed visual is not promoted
because every other candidate failed too.

## Evaluation Corpus

`tests/fixtures/hyperframes_semantic_cases.json` is a golden semantic corpus covering
renderable and rejectable cases. The evaluation harness checks:

- scene classification
- required-label retention
- forbidden-label absence
- numeric provenance
- compiler acceptance/rejection
- valid deterministic composition
- semantic metadata

The corpus is the extension point for future scene families. A new pattern should not
enter production without a source-backed positive fixture and a nearby negative
fixture.

## Manual Visual Placement

`add_visual_asset` provides exact-time insertion for project-local:

- HTML
- MP4/MOV/M4V/WebM
- GIF
- PNG/JPEG/WebP/BMP

The tool validates project-safe paths, file type, size, timing, and composition mode.
Images and GIFs are normalized with FFmpeg. Local HTML is checked for remote references
and Node/shell execution hooks, copied into an isolated render bundle, and rendered
through the local HyperFrames CLI.

Manual placements bypass transcript scoring by design. They remain timeline operations,
so undo, redo, and rebuild replay the exact validated overlay.

## Auto Shorts Reliability

Auto Shorts now normalizes model-authored numeric fields before rendering:

- duration
- quality floor
- source-range speed
- zoom
- intensity-like values
- range indices and timing

Named values such as `high`, `medium`, `fast`, or `slow` map to bounded numeric
values instead of reaching `float()` and crashing.

Each bundle has:

```text
drafts/
accepted/
rejected/
```

Rendered files are created in `drafts/`, promoted to `accepted/` only after transcript
and output QA, and retained in `rejected/` with exact reasons when they fail. User
messages distinguish rendered, accepted, and rejected counts and point to retained
artifacts.

## Scaling and Export

`upscale_video` is a practical FFmpeg export path:

- Lanczos scaling
- `fit`: preserve aspect ratio and pad
- `fill`: preserve aspect ratio and crop
- `stretch`: exact dimensions with possible distortion, only when requested
- output path validation
- size estimation and disk-space check
- artifact history

The command explicitly reports that this is resize/export scaling, not AI
super-resolution. Existing presets such as `youtube_1080p` remain supported.

## Renderer Diagnostics

`vex renderers doctor` reports:

- HyperFrames CLI path and aggregate readiness
- Node.js version and major version
- FFmpeg path and version
- Manim path and version
- Blender path and version
- renderer capability payload

HyperFrames readiness requires the local CLI, Node.js 22+, and FFmpeg. Renderer
failures preserve exact log paths in render metadata.

## Manifests and Artifacts

Auto Visuals manifests now include:

- coverage and density controls
- transcript source and timed segment count
- planning preview
- creative graph and visual narrative program
- Auto Visuals Director report
- HyperFrames compiler report
- Visual Explanation IR
- storyboard and production contract
- semantic continuity
- source asset grounding
- renderer and variant metadata
- semantic, vision, rendered, and final timeline QA
- selected overlays
- render and rejection failures

HyperFrames variant directories retain:

- generated HTML
- validation and lint output
- render log
- preview frames
- composition metadata
- semantic QA JSON
- optional vision QA JSON
- typed bespoke scene program when used
- variant report and final selection provenance

This changes a bad result from "the visual looked wrong" into an inspectable failure at
a named architecture boundary.

## Safety Boundaries

The production safety model now includes:

- project-safe local path resolution
- file type and size limits
- no remote URLs in manual or generated HTML
- no arbitrary shell or Node execution in HTML
- typed Blender specs instead of raw generated Blender Python
- typed bespoke HyperFrames DSL instead of raw generated application code
- deterministic seekable timelines
- subprocess argv construction
- explicit renderer logs
- hard semantic rejection before rendering
- hard QA rejection before compositing

Explicit user counts and manual timing can increase effort or bypass relevance
scoring. They cannot bypass these safety boundaries.

## Why This Improves Vex

### Relevance

Visuals are selected by explanation type and grounded role prerequisites, not by
template aesthetics. Unsupported content is skipped rather than decorated.

### Truthfulness

Facts, labels, quotes, interface states, and numeric claims retain source provenance.
The semantic signature makes the accepted explanation traceable through output QA.

### Visual quality

The renderer receives a structured visual argument with a motion spine, not a loose
paragraph. Curated blueprints and bespoke typed composition can spend visual complexity
on the relationship that matters.

### Consistency

Concept color, motif, object identity, transition contracts, and final-state holds
survive across scenes and variants.

### User control

Counts, density, exact-time assets, scaling mode, renderer choice, and QA settings are
explicit. Defaults remain conservative.

### Debuggability

Planning, compiler, renderer, semantic QA, vision QA, final QA, and composite failures
are separated and persisted with exact artifacts.

### Operational reliability

Typed numeric normalization, retained rejected outputs, incremental run status, path
validation, dependency diagnostics, and strict promotion remove several silent or
misleading failure modes.

## Configuration

Relevant environment variables:

```env
AUTO_BROLL_MAX_OVERLAYS=24
AUTO_VISUALS_MAX_VISUALS=32
HYPERFRAMES_CLI_PATH=hyperframes
HYPERFRAMES_LINT_TIMEOUT_SEC=90
HYPERFRAMES_RENDER_TIMEOUT_SEC=0
HYPERFRAMES_RENDER_QUALITY=
HYPERFRAMES_VARIANT_COUNT=3
HYPERFRAMES_PROOF_CANDIDATE_COUNT=4
HYPERFRAMES_QA_MODE=hybrid
HYPERFRAMES_ENABLE_VISION_QA=true
HYPERFRAMES_ENABLE_COUNTERFACTUAL_QA=true
HYPERFRAMES_VISION_MODEL=
HYPERFRAMES_BLIND_DECODER_MIN_SCORE=0.68
HYPERFRAMES_MIN_QUALITY_SCORE=0.78
```

## HyperFrames Visual Proof Search Extension

HyperFrames automation now compiles a signed visual claim graph before rendering.
Curated blueprints are ranked as priors and expanded into a structural proof-program
tournament. The default four candidates use different spatial and motion encodings,
not cosmetic themes.

Rendered frames are decoded by a blind inverse evaluator that does not receive the
transcript, intended thesis, expected labels, storyboard, blueprint, claim graph, or
production contract. Deterministic grading then compares the inferred thesis, objects,
directed relations, and sequence with the signed graph.

Relation-ablation and temporal-scramble probes test whether comprehension depends on
the authored visual grammar. Promotion prioritizes decoded proof and counterfactual
sensitivity before conventional polish. Bounded repair runs only after all original
candidates fail, so one candidate can no longer mutate another candidate's search
path.

The complete design and artifact contract are documented in
[`hyperframes-visual-proof-search.md`](hyperframes-visual-proof-search.md).

## Verification Strategy

The implementation includes focused tests for:

- coverage-policy intent compilation
- count caps above the previous limits
- SRT-only timed transcript loading
- exact-time manual insertion and replay
- unsafe local asset rejection
- scale-mode FFmpeg filters
- renderer diagnostics
- Auto Shorts named numeric values and output separation
- semantic IR grounding and numeric provenance
- 12-case HyperFrames golden evaluation
- blueprint selection and compiler rejection
- deterministic semantic composition
- safe bespoke authoring
- skill-pack contract
- four-time semantic QA
- optional vision QA behavior
- strict failed-variant rejection
- Auto Visuals compiler integration
- source-frame embedding and path denial
- continuity preservation

### Verification results

- Full repository suite: `304 passed`
- Focused HyperFrames and Auto Visuals suite: `77 passed`
- Python compile checks: passed
- `git diff --check`: passed
- npm production dependency audit: zero known vulnerabilities
- Real local HyperFrames smoke render: `1280x720`, `4.0s`, four QA frames,
  semantic/local QA passed at `0.9279`, `progressive_stack` / `layered_flow`
  promoted with quality score `1.0`
- Renderer doctor: HyperFrames, Node.js 22, FFmpeg, and Blender available; Manim
  not installed on the verification machine
- Remaining warning: upstream `google.genai` deprecation warning under Python 3.14

The release check covers the complete repository `pytest` suite plus Python compile
checks and whitespace validation.

## Deliberate Limitations

- Scaling is not AI super-resolution.
- Blind inverse-decoder and counterfactual QA require compatible provider
  credentials/model access. Hybrid mode records their unavailability and continues
  with deterministic semantic QA.
- HyperFrames cannot make an unsupported or vague idea explanatory; it rejects it.
- Legacy manual templates remain for backwards compatibility.
- Renderer reroute recommendations do not fabricate missing assets or silently change
  a strict renderer request.
- `run_status.json` is durable progress reporting, not yet a resumable background job
  queue. Durable job recovery remains future platform work.

These limits are explicit so Vex fails honestly instead of appearing successful with
an irrelevant or unverifiable result.
