# Remotion Visual Compiler Architecture

Vex treats Remotion as a deterministic visual compiler and renderer, not as a
free-form JSX generation target. Reasoning can propose visual concepts, but
typed contracts own evidence, capabilities, geometry, motion, execution, media
encoding, and publication.

## Production Pipeline

1. `visual_explanation.py` compiles transcript evidence into facts, objects,
   relations, required labels, forbidden claims, and an explicit render or
   reject decision.
2. The communication and concept-search layers create six materially distinct
   visual directions with premise, mechanism, proof, and final-hold reference
   frames.
3. `vex_visuals.open_visual_program` authors bounded Open Visual Program
   candidates. Every semantic element binds to a supplied fact, object, or
   relation; every accepted candidate is validated and signed.
4. `vex_remotion.compiler` selects a supported scene family, carries the exact
   Open Visual Programs into `RemotionSceneProgram` v4, and compiles the selected
   program into signed SceneGraph v2.
5. For a final render with multiple candidates, the Remotion adapter compiles
   each candidate without allowing substitution, runs structural QA, and renders
   three contact-sheet frames for every valid candidate in one browser session.
6. The rendered search score combines measured aesthetics (68 percent), semantic
   grounding (16 percent), and structural QA (16 percent). The signed preflight
   report selects one exact program; the final compiler cannot silently rerun
   the static tournament.
7. The SceneGraph runtime solves constraints and renders specialized DOM/SVG
   primitives. Remotion animation derives only from frame number, FPS, signed
   keyframes, and deterministic inputs.
8. Compile-time structural QA and 16-sample temporal render QA must pass before
   Auto Visuals can publish the asset.
9. Final opaque visuals are ProRes HQ masters. Transparent overlays are ProRes
   4444 masters. Preview renders remain half-resolution H.264 for bounded repair
   search, and a winning preview is always rerendered at final fidelity before
   publication.

## Contract Stack

### Open Visual Program v1

Open Visual Program is the authored semantic language. It contains normalized
elements, evidence bindings, relations, constraints, semantic motion tracks,
resource bounds, and the creative concept. It cannot contain executable code,
remote asset fetches, unseeded randomness, or filesystem access.

### SceneGraph v2

`vex-scene-graph-v2` is the renderer capability plan. It adds:

- an explicit canvas, safe area, design system, typography floor, and depth model
- typed primitives with requested and fallback renderer backends
- routed semantic relations and attachment telemetry
- prioritized constraints and a deterministic motion graph
- local-only asset declarations with provenance
- a quality contract and required telemetry sample times
- a SHA-256 signature over the canonical graph

The packaged JSON Schema is `vex_visuals/scene_graph_v2.schema.json`.

### Remotion Scene Program v4

The scene program is the final immutable input to React. It includes source
evidence, the selected Open Visual Program, all bounded candidates and tournament
diagnostics, the signed SceneGraph, responsive layout data, semantic beats, and
quality thresholds.

### Media Contract v1

The render request declares codec, container, render pixel format, expected
encoded pixel format, ProRes profile, alpha state, and BT.709 color metadata.
Vex probes the output and rejects codec, pixel-format, alpha, or final-master
color drift.

## Runtime Capabilities

The capability registry maps authored elements to specialized primitives:

| Primitive | Primary runtime | Purpose |
| --- | --- | --- |
| `data_chart` | SVG | Evidence-bound quantitative or state marks |
| `graph_node` | SVG/DOM | Framed mechanism states and resolved outcomes |
| `kinetic_text_run` | DOM | Measured, clipped semantic typography |
| `metric_mark` | DOM | Quantitative focal marks |
| `masked_media` | DOM | Provenance-bound local media and masks |
| `vector_icon`, `vector_path`, `vector_shape` | SVG | Icons, routes, and geometry |
| `semantic_token`, `text_block`, `group` | DOM | Supporting semantic structure |
| `particle_field` | Declared Skia lane, DOM fallback | Bounded decorative fields |

The schema reserves bounded Skia, Three, Rive, and Lottie backend lanes, but the
signed production registry currently exposes DOM/SVG primitives plus one
Skia-requested particle primitive with a DOM fallback. Three, Rive, and Lottie
cannot execute until a primitive and bounded fallback are explicitly registered.

The constraint solver supports safe-area containment, alignment, row/column
distribution, overlap avoidance, minimum gaps, and parent-first containment.
For containment, the first target is the container and every subsequent target
is fitted within its padded bounds. Connectors use direct, cubic, or orthogonal
routing and animate their authored path progress.

## Quality Gates

### Structural QA

`remotion-structural-qa-v1` executes before Chromium for the final program and
before candidate contact sheets. It verifies:

- SceneGraph schema and signature
- solved safe-area containment
- overlap and minimum-gap constraints
- exact required object and relation binding coverage
- text capacity at the runtime font floor
- relation endpoint separation and attachment
- animated translation/scale safety at telemetry times
- bounded simultaneous semantic motion

Hard structural issues prevent rendering. Reports include the solved normalized
layout so geometry decisions are auditable.

### Temporal Render QA

`remotion-render-qa-v5` samples up to 16 frames drawn from uniform coverage,
authored beats, SceneGraph telemetry times, and the final hold. It measures:

- semantic score, contrast, occupancy, entropy, and aesthetic hierarchy
- motion delta, changed area, normalized motion rate, acceleration, and settle
- mean luminance, abrupt global luminance changes, and alternating flicker
- alpha occupancy for transparent overlays
- initial readability for opaque inserts and populated final state for all media

An intentionally transparent entrance is allowed for an alpha overlay; an empty
or illegible resolved overlay is not.

## Rendering and Caching

`renderers/remotion_runner.mjs` opens one Remotion browser and reuses it for
composition selection, candidate stills, and the requested media render. Bundles
are content-addressed by the entry JSX, SceneGraph runtime, and exact package
lock. Valid bundles are reused across independent jobs from Vex's user-data
cache; incomplete or mismatched entries are rebuilt atomically.

Candidate still mode renders at most eight supplied programs and at most twelve
fractions per program. The current compiler supplies six concept treatments and
the rendered preflight samples three frames from each.

## Media Policy

| Fidelity | Output | Codec/profile | Pixel format | Alpha | Color contract |
| --- | --- | --- | --- | --- | --- |
| Preview | `visual.mp4` | H.264 | `yuv420p` | No | BT.709 matrix |
| Final opaque | `visual.mov` | ProRes HQ | `yuv422p10le` | No | BT.709 matrix, primaries, transfer |
| Final overlay | `visual.mov` | ProRes 4444 | render `yuva444p10le`, encoded `yuva444p12le` | Yes | BT.709 matrix, primaries, transfer |

Frames are rendered as PNG before encoding. If the platform ProRes muxer omits
QuickTime color metadata, Vex performs an atomic stream-copy remux with explicit
BT.709 tags and probes the result again. It never fixes a media-contract failure
by transcoding to a weaker codec.

## Trust and Failure Boundaries

- Agent-authored executable JSX never crosses the boundary.
- Numeric facts without source provenance fail compilation.
- A candidate whose program identity changes during compilation is rejected,
  not substituted.
- Remote scene assets and undeclared renderer capabilities fail validation.
- A preview asset cannot become the published asset.
- Structural, temporal, media, or alpha contract failures remain hard failures.
- Provider outages reduce concept variety but deterministic candidates remain.
- Existing legacy Remotion scene families are compatibility fallbacks only when
  no valid SceneGraph is available.

## Artifacts

Each final job can contain:

- `remotion_spec.json`
- `remotion_scene_program.json`
- `remotion_compiler_report.json`
- `remotion_structural_qa.json`
- `candidate_preflight/remotion_candidate_preflight.json`
- `candidate_preflight/remotion_preview_frames/**`
- `render_request.json`
- `remotion_result.json`
- `remotion_render.log`
- `remotion_color_metadata.log` when normalization was required
- `remotion_qa.json`
- `remotion_qa_frames/**`
- `remotion_metadata.json`
- `visual.mov` for final media or `visual.mp4` for preview media

The candidate preflight report and SceneGraph are signed. Renderer metadata
records bundle identity/cache state, selected program identity, structural and
temporal QA, the media contract, color normalization, and probed stream facts.

## Configuration

| Variable | Default | Bound or behavior |
| --- | ---: | --- |
| `OPEN_VISUAL_PROGRAM_CANDIDATES` | `6` | 1-6 complete candidates |
| `OPEN_VISUAL_PROGRAM_AUTHORING_ATTEMPTS` | `2` | Initial attempt plus one repair |
| `OPEN_VISUAL_PROGRAM_MIN_SCORE` | `0.78` | Static grounding floor |
| `REMOTION_RENDER_TIMEOUT_SEC` | `0` | Disables the outer process timeout |
| `REMOTION_RENDER_CONCURRENCY` | empty | Remotion default or explicit number/percentage |

Bundle cache location is an internal user-data path and does not depend on the
current working directory.
