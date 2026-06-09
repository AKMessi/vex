# Vex Architecture Upgrade Plan

> Implementation note: the feedback-driven automation controls, manual visual
> insertion, scaling diagnostics, Auto Shorts QA hardening, and HyperFrames semantic
> compiler described by the completed 2026 upgrade are documented in
> [feedback-driven-automation-report.md](feedback-driven-automation-report.md).
> The remaining sections in this file describe longer-term platform work.

This plan is intentionally larger than a cleanup pass. It describes the upgrades that would move Vex from a capable terminal agent into a production-grade video automation platform with durable jobs, reproducible timelines, safer generated media, and clearer extension boundaries.

## Current Architecture Review

Vex already has a useful separation between the CLI, agent loop, provider adapters, editing tools, renderers, and project state. The strongest parts of the codebase are the deterministic intent compiler, the timeline rebuild path, renderer fallback strategy, and the transcript-aware visual planning pipeline.

The main limits are architectural rather than cosmetic:

- Tool execution is synchronous and tied to the REPL process, so long renders and network downloads do not have durable job recovery.
- Timeline operations are stored as loose dictionaries, which makes migrations, validation, and cross-version replay harder as features grow.
- Renderers share the same Python process boundary as orchestration, even though Hyperframes, Manim, Blender, FFmpeg, and LLM-generated code all have different failure and isolation profiles.
- Asset metadata is spread across timeline params, manifests, and artifacts, which makes garbage collection, deduplication, and provenance harder.
- Provider retries exist, but there is no central policy for budgets, rate limits, circuit breaking, or model capability routing.
- Tests cover important deterministic paths, but full media integration, crash recovery, and renderer isolation need a stronger test pyramid.

## Recent Hyperframes Upgrade

The first production slice of the renderer-quality roadmap is now in place for Hyperframes auto visuals:

- planned visuals compile into a typed design IR before HTML is authored
- art-direction profiles control palette, typography, density, safe areas, layout classing, and motion intensity
- each visual can render multiple deterministic variants, with picture-in-picture capped to avoid wasted work
- QA extracts preview frames and scores contrast, occupancy, dead space, edge safety, text overflow risk, and motion delta
- manifests, notes, metadata, QA JSON, validation output, logs, HTML, and selected variant provenance are persisted for every generated visual

## Visual Narrative Program Upgrade

The next production slice moves auto visuals from isolated transcript-card inserts toward a video-level directed visual system:

- auto visuals now build a `VisualNarrativeProgram` before individual specs are rendered
- the program stores chapters, concept memory, density targets, continuity groups, visual episodes, episode beats, and transition contracts
- the LLM planner receives the program as directing context instead of only a flat card list
- normalized specs inherit program context, episode context, recurring concept IDs, visual beats, transition metadata, and QA contracts
- Hyperframes metadata carries this context through render artifacts so selected variants remain explainable
- the compositor can use entry and exit transition handles for full-screen replacements instead of always hard-cutting generated visuals into the source video
- Hyperframes has expanded sequence templates for causal chains, flywheel loops, decision matrices, anatomy cutaways, stack rankings, contrast ladders, proof sequences, and narrative arcs

## Hyperframes Coverage Upgrade

Auto visuals now uses a context-aware visual budget instead of a fixed low default. The budget considers clip duration, transcript card count, visualizability, intuition payoff, and explicit data/process/contrast cues, then caps the final plan at a production-safe maximum. Premium plans can backfill from deterministic fallback candidates after the director and critic passes, so a strong first visual no longer prevents additional high-signal beats from being rendered.

Hyperframes also includes a broader deterministic template pack for common explainer needs: concept maps, problem/solution pivots, myth-busters, checklists, radar scans, opportunity maps, scorecards, pipeline X-rays, decision trees, momentum waves, focus rings, filmstrip timelines, quote breakdowns, market maps, mechanism blueprints, and data pulses. These remain typed specs compiled by Vex into self-contained seekable HTML; the LLM chooses templates and parameters, not raw rendering code.

## Auto Visuals Director V3

Auto visuals now has a local director gate after LLM planning and before compositing. Vex samples tiny source frames with FFmpeg rawvideo pipes, scores source richness versus visual need, routes non-math explanatory visuals toward Hyperframes, keeps Manim for formula/geometry-heavy scenes, and rejects visuals whose copy does not overlap the transcript evidence. Rendered Hyperframes variants and Manim scenes also pass a final metadata QA gate, so a failed renderer output is dropped instead of being composited as the least-bad candidate. Full-screen generated replacements receive soft dissolve transition handles when the visual program did not specify one, keeping timeline ownership in FFmpeg while avoiding abrupt slide cuts.

## Auto Effects Director

Auto effects now separates subtitle-beat selection from final motion rendering. The subtitle planner still chooses candidate emphasis moments, but `effects.motion` converts that plan into a typed `motion-director-v1` camera timeline with taste profiles, bounded scale, safe anchors, pan limits, style strength, and motion QA. The FFmpeg compiler consumes this motion plan when present, using a single resolved camera path instead of additive stacked zoom/pan deltas. The tool also runs tiny raw-frame preview QA after rendering, without saving screenshots, so broken or blank effected output is rejected before the project state is updated. Legacy stored effect plans continue to replay through the older compiler path.

## Local Creative Intelligence Upgrade

The newest local-first production slice gives the highest-level automation features a shared creative substrate instead of separate one-off heuristics:

- `tools/creative_intelligence.py` builds a typed `VideoUnderstandingGraph` with transcript excerpts, topic weights, semantic beats, retention moments, scene cuts, platform profiles, and quality contracts
- auto shorts uses the graph for candidate retention scoring, visual-opportunity signals, topic alignment, continuity risk, graph-searched multi-source `ShortEditPlan` synthesis, pre-render edit validation, final metadata, and bundle manifests
- auto visuals annotates visual cards with graph signals, carries graph summaries into the visual narrative program, records graph-backed plan QA, and persists the graph in the manifest
- auto color grading builds a deterministic Color Grade Director plan with subject-aware protection, per-shot candidate scoring, scene consistency smoothing, local color-grade quality contracts, and creative QA alongside the existing technical grade validation
- `tools/creative_qa.py` creates reusable `CreativeQualityReport` payloads for shorts candidates, visual plans, and color grades
- `tools/creative_registry.py` writes a local atomic `creative_runs.json` registry per project so recent shorts, visuals, and grade runs are discoverable from `/status`, `/creative-runs`, and `vex creative-runs`

This keeps the upgrade local-first: no SaaS API, no cloud state, and no remote dependency is required for the new architecture.

## Blender 3D Asset Renderer

Blender support is intentionally an asset-rendering backend, not a second editor. Vex owns the project state, timeline, undo/rebuild, and FFmpeg compositing. Blender receives a typed `BlenderVisualSpec`, renders deterministic 3D assets, and returns metadata for the normal generated-visual pipeline.

- supported 3D templates include `three_d_title`, `floating_3d_label`, `object_orbit`, `logo_reveal`, `screen_pointer_3d`, `data_tunnel`, and `product_model_spin`
- overlay specs render transparent frames and are encoded as alpha-capable assets before Vex composites them
- model imports are restricted to local `.glb`, `.gltf`, `.obj`, and `.blend` files inside workspace-safe roots
- the agent may choose templates and parameters, but Vex never executes raw LLM-authored Blender Python

## Target Architecture

The target shape is a modular local-first platform:

1. Core domain package: typed timeline operations, project manifests, asset records, render job specs, and schema migrations.
2. Durable job runner: queue long-running work, persist progress, resume interrupted renders, and expose cancellation.
3. Media engine boundary: isolate FFmpeg/MoviePy operations behind a typed service contract with command auditing and fixture-based integration tests.
4. Renderer sandbox layer: execute generated Manim/Blender scenes in isolated workspaces with strict timeouts, resource limits, validation reports, and artifact promotion.
5. Provider gateway: centralize model selection, retry policy, tool schema adaptation, prompt versioning, and trace capture.
6. Asset registry: track source media, generated assets, transcripts, exports, attribution, checksums, and cleanup policy in one place.
7. Observability and QA: structured logs, machine-readable traces, render quality reports, regression screenshots, and reproducible support bundles.
8. Extension API: make tools, renderers, export presets, and style packs loadable without editing core modules.

## Upgrade Phases

### Phase 1: Stabilize the Core

- Replace dictionary timeline params with typed operation models.
- Add schema versions to project state and timeline entries.
- Introduce project-state migrations with backwards-compatible loading.
- Keep all state writes atomic and add recovery tests for interrupted saves.
- Normalize command errors into user-facing messages plus debug metadata.
- Add integration fixtures for no-audio, variable-fps, vertical, square, and path-with-quotes videos.

### Phase 2: Durable Execution

- Add a local job table under each project directory.
- Run long tasks as resumable jobs: transcription, YouTube download, auto-shorts, auto-b-roll, auto-visuals, render, export.
- Persist job progress, stdout/stderr tails, current artifact paths, and cancellation state.
- Add a `vex jobs` CLI surface for list, inspect, cancel, retry, and resume.
- Make the REPL attach to existing jobs instead of owning their lifetime.

### Phase 3: Timeline IR and Rebuild Engine

- Define a canonical timeline IR with typed operation classes.
- Give each operation explicit inputs, outputs, side effects, and replay requirements.
- Store asset IDs instead of raw paths wherever possible.
- Add deterministic timeline validation before execution and before rebuild.
- Support partial rebuild from the last valid checkpoint instead of always replaying from source.
- Add timeline diff and dry-run output for complex chained edits.

### Phase 4: Renderer Isolation

- Move Hyperframes, Manim, and Blender execution into isolated render jobs.
- Separate generated code validation, preview render, QA scoring, final render, and promotion into explicit stages.
- Cache successful render artifacts by spec hash.
- Add renderer health checks and capability probes at startup.
- Enforce per-render time, disk, and process limits.
- Store every renderer's source, config, logs, preview frames, QA report, and final asset in the asset registry.

### Phase 5: Provider Gateway

- Create a single gateway for Gemini, Claude, and future providers.
- Route tasks by capability: tool calling, long-context transcript planning, visual scene generation, critique, and metadata writing.
- Add prompt version IDs to traces and generated artifacts.
- Add token, latency, retry, and failure metrics per provider.
- Support offline deterministic fallback for core edits when provider calls fail.
- Add golden prompt tests for intent-to-tool and visual-plan outputs.

### Phase 6: Asset Registry and Storage Policy

- Introduce `assets.json` or a small SQLite catalog per project.
- Track every source, working copy, transcript, generated visual, b-roll clip, export, and manifest.
- Store checksum, media metadata, dependency links, attribution, and retention class.
- Add cleanup commands for temp files, stale renders, failed jobs, and retained checkpoints.
- Make export bundles reproducible from the registry plus timeline.

### Phase 7: Product Surface

- Add a non-interactive `vex plan` command that compiles natural language into a reviewable edit plan.
- Add a `vex apply-plan` command for reproducible automation.
- Add structured JSON output modes for scripts and CI.
- Add plugin discovery for tools/renderers/style packs.
- Add a lightweight local API server for GUI or external automation clients.

## Immediate Engineering Standards

- Every state mutation should be either atomic or recoverable.
- Every subprocess command should be built from argv lists, logged in debug traces, and tested for hostile paths.
- Every operation that can run without audio must explicitly support no-audio inputs.
- Every long-running tool should emit progress that can survive REPL redraws and future job persistence.
- Every generated artifact should carry enough provenance to reproduce or explain it.
- Every user-facing failure should include the next useful action, not only the raw dependency traceback.

## First Milestones

1. Introduce typed timeline operation models beside the existing dict format.
2. Add migration-aware project loading and save schema versions.
3. Add a local durable job runner for exports and auto-visuals.
4. Move renderer execution logs and QA reports into a single asset registry.
5. Build a media fixture integration suite that runs in CI with FFmpeg.
6. Add `vex plan` and `vex jobs` before adding any GUI surface.
