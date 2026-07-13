# Changelog

All notable Vex changes are recorded here. Vex follows Semantic Versioning while
the public interface remains pre-1.0.

## [Unreleased]

### Added

- Add Visual Director v2 with signed atomic communication contracts, six-lane
  visual concept search, four-frame reference boards, and bounded alternate
  concept rendering shared by HyperFrames and Remotion.
- Add blind multimodal semantic, design, temporal, and technical verification
  with Gemini/Claude failover, retries, circuit breaking, content-addressed
  caching, explicit quality states, and order-reversed pairwise ranking.
- Add typed counterexample-guided repair for semantic encoding, composition,
  copy fidelity, motion causality, and execution defects, plus portfolio-level
  creative identity, repetition gates, diversity metrics, and telemetry.

### Changed

- Verify final rendered evidence at the primary and reserve publication
  boundaries; only monotonic repairs can advance, while independently verified
  evidence may override known-soft renderer heuristics.
- Compile reserve opportunities with the same model-authored Open Visual search
  as primary opportunities and route directed HyperFrames requests through the
  shared communication, concept, verification, and repair architecture.

### Fixed

- Prevent valid Remotion scenes from being discarded solely because a local
  pixel-motion heuristic reports no meaningful motion when blind frame evidence
  verifies the intended causal sequence.
- Prevent verifier outages from silently claiming success or eliminating every
  candidate: strict mode remains unverified, while balanced mode publishes only
  a clearly labeled degraded candidate that passed hard local gates and a local
  quality floor.
- Pin the Windows ARM64 Sharp binary used by HyperFrames and require a real
  native-load probe during managed installation, renderer availability checks,
  and diagnostics so a present-but-unloadable CLI is never reported as ready.

## [0.1.0rc21] - 2026-07-13

### Added

- Add Open Visual Program v1, a packaged JSON Schema 2020-12 scene graph shared
  by HyperFrames and Remotion with signed evidence bindings, normalized layout,
  semantic relations, deterministic motion tracks, constraints, and resource
  budgets.
- Add evidence-bound model authoring with distinct concept candidates, one
  validation-aware retry, deterministic fallback programs, novelty fingerprints,
  semantic-fitness ranking, and persisted tournament diagnostics.
- Add generic Open Visual Program runtimes for Remotion and HyperFrames plus
  bounded typed patch operations that re-sign and revalidate every repair.

### Changed

- Treat template and scene-family libraries as compatibility fallbacks and
  authoring priors instead of the primary automatic composition boundary.
- Map HyperFrames proof variants to distinct open-program concepts and include
  semantic fitness in final candidate selection.
- Render Remotion scenes from arbitrary validated primitives and frame-driven
  property tracks while preserving existing render and timeline QA telemetry.

### Fixed

- Preserve complete mechanism explanations in the open scene graph when legacy
  renderer layouts omit or collapse a grounded result stage.
- Prevent fixed-size renderer typography and legacy palette precedence from
  clipping long labels or producing low-contrast open-program text.
- Keep HyperFrames entrance animation from overwriting semantic transformation
  motion on the same visual element.

## [0.1.0rc20] - 2026-07-12

### Added

- Add a signed Visual Explanation IR handoff from opportunity planning to both
  renderers, preserving validated evidence, objects, relations, and display
  titles through presentation-copy normalization.
- Add choreography-aware Remotion motion QA that samples authored beat windows,
  measures changed image area, and verifies the final readable hold.
- Add semantic relation connectors, role labels, and authored relation-phase
  animation to Remotion mechanism scenes.

### Changed

- Scope prior failure memory to exact compiler-rejected opportunities for the
  requested renderer; render QA failures now influence quality learning without
  blacklisting their transcript evidence.
- Restrict relationship-heavy HyperFrames scenes to diagrammatic, spatial,
  data-sculpture, or editorial media and use content-aware display type sizing.
- Give semantic HyperFrames explanations a larger evidence-copy budget while
  tightening edge intrusion limits for kinetic typography.

### Fixed

- Prevent a failed Remotion render from blocking every overlapping opportunity
  in a later HyperFrames run.
- Prevent valid preflight opportunities from failing after shortened display
  labels are rebuilt into fragmented semantic facts.
- Prevent Remotion motion QA from sampling only settled frames and falsely
  rejecting animated scenes as static.
- Prevent oversized HyperFrames headlines from clipping or colliding with
  evidence labels, and avoid duplicating full takeaways as scene titles when a
  concise transcript-grounded episode title is available.

## [0.1.0rc19] - 2026-07-11

### Added

- Add a renderer-neutral, signed Creative Direction Program for HyperFrames and
  Remotion with semantic medium selection, focal hierarchy, composition
  grammar, palette roles, typography, material, motif, choreography phases,
  negative-space targets, and accessibility constraints.
- Add a shared rendered-frame aesthetic critic for hierarchy, balance, edge
  safety, visual depth, palette vitality, negative space, and settled motion.
- Add Remotion medium treatments for data sculpture, editorial collage,
  spatial metaphor, kinetic typography, and grounded annotation rails.

### Changed

- Compile every HyperFrames proof candidate and Remotion scene against the same
  video design bible and creative-direction harness.
- Blend measured aesthetic quality into HyperFrames variant selection and
  Remotion publication QA while preserving semantic hard gates.
- Carry creative direction IDs, signatures, media, and critic artifacts into
  renderer metadata and Auto Visuals compiler reports.

### Fixed

- Derive panel foreground color from the directed panel surface to prevent
  unreadable dark-on-dark or light-on-light compositions.
- Preserve full evidence-backed quote language and render grounded constraints
  without misrepresenting them as process stages.
- Exclude intentional registration rails from edge-crowding measurements while
  retaining content edge-safety enforcement.

## [0.1.0rc18] - 2026-07-11

### Added

- Add a typed, signed Remotion scene program that compiles transcript evidence
  into constrained metric, mechanism, contrast, timeline, interface, or
  emphasis compositions before Chromium starts.
- Add rendered-frame Remotion QA for contrast, occupancy, information density,
  motion, semantic quality, and final-hold validation.
- Add responsive landscape, square, and portrait React compositions using
  frame-driven animation and measured text fitting.

### Changed

- Precompile Remotion primary and reserve opportunities in Auto Visuals so
  unsupported or ungrounded scenes are rejected cheaply and valid reserves can
  replace them before rendering.
- Replace the embedded legacy Remotion JSX template with one packaged,
  testable composition entrypoint and require `@remotion/layout-utils` in the
  managed renderer runtime.

### Fixed

- Reject transcript-grounded Remotion programs with invented numeric facts or
  a non-renderable visual explanation policy.
- Prevent duplicated metric units, summary text masquerading as a process
  stage, square-layout clipping, portrait footer collisions, and post-render
  duration fallback failures.

## [0.1.0rc17] - 2026-07-11

### Added

- Add `VEX_NODE_PATH` and `VEX_NPM_PATH` for selecting an isolated renderer
  runtime without replacing the system Node.js installation.

### Fixed

- Rebuild the managed renderer runtime when the selected Node architecture
  changes, ensuring native Remotion, Rspack, and Sharp packages match Node.
- Launch managed HyperFrames entrypoints with Vex's selected Node executable
  instead of an ambient shell shim.
- Prefer the verified managed Remotion runtime over stale checkout packages and
  load Rspack's native binding during availability checks.

## [0.1.0rc16] - 2026-07-11

### Added

- Add Remotion as a first-class third Auto Visuals renderer in the agent,
  deterministic intent compiler, tool schema, CLI, renderer registry,
  diagnostics, packaging, and installed-wheel runtime.
- Add `vex renderers install remotion` for the shared version-locked Node
  renderer runtime.

### Changed

- Verify exact HyperFrames, Remotion, React, renderer, and bundler package
  versions before promoting a managed renderer runtime installation.
- Harden project transactions, timeline replay, concurrent mutation execution,
  media subprocess deadlines, provider/download boundaries, and creative run
  artifact isolation.

## [0.1.0rc15] - 2026-07-01

### Added

- Add `add_song`, a production Music Director path for adding songs,
  soundtracks, intro/outro cues, and timed music segments. It routes requests
  through a bounded song-mix skill graph, builds deterministic FFmpeg
  filtergraphs with loop/trim placement, fades, speech ducking, limiting, and
  loudness normalization, writes auditable song-mix bundles, rejects failed
  audio QA before timeline promotion, and exposes CLI, intent, and tool-schema
  support.

## [0.1.0rc14] - 2026-06-30

### Added

- Add a Video Generation Skill Graph runtime for `generate_video`. The new
  showrunner layer classifies the whole video into a production skill, assigns
  every beat a source-grounded visual skill, seeds HyperFrames semantic frames,
  steers proof tournaments and native motion, writes `VIDEO_SKILL_GRAPH.json`,
  and makes skill coverage part of portfolio and generated-video QA.
- Add an Auto Visuals Skill Graph runtime that routes transcript opportunities
  through executable visual skills before model planning, attaches deterministic
  plan seeds, skill slices, slot schemas, reject rules, proof priors, and
  visual-world priors, and records skill contracts in manifests and accepted
  overlays.

## [0.1.0rc13] - 2026-06-30

### Added

- Add typed project and timeline schema migration with stable operation IDs,
  schema versions, and backward-compatible legacy project loading.
- Add a project asset registry and content-addressed media cache, integrated
  into transactional working-file promotion.
- Add explicit tool contracts, a durable project job runner, and CLI surfaces
  for `jobs`, `queue-job`, and `run-job`.
- Add renderer job manifests for Auto Visuals, capturing render status,
  request metadata, output paths, artifacts, failures, and renderer timing
  context in `render_job.json`.
- Add a provider gateway with model capability metadata, normalized provider
  events, response validation, and sanitized tool-result formatting.
- Add durable `plan`, `plans`, and `apply-plan` commands for deterministic
  reviewable edit plans.
- Add NLE interoperability exports through `nle-export`, producing Vex
  timeline JSON, FCPXML, and EDL sidecars.
- Add an intent evaluation harness and `eval-intents` command for
  machine-readable planning regression reports.
- Add a manifest-based plugin extension API with explicit opt-in plugin tool
  executor loading and `plugins` discovery.

### Changed

- Route `trim_clip` through transactional promotion so project state changes
  only after outputs, metadata, asset registration, and cache insertion
  succeed.
- Include the new architecture modules in package metadata and release
  artifact validation.

### Quality Architecture

- Vex now has a durable editing harness spine: typed timeline IR, tool
  contracts, asset registry, content cache, job ledger, renderer manifests,
  provider gateway, reviewable plans, NLE export, evaluation reports, and
  plugin discovery.

## [0.1.0rc12] - 2026-06-28

### Added

- Add a generated-video Director Crew layer that repairs weak prompt-only
  scripts, writes `DIRECTOR_CREW.json`, and gives every beat a source-grounded
  objective, viewer question, visual job, motion intent, and anti-filler rule.
- Add a per-beat HyperFrames variant tournament for generated videos so
  semantic worlds, proof programs, scene programs, canvas motion, and contract
  coverage compete before a beat is accepted.
- Add a generated-video portfolio judge that rejects generic scripts, repeated
  visual worlds, missing tournament records, weak native-motion coverage, and
  incomplete director contracts before a generated project is reported as
  successful.

### Changed

- Upgrade the pinned managed HyperFrames runtime from `0.6.113` to `0.7.17`.
- Make `generate_video` fail visibly when generated-video QA rejects the
  project, while still leaving the project artifacts and manifest for debugging.

### Fixed

- Keep generic director story roles such as "problem", "promise", "input", and
  "payoff" out of generated-video semantic required labels so source-provenance
  checks reject fewer valid beats for the wrong reason.
- Strip prompt command verbs such as "show" and "explain" before generated-video
  script direction, titles, and beat contracts so command words do not become
  fake visual concepts.
- Preserve `id="root"` in external HyperFrames sub-composition templates and
  rewrite active root-class CSS selectors to `#root`, satisfying HyperFrames
  0.7 lint scoping without losing generated-video styling.

## [0.1.0rc11] - 2026-06-22

### Added

- Add a native HyperFrames motion compiler for `generate_video`: generated
  projects now persist `MOTION_PLAN.json`, `motion_cues.json`, template-backed
  per-beat `compositions/*.html`, transition overlays, and cue-driven motion
  metadata while rendering visible native beat surfaces in the root timeline.
- Expose showcase render controls for generated videos through the CLI and tool
  schema, including `fps`, `quality`, `render_resolution`, and `workers`.

### Changed

- Orchestrate generated videos with root inline native beat surfaces plus
  traceable `data-external-composition-src` artifact links, avoiding renderer
  blind spots while preserving per-beat HyperFrames composition files.
- Upgrade the pinned managed HyperFrames runtime from `0.6.112` to `0.6.113`.

### Fixed

- Make `add_auto_visuals` use a tiered subtitle opportunity planner so
  source-grounded assistive visuals can still be selected when no strict
  proof-grade HyperFrames opportunity clears the primary threshold.
- Stage HyperFrames lint and render work in short temporary directories on
  Windows before copying outputs back to project bundles, avoiding `mkdtemp`
  failures in long auto-visual artifact paths.
- Remove non-packaged `Arial Narrow` and `Roboto Condensed` font declarations
  from HyperFrames visual-world templates so HyperFrames 0.6.113 does not reject
  generated variants during lint.

## [0.1.0rc10] - 2026-06-19

### Added

- Add the generated-video Semantic Cinematographer, which compiles each script
  beat into grounded HyperFrames visual-world compositions before rendering.
- Record per-beat cinematography metadata, scoped composition files, and final
  rendered visual QA in generated-video manifests.

### Changed

- Route generated videos through scoped, fail-visible semantic visual clips
  instead of generic background scenes, preserving grounded objects even when
  nested composition script execution is unavailable.
- Improve sparse-attention prompt planning with concrete explanatory beats
  instead of generic filler narration.

### Fixed

- Prevent generated-video semantic visuals from disappearing behind the root
  shell by demoting inline cinematic roots to ordinary timed clips.
- Treat generated-video copy-density and edge-distance checks as soft warnings
  when the rendered semantic visual score already clears the quality floor.

## [0.1.0rc9] - 2026-06-19

### Added

- Add `generate_video`, a standalone audio-first HyperFrames generator that
  creates a new video project from a prompt or script without requiring a
  source clip.
- Generate script, storyboard, design notes, beat graph, HyperFrames HTML,
  optional TTS narration, optional transcript timing, QA, manifest, and final
  render artifacts under `~/.video-agent/generated_videos`.
- Expose generation through the LLM tool schema, deterministic intent parsing,
  no-project REPL commands, and `vex generate-video`.

### Changed

- Upgrade the pinned managed HyperFrames runtime from `0.6.99` to `0.6.112`.
- Declare `kokoro-onnx` and `soundfile` as direct runtime dependencies so
  pipx installs can use HyperFrames TTS sound generation.

### Fixed

- Reinstall the managed HyperFrames runtime when the pinned package version or
  package-lock digest changes, instead of reusing a stale version-scoped
  runtime directory.
- Fall back to estimated audio timing with a clear warning when HyperFrames
  transcription fails and strict audio timing is not requested.

## [0.1.0rc8] - 2026-06-18

### Added

- Add directed HyperFrames visual ideas to `add_auto_visuals`, allowing a user
  to describe a custom visual metaphor while Vex grounds labels, facts, and
  relations in the selected transcript window.
- Expose directed HyperFrames visuals through deterministic intent parsing,
  the LLM tool schema, and CLI options such as `--visual-idea`, `--start`,
  `--end`, and `--trigger-text`.

### Changed

- Bias the primary signed Visual World Program toward the user-requested medium
  family, such as data sculpture for particle/compression ideas, while keeping
  the HyperFrames proof tournament visually diverse.

### Fixed

- Prevent user art-direction terms from becoming required factual labels unless
  those terms are present in the transcript evidence.
- Reject explicitly timestamped directed visuals when no overlapping transcript
  evidence exists instead of silently borrowing an unrelated subtitle window.

## [0.1.0rc7] - 2026-06-15

### Added

- Render set-partition proofs as executable data sculptures with source
  particles, a compression lens, and grounded crystalline memory nodes.
- Install and exercise every attested release wheel through isolated `pipx`
  verification before creating a GitHub release.

### Changed

- Publish release candidates to TestPyPI only when the repository variable
  `ENABLE_TESTPYPI_PUBLISH` is enabled after trusted-publisher setup.
- Upgrade release artifact transfer to `upload-artifact` 7.0.1 and
  `download-artifact` 8.0.1.

### Fixed

- Stop the legacy semantic-partition grid from shadowing a signed Visual World
  Program.
- Keep GitHub prerelease creation available when TestPyPI has not yet been
  configured, while still blocking it on any enabled TestPyPI verification
  failure.
- Place the exact grounded compression label in the resolved frame so semantic
  screenshot QA can verify every required object.

## [0.1.0rc6] - 2026-06-15

### Added

- Compile every automatic HyperFrames proof candidate through a signed Visual
  World Program that selects a complete medium, canvas, material, typography,
  camera, motion choreography, and background system.
- Render deterministic kinetic typography, editorial collage, data sculpture,
  spatial metaphor, technical-system, product-interface, and source-media
  compositions while preserving signed object, relation, and evidence IDs.
- Coordinate long videos through one design bible with rolling anti-repetition
  constraints across primary and reserve visual opportunities.
- Fingerprint rendered frames by color distribution, luminance, saturation,
  edge density, medium, background, and motion grammar.

### Changed

- Upgrade the pinned HyperFrames runtime from `0.5.7` to `0.6.99`.
- Search proof structure and visual medium independently instead of treating
  palette and panel layout as sufficient candidate diversity.
- Reward medium, canvas, background, and motion diversity in portfolio
  selection, then reject neighboring renders that remain perceptually
  repetitive.
- Use light editorial, chromatic, paper, spatial, data-field, source-media, and
  workspace canvases instead of forcing every scene through the same dark grid.

### Fixed

- Stop automatic semantic scenes from collapsing into the same background,
  rectangular card, and connector-line composition.
- Decode managed HyperFrames installation output explicitly as UTF-8 on Windows
  instead of inheriting the active locale.
- Require the Visual World compiler, renderer, fingerprint QA, and portfolio
  optimizer in wheel, source distribution, and pipx release verification.

### Quality Architecture

- HyperFrames generation now follows:
  `signed proof program -> signed visual world -> specialized deterministic
  renderer -> semantic QA + aesthetic QA + cross-scene diversity QA`.

## [0.1.0rc5] - 2026-06-15

### Added

- Verify Pillow and ImageIO with an in-memory PNG round trip before generated
  visual planning or renderer selection.
- Report the active imaging runtime, installed versions, and an exact pipx
  repair command through `vex renderers doctor`.

### Changed

- Declare Pillow as a direct Vex runtime dependency because generated-visual
  capture and QA require it through ImageIO.
- Validate the real imaging stack inside clean pipx wheel and release
  installations.

### Fixed

- Fail generated Auto Visuals immediately with an actionable diagnostic when a
  partial Pillow installation is missing `PIL.Image`.
- Mark HyperFrames and Manim unavailable when their image capture and QA stack
  cannot encode and decode frames.

## [0.1.0rc4] - 2026-06-15

### Added

- Compile Auto Shorts from stable semantic transcript units instead of
  arbitrary word-count and duration sentence cuts.
- Plan long videos chapter by chapter with exact subtitle unit IDs, then run a
  global candidate tournament with prevalidated reserves.
- Record story-planner, model-selection, fallback, rejection, and reserve
  recovery provenance in every Auto Shorts manifest.

### Changed

- Prefer complete contiguous stories and allow stitched shorts only when their
  ranges are chronological, boundary-complete, and causally connected.
- Render beyond the primary selections only when needed to replace a failed
  preflight or post-render candidate.
- Reduce heuristic role bonuses so labels such as hook and payoff cannot
  overpower transcript continuity.

### Fixed

- Stop selecting mechanically truncated `transcript.sentences.json` fragments
  when higher-quality Whisper segments are available.
- Reject model-selected seeds with abrupt starts, dangling endings, failed
  story critics, or excessive continuity risk.
- Preserve quality-gate reason strings as complete reasons instead of
  serializing them as individual characters.
- Expose reasoning-provider outages and deterministic fallback decisions
  instead of silently labeling fallback selections as model choices.

### Quality Architecture

- Auto Shorts now use a Shorts Story Compiler: semantic transcript units,
  hierarchical chapter discovery, deterministic source-range compilation,
  cold-viewer story criticism, portfolio selection, and bounded reserve
  recovery.

## [0.1.0rc3] - 2026-06-15

### Added

- Plan long-form Auto Visuals through hierarchical subtitle episodes and
  multi-sentence opportunity windows before asking the model to author scenes.
- Preflight every opportunity against the HyperFrames semantic compiler and
  keep globally scheduled reserve opportunities for bounded recovery.

### Changed

- Preserve semantic episode boundaries through the video-level visual program
  instead of reducing long videos to a small fixed chapter count.
- Feed the model window-local evidence, executable scene contracts, and
  compiler preflight results rather than isolated subtitle lines.
- Validate clean wheels and published pipx installations for the long-form
  planner and Auto Visuals integration modules.

### Fixed

- Replace failed primary opportunities after semantic compilation, renderer QA,
  or final timeline QA without forcing unrelated filler visuals.
- Remember failed source subtitle cards across project-local Auto Visuals runs
  so repeated attempts do not regenerate the same rejected concept.
- Reuse a compatible system Whisper installation from pipx through the bounded
  external transcription worker.

### Quality Architecture

- Auto Visuals now use an executable opportunity contract: local evidence,
  semantic episode, scene type, required labels, renderability preflight,
  global schedule decision, and reserve-recovery provenance.
- Successful and failed manifests record the complete opportunity plan,
  compiler substitutions, render recoveries, and semantic source-card lineage.

## [0.1.0rc2] - 2026-06-13

### Changed

- Validate clean wheel installs and published releases through pinned `pipx`
  environments, matching the recommended user installation path.

### Fixed

- Reuse a discoverable system Whisper installation through a bounded external
  worker, or install and verify Whisper inside Vex when no compatible runtime
  exists.
- Keep HyperFrames renderer capabilities synchronized with the canonical
  composer template registry, including semantic partition scenes.
- Fail renderer preflight before expensive visual planning and report
  unsupported templates separately from unavailable renderer dependencies.
- Decode HyperFrames and FFmpeg subprocess output explicitly as UTF-8 so
  Windows locale defaults cannot crash rendering with `UnicodeDecodeError`.
- Reject ambiguous quantitative plans unless their relationships are supported
  by an executable, evidence-backed semantic model.
- Make release resource checks insensitive to platform line endings while still
  requiring semantically identical packaged manifests and configuration.
- Keep release tag tests tied to the canonical runtime version so version bumps
  do not require a second hard-coded test update.

### Quality Architecture

- Visual Explanation IR v2 records relation-level provenance instead of treating
  grounded endpoint labels as proof of a causal or transformational edge.
- Deterministically recover token partition scenes such as
  `32 / 4 = 8` from noisy transcripts and render inspectable token-to-block
  geometry instead of generic metric cards.
- Keep internal directing instructions out of visible headlines and reduce
  scenes to the smallest set of source-backed semantic objects.

## [0.1.0rc1] - Unreleased

### Added

- Installable `vex-video` Python distribution with the existing `vex` command.
- Version-scoped managed HyperFrames runtime installation through
  `vex renderers install hyperframes`.
- Configuration bootstrap through `vex setup config`.
- Multi-platform tests and clean-wheel smoke tests on Linux, Windows, and macOS.
- Tag-gated TestPyPI/PyPI Trusted Publishing with checksums, provenance
  attestations, post-publish installation verification, and GitHub Releases.
- Release-candidate and stable-release operating procedures.

### Changed

- Reset the public version line to `0.1.0rc1` to establish honest pre-1.0
  compatibility expectations before the first formal release.
- Made one Python module the canonical version authority.
- Moved Manim out of the mandatory dependency set and into the `manim` and `all`
  optional extras.
- Restricted HyperFrames discovery to an explicit configured path, the
  repository runtime used for development, or Vex's verified managed runtime.

### Fixed

- Prevented CLI startup failure when an older editable environment does not
  contain `platformdirs`; Vex now resolves standard Linux, macOS, and Windows
  user-data directories with the Python standard library.

### Quality Architecture

- Auto Visuals now compile evidence into signed visual claim graphs and
  role-constrained semantic blueprints.
- HyperFrames candidates compete as structurally distinct proof programs and
  pass blind inverse-decoder and counterfactual QA before promotion.
- Automatic HyperFrames candidates now compile to signed Scene Program V2
  programs with element/relation telemetry and adaptive semantic frame capture.
- Blind, grounded, and design critics now emit typed visual counterexamples that
  drive evidence-preserving patch operations.
- HyperFrames repair is monotonic and bounded; repaired candidates must improve
  without quality or semantic regressions and pass an independent final judge.
- Auto Visuals optimize coherent sets, compare compatible renderer outputs,
  learn bounded project-local quality priors, and validate the final composite.
- Auto B-roll shares final composite QA before project-state promotion.

[Unreleased]: https://github.com/AKMessi/vex/compare/v0.1.0rc15...HEAD
[0.1.0rc15]: https://github.com/AKMessi/vex/compare/v0.1.0rc14...v0.1.0rc15
[0.1.0rc14]: https://github.com/AKMessi/vex/compare/v0.1.0rc13...v0.1.0rc14
[0.1.0rc13]: https://github.com/AKMessi/vex/compare/v0.1.0rc12...v0.1.0rc13
[0.1.0rc12]: https://github.com/AKMessi/vex/compare/v0.1.0rc11...v0.1.0rc12
[0.1.0rc11]: https://github.com/AKMessi/vex/compare/v0.1.0rc10...v0.1.0rc11
[0.1.0rc10]: https://github.com/AKMessi/vex/compare/v0.1.0rc9...v0.1.0rc10
[0.1.0rc9]: https://github.com/AKMessi/vex/compare/v0.1.0rc8...v0.1.0rc9
[0.1.0rc8]: https://github.com/AKMessi/vex/compare/v0.1.0rc7...v0.1.0rc8
[0.1.0rc7]: https://github.com/AKMessi/vex/compare/v0.1.0rc6...v0.1.0rc7
[0.1.0rc6]: https://github.com/AKMessi/vex/compare/v0.1.0rc5...v0.1.0rc6
[0.1.0rc5]: https://github.com/AKMessi/vex/compare/v0.1.0rc4...v0.1.0rc5
[0.1.0rc4]: https://github.com/AKMessi/vex/compare/v0.1.0rc3...v0.1.0rc4
[0.1.0rc3]: https://github.com/AKMessi/vex/compare/v0.1.0rc2...v0.1.0rc3
[0.1.0rc2]: https://github.com/AKMessi/vex/compare/v0.1.0rc1...v0.1.0rc2
[0.1.0rc1]: https://github.com/AKMessi/vex/releases/tag/v0.1.0rc1
