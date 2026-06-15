# Changelog

All notable Vex changes are recorded here. Vex follows Semantic Versioning while
the public interface remains pre-1.0.

## [Unreleased]

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

[Unreleased]: https://github.com/AKMessi/vex/compare/v0.1.0rc3...HEAD
[0.1.0rc3]: https://github.com/AKMessi/vex/compare/v0.1.0rc2...v0.1.0rc3
[0.1.0rc2]: https://github.com/AKMessi/vex/compare/v0.1.0rc1...v0.1.0rc2
[0.1.0rc1]: https://github.com/AKMessi/vex/releases/tag/v0.1.0rc1
