# Changelog

All notable Vex changes are recorded here. Vex follows Semantic Versioning while
the public interface remains pre-1.0.

## [Unreleased]

## [0.1.0rc2] - 2026-06-13

### Changed

- Validate clean wheel installs and published releases through pinned `pipx`
  environments, matching the recommended user installation path.

### Fixed

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

[Unreleased]: https://github.com/AKMessi/vex/compare/v0.1.0rc2...HEAD
[0.1.0rc2]: https://github.com/AKMessi/vex/compare/v0.1.0rc1...v0.1.0rc2
[0.1.0rc1]: https://github.com/AKMessi/vex/releases/tag/v0.1.0rc1
