# Vex Release Architecture Report

## Objective

The release system turns Vex from a repository-only application into a
versioned, reproducible install while preserving explicit control over external
renderers and public publication.

## Package Identity

- Product and CLI remain `Vex` and `vex`.
- The Python distribution is `vex-video` because the `vex` name on PyPI belongs
  to an unrelated project.
- `vex_runtime.__version__` is the only version authority.
- The first formal line starts at `0.1.0rc1`, reflecting pre-1.0 compatibility
  rather than claiming a mature stable API.

This removes version drift between CLI output, package metadata, tags, and
release artifacts.

## Dependency Boundaries

The default Python install contains the editing and agent runtime. Whisper and
Manim are optional extras. Blender, FFmpeg, Node.js, and npm remain external
system tools.

Moving Manim out of the default dependency graph reduces install time, native
build failures, and unnecessary system-library requirements for users who only
need FFmpeg, B-roll, HyperFrames, or general editing.

## Managed HyperFrames Runtime

Vex packages a pinned npm manifest and lockfile. The explicit installer:

1. verifies Node.js 22+ and npm;
2. installs into a staging directory with `npm ci`;
3. retains a failure log when installation fails;
4. verifies the exact HyperFrames CLI version;
5. records the lockfile digest and tool versions;
6. atomically promotes the completed runtime;
7. scopes the runtime to the installed Vex version.

Renderer discovery accepts only an explicit configured path, the repository
runtime for source development, or the managed runtime. It no longer executes a
HyperFrames binary from arbitrary current-directory `node_modules`.

This makes wheel installs functional without a Git checkout while reducing
dependency substitution and partial-install risks.

## Build And CI Gates

CI now covers:

- Python 3.11 and 3.14 on Linux;
- Python 3.12 on Windows and macOS;
- the complete pytest suite;
- Python compilation and whitespace checks;
- isolated wheel and sdist builds;
- `twine` metadata validation;
- archive content and secret-exclusion inspection;
- clean-wheel installation without a source checkout;
- CLI, config, doctor, and packaged-resource smoke tests.

The archive validator enforces distribution name/version, required runtime
resources, lockfile equality, and the absence of `.env`, `node_modules`, Git
metadata, and Python caches.

## Publication Trust Boundary

Release publication runs only for `v*` tags. The tag must equal the canonical
version exactly.

The build job has no package-index credential. It tests, builds, validates,
checksums, attests, and uploads immutable workflow artifacts. A separate job
installs the attested wheel through `pipx` before any GitHub release is created.
Environment-protected publication jobs receive short-lived OIDC tokens for
TestPyPI or PyPI.

Release candidates publish to TestPyPI when the trusted publisher is explicitly
enabled; otherwise the verified GitHub wheel remains the candidate
distribution. Stable versions publish to PyPI. Each enabled index publication
is downloaded and installed before the corresponding GitHub Release is
created.

This limits credential exposure, prevents branch pushes from publishing, and
ensures the GitHub Release represents a package users can actually install.

## Supply-Chain Evidence

Every release bundle includes:

- wheel;
- source distribution;
- `SHA256SUMS`;
- GitHub build provenance attestation;
- generated release notes.

Dependencies in GitHub workflows are pinned to reviewed action commits, and
Dependabot proposes controlled updates.

## Operational Safety

Publication remains a two-stage human decision:

1. merge and validate release infrastructure or a version bump;
2. separately approve the protected environment deployment after pushing the
   intentional release tag.

Bad published versions are never overwritten. They are yanked and replaced by a
higher version. Managed renderer runtimes and user projects are not deleted
during package uninstall or upgrade.

## Result

Vex gains a repeatable install path, smaller default dependency surface,
cross-platform package evidence, explicit renderer setup, honest compatibility
semantics, and a release process that is difficult to trigger accidentally and
does not depend on long-lived package-index secrets.
