# Compatibility Policy

## Versioning

Vex uses Semantic Versioning.

- `0.y.z`: public interfaces can still change while Vex is pre-1.0.
- Patch releases preserve documented CLI behavior unless a security or data
  integrity issue requires a correction.
- Release candidates such as `0.1.0rc1` are validation builds.
- `1.0.0` will mark the first stable compatibility contract.

## Supported Runtime

- Python: 3.11 through 3.14
- FFmpeg: required; recent maintained builds are recommended
- Node.js: 22 or newer for HyperFrames
- HyperFrames: version-locked by each Vex release
- Manim: optional extra for specialist math rendering
- Blender: optional external executable

CI runs tests across Linux, Windows, and macOS. Full media/render integration is
validated primarily on Linux, with clean-wheel CLI installation covered on all
three operating-system families.

## Compatibility Surfaces

The following are public compatibility surfaces:

- the `vex` command and documented subcommands
- project-state and timeline replay behavior
- manifest schemas written into project bundles
- renderer install and diagnostic commands
- documented environment variables

Internal Python modules are not a stable library API before 1.0 unless explicitly
documented otherwise.

## Deprecation

Before 1.0, avoidable breaking changes should still be announced in the
changelog. After 1.0, a documented public interface should receive a deprecation
period before removal unless retaining it creates a security or data-loss risk.

