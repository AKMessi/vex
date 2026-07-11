# Vex Renderer Guidance

Use `vex renderers doctor` to check local renderer availability before long generated-visual jobs.

Install the shared version-locked HyperFrames and Remotion runtime with either
first-class command:

```bash
vex renderers install hyperframes
vex renderers install remotion
```

The installer requires Node.js 22+ and npm, executes the packaged `npm ci`
lockfile, verifies the installed HyperFrames version, and atomically promotes
the runtime into Vex's user data directory. It verifies the exact Remotion,
Remotion renderer/bundler, React, React DOM, and HyperFrames package versions.
It never trusts a global
`hyperframes` executable or arbitrary current-directory `node_modules`.

Remotion uses the source checkout's Node runtime and exact package lock:

```bash
npm ci
vex renderers doctor
```

Remotion's native compositor currently supports Windows x64, Linux, and macOS.
On Windows ARM, use x64 Node under emulation or WSL/Linux; `vex renderers doctor`
reports the active Node architecture before a render starts.

`renderer=remotion` renders a job-scoped React composition through Remotion's
local SSR path and returns a normal MP4 asset to Vex's existing timeline
compositor. It does not take ownership of timeline edits, undo, or final FFmpeg
composition.

## Renderer Fit

HyperFrames is the default fit for evidence-backed explainers: UI/process diagrams, timelines, measured changes, causal relationships, decisions, architecture flows, and custom HTML-like motion.

Remotion is best for React-driven data cards, UI callouts, timelines,
comparison scenes, and reusable programmatic explainer templates. It is a strict
renderer choice via `renderer=remotion`; flexible `renderer=auto` can include it
in renderer tournaments.

Automatic HyperFrames work does not select a decorative template directly. Vex first
builds `VisualExplanationIR`, signs a visual claim graph, reviews a semantic
storyboard, ranks curated role-constrained blueprints, and builds a structural
proof-program tournament. Unsupported ideas are rejected before HTML generation.

Manim is best for formulas, geometry, graphs, and math animations where symbolic layout matters.

Blender is best for 3D titles, product/model spins, logo reveals, and spatial overlays.

FFmpeg asset insertion is best for local MP4/WebM/GIF/image inserts and practical video scaling/export.

## HyperFrames Quality Architecture

- 23 curated semantic blueprints cover 11 explanation scene types and act as search priors.
- Five structural encoding families generate distinct explanatory programs.
- Required labels, objects, directed relations, motion, screenshot tests, forbidden content, and signatures travel with every render.
- Interface scenes prefer a real source-video frame when the source is a screen recording or slide.
- Four sampled times are checked for required objects, meaningful motion, and a readable resolved hold.
- Blind inverse decoding receives no intended answer and must recover the thesis, objects, relations, and sequence.
- Relation-ablation and temporal-scramble counterfactuals test whether the authored grammar is necessary.
- A bounded repair pass may fix grounded copy placement, object coverage, semantic motion, or final hold only after untouched candidates compete.
- Failed variants are never promoted as the least-bad result.
- Legacy templates remain available for compatibility, but automatic generation compiles to `semantic_*` stage families.

## Native Generated-Video Architecture

`vex generate-video` uses HyperFrames as a native motion-production runtime, not
just an HTML slide renderer. After script, audio, transcript, beat graph, and
semantic cinematography planning, Vex writes:

This path is still very early-stage. It is useful for proof videos and focused
technical explainers, but it still lacks a lot of the range, taste, long-form
planning, and consistency expected from a mature general-purpose video generator.
Expect to iterate on prompts, scripts, and outputs.

- `MOTION_PLAN.json` with per-beat technique, camera, transition, caption, and
  capability choices.
- `motion_cues.json` with deterministic word/audio cue envelopes.
- template-backed `compositions/*.html` beat documents that keep registered
  HyperFrames timelines.
- a root `index.html` with render-visible inline native beat surfaces,
  traceable `data-external-composition-src` artifact links, transitions,
  captions, audio, cue variables, and beat-local motion metadata.

For showcase renders, prefer `vex generate-video ... --quality high --fps 60
--render-resolution 4k` when runtime cost is acceptable.

See [feedback-driven-automation-report.md](feedback-driven-automation-report.md)
for the full architecture and artifact contract.
See [hyperframes-visual-proof-search.md](hyperframes-visual-proof-search.md) for the
proof-search and blind-decoder design.

## Examples

- `Use HyperFrames from 01:12 to 01:18 with a dark blue background, yellow title, three animated cards.`
- `Use Manim to animate this equation at 00:45.`
- `Add auto visuals using Remotion for the UI and timeline scenes.`
- `Use Blender to add a rotating 3D title from 00:10 to 00:14.`
- `add_visual_asset --start 12.5 --end 18 --asset path/to/visual.html --mode replace`
- `add_visual_asset --start 12 --end 16 --asset overlay.mp4 --mode overlay`
- `upscale_video --resolution 1920x1080`

## Scaling

`scale to 1080p` means FFmpeg resize/export with Lanczos filtering. This pass does not include AI super-resolution.

Scale modes:

- `fit`: preserve aspect ratio and pad if needed.
- `fill`: preserve aspect ratio and crop to fill the frame.
- `stretch`: resize exactly and distort if the aspect ratio differs.

## Skill Pack

The repository includes `skills/vex-hyperframes-director`. It defines mandatory
grounding rules, scene-family selection, deterministic motion guidance, QA order,
bounded repair, rerouting, and regression requirements. Runtime skill retrieval
also injects scene-specific guidance into composition metadata.
