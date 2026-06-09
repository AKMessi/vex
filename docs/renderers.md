# Vex Renderer Guidance

Use `vex renderers doctor` to check local renderer availability before long generated-visual jobs.

## Renderer Fit

HyperFrames is the default fit for evidence-backed explainers: UI/process diagrams, timelines, measured changes, causal relationships, decisions, architecture flows, and custom HTML-like motion.

Automatic HyperFrames work does not select a decorative template directly. Vex first
builds `VisualExplanationIR`, reviews a semantic storyboard, selects a curated
role-constrained blueprint, and signs a production contract. Unsupported ideas are
rejected before HTML generation.

Manim is best for formulas, geometry, graphs, and math animations where symbolic layout matters.

Blender is best for 3D titles, product/model spins, logo reveals, and spatial overlays.

FFmpeg asset insertion is best for local MP4/WebM/GIF/image inserts and practical video scaling/export.

## HyperFrames Quality Architecture

- 23 curated semantic blueprints cover 11 explanation scene types.
- Required labels, objects, motion, screenshot tests, forbidden content, and a semantic signature travel with every render.
- Interface scenes prefer a real source-video frame when the source is a screen recording or slide.
- Four sampled times are checked for required objects, meaningful motion, and a readable resolved hold.
- Optional vision critique supplements deterministic QA; it does not replace contract checks.
- A bounded repair pass may fix grounded copy placement, object coverage, semantic motion, or final hold.
- Failed variants are never promoted as the least-bad result.
- Legacy templates remain available for compatibility, but automatic generation compiles to `semantic_*` stage families.

See [feedback-driven-automation-report.md](feedback-driven-automation-report.md)
for the full architecture and artifact contract.

## Examples

- `Use HyperFrames from 01:12 to 01:18 with a dark blue background, yellow title, three animated cards.`
- `Use Manim to animate this equation at 00:45.`
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
