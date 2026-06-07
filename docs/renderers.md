# Vex Renderer Guidance

Use `vex renderers doctor` to check local renderer availability before long generated-visual jobs.

## Renderer Fit

HyperFrames is the default fit for explainers: slides, UI/process diagrams, timelines, data cards, comparisons, and custom HTML-like motion.

Manim is best for formulas, geometry, graphs, and math animations where symbolic layout matters.

Blender is best for 3D titles, product/model spins, logo reveals, and spatial overlays.

FFmpeg asset insertion is best for local MP4/WebM/GIF/image inserts and practical video scaling/export.

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
