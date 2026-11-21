# Hybrid film production

Vex turns a creative brief into an editable motion-design film. One model
request writes the narration and authors the shots. The model chooses the shot
count, palette, framing, drawings, on-screen copy, and movement. Each shot pairs
an observable visual intent with the artwork that should deliver it. The default
automatic renderer selects this workflow when a text model is configured. A
silent film can specify relative shot durations with no spoken text. A single
shot is a valid film.

The earlier lens film exposed a gap between direction and rendering. The
model asked for the viewer to see a lens move and change focus. HyperFrames
received three spoken phrases and animated them as a list. The earlier review
could accept those phrases as evidence. The current flow carries the model's
actual drawings and timing through to the encoded frames, where the reviewer
checks that the intended action is visible.

| Area | Implementation |
| --- | --- |
| Artwork and motion | The director authors SVG fragments with ordered layers, paths, masks, gradients, and frame-addressed transforms. Trusted Vex code evaluates the same motion keys in either browser renderer. |
| HyperFrames | Renders authored scenes on a seekable HTML timeline when selected for a shot. |
| Manim | Handles optional native geometry such as point transformations or a convex lens ray diagram. |
| Remotion | Renders inline vector scenes, cuts native clips together, and mixes narration, captions, and music on the final timeline. |
| Verification and repair | DeepSeek Flash reviews the encoded export against its intent and narration. One repair can update flagged shots, reuse native clips, and compare the proposed change on the same absolute frames before encoding a candidate. |

The rendering contract accepts declarative artwork with local references and
bounded motion keys. It validates markup before injecting it into a browser.
A malformed treatment gets one validation-guided retry. When only the narration
misses its duration-derived word range, the retry replaces only the spoken
lines in the validated artwork. The direction is saved before audio synthesis or
rendering. A visual repair preserves narration, shot intent, renderer
selection, palette, and scheduled cuts. Vex selects a candidate when technical
and visual QA pass. [Manim scenes](https://docs.manim.community/en/stable/reference/manim.scene.scene.Scene.html) · [HyperFrames compositions](https://hyperframes.heygen.com/concepts/compositions) · [Remotion frame-based animation](https://www.remotion.dev/docs/flickering)

## Evidence-based repair

The repair request gives hosted DeepSeek Flash the original pixels and findings
for up to three flagged shots. Vex renders the baseline and proposal in the same
Remotion browser session at matching absolute frame numbers. It samples the
opening, development, and resolved state, adds a changed motion key when space
permits, and checks the opening of the next untouched shot. Each target needs an
observable difference. The neighboring samples need to stay stable. A
transition-only edit reuses the native Manim or HyperFrames clip underneath it.

After an encode, an independent review receives the candidate frames, shot
intent, and narration. The repair request uses prior findings, and the local
comparison uses both versions. Keeping that context out of the final review
reduces the chance of the model repeating its earlier judgment. The paired PNGs,
per-frame difference measurements, and input and PNG hashes are recorded in the
repair directory. Sampling covers specific instants, so brief changes between
samples may escape detection. Accepted repairs pay for both the still
comparison and the full export.

On a 108-frame fixture composed from all three engines, adding fully transparent
artwork produced zero changed samples. The paired path rejected it in 10.3
seconds using four frame pairs. Encoding the same proposal and checking the
encoded output took 28.1 seconds on the same machine and reached model review.
These are single-run measurements. A separate transition edit passed the
comparison, kept the adjacent shot stable, and reused both native clips.

VIVID reports results from iterative observation and editing. Vex applies that
idea at the repair boundary with a shared clock, explicit edit scope, and a
local pre-encode decision. [VIVID](https://arxiv.org/abs/2611.20688) · [Remotion frame selection](https://www.remotion.dev/docs/options/frame)

## Relationships between artwork layers

An artwork layer can use `parent` to inherit the motion of an earlier layer and
`pivot` to set the local center for scaling and rotation. An indicator can ride a
moving part while animating its own size. The layer order still controls paint
order. Both browser renderers use the same evaluator for the full hierarchy.
Existing artwork uses the original canvas-centered motion by default.

A repair can update the SVG, motion keys, parent, or pivot of named layers
while preserving the rest of the artwork. Full artwork replacement is available
for a change to the composition or layer structure. This helps when the scene
looks right at the opening and ending but drifts during movement. The placement
relationship is expressed once and applied at every frame. [glTF transform hierarchy](https://github.khronos.org/glTF-Tutorials/gltfTutorial/gltfTutorial_004_ScenesNodes.html)

In a short 480×320 clip, a shrinking detail that used the canvas center
drifted up to 70.2 pixels from its moving parent at the three sampled
timestamps. With layer attachment and a local pivot, the maximum measured
offset was 0.42 pixels across samples from HyperFrames and Remotion.

On a 1080p camera-focus film, the automated DeepSeek repairs still missed the
visual relationship. An operator edited the same direction to attach the blur
indicator to the sensor. DeepSeek repeated both earlier defects when given the
old findings and both sets of frames. Reviewing that same operator-edited export
with candidate frames alone, it passed all three shots and identified the
corrected movement. The final-review request used nine images instead of
fifteen in this trial. Candidate-only review also passed an automated version
whose sensor barely moved in the second shot. Model review remains a judgment
call for visual accuracy. The local comparison verifies visible change and
scope on its sampled frames.

## Run locally

Install the pinned browser renderers. Install the Manim extra for native
geometry shots.

```bash
python -m pip install -e '.[manim]'
vex renderers install hyperframes
vex renderers install remotion
```

Configure the supported DeepSeek endpoint through Vex's OpenAI-compatible
provider. Keep `OPENAI_COMPAT_API_KEY` in the environment or an ignored
`.env` file.

```dotenv
PROVIDER=openai_compatible
OPENAI_COMPAT_BASE_URL=https://api.deepseek.com
OPENAI_COMPAT_MODEL=deepseek-flash
OPENAI_COMPAT_MAX_TOKENS=12288
```

```bash
vex generate-video "Explain a scattered signal becoming a focused pulse" --renderer hybrid --duration 18
```

`deepseek-flash` is the official DeepSeek V4.1 Flash model ID. It receives
the creative brief and sampled frames for review. Other configured text
models can author the treatment and use the local delivery checks.
Vex requests JSON output, uses low thinking effort for the first direction,
and reserves the retry and review output budgets for their JSON responses.
[DeepSeek V4.1 Flash](https://api-docs.deepseek.com/news/news260910/) · [DeepSeek image inputs](https://api-docs.deepseek.com/guides/vision/) · [DeepSeek JSON output](https://api-docs.deepseek.com/guides/json_mode/) · [DeepSeek thinking mode](https://api-docs.deepseek.com/guides/thinking_mode/)

Use `--no-render --no-audio` to author the direction and renderer sources.
Replay the saved direction locally to iterate on export settings without
another authoring or review request. Keep the original aspect and target duration
when reproducing the edit.

```bash
vex generate-video "Focused pulse" --renderer hybrid --duration 18 --film-direction path/to/film_direction.json
```

A rendered project contains `film_direction.json`, editable shot sources under
`shots/`, the Remotion composition and media, `film_production.json`, and
`generated_video_qa.json`. The manifest points to the selected composition,
direction, and export when Vex accepts a repair under `film_repair/`.

Authored browser shots support vector motion design. Native footage synthesis,
character rigs, and arbitrary animation code are outside this contract. The
rendered regression exercises all three engines and a simulated DeepSeek repair
on decoded pixels.

```bash
VEX_RUN_HYBRID_RENDER=1 python -m pytest -q tests/test_hybrid_film.py
```
