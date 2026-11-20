# Hybrid film production

Vex turns a creative brief into an editable motion-design film. One model
request writes the narration and authors the shots. The model chooses the shot
count, palette, framing, drawings, on-screen copy, and movement. Each shot pairs
an observable visual intent with the artwork that should deliver it.

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
| Verification and repair | Vex samples the encoded export. DeepSeek Flash reviews the visible scene against its intent and narration. One repair can update flagged shots, reuse unchanged native clips, and submit the candidate for a second review. |

The rendering contract accepts declarative artwork with local references and
bounded motion keys. It validates markup before injecting it into a browser.
A malformed treatment gets one validation-guided retry. The validated direction
is saved before audio synthesis or rendering. A repair preserves narration, shot
intent, renderer selection, palette, and scheduled cuts. Vex keeps the original
and candidate exports and selects the candidate when technical and visual QA
pass. [Manim scenes](https://docs.manim.community/en/stable/reference/manim.scene.scene.Scene.html) · [HyperFrames compositions](https://hyperframes.heygen.com/concepts/compositions) · [Remotion frame-based animation](https://www.remotion.dev/docs/flickering)

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
