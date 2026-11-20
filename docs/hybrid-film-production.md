# Hybrid film production

Vex can direct, render, and review an animated film from one brief. The LLM
defines the overall sequence of shots, spoken beats, source-grounded display
copy, palette, motion actions, and transitions before the renderer starts.
Each shot has a viewer objective and a typed visual plan.

| Area | Implementation |
| --- | --- |
| Spatial transformation | Manim preserves the identity of each mark as it moves between configurations. The director can author bounded before and after coordinates or select a convex-lens ray diagram when supported by the narration. |
| Process | HyperFrames uses a seekable HTML timeline to reveal two or three source-grounded stages. |
| Editorial and final edit | Remotion renders the editorial beats, places the two native clip types at the scheduled cut frames, and mixes narration, cut-aware captions, and music. |
| Verification and repair | Vex decodes frames from the encoded export. DeepSeek Flash reviews the frames against the spoken source. One bounded repair can modify the flagged shots and reuse unchanged native clips, with a second review of the complete export. |

The treatment is data. The three renderer adapters own HTML, Python, and React
execution. A malformed treatment gets one validation-guided retry. A failed
native shot retains its source and renderer logs. A repair preserves the
narration, renderer roles, palette, and frame schedule. Vex keeps the original
and candidate exports available and selects the candidate when technical and
visual QA pass. [Manim scenes](https://docs.manim.community/en/stable/reference/manim.scene.scene.Scene.html) · [HyperFrames compositions](https://hyperframes.heygen.com/concepts/compositions) · [Remotion exact-frame media](https://www.remotion.dev/docs/offthreadvideo)

## Run locally

Install the Manim extra and the pinned browser renderers:

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
OPENAI_COMPAT_MAX_TOKENS=8192
```

```bash
vex generate-video "Explain a scattered signal becoming a focused pulse" --renderer hybrid --duration 18
```

`deepseek-flash` is the official DeepSeek V4.1 Flash model ID. It receives
the creative brief and sampled frames for review. Other configured text
models can author the treatment and use the local delivery checks.
[DeepSeek V4.1 Flash](https://api-docs.deepseek.com/news/news260910/) · [DeepSeek image inputs](https://api-docs.deepseek.com/guides/vision/)

Use `--no-render --no-audio` to generate the plan and renderer sources
without native renders. A rendered project contains its source files under
`shots/`, the Remotion composition and media, `film_production.json`,
and `generated_video_qa.json`. Model and repair reports appear when those
stages run. When a repair is accepted, the manifest points to the selected
composition and export under `film_repair/`.

The current visual vocabulary covers abstract geometry, process sequences
and editorial typography. Live-action footage, character rigs, and arbitrary
animation code are outside this vocabulary. The rendered regression runs
all three engines and a simulated DeepSeek repair against real decoded pixels:

```bash
VEX_RUN_HYBRID_RENDER=1 python -m pytest -q tests/test_hybrid_film.py
```
