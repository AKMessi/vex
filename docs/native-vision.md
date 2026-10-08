# Native vision and Groq

Vex supports `PROVIDER=groq` for native tools/streaming and `qwen/qwen3.8-27b` for visual authoring, frame verification, and targeted scene repair. The existing `httpx` dependency supplies the transport; no additional SDK is required.

```dotenv
GROQ_API_KEY=<your existing key>
GROQ_MODEL=qwen/qwen3.8-27b
VISUAL_AUTHORING_PROVIDER=groq
VISUAL_AUTHORING_MODEL=qwen/qwen3.8-27b
VISUAL_DIRECTOR_GROQ_VISION_MODEL=qwen/qwen3.8-27b
VISUAL_DIRECTOR_VISION_REPAIR=true
```

These visual overrides do not require changing the main chat provider. Setting `PROVIDER=groq` selects Groq for the main agent too.

The model receives numbered chronological QA frames. Vex packs all selected frames into at most three images to respect the model's input-image limit. A blind evaluator sees viewer questions and pixels; a separate repair request sees the current program, grounded source contract, failures, and pixels. Repairs produce bounded operations on existing scene elements/tracks. Vex revalidates evidence, geometry, motion, and signatures, renders the change, and independently judges the new video. A model suggestion is not publication approval.

Use `GROQ_REASONING_EFFORT=none` for efficient requests or a supported reasoning level (`default`, `low`, `medium`, `high`) when measured quality justifies it. Requests reject truncated output. Existing retries and circuits remain bounded. Invalid/unavailable repair proposals retain deterministic repair.

On 9 October 2026, the configured key authenticated against Groq's live model catalog, which listed the exact model. A live synthetic-image request correctly read `VEX VISION SMOKE` with JSON output. No key was printed or added to Git.

API behavior was checked against [the Groq model documentation](https://console.groq.com/docs/model/qwen/qwen3.8-27b) and [vision documentation](https://console.groq.com/docs/vision). This model is listed as preview; Vex keeps the provider/model configurable.
