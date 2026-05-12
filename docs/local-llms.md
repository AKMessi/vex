# Local LLM Setup

Vex can run its agent planning layer against any OpenAI-compatible local chat server. The same adapter supports Ollama, LM Studio, llama.cpp server, vLLM, LocalAI, and similar runtimes that expose `/v1/chat/completions`.

Local models are useful when you want privacy, offline iteration, or lower marginal cost. They are also more variable than Gemini or Claude. The most important requirement is reliable tool calling, because Vex edits video by asking the model to choose structured tools.

## Provider Options

Use one of these values in `.env`:

```env
PROVIDER=openai_compatible
PROVIDER=ollama
PROVIDER=lmstudio
PROVIDER=llama_cpp
```

`openai_compatible` is the generic mode. The other three are convenience aliases with sensible default URLs.

## Generic OpenAI-Compatible Server

Use this for vLLM, LocalAI, a custom gateway, or any server that implements OpenAI-compatible chat completions.

```env
PROVIDER=openai_compatible
OPENAI_COMPAT_BASE_URL=http://localhost:11434/v1
OPENAI_COMPAT_API_KEY=
OPENAI_COMPAT_MODEL=qwen2.5-coder:14b
OPENAI_COMPAT_TIMEOUT_SEC=120
OPENAI_COMPAT_MAX_TOKENS=4096
OPENAI_COMPAT_TEMPERATURE=0.2
```

`OPENAI_COMPAT_API_KEY` can be empty for local servers that do not require auth. If your server expects a bearer token, set it there.

## Ollama

Start Ollama and pull a model:

```bash
ollama serve
ollama pull qwen2.5-coder:14b
```

Configure Vex:

```env
PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=qwen2.5-coder:14b
```

Ollama must be running before you ask Vex to execute an agent request. If Vex can start but fails on the first model call, check that `ollama serve` is active and that the model name exactly matches `ollama list`.

## LM Studio

In LM Studio:

1. Download a chat or coder model with tool-calling support.
2. Open the local server tab.
3. Start the OpenAI-compatible server.
4. Note the model id shown by LM Studio.

Configure Vex:

```env
PROVIDER=lmstudio
LM_STUDIO_BASE_URL=http://localhost:1234/v1
LM_STUDIO_MODEL=your-loaded-model-id
```

If LM Studio reports a different port, update `LM_STUDIO_BASE_URL`.

## llama.cpp Server

Start a llama.cpp OpenAI-compatible server with your model:

```bash
llama-server -m /path/to/model.gguf --host 127.0.0.1 --port 8080
```

Configure Vex:

```env
PROVIDER=llama_cpp
LLAMA_CPP_BASE_URL=http://localhost:8080/v1
LLAMA_CPP_MODEL=your-model-name
```

Some llama.cpp builds expose model names differently. If your server ignores the model field, keep any non-empty model name in the env file.

## Model Guidance

Prefer instruction-tuned coder or agent models that can produce structured tool calls. Small models can work for deterministic Vex commands, but complex editing requests need stronger reasoning and cleaner JSON/tool behavior.

Good starting points:

- `qwen2.5-coder:14b` for Ollama if your machine has enough memory.
- A 7B or 8B instruct/coder model for lighter machines.
- Larger local models for visual planning, transcript reasoning, and multi-step edits.

Avoid base models that are not instruction tuned. They usually fail tool-calling workflows.

## How Vex Talks To Local Models

Vex sends:

- a system prompt with current project state
- conversation messages
- OpenAI-format tool schemas
- `tool_choice=auto`

Vex accepts:

- native OpenAI-compatible `tool_calls`
- JSON-only fallback tool calls for local models that emit tool JSON as text

The fallback exists for resilience, but native tool calls are preferred.

## Smoke Test

After configuring `.env`, run:

```bash
vex
```

Then try a simple request against a loaded project:

```text
Vex > show video metadata
```

That should call `get_video_info`. If the model answers in prose instead of using a tool, the selected local model probably does not follow tool schemas well enough for agentic editing.

## Troubleshooting

`Could not connect`

- Start the local model server.
- Check the base URL and port.
- Confirm the URL includes `/v1`.

`Local provider returned invalid JSON arguments`

- Try a stronger instruction-tuned model.
- Lower temperature with `OPENAI_COMPAT_TEMPERATURE=0`.
- Prefer native tool-call-capable runtimes when available.

`Model returns text instead of tool calls`

- Use a model known to support tool calling.
- Try `qwen2.5-coder`, a larger instruct model, or an LM Studio model with tool support.
- Keep deterministic commands simple when using smaller local models.

Slow responses

- Use a smaller model.
- Reduce `OPENAI_COMPAT_MAX_TOKENS`.
- Use a GPU-backed runtime.
- Increase `OPENAI_COMPAT_TIMEOUT_SEC` for large local models.

## Current Limits

Local LLM support covers the agent planning provider. Vex still uses local non-LLM tools separately, including FFmpeg, Whisper, Hyperframes, Manim, and optional Blender. API-backed features still need their own keys when used, such as Pexels stock footage search.
