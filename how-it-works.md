# How Vex Works

This document explains Vex end to end: what it does, how it is structured, how user requests flow through the system, how each editing capability works, and how project state is kept reliable across sessions.

If `README.md` is the public landing page, this file is the deeper technical map.

## What Vex Is

Vex is a terminal-first AI video editing agent.

At a high level, it combines:

- a conversational CLI built with Typer and Rich
- a provider-agnostic LLM loop
- a structured project state layer
- a library of editing tools
- an FFmpeg and MoviePy execution engine

The design goal is simple:

1. let the user speak naturally
2. keep the real editing state outside the model
3. execute deterministic video operations on a working copy
4. persist everything needed to resume, undo, redo, and export later

## What Vex Does

Vex currently supports all of the following.

### Conversational video editing

- understand natural-language editing requests
- detect video file paths directly from user messages
- auto-create or auto-load projects when a path is referenced
- keep the user inside a continuous REPL session

### Video editing operations

- inspect metadata
- trim clips
- merge clips
- adjust speed
- apply fade in / fade out / fade-through-black transitions
- add timed text overlays
- remove silent gaps
- extract selected highlight segments

### Audio operations

- extract audio
- replace audio
- mix new audio with original audio
- mute specific segments

### Transcript and subtitle workflows

- transcribe video with local Whisper
- generate `transcript.txt`
- generate `transcript.srt`
- burn subtitles into video
- summarize a long clip into highlights using transcript-aware LLM selection

### Project and workflow features

- persistent saved projects
- working copy protection
- timeline history
- undo / redo via rebuild
- export presets
- direct CLI exports
- streaming terminal feedback
- multi-provider support

## Product Philosophy

The key architectural idea in Vex is that the model is not the source of truth.

The LLM decides which tool to call and with what arguments, but the important state lives in structured Python objects and project JSON:

- current working file
- source file path
- current metadata
- timeline of applied operations
- redo stack
- session log
- selected provider and model

That is why Vex is much more reliable than a pure “chat + hope” workflow. The model can drift in wording, but the project state stays explicit.

## The Main Runtime Flow

When a user runs `vex`, the system goes through this flow.

### 1. Startup and configuration

Vex loads configuration from environment variables and `.env`.

Important settings:

- `PROVIDER`
- `GEMINI_API_KEY`
- `GEMINI_MODEL`
- `ANTHROPIC_API_KEY`
- `CLAUDE_MODEL`
- `AGENT_PROJECTS_DIR`
- `FFMPEG_PATH`
- `WHISPER_MODEL`

At startup, Vex validates:

- provider value
- required API key for the selected provider
- FFmpeg availability
- project storage directory existence

### 2. CLI entry

The Typer app exposes:

- `vex`
- `vex start`
- `vex repl`
- `vex run`
- `vex projects`
- `vex export`
- `vex --version`

The default mode is `vex` with no subcommand.

Behavior:

- if exactly one saved project exists, Vex resumes it automatically
- otherwise, Vex opens a clean conversational prompt

### 3. Path detection inside the REPL

Before any user message is sent to the agent, Vex scans it for a video path.

It supports:

- Windows paths like `D:\videos\clip.mp4`
- Unix paths like `/home/user/video.mp4`
- quoted paths
- whitespace-separated path tokens

Supported video extensions:

- `.mp4`
- `.mov`
- `.avi`
- `.mkv`
- `.webm`
- `.m4v`
- `.flv`

If a referenced file already belongs to a saved project, that project is loaded.

If not, Vex creates a new project automatically and copies the source video into the project working directory.

### 4. Agent loop

Once a project is loaded, `VideoAgent` runs the core tool loop.

That loop works like this:

1. append the user message to the session conversation
2. build a system prompt from the current project state
3. send conversation + tool schemas to the selected provider
4. if the provider returns tool calls:
   - execute tools one by one
   - update project state
   - append tool results back into the conversation
   - loop again
5. if the provider returns text:
   - save it as the assistant response
   - persist the full session log
   - return control to the REPL

This loop is capped at 10 iterations to avoid runaway tool recursion.

## The System Prompt

Vex does not send a blank prompt to the model.

Each turn includes a system prompt that injects current project context:

- project name
- provider and model
- working file path
- duration
- resolution
- fps
- number of timeline operations
- last operation description

The prompt also instructs the model to:

- inspect metadata before making decisions if needed
- break complex requests into sequential tool calls
- keep responses concise
- preserve original files
- use suggestions in a specific format

This is one of the main reasons Vex holds together better than a naive chat wrapper.

## Provider Layer

Vex supports two LLM backends behind a shared interface.

### Gemini

Gemini support is implemented with `google-genai`.

What the Gemini adapter does:

- creates a `genai.Client`
- converts Vex tool schemas into Gemini function declarations
- sanitizes JSON schema fields for Gemini compatibility
- translates neutral conversation messages into Gemini content parts
- preserves Gemini function call parts so tool follow-up messages keep their required metadata
- streams partial text responses and accumulates tool calls across all stream chunks
- disables Gemini thinking mode with `thinking_budget=0`

### Claude

Claude support is implemented with Anthropic’s SDK.

What the Claude adapter does:

- creates an `Anthropic` client
- translates Vex tool schemas into Claude tool definitions
- converts neutral messages into Claude-native text, tool use, and tool result messages
- supports streaming text output
- extracts tool calls from final Claude responses

### Default provider behavior

Gemini is the default provider.

Claude is supported only when explicitly selected.

The clip summarizer follows the active provider now as well, so Gemini users are not forced into a Claude-specific path.

## Project State Model

Every project is represented by `ProjectState`.

Stored fields include:

- `project_id`
- `project_name`
- `created_at`
- `updated_at`
- `source_files`
- `working_file`
- `working_dir`
- `output_dir`
- `timeline`
- `redo_stack`
- `session_log`
- `metadata`
- `provider`
- `model`

### Why this matters

This state is what makes Vex reliable.

Instead of asking the model to “remember” the edit history, Vex stores the actual edit history explicitly and persists it to disk as JSON.

### What gets persisted

For each project, Vex stores:

- the original source path
- the current working file path
- current video metadata
- every timeline operation
- redo information
- the saved conversation history

### What does not happen

Vex does not edit the original source file directly.

Every destructive-looking operation is actually performed against the current working copy in the project directory.

## Working Copy Safety Model

When a new project is created:

1. Vex creates a unique project directory
2. it copies the original video into that directory
3. it probes the copied file for metadata
4. it sets that copied file as `working_file`

From then on, every edit generates a new output file with a unique filename.

So the flow looks like:

- original source file stays untouched
- project source copy becomes the first working file
- each edit creates another derived file
- `working_file` is updated to point to the newest output

## Timeline and Undo / Redo

Every state-changing tool records a timeline operation.

Each operation stores:

- operation name
- normalized params
- timestamp
- result file
- human-readable description

Undo is not implemented as “delete the last file and hope”.

Instead, Vex rebuilds the project by replaying the timeline from the original source file, excluding the undone step.

That makes undo and redo much more deterministic.

### How undo rebuild works

The rebuild logic:

1. start from the original source file
2. iterate through the saved timeline
3. re-run each operation in order with stored params
4. produce a fresh working file
5. update metadata
6. save project state

Supported replayed operations currently include:

- `trim_clip`
- `merge_clips`
- `adjust_speed`
- `add_transition`
- `add_text_overlay`
- `replace_audio`
- `mute_segment`
- `trim_silence`
- `burn_subtitles`
- `summarize_clip`

## The Execution Engine

The low-level editing work happens in `engine.py`.

This layer is intentionally deterministic and mostly FFmpeg-based.

### Shared engine behavior

Most engine functions:

- generate a unique output path in the project working directory
- call FFmpeg or MoviePy
- raise `VideoEngineError` on failure
- return the new output path on success

### Metadata probing

`probe_video()` uses `ffprobe` to extract:

- duration
- fps
- width
- height
- codec
- whether audio exists
- file size
- container format

### Timestamp parsing

`parse_timestamp()` is the common parser used by tools.

It supports:

- raw seconds like `30`
- suffixed seconds like `30s`
- `MM:SS`
- `HH:MM:SS`
- decimal seconds

### Trim

`trim()` uses FFmpeg input seeking and duration clipping.

Output:

- H.264 video
- AAC audio
- `+faststart` enabled

### Merge

`merge()` first normalizes inputs so concat is safe.

That normalization step:

- scales videos to a common resolution
- pads with black bars if needed
- normalizes fps
- resamples audio
- synthesizes silent audio if a clip has no audio

Then Vex writes a concat list file and uses FFmpeg concat demuxing to merge the clips.

### Extract segments

`extract_segments()` is the highlight-cut helper.

It:

- trims each requested segment to a temp file
- returns a single trim directly if only one segment exists
- otherwise merges the trimmed segments together

This is what powers `summarize_clip`.

### Speed changes

`adjust_speed()` supports:

- full-clip speed changes
- segment-only speed changes

Audio tempo changes are split into chained `atempo` filters so FFmpeg stays within supported ranges.

### Transitions

Vex currently supports:

- `fade_in`
- `fade_out`
- fade-through-black behavior when the tool requests `crossfade` on a single clip

Single-clip “crossfade” is implemented as a fade out followed by a fade in.

### Text overlays

Text overlays use MoviePy rather than raw FFmpeg drawing filters.

This makes it easier to support:

- caption-style wrapping
- multiple anchor positions
- timed overlays
- optional translucent background blocks

### Audio extraction

`extract_audio()` can output:

- `mp3`
- `wav`
- `aac`

### Audio replacement and mixing

`replace_audio()` supports two modes:

- full replacement
- mixing external audio with the original

Mixing uses FFmpeg volume filters plus `amix`.

### Muting

`mute_segment()` uses an FFmpeg volume filter with time-based enable logic.

### Silence trimming

`trim_silence()` works in two phases:

1. run FFmpeg’s `silencedetect` filter
2. parse the reported `silence_start` and `silence_end` timestamps

From there, it constructs “keep” segments representing all non-silent ranges and then reuses `extract_segments()` to build the final cut.

### Subtitle burning

`burn_subtitles()` uses FFmpeg’s `subtitles` filter and force-style settings.

It handles:

- font size
- primary text color
- outline color
- subtitle position
- path escaping for FFmpeg filter syntax

### Export

`export()` applies preset-driven output settings and streams progress by parsing FFmpeg time markers from stderr.

It supports:

- video exports with codec, bitrate, resolution, fps, and faststart
- audio-only exports
- progress callbacks

### Utility helpers

The engine also contains helpers for:

- frame extraction
- output size estimation
- disk space checks
- generating silent audio clips

## Tool Execution Model

Tools are thin wrappers around engine functions and state updates.

Each tool returns a standard result payload with:

- `success`
- `message`
- `suggestion`
- `updated_state`
- `tool_name`

This standardization makes provider integration much easier.

### Tool-by-tool breakdown

#### `get_video_info`

What it does:

- probes the current working video
- refreshes metadata in state
- saves the state

#### `trim_clip`

What it does:

- parses timestamps
- trims the current working video
- updates `working_file`
- refreshes metadata
- appends a timeline operation

#### `merge_clips`

What it does:

- resolves and validates external file paths
- merges the current working file with them
- warns if auto-scaling was needed
- stores the operation in timeline

#### `adjust_speed`

What it does:

- validates speed factor
- optionally parses segment timestamps
- calls engine speed adjustment
- records the operation

#### `add_transition`

What it does:

- chooses fade behavior based on requested transition and position
- updates the working file
- records the operation

#### `add_text_overlay`

What it does:

- validates overlay position
- parses start and end timestamps
- applies MoviePy text rendering
- stores the result in timeline

#### `extract_audio`

What it does:

- extracts audio from the current working file
- optionally moves it to a user-provided path

This tool does not mutate the video project timeline.

#### `replace_audio`

What it does:

- validates the provided audio file
- replaces or mixes audio
- updates working file and metadata
- records the operation

#### `mute_segment`

What it does:

- parses timestamps
- silences the selected range
- records the operation

#### `trim_silence`

What it does:

- reads silence thresholds from params
- removes silent gaps
- updates the working file
- records the operation

#### `transcribe_video`

What it does:

- loads local Whisper
- transcribes the current working file
- writes `transcript.txt`
- writes `transcript.srt`
- returns a transcript preview

It does not modify the video itself.

#### `burn_subtitles`

What it does:

- resolves the SRT path
- defaults to the project’s `transcript.srt`
- validates subtitle position
- burns subtitles into a new video file
- records the operation

#### `summarize_clip`

What it does:

1. ensure transcript artifacts exist
2. auto-run transcription first if needed
3. parse `transcript.srt` into timestamped transcript chunks
4. ask the active LLM provider to return a JSON array of highlight segments
5. merge overlapping segments
6. extract and merge those segments into a shorter cut
7. update state and timeline

This tool is one of the more “agentic” parts of Vex because it combines:

- local preprocessing
- LLM selection
- deterministic media operations

#### `export_video`

What it does:

- loads preset settings
- applies custom overrides if provided
- checks disk space
- exports the current working file

This tool produces a final deliverable but does not change the active working timeline.

#### `undo` and `redo`

What they do:

- manipulate `timeline` and `redo_stack`
- rebuild project state by replaying operations

## Export Presets

Vex ships with named export presets for common targets:

- `youtube_1080p`
- `youtube_4k`
- `instagram_reels`
- `instagram_square`
- `tiktok`
- `twitter_x`
- `podcast_audio`
- `custom`

Each preset can define:

- resolution
- video codec
- audio codec
- video bitrate
- audio bitrate
- fps
- output format
- audio-only behavior

## Streaming User Experience

The REPL is designed to feel alive while work is happening.

Vex uses Rich for:

- the startup banner
- project info panels
- timeline tables
- project tables
- streaming assistant output
- tool progress spinners
- export progress bars

During a tool-based agent turn:

- model text can stream into a live panel
- tool start and finish events update progress feedback
- the final assistant response is printed once the loop completes

## Suggestions

Vex supports a lightweight suggestion system.

If the assistant includes lines in the form:

`[SUGGESTION]: ...`

the agent extracts them separately and the REPL displays them in a highlighted panel.

This is used for follow-up guidance such as:

- warning about mismatched clip resolutions
- warning about aggressive speed changes
- offering caption-related next steps

## Reliability Model

Vex does not try to solve reliability purely with prompting.

It combines several safeguards:

- structured project state
- explicit tool schemas
- deterministic engine functions
- standardized tool result payloads
- JSON-persisted timeline history
- replay-based undo / redo
- source file protection
- provider abstraction instead of provider-specific application logic everywhere

## Memory and Context Handling

Vex does not yet implement a retrieval-ranked long-term memory layer.

What it does instead:

- persists session log
- rebuilds context from structured project state each turn
- injects authoritative project facts into the system prompt

This means:

- edit state remains reliable
- long conversational preference memory can still degrade in very long sessions

That tradeoff is intentional for now. The system favors explicit state over hidden memory magic.

## Limitations

Current limitations include:

- FFmpeg must be installed
- subtitle burning depends on FFmpeg subtitle filter support
- Whisper must be installed for transcription
- MoviePy text rendering on Windows may require ImageMagick
- very long conversation threads can still drift on non-structured preferences
- undo replay depends on required source assets still existing

## Why the Architecture Works

Vex works well because it separates concerns cleanly:

- the CLI owns interaction
- the provider layer owns model-specific protocol details
- the agent loop owns reasoning + tool orchestration
- the tools own validation + state updates
- the engine owns deterministic media processing
- the state layer owns persistence and recoverability

That split is what makes the system understandable, debuggable, and extensible.

## How to Extend Vex

If you want to add a new capability, the pattern is straightforward:

1. add an engine function in `engine.py`
2. add a tool executor in `tools/`
3. register its schema in `prompts.py`
4. register its executor in `tools/__init__.py`
5. add replay support in `tools/undo.py` if it mutates the project timeline

That is the main extension contract across the codebase.

## Short Version

Vex is not just “an LLM hooked up to FFmpeg”.

It is a structured editing system where:

- the model plans
- tools validate
- the engine executes
- state persists
- the timeline makes everything recoverable

That is the core of how the whole project works.
