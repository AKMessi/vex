# Vex

Vex is an open-source AI video editing agent for the terminal.

You launch `vex`, talk to it in plain English, point at a video file, and it edits a safe working copy of your footage using FFmpeg, MoviePy, and an LLM-driven tool loop.

It is built for people who want the speed of CLI workflows without giving up conversational editing.

## Why Vex

- Natural language first: tell Vex what you want instead of memorizing editing syntax
- Zero-setup interaction: type `vex` and start talking
- Original footage stays untouched: edits always happen on a project working copy
- Stateful projects: resume later with timeline history intact
- Real editing tools: trims, overlays, audio edits, subtitle burn-in, silence cleanup, exports, and more
- Multi-provider ready: Gemini by default, Claude when you explicitly choose it
- Terminal-native: fast, scriptable, and easy to integrate into your workflow

## What Vex Can Do

### Core editing

- Inspect video metadata
- Trim clips by timestamp
- Merge multiple clips
- Adjust playback speed for a full clip or a selected segment
- Add fade in, fade out, and fade-through-black transitions
- Add timed text overlays
- Remove silent gaps from raw footage
- Extract selected highlight segments into a shorter cut

### Audio

- Extract audio as `mp3`, `wav`, or `aac`
- Replace a video's audio track
- Mix external audio with original audio
- Mute a selected time range

### Captions and transcript workflows

- Transcribe video locally with Whisper
- Generate `transcript.txt` and `transcript.srt`
- Burn subtitles directly into the video from an SRT file
- Auto-summarize long clips into highlight cuts using transcript-aware segment selection
- Auto-create multiple vertical shorts with captions, ranking, hooks, metadata, and a bundle manifest
- Score each generated short with explainable viral factors
- Generate timestamped B-roll suggestions for each short
- Add transcript-driven punch-in moments for emphasis inside generated shorts

### Export and delivery

- Export with built-in presets for YouTube, Instagram, TikTok, X, and podcast audio
- Export directly from the REPL or from the CLI
- Estimate output size and check disk space before export

### Project system

- Persistent saved projects
- Auto-load a project when you mention a known video path
- Deterministic project loading
- Undo and redo through timeline rebuild
- Timeline inspection and project summaries

## The Intended Experience

This is the default workflow:

```text
> vex

[Vex banner]
Vex > trim the first 30 seconds of "D:\videos\clip.mp4"
Vex > remove awkward pauses
Vex > burn subtitles
Vex > export it for instagram
Vex > /quit
```

You do not need to type subcommands inside the session.

If no project is loaded, include a video path in your first message and Vex handles the rest.

## Installation

### Requirements

- Python 3.11+
- FFmpeg installed and available on `PATH`
- `yt-dlp` available through the Python environment for YouTube downloads

FFmpeg install:

- macOS: `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt install ffmpeg`
- Windows: install from `https://ffmpeg.org/download.html` and add `ffmpeg/bin` to `PATH`

### Install Vex

```bash
git clone https://github.com/AKMessi/vex.git
cd vex
pip install -e .
```

After install, you should be able to launch Vex with:

```bash
vex
```

### Windows PATH note

If `vex` is not recognized after install, add your Python Scripts directory to `PATH`.

Example:

```text
C:\Users\aarya\AppData\Roaming\Python\Python314\Scripts
```

Your path may differ depending on your Python installation.

## Configuration

Copy the environment example:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Then configure your provider.

### Default provider

Vex defaults to Gemini.

```env
PROVIDER=gemini
GEMINI_API_KEY=your_key_here
```

### Claude support

If you want Claude instead:

```env
PROVIDER=claude
ANTHROPIC_API_KEY=your_key_here
```

### Important settings

- `PROVIDER`
- `GEMINI_API_KEY`
- `GEMINI_MODEL`
- `ANTHROPIC_API_KEY`
- `CLAUDE_MODEL`
- `AGENT_PROJECTS_DIR`
- `FFMPEG_PATH`
- `WHISPER_MODEL`

## Quick Start

### First run

```bash
vex
```

Then type a normal sentence with a video path:

```text
Vex > trim the first 30 seconds of "D:\videos\clip.mp4"
```

Then continue naturally:

```text
Vex > remove the silent pauses
Vex > add "DealScout" at the bottom for the first 4 seconds
Vex > export it for youtube
Vex > /quit
```

### If a project already exists

- If exactly one saved project exists, Vex resumes it automatically
- If multiple saved projects exist, Vex starts clean and waits for a file path or an explicit project command
- If you mention a video path that already belongs to a saved project, Vex reuses that project instead of creating a duplicate

## Natural-Language Examples

### Basic trim

```text
Vex > trim the first 15 seconds of "D:\videos\intro.mp4"
```

### Speed up a section

```text
Vex > speed up the section from 1:10 to 1:35 by 1.25x
```

### Remove pauses

```text
Vex > remove the silent gaps
```

### Add subtitles

```text
Vex > transcribe this video
Vex > burn subtitles in yellow at the bottom
```

### Summarize a long video

```text
Vex > make this a 60 second highlight reel
```

### Create viral-style shorts

```text
Vex > turn this podcast into 4 YouTube Shorts with captions
```

### Create shorts from a YouTube link

```text
Vex > make 3 shorts from https://www.youtube.com/watch?v=example123
```

### Export for social

```text
Vex > export it for instagram
```

## Full Tool Surface

These are the editing tools Vex exposes to the agent loop.

| Tool | What it does |
|---|---|
| `get_video_info` | Reads duration, resolution, fps, codec, audio presence, format, and size |
| `trim_clip` | Trims the current working video to a selected range |
| `merge_clips` | Concatenates the current working clip with one or more external clips |
| `adjust_speed` | Changes playback speed globally or for a selected segment |
| `add_transition` | Adds `fade_in`, `fade_out`, or fade-through-black style transitions |
| `add_text_overlay` | Adds timed text overlays to the video |
| `extract_audio` | Exports audio from the current working video |
| `replace_audio` | Replaces or mixes audio with an external track |
| `mute_segment` | Silences audio in a selected time range |
| `trim_silence` | Detects and removes silent gaps from the video |
| `burn_subtitles` | Burns subtitles from an SRT file directly into the video |
| `transcribe_video` | Generates `transcript.txt` and `transcript.srt` using Whisper |
| `summarize_clip` | Uses transcript-aware LLM selection to build a shorter highlight cut |
| `create_auto_shorts` | Builds multiple ranked vertical shorts with transcript analysis, captions, metadata, and a manifest bundle |
| `export_video` | Exports the working video with a named preset |
| `undo` | Rebuilds the project without the last operation |
| `redo` | Reapplies the most recently undone operation |

## CLI Commands

Vex supports both a conversational mode and explicit power-user commands.

### `vex`

Start the interactive REPL.

### `vex shorts`

Generate a packaged shorts bundle directly from an existing project.

```bash
vex shorts --project <project-id> --count 4 --target-platform youtube_shorts
```

### `vex youtube-shorts`

Download a YouTube video and immediately run the auto shorts workflow.

```bash
vex youtube-shorts "https://www.youtube.com/watch?v=example123" --count 4
```

- resumes the only saved project automatically
- otherwise waits for natural-language input
- if no video is loaded, include a file path in your message

### `vex start <video_path> [--name TEXT]`

Create a new project explicitly and open the REPL.

```bash
vex start "D:\videos\clip.mp4" --name "Launch Cut"
```

### `vex repl [--project TEXT]`

Open the REPL for an existing project.

```bash
vex repl
vex repl --project 7e5a4d1c
```

### `vex run "<instruction>" --project TEXT`

Run a single instruction against a saved project and exit.

```bash
vex run "export it for instagram" --project 7e5a4d1c
```

### `vex projects`

List saved projects.

### `vex export <preset_name> --project TEXT [--output TEXT]`

Export without entering the REPL.

```bash
vex export instagram_reels --project 7e5a4d1c --output "D:\exports\clip.mp4"
```

### `vex --version`

Show the installed version.

## REPL Slash Commands

These commands work only inside the interactive session.

| Command | Action |
|---|---|
| `/status` | Show the current project summary |
| `/timeline` | Show the applied timeline |
| `/undo` | Undo the last edit |
| `/redo` | Redo the last undone edit |
| `/export <preset>` | Export immediately with a preset |
| `/provider` | Show the active provider and model |
| `/projects` | List saved projects |
| `/help` | Show available slash commands |
| `/quit` or `/exit` | Save and exit |

## Export Presets

Built-in presets:

| Preset | Description | Format |
|---|---|---|
| `youtube_1080p` | YouTube 1080p HD | `mp4` |
| `youtube_4k` | YouTube 4K UHD | `mp4` |
| `instagram_reels` | Instagram Reels / Stories vertical | `mp4` |
| `instagram_square` | Instagram square feed | `mp4` |
| `tiktok` | TikTok vertical | `mp4` |
| `twitter_x` | X / Twitter landscape | `mp4` |
| `podcast_audio` | Audio-only podcast export | `mp3` |
| `custom` | Start from your own settings | variable |

## How Project Loading Works

- Vex never edits the original source file directly
- each project stores:
  - original source path
  - working copy path
  - timeline operations
  - provider and model
  - session log
  - metadata
- when you mention a video path in the REPL:
  - Vex checks for an existing project for that exact source file
  - if found, it loads that project
  - otherwise, it creates a new project automatically

Default project storage:

```text
~/.video-agent/projects/
```

You can override that with `AGENT_PROJECTS_DIR`.

## Architecture

| Path | Responsibility |
|---|---|
| `main.py` | CLI, REPL, auto-loading, slash commands, Rich terminal UI |
| `agent.py` | Provider-agnostic agent loop and tool orchestration |
| `providers/` | Gemini and Claude adapters behind one interface |
| `tools/` | Agent-callable editing tools |
| `engine.py` | FFmpeg and MoviePy operations |
| `state.py` | Persistent project state and timeline history |
| `presets/export_presets.json` | Built-in export presets |

## Dependencies and Runtime Notes

### FFmpeg is mandatory

Vex depends on FFmpeg for:

- metadata probing
- trims and merges
- audio processing
- subtitle burn-in
- silence detection
- exports

### Whisper is optional

`transcribe_video` requires `openai-whisper` and a local environment capable of running it.

### Text overlays on Windows may require ImageMagick

MoviePy text rendering on Windows can require ImageMagick.

If timed text overlays fail, install it from:

```text
https://imagemagick.org/script/download.php#windows
```

### Subtitle burning depends on FFmpeg subtitle support

Most FFmpeg builds support the `subtitles` filter. If yours does not, subtitle burn-in may fail until you install a build with subtitle filter support.

## Troubleshooting

### `vex` is not recognized

Add your Python Scripts directory to `PATH`, then restart the terminal.

### `FFmpeg was not found in PATH`

Install FFmpeg and verify `ffmpeg` works in the terminal before launching Vex.

### Vex says no video is loaded

Start with a normal sentence that includes a file path:

```text
Vex > trim the first 10 seconds of "D:\videos\clip.mp4"
```

### Subtitle burn-in fails

Check that:

- your `transcript.srt` exists
- your FFmpeg build supports the `subtitles` filter
- the subtitle path is valid on your OS

### Text overlay fails on Windows

Install ImageMagick and retry.

### Transcription fails

Make sure Whisper is installed and usable in your Python environment.

### Summarization does not work

Make sure:

- the active provider API key is configured
- transcription completed successfully
- `transcript.txt` and `transcript.srt` exist in the project working directory

## Contributing

Issues and PRs are welcome.

If you report a bug, include:

- OS
- Python version
- active provider
- the exact command or REPL input
- the traceback or terminal output
- whether the issue happens on a fresh project or a resumed project

## License

MIT. See [LICENSE](LICENSE).
