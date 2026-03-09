# Vex

Vex is a Python CLI video editing agent that turns natural-language instructions into concrete editing operations on a safe working copy of your footage. It supports interactive and one-shot workflows, keeps persistent project state, and swaps between Gemini and Claude through a provider abstraction layer.

## Prerequisites

- Python 3.11 or newer
- FFmpeg installed and available on your `PATH`

FFmpeg installation:

- macOS: `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt install ffmpeg`
- Windows: download from `https://ffmpeg.org/download.html`, then add `ffmpeg/bin` to `PATH`

## Installation

1. Clone this project.
2. Create and activate a Python 3.11+ virtual environment.
3. Install dependencies:

```bash
pip install -e .
```

## Configuration

1. Copy `.env.example` to `.env`.
2. Fill in the provider API key you want to use.
3. Set `PROVIDER=gemini` or `PROVIDER=claude`.

## Switching Providers

Use `PROVIDER=gemini` for Google Gemini or `PROVIDER=claude` for Anthropic Claude.

## CLI Commands

Examples below assume you run the CLI with `python main.py ...`.

- Start a new interactive project:

```bash
python main.py start ~/Downloads/footage.mp4 --name "My Vlog"
```

- Reopen the latest project in REPL mode:

```bash
python main.py repl
```

- Reopen a specific project:

```bash
python main.py repl --project 0f7b90f0
```

- Run one instruction and exit:

```bash
python main.py run "trim the first 30 seconds and export for youtube" --project 0f7b90f0
```

- List projects:

```bash
python main.py projects
```

- Export directly:

```bash
python main.py export youtube_1080p --project 0f7b90f0 --output ~/Downloads/my_vlog.mp4
```

## REPL Slash Commands

| Command | Action |
|---|---|
| `/status` | Show the current project summary |
| `/timeline` | Show timeline operations in a table |
| `/undo` | Undo the most recent edit by rebuilding from source |
| `/redo` | Redo the most recently undone edit |
| `/export <preset>` | Export immediately using a preset |
| `/provider` | Show the active provider and model |
| `/projects` | List all saved projects |
| `/help` | Show available slash commands |
| `/quit` or `/exit` | Save and leave the REPL |

## Export Presets

| Preset | Description | Format |
|---|---|---|
| `youtube_1080p` | YouTube - 1080p HD | `mp4` |
| `youtube_4k` | YouTube - 4K UHD | `mp4` |
| `instagram_reels` | Instagram Reels / Stories - 9:16 vertical | `mp4` |
| `instagram_square` | Instagram Feed - 1:1 square | `mp4` |
| `tiktok` | TikTok - 9:16 vertical | `mp4` |
| `twitter_x` | Twitter / X - landscape | `mp4` |
| `podcast_audio` | Podcast - audio only MP3 | `mp3` |
| `custom` | Start from a custom settings object | variable |

## Example Session

```text
$ python main.py start ~/Downloads/footage.mp4 --name "My Vlog"

[Vex banner]

Vex > cut out the first 45 seconds and the last minute
get_video_info...
trim_clip...
trim_clip...
Done. Removed intro and outro.
[SUGGESTION]: The cut lands abruptly - reply 'yes' to apply or continue.

Vex > yes
add_transition...
Crossfade applied.

Vex > add "TechWithAlex" at the bottom for the first 5 seconds
add_text_overlay...
Text overlay added.

Vex > /export youtube_1080p
Exporting... 100%
Saved: ~/Downloads/My_Vlog_youtube_1080p.mp4

Vex > /timeline
1  trim_clip
2  trim_clip
3  add_transition
4  add_text_overlay

Vex > /quit
Project saved. Goodbye.
```

## Architecture Overview

`agent.py` owns a provider-agnostic editing loop built around unified neutral messages, a shared tool schema list, and a tool executor registry. The `providers/` package hides all Gemini and Claude SDK details behind `BaseLLMProvider`, while `state.py` persists project timelines and session logs to JSON so edits can be resumed and undo/redo can rebuild from original sources. `engine.py` centralizes FFmpeg and MoviePy operations, and `tools/` provides small executors that validate inputs, call engine functions, update state, and return structured tool results.
