# Smartie Bundle Import

Smartie support is an optional import path. Smartie remains the lightweight recorder; Vex acts as the post-production and render backend when you point it at a Smartie recording bundle.

## Bundle Format

A Smartie bundle is a directory with these required files:

```text
manifest.json
recording.webm
attention.timeline.json
```

`manifest.json` may declare another source video path instead of `recording.webm`. Vex accepts `recording`, `recording_file`, `recording_path`, `source_video`, `video`, `video_path`, and equivalent nested `files` entries.

Optional files:

```text
recording.smartie.json
preview-thumbnails/
```

Missing optional files do not fail import.

## Import

Create a new Vex project from a Smartie bundle:

```bash
vex import-smartie /path/to/smartie-bundle --project "Smartie Demo"
```

`--project` can be:

- an existing Vex project id, which updates that project from the Smartie bundle
- a new project name
- a filesystem directory for an explicit project location

Import validates the bundle, copies the source video into the Vex project working directory, parses `attention.timeline.json`, and writes a deterministic Smartie effect plan. It does not render by default.

## Render

Render the Smartie attention camera plan immediately:

```bash
vex import-smartie /path/to/smartie-bundle --project "Smartie Demo" --render
```

This compiles Smartie attention telemetry into `smart_zoom_segment` effects and sends them through the existing Vex timed-effect FFmpeg pipeline. Audio is preserved through the same path used by existing auto-effects.

The importer also supports module execution:

```bash
python -m tools.import_smartie /path/to/smartie-bundle --project "Smartie Demo" --render
```

## Planning Model

The Smartie planner converts attention telemetry into a typed camera plan. It smooths noisy points, merges nearby intent, avoids micro-zooms, enforces minimum shot duration, caps zoom speed, keeps focus centers inside renderable bounds, and leaves wide shots wherever attention data is weak.

The first camera effect is:

```text
smart_zoom_segment
```

Each segment stores start/end time, normalized focus coordinates, target scale, easing/smoothing metadata, confidence, cue type, and the reason for the move.

## Current Limitations

- Smartie import uses the source recording and attention telemetry only; preview thumbnails are recorded as metadata but are not analyzed.
- Rendering uses Vex's FFmpeg effect compiler. It intentionally avoids full-frame video analysis during import.
- The planner is deterministic and conservative. Weak or noisy telemetry may produce few or no zoom segments instead of forcing distracting motion.
- Smartie support is additive and opt-in. Existing Vex commands, agentic editing workflows, auto-visuals, auto-effects, and render paths remain unchanged unless you run the Smartie import path.
