# Installing And Upgrading Vex

## Recommended Install

Requirements:

- Python 3.11 or newer
- FFmpeg on `PATH`
- `pipx`

Install:

```bash
pipx install vex-video
vex --version
vex setup config
vex renderers doctor
```

HyperFrames visuals additionally require Node.js 22+ and npm:

```bash
vex renderers install hyperframes
vex renderers doctor
```

## Optional Features

```bash
pipx install "vex-video[transcription]"
pipx install "vex-video[manim]"
pipx install "vex-video[all]"
```

For an existing Vex installation:

```bash
vex setup transcription
```

This reuses a compatible system Whisper runtime when one is discoverable,
otherwise it installs and verifies Whisper in the Python environment that runs
Vex. Packages installed into system Python are intentionally isolated from
pipx; Vex crosses that boundary through its external worker when the
interpreter is on `PATH`, or when `WHISPER_PYTHON_PATH` points to it.

Manim also has platform-specific Cairo, Pango, and LaTeX requirements. Blender
is installed separately and discovered from `PATH` or `BLENDER_PATH`.

## Upgrade

```bash
pipx upgrade vex-video
vex renderers install hyperframes
vex renderers doctor
```

Each Vex version uses its own managed HyperFrames runtime directory. This avoids
silently changing renderer dependencies underneath an older Vex installation.

## Install A Release Candidate

Release candidates are published as pipx-verified GitHub prereleases:

```bash
VERSION=0.1.0rc7
pipx install \
  "https://github.com/AKMessi/vex/releases/download/v${VERSION}/vex_video-${VERSION}-py3-none-any.whl"
vex --version
```

Configured release pipelines also publish and verify candidates on TestPyPI.
Use release candidates for validation, not production projects.

## Roll Back

List published versions on PyPI, then install the required version explicitly:

```bash
pipx install --force "vex-video==0.1.0"
vex --version
vex renderers install hyperframes
```

Project files remain in `AGENT_PROJECTS_DIR`; changing the Vex executable does
not delete projects. Back up a project before opening it with an older version
when a release changes project-state schemas.

## Uninstall

```bash
pipx uninstall vex-video
```

The managed renderer runtime and Vex projects are user data and are not deleted
automatically. This prevents package removal from destroying work.
