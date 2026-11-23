# Editable production cuts

Studio and the CLI share the same preview and save transaction. The current
cut has a revision number. A preview validates a patch against that revision,
renders sample frames, and stores a receipt. Saving checks the project revision
and the composition, runtime and media hashes from the preview. The saved cut
joins the project's Undo and Redo history after delivery checks pass.

In Studio, open a generated film and use **Revise this cut**. You can also open
H.264 MP4 footage at a supported integer frame rate with **Open footage as an
editable cut**. This starts from the current footage as one source clip.

Select **Live timeline** to play the composition directly in Studio. Scrub the
shared playhead, drag shots to reorder them, or select a shot and use the move
and split controls. Picture, voice, music and captions have separate lanes.
The draft uses the same composition as final delivery. **Review and save**
opens the existing verified revision workflow. Live playback requires an
installed Vex Remotion runtime and browser-compatible source media.

- Splitting, reordering and trim operations keep the original motion times,
  narration ranges and caption cues attached to their source footage. Music is
  a separate backing track for the whole cut.
- **Rewrite narration** records a short line with local HyperFrames speech.
  Listen to the take while reviewing the frames. The line must fit the shot's
  existing time. Vex distributes the new phrase captions across that shot.
- **Direct a shot in plain English** asks the configured model for a scoped
  proposal. It can change captions, authored artwork, narration, or generate a
  dedicated HyperFrames or Manim scene. The model receives the selected shot and
  saved direction notes.
  A proposed Manim scene uses the trusted geometry renderer and requires the
  Manim extra.
- **Project direction and locked shots** stores recurring instructions and
  protects selected shots in production proposals. Ordinary timeline effects
  that bake their result can be opened as a new footage source after editing.
- **Visual references** attaches up to four still images with notes. Choose
  whether to include the images when directing a shot with a vision model.
  Reference images travel with a production archive.
- **Find footage or spoken words** searches the project locally. Transcripts
  appear when they belong to the current footage. Matching timed results seek
  directly to the relevant part of the cut.

The command line exposes the same workflow.

```bash
vex production-cut --project PROJECT_ID
vex production-cut --project PROJECT_ID --attach
vex production-cut --project PROJECT_ID --patch patch.json
vex production-cut --project PROJECT_ID --instruct "Highlight the main curve in amber" --shot beat_02
vex production-reference ./palette.png --project PROJECT_ID --note "Use this palette"
vex production-cut --project PROJECT_ID --instruct "Match the reference palette" --shot beat_02 --use-references
vex production-cut --project PROJECT_ID --commit PREVIEW_ID
vex find-media "main curve" --project PROJECT_ID
vex measure-audio --project PROJECT_ID
vex production-bundle ./portable-cut --project PROJECT_ID
vex production-bundle ./portable-cut
vex production-archive ./portable-history --project PROJECT_ID
vex production-archive ./portable-history --relink-from ./moved-media
vex production-archive ./portable-history
vex relink-source ./moved-original.mp4 --project PROJECT_ID
```

A patch contains the current revision and a list of operations. Use IDs from
`vex production-cut --project PROJECT_ID`.

```json
{
  "revision": 2,
  "operations": [
    {"op": "caption", "id": "CUE_ID", "text": "Light bends when it enters glass."}
  ]
}
```

Use `trim` with start and end in seconds, `split` with a shot ID and timeline
time in seconds, `reorder` with the ordered list of shot IDs, `canvas` with
width and height, `music_gain` with a gain from zero to one, or `narration`
with a shot ID and text. A local bundle contains the saved composition, referenced
media, and delivered MP4, MOV or WebM. Opening it verifies the listed file hashes
and creates a local copy with a new project ID.

Use a production archive to carry the active cut and its Undo and Redo history.
Identical media is stored once. A copied archive can locate missing media by
checksum even if the original filename changed. For an ordinary footage project,
`relink-source` checks the moved original against the copy Vex imported.

Caption-only changes can reuse the verified picture outside the changed frame
interval. Narration takes and newly rendered native assets are cached between
preview and delivery. Changes to the audio, canvas, or overall time may need a
full final composition render. Studio generation continues if the server stops
while its detached worker is alive. If the worker itself stops unexpectedly,
Studio records the interruption. **Retry from saved stages** can reuse the
completed direction and verified narration recording. A provider request
interrupted before Vex can save its answer may incur a charge when retried.

Vex records provider-reported model usage when available and keeps a
conservative estimate for interrupted calls. Set `VEX_PROVIDER_MAX_USD` to an
estimated spend ceiling and `VEX_PROVIDER_RATES` to a JSON object keyed by
`provider/model` with `input` and `output` prices per million tokens. Use the
rates from your provider account. Vex reserves estimated cost before dispatch
and rejects unpriced requests when the ceiling is enabled. Studio production
retries share the original checkpoint workspace and spend ledger.

Inspect usage with `vex model-usage --project PROJECT_ID` or point
`vex model-usage` at a generation or checkpoint folder. This ledger uses your
rates and the provider's reported usage, so the provider invoice remains the
source for the final charge.

### Custom native shots

Configure a local Linux Docker Engine to opt in to source written directly in
Remotion, Manim, or HyperFrames. From a Vex source checkout, build the image and
set `VEX_SANDBOX_IMAGE` to the `sha256:…` ID printed by the second command.

```sh
docker build -f sandbox/Dockerfile -t vex-native .
docker image inspect --format '{{.Id}}' vex-native
vex sandbox-render scene.py scene.mp4 --engine manim --width 640 --height 360 --fps 24 --frames 48
```

Remotion source registers a composition named `VexScene`. Manim source declares
a `VexScene` class. HyperFrames source declares its root composition and timing
in a complete HTML document. The same command accepts explicit local `--asset`
inputs. Generated clips are silent and must match the requested frame count.
In Studio, select a shot and enable native code while asking for a proposal.
Vex previews the selected shot before it can replace the current cut. Jobs run
without network access, as an unprivileged user, with a read-only root and a
temporary input directory containing only the submitted files.

### OpenTimelineIO handoff

Install `vex-video[interop]` to exchange `.otio` cuts with another editor.

```sh
vex nle-export --project PROJECT_ID --format otio --output-dir ./handoff
vex nle-import ./handoff/CUT.otio --media-root ./MEDIA_DIRECTORY
```

Ordinary footage exports retain source ranges when the cut can be expressed
as straight edits. Generated films export a clip per shot using ranges in the
finished video to preserve the visible graphics, captions, and mix. The import
supports a straight-cut picture track with optional matching source audio.
It validates the frame grid, copies the referenced local files into the new
project, and preserves the edit points. Use a Vex production archive to carry
the original editable renderer layers and full saved cut history.

### Blind quality review

Use `vex blind-review comparisons.json ./study` to package identical briefs
rendered by two different versions. The manifest has a `cases` array. Each
entry contains `id`, `category`, `baseline`, and `candidate`, with local H.264
MP4 paths. Give reviewers only the `study/review` folder. Alternate which
reviewers open `index.html` and `mirror.html` to balance the viewing order.
Reviewers play each pair, choose a preference or tie, and download a ballot.

Run `vex score-review ./study/study-key.json ballot1.json ballot2.json` to tally
wins, ties, and comments. Keep the key private until voting has ended. Compare
these preferences with measured render time and model usage from the same
briefs before deciding whether a change improved the workflow.

To compare a caption edit against a full rebuild on a verified composition,
run `python -m scripts.benchmark_production_revision PROJECT OUTPUT --caption 0`
from a source checkout. The report includes elapsed time, frame counts and
pixel differences near the caption boundary.
