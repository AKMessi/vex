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
- **Find footage or spoken words** searches the project locally. Transcripts
  appear when they belong to the current footage. Matching timed results seek
  directly to the relevant part of the cut.

The command line exposes the same workflow.

```bash
vex production-cut --project PROJECT_ID
vex production-cut --project PROJECT_ID --attach
vex production-cut --project PROJECT_ID --patch patch.json
vex production-cut --project PROJECT_ID --instruct "Highlight the main curve in amber" --shot beat_02
vex production-cut --project PROJECT_ID --commit PREVIEW_ID
vex find-media "main curve" --project PROJECT_ID
vex measure-audio --project PROJECT_ID
vex production-bundle ./portable-cut --project PROJECT_ID
vex production-bundle ./portable-cut
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

Caption-only changes can reuse the verified picture outside the changed frame
interval. Narration takes and newly rendered native assets are cached between
preview and delivery. Changes to the audio, canvas, or overall time may need a
full final composition render. Studio generation continues if the server stops
while its detached worker is alive. If the worker itself stops unexpectedly,
Studio records the interruption. **Retry from saved stages** can reuse the
completed direction and verified narration recording. A provider request
interrupted before Vex can save its answer may incur a charge when retried.

To compare a caption edit against a full rebuild on a verified composition,
run `python -m scripts.benchmark_production_revision PROJECT OUTPUT --caption 0`
from a source checkout. The report includes elapsed time, frame counts and
pixel differences near the caption boundary.
