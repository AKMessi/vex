# Remotion video production

Vex produces complete videos as frame-addressed Remotion compositions. The script, narration timestamps, director contract, and video skill graph inform each shot. Shot candidates compile against a design canvas for the chosen orientation and compete on semantic and structural scores. Panels representing a source object need a grounded label or an explicit outcome mark. Repeating the previous visual medium lowers a candidate's score. When no candidate passes, Vex renders typography drawn from the narration and records the compiler findings.

For prompt-only jobs, the configured text provider can produce a treatment with one spoken sentence and one quoted focus per shot. The parser checks the timing budget, exact focus grounding, and supplied digit values before any speech or rendering jobs start. The accepted narration becomes the source for both timing and visual evidence. For authored scripts, Vex uses the supplied text path directly. The manifest records the treatment provider or an offline fallback.

## Build and verify a production

```bash
vex renderers install remotion
vex generate-video "Explain retrieval and ranking" --renderer remotion --aspect portrait --output-dir ./video-projects
VEX_RUN_RENDER_TESTS=1 python -m pytest -q tests/test_remotion_generation_render.py
```

### Gemini production mode

Set `GEMINI_API_KEY` to a Gemini Developer API key from a Free Tier project
with Cloud Billing disabled. Google AI Pro includes models such as Argon and
Neon in AI Studio's web interface. Vex uses the API project's own model access
and quota. [Google AI plans](https://aistudio.google.com/docs/google-ai-plans)

```bash
vex gemini-models
vex generate-video "Explain retrieval and ranking" --renderer remotion --quality high --gemini-production
```

The catalog command lists models visible to the API key. The AI Studio rate
limit page shows the project's active requests-per-day quota. Production mode
defaults to free-tier Gemini 4 Hydrogen for brief authoring, grounded program
proposals for up to four shots, and a separate review of frames decoded from
each shot in the final export. Accepted model proposals compete with local
candidates through Remotion preflight. [Active project limits](https://aistudio.google.com/rate-limit)
Missing or failed model review fails QA and leaves the rendered artifact and
its `gemini_review.json` available for revision. The report binds its verdicts
to the video hash and the local delivery receipt.

Override the director with `VEX_REMOTION_GEMINI_DIRECTOR_MODEL` and the reviewer
with `VEX_REMOTION_GEMINI_CRITIC_MODEL`. Gemini 3.5 Flash-Lite is a free-tier
alternative if your project shows a higher daily quota for it. Production mode sends briefs and sampled rendered
frames to the Gemini API when the relevant stages run. [Gemini API pricing](https://aistudio.google.com/docs/pricing)

The composition uses the same input props for metadata selection and rendering. Remotion's sequences give shots and captions local frame clocks. Project media is bundled for the job, and media-free jobs can reuse Vex's shared bundle cache. [Remotion input props](https://www.remotion.dev/docs/passing-props) · [Sequences](https://www.remotion.dev/docs/sequence) · [Bundling](https://www.remotion.dev/docs/bundle)

High-quality renders compare the two highest-ranked layouts on up to four shots sampled across the timeline. One browser session captures four frames per layout. Candidates with observed clipping or missing font readiness are rejected from preflight selection. Frame aesthetics contribute to ranking, and Vex records frame digests and measured defects in the shot report. Shots without usable preflight evidence keep the structurally accepted selection.

| Artifact | What it records |
| --- | --- |
| `remotion/input_props.json` | Canvas, frame rate, contiguous shot intervals, selected programs, caption intervals, and audio tracks |
| `remotion/shot_direction.json` | Candidate scores, structural failures, selected program, and typography fallback |
| `generated_video_qa.json` | Final media checks, exact sampled frame indices and digests, and export digest |
| `revision.json` | Original and revised timeline digests, asset digests, render log, and delivery receipt |

## Preview, deliver, and edit a saved production

A generated project can serve as the source for multiple deliveries. Vex reads
the saved frame clock and media, stages pending native shots, and renders
requested frames or a complete export. These steps use saved direction and
audio. A hybrid project with an accepted repair uses that composition when
opened from the project root. Specify its original `remotion/` directory to
work from the first version.

1. Run `vex inspect-video-project PROJECT` for shot IDs, renderers, half-open
   frame intervals, and pending native shots. Add `--json` for automation.
2. Run `vex render-video-project PROJECT --stage-only` to materialize pending
   Manim and HyperFrames assets and save a reusable production folder.
3. Run `vex render-video-project PREPARED --frame 0 --frame 60` to capture
   the requested zero-based frames at a maximum 960-pixel long edge.
   Each preview records the frame number and PNG digest in `revision.json`.
4. Run `vex render-video-project PREPARED --format mov --quality draft --add-to-studio`
   to verify a new export and register a local editing project. Studio records
   the originating manifest and keeps an editable media copy.

The same Remotion composition renders previews and deliveries from the saved
timeline. Sampling a handful of frames avoids a complete encode on each review
pass. Native adapter code comes from the installed runtime and uses saved
declarative direction. [Remotion frame capture](https://www.remotion.dev/docs/renderer/render-frames) · [HyperFrames composition timing](https://github.com/heygen-com/hyperframes/blob/main/docs/concepts/compositions.mdx) · [Manim output settings](https://docs.manim.community/en/stable/tutorials/output_and_config.html)

## Follow-up work

1. **Repair failed shots by editing the timeline.** Use the verified export's frame indices to request a typed scene-graph patch for one failed shot. Render the patched window for comparison, then verify the full export again after accepting a change. Keep earlier revisions available for review.
2. **Measure viewer outcomes against general-purpose agents.** Give Vex, Claude Code, and Codex the same briefs, source assets, pinned Remotion environment, and model budget. Blind the outputs and collect viewer preference, factual recall, and editability scores. Track render completion, render time, cost, and time to deliver a requested revision. The deterministic render fixtures verify pipeline behavior. A comparative quality claim requires the viewer evaluation.
