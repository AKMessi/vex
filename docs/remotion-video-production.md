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

Set `GEMINI_API_KEY` to a Gemini Developer API project key in your environment or
Vex `.env`. Google AI Pro's AI Studio web quota is separate from API billing.
Eligible subscribers can claim $10 USD per month in Google Cloud credits for an
eligible billed project. [Google AI plans](https://aistudio.google.com/docs/google-ai-plans) · [Developer Program credits](https://developers.google.com/profile/help/benefits)

```bash
vex gemini-models
vex generate-video "Explain retrieval and ranking" --renderer remotion --quality high --gemini-production
```

The catalog command lists models visible to the API key. Generation quotas and
billing depend on the project. Production mode uses Gemini 4 Argon to author
the brief and propose a grounded program for up to four shots. Accepted model
proposals compete with local candidates through Remotion preflight. Gemini 4
Carbon independently reviews frames decoded from each shot in the final export.
Missing or failed model review fails QA and leaves the rendered artifact and
its `gemini_review.json` available for revision. The report binds its verdicts
to the video hash and the local delivery receipt.

Override the director with `VEX_REMOTION_GEMINI_DIRECTOR_MODEL` and the reviewer
with `VEX_REMOTION_GEMINI_CRITIC_MODEL`. For a free-tier smoke test, set both
to `gemini-4-hydrogen`. Production mode sends briefs and sampled rendered
frames to the Gemini API when the relevant stages run. [Gemini API pricing](https://aistudio.google.com/docs/pricing)

The composition uses the same input props for metadata selection and rendering. Remotion's sequences give shots and captions local frame clocks. Project media is bundled for the job, and media-free jobs can reuse Vex's shared bundle cache. [Remotion input props](https://www.remotion.dev/docs/passing-props) · [Sequences](https://www.remotion.dev/docs/sequence) · [Bundling](https://www.remotion.dev/docs/bundle)

High-quality renders compare the two highest-ranked layouts on up to four shots sampled across the timeline. One browser session captures four frames per layout. Candidates with observed clipping or missing font readiness are rejected from preflight selection. Frame aesthetics contribute to ranking, and Vex records frame digests and measured defects in the shot report. Shots without usable preflight evidence keep the structurally accepted selection.

| Artifact | What it records |
| --- | --- |
| `remotion/input_props.json` | Canvas, frame rate, contiguous shot intervals, selected programs, caption intervals, and audio tracks |
| `remotion/shot_direction.json` | Candidate scores, structural failures, selected program, and typography fallback |
| `generated_video_qa.json` | Final media checks, exact sampled frame indices and digests, and export digest |
| `revision.json` | Original and revised timeline digests, asset digests, render log, and delivery receipt |

## Highest-leverage follow-up work

1. **Repair failed shots by editing the timeline.** Use the verified export's frame indices to request a typed scene-graph patch for one failed shot. Render the patched window for comparison, then verify the full export again after accepting a change. Keep earlier revisions available for review.
2. **Measure viewer outcomes against general-purpose agents.** Give Vex, Claude Code, and Codex the same briefs, source assets, pinned Remotion environment, and model budget. Blind the outputs and collect viewer preference, factual recall, and editability scores. Track render completion, render time, cost, and time to deliver a requested revision. The deterministic render fixtures verify pipeline behavior. A comparative quality claim requires the viewer evaluation.
