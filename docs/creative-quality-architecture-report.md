# Creative Quality Architecture Upgrade

## Status

Implemented, tested, committed, and pushed to `main` on June 11, 2026.

This upgrade addresses the largest quality gaps found in a repository-wide review of
Auto Visuals, HyperFrames renderer routing, creative-run persistence, final
compositing, Auto B-roll, Shorts, effects, grading, export, and project-state
promotion.

The central change is that Vex no longer treats one locally successful decision as
proof that the final edit is good. Quality is now evaluated at four different levels:

1. candidate relevance
2. coherence of the selected set
3. quality of the rendered asset
4. integrity of the final encoded composite

Historical outcomes can influence future ranking, but only through bounded priors.
They cannot bypass semantic, path, render, timing, or publish QA.

## Commit Ledger

| Commit | Architecture change |
| --- | --- |
| `034bdb9` | Deterministic creative-set optimization for plan and post-render selection |
| `4d46b0d` | Bounded renderer quality tournaments and ranked renderer API |
| `4c427bc` | Closed-loop creative quality priors from actual render outcomes |
| `4c7fcf2` | Final encoded-composite acceptance gate for Auto Visuals |
| `df9fa83` | Shared composite acceptance gate for Auto B-roll |

## Audit Findings

### Candidate selection was locally correct but globally weak

Auto Visuals previously accepted candidates independently, sorted them by timestamp,
and truncated the list to `max_visuals`. A later high-value explanation could lose to
an earlier weaker candidate simply because it appeared first.

Post-render QA had the same structural problem. It walked overlays in time order and
rejected the later item when two visuals overlapped, even if the later render had much
stronger semantic or visual evidence.

### Renderer success was mistaken for renderer quality

Renderer resolution used static capability scores and returned the first renderer
that completed. Render QA ran only afterward. If that asset failed QA, Vex dropped the
visual instead of comparing it with another compatible renderer.

### The creative registry was write-only

`creative_runs.json` preserved manifests and summary scores, but no planning or
renderer decision consumed those outcomes. Repeated project-local evidence therefore
could not improve future runs.

### Final composites were not verified

Auto Visuals and Auto B-roll verified plans or source assets, then accepted the
FFmpeg output after metadata probing. The system did not prove that the selected
visual was actually present in the encoded edit, that audio survived, or that
duration and resolution remained stable.

### Other release paths already had stronger boundaries

Auto Shorts validates rendered outputs and separates accepted, rejected, and draft
artifacts. Auto Effects validates actual output metadata and preview frames. Export
has encode validation. Color grading has perceptual and creative evaluation. The
largest comparable release-boundary gap was shared by Auto Visuals and Auto B-roll.

## New End-to-End Flow

```text
transcript / imported SRT / source frames
  -> VideoUnderstandingGraph
  -> visual candidates
  -> independent semantic and director gates
  -> creative-set optimizer
  -> HyperFrames semantic compiler / specialist renderer path
  -> renderer quality tournament when routing is flexible
  -> rendered-asset QA
  -> post-render creative-set optimizer
  -> FFmpeg composite
  -> encoded-composite acceptance QA
  -> project-state promotion after publish QA
  -> outcome registry
  -> bounded priors for a future run
```

Strict renderer requests stay strict. Manual paths and hard failures remain outside
the learned policy.

## Creative-Set Optimizer

`tools/creative_optimizer.py` introduces `creative-set-optimizer-v1`.

The optimizer treats the requested visuals as a portfolio rather than unrelated
items. Its objective combines:

- director or rendered QA score
- transcript-copy alignment
- graph visual opportunity
- retention value
- topic alignment
- confidence
- coverage of graph beats and recurring concepts
- chapter or episode coverage
- intent and renderer diversity
- penalties for repeated intent-template signatures
- hard timing-conflict constraints

Selection is deterministic. Stable time and ID keys resolve equal scores. A bounded
local-improvement pass can replace an initially selected candidate with a stronger
candidate or a candidate that improves semantic coverage.

The optimizer runs twice:

1. after Auto Visuals Director, before expensive rendering
2. after render QA, before final compositing

This means a visually stronger overlapping render can replace a weaker earlier one.
Manifests record:

- objective score
- selected candidate evidence
- portfolio contribution per selected visual
- beat and concept coverage
- intent and renderer diversity
- every excluded candidate and its reason
- conflicting visual IDs

Duplicate planner IDs are normalized internally so candidates are not silently lost.

## Renderer Quality Tournament

`renderers.rank_renderers()` exposes an ordered, typed list of compatible renderer
matches. `resolve_renderer()` remains the single-choice compatibility API.

For flexible `auto` or `both` runs, Auto Visuals can execute
`renderer-quality-tournament-v1`:

- candidates are filtered by semantic suitability
- each renderer uses an isolated workspace
- failed renderers do not consume the successful-contender budget
- every completed contender receives the same rendered-asset QA
- passing contenders outrank failing contenders
- final promotion uses render QA, then resolver score as a deterministic tie-break
- all attempts, errors, QA reports, paths, and scores remain in renderer metadata

The default tournament size is two successful contenders and is capped at three.
Strict `renderer=hyperframes`, `renderer=manim`, `renderer=ffmpeg`, and
`renderer=blender` requests still use one renderer and do not silently reroute.

This changes fallback from failure-only behavior into quality promotion where the
content genuinely supports multiple engines.

## Closed-Loop Quality Priors

`tools/creative_registry.py` now builds `creative-quality-policy-v1` from compact,
text-free outcome signals stored in prior project-local Auto Visuals records.

Signals contain:

- renderer
- visual intent
- template
- rendered QA score
- rendered QA verdict
- whether that contender was ultimately published

The policy uses:

- the latest 30 matching runs
- at most 64 outcome signals per run
- recency weighting
- a minimum of three observations per bucket
- Bayesian shrinkage toward project-wide quality
- a hard adjustment bound of `-0.04` to `+0.04`

Priors exist for renderer, renderer-plus-intent, and template. They can slightly
adjust director scoring, creative-set scoring, and tournament ordering. They do not
alter hard QA floors or turn a failing render into a passing render.

Failed-QA runs now write manifests and registry records too. Vex therefore learns
from rejected renders instead of learning only from successful outputs.

Malformed labels are normalized, non-finite quality values are discarded, booleans
are normalized, and sparse history has exactly zero effect.

## Encoded-Composite Acceptance Gate

`tools/composite_qa.py` introduces `visual-composite-qa-v1`, shared by Auto Visuals
and Auto B-roll.

Before project state is changed, the gate verifies:

- output duration remains within a frame-aware tolerance
- output resolution matches the source
- source audio is preserved when present
- the output file has nonzero media size
- every full-screen replacement asset exists
- the midpoint of each replacement is extractable from both asset and final output
- the encoded output frame is perceptually similar to the promoted asset
- the final frame is not effectively blank

Frames are extracted through an argv-only FFmpeg command at `48x27` RGB. The same
fill-and-crop geometry used by the compositor is applied before comparison. The
default similarity floor is `0.72`.

If composite QA fails:

- the project working file is not changed
- no timeline operation is committed
- the unpromoted output is retained for diagnosis
- `run_status.json` records the failed phase and full report
- a failed manifest records exact issues and artifacts
- the user receives the manifest path and concrete failure reasons

This makes final publication transactional with respect to visual QA. A generated or
stock asset is not considered successful until it survives the actual composite.

## Configuration

```dotenv
AUTO_VISUALS_RENDERER_TOURNAMENT_SIZE=2
VISUAL_COMPOSITE_SIMILARITY_FLOOR=0.72
```

`AUTO_VISUALS_RENDERER_TOURNAMENT_SIZE` accepts `1` through `3`.

`VISUAL_COMPOSITE_SIMILARITY_FLOOR` is clamped to `0.50` through `0.98`.
`AUTO_VISUALS_COMPOSITE_SIMILARITY_FLOOR` remains accepted as a backward-compatible
environment alias.

Existing controls remain unchanged:

- generic commands default to `quality_only`
- explicit counts use stronger coverage pressure
- strict renderer requests do not reroute
- hard semantic, path, timing, render, and QA failures cannot be bypassed

## Runtime and Manifest Changes

Auto Visuals planning previews now expose:

- renderer selection strategy
- tournament size
- estimated render count
- creative policy version and sample counts

Auto Visuals manifests additionally expose:

- creative-set optimization reports
- renderer tournament attempts
- creative policy snapshot
- rendered contender outcomes
- final composite QA

Auto B-roll manifests additionally expose:

- final composite QA
- failed unpromoted output when publication is rejected

Project artifact summaries and notes include composite QA scores.

## How This Improves Vex

### Better explanation coverage

Selection now rewards distinct semantic beats and concepts, so the budget is less
likely to be consumed by repetitive visuals around the same idea.

### Better use of rendering cost

Weak candidates are removed before rendering, but flexible renderer runs can spend
bounded extra work when a real quality comparison is possible.

### Fewer false successes

A renderer exit code, an MP4 file, or a plausible metadata probe is no longer enough.
The final encoded edit must contain the intended replacement and preserve core media
properties.

### Better failure recovery

Failed outputs are retained and explained without corrupting project state. Users can
inspect manifests and `run_status.json` instead of reverse-engineering a generic
failure message.

### Conservative project adaptation

Repeated local outcomes can improve future ranking, but the bounded prior prevents a
small or stale history from dominating current evidence.

### More reproducible behavior

Selection, tie-breaking, score bounds, contender limits, and failure reasons are
deterministic and persisted.

## Verification

Final verification on June 11, 2026:

- full repository suite: `320 passed`
- Python compile checks: passed
- `git diff --check`: passed
- focused optimizer/director/semantic QA suite: `26 passed`
- focused renderer promotion suite: `39 passed`
- focused outcome-learning suite: `43 passed`
- focused composite/publish suite: `47 passed`
- focused shared B-roll/config suite: `42 passed`
- real FFmpeg composite smoke:
  - source: `640x360`, `5.0s`, H.264 plus AAC audio
  - replacement asset: `640x360`, `2.0s`
  - output duration: exact
  - output resolution: preserved
  - source audio: preserved
  - replacement-frame similarity: `0.9996`
  - composite QA score: `0.9998`

The only suite warning is an upstream `google.genai` deprecation warning under
Python 3.14.

## Remaining Platform Work

The audit did not justify large unrelated rewrites in this pass. The next platform
upgrades remain:

- durable background jobs with resume, cancellation, and retry
- typed, migration-aware timeline operations
- a unified content-addressed asset registry and render cache
- renderer process isolation and resource limits
- full media fixture coverage for vertical, square, variable-frame-rate, and
  path-hostile inputs
- project-level evaluation dashboards that compare quality distributions over time

These are important, but they do not replace the quality boundaries implemented
here. The current release specifically removes the highest-impact ways a visually
weak or broken automatic result could be selected, promoted, and committed.
