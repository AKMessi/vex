# Visual harness implementation ledger

Branch: `feat/visual-harness-native-vision`. Each tested batch is committed and pushed.

## Completed batches

1. **Publication integrity:** negation/direction guards, validated communication contracts, zero-frame rejection, chronological evidence selection, final Remotion re-verification, content-bound receipts, nonregressing repair eligibility, complete-contract cache keys, pairwise caching, concurrent-safe cache writes. Added adversarial regression cases. Full suite: 764 passed.
2. **Groq and native vision repair:** native tools/streaming provider, image+text transport, at most three chronological contact sheets, Qwen visual authoring/QA, bounded pixel-conditioned patch proposals, revalidation and independent re-render verification. Configured local visual overrides without committing secrets. Live catalog and synthetic-image smoke passed; 55 provider/visual tests and 17 Groq/packaging tests passed.
3. **Executable scene contracts:** full-keyframe shared motion evaluation including physical spring curves, cross-runtime conformance tests, grounded chart series, vector paths/icons, image registry binding, nested render groups, common Hyperframes constraint solving, bundled Inter with OFL license, browser text/layout measurements. Real Remotion 1280×720/30 FPS render: quality passed, local score 0.8395, 17 browser samples passed. Hyperframes HTML browser probe: 5 samples passed. These results establish working rendering/checks, not a human preference improvement percentage.

4. **Durable execution and native QA routing:** SQLite stages with leases/heartbeats, dependency and output hashing, shared process-visible call/token/render budgets, cancellation/deadline checks, bounded isolated renderer processes, and native Groq/Gemini/Claude transports for legacy Hyperframes critics. A real isolated Remotion render followed by an identical request produced one render and a stage-cache hit. Recovery/budget, provider, and workflow tests passed. Full suite after batch 3: 771 passed.

5. **Reference-guided design and common selection:** concrete grounded keyframe sketches and verified prior frame references reach Qwen's native image input; project-local bounded experience storage records attribution and verifies reference hashes; stable semantic identities and transcript-word cues accompany signed programs; cross-renderer and Remotion preview selection use the shared verifier; Hyperframes relation ablation can rerender only the selected program relations instead of erasing a broad rectangle. Removed fabricated fallback percentages. Targeted suites: 58, 42, 55, and 19 tests passed in the relevant checks.

6. **Shared workflow policy and bounded context:** full-video beat windows use the same blind meaning gate, typed repair/render path, final composite re-verification, and receipts as Auto Visuals. Dispatch context compaction keeps tool-call/result exchanges together and retains user constraints while the original log stays authoritative. Native Qwen reference authoring now uses strict schema-constrained compact patches and quota-aware output limits; a live request produced a valid model-authored program with no warnings/rejections. Shared portfolio, provider, generation, evidence and native integration checks passed (29 and 34 tests in the targeted runs).

7. **Source-based delivery and exported pixel evidence:** visual layers persist in the rational edit graph, including asset offsets after cuts/trims. One FFmpeg pass renders supported source spans and approved layers; fractional FPS is preserved, balanced delivery uses CRF 16, and an optional ProRes master is available. Composite QA now samples four native frames for replace, PiP, and alpha modes. A real 30000/1001-FPS source with audio and an approved Remotion asset passed duration/audio checks and four delivered-frame comparisons (0.9846–0.9893 similarity). Unsupported legacy edits retain explicit rendered anchors. Environment-dependent no-credential tests now clear both native providers.

8. **Rendered regression and operational hardening:** added a fixed six-case corpus across all three aspect ratios, actual renderer/video/frame evidence, comparisons, human-rating ingestion, and a pinned browser/render CI workflow. Real renders exposed and corrected portrait layout, low-contrast typography, missing chart labels/units, legacy capture timing, worker-policy drift and system-temp disk exhaustion. Browser measurements now include text contrast and observed panel coverage. Authoring/compilation stages resume from SQLite, failed proposals remain retryable, and native patches bind to their source program. Groq supports separate reasoning settings by visual role, bounded input-resolution recovery and strict repair schemas. Live Qwen QA inspected actual Remotion frames; a live repair produced three validated proposals.

Final validation: **797 tests passed**, all **12 rendered corpus cases passed local/technical QA**, and all **325 Python source files parsed**. Wheel/source builds, metadata checks and package-resource checks passed. The offline corpus used **zero model calls**; its semantic states remain unverified. A separate live Qwen eight-frame check returned usable critique and withheld publication for design/temporal weaknesses, confirming that a local pass cannot bypass independent verification. No human preference gain is inferred from these results.

## Implementation map

9. **Cross-platform CI follow-up:** the new GitHub render job exposed npm's Unix symlink layout in the installer. Package-root discovery now accepts both `.bin` launchers and resolved `hyperframes/dist/cli.js` paths and retains the explicitly selected Node executable. Added direct-entrypoint and symlink regression cases, and enabled the existing OS/Python/package CI matrix on this implementation branch. Installation diagnostics are uploaded when setup fails.

| Review upgrade | Core implementation |
|---|---|
| 1. Correct meaning | `communication_contract.py`: polarity/direction guards, object-scoped quantities; independent `verifier.py` |
| 2. Final evidence | `evidence.py`, `workflow_stages.py`, final Remotion re-verification and hashed publication receipts |
| 3. Faithful motion | shared `visual_motion.mjs` and `motion_state.py`, complete authored keyframes and springs |
| 4. Typed visual language | Open Visual Program/SceneGraph schemas, grounded data, geometry, registry assets and parent groups |
| 5. Actual layout | browser telemetry, font readiness, glyph bounds, clipping, contrast and observed coverage |
| 6. Image references | grounded keyframe sketches and verified frame references in native designer requests |
| 7. Shared selection | rendered cross-renderer tournament and Remotion preflight use independent eligibility |
| 8. Targeted repair | native frame counterexamples, bounded strict patches, source binding and monotonic selection |
| 9. Continuity | semantic identities, consistent symbols, transcript cues and reading-hold requirements |
| 10. Benchmark | `benchmark.py`, fixed packaged corpus, comparison/calibration reports, rendered CI |
| 11. Shared workflows | `harness.py`, Auto Visuals and generated-video beat verification/repair |
| 12. Routing/budgets | native multimodal transport, Groq role settings, shared usage budgets and evidence caches |
| 13. Recovery/isolation | SQLite authoring/render stages, leases/heartbeats, isolated workers, cancellation and scratch cleanup |
| 14. Delivery | rational edit-graph visual layers, source-based rendering, fractional FPS and native composite evidence |
| 15. Experience | bounded project memory, attribution and hash-checked verified references |
| 16. Typed stages/context | common evidence/publication stages, workflow profiles, durable compile/render boundaries and bounded dispatch context |

## Validation limits

These are engineering implementations and regression checks. No human preference improvement percentage is claimed: the rating interface ships without invented ratings. The fixed corpus complements adversarial tests, but larger held-out creator briefs and multiple independent human reviewers are still needed to calibrate quality judgments. Offline benchmark semantics remain explicitly unverified; `--vision` enables the separate model gate.

Unsupported legacy edits retain rendered anchors. Browser telemetry does not prove arbitrary SVG path clearance or all nested connector routes. The existing large planning executor and compatibility adapters remain; this change adds reusable stage boundaries rather than claiming every legacy function has been replaced. Native generation, source-grounded imports and artistic/manual content continue to require workflow-appropriate checks.

The local configuration uses Groq Qwen for visual authoring/vision/repair, with strict publication verification and a quota-aware output cap. General chat keeps its existing provider. API keys, generated media and the existing `deliverables/` directory are not committed.
