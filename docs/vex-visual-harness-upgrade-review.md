# Vex visual quality and architecture review

Reviewed 9 October 2026 against commit `0b67a9bc64fb9ccce79ab8965692483ea9305780`.

This is a review and proposed implementation sequence. No runtime code, configuration, dependencies, or existing tests were changed. The existing untracked `deliverables/` directory was preserved.

## Recommended upgrades, in simple words

| # | Upgrade | Problem it solves | How it improves Vex |
|---|---|---|---|
| 1 | Check actual meaning, including direction and negation | The semantic scorer can approve an explanation that says the opposite of the source | Correct explanations become a hard requirement, so attractive but misleading visuals cannot win |
| 2 | Verify the exact final video with complete frame evidence | Early frames and preview approval can stand in for evidence about the final output | The video inserted into the timeline is the video that passed the checks |
| 3 | Execute every authored motion track faithfully | Hyperframes reduces rich tracks to simple endpoint changes; spring names in the shared Remotion graph map to polynomial easing | Objects move, transform, reveal, and settle according to the explanation instead of repeating generic entrance animations |
| 4 | Expand the typed visual language and enforce real renderer capabilities | The schema and renderers disagree about data, artwork, groups, and assets | Vex can create accurate charts, meaningful icons, custom geometry, and real source-image compositions |
| 5 | Measure layout and typography in the actual browser | Estimated character widths and declared rectangles can differ from what viewers see | Text stays readable, diagrams fit, and connectors remain attached across landscape, square, and portrait formats |
| 6 | Give the designer actual visual references | Current reference boards describe images in text | Vex can aim for a concrete visual standard and compare the rendered result against it |
| 7 | Select candidates through one shared preview tournament | Different pipelines rank different proxies and discard alternatives before independent verification | The clearest rendered explanation wins, with controlled rendering and model cost |
| 8 | Repair the specific visible failure | Repairs often apply broad operations; counterfactual checks erase broad image regions | Vex fixes the failed element, timing, or relation while preserving the parts that already work |
| 9 | Plan visual timing and identity across the whole video | Diversity scoring does not ensure stable symbols, reading time, or narration alignment | Repeated concepts remain recognizable and explanatory changes appear when they are spoken |
| 10 | Benchmark real rendered quality | Passing compiler and mocked tests does not establish motion-design quality | Each upgrade must demonstrate better comprehension and visual preference on a fixed, human-calibrated corpus |
| 11 | Use the same quality harness across generation routes | Auto Visuals, directed/manual visuals, and full-video generation have different orchestration and gates | Improvements reach every supported workflow and quality policy becomes consistent |
| 12 | Share model routing, budgets, and evidence caches | Planning, vision, repairs, and pairwise comparisons use separate policies and SDK paths | Vex spends compute on difficult scenes, supports suitable models consistently, and avoids repeated work |
| 13 | Resume work by stage and isolate renderer workers | Job status persists, but interrupted visual pipelines generally restart the executor | Long runs recover completed work, cancellation stops child processes, and failed renders do not waste the whole job |
| 14 | Preserve quality through compositing and export | Re-encoding, integer FPS conversion, and coarse composite QA can damage approved graphics | Sharp text, stable timing, clean transitions, and readable overlays survive into the delivered video |
| 15 | Remember verified failures and successes | Aggregate scores and renderer priors do not explain which change helped | Future scenes reuse relevant, tested lessons rather than repeating known mistakes |
| 16 | Split orchestration into typed stages | Large functions and overlapping representations make quality policy hard to maintain | Renderer upgrades become easier to test, reuse, debug, and migrate safely |

The largest direct quality gains should come from 1–8. Upgrade 10 should begin alongside 1–2, because it supplies the measurement needed to establish whether the other changes help. The rankings are engineering judgments from this review, not measured Vex improvement percentages.

## What already exists

Vex already has a valuable foundation: source-grounded `VisualExplanationIR`, communication contracts, concept search, Open Visual Program, SceneGraph v2, creative direction, structural QA, frame critics, bounded repair, an order-reversed pairwise judge, portfolio selection, project snapshots, a SQLite execution ledger, asset metadata, content-addressed media storage, and a rational edit graph.

These proposals extend that foundation. In particular, adding another named director, another scene representation, or another blanket instruction to produce beautiful visuals would duplicate existing machinery without addressing the execution and verification gaps below.

Some existing architecture documents describe work that is now implemented. The code, test results, and observed artifacts were treated as the authority for this review.

## Scope and validation

- Inventoried all 299 tracked Python/JavaScript source and test files: 128,267 lines, including 100 test files. All tracked Python files parsed successfully.
- Mapped 195 runtime Python modules and 543 resolved internal import edges. This static map does not capture every dynamic import or import embedded inside generated code strings.
- Read the main Auto Visuals execution path, primary and reserve planning, renderer tournaments, shared direction, semantic verification, repair, frame sampling, both target renderer adapters and their scene runtimes, and full-video generation.
- Inspected related project state, source timing, compositing, asset/cache/promotion, provider, job/Studio, editing, effects, shorts, grading, plugin, packaging, and CI boundaries and their relevant tests.
- Ran `python -m pytest -q`: **754 passed, 4 failed**, in 71.29 seconds on this Windows/Python 3.14 environment.
- Re-ran the four failures individually as a group; all four reproduced.
- Ran small offline behavior probes using existing test fixtures. They reproduced the false semantic approval, zero-frame publication, first-four-frame selection, failed-final-QA merge, cache aliasing, and chart-data schema mismatch described below. No paid model requests were made.
- Visually inspected an existing Hyperframes contact sheet and its QA artifacts. This was a historical output, not a newly generated benchmark result.

This is a repository-wide architecture review with deep inspection of the target visual paths. It is not a claim that every line of the approximately 107,000 runtime source lines was manually audited, or that every external integration was exercised. Fresh model-backed quality comparisons, fault-injection recovery tests, and new end-to-end renders remain necessary to measure the proposed upgrades.

## Concrete findings

### F1: Opposite meanings can pass semantic verification

**Observed, reproduced.** `semantic_text_score()` and `_semantic_recovery_score()` in `vex_visuals/communication_contract.py:475` and `:620` principally use token overlap, synonyms, and numeric presence. They do not establish subject/object direction or proposition polarity.

Using the existing communication fixture and normal verifier payload, I replaced every decoded answer and every sequence statement with `It is false that <expected statement>`, and similarly negated the thesis. `evaluate_verifier_payload()` returned:

```json
{"state":"verified","publishable":true,"semantic_passed":true,"semantic_score":1.0,"semantic_issues":[]}
```

The scorer also returned `1.0` for the pair `The planner selects the tool` / `The tool selects the planner`.

This reproduces a grading defect; it does not establish how often the vision model makes these mistakes on real renders. It means the grader cannot reliably reject them when they occur. Fix before treating verifier scores as optimization targets.

### F2: Missing frame evidence can still permit publication

**Observed, reproduced.** In `vex_visuals/verifier.py:148`, missing frames go through `_unavailable_report()`. Its balanced-mode degraded decision depends on `local_gate_passed` and `strict`, regardless of the reason verification is unavailable.

`direct_rendered_visual()` with strong passing local QA, `strict=False`, no extracted frames, and zero repair rounds returned:

```json
{"passed":true,"publishable":true,"state":"degraded","frames":0,"issues":["visual_verifier_has_no_frames"]}
```

A provider outage and missing visual evidence require different publication rules. Zero-frame evidence must fail closed.

### F3: Shared verification can omit most of the animation

**Observed, reproduced.** `tools/auto_visuals.py:2474` returns `existing[:4]` when at least four saved QA frames exist. Remotion render QA samples numerous chronological frames starting at early fractions; the default list starts at `0.03, 0.08, 0.16, 0.26` and ends at `0.94`.

A ten-frame artifact probe selected frame 1 through frame 4 and omitted the final frame. Hyperframes can produce an adaptive plan with up to eight captures; the same prefix truncation applies there. Four frames can be a useful budget, but they must be selected for semantic coverage rather than position in a file list.

The extraction fallback also does not pass the asset's actual FPS into `extract_quality_frames()`, which defaults to 30. This can select the wrong frame index for media rendered at another frame rate.

### F4: A final Remotion render can inherit preview approval

**Observed control flow; merge behavior reproduced.** `tools/auto_visuals.py:2331` independently evaluates and selects a repaired preview. It then renders the finalist again at final fidelity and computes fresh local QA. The final render is not independently reverified before `_merge_visual_director_quality()` uses the original outcome.

At `tools/auto_visuals.py:2532`, the merged result takes `passed=outcome.passed`, and drops local issues when the old outcome passed. A probe merged failing final local QA containing `remotion_render_final_frame_is_visually_empty` with a previously verified outcome:

```json
{"final_local_passed":false,"merged_passed":true,"merged_issues":[]}
```

The final asset, final program, fresh frames, and final quality decision must remain bound together.

### F5: Hyperframes does not faithfully execute the shared authored program

**Observed implementation.** `vex_hyperframes/open_visual_runtime.py:151` extracts first/last track values, computes translation from endpoint differences, scales translation using fixed 1280/720 constants, and applies several properties through global `--route-progress`.

`vex_hyperframes/composer.py:2606` drives actors through generic rise/slide/pop entrance logic and shared timing. Intermediate keyframes, per-track easing, independent relation reveal tracks, arbitrary opacity trajectories, and static transform offsets are not faithfully represented by this lowering.

For example, a track that travels away and returns to its initial position has zero endpoint displacement. Endpoint-only translation cannot preserve that journey. This applies to the Open Visual Program path; it does not mean every legacy Hyperframes composition lacks animation.

### F6: The authored schema and execution capabilities disagree

**Observed; data mismatch reproduced.** The Open Visual Program schema forbids unknown element properties and does not declare `data` or `parent_id`. Meanwhile, `vex_visuals/scene_graph.py:590` reads both into runtime nodes.

Adding chart data to a valid fixture program, re-signing it, and validating produced:

```text
open_visual_schema:elements.0:additionalProperties
Additional properties are not allowed ('data' was unexpected)
```

The shared Remotion graph's `DataChart` reads `node.content.data`, but normal authored programs cannot supply that field through the packaged schema. `VectorIcon` and `VectorPathNode` use fixed checkmark/curve geometry. Parent IDs are validated in the graph, but `SceneGraphLayer` renders nodes as a flat list rather than a nested transform tree.

The Hyperframes Open Visual Program path handles chart bars with formula-generated heights, and does not render authored local image contents in `_element_markup()`. The richer legacy Visual World path has different behavior; its successful screenshot is not evidence that the shared open-program path supports those same capabilities.

### F7: Motion and layout semantics vary between implementations

**Observed implementation.** In `renderers/remotion_scene_graph.jsx:27`, `spring_snappy` and `spring_gentle` map to cubic/smoothstep formulas rather than physics-based spring evaluation. Static Python structural QA duplicates the JavaScript layout and easing logic.

`vex_remotion/structural_qa.py:605` estimates text width using character-count multipliers, while the renderer uses browser `fitText()`. Runtime telemetry attributes declare geometry and text probes, but `renderers/remotion_runner.mjs` does not collect a full per-frame measured DOM/text-bound report. Declared coordinates are useful intent, not actual glyph bounds.

### F8: Rendered candidate selection uses different quality objectives

**Observed control flow.** Remotion's still preflight at `renderers/remotion_renderer.py:361` ranks candidates using 68% pixel-aesthetic score plus static semantic/structural scores, at fixed fractions `0.08, 0.42, 0.82`. Hyperframes performs its own variant/critic/final-judge sequence.

The cross-renderer tournament at `tools/auto_visuals.py:3067` chooses one asset using local QA before the shared independent director sees it. If that asset fails independent semantic verification, an independently better discarded renderer contender is not automatically reconsidered by this tournament.

Already existing order-reversed pairwise judging should be preserved. The missing piece is a comparable evidence and eligibility policy before alternatives disappear.

### F9: Current visual references are descriptions

**Observed implementation.** `vex_visuals/concept_search.py` builds reference boards containing purpose, composition, transformation, focal element, camera, and fractions. `generative_authoring.py` sends their JSON in a text reasoning prompt. These are useful storyboard contracts, but they are not reference images or visual exemplars.

The visual designer cannot compare typography, material, proportion, and composition with an actual target from those descriptions alone.

### F10: Repair and counterfactual evidence need stronger attribution

**Observed implementation.** Shared repair planning groups failures into broad semantic/design/temporal/technical operations. Authoring retry includes validation errors but not the failed program itself. `_alternate_program()` selects untried candidates largely using concept difference and ID order.

In the shared director, repaired candidates enter the candidate pool before the monotonic-progress decision; final publication selection considers publication-ready candidates even if a repair did not advance the search. The intended acceptance policy should be explicit.

Hyperframes counterfactual ablation at `vex_hyperframes/inverse_decoder.py:426` masks a broad rectangle based on encoding family. It can remove unrelated text/artwork or miss the actual relation. Its score delta is useful heuristic evidence, not proof that a particular relation is necessary.

### F11: Full-video generation has a separate quality pipeline

**Observed execution paths.** `video_generation/pipeline.py:44` uses its own director, skill graph, cinematography, motion planning, portfolio judge, HTML project, and rendered cinematography evaluation. It does not call the shared `direct_rendered_visual()` verification/repair search for every generated beat.

Manual/directed paths also have eligibility differences. There should be explicit workflow profiles using one publication policy, with imported assets receiving checks appropriate to their mode.

### F12: Model policy, caching, and recovery are partially duplicated

**Observed implementation; cache alias reproduced.** Main chat uses `ProviderGateway`, but visual reasoning in `broll_intelligence.py:766` and vision critics instantiate provider SDKs directly. Concept/program authoring only enables model calls for Gemini and Claude. The shared verifier likewise configures Gemini/Claude endpoints; Hyperframes' older critics are Gemini-specific.

`config.build_gemini_generation_config()` globally requests zero thinking budget for Gemini-prefixed models. This is a uniform policy rather than a stage-specific, capability-aware choice. Its optimal value for Vex has not been benchmarked.

Planning already has call and wall-time budgets. These do not form one budget covering all nested variants, critics, counterfactual calls, repairs, pairwise calls, and final renders.

The verifier cache hashes the supplied contract signature rather than the complete validated contract/prompt. Changing contract content while retaining its signature produced an identical cache key. The existing contract validator is not enforced by the verifier entry points. `compare_visual_candidates()` explicitly discards `cache_dir`, despite documentation describing pairwise caching. `_write_cache()` uses a fixed `.tmp` name, which can collide under concurrent identical requests.

The execution ledger and job checkpoints already exist. `run_tool_job()` still calls the whole tool executor again on recovery; completed visual-stage outputs are not a resumable dependency graph. Browser renderers remain deliberately serial in Auto Visuals. That is understandable today, but a resource-aware worker scheduler can recover throughput without unconstrained concurrency.

### F13: Delivery can reduce the quality established upstream

**Observed implementation.** `engine.apply_visual_overlays()` converts the output to H.264/YUV420P and rounds FPS using `math.ceil()`. The rational edit graph currently models some source timing, but most legacy mutations become rendered anchors and are re-encoded.

`tools/composite_qa.py` samples one midpoint for each fullscreen replacement at 48×27 pixels, and skips visual-presence sampling for picture-in-picture/alpha overlays. These checks catch gross absence and metadata drift but cannot establish final text readability, edge timing, thin-line integrity, or correct alpha composition.

The existing Remotion media/output contract and atomic promotion helper should be reused and extended through delivery. The proposal is not to force every delivery format to use expensive lossless video; preserve a suitable master and validate the actual target export.

## Architecture and acceptance criteria

### 1. Claim-aware semantic verification

Represent critical claims as structured subject–predicate–object assertions with direction, polarity, units, quantities, and temporal dependencies. Keep the blind decoder independent of expected answers; align its decoded entities/relations to the evidence in a separate grader. Use lexical matching for aliases and candidate retrieval, not as sufficient evidence of entailment.

When structured alignment is ambiguous, a bounded entailment check should return supported, contradicted, or unknown. Unknown should trigger more evidence or a repair according to the workflow policy. Validate the communication contract and bind it to the source IR at the boundary.

Acceptance: all current valid paraphrases still pass; negated claims, reversed arrows, wrong units, swapped quantities, unsupported comparisons, and altered contracts fail. Use adversarial fixtures before changing the publication threshold.

### 2. Final artifact verification and evidence coverage

Introduce a verification receipt containing final asset hash, program hash, source contract hash, runtime version, dimensions/FPS, frame timestamps/hashes, verifier version, and publication state. A new render or material program change invalidates the previous receipt.

Choose captures for premise, each required proof relation, resolved outcome, and the final hold. Add adjacent frames around important transitions and short dense temporal windows for flicker/smoothness checks. A few sparse stills cannot establish continuous motion smoothness. Use actual media FPS, not an implicit 30 FPS assumption.

Acceptance: missing/corrupt frames never publish; final-hold coverage survives frame budgets; a failed final render cannot inherit preview approval; the receipt references the exact delivered asset. Balanced provider-outage fallback remains an explicit policy only when the required local and frame evidence exists.

### 3. Faithful motion execution

Define one pure motion evaluator: `state = evaluate(program, frame, fps)`. It must handle every keyframe, easing, property, initial offset, visibility interval, and relation reveal. Compile Hyperframes into a registered seekable timeline or a pure seek evaluator; keep Remotion driven by frame/time. Use actual canvas dimensions for normalized translation.

Support real springs through explicit stiffness/damping/mass parameters or deterministic sampled curves. Update QA to evaluate the same curves and sample their extrema. Distinguish spring behavior from polynomial easing rather than attaching spring names to unrelated curves.

Acceptance: three-keyframe return journeys, delayed independent relation reveals, nonzero initial offsets, opacity holds/exits, and out-of-order seeking match the authored states in both backends. Repeated seeking to the same frame produces the same result.

### 4. Expressive typed primitives and capability checks

Version the shared authored language incrementally. Add bounded vector geometry, named icon resources, grounded chart series with units and baseline policy, explicit mask/clip behavior, nested groups, and registry-backed image assets. Data values need source evidence bindings, not merely a chart-level object binding.

Publish an actual per-renderer capability manifest. Reject unsupported required semantics before rendering; allow only documented equivalent fallbacks. A renderer must not claim a backend/property and silently draw a generic substitute. Start with DOM/SVG; add specialized backends only when the benchmark demonstrates a benefit.

Acceptance: nonuniform chart data produces corresponding marks; negative values retain their sign; distinct icons and paths render distinctly; a local image displays its registered bytes; group transforms affect children; unsupported features are explicit errors or declared alternatives.

### 5. Browser-measured layout and typography

Package stable licensed font files and await font readiness. Collect actual element and text bounds, line counts, effective opacity, clipping, contrast against the visible background, and transformed relation endpoints at selected frames. Reuse a shared solved layout rather than keeping independently evolving Python and JavaScript solvers.

Prefer reflow, shorter grounded copy, a different composition, or another scene over shrinking required text below the reading floor. Evaluate typography at the final viewport and at realistic mobile viewing sizes.

Acceptance: long labels, unbroken technical terms, Unicode/multilingual copy, portrait formats, masked text, nested transforms, and moving connectors are verified through browser measurements and rendered evidence.

### 6. Actual visual reference packs

Add a project design pack with typography, spacing, palette, materials, reference images, and representative approved clips. Produce concrete keyframe sketches/contact sheets for a small number of viable concepts, and feed those images plus grounded scene intent to a multimodal designer.

Keep factual assets distinct from stylistic references. Reference imagery controls appearance; only source evidence authorizes factual claims. Start with locally supplied or curated licensed examples and rendered sketches, without requiring image generation for every scene.

Acceptance: reference and generated frames are directly inspectable; authoring actually receives image inputs; measurable layout/typography/style adherence improves in blinded comparison at a fixed budget.

### 7. Shared progressive candidate search

Use one sequence: semantic/static checks → low-cost stills → short motion previews → independent eligibility → calibrated pairwise selection → final render → final verification. Render only a bounded diverse shortlist; do not force the same number of attempts for every scene.

Expose contenders from both renderers to the shared selector. Retain original valid candidates and select for comprehension first, then design and motion, then cost. Preserve order reversal and record disagreement/ties. Store preview/final identities and selection provenance.

Acceptance: a misleading high-aesthetic candidate loses to a correct one; a failed local winner allows another verified renderer contender to win; simple scenes stop early; identical inputs reuse work; measured total cost stays within the run budget.

### 8. Attributed, targeted repair

Create typed counterexamples with frame/time, element/relation identity, observed defect, violated requirement, evidence crop/measurement, and allowed patch operations. Provide the failed program and relevant visual evidence to the repair author, keeping the independent final judge separate.

Check a repair's prerequisites, postconditions, and regressions in meaning, layout, and timing. Keep the best eligible candidate; stop on repeated program/evidence hashes, exhausted budget, or no measured improvement. Provider unavailability should invoke provider recovery rather than automatically modifying correct scene geometry.

Replace broad raster masking with controlled program-level ablation or exact relation masks from measured telemetry, and re-render at the same times. Treat redundant labels and geometry as potentially helpful: comprehension surviving one removed channel is not automatically a design failure.

Acceptance: one clipped label is fixed without moving the whole diagram; wrong motion direction is fixed without changing evidence; an ablation changes only its intended target; rejected/regressing repairs cannot silently win.

### 9. Narration and cross-scene continuity

Extend the existing narrative program/style bible with stable semantic object IDs, visual symbol assignments, absolute timing cues, narration phrase anchors, and readable hold duration. Keep palette/symbol identity consistent where meaning recurs, while permitting changes in composition and shot scale.

Use word timing confidence explicitly. On weak transcription, enlarge timing tolerance or use estimated cues with an honest label. Measure reading time from actual rendered copy and intended audience rather than requiring a fixed final fraction for every scene.

Acceptance: the same concept retains its visual identity across shots; important transformations land near their spoken phrase; viewers can read final states; scene diversity does not obscure causal continuity.

### 10. A rendered benchmark and evaluator calibration

Extend the existing semantic fixtures into separate rendered regression and quality suites. Include several domains, real creator briefs, all target aspect ratios, and deliberately bad outputs: reversed relations, tiny text, absent final states, near-static animation, fake charts, wrong quantities, alpha failures, and misleading polishing.

Measure comprehension, false acceptance, false rejection, text/layout correctness, temporal integrity, blind preference, total model/render cost, latency, and successful publication rate. Keep held-out cases and calibrate judges against human ratings. Report repeated trials and uncertainty for model-backed comparisons.

Add a small pinned browser/render integration set in CI, with larger quality runs on demand or scheduled separately. Current mocked/synthetic tests remain valuable regression checks.

Acceptance: each claimed upgrade wins a controlled before/after evaluation, preserves known working scenes, and does not improve style scores by weakening factual gates.

### 11. One generation harness for all workflows

Extract reusable planning, authoring, compilation, preview, verification, repair, selection, and delivery stages. Let Auto Visuals, full-video generation, directed visuals, and imports choose explicit profiles while sharing artifacts, failure types, budgets, and publication rules.

Imported content can use an import profile, and a manual request need not invoke concept search. Each profile must declare what was and was not verified.

Acceptance: the same failing scene is rejected consistently across entry points; shared quality improvements reach full-video beats; manifests use one evidence vocabulary; existing CLI and Studio behavior remains compatible.

### 12. Role-based model policy and complete cost accounting

Extend the gateway with text/vision/structured-output capability negotiation, model roles, usage accounting, deadlines, bounded retry policy, and circuit behavior. Assign planning, visual authoring, repair, and independent judging by measured task performance. Permit compatible local models when they actually support the required modality and schema.

Make reasoning effort configurable by role and supported model version. Measure whether additional reasoning improves scene design before making it the default. Avoid model-name-only assumptions.

Use one run budget for all nested model calls and render work. Cache validated results using canonical contract, prompt/policy versions, ordered evidence hashes, model, runtime/font/assets, and output settings. Implement pairwise caching, safe unique temporary writes, and concurrent single-flight work sharing.

Acceptance: equivalent requests reuse results; changed content invalidates cache even when a supplied signature is stale; no hidden pairwise/counterfactual spend escapes the budget; retry and capability behavior is consistent across entry points.

### 13. Durable stage execution and renderer workers

Build on the current SQLite execution ledger and checkpoints. Persist stage inputs, dependency hashes, status, outputs, attempts, quality receipts, resource reservation, and failure attribution. Resume only stages whose dependencies and outputs remain valid.

Use renderer worker processes with leases, heartbeats, deadlines, cooperative cancellation, child-process cleanup, and bounded CPU/RAM/browser slots. Keep project mutation exclusive even when independent rendering is concurrent. Do not introduce a distributed queue until local worker recovery is proven and needed.

Acceptance: terminate after compilation/render/QA and resume without repeating valid completed stages; orphaned workers are detected; cancellation stops child processes; project publication remains atomic; concurrency respects measured resource ceilings.

### 14. Delivery quality and source-based composition

Extend the existing rational edit graph to overlays/effects and compose from original sources plus approved assets where possible. Keep a suitable high-quality master; encode the delivery version according to its target. Make color space, alpha, scaling, sampling, and frame rate part of the delivery contract.

Verify the composite at entry, proof, hold, and exit times, including picture-in-picture and alpha modes. Check final readability at useful resolution, not just 48×27 similarity. Preserve source timing or use an explicit conversion policy rather than implicit FPS rounding.

Acceptance: 30000/1001 source timing survives supported paths; repeated edits do not repeatedly degrade the same master; exported text/lines remain legible; overlays occupy their promised location/time; audio and alpha composition remain correct.

### 15. Verified experience memory

Extend creative history with brief/evidence signature, renderer/runtime version, failure category, targeted patch, before/after measurements, verdict confidence, and human preference when available. Retrieve a small number of relevant examples by mechanism, layout, domain, and failure mode.

Separate observations from inferred lessons; expire incompatible versions and avoid suppressing valid opportunities because an older renderer failed. Memories influence the proposal, never bypass the new evidence gate.

Acceptance: a known recurring defect appears less often on held-out related briefs; retrieved lessons explain why they apply; context remains bounded; stale negative history does not block a now-supported visual.

### 16. Modular stages and typed contracts

The largest runtime function is `tools/auto_visuals.py::execute`, at 1,570 lines; it imports 30 internal modules. Extract stages with typed inputs/outputs and typed failures: evidence, authoring, capability, runtime, verification, budget, and delivery.

Retain the distinction between source facts, creative intent, executable scene, and observed evidence. Make the existing Open Visual Program the authored scene contract and the scene graph a derived execution artifact; do not let overlapping legacy representations each become a competing authority.

Add compatibility adapters and migrations before retiring old paths. Share promotion, frame extraction, artifact writing, and policy implementation across tools. Add context compaction to the main agent that preserves user preferences, active task state, evidence/artifact references, and complete tool-call/result pairs.

Acceptance: pipeline stages can be replayed/tested separately; renderer failures retain attribution; migration fixtures remain supported; long sessions retain task facts without sending an unbounded conversation and full tool catalog on every pass.

## Research basis and limits

These sources informed the architecture; the application to Vex is my engineering inference. Their reported gains are not Vex benchmark results.

1. [Harness Engineering: Anatomy, Architecture, and Evolution of Coding Agents](https://arxiv.org/html/2609.00006v1). A source-level comparison of production harness subsystems. Useful for explicit loop/context/tool/runtime boundaries and avoiding speculative framework layers. It is descriptive, not proof that a specific architecture will improve Vex.
2. [AI Harness Engineering: A Runtime Substrate for Foundation-Model Software Agents](https://arxiv.org/html/2605.13357v1). Emphasizes named resources, failure attribution, and requirement-linked verification evidence. Applied here as typed visual counterexamples, stage budgets, and final verification receipts; the paper's controlled software task is not a visual-quality benchmark.
3. [Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents). Supports explicit progress artifacts, incremental work, and verification across long sessions. Applied to render-stage recovery and bounded project memory.
4. [Harness design for long-running application development](https://www.anthropic.com/engineering/harness-design-long-running-apps). Describes distinct planning, generation, and evaluation roles. Applied as clear stage responsibilities and evaluator independence; Vex need not instantiate a separate agent for every responsibility.
5. [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents). Separates outcomes from agent claims, and discusses deterministic/model/human graders plus capability and regression suites. Applied to rendered Vex evaluation and calibrated judges.
6. [Design2Code](https://arxiv.org/html/2403.03163v2). Shows why rendered layout and element measurements matter; its revision experiments also caution against assuming self-feedback reliably improves all models. Applied to typography/layout probes and controlled visual comparisons, with additional temporal checks for video.
7. [UI2CodeN](https://arxiv.org/html/2511.08195v1). Studies generation/editing/polishing using rendered visual feedback. Applied to reference + current render + scene program inputs for targeted repair. Its results concern UI generation and trained models, not a direct guarantee for motion graphics.
8. [Self-Refine](https://arxiv.org/pdf/2303.17651). Motivates an explicit generation–feedback–revision cycle. Applied as bounded targeted revision with external evidence and regression checks rather than unverified self-praise.
9. [Reflexion](https://arxiv.org/html/2303.11366v4). Motivates compact experience memory based on task feedback. Applied to verified visual failure/repair examples, retaining its limitation that success depends on the feedback's reliability.

The runtime proposals also align with [Remotion spring documentation](https://www.remotion.dev/docs/spring), [Remotion text measurement](https://www.remotion.dev/docs/layout-utils/measure-text), [font loading guidance](https://www.remotion.dev/docs/google-fonts/), and Hyperframes' [determinism](https://hyperframes.app/docs/2-concepts/4-determinism) and [GSAP timeline guidance](https://hyperframes.app/docs/3-guides/3-gsap-animation). Compatibility must be verified against the repository's pinned Remotion `4.0.487` and Hyperframes `0.7.17`; current documentation can describe newer behavior.

## Proposed implementation sequence

1. **Establish a trustworthy baseline:** add the reproduced negative cases and a small rendered corpus; implement 1–2; isolate the four existing test failures. Preserve working paraphrase and renderer behavior.
2. **Make the scene contract executable:** implement 3–5 with cross-renderer conformance fixtures and real browser measurements. Add richer primitives behind an explicit version/capability contract.
3. **Raise creative quality:** implement 6–9 using the benchmark, matched budgets, targeted repair, and human-calibrated pairwise comparison.
4. **Unify execution and recovery:** extract stages as needed for 11–13 and 16; extend existing ledger, gateway, cache, and promotion rather than replacing them wholesale.
5. **Verify delivery and learning:** implement 14–15; run long-video, repeated-edit, crash-recovery, and held-out quality comparisons. Promote only demonstrated improvements.

Visual-quality optimization and platform refactoring should be separate, reviewable changes with observable acceptance criteria. First prioritize false approval and lost authored meaning; then measure beauty, timing, continuity, and cost.

## Baseline test failures

| Test | Observed cause in this environment | Interpretation |
|---|---|---|
| `test_hyperframes_command_does_not_fall_back_to_global_or_cwd_path` | The real managed Hyperframes runtime was resolved although the test expected no CLI | Test runtime discovery is not fully isolated; this failure alone does not prove a renderer defect |
| `test_managed_runtime_install_is_locked_verified_and_reused` | The real configured sibling `npm.cmd` was selected instead of the fake `/tools/npm` expected by the subprocess mock | Environment/configuration-sensitive runtime test |
| `test_managed_runtime_reinstalls_when_lock_digest_changes` | Same real npm selection mismatch | Environment/configuration-sensitive runtime test |
| `test_streaming_multipart_parser_writes_media_to_private_temp_file` | Windows reports directory mode bits differently from the expected POSIX `0700` | Platform-specific assertion; validate the intended Windows access behavior explicitly |

These failures preceded any changes and remain unfixed in this proposal-first pass. The semantic/publication defects were found by additional probes despite the existing visual tests passing. A green unit suite and a high heuristic visual score are therefore insufficient evidence for the requested quality goal.
