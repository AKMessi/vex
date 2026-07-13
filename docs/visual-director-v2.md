# Visual Director v2

Visual Director v2 is the renderer-neutral publication boundary for automatic
HyperFrames and Remotion visuals. It gives the reasoning model broad control over
visual language while keeping evidence, execution, and publication inside typed,
auditable contracts.

## Runtime Pipeline

1. `VisualExplanationIR` supplies the only allowed facts, objects, relations,
   labels, exact numbers, and forbidden claims.
2. The communication compiler turns that evidence into signed atomic
   propositions, temporal dependencies, a concise thesis and takeaway, and blind
   viewer questions.
3. Concept search creates six materially different lanes: physical
   transformation, data explanation, spatial metaphor, editorial kinetic,
   source-grounded collage, and dimensional system.
4. Each concept receives a four-frame reference board at premise, mechanism,
   proof, and final-hold moments. The selected concept is compiled into a signed
   Open Visual Program; deterministic programs remain available when model
   authoring fails.
5. The renderer produces the actual video. The adapter extracts four reference-
   aligned frames from renderer artifacts or FFmpeg.
6. A blind vision critic sees the frames and viewer questions, but not the
   expected answers, transcript, concept brief, or scene graph. It scores semantic
   recovery, hierarchy and design, causal motion, and technical integrity.
7. A publishable candidate can trigger one bounded alternate-concept render.
   Verified alternatives enter an order-reversed pairwise tournament so a judge
   must express the same preference when A/B positions are swapped.
8. Failed candidates produce typed repair operations. Vex promotes an untried
   concept or patches copy, hierarchy, motion causality, final hold, or layout;
   re-signs the program; re-renders it; and advances only monotonic improvements.
9. Final timeline QA evaluates the whole portfolio for repeated concept lane,
   medium, motion grammar, composition, palette, renderer, and perceptual output.

## Publication States

| State | Meaning | Publication |
| --- | --- | --- |
| `verified` | Independent frame evidence passes semantic, design, temporal, and technical gates | Yes |
| `degraded` | Independent verification is unavailable, but balanced mode has strong local evidence and no hard defect | Yes, explicitly labeled |
| `unverified` | The verifier is unavailable in strict mode or local fallback evidence is insufficient | No |
| `rejected` | Evidence recovery, unsupported-claim, design, motion, or technical gates failed | No |

Unsupported claims, blank output, clipped or illegible required text, collisions,
missing final state, and semantic-program failures remain hard local defects. A
verified multimodal report may override a soft heuristic such as a pixel-motion
threshold, but never a hard defect. Balanced outage fallback also requires a
minimum local score, so provider failure is not a general QA bypass.

The blind decoder is evaluated as one evidence bundle rather than isolated
answers. Object-specific questions disambiguate inputs, intermediate states,
and results; semantic recovery may use the decoded thesis and adjacent
chronological frames, but exact quantities and proposition dependencies still
gate publication. This avoids false rejection when one short answer is vague
without weakening numeric or causal correctness.

## Reliability And Cost Bounds

- Gemini and Claude vision endpoints are tried independently with bounded retries.
- A provider/model circuit opens after repeated failures and closes after a
  configured cooldown.
- Blind and pairwise responses use content-addressed caches keyed by frame bytes,
  contract signature, provider, model, operation, and verifier version.
- Full-render search defaults to two candidates and is capped at three.
- Repair defaults to two rounds and is capped at four.
- Provider outages skip alternate search and pairwise spend.
- Strict renderer requests remain renderer-locked during repair.

## Artifacts And Telemetry

Every selected renderer job writes `visual_director_report.json` and records its
path in `RenderedAsset.artifact_paths`. Auto Visuals manifests retain candidate
states, blind answers, dimension scores, typed repair history, pairwise comparisons,
selected creative identity, portfolio metrics, and all explicit degradation or
rejection reasons.

The project-local creative registry receives renderer, intent, QA score,
publication status, verifier state and score, repair count, concept lane, concept
medium, and motion grammar. These signals can adjust conservative future priors;
they cannot bypass evidence or publication hard gates.

## Configuration

| Variable | Default | Bound |
| --- | ---: | ---: |
| `VISUAL_DIRECTOR_ENABLED` | `true` | Boolean |
| `VISUAL_DIRECTOR_VERIFICATION_MODE` | `balanced` | `strict`, `balanced`, `off` |
| `VISUAL_DIRECTOR_RENDER_CANDIDATES` | `2` | 1-3 |
| `VISUAL_DIRECTOR_MAX_REPAIR_ROUNDS` | `2` | 0-4 |
| `VISUAL_DIRECTOR_MIN_REPAIR_DELTA` | `0.025` | 0-0.25 |
| `VISUAL_DIRECTOR_PAIRWISE_TOP_K` | `3` | 1-4 |
| `VISUAL_DIRECTOR_VERIFIER_RETRIES` | `2` | 1-3 |
| `VISUAL_DIRECTOR_CIRCUIT_FAILURES` | `2` | 1-8 |
| `VISUAL_DIRECTOR_CIRCUIT_COOLDOWN_SEC` | `45` | 1-600 |

`VISUAL_DIRECTOR_VISION_MODEL` and
`VISUAL_DIRECTOR_CLAUDE_VISION_MODEL` override the configured Gemini and Claude
vision models. When left empty, Vex uses the existing provider model settings.

## Source Boundaries

- `vex_visuals/communication_contract.py`: signed meaning contract and semantic scoring
- `vex_visuals/concept_search.py`: concept lanes, model normalization, reference boards
- `vex_visuals/verifier.py`: blind VQA, provider reliability, cache, pairwise tournament
- `vex_visuals/repair.py`: typed repairs and monotonic improvement assessment
- `vex_visuals/director.py`: renderer-neutral test-time search and publication policy
- `vex_visuals/portfolio.py`: set-level creative identity and diversity report
- `tools/auto_visuals.py`: renderer adapters, primary/reserve integration, manifests
