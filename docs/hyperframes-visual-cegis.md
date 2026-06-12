# HyperFrames Counterexample-Guided Visual Synthesis

## Problem

The previous HyperFrames architecture could select structurally different candidates
and reject weak blind decodes, but repair still operated on broad directives. It could
not point from a bad pixel region back to the exact object, relation, beat, evidence
binding, or layout instruction that caused the failure.

## Architecture

Automatic HyperFrames now behaves as a counterexample-guided visual compiler:

1. Evidence is compiled into Visual Explanation IR and a signed claim graph.
2. Proof candidates compile into signed Scene Program V2 programs.
3. Adaptive captures are taken at semantic beat resolution, required relation reveal,
   and final hold.
4. Blind, grounded, and design critics produce typed counterexamples.
5. A closed patch DSL changes only validated visual-program parameters.
6. The candidate is rerendered and compared with the previous attempt.
7. A repair survives only when improvement is monotonic.
8. A fresh final judge receives no repair history and decides release.

## Scene Program V2

Every automatic candidate contains stable:

- element, object, relation, beat, motion, and evidence IDs
- DOM selectors
- normalized geometry
- visibility intervals
- relation reveal timing and strength
- graph, semantic, and program signatures

The renderer writes `scene_program_v2.json` and `render_trace.json`, making every
sampled frame traceable to editable program instructions.

## Critics

The blind critic evaluates what can be understood without the intended answer. The
grounded critic checks relevance, source fidelity, unsupported implications, and real
interface evidence. The design critic checks hierarchy, density, typography, overflow,
collision, pacing, and professional finish.

All failures become `VisualCounterexample` records. Free-form critic prose is never an
editing interface.

## Repair Safety

The patch DSL permits bounded geometry, layout, hierarchy, timing, relation-strength,
source-binding, proof-encoding, and rerouting operations. It cannot add copy, facts,
metrics, entities, arbitrary HTML, scripts, remote URLs, or shell execution.

Before and after every patch, Vex verifies signatures, evidence preservation, required
object and relation coverage, relation endpoints, relation types, geometry, and timing.

## Promotion Policy

Vex rejects a repair when:

- hard failures increase
- object or relation coverage drops
- render quality materially drops
- a new failure class appears without resolving a harder failure
- the configured minimum improvement is not reached

Passing renderer QA is not sufficient. Structured critics and the independent final
judge must also pass. When every candidate fails, Vex drops or reroutes the visual
instead of selecting the least-bad output.

## Artifacts

Each candidate can now include:

- `scene_program_v2.json`
- `render_trace.json`
- `frame_contact_sheet.png`
- `blind_critic.json`
- `grounded_critic.json`
- `design_critic.json`
- `counterexamples.json`
- `patch_round_*.json`
- `before_after_*.json`
- `repair_history.json`
- `final_independent_verdict.json`

## Configuration

```env
HYPERFRAMES_ENABLE_CEGIS=true
HYPERFRAMES_MAX_CRITIC_FRAMES=8
HYPERFRAMES_MAX_REPAIR_ROUNDS=3
HYPERFRAMES_MIN_REPAIR_DELTA=0.025
```

Legacy decorative templates remain available for manual compatibility. Automatic
HyperFrames visuals must compile through semantic stages and Scene Program V2.
