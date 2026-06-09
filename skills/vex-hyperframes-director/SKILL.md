---
name: vex-hyperframes-director
description: Direct, implement, debug, or review Vex HyperFrames explainer visuals using evidence-backed semantic contracts, curated blueprints, deterministic HTML motion, and semantic QA. Use for Auto Visuals HyperFrames quality work, new semantic scene families, storyboard or production-contract changes, generated HTML safety, renderer failures, irrelevant or fabricated visuals, template replacement, and HyperFrames skill-pack improvements.
---

# Vex HyperFrames Director

Build explanations, not themed slides. Treat transcript evidence as the truth boundary and reject unsupported visuals before HTML generation.

## Workflow

1. Inspect `visual_explanation.py`, `vex_hyperframes/compiler.py`, and the relevant caller before changing renderer code.
2. Build or validate Visual Explanation IR. Require evidence-backed facts, typed objects, semantic beats, required labels, and explicit rejection reasons.
3. Select a curated blueprint by scene type and grounded role prerequisites. Do not select by aesthetic keywords alone.
4. Build the storyboard and production contract before composition. Require a screenshot test and semantic signature.
5. Compile only contract-backed content into deterministic, self-contained HyperFrames HTML.
6. Validate HTML timing and safety, render multiple time samples, then run semantic QA before visual-quality QA.
7. Repair only bounded, diagnosable failures. Reject or reroute when evidence or required roles are missing.
8. Add golden fixtures and regression tests for every new explanation pattern.

## Hard Rules

- Never invent metrics, thresholds, interface states, entities, risks, steps, or outcomes.
- Never silently fall back from an unsupported semantic request to a quote or generic card layout.
- Never let visual polish override a failed grounding, label, object, beat, or screenshot check.
- Keep all motion seekable through `window.__timelines[compositionId]`.
- Keep generated compositions local-only and free of remote URLs, arbitrary scripts, and shell execution.
- Preserve object identity when a state transforms, a token moves, or ownership changes.
- Keep legacy templates backward-compatible, but route automatic generation through semantic blueprints.

## References

- Read [contracts.md](references/contracts.md) when changing IR, storyboard, production contracts, or rejection behavior.
- Read [blueprints.md](references/blueprints.md) when adding or selecting scene types and blueprint variants.
- Read [motion-and-layout.md](references/motion-and-layout.md) when implementing HTML/CSS/timeline behavior.
- Read [qa-and-repair.md](references/qa-and-repair.md) when changing frame inspection, semantic QA, retries, or rerouting.
- Read [examples.md](references/examples.md) for expected contract shapes and user-request mappings.

## Verification

Run focused semantic/compiler/composer tests first, then renderer and Auto Visuals tests. Finish with the complete pytest suite when the change affects planning, rendering, QA, or compositing.
