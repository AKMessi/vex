---
name: vex-hyperframes-visual-repair
description: Diagnose and repair Vex HyperFrames renders with Scene Program V2, adaptive frame capture, blind/grounded/design critics, typed visual counterexamples, evidence-preserving patch operations, monotonic repair acceptance, and independent final judging. Use when a HyperFrames visual is irrelevant, generic, unreadable, semantically wrong, visually weak, or failing QA.
---

# Vex HyperFrames Visual Repair

Repair the executable visual program, not the rendered pixels and not free-form HTML.

## Workflow

1. Read `scene_program_v2.json`, `render_trace.json`, and `frame_contact_sheet.png`.
2. Compare `blind_critic.json`, `grounded_critic.json`, and `design_critic.json`.
3. Convert failures into typed counterexamples with exact element IDs, relation IDs, frame IDs, evidence IDs, and allowed repairs.
4. Generate a signed `VisualPatchSet` against the current program signature.
5. Apply only the closed patch DSL and validate the complete scene program.
6. Rerender and compare before/after snapshots.
7. Keep the repair only when it is monotonic: fewer hard failures or a sufficient score gain with no quality, coverage, or issue regression.
8. Run the independent final judge without repair history.

## Hard Rules

- Never invent copy, facts, metrics, entities, interface states, or assets.
- Never rewrite arbitrary HTML, CSS, JavaScript, or shell commands.
- Never change claim-graph endpoints, relation types, evidence IDs, graph signature, or semantic signature.
- Never remove required objects or relations.
- Never publish a failed candidate because it is the best available candidate.
- Reroute to source footage, Manim, Blender, or no visual when the signed contract cannot be satisfied by HyperFrames.

## References

- Read [counterexamples.md](references/counterexamples.md) for critic output requirements.
- Read [patch-dsl.md](references/patch-dsl.md) for allowed operations and invariants.
- Read [release-gate.md](references/release-gate.md) for monotonic acceptance and final judging.
