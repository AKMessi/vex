# Motion And Layout

## Deterministic Timing

- Register one timeline at `window.__timelines[compositionId]`.
- Implement `time`, `seek`, and `progress` from the requested render time.
- Avoid `requestAnimationFrame`, `setInterval`, and uncontrolled CSS animation clocks.
- Give every visible timed element explicit clip timing and a non-overlapping track.

## Semantic Motion

Use motion to expose:

- state registration and comparison
- signal propagation
- mechanism activation
- intervention
- route progression
- ownership handoff
- branch selection
- evidence accumulation
- resolved-state hold

Decorative motion may support these events but cannot replace them.

## Object Continuity

Keep the same object identity when:

- before transforms into after
- a request crosses services
- a process token changes owner
- a story subject moves through failure and recovery
- a metric changes after intervention

## Layout

- Use one dominant explanatory spine.
- Keep headline treatment subordinate to the explanation.
- Reserve the subtitle-safe zone.
- Limit simultaneous conceptual objects.
- Use compact grounded labels, not transcript paragraphs.
- Make the resolved frame readable as a still.
- Keep text inside stable responsive bounds at landscape, square, and vertical targets.

## Structural Encoding Families

- `linear_trace`: one directional explanatory route.
- `split_register`: matched states, branches, or ownership zones.
- `layered_flow`: responsibilities or evidence accumulated by depth.
- `focal_gate`: one mechanism, intervention, decision, or source focus controls the result.
- `radial_evidence`: grounded support converges on one central claim.

Encoding families must change spatial reasoning and motion grammar, not merely colors,
fonts, or decorative treatment.

## Interface Scenes

Render only source-backed controls and states. Do not add progress percentages, fake status rows, notifications, logs, or dashboards to make the frame feel complete.
