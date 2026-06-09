# Examples

## Metric Intervention

Request: "Show latency falling from 420 ms to 180 ms after caching."

Compile:

- scene type: `metric_intervention`
- objects: `420 ms`, `Enable caching`, `180 ms`
- blueprint: intervention trace or grounded threshold variant
- hard rule: do not add another percentage, target, or trend

## Grounded Interface

Request: "Show the failed shot, open its render log, then retry that shot."

Compile:

- scene type: `grounded_interface_walkthrough`
- objects: failed shot, render log, retry action
- prefer real captured UI when available
- hard rule: do not invent status percentages or generic workflow rows

## Decision Gate

Request: "If transcript confidence is low, request review; otherwise continue rendering."

Compile:

- scene type: `decision_branch`
- condition: transcript confidence
- low branch: request review
- high branch: continue rendering
- activate branches sequentially, never simultaneously

## Unsupported Idea

Request: "Show some risks we should think about."

Reject because no risk entities or evidence are named. Do not create Risk 1, Risk 2, or Risk 3.
