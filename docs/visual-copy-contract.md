# Evidence-led visual copy

Auto Visuals separates source evidence, semantic claims, visible copy, and rendering geometry.
Renderers must not derive on-screen text directly from transcript windows or clip text to fit a
layout.

## Contract flow

1. `visual_intelligence.py` extracts candidate evidence and excludes numeric identifiers such as
   model versions from measured facts.
2. `visual_explanation.py` builds facts, objects, relations, and deterministic executable models.
3. `visual_copy_contract.py` authorizes complete display phrases against a specific fact or object.
   The signed contract records the phrase, binding, evidence IDs, grounding class, confidence, and
   whether the phrase is required. Consumers verify the signature and fail closed after mutation.
4. Remotion and Hyperframes consume the same approved phrases. A phrase is kept whole or omitted;
   renderers cannot publish a first-N-word fragment.
5. Open Visual Program validation checks copy against its local semantic binding. Global transcript
   token overlap is not sufficient.

The LLM remains free to author geometry, visual metaphor, composition, and semantic motion. It may
also propose concise copy, but that copy must survive the same claim-local evidence contract before
it can render.

## Failure policy

The planner rejects an opportunity before rendering when required copy is dangling, deictic,
generic, low-signal, unrelated to the current evidence window, or missing a claim binding. It then
uses the existing reserve-opportunity path instead of publishing a semantically weak visual.
