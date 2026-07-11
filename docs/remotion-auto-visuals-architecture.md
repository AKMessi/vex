# Remotion Auto Visuals Architecture

Vex treats Remotion as a semantic renderer, not a free-form JSX generator. The
agent proposes an opportunity, but deterministic code decides whether the
evidence can support a visual and exactly what the composition must communicate.

## Pipeline

1. Auto Visuals identifies transcript windows and reserve opportunities.
2. `visual_explanation.py` extracts grounded facts, objects, relations, beats,
   required labels, forbidden content, and an explicit render or reject policy.
3. `vex_remotion.compiler` compiles that evidence into a typed
   `RemotionSceneProgram`.
4. Hard constraints remove scene families that cannot be supported. Soft
   constraints rank the remaining metric, mechanism, contrast, timeline,
   interface, and emphasis layouts.
5. The compiler selects an orientation-specific layout, removes redundant
   summary nodes, creates a quality contract, and signs the canonical program.
6. Auto Visuals rejects invalid primaries before Chromium starts and promotes a
   non-overlapping compiled reserve when available.
7. `renderers/remotion_entry.jsx` renders only the signed program. It uses
   `calculateMetadata`, frame-time animation, measured text fitting, explicit
   safe areas, and a final hold.
8. `vex_remotion.qa` extracts early, middle, and final frames from the rendered
   video and gates publication on contrast, occupancy, motion, information
   density, and the compiler's semantic score.

## Trust Boundaries

- Transcript-grounded programs fail when the explanation IR says reject.
- Numeric facts must have source provenance; raw agent metrics cannot override
  this rule.
- Generic labels and structurally unsupported scene families fail compilation.
- React receives a bounded data program, never agent-authored executable JSX.
- The renderer records the program, compiler report, render request, render log,
  sampled QA frames, QA report, and output metadata in the render job directory.

## Scene Program

The signed scene program contains:

- source-grounded title and takeaway
- typed nodes with roles, details, metric values, emphasis, and fact IDs
- relation edges with provenance and required status
- timed semantic beats expressed as duration fractions
- orientation, layout variant, density, safe margin, and measured title lines
- required visible labels, required edges, forbidden content, and visual QA
  thresholds
- evidence records, grounding mode, semantic score, and compiler warnings

The signature makes program identity stable across planning, rendering, QA, and
manifest inspection.

## Responsive Composition

Landscape, square, and portrait are distinct layout modes. They do not scale a
single fixed 1280x720 canvas into every target. Each mode owns content widths,
card geometry, flow direction, title fitting, and footer safe areas. Text is
measured with `@remotion/layout-utils`; animation derives only from Remotion's
current frame and video configuration.

## Validation

The regression suite covers:

- all renderable and rejected cases in the semantic golden corpus
- invented metric rejection and metric fallback behavior
- signed relation-preserving programs
- portrait layout selection and summary-node removal
- primary rejection with compiled reserve substitution
- packaged wheel and source distribution contents
- synthetic animated and blank-frame QA behavior
- real landscape, portrait, and square renders across mechanism, metric,
  contrast, and interface families

Rendered artifacts are accepted only when both semantic compilation and actual
frame QA pass.
