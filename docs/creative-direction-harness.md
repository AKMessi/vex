# Creative Direction Harness

HyperFrames and Remotion share one deterministic creative-direction layer. The
renderer is an execution backend; it does not independently decide what visual
medium, hierarchy, palette, or motion language an explanation should use.

## Direction Program

`vex_visuals.creative_direction` compiles transcript-grounded semantic objects
and relations into a signed `CreativeDirectionProgram` containing:

- semantic medium family: data sculpture, diagrammatic system, editorial
  collage, kinetic typography, product interface, source media composite, or
  spatial metaphor
- orientation-specific composition grammar, focal point, reading path,
  density, negative-space target, panel budget, layer count, and safe margin
- focal and supporting object IDs plus required relation IDs
- palette roles with enforced foreground contrast, typography system, shape
  language, material, depth model, motif, texture, and corner policy
- establish, reveal, relate, resolve, and readable-hold choreography phases
- hard quality limits for text and graphic contrast, hierarchy, balance, edge
  intrusion, palette vitality, global motion, and color-only encoding

The compiler uses semantic fit, orientation, source availability, proof-world
alignment, variant ordinal, and recent visual history. The resulting signature
prevents renderers or agents from silently changing the direction after
compilation.

## Renderer Adapters

HyperFrames compiles a direction for every proof candidate. Its existing visual
world remains responsible for detailed medium geometry, while the direction
adds a validated palette, registration motif, choreography contract, metadata,
and aesthetic gate.

Remotion embeds the direction in its signed scene program and renders it
directly. Data scenes gain orbital evidence fields and measured progress;
contrast scenes use editorial material and asymmetry; mechanisms gain spatial
tracks and depth; quotes use a dedicated kinetic type lockup; interface scenes
retain product surfaces. Grounded constraints render as annotation rails rather
than fake process nodes.

## Shared Aesthetic Critic

`vex_visuals.aesthetic_critic` evaluates representative rendered frames, not
template names or self-reported quality. It measures:

- saliency hierarchy and focal balance against the compiled target
- content occupancy and negative space against the medium contract
- edge intrusion while excluding intentional outer registration rails
- luminance separation and local variation as a depth signal
- palette vitality and dominant color-role count
- motion magnitude, simultaneity, and whether the final state settles

The report is persisted beside renderer artifacts. HyperFrames blends it into
variant ranking; Remotion blends it into render QA. Hard critic failures block
promotion in both paths.

## Motion Principles

The choreography contract establishes stable context first, staggers evidence,
uses motion to explain relations, resolves onto the focal outcome, and reserves
the final 20 percent for reading. Remotion animations remain entirely driven by
the current frame. Nonessential simultaneous full-scene motion and color-only
semantic encoding are forbidden.

## Validation

The harness is covered by deterministic signature, semantic medium, palette
continuity, tamper rejection, synthetic frame-critic, golden semantic corpus,
packaging, and renderer integration tests. Production validation also renders
landscape, square, and portrait Remotion media plus a complete HyperFrames proof
candidate with shared aesthetic scoring.
