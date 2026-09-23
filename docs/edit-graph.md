# Edit graph and NLE handoff (v1)

Vex now records a versioned `edit_graph` in each project revision. It describes
the relationship between output time and source time using exact rational
numbers. A graph contains source media references and ordered, contiguous clip
spans; each span has a source range and an output range. The source copy is
retained in the project working directory on import.

`trim_clip` composes its range over the current graph. A successful trim keeps
source lineage only when the actual rendered duration is within the validation
tolerance (0.1 seconds or two frames, whichever is greater). Unsupported
operations, or trims whose output timing does not match, become an explicit
`rendered_anchor` pointing at the promoted media. This keeps the graph truthful
without claiming that effects, transitions, audio mixes, or semantic edits can
yet be reconstructed from original sources. Legacy tools that save through
`ProjectState.apply_operation` also become rendered anchors. Older projects with no graph remain
supported and are exported as `flattened` timelines.

Undo/redo and timeline rebuild re-derive source mapping for trim-only histories.
Other rebuilt histories are anchored to the actual render. The graph is saved
with the same catalog revision as a promoted output's asset and cache record.
Studio displays real span lengths and their provenance; it does not pretend
legacy operation history represents individual clips.

NLE timeline JSON is schema 2 and includes the graph and a `handoff_mode` of
`source`, `rendered_anchor`, or `flattened`. FCPXML and EDL exports emit actual
cut entries for graph spans, with rational FCPXML timing. EDL necessarily rounds
to frame timecodes. A retimed span is rejected rather than exported
incorrectly. These exports are handoff formats, not effect round-trips: text,
color, transitions, audio, and other rendered changes are represented by the
rendered anchor, while legacy operation descriptions remain informational.

## Next architecture step

The graph is currently a non-destructive timing model, not a complete render
compiler. The next phase should introduce typed video/audio/effect nodes,
content-addressed source retention, a deterministic graph compiler and proxy
path, and frame/colour/audio validation against exports. EDL's single-track
limits and variable-frame-rate source handling also need explicit policies
before claiming full NLE interchange.
