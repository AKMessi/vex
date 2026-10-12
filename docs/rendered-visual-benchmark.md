# Rendered visual benchmark

The fixed corpus covers quantities, direction, negation, grounded charts, long text, and an intermediate return keyframe across landscape, portrait, and square. Each run creates actual Remotion/Hyperframes video assets, chronological decoded frames, local quality evidence, timing/usage records, content hashes, `report.json`, and `gallery.html`.

```powershell
python -m vex_visuals.benchmark --output .analysis/benchmark-current
python -m vex_visuals.benchmark --output .analysis/benchmark-smoke --cases compression_landscape routing_portrait
```

Rendering is offline: designer/critic API calls are disabled through an explicit policy propagated to isolated workers. The normal compiler, renderer and local gates remain active. A failed case exits nonzero and stays visible in the report; the suite never silently relaxes quality thresholds to produce a green run. `.github/workflows/visual-render.yml` installs the pinned runtime and executes the small rendered set. Larger runs are local/on demand.

For native semantic evaluation, add `--vision`. This uses the configured keys and consumes provider quota. Offline outputs deliberately retain `semantic_state=unverified`. Local QA scores are heuristics; they do not establish comprehension or human preference. Usage records count calls/tokens/renders, not a fabricated dollar price. Render stages can reuse valid outputs, so cached latency must not be treated as cold-render speed.

To compare two implementations, run the same command in each checkout and keep both reports:

```powershell
python -m vex_visuals.benchmark --output .analysis/benchmark-after --compare path/to/before/report.json
```

Comparisons require the same corpus digest, cases, renderers, canvas dimensions, and vision policy. `comparison.json` reports technical regressions and measured score/latency deltas. It explicitly does not convert those deltas into human quality gains. Corpus changes require fresh baselines.

For independent review, inspect the gallery without showing reviewers the implementation label or local scores. Ask comprehension questions from the source contract, then rate comprehension and visual preference separately. Use multiple reviewers and held-out real briefs before making broad quality claims. Save genuine ratings as a JSON array:

```json
[{"key":"compression_landscape:remotion","reviewer":"reviewer-1","comprehension":4,"visual_preference":4}]
```

```powershell
python -m vex_visuals.benchmark --output .analysis/benchmark-current --ratings ratings.json
```

This writes `calibrated_report.json` without rerendering. Duplicate reviewer/case pairs, unknown cases and invalid ratings are rejected. The descriptive false-acceptance statistic counts vision-verified scenes rated below 3/5 for comprehension. Ratings of the same scene are correlated; these summaries are not confidence intervals or proof of improvement. No human ratings ship with the corpus.

The adversarial semantic, evidence, motion, grounding and delivery regression tests complement this rendered corpus. Reversed roles, negated claims, absent final frames, missing evidence, fake chart values and rejected repairs must remain rejected even when the associated visual looks polished.
