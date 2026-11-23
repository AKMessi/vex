"""Prepare blinded video pairs and score independently collected ballots."""
from __future__ import annotations

import html
import json
import random
import re
import secrets
import tempfile
from pathlib import Path

import config
from engine import _run_command, probe_video
from vex_visuals.evidence import file_digest


VERSION = "vex-blind-review-v1"
IDENTITY = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}")


def _json(path: Path, limit: int = 1 << 20) -> dict:
    if not 0 < path.stat().st_size <= limit:
        raise ValueError(f"Choose a review JSON file up to {limit // 1024} KiB.")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("Review JSON must contain an object.")
    return value


def prepare_review(manifest: Path, output: Path, *, seed: int | None = None) -> dict:
    manifest = manifest.resolve(strict=True)
    inputs = _json(manifest).get("cases")
    if not isinstance(inputs, list) or not 1 <= len(inputs) <= 60:
        raise ValueError("Provide a manifest with 1 to 60 comparison cases.")
    randomizer = random.Random(seed if seed is not None else secrets.randbits(128))
    cases, seen = [], set()
    for row in inputs:
        if not isinstance(row, dict):
            raise ValueError("Each comparison case must contain an ID, category, baseline and candidate.")
        identity, category = row.get("id"), row.get("category")
        if (not isinstance(identity, str) or not IDENTITY.fullmatch(identity) or identity in seen
                or not isinstance(category, str) or not 1 <= len(category.strip()) <= 80):
            raise ValueError("Each comparison needs a unique case ID and a short category.")
        seen.add(identity)
        sources = {}
        for role in ("baseline", "candidate"):
            value = row.get(role)
            if not isinstance(value, str) or not value:
                raise ValueError("Each comparison needs a baseline and a candidate MP4.")
            path = (manifest.parent / value).resolve(strict=True)
            if not path.is_file() or path.suffix.lower() != ".mp4":
                raise ValueError("Review files must be MP4 videos.")
            media = probe_video(str(path))
            if media["codec"] != "h264" or media["duration_sec"] <= 0:
                raise ValueError("Use a playable H.264 MP4 to help reviewers open both clips.")
            sources[role] = path
        cases.append((identity, category.strip(), sources))
    randomizer.shuffle(cases)
    study = secrets.token_hex(16)
    output = output.expanduser().resolve()
    if output.exists():
        raise ValueError("Choose a new directory for the review packet.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".vex-review-", dir=output.parent) as directory:
        packet = Path(directory) / "packet"
        review = packet / "review"
        review.mkdir(parents=True)
        key = {"schema": VERSION, "study": study, "cases": {}}
        # Alternate which role appears first, with a random starting side.
        flip = randomizer.randrange(2)
        for index, (identity, category, sources) in enumerate(cases, 1):
            public_id = f"{index:03d}"
            first_role = "candidate" if (index + flip) % 2 else "baseline"
            second_role = "baseline" if first_role == "candidate" else "candidate"
            hashes = {}
            for side, role in zip("AB", (first_role, second_role), strict=True):
                target = review / f"{public_id}-{side}.mp4"
                _run_command([config.FFMPEG_PATH, "-nostdin", "-i", str(sources[role]), "-map", "0:v:0",
                              "-map", "0:a:0?", "-map_metadata", "-1", "-map_metadata:s", "-1", "-map_chapters", "-1",
                              "-c", "copy", "-movflags", "+faststart", str(target)], "Could not prepare the blind review clip")
                hashes[side] = file_digest(target)
            key["cases"][public_id] = {"id": identity, "category": category, "A": first_role, "B": second_role,
                                          "sha256": hashes}
        for layout, name in (("forward", "index.html"), ("mirror", "mirror.html")):
            (review / name).write_text(_review_page(study, layout, key["cases"]), encoding="utf-8")
        (packet / "study-key.json").write_text(json.dumps(key, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        packet.rename(output)
    return {"study": study, "cases": len(cases), "review": str(output / "review"), "key": str(output / "study-key.json")}


def _review_page(study: str, layout: str, cases: dict) -> str:
    blocks = []
    for identity, row in cases.items():
        media = []
        files = "AB" if layout == "forward" else "BA"
        for side, file_side in zip("AB", files, strict=True):
            media.append(f'<div><h3>Clip {side}</h3><video controls playsinline preload="metadata" '
                         f'aria-label="Case {identity} clip {side}" src="{identity}-{file_side}.mp4"></video></div>')
        options = "".join(f'<label><input type="radio" name="v{identity}" value="{vote}">{label}</label>'
                          for vote, label in (("A", "A"), ("B", "B"), ("tie", "Tie")))
        blocks.append(f'<section data-case="{identity}"><h2>Comparison {identity} · {html.escape(row["category"])}</h2>'
                      f'<div class="pair">{"".join(media)}</div><fieldset><legend>Overall preference</legend>{options}</fieldset>'
                      '<label>Why did you choose it? <textarea rows="2" maxlength="500"></textarea></label></section>')
    return f'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Video comparison</title>
<style>
:root {{color-scheme: dark; background:#12161d; color:#f6f7fa; font: 16px system-ui,sans-serif;}}
body {{max-width:72rem; padding:1rem; margin:auto;}}
section {{margin:2rem 0; padding:1rem; border:1px solid #566170; border-radius:.6rem;}}
.pair {{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,20rem),1fr));gap:1rem;}}
video,textarea {{width:100%;}} video {{max-height:65vh; background:#000;}}
fieldset {{margin:1rem 0;}} fieldset label {{display:inline-block; padding:.5rem 1rem .5rem 0;}}
button {{padding:.6rem 1rem;}} input {{margin-right:.4rem;}}
</style>
<main><h1>Video comparison</h1><p>Watch both clips before you vote. Consider clarity, pacing, and whether the video works for its intended audience.</p>
<form id="ballot"><label>Reviewer ID <input name="reviewer" required pattern="[a-zA-Z0-9][a-zA-Z0-9_.\\-]{{0,63}}" maxlength="64"></label>
{"".join(blocks)}<button type="submit">Download ballot</button> <output id="status" aria-live="polite"></output></form></main>
<script>
const form = document.querySelector('#ballot');
const status = document.querySelector('#status');
form.addEventListener('submit', event => {{
  event.preventDefault();
  const votes = [...form.querySelectorAll('section')].flatMap(section => {{
    const preference = section.querySelector('input:checked')?.value;
    return preference ? [{{case: section.dataset.case, preference, notes: section.querySelector('textarea').value.trim()}}] : [];
  }});
  if (!votes.length) {{status.textContent = 'Vote on at least one comparison.'; return;}}
  const reviewer = form.elements.reviewer.value;
  const ballot = {{schema: '{VERSION}', study: '{study}', layout: '{layout}', reviewer, votes}};
  const url = URL.createObjectURL(new Blob([JSON.stringify(ballot, null, 2)], {{type: 'application/json'}}));
  const link = document.createElement('a');
  link.href = url; link.download = `video-review-${{reviewer}}.json`; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  status.textContent = `Downloaded ${{votes.length}} vote(s).`;
}});
</script></html>
'''


def score_review(key_path: Path, ballot_paths: list[Path]) -> dict:
    key = _json(key_path)
    cases = key.get("cases")
    if (key.get("schema") != VERSION or not isinstance(key.get("study"), str) or not isinstance(cases, dict)
            or not 1 <= len(cases) <= 60 or any(not isinstance(row, dict) or not {"id", "category", "A", "B"} <= row.keys()
                                               for row in cases.values())):
        raise ValueError("Choose the study's private key file.")
    totals = {"baseline": 0, "candidate": 0, "tie": 0}
    scored = {identity: {"id": row["id"], "category": row["category"], "notes": [], **totals} for identity, row in cases.items()}
    seen, reviewers = set(), set()
    for path in ballot_paths:
        ballot = _json(path)
        reviewer, layout, votes = ballot.get("reviewer"), ballot.get("layout"), ballot.get("votes")
        if (ballot.get("schema") != VERSION or ballot.get("study") != key.get("study")
                or not isinstance(reviewer, str) or not IDENTITY.fullmatch(reviewer)
                or layout not in ("forward", "mirror") or not isinstance(votes, list) or not 1 <= len(votes) <= len(cases)):
            raise ValueError(f"The ballot {path.name} does not match the study.")
        reviewer = reviewer.casefold()
        reviewers.add(reviewer)
        for vote in votes:
            if not isinstance(vote, dict):
                raise ValueError(f"Invalid vote in {path.name}.")
            identity, preference = vote.get("case"), vote.get("preference")
            if not isinstance(identity, str) or identity not in cases or preference not in ("A", "B", "tie"):
                raise ValueError(f"Invalid comparison or vote in {path.name}.")
            pair = (reviewer, identity)
            if pair in seen:
                raise ValueError(f"Reviewer {reviewer} voted twice on comparison {identity}.")
            seen.add(pair)
            side = {"A": "B", "B": "A"}[preference] if layout == "mirror" and preference != "tie" else preference
            result = "tie" if side == "tie" else cases[identity][side]
            if result not in totals:
                raise ValueError("The study key contains an invalid role.")
            totals[result] += 1
            scored[identity][result] += 1
            notes = vote.get("notes", "")
            if not isinstance(notes, str) or len(notes) > 500:
                raise ValueError(f"Invalid notes in {path.name}.")
            if notes.strip():
                scored[identity]["notes"].append({"preference": result, "text": notes.strip()})
    decisive = totals["candidate"] + totals["baseline"]
    return {"schema": VERSION, "study": key.get("study"), "reviewers": len(reviewers), "votes": len(seen),
            **totals, "candidate_share_of_decisive_votes": totals["candidate"] / decisive if decisive else None,
            "cases": list(scored.values())}
