import json
import shutil
import subprocess
from pathlib import Path

import pytest

from vex_visuals.blind_review import VERSION, prepare_review, score_review


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="Requires FFmpeg fixtures")
def test_packet_hides_source_names_and_scores_mirrored_ballots(tmp_path):
    files = [tmp_path / filename for filename in ("original-branch.mp4", "new-branch.mp4")]
    for path, color in zip(files, ("blue", "green"), strict=True):
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"color={color}:s=320x180:d=0.3",
                        "-metadata", "title=SECRET-IDENTITY", "-c:v", "libx264", str(path)], check=True, timeout=15)
    manifest = tmp_path / "pairs.json"
    manifest.write_text(json.dumps({"cases": [{"id": "secret-case-name", "category": "Motion <test>",
                                              "baseline": files[0].name, "candidate": files[1].name}]}), encoding="utf-8")
    result = prepare_review(manifest, tmp_path / "study", seed=7)
    review = Path(result["review"])
    key = Path(result["key"])
    identity = json.loads(key.read_text(encoding="utf-8"))["cases"]["001"]
    assert {path.name for path in review.iterdir()} == {"001-A.mp4", "001-B.mp4", "index.html", "mirror.html"}
    assert b"SECRET-IDENTITY" not in (review / "001-A.mp4").read_bytes()
    for path in review.glob("*.html"):
        page = path.read_text(encoding="utf-8")
        assert "Motion &lt;test&gt;" in page
        assert all(word not in page for word in ("original-branch", "secret-case-name", "study-key.json"))
    candidate_side = "A" if identity["A"] == "candidate" else "B"
    ballot_paths = []
    for reviewer, layout, preference in (("a", "forward", candidate_side), ("b", "mirror", candidate_side),
                                          ("c", "forward", "tie")):
        ballot = tmp_path / f"{reviewer}.json"
        ballot.write_text(json.dumps({"schema": VERSION, "study": result["study"], "reviewer": reviewer,
                                       "layout": layout, "votes": [{"case": "001", "preference": preference,
                                                                     "notes": "Test fixture"}]}), encoding="utf-8")
        ballot_paths.append(ballot)
    report = score_review(key, ballot_paths)
    assert (report["candidate"], report["baseline"], report["tie"], report["reviewers"]) == (1, 1, 1, 3)
    assert report["candidate_share_of_decisive_votes"] == 0.5
    assert len(report["cases"][0]["notes"]) == 3
    assert score_review(key, [ballot_paths[2]])["candidate_share_of_decisive_votes"] is None
    with pytest.raises(ValueError, match="voted twice"):
        score_review(key, [*ballot_paths, ballot_paths[0]])
