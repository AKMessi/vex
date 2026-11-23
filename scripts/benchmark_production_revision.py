"""Compare one caption edit against a complete rebuild of the same composition."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

import config
from video_generation.production_timeline import cue_id
from video_generation.remotion_revisions import _composition_dir, _read_object, revise_remotion_project


def frame_rgb(path: Path, number: int) -> np.ndarray:
    raw = subprocess.check_output([
        config.FFMPEG_PATH, "-nostdin", "-v", "error", "-i", str(path),
        "-vf", f"select=eq(n\\,{number}),scale=640:-2", "-frames:v", "1",
        "-pix_fmt", "rgb24", "-f", "rawvideo", "-",
    ], timeout=120)
    if not raw:
        raise ValueError(f"Output did not contain frame {number}.")
    return np.frombuffer(raw, dtype=np.uint8).astype(np.int16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path, help="Verified Remotion composition or project folder")
    parser.add_argument("output", type=Path, help="Folder for benchmark output")
    parser.add_argument("--caption", type=int, default=0, help="Zero-based caption index")
    parser.add_argument("--workers", default="2")
    options = parser.parse_args()
    config.reload_settings()
    source = _composition_dir(options.project)
    timeline = _read_object(source / "input_props.json")["timeline"]
    cue = timeline["captions"][options.caption]
    edits = [{"op": "caption", "id": cue_id(cue), "text": "Revised caption for render comparison."}]
    results = {}
    videos = {}
    for name, full in (("incremental", False), ("full", True)):
        start = time.perf_counter()
        video, manifest, receipt = revise_remotion_project(
            source, replacements={}, edits=edits, output_dir=options.output / name,
            workers=options.workers, full_render=full,
        )
        if not receipt["passed"]:
            raise RuntimeError(f"Delivery did not pass. See {manifest}")
        results[name] = {"elapsed_seconds": round(time.perf_counter() - start, 3),
                         "rendering": receipt["rendering"], "output": str(video)}
        videos[name] = video
    if results["incremental"]["rendering"]["mode"] != "shot_ranges":
        raise RuntimeError(f"Input did not qualify for frame reuse: {results['incremental']['rendering']}")
    left, right = cue["start_frame"], cue["start_frame"] + cue["duration_frames"]
    frames = {left, (left + right) // 2, right - 1, max(0, left - 1), min(right, timeline["duration_frames"] - 1)}
    results["comparison"] = [
        {"frame": frame, "mean_rgb_error": round(float(np.abs(
            frame_rgb(videos["incremental"], frame) - frame_rgb(videos["full"], frame)
        ).mean()), 3)}
        for frame in sorted(frames)
    ]
    if any(row["mean_rgb_error"] > 3 for row in results["comparison"]):
        raise RuntimeError("Sampled frames differed from the full rebuild beyond the tolerance.")
    report = options.output / "comparison.json"
    report.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(report.resolve())


if __name__ == "__main__":
    main()
