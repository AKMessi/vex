from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from vex_visuals.evidence import file_digest, payload_digest
from vex_visuals.frame_sampling import extract_native_frames


def check_remotion_delivery(
    timeline_path: Path, output_path: Path, metadata: dict[str, Any],
) -> dict[str, Any]:
    props = json.loads(timeline_path.read_text(encoding="utf-8"))
    timeline = props["timeline"]
    fps = timeline["fps"]
    duration = timeline["duration_frames"] / fps
    format_name = output_path.suffix
    expected_codec = {".mp4": "h264", ".mov": "prores", ".webm": "vp9", ".gif": "gif"}[format_name]
    issues = [f"rendered_{key}_mismatch" for key in ("width", "height") if timeline[key] != metadata.get(key)]
    if expected_codec != metadata.get("codec"):
        issues.append("rendered_codec_mismatch")
    # GIF delays are stored as centiseconds. Validate its duration at container precision.
    if format_name != ".gif" and abs(float(metadata.get("fps") or 0) - fps) > 0.01:
        issues.append("rendered_fps_mismatch")
    duration_tolerance = max(1 / fps, 0.05)
    if format_name == ".gif":
        duration_tolerance += 0.01
    if abs(duration - metadata["duration_sec"]) > duration_tolerance:
        issues.append("rendered_frame_clock_duration_mismatch")
    audio_expected = bool(timeline["audio"]["narration"] or timeline["audio"]["music"])
    if audio_expected != bool(metadata["has_audio"]):
        issues.append("rendered_audio_track_missing" if audio_expected else "rendered_unexpected_audio_track")

    evidence_dir = output_path.parent / "delivery_frames"
    scene_samples = []
    for scene_index, scene in enumerate(timeline["scenes"], 1):
        start = scene["start_frame"]
        last = scene["duration_frames"] - 1
        frames = sorted({start, start + round(last * 0.55), start + last})
        scene_samples.append((scene, [
            (evidence_dir / f"shot_{scene_index:03d}_frame_{frame:06d}.png", frame)
            for frame in frames
        ]))
    samples = [(path, frame / fps) for _, rows in scene_samples for path, frame in rows]
    paths, extraction_errors = extract_native_frames(
        output_path, samples, fps=fps, duration_sec=duration, max_dimension=720,
    )
    if extraction_errors or len(paths) != len(samples):
        issues.append("rendered_frame_evidence_incomplete")
    available = set(paths)
    shots = []
    for scene, rows in scene_samples:
        frames = [
            {
                "frame": frame,
                "path": path.relative_to(output_path.parent).as_posix(),
                "sha256": file_digest(path),
                "detail_fraction": _detail_fraction(path),
            }
            for path, frame in rows if path in available
        ]
        # The resolved frame should still contain visible foreground detail.
        if frames and not scene.get("media") and frames[-1]["detail_fraction"] < 0.002:
            issues.append(f"rendered_shot_has_no_visible_detail:{scene['id']}")
        shots.append({
            "id": scene["id"],
            "program_id": (scene.get("program") or {}).get("program_id") if not scene.get("media") else None,
            "media": scene.get("media"),
            "frames": frames,
        })
    receipt = {
        "version": "vex-remotion-delivery-v1",
        "passed": not issues,
        "issues": issues,
        "expected_duration_sec": duration,
        "timeline_sha256": payload_digest(props),
        "output_sha256": file_digest(output_path),
        "scene_count": len(shots),
        "typography_fallbacks": [
            scene["id"] for scene in timeline["scenes"] if not scene.get("program") and not scene.get("media")
        ],
        "extraction_errors": extraction_errors,
        "shots": shots,
    }
    return {**receipt, "receipt_sha256": payload_digest(receipt)}


def _detail_fraction(path: Path) -> float:
    with Image.open(path) as frame:
        frame.thumbnail((320, 320))
        pixels = np.asarray(frame.convert("RGB"), dtype=np.int16)
    background = np.median(pixels.reshape(-1, 3), axis=0)
    deviation = np.max(np.abs(pixels - background), axis=2)
    return round(float(np.mean(deviation >= 22)), 6)
