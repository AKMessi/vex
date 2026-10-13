"""Check the delivered pixels for fullscreen, PiP, and alpha compositions."""

from __future__ import annotations
from pathlib import Path
import re
from PIL import Image, ImageOps
import numpy as np
from engine import probe_video, _transition_duration_sec
from vex_visuals.frame_sampling import extract_native_frames


def evaluate_native_composite(
    source_path, output_path, overlays, *, source_meta, output_meta
):
    root = Path(output_path).parent / "composite_evidence" / Path(output_path).stem
    fps = float(output_meta.get("fps") or source_meta.get("fps") or 30)
    duration = float(output_meta.get("duration_sec") or 0)
    width, height = int(output_meta.get("width") or 0), int(
        output_meta.get("height") or 0
    )
    if width <= 0 or height <= 0 or duration <= 0:
        return {
            "passed": False,
            "issues": ["composite_evidence_invalid_metadata"],
            "samples": [],
        }
    all_samples = []
    windows = []
    for index, overlay in enumerate(overlays):
        start = float(overlay.get("start") or 0)
        span = float(overlay.get("end") or 0) - start
        if span <= 0:
            continue
        in_time = _transition_duration_sec(overlay.get("transition_in"))
        out_time = _transition_duration_sec(overlay.get("transition_out"))
        times = [
            min(
                max(span * fraction, in_time + 2 / fps),
                max(span - out_time - 2 / fps, span * 0.5),
            )
            for fraction in [0.18, 0.5, 0.78, 0.94]
        ]
        targets = [
            root / f"overlay_{index:03d}_frame_{sample:02d}.png" for sample in range(4)
        ]
        all_samples.extend(
            (target, start + time) for target, time in zip(targets, times)
        )
        windows.append((overlay, targets, times))
    frames, errors = extract_native_frames(
        output_path, all_samples, fps=fps, duration_sec=duration
    )
    if len(frames) != len(all_samples):
        return {
            "passed": False,
            "issues": ["composite_native_frames_unavailable"],
            "errors": errors,
            "samples": [],
        }
    source_targets = [
        (root / f"source_{index:03d}.png", time)
        for index, (_, time) in enumerate(all_samples)
    ]
    source_frames, source_errors = extract_native_frames(
        source_path,
        source_targets,
        fps=float(source_meta.get("fps") or fps),
        duration_sec=float(source_meta.get("duration_sec") or duration),
    )
    if len(source_frames) != len(source_targets):
        return {
            "passed": False,
            "issues": ["composite_source_frames_unavailable"],
            "errors": source_errors,
            "samples": [],
        }
    records = []
    issues = []
    cursor = 0
    for index, (overlay, targets, times) in enumerate(windows):
        asset = Path(str(overlay.get("asset_path") or ""))
        if not asset.is_file():
            issues.append("composite_asset_missing")
            cursor += len(targets)
            continue
        asset_meta = probe_video(str(asset))
        asset_samples = [
            (root / f"asset_{index:03d}_{number:02d}.png", time)
            for number, time in enumerate(times)
        ]
        asset_frames, asset_errors = extract_native_frames(
            asset,
            asset_samples,
            fps=float(asset_meta.get("fps") or fps),
            duration_sec=float(asset_meta.get("duration_sec") or 0),
        )
        if len(asset_frames) != len(targets):
            issues.append("composite_asset_frames_unavailable")
            cursor += len(targets)
            continue
        mode = str(overlay.get("compose_mode") or "replace")
        for target, asset_frame, local_time in zip(targets, asset_frames, times):
            with Image.open(target) as image:
                actual = image.convert("RGB")
            with Image.open(asset_frame) as image:
                visual = image.convert("RGBA")
            with Image.open(source_frames[cursor]) as image:
                base = image.convert("RGBA").resize((width, height))
            cursor += 1
            if mode == "picture_in_picture":
                visual_width = round(width * float(overlay.get("scale") or 0.42))
                visual = visual.resize(
                    (
                        visual_width,
                        max(2, round(visual.height * visual_width / visual.width)),
                    ),
                    Image.Resampling.LANCZOS,
                )
                margin = int(overlay.get("margin") or 24)
                position = str(overlay.get("position") or "bottom_right")
                x = (
                    margin
                    if "left" in position
                    else (
                        width - visual.width - margin
                        if "right" in position
                        else (width - visual.width) // 2
                    )
                )
                y = (
                    margin
                    if position.startswith("top")
                    else (
                        height - visual.height - margin
                        if position.startswith("bottom")
                        else (height - visual.height) // 2
                    )
                )
                expected = base.copy()
                expected.alpha_composite(visual, (x, y))
                box = (x, y, x + visual.width, y + visual.height)
                left = np.asarray(actual.crop(box).resize((320, 180)), dtype=np.float32)
                right = np.asarray(
                    expected.convert("RGB").crop(box).resize((320, 180)),
                    dtype=np.float32,
                )
            else:
                visual = ImageOps.fit(
                    visual, (width, height), method=Image.Resampling.LANCZOS
                )
                expected = (
                    visual.convert("RGB")
                    if mode == "replace"
                    else Image.alpha_composite(base, visual).convert("RGB")
                )
                left = np.asarray(actual.resize((480, 270)), dtype=np.float32)
                right = np.asarray(expected.resize((480, 270)), dtype=np.float32)
                if mode == "overlay":
                    alpha = (
                        np.asarray(
                            visual.getchannel("A").resize((480, 270)), dtype=np.float32
                        )
                        / 255
                    )
                    mask = alpha > 0.1
                    if mask.any():
                        left, right = left[mask], right[mask]
            similarity = 1 - float(np.mean(np.abs(left - right))) / 255
            if similarity < 0.82:
                issues.append("composite_native_visual_mismatch")
            records.append(
                {
                    "visual_id": overlay.get("visual_id"),
                    "mode": mode,
                    "asset_time_sec": local_time,
                    "frame_path": str(target),
                    "similarity": round(similarity, 4),
                }
            )
    return {
        "version": "vex-native-composite-evidence-v1",
        "passed": not issues,
        "issues": list(dict.fromkeys(issues)),
        "samples": records,
    }
