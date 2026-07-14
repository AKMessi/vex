from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

import config


def extract_native_frames(
    video_path: str | Path,
    samples: Sequence[tuple[Path, float]],
    *,
    fps: float,
    duration_sec: float,
    timeout_sec: float = 90.0,
) -> tuple[list[Path], list[str]]:
    """Decode requested frames in one forward pass.

    Timestamp seeks can produce partially reconstructed frames with some Windows
    FFmpeg builds. Selecting decoded frame indices avoids feeding those artifacts
    to visual critics while also decoding the source only once.
    """

    if not samples:
        return [], []
    safe_fps = max(1.0, float(fps or 30.0))
    safe_duration = max(0.0, float(duration_sec or 0.0))
    max_frame_index = max(0, math.ceil(safe_duration * safe_fps) - 1)
    indexed_samples: list[tuple[Path, int]] = []
    for target, time_sec in samples:
        output_path = Path(target)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.unlink(missing_ok=True)
        frame_index = max(0, round(max(0.0, float(time_sec or 0.0)) * safe_fps))
        if safe_duration > 0:
            frame_index = min(frame_index, max_frame_index)
        indexed_samples.append((output_path, frame_index))

    unique_indices = sorted({frame_index for _, frame_index in indexed_samples})
    select_filter = "select=" + "+".join(
        f"eq(n\\,{frame_index})" for frame_index in unique_indices
    )
    errors: list[str] = []
    with tempfile.TemporaryDirectory(prefix="vex-native-frames-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        output_pattern = temp_dir / "sample_%04d.png"
        command = [
            config.FFMPEG_PATH,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vf",
            select_filter,
            "-fps_mode",
            "vfr",
            "-y",
            str(output_pattern),
        ]
        try:
            result = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=max(5.0, float(timeout_sec)),
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return [], [str(exc)]
        generated = sorted(temp_dir.glob("sample_*.png"))
        if result.returncode != 0 or len(generated) != len(unique_indices):
            detail = (result.stderr or "").strip()
            errors.append(
                detail
                or (
                    "native_frame_extraction_count_mismatch: "
                    f"expected={len(unique_indices)} actual={len(generated)}"
                )
            )
            return [], errors

        generated_by_index = dict(zip(unique_indices, generated, strict=True))
        for target, frame_index in indexed_samples:
            shutil.copyfile(generated_by_index[frame_index], target)

    paths = [target for target, _ in indexed_samples if target.is_file()]
    if len(paths) != len(indexed_samples):
        errors.append(
            "native_frame_materialization_count_mismatch: "
            f"expected={len(indexed_samples)} actual={len(paths)}"
        )
    return paths, errors
