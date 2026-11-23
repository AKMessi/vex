"""Measure the delivered audio without modifying its mix."""
from __future__ import annotations

import json
import math
import re
import subprocess
from pathlib import Path

import config
from engine import probe_video


def measure_audio(path: Path) -> dict:
    if not probe_video(str(path))["has_audio"]:
        return {"has_audio": False}
    command = [
        config.FFMPEG_PATH, "-nostdin", "-hide_banner", "-i", str(path), "-map", "0:a:0", "-vn",
        "-af", "loudnorm=print_format=json", "-f", "null", "-",
    ]
    result = subprocess.run(command, capture_output=True, encoding="utf-8", errors="replace",
                            timeout=120, check=False)
    matches = re.findall(r"\{[^{}]*\"input_i\"[^{}]*\}", result.stderr)
    if result.returncode or not matches:
        raise ValueError("FFmpeg could not measure the delivered audio.")
    values = json.loads(matches[-1])
    metrics = {"integrated_lufs": float(values["input_i"]), "true_peak_dbtp": float(values["input_tp"])}
    return {"has_audio": True, **{name: round(value, 2) if math.isfinite(value) else None
                                   for name, value in metrics.items()}}
