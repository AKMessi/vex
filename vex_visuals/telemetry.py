from __future__ import annotations
import json
import os
import subprocess
import re
from pathlib import Path
from importlib.resources import files
from typing import Any
from vex_runtime.hyperframes import (
    resolve_node_executable,
    resolve_hyperframes_cli_path,
)


def evaluate_browser_telemetry(samples: list[dict[str, Any]]) -> dict[str, Any]:
    issues = []
    for sample in samples:
        if not sample.get("fonts_ready"):
            issues.append("browser_fonts_not_ready")
        for node in sample.get("nodes") or []:
            if not node.get("visible"):
                continue
            if node.get("clipped"):
                issues.append("browser_clipped_text:" + str(node.get("element_id")))
            if node.get("text") and float(node.get("font_size") or 0) < 16:
                issues.append("browser_illegible_text:" + str(node.get("element_id")))
            if node.get("text"):
                contrast = measured_text_contrast(node)
                if contrast is not None and contrast < 4.5:
                    issues.append(
                        "browser_low_contrast_text:" + str(node.get("element_id"))
                    )
    return {
        "version": "vex-browser-telemetry-v1",
        "available": bool(samples),
        "passed": bool(samples) and not issues,
        "issues": list(dict.fromkeys(issues)),
        "sample_count": len(samples),
    }


def measured_text_contrast(node: dict) -> float | None:
    def luminance(color):
        values = [float(value) for value in re.findall(r"[\d.]+", str(color or ""))]
        if len(values) not in (3, 4) or len(values) == 4 and values[3] < 0.99:
            return None
        channels = [value / 255 for value in values[:3]]
        channels = [
            value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
            for value in channels
        ]
        return sum(
            value * weight for value, weight in zip(channels, [0.2126, 0.7152, 0.0722])
        )

    foreground = luminance(node.get("text_color"))
    background = luminance(node.get("background_color"))
    if foreground is None or background is None:
        return None
    light, dark = sorted([foreground, background], reverse=True)
    return (light + 0.05) / (dark + 0.05)


def observed_layout_occupancy(samples: list[dict]) -> float | None:
    """Union of observed visible text/panel bounds in the resolved browser frame.

    Sparse ink can still represent substantial outlined panels. Rasterize bounds
    on a small grid so overlapping/nested boxes cannot inflate the area sum.
    """
    if not samples:
        return None
    import numpy as np

    sample = samples[-1]
    viewport = sample.get("viewport") or {}
    width, height = float(viewport.get("width") or 0), float(
        viewport.get("height") or 0
    )
    if not width or not height:
        return None
    mask = np.zeros((180, 320), dtype=bool)
    for node in sample.get("nodes") or []:
        if not node.get("visible"):
            continue
        bounds = node.get("bounds") or {}
        x, y = float(bounds.get("x") or 0), float(bounds.get("y") or 0)
        right, bottom = x + float(bounds.get("width") or 0), y + float(
            bounds.get("height") or 0
        )
        x1, x2 = [max(0, min(320, round(value / width * 320))) for value in [x, right]]
        y1, y2 = [
            max(0, min(180, round(value / height * 180))) for value in [y, bottom]
        ]
        mask[y1:y2, x1:x2] = True
    return float(mask.mean())


def probe_html_layout(
    html_path: Path,
    *,
    width: int,
    height: int,
    duration_sec: float,
    fps: float,
    output_dir: Path,
    ablated_relation_ids: list[str] | None = None,
    fractions: list[float] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "browser_telemetry.json"
    request_path = output_dir / "browser_probe_request.json"
    capture_dir = output_dir / "ablation_frames" if ablated_relation_ids else None
    request_path.write_text(
        json.dumps(
            {
                "html_path": str(html_path.resolve()),
                "width": width,
                "height": height,
                "duration_sec": duration_sec,
                "fps": fps,
                "output_path": str(output.resolve()),
                "fractions": fractions or [0.03, 0.42, 0.68, 0.9, 0.97],
                "ablated_relation_ids": ablated_relation_ids or [],
                "capture_dir": str(capture_dir.resolve()) if capture_dir else None,
            }
        ),
        encoding="utf-8",
    )
    node = resolve_node_executable()
    if not node:
        return {
            "available": False,
            "passed": False,
            "issues": ["browser_probe_node_unavailable"],
        }
    env = os.environ.copy()
    cli = resolve_hyperframes_cli_path()
    node_root = cli.parent.parent.parent if cli else Path(__file__).parents[1]
    env["VEX_REMOTION_NODE_ROOT"] = str(node_root)
    try:
        result = subprocess.run(
            [
                node,
                str(files("renderers").joinpath("visual_probe.mjs")),
                str(request_path),
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if result.returncode != 0 or not output.is_file():
            return {
                "available": False,
                "passed": False,
                "issues": ["browser_probe_failed"],
                "error": (result.stderr or "")[-1000:],
            }
        payload = json.loads(output.read_text(encoding="utf-8"))
        return {
            **payload,
            "quality": evaluate_browser_telemetry(payload.get("samples") or []),
            "frame_paths": (
                [str(path) for path in sorted(capture_dir.glob("frame_*.png"))]
                if capture_dir
                else []
            ),
        }
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        return {
            "available": False,
            "passed": False,
            "issues": ["browser_probe_failed"],
            "error": type(exc).__name__,
        }
