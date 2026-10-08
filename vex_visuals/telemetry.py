from __future__ import annotations
import json
import os
import subprocess
from pathlib import Path
from importlib.resources import files
from typing import Any
from vex_runtime.hyperframes import resolve_node_executable, resolve_hyperframes_cli_path


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
    return {"version": "vex-browser-telemetry-v1", "available": bool(samples), "passed": bool(samples) and not issues, "issues": list(dict.fromkeys(issues)), "sample_count": len(samples)}


def probe_html_layout(html_path: Path, *, width: int, height: int, duration_sec: float, fps: float, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "browser_telemetry.json"
    request_path = output_dir / "browser_probe_request.json"
    request_path.write_text(json.dumps({"html_path": str(html_path.resolve()), "width": width, "height": height, "duration_sec": duration_sec, "fps": fps, "output_path": str(output.resolve()), "fractions": [.03, .42, .68, .9, .97]}), encoding="utf-8")
    node = resolve_node_executable()
    if not node:
        return {"available": False, "passed": False, "issues": ["browser_probe_node_unavailable"]}
    env = os.environ.copy()
    cli = resolve_hyperframes_cli_path()
    node_root = cli.parent.parent.parent if cli else Path(__file__).parents[1]
    env["VEX_REMOTION_NODE_ROOT"] = str(node_root)
    try:
        result = subprocess.run([node, str(files("renderers").joinpath("visual_probe.mjs")), str(request_path)], env=env, capture_output=True, text=True, timeout=60, check=False)
        if result.returncode != 0 or not output.is_file():
            return {"available": False, "passed": False, "issues": ["browser_probe_failed"], "error": (result.stderr or "")[-1000:]}
        payload = json.loads(output.read_text(encoding="utf-8"))
        return {**payload, "quality": evaluate_browser_telemetry(payload.get("samples") or [])}
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "passed": False, "issues": ["browser_probe_failed"], "error": type(exc).__name__}
