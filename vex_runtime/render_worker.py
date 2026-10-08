"""Isolated renderer entry point. Requests contain typed data, never source code."""
from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path
import sys
import config
from renderers import get_renderer
from vex_runtime.visual_run import visual_run


def main() -> None:
    request_path = Path(sys.argv[1]).resolve(strict=True)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    config.reload_settings()
    output = request_path.with_suffix(".result.json")
    try:
        with visual_run(request["project_root"],run_id=request["run_id"]):
            renderer = get_renderer(request["renderer"])
            asset = renderer.render(request["spec"], Path(request["render_root"]), request["width"], request["height"], request["fps"])
            payload = {"success":True,"asset":asdict(asset)}
    except Exception as exc:
        payload = {"success":False,"error":f"{type(exc).__name__}: {exc}"}
    output.write_text(json.dumps(payload,allow_nan=False),encoding="utf-8")
    if not payload["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
