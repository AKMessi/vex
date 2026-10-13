from __future__ import annotations
import json
import os
from pathlib import Path
import signal
import shutil
import tempfile
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from vex_runtime.visual_run import VisualRun, VisualRunError

_SLOTS = threading.BoundedSemaphore(2)


@contextmanager
def _slot(run):
    while not _SLOTS.acquire(timeout=0.2):
        run.consume("calls", 0)
    try:
        yield
    finally:
        _SLOTS.release()


def _stop_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            timeout=10,
            check=False,
        )
    else:
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=3)


def render_isolated(
    run: VisualRun,
    renderer: str,
    spec: dict,
    *,
    render_root: Path,
    width: int,
    height: int,
    fps: float,
) -> dict:
    from vex_runtime.renderer_policy import renderer_policy

    request_dir = run.root / "workers"
    request_dir.mkdir(exist_ok=True)
    request_path = request_dir / f"{uuid.uuid4().hex}.json"
    request_path.write_text(
        json.dumps(
            {
                "project_root": str(run.root.parent),
                "run_id": run.run_id,
                "renderer": renderer,
                "spec": spec,
                "render_root": str(render_root.resolve()),
                "width": width,
                "height": height,
                "fps": fps,
                "policy": renderer_policy(),
            },
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    result_path = request_path.with_suffix(".result.json")
    log_path = request_path.with_suffix(".log")
    scratch_root = Path(tempfile.gettempdir()) / "vex-render-workers"
    if shutil.disk_usage(scratch_root.parent).free < 1536 * 1024 * 1024:
        scratch_root = Path(run.root.anchor)/"vex-render-tmp" if os.name=="nt" else run.root.parent/".render-tmp"
    scratch_root.mkdir(parents=True,exist_ok=True)
    with _slot(run), tempfile.TemporaryDirectory(prefix="worker-",dir=scratch_root) as scratch, log_path.open("w", encoding="utf-8") as log:
        run.consume("calls", 0)
        environment = os.environ.copy()
        environment.update(TMP=scratch,TEMP=scratch,TMPDIR=scratch)
        process = subprocess.Popen(
            [sys.executable, "-m", "vex_runtime.render_worker", str(request_path)],
            stdout=log,
            stderr=log,
            stdin=subprocess.DEVNULL,
            env=environment,
            start_new_session=os.name != "nt",
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            while process.poll() is None:
                run.consume("calls", 0)
                time.sleep(0.2)
        except BaseException:
            _stop_tree(process)
            raise
    if not result_path.is_file():
        raise VisualRunError(
            f"Renderer worker exited without an artifact; inspect {log_path}"
        )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if process.returncode != 0 or not result.get("success"):
        raise VisualRunError(
            result.get("error") or f"Renderer worker failed; inspect {log_path}"
        )
    return result["asset"]
