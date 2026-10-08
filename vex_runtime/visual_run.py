"""Shared, process-visible visual budgets and stage recovery."""
from __future__ import annotations
from contextlib import contextmanager, closing
from contextvars import ContextVar
import json
import os
from pathlib import Path
import sqlite3
import time
import uuid
from typing import Any, Callable
from vex_visuals.evidence import file_digest, payload_digest


class VisualRunError(RuntimeError):
    pass


_CURRENT: ContextVar["VisualRun | None"] = ContextVar("vex_visual_run", default=None)


class VisualRun:
    def __init__(self, root: str | Path, *, max_model_calls: int = 96, max_tokens: int = 1_000_000, max_renders: int = 128, timeout_sec: float = 7200, run_id: str | None = None):
        self.root = Path(root).resolve() / ".visual-harness"
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "runs.sqlite3"
        self.run_id = run_id or uuid.uuid4().hex
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, started REAL, calls INTEGER, tokens INTEGER, renders INTEGER, max_calls INTEGER, max_tokens INTEGER, max_renders INTEGER, deadline REAL, cancelled INTEGER)")
            db.execute("CREATE TABLE IF NOT EXISTS stages (key TEXT PRIMARY KEY, name TEXT, status TEXT, owner TEXT, lease REAL, result TEXT, outputs TEXT, error TEXT)")
            db.execute("INSERT OR IGNORE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?)", (self.run_id,time.time(),0,0,0,max_model_calls,max_tokens,max_renders,time.time()+timeout_sec,0))

    @contextmanager
    def connect(self):
        with closing(sqlite3.connect(self.path, timeout=30)) as db:
            db.row_factory = sqlite3.Row
            with db:
                yield db

    def consume(self, resource: str, amount: int = 1) -> None:
        if resource not in {"calls", "tokens", "renders"} or amount < 0:
            raise ValueError("Invalid visual run resource")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM runs WHERE id=?",(self.run_id,)).fetchone()
            if row["cancelled"] or time.time() > row["deadline"]:
                raise VisualRunError("Visual run cancelled or deadline exhausted")
            ceiling = {"calls":"max_calls","tokens":"max_tokens","renders":"max_renders"}[resource]
            if row[resource]+amount > row[ceiling]:
                raise VisualRunError(f"Visual {resource} budget exhausted")
            db.execute(f"UPDATE runs SET {resource}={resource}+? WHERE id=?",(amount,self.run_id))

    def snapshot(self) -> dict[str, Any]:
        with self.connect() as db:
            return dict(db.execute("SELECT * FROM runs WHERE id=?",(self.run_id,)).fetchone())

    def cancel(self) -> None:
        with self.connect() as db:
            db.execute("UPDATE runs SET cancelled=1 WHERE id=?",(self.run_id,))

    def stage(self, name: str, inputs: dict, work: Callable[[], dict], *, output_paths: Callable[[dict], list[str]] = lambda _: []) -> tuple[dict, bool]:
        self.consume("calls", 0)
        key = payload_digest({"version":"vex-visual-stages-v1", "name":name,"inputs":inputs})
        owner = f"{os.getpid()}:{uuid.uuid4().hex}"
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM stages WHERE key=?",(key,)).fetchone()
            if row and row["status"] == "succeeded":
                try:
                    outputs = json.loads(row["outputs"])
                    if all(Path(path).is_file() and file_digest(path)==digest for path,digest in outputs.items()):
                        return json.loads(row["result"]), True
                except (OSError,ValueError,TypeError):
                    pass
            from vex_runtime.locking import process_is_running
            if row and row["status"] == "running" and row["lease"] > time.time() and process_is_running(int(row["owner"].split(":")[0])):
                raise VisualRunError(f"Visual stage is owned by another worker: {name}")
            db.execute("INSERT OR REPLACE INTO stages VALUES (?,?,?,?,?,?,?,?)",(key,name,"running",owner,time.time()+120,"","",""))
        import threading
        finished = threading.Event()
        def heartbeat():
            while not finished.wait(15):
                with self.connect() as db:
                    db.execute("UPDATE stages SET lease=? WHERE key=? AND owner=?",(time.time()+120,key,owner))
        thread = threading.Thread(target=heartbeat,daemon=True)
        thread.start()
        try:
            result = work()
            outputs = {str(Path(path).resolve()):file_digest(path) for path in output_paths(result)}
            self.consume("calls",0)
            with self.connect() as db:
                cursor = db.execute("UPDATE stages SET status='succeeded',result=?,outputs=?,lease=0 WHERE key=? AND owner=?",(json.dumps(result,allow_nan=False),json.dumps(outputs),key,owner))
                if cursor.rowcount != 1:
                    raise VisualRunError("Visual stage lost its worker lease")
            return result, False
        except BaseException as exc:
            with self.connect() as db:
                db.execute("UPDATE stages SET status='failed',error=?,lease=0 WHERE key=? AND owner=?",(type(exc).__name__,key,owner))
            raise
        finally:
            finished.set()
            thread.join(timeout=1)


def current_visual_run() -> VisualRun | None:
    return _CURRENT.get()


@contextmanager
def visual_run(root: str | Path, **limits):
    existing = _CURRENT.get()
    if existing:
        yield existing
        return
    run = VisualRun(root,**limits)
    token = _CURRENT.set(run)
    try:
        yield run
    finally:
        _CURRENT.reset(token)


def consume_visual_resource(resource: str, amount: int = 1) -> None:
    run = _CURRENT.get()
    if run:
        run.consume(resource,amount)


@contextmanager
def model_budget(prompt: str, max_tokens: int, image_count: int = 0):
    run = _CURRENT.get()
    reserved = len(prompt) + max_tokens + image_count * 4096
    usage = {}
    if run:
        run.consume("calls")
        run.consume("tokens",reserved)
    try:
        yield usage
    finally:
        if run and usage.get("total_tokens") is not None:
            actual = max(0,int(usage["total_tokens"]))
            refund = max(0,reserved-actual)
            with run.connect() as db:
                db.execute("UPDATE runs SET tokens=max(0,tokens-?) WHERE id=?",(refund,run.run_id))


def request_visual_cancellation(root: str | Path) -> int:
    path = Path(root).resolve() / ".visual-harness/runs.sqlite3"
    if not path.is_file():
        return 0
    with closing(sqlite3.connect(path,timeout=30)) as db, db:
        return db.execute("UPDATE runs SET cancelled=1 WHERE deadline>? AND cancelled=0",(time.time(),)).rowcount
