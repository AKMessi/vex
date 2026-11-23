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
from vex_runtime.provider_meter import cost_micro, initialize_meter, load_policy, request_usage


class VisualRunError(RuntimeError):
    pass


_CURRENT: ContextVar["VisualRun | None"] = ContextVar("vex_visual_run", default=None)


class VisualRun:
    def __init__(self, root: str | Path, *, max_model_calls: int = 96, max_tokens: int = 1_000_000, max_renders: int = 128, timeout_sec: float = 7200, run_id: str | None = None):
        self.root = Path(root).resolve() / ".visual-harness"
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "runs.sqlite3"
        self.run_id = run_id or uuid.uuid4().hex
        self.spend_limit, self.rates = load_policy()
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, started REAL, calls INTEGER, tokens INTEGER, renders INTEGER, max_calls INTEGER, max_tokens INTEGER, max_renders INTEGER, deadline REAL, cancelled INTEGER)")
            db.execute("CREATE TABLE IF NOT EXISTS stages (key TEXT PRIMARY KEY, name TEXT, status TEXT, owner TEXT, lease REAL, result TEXT, outputs TEXT, error TEXT)")
            initialize_meter(db)
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

    def reserve_request(self, provider: str, model: str, inputs: int, outputs: int) -> tuple[str, Any]:
        rates = self.rates.get(f"{provider}/{model}")
        estimate = cost_micro(rates, inputs, outputs)
        request_id = uuid.uuid4().hex
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM runs WHERE id=?", (self.run_id,)).fetchone()
            if row["cancelled"] or time.time() > row["deadline"]:
                raise VisualRunError("Visual run cancelled or deadline exhausted")
            if row["calls"] >= row["max_calls"]:
                raise VisualRunError("Visual calls budget exhausted")
            if row["tokens"] + inputs + outputs > row["max_tokens"]:
                raise VisualRunError("Visual tokens budget exhausted")
            if self.spend_limit is not None:
                if estimate is None:
                    raise VisualRunError(f"Set an input and output rate for {provider}/{model} to use a spend limit.")
                total = db.execute("SELECT COALESCE(SUM(cost_micro), 0) FROM model_usage").fetchone()[0]
                if total + estimate > self.spend_limit:
                    raise VisualRunError("The estimated provider spend limit would be exceeded.")
            db.execute("UPDATE runs SET calls=calls+1,tokens=tokens+? WHERE id=?", (inputs + outputs, self.run_id))
            db.execute("INSERT INTO model_usage VALUES (?,?,?,?,?,?,?,?,?)",
                       (request_id, self.run_id, provider, model, "reserved", inputs, outputs, estimate, time.time()))
        return request_id, rates

    def finish_request(self, request_id: str, rates, usage: dict) -> None:
        counted = request_usage(usage)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT input_tokens, output_tokens FROM model_usage WHERE id=?", (request_id,)).fetchone()
            total = usage.get("total_tokens")
            if counted is None and type(total) is int and total >= 0:
                # Some transports expose only the total. Price those tokens at
                # the more expensive rate and retain the distinction in the ledger.
                counted = (total, 0) if rates and rates[0] >= rates[1] else (0, total)
                status = "reported_total"
            else:
                status = "reported_split" if counted is not None else "estimated"
            if counted is None:
                db.execute("UPDATE model_usage SET status=? WHERE id=?", (status, request_id))
                return
            inputs, outputs = counted
            delta = inputs + outputs - row["input_tokens"] - row["output_tokens"]
            db.execute("UPDATE runs SET tokens=max(0,tokens+?) WHERE id=?", (delta, self.run_id))
            db.execute("UPDATE model_usage SET status=?,input_tokens=?,output_tokens=?,cost_micro=? WHERE id=?",
                       (status, inputs, outputs, cost_micro(rates, inputs, outputs), request_id))

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


def spend_limit_enabled() -> bool:
    return bool(os.getenv("VEX_PROVIDER_MAX_USD", "").strip())


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
def model_budget(prompt: str, max_tokens: int, image_count: int = 0, *, provider: str = "", model: str = ""):
    run = _CURRENT.get()
    if run is None and spend_limit_enabled():
        raise VisualRunError("Run this provider request from a project so Vex can enforce its spend limit.")
    inputs = len(prompt.encode("utf-8")) + image_count * 4096
    usage = {}
    if run:
        request_id, rates = run.reserve_request(provider or "unknown", model or "unknown", inputs, max_tokens)
    try:
        yield usage
    finally:
        if run:
            run.finish_request(request_id, rates, usage)


def request_visual_cancellation(root: str | Path) -> int:
    path = Path(root).resolve() / ".visual-harness/runs.sqlite3"
    if not path.is_file():
        return 0
    with closing(sqlite3.connect(path,timeout=30)) as db, db:
        return db.execute("UPDATE runs SET cancelled=1 WHERE deadline>? AND cancelled=0",(time.time(),)).rowcount
