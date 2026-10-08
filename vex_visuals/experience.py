"""Small, attributable memories and reference frames; never publication authority."""
from __future__ import annotations
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import time
from vex_visuals.evidence import file_digest, payload_digest


def _database(root):
    path = Path(root)/".visual-harness/experience.sqlite3"
    path.parent.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(path,timeout=15)
    db.row_factory=sqlite3.Row
    db.execute("CREATE TABLE IF NOT EXISTS experiences (id TEXT PRIMARY KEY, scene_type TEXT, renderer TEXT, verified INTEGER, score REAL, updated REAL, payload TEXT)")
    return db


def record_visual_experience(root, spec, asset, outcome) -> None:
    selected=outcome.selected
    frames=[{"path":str(path),"sha256":file_digest(path)} for path in selected.frame_paths if Path(path).is_file()]
    payload={"version":"vex-visual-experience-v1","verifier_version":selected.verification.version,"source_ir_signature":(spec.get("visual_communication_contract") or {}).get("source_ir_signature"),"issues":selected.verification.issues,"repair_history":outcome.repair_history,"frames":frames,"program_id":(spec.get("open_visual_program") or {}).get("program_id"),"runtime_version":outcome.version,"quality_state":selected.verification.state.value}
    key=payload_digest({"program":spec.get("open_visual_program"),"renderer":asset.renderer,"frames":frames})
    with closing(_database(root)) as db, db:
        db.execute("INSERT OR REPLACE INTO experiences VALUES (?,?,?,?,?,?,?)",(key,(spec.get("visual_explanation_ir") or {}).get("scene_type",""),asset.renderer,int(selected.verification.verified and outcome.passed),selected.verification.score,time.time(),json.dumps(payload)))
        db.execute("DELETE FROM experiences WHERE id IN (SELECT id FROM experiences ORDER BY updated DESC LIMIT -1 OFFSET 200)")


def relevant_visual_experiences(root, ir, *, limit=3) -> dict:
    with closing(_database(root)) as db:
        rows=db.execute("SELECT * FROM experiences WHERE scene_type=? ORDER BY verified DESC,score DESC,updated DESC LIMIT ?",(ir.get("scene_type",""),max(1,min(limit,3)))).fetchall()
    examples=[]
    references=[]
    for row in rows:
        payload=json.loads(row["payload"])
        from vex_visuals.verifier import VISUAL_VERIFIER_VERSION
        if payload.get("verifier_version")!=VISUAL_VERIFIER_VERSION:
            continue
        examples.append({"quality_state":payload["quality_state"],"issues":payload["issues"],"repair_history":payload["repair_history"][-2:],"runtime_version":payload["runtime_version"]})
        if row["verified"] and row["score"]>=.72:
            for frame in payload["frames"][-1:]:
                try:
                    if file_digest(frame["path"])==frame["sha256"]:
                        references.append(frame["path"])
                except OSError:
                    pass
    return {"examples":examples,"reference_paths":references,"instruction":"References show visual style only. Current source evidence determines all facts and quantities."}
