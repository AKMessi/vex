"""Repeatable rendered evidence, comparisons, and optional human calibration.

The offline suite measures renderer correctness. Pixel heuristics never stand in
for comprehension or human preference. Native vision is an explicit opt-in.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from html import escape
from urllib.parse import quote
from importlib.resources import files
import json
from pathlib import Path
import subprocess
import time

import config
from engine import probe_video
from renderers import get_renderer, render_with_manifest
from vex_runtime.visual_run import visual_run
from vex_visuals.communication_contract import build_communication_contract
from vex_visuals.evidence import file_digest, payload_digest
from vex_visuals.frame_sampling import extract_native_frames
from vex_visuals.generative_authoring import compile_open_visual_program_for_spec
from vex_visuals.open_visual_program import (
    sign_open_visual_program,
    validate_open_visual_program,
)
from vex_visuals.verifier import run_visual_verifier

VERSION = "vex-rendered-benchmark-v1"
FRACTIONS = [0.03, 0.42, 0.68, 0.94]


def corpus() -> dict:
    return json.loads(
        files("vex_visuals")
        .joinpath("benchmark_cases.json")
        .read_text(encoding="utf-8")
    )


def compile_case(case: dict, *, fps: int = 24) -> dict:
    facts, objects, relations = [], [], []
    evidence = (
        case["title"]
        + ". "
        + " ".join(f"{step[0]}: {step[1]}." for step in case["steps"])
    )
    for index, (label, meaning) in enumerate(case["steps"], 1):
        facts.append(
            {
                "fact_id": f"fact_{index}",
                "fact_type": "mechanism",
                "label": meaning,
                "evidence_ids": ["source"],
                "confidence": 1.0,
                "grounding": "transcript_exact",
            }
        )
        objects.append(
            {
                "object_id": f"object_{index}",
                "role": (
                    "input"
                    if index == 1
                    else "result" if index == len(case["steps"]) else "mechanism"
                ),
                "label": label,
                "meaning": meaning,
                "fact_ids": [f"fact_{index}"],
            }
        )
        if index > 1:
            relations.append(
                {
                    "relation_id": f"relation_{index-1}",
                    "source_id": f"object_{index-1}",
                    "target_id": f"object_{index}",
                    "relation_type": case["relation_type"],
                    "evidence_ids": ["source"],
                    "required": True,
                }
            )
    facts.append(
        {
            "fact_id": "fact_title",
            "fact_type": "context",
            "label": case["title"],
            "evidence_ids": ["source"],
            "confidence": 1.0,
            "grounding": "transcript_exact",
        }
    )
    ir = {
        "version": "visual-explanation-v3",
        "visual_id": case["case_id"],
        "scene_type": "guided_process",
        "thesis": case["title"],
        "takeaway": case["takeaway"],
        "render_policy": "render",
        "viewer_question": "How does the source-backed process work?",
        "duration_sec": 4.8,
        "composition_mode": "replace",
        "evidence": [
            {
                "evidence_id": "source",
                "source_type": "transcript",
                "text": evidence,
                "confidence": 1.0,
            }
        ],
        "facts": facts,
        "objects": objects,
        "relations": relations,
        "beats": [
            {
                "beat_id": f"beat_{index}",
                "subject_id": obj["object_id"],
                "target_id": (
                    objects[index + 1]["object_id"] if index + 1 < len(objects) else ""
                ),
                "action": (
                    "reveal"
                    if index == 0
                    else "lock_focus" if index == len(objects) - 1 else "advance_route"
                ),
                "phase": (
                    "establish"
                    if index == 0
                    else "resolve" if index == len(objects) - 1 else "explain"
                ),
                "start_fraction": index * 0.3,
                "end_fraction": 0.3 + index * 0.3,
            }
            for index, obj in enumerate(objects)
        ],
        "required_labels": [step[0] for step in case["steps"]],
        "forbidden_content": ["invented quantities"],
    }
    from visual_copy_contract import build_visual_copy_contract

    copy = build_visual_copy_contract(
        source_text=evidence,
        evidence=ir["evidence"],
        facts=facts,
        objects=objects,
        required_labels=ir["required_labels"],
        title_candidates=[case["title"]],
        takeaway_candidates=[case["takeaway"]],
    )
    ir["metadata"] = {"visual_copy_contract": copy.to_dict()}
    from vex_hyperframes.compiler import compile_hyperframes_plan

    native = compile_hyperframes_plan(
        {
            "visual_id": case["case_id"],
            "duration": 4.8,
            "sentence_text": evidence,
            "context_text": evidence,
            "visual_explanation_ir": ir,
            "hyperframes_proof_candidate_count": 1,
        }
    )
    if not native.passed:
        raise ValueError(
            f"Corpus native contracts invalid: {case['case_id']}: {native.issues}"
        )
    spec, result = compile_open_visual_program_for_spec(
        {
            **native.renderer_spec,
            "visual_id": case["case_id"],
            "duration": 4.8,
            "template": "semantic_transform",
            "headline": case["title"],
        },
        ir=ir,
        width=case["width"],
        height=case["height"],
        fps=fps,
        enable_model_authoring=False,
        candidate_count=1,
    )
    if not result.passed:
        raise ValueError(f"Corpus compilation failed: {case['case_id']}: {result}")
    program = spec["open_visual_program"]
    if case.get("chart"):
        target = next(
            item
            for item in program["elements"]
            if item.get("binding", {}).get("id") == case["chart"]["object_id"]
        )
        target.update(type="chart", data=case["chart"]["values"])
    if case.get("return_motion"):
        target = next(
            item
            for item in program["elements"]
            if item.get("binding", {}).get("kind") == "object"
        )
        program["tracks"] = [
            track
            for track in program["tracks"]
            if not (
                track["target_id"] == target["element_id"]
                and track["property"] == "translate_x"
            )
        ]
        program["tracks"].append(
            {
                "track_id": "return_journey",
                "target_id": target["element_id"],
                "property": "translate_x",
                "semantic_intent": "The request travels to review and returns for revision",
                "keyframes": [
                    {"t": 0, "value": 0, "easing": "linear"},
                    {"t": 0.42, "value": 0.05, "easing": "ease_in_out"},
                    {"t": 0.68, "value": 0, "easing": "ease_in_out"},
                    {"t": 1, "value": 0, "easing": "linear"},
                ],
            }
        )
    program = sign_open_visual_program(program)
    validation = validate_open_visual_program(program, ir=ir)
    if not validation.passed:
        raise ValueError(
            f"Corpus program invalid: {case['case_id']}: {validation.errors}"
        )
    # Recompile the derived graph when corpus features change the authored program.
    from vex_visuals.scene_graph import compile_scene_graph

    spec["open_visual_program"] = program
    spec["open_visual_program_candidates"] = [program]
    spec["scene_graph_v2"] = compile_scene_graph(
        program, creative_direction=spec.get("creative_direction") or {}
    )
    spec["visual_explanation_ir"] = ir
    spec["visual_communication_contract"] = build_communication_contract(ir).to_dict()
    return spec


@contextmanager
def offline_render_policy():
    overrides = {
        "GROQ_API_KEY": "",
        "GEMINI_API_KEY": "",
        "ANTHROPIC_API_KEY": "",
        "HYPERFRAMES_ENABLE_VISION_QA": False,
        "HYPERFRAMES_QA_MODE": "local",
        "HYPERFRAMES_VARIANT_COUNT": 1,
        "HYPERFRAMES_ENABLE_CEGIS": False,
    }
    previous = {key: getattr(config, key, None) for key in overrides}
    try:
        for key, value in overrides.items():
            setattr(config, key, value)
        yield
    finally:
        for key, value in previous.items():
            setattr(config, key, value)


def run_benchmark(
    output_dir: Path,
    *,
    case_ids: list[str] | None = None,
    renderers: tuple[str, ...] = ("remotion", "hyperframes"),
    vision: bool = False,
) -> dict:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = corpus()
    cases = [
        case for case in dataset["cases"] if not case_ids or case["case_id"] in case_ids
    ]
    if not cases or case_ids and set(case_ids) - {case["case_id"] for case in cases}:
        raise ValueError("Unknown or empty benchmark case selection")
    if not renderers or set(renderers) - {"remotion", "hyperframes"}:
        raise ValueError("Benchmark renderer must be remotion or hyperframes")
    records = []
    with visual_run(output_dir) as run:
        for case in cases:
            spec = compile_case(case)
            for renderer_name in renderers:
                started = time.monotonic()
                key = f"{case['case_id']}:{renderer_name}"
                record = {
                    "key": key,
                    "case_id": case["case_id"],
                    "renderer": renderer_name,
                    "canvas": [case["width"], case["height"]],
                    "tags": case["tags"],
                    "technical_passed": False,
                    "semantic_state": "unverified",
                    "frame_paths": [],
                }
                try:
                    with offline_render_policy():
                        asset = render_with_manifest(
                            get_renderer(renderer_name),
                            spec,
                            render_root=output_dir / renderer_name,
                            width=case["width"],
                            height=case["height"],
                            fps=24,
                        )
                    metadata = probe_video(asset.asset_path)
                    samples = [
                        (
                            output_dir
                            / "frames"
                            / key.replace(":", "_")
                            / f"frame_{index:02d}.png",
                            asset.duration_sec * fraction,
                        )
                        for index, fraction in enumerate(FRACTIONS)
                    ]
                    frames, errors = extract_native_frames(
                        asset.asset_path,
                        samples,
                        fps=float(metadata.get("fps") or 24),
                        duration_sec=asset.duration_sec,
                    )
                    local = dict(asset.metadata or {})
                    selection = local.get("variant_selection") or {}
                    record.update(
                        asset_path=asset.asset_path,
                        asset_sha256=file_digest(asset.asset_path),
                        frame_paths=[str(path) for path in frames],
                        frame_sha256=[file_digest(path) for path in frames],
                        local_score=float(
                            local.get(
                                "quality_score",
                                selection.get("selected_quality_score", 0),
                            )
                            or 0
                        ),
                        local_quality=local.get("quality_components")
                        or local.get("qa")
                        or {},
                        technical_passed=bool(
                            local.get(
                                "quality_passed",
                                selection.get("selected_quality_passed"),
                            )
                        )
                        and len(frames) == 4,
                        extraction_errors=errors,
                    )
                    if vision:
                        verdict = run_visual_verifier(
                            frames,
                            spec["visual_communication_contract"],
                            strict=True,
                            local_gate_passed=record["technical_passed"],
                            local_score=record["local_score"],
                            cache_dir=output_dir / "verifier_cache",
                        )
                        record.update(
                            semantic_state=verdict.state.value,
                            verifier=verdict.to_dict(),
                        )
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
                record["elapsed_sec"] = round(time.monotonic() - started, 3)
                records.append(record)
                print(
                    f"{key}: technical={record['technical_passed']}, semantic={record['semantic_state']}, seconds={record['elapsed_sec']}",
                    flush=True,
                )
        usage = run.snapshot()
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    report = {
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "revision": revision,
        "corpus_digest": payload_digest(dataset),
        "vision_requested": vision,
        "passed": all(
            record["technical_passed"]
            and (not vision or record["semantic_state"] == "verified")
            for record in records
        ),
        "usage": {key: usage[key] for key in ["calls", "tokens", "renders"]},
        "cases": records,
        "human_calibration": {"status": "unrated", "rating_count": 0},
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    write_gallery(report, output_dir / "gallery.html")
    return report


def write_gallery(report: dict, path: Path) -> None:
    sections = []
    for record in report["cases"]:
        frames = "".join(
            f'<img loading="lazy" src="{escape(quote(Path(frame).relative_to(path.parent).as_posix()),quote=True)}" alt="Chronological QA frame">'
            for frame in record.get("frame_paths") or []
        )
        sections.append(
            f'<section><h2>{escape(record["key"])}</h2><p>Technical pass: {record["technical_passed"]}; semantic state: {escape(record["semantic_state"])}</p><div>{frames}</div><p>{escape(record.get("error",""))}</p></section>'
        )
    path.write_text(
        '<!doctype html><meta charset="utf-8"><title>Vex rendered benchmark</title><style>body{font:16px system-ui;background:#111;color:#eee;margin:32px}section{margin:32px 0}div{display:flex;gap:12px;flex-wrap:wrap}img{max-width:320px;max-height:340px;object-fit:contain}</style><h1>Rendered evidence</h1><p>Chronological frames. Technical scores are uncalibrated. Rate comprehension and visual preference independently; no human ratings have been invented.</p>'
        + "".join(sections),
        encoding="utf-8",
    )


def compare_reports(before: dict, after: dict) -> dict:
    if before["corpus_digest"] != after["corpus_digest"]:
        raise ValueError("Comparison requires the same fixed corpus")
    left = {item["key"]: item for item in before["cases"]}
    right = {item["key"]: item for item in after["cases"]}
    if (
        not left
        or left.keys() != right.keys()
        or before["vision_requested"] != after["vision_requested"]
    ):
        raise ValueError(
            "Comparison requires identical cases, renderers, and vision policy"
        )
    rows = []
    for key in sorted(left):
        a, b = left[key], right[key]
        if a["canvas"] != b["canvas"]:
            raise ValueError("Comparison requires identical canvas dimensions")
        rows.append(
            {
                "key": key,
                "technical_regression": a["technical_passed"]
                and not b["technical_passed"],
                "technical_improvement": not a["technical_passed"]
                and b["technical_passed"],
                "local_score_delta": round(
                    b.get("local_score", 0) - a.get("local_score", 0), 4
                ),
                "elapsed_delta_sec": round(b["elapsed_sec"] - a["elapsed_sec"], 3),
                "semantic_before": a["semantic_state"],
                "semantic_after": b["semantic_state"],
            }
        )
    return {
        "corpus_digest": after["corpus_digest"],
        "cases": rows,
        "technical_regressions": sum(row["technical_regression"] for row in rows),
        "human_quality_claim": "Requires independent human ratings; local score delta is not a preference gain.",
    }


def calibrate_report(report: dict, ratings: list[dict]) -> dict:
    records = {record["key"]: record for record in report["cases"]}
    seen = set()
    for rating in ratings:
        identity = (rating.get("reviewer"), rating.get("key"))
        if not identity[0] or identity in seen or identity[1] not in records:
            raise ValueError("Ratings require unique reviewers and known case keys")
        seen.add(identity)
        for metric in ["comprehension", "visual_preference"]:
            value = rating.get(metric)
            if type(value) not in (int, float) or not 1 <= value <= 5:
                raise ValueError("Human ratings must be between 1 and 5")
    verified = [
        rating
        for rating in ratings
        if records[rating["key"]]["semantic_state"] == "verified"
    ]
    return {
        "status": "rated" if ratings else "unrated",
        "rating_count": len(ratings),
        "reviewer_count": len({rating["reviewer"] for rating in ratings}),
        "mean_comprehension": (
            sum(rating["comprehension"] for rating in ratings) / len(ratings)
            if ratings
            else None
        ),
        "mean_visual_preference": (
            sum(rating["visual_preference"] for rating in ratings) / len(ratings)
            if ratings
            else None
        ),
        "verified_rating_count": len(verified),
        "false_acceptance_rate": (
            sum(rating["comprehension"] < 3 for rating in verified) / len(verified)
            if verified
            else None
        ),
        "false_acceptance_definition": "Verified outputs rated below 3/5 comprehension. Multiple ratings of one scene are correlated; this is descriptive, not a statistical quality claim.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(".analysis/rendered-benchmark")
    )
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--renderers", nargs="+", default=["remotion", "hyperframes"])
    parser.add_argument(
        "--vision",
        action="store_true",
        help="Use configured native vision APIs; consumes model quota",
    )
    parser.add_argument(
        "--compare", type=Path, help="Compare an existing report with the new run"
    )
    parser.add_argument(
        "--ratings",
        type=Path,
        help="Calibrate an existing output/report.json without rerendering",
    )
    args = parser.parse_args()
    if args.ratings:
        report = json.loads((args.output / "report.json").read_text(encoding="utf-8"))
        report["human_calibration"] = calibrate_report(
            report, json.loads(args.ratings.read_text(encoding="utf-8"))
        )
        (args.output / "calibrated_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    else:
        report = run_benchmark(
            args.output,
            case_ids=args.cases,
            renderers=tuple(args.renderers),
            vision=args.vision,
        )
        if args.compare:
            comparison = compare_reports(
                json.loads(args.compare.read_text(encoding="utf-8")), report
            )
            (args.output / "comparison.json").write_text(
                json.dumps(comparison, indent=2), encoding="utf-8"
            )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
