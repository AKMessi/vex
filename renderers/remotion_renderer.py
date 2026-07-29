from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import config
from engine import probe_video
from renderers.base import (
    RenderedAsset,
    RendererStatus,
    VisualRenderer,
    VisualRendererError,
    safe_render_job_dir,
)
from vex_runtime.hyperframes import (
    managed_renderer_runtime_dir,
    node_major_version,
    node_platform_arch,
    node_runtime_identity,
    renderer_native_runtime_status,
    resolve_node_executable,
)
from vex_runtime.paths import data_dir
from vex_remotion import (
    compile_remotion_scene_program,
    evaluate_remotion_render,
    evaluate_remotion_structure,
)
from vex_visuals.aesthetic_critic import evaluate_frame_aesthetics


REMOTION_COMPOSITION_ID = "VexAutoVisual"
REMOTION_PACKAGE_VERSION = "4.0.487"
REMOTION_MEDIA_CONTRACT_VERSION = "vex-remotion-media-contract-v1"

SUPPORTED_REMOTION_TEMPLATES = {
    "semantic_architecture",
    "semantic_causal",
    "semantic_decision",
    "semantic_interface",
    "semantic_metric",
    "semantic_narrative",
    "semantic_quote",
    "semantic_route",
    "semantic_transform",
    "data_journey",
    "signal_network",
    "kinetic_route",
    "spotlight_compare",
    "interface_cascade",
    "ribbon_quote",
    "causal_chain",
    "flywheel_loop",
    "decision_matrix",
    "anatomy_cutaway",
    "stack_ranking",
    "contrast_ladder",
    "proof_sequence",
    "narrative_arc",
    "concept_map",
    "problem_solution",
    "myth_buster",
    "checklist_reveal",
    "risk_radar",
    "opportunity_map",
    "scorecard",
    "pipeline_xray",
    "decision_tree",
    "momentum_wave",
    "focus_ring",
    "timeline_filmstrip",
    "quote_breakdown",
    "market_map",
    "mechanism_blueprint",
    "data_pulse",
    "metric_callout",
    "keyword_stack",
    "timeline_steps",
    "comparison_split",
    "quote_focus",
    "system_flow",
    "stat_grid",
}


def _safe_scene_name(spec_id: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(spec_id or "visual")).strip("_")
    return cleaned or "auto_visual"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _runner_path() -> Path:
    return Path(__file__).resolve().with_name("remotion_runner.mjs")


def _entry_template_path() -> Path:
    return Path(__file__).resolve().with_name("remotion_entry.jsx")


def _scene_graph_runtime_path() -> Path:
    return Path(__file__).resolve().with_name("remotion_scene_graph.jsx")


def _node_major_version() -> int | None:
    return node_major_version()


def _remotion_timeout_sec() -> int | None:
    try:
        timeout = int(getattr(config, "REMOTION_RENDER_TIMEOUT_SEC", 0))
    except (TypeError, ValueError):
        timeout = 0
    if timeout <= 0:
        return None
    return max(30, timeout)


def _remotion_timeout_ms() -> int:
    timeout = _remotion_timeout_sec()
    if timeout is None:
        return 30000
    return max(10000, int(timeout * 1000))


def _remotion_concurrency() -> str:
    return str(getattr(config, "REMOTION_RENDER_CONCURRENCY", "") or "").strip()


def _render_fidelity(spec: dict[str, Any]) -> str:
    fidelity = str(spec.get("remotion_render_fidelity") or "final").strip().lower()
    return fidelity if fidelity in {"final", "preview"} else "final"


def _as_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _alpha_requested(spec: dict[str, Any]) -> bool:
    overlay = str(spec.get("composition_mode") or "").strip().lower() == "overlay"
    alpha = _as_bool(spec.get("alpha"), overlay)
    return alpha and _as_bool(spec.get("transparent_background"), alpha)


def _render_output_policy(spec: dict[str, Any]) -> dict[str, Any]:
    fidelity = _render_fidelity(spec)
    if fidelity == "preview":
        return {
            "version": REMOTION_MEDIA_CONTRACT_VERSION,
            "fidelity": fidelity,
            "filename": "visual.mp4",
            "container": "mp4",
            "codec": "h264",
            "pixel_format": "yuv420p",
            "encoded_pixel_format": "yuv420p",
            "prores_profile": None,
            "color_space": "bt709",
            "color_primaries": None,
            "color_transfer": None,
            "color_range": "tv",
            "image_format": "png",
            "has_alpha": False,
        }
    has_alpha = _alpha_requested(spec)
    return {
        "version": REMOTION_MEDIA_CONTRACT_VERSION,
        "fidelity": fidelity,
        "filename": "visual.mov",
        "container": "mov",
        "codec": "prores",
        "pixel_format": "yuva444p10le" if has_alpha else "yuv422p10le",
        "encoded_pixel_format": "yuva444p12le" if has_alpha else "yuv422p10le",
        "prores_profile": "4444" if has_alpha else "hq",
        "color_space": "bt709",
        "color_primaries": "bt709",
        "color_transfer": "bt709",
        "color_range": "tv",
        "image_format": "png",
        "has_alpha": has_alpha,
    }


def _media_contract_issues(
    metadata: dict[str, Any],
    policy: dict[str, Any],
) -> list[str]:
    issues: list[str] = []
    if str(metadata.get("codec") or "").lower() != str(policy["codec"]).lower():
        issues.append(
            "codec_mismatch:"
            f"expected={policy['codec']} actual={metadata.get('codec')}"
        )
    if str(metadata.get("pix_fmt") or "").lower() != str(
        policy["encoded_pixel_format"]
    ).lower():
        issues.append(
            "pixel_format_mismatch:"
            f"expected={policy['encoded_pixel_format']} "
            f"actual={metadata.get('pix_fmt')}"
        )
    if str(metadata.get("color_space") or "").lower() != str(
        policy["color_space"]
    ).lower():
        issues.append(
            "color_space_mismatch:"
            f"expected={policy['color_space']} actual={metadata.get('color_space')}"
        )
    if policy.get("color_primaries") and str(
        metadata.get("color_primaries") or ""
    ).lower() != str(policy["color_primaries"]).lower():
        issues.append(
            "color_primaries_mismatch:"
            f"expected={policy['color_primaries']} "
            f"actual={metadata.get('color_primaries')}"
        )
    if policy.get("color_transfer") and str(
        metadata.get("color_transfer") or ""
    ).lower() != str(policy["color_transfer"]).lower():
        issues.append(
            "color_transfer_mismatch:"
            f"expected={policy['color_transfer']} "
            f"actual={metadata.get('color_transfer')}"
        )
    if bool(metadata.get("has_alpha")) != bool(policy.get("has_alpha")):
        issues.append(
            "alpha_mismatch:"
            f"expected={bool(policy.get('has_alpha'))} "
            f"actual={bool(metadata.get('has_alpha'))}"
        )
    return issues


def _color_metadata_remux_command(
    source: Path,
    target: Path,
    policy: dict[str, Any],
) -> list[str]:
    return [
        str(config.FFMPEG_PATH),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-map",
        "0",
        "-c",
        "copy",
        "-colorspace",
        str(policy["color_space"]),
        "-color_primaries",
        str(policy["color_primaries"]),
        "-color_trc",
        str(policy["color_transfer"]),
        "-color_range",
        str(policy["color_range"]),
        "-movflags",
        "+write_colr",
        "-y",
        str(target),
    ]


def _normalize_color_metadata(
    output_path: Path,
    *,
    policy: dict[str, Any],
    log_path: Path,
) -> None:
    temporary_path = output_path.with_name(
        f"{output_path.stem}.color-normalized{output_path.suffix}"
    )
    temporary_path.unlink(missing_ok=True)
    command = _color_metadata_remux_command(
        output_path,
        temporary_path,
        policy,
    )
    try:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(
                30,
                int(getattr(config, "FFMPEG_RENDER_TIMEOUT_SEC", 120) or 120),
            ),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        temporary_path.unlink(missing_ok=True)
        raise VisualRendererError(
            f"Could not normalize Remotion color metadata: {exc}"
        ) from exc
    _write_command_log(log_path, command, result)
    if result.returncode != 0 or not temporary_path.is_file():
        temporary_path.unlink(missing_ok=True)
        detail = (result.stderr or result.stdout or "").strip()
        raise VisualRendererError(
            "Could not normalize Remotion color metadata: "
            + (detail or "FFmpeg produced no output")
        )
    temporary_path.replace(output_path)


def _remotion_bundle_cache_dir() -> Path:
    return data_dir() / "renderers" / "remotion" / "bundles-v2"


def _write_command_log(path: Path, command: list[str], result: subprocess.CompletedProcess[str]) -> None:
    path.write_text(
        "\n".join(
            [
                "$ " + " ".join(command),
                "",
                f"exit_code={result.returncode}",
                "",
                "[stdout]",
                result.stdout or "",
                "",
                "[stderr]",
                result.stderr or "",
            ]
        ),
        encoding="utf-8",
    )


def _report_signature(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _candidate_preflight(
    spec: dict[str, Any],
    *,
    job_dir: Path,
    node_path: str,
    node_root: Path,
    width: int,
    height: int,
    fps: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates = [
        dict(item)
        for item in spec.get("open_visual_program_candidates") or []
        if isinstance(item, dict)
    ]
    if _render_fidelity(spec) != "final" or len(candidates) < 2:
        return dict(spec), {
            "version": "vex-remotion-rendered-preflight-v1",
            "enabled": False,
            "reason": (
                "preview_render"
                if _render_fidelity(spec) != "final"
                else "insufficient_candidates"
            ),
        }

    preflight_dir = job_dir / "candidate_preflight"
    preflight_dir.mkdir(parents=True, exist_ok=True)
    entry_template = _entry_template_path()
    runtime_template = _scene_graph_runtime_path()
    if not entry_template.is_file() or not runtime_template.is_file():
        return dict(spec), {
            "version": "vex-remotion-rendered-preflight-v1",
            "enabled": False,
            "reason": "runtime_templates_unavailable",
        }
    (preflight_dir / "entry.jsx").write_bytes(entry_template.read_bytes())
    (preflight_dir / "remotion_scene_graph.jsx").write_bytes(
        runtime_template.read_bytes()
    )

    batch: list[dict[str, Any]] = []
    programs_by_candidate: dict[str, dict[str, Any]] = {}
    structural_reports_by_candidate: dict[str, dict[str, Any]] = {}
    source_candidates: dict[str, dict[str, Any]] = {}
    rejected: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates[:8]):
        candidate_id = _safe_scene_name(
            str(candidate.get("program_id") or f"candidate_{index + 1:02d}")
        )
        if candidate_id in programs_by_candidate:
            candidate_id = f"{candidate_id}_{index + 1:02d}"
        candidate_spec = {
            **dict(spec),
            "open_visual_program": candidate,
            "open_visual_program_candidates": [candidate],
            "open_visual_tournament": {},
        }
        compilation = compile_remotion_scene_program(
            candidate_spec,
            width=width,
            height=height,
            fps=fps,
        )
        if not compilation.passed or compilation.program is None:
            rejected.append(
                {
                    "candidate_id": candidate_id,
                    "program_id": str(candidate.get("program_id") or ""),
                    "errors": list(compilation.errors[:6]),
                }
            )
            continue
        program = compilation.program.to_dict()
        compiled_program_id = str(
            (program.get("open_visual_program") or {}).get("program_id") or ""
        )
        requested_program_id = str(candidate.get("program_id") or "")
        if compiled_program_id != requested_program_id:
            rejected.append(
                {
                    "candidate_id": candidate_id,
                    "program_id": requested_program_id,
                    "errors": [
                        "candidate_program_was_rejected_or_substituted",
                        *list(compilation.warnings[:5]),
                    ],
                }
            )
            continue
        structural_qa = evaluate_remotion_structure(program)
        if not structural_qa.passed:
            rejected.append(
                {
                    "candidate_id": candidate_id,
                    "program_id": requested_program_id,
                    "errors": list(structural_qa.issues[:6]),
                }
            )
            continue
        programs_by_candidate[candidate_id] = program
        structural_reports_by_candidate[candidate_id] = structural_qa.to_dict()
        source_candidates[candidate_id] = candidate
        batch.append(
            {
                "candidate_id": candidate_id,
                "input_props": {
                    "program": program,
                    "compositionId": REMOTION_COMPOSITION_ID,
                    "transparent": _alpha_requested(spec),
                },
            }
        )
    if len(batch) < 2:
        return dict(spec), {
            "version": "vex-remotion-rendered-preflight-v1",
            "enabled": False,
            "reason": "insufficient_valid_candidates",
            "rejected": rejected,
        }

    input_props_path = preflight_dir / "input_props.json"
    batch_path = preflight_dir / "candidate_input_props.json"
    request_path = preflight_dir / "render_request.json"
    result_path = preflight_dir / "remotion_result.json"
    log_path = preflight_dir / "remotion_render.log"
    report_path = preflight_dir / "remotion_candidate_preflight.json"
    input_props_path.write_text(
        json.dumps(batch[0]["input_props"], indent=2),
        encoding="utf-8",
    )
    batch_path.write_text(json.dumps(batch, indent=2), encoding="utf-8")
    request_path.write_text(
        json.dumps(
            {
                "composition_id": REMOTION_COMPOSITION_ID,
                "render_mode": "stills",
                "candidate_input_props_file": batch_path.name,
                "sample_fractions": [0.08, 0.42, 0.82],
                "timeout_sec": _remotion_timeout_sec(),
                "concurrency": _remotion_concurrency(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    command = [node_path, str(_runner_path()), str(preflight_dir)]
    env = os.environ.copy()
    env["VEX_REMOTION_NODE_ROOT"] = str(node_root)
    env["VEX_REMOTION_TIMEOUT_MS"] = str(_remotion_timeout_ms())
    env["VEX_REMOTION_BUNDLE_CACHE_DIR"] = str(_remotion_bundle_cache_dir())
    if _remotion_concurrency():
        env["VEX_REMOTION_CONCURRENCY"] = _remotion_concurrency()
    try:
        result = subprocess.run(
            command,
            cwd=str(node_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=max(_remotion_timeout_sec() or 120, 120) + 45,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return dict(spec), {
            "version": "vex-remotion-rendered-preflight-v1",
            "enabled": False,
            "reason": "preflight_process_failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
    _write_command_log(log_path, command, result)
    if result.returncode != 0 or not result_path.is_file():
        return dict(spec), {
            "version": "vex-remotion-rendered-preflight-v1",
            "enabled": False,
            "reason": "preflight_render_failed",
            "error": (result.stderr or result.stdout or "").strip()[-2000:],
        }
    try:
        render_result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return dict(spec), {
            "version": "vex-remotion-rendered-preflight-v1",
            "enabled": False,
            "reason": "preflight_result_invalid",
            "error": f"{type(exc).__name__}: {exc}",
        }

    records: list[dict[str, Any]] = []
    for item in render_result.get("candidate_stills") or []:
        if not isinstance(item, dict):
            continue
        candidate_id = str(item.get("candidate_id") or "")
        program = programs_by_candidate.get(candidate_id)
        if program is None:
            continue
        frame_paths = [
            Path(path)
            for path in item.get("frame_paths") or []
            if Path(path).is_file()
        ]
        aesthetic = evaluate_frame_aesthetics(
            frame_paths,
            dict(program.get("creative_direction") or {}),
        )
        semantic_score = max(
            0.0,
            min(float(program.get("semantic_score") or 0.0), 1.0),
        )
        structural_report = structural_reports_by_candidate[candidate_id]
        structural_score = max(
            0.0,
            min(float(structural_report.get("score") or 0.0), 1.0),
        )
        rendered_score = (
            aesthetic.score * 0.68
            + semantic_score * 0.16
            + structural_score * 0.16
        )
        records.append(
            {
                "candidate_id": candidate_id,
                "program_id": str(
                    source_candidates[candidate_id].get("program_id") or ""
                ),
                "eligible": bool(aesthetic.passed and len(frame_paths) >= 3),
                "score": round(rendered_score, 4),
                "semantic_score": round(semantic_score, 4),
                "structural_qa": structural_report,
                "aesthetic": aesthetic.to_dict(),
                "frame_paths": [str(path) for path in frame_paths],
            }
        )
    eligible = [item for item in records if bool(item.get("eligible"))]
    eligible.sort(
        key=lambda item: (
            float(item.get("score") or 0.0),
            str(item.get("program_id") or ""),
        ),
        reverse=True,
    )
    if not eligible:
        unsigned_report = {
            "version": "vex-remotion-rendered-preflight-v1",
            "enabled": True,
            "passed": False,
            "reason": "no_rendered_candidate_passed",
            "selected_program_id": str(
                (spec.get("open_visual_program") or {}).get("program_id") or ""
            ),
            "candidates": records,
            "rejected": rejected,
            "bundle_fingerprint": str(
                render_result.get("bundle_fingerprint") or ""
            ),
            "bundle_cache_hit": bool(render_result.get("bundle_cache_hit")),
        }
        report = {
            **unsigned_report,
            "signature": _report_signature(unsigned_report),
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return dict(spec), report

    selected = eligible[0]
    selected_candidate = source_candidates[str(selected["candidate_id"])]
    selected_spec = {
        **dict(spec),
        "open_visual_program": selected_candidate,
        "open_visual_program_candidates": [selected_candidate],
        "open_visual_tournament": {},
    }
    unsigned_report = {
        "version": "vex-remotion-rendered-preflight-v1",
        "enabled": True,
        "passed": True,
        "selection_mode": "rendered_contact_sheet",
        "selected_program_id": str(selected.get("program_id") or ""),
        "requested_candidate_count": len(candidates[:8]),
        "rendered_candidate_count": len(records),
        "sample_fractions": [0.08, 0.42, 0.82],
        "candidates": records,
        "rejected": rejected,
        "bundle_fingerprint": str(render_result.get("bundle_fingerprint") or ""),
        "bundle_cache_hit": bool(render_result.get("bundle_cache_hit")),
    }
    report = {
        **unsigned_report,
        "signature": _report_signature(unsigned_report),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return selected_spec, report


def _candidate_node_roots(node_path: str | None = None) -> list[Path]:
    candidates: list[Path] = []
    managed_runtime = managed_renderer_runtime_dir(node_path)
    for candidate in (managed_runtime, _repo_root()):
        if candidate is None:
            continue
        resolved = candidate.expanduser().resolve(strict=False)
        if resolved not in candidates:
            candidates.append(resolved)
    return candidates


def _probe_node_packages_at(
    root: Path,
    *,
    node_path: str | None = None,
) -> tuple[bool, str]:
    selected_node = node_path or resolve_node_executable()
    if not selected_node:
        return False, "Node.js is unavailable; check PATH or VEX_NODE_PATH."
    status = renderer_native_runtime_status(
        node_path=selected_node,
        node_root=root,
        require_remotion=True,
    )
    if bool(status.get("available")):
        return True, ""
    return False, str(status.get("reason") or "Remotion native runtime is unavailable.")


def _find_remotion_node_root(
    node_path: str | None = None,
) -> tuple[Path | None, str]:
    selected_node = node_path or resolve_node_executable()
    reasons: list[str] = []
    for candidate in _candidate_node_roots(selected_node):
        ok, reason = _probe_node_packages_at(candidate, node_path=selected_node)
        if ok:
            return candidate, ""
        if reason:
            reasons.append(reason)
    identity = node_runtime_identity(selected_node)
    identity_detail = (
        f" for {identity.runtime_key}" if identity is not None else ""
    )
    detail = "; ".join(reasons[:3])
    return (
        None,
        (
            "Remotion packages are not installed. Run `npm ci` in a source "
            "checkout or `vex renderers install remotion` for the managed "
            f"renderer runtime{identity_detail}."
            f"{f' Details: {detail}' if detail else ''}"
        ),
    )


def _run_node_package_probe(node_path: str | None = None) -> tuple[bool, str]:
    root, reason = _find_remotion_node_root(node_path)
    return root is not None, reason


def _node_platform_arch() -> tuple[str | None, str | None, str]:
    node_path = resolve_node_executable()
    if not node_path:
        return None, None, "Node.js is unavailable; check PATH or VEX_NODE_PATH."
    platform, arch = node_platform_arch(node_path)
    if not platform or not arch:
        return None, None, "Could not inspect the selected Node.js platform."
    return platform, arch, ""


def _remotion_platform_blocker(platform: str | None, arch: str | None) -> str:
    if not platform or not arch:
        return ""
    if platform == "win32" and arch != "x64":
        return (
            "Remotion 4.0.487 does not ship a Windows "
            f"{arch} compositor package. Use x64 Node/npm on Windows ARM, "
            "run from WSL/Linux, or use another renderer on this machine."
        )
    if platform not in {"win32", "linux", "darwin"}:
        return f"Remotion local rendering is not supported on Node platform {platform!r}."
    return ""


def _build_input_props(
    spec: dict[str, Any],
    *,
    width: int,
    height: int,
    fps: float,
) -> dict[str, Any]:
    compilation = compile_remotion_scene_program(
        spec,
        width=width,
        height=height,
        fps=fps,
    )
    if not compilation.passed or compilation.program is None:
        detail = ", ".join(compilation.errors[:6]) or "unknown semantic compiler failure"
        raise VisualRendererError(f"Remotion scene compilation failed: {detail}")
    return {
        "program": compilation.program.to_dict(),
        "compositionId": REMOTION_COMPOSITION_ID,
    }


def _template_family(template: str, intent_type: str) -> str:
    value = f"{template} {intent_type}".lower()
    if re.search(r"data|metric|score|risk|proof|stat", value):
        return "data"
    if re.search(r"compare|contrast|decision|myth|problem", value):
        return "contrast"
    if re.search(r"timeline|sequence|narrative|route|journey", value):
        return "timeline"
    if re.search(r"interface|ui", value):
        return "interface"
    if re.search(r"quote|keyword|focus|emphasis", value):
        return "emphasis"
    return "mechanism"


class RemotionRenderer(VisualRenderer):
    name = "remotion"
    supported_templates = SUPPORTED_REMOTION_TEMPLATES

    def availability(self) -> RendererStatus:
        runner = _runner_path()
        if not runner.is_file():
            return RendererStatus(False, "Remotion runner is missing from the Vex installation.")
        node_major = _node_major_version()
        if node_major is None:
            return RendererStatus(
                False,
                "Node.js is unavailable; install Node.js 22+ or set VEX_NODE_PATH.",
            )
        if node_major < 22:
            return RendererStatus(False, f"Node.js {node_major} is too old for Vex's renderer runtime; install Node.js 22+.")
        platform, arch, platform_reason = _node_platform_arch()
        if platform_reason:
            return RendererStatus(False, platform_reason)
        platform_blocker = _remotion_platform_blocker(platform, arch)
        if platform_blocker:
            return RendererStatus(False, platform_blocker)
        packages_ok, package_reason = _run_node_package_probe()
        if not packages_ok:
            return RendererStatus(False, package_reason)
        return RendererStatus(True, "")

    def score_spec(self, spec: dict[str, Any]) -> float:
        if not self.supports(spec):
            return -1.0
        template = str(spec.get("template") or "").strip().lower()
        intent_type = str(spec.get("visual_intent_type") or "").strip().lower()
        renderer_hint = str(spec.get("renderer_hint") or "").strip().lower()
        visual_hint = str(spec.get("visual_type_hint") or "").strip().lower()
        composition = str(spec.get("composition_mode") or "").strip().lower()
        if intent_type == "spatial_3d" and renderer_hint != "remotion":
            return -1.0
        score = 0.92
        if renderer_hint == "remotion":
            score += 0.22
        if template.startswith("semantic_"):
            score += 0.16
        if _template_family(template, intent_type) in {"data", "mechanism", "contrast", "timeline", "interface"}:
            score += 0.18
        if visual_hint in {"product_ui", "process", "abstract_motion", "data_graphic"}:
            score += 0.08
        if composition == "replace":
            score += 0.08
        if intent_type == "math_or_formula" and renderer_hint != "remotion":
            score -= 0.3
        if intent_type == "spatial_3d":
            score -= 0.24
        return round(score, 3)

    def capability_summary(self) -> dict[str, Any]:
        base = super().capability_summary()
        base["render_model"] = "local_remotion_ssr"
        base["package_version"] = REMOTION_PACKAGE_VERSION
        base["composition_id"] = REMOTION_COMPOSITION_ID
        base["scene_program_version"] = "remotion-scene-program-v4"
        base["scene_graph_version"] = "vex-scene-graph-v2"
        base["render_qa_version"] = "remotion-render-qa-v5"
        return base

    def render(
        self,
        spec: dict[str, Any],
        render_root: Path,
        width: int,
        height: int,
        fps: float,
    ) -> RenderedAsset:
        status = self.availability()
        if not status.available:
            raise VisualRendererError(status.reason)

        spec_id = str(spec.get("visual_id") or spec.get("id") or "visual")
        scene_name = _safe_scene_name(spec_id)
        job_dir = safe_render_job_dir(render_root, spec_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        media_contract = _render_output_policy(spec)
        output_path = job_dir / str(media_contract["filename"])
        entry_path = job_dir / "entry.jsx"
        scene_graph_runtime_path = job_dir / "remotion_scene_graph.jsx"
        spec_path = job_dir / "remotion_spec.json"
        input_props_path = job_dir / "input_props.json"
        scene_program_path = job_dir / "remotion_scene_program.json"
        compiler_report_path = job_dir / "remotion_compiler_report.json"
        request_path = job_dir / "render_request.json"
        result_path = job_dir / "remotion_result.json"
        log_path = job_dir / "remotion_render.log"
        metadata_path = job_dir / "remotion_metadata.json"
        structural_qa_path = job_dir / "remotion_structural_qa.json"
        color_metadata_log_path = job_dir / "remotion_color_metadata.log"

        node_path = resolve_node_executable()
        if not node_path:
            raise VisualRendererError("Node.js is unavailable; check PATH or VEX_NODE_PATH.")
        node_root, node_root_reason = _find_remotion_node_root(node_path)
        if node_root is None:
            raise VisualRendererError(node_root_reason)
        render_spec, candidate_preflight = _candidate_preflight(
            spec,
            job_dir=job_dir,
            node_path=node_path,
            node_root=node_root,
            width=width,
            height=height,
            fps=fps,
        )
        compilation = compile_remotion_scene_program(
            render_spec,
            width=width,
            height=height,
            fps=fps,
        )
        if not compilation.passed or compilation.program is None:
            compiler_report_path.write_text(
                json.dumps(compilation.to_dict(), indent=2),
                encoding="utf-8",
            )
            detail = ", ".join(compilation.errors[:6]) or "unknown semantic compiler failure"
            raise VisualRendererError(
                f"Remotion scene compilation failed for {spec_id}: {detail}"
            )
        program = compilation.program.to_dict()
        structural_qa = evaluate_remotion_structure(program)
        structural_qa_path.write_text(
            json.dumps(structural_qa.to_dict(), indent=2),
            encoding="utf-8",
        )
        if not structural_qa.passed:
            detail = ", ".join(structural_qa.issues[:6])
            raise VisualRendererError(
                f"Remotion structural QA failed for {spec_id}: {detail}"
            )
        input_props = {
            "program": program,
            "compositionId": REMOTION_COMPOSITION_ID,
            "transparent": bool(media_contract["has_alpha"]),
        }
        entry_template = _entry_template_path()
        if not entry_template.is_file():
            raise VisualRendererError("Remotion React entry template is missing from the Vex installation.")
        scene_graph_runtime_template = _scene_graph_runtime_path()
        if not scene_graph_runtime_template.is_file():
            raise VisualRendererError(
                "Remotion SceneGraph runtime is missing from the Vex installation."
            )
        entry_path.write_bytes(entry_template.read_bytes())
        scene_graph_runtime_path.write_bytes(scene_graph_runtime_template.read_bytes())
        spec_path.write_text(json.dumps(render_spec, indent=2), encoding="utf-8")
        input_props_path.write_text(json.dumps(input_props, indent=2), encoding="utf-8")
        scene_program_path.write_text(json.dumps(program, indent=2), encoding="utf-8")
        compiler_report_path.write_text(
            json.dumps(compilation.to_dict(), indent=2),
            encoding="utf-8",
        )
        request_path.write_text(
            json.dumps(
                {
                    "composition_id": REMOTION_COMPOSITION_ID,
                    "output_path": str(output_path),
                    "width": width,
                    "height": height,
                    "fps": fps,
                    "timeout_sec": _remotion_timeout_sec(),
                    "concurrency": _remotion_concurrency(),
                    "render_mode": _render_fidelity(spec),
                    "preview_scale": 0.5,
                    "transparent": bool(media_contract["has_alpha"]),
                    "media_contract": media_contract,
                    "candidate_preflight": {
                        "enabled": bool(candidate_preflight.get("enabled")),
                        "selected_program_id": str(
                            candidate_preflight.get("selected_program_id") or ""
                        ),
                        "signature": str(candidate_preflight.get("signature") or ""),
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        request_payload = json.loads(request_path.read_text(encoding="utf-8"))
        request_payload["node_root"] = str(node_root)
        request_path.write_text(json.dumps(request_payload, indent=2), encoding="utf-8")
        command = [node_path, str(_runner_path()), str(job_dir)]
        env = os.environ.copy()
        env["VEX_REMOTION_NODE_ROOT"] = str(node_root)
        env["VEX_REMOTION_TIMEOUT_MS"] = str(_remotion_timeout_ms())
        env["VEX_REMOTION_BUNDLE_CACHE_DIR"] = str(_remotion_bundle_cache_dir())
        if _remotion_concurrency():
            env["VEX_REMOTION_CONCURRENCY"] = _remotion_concurrency()
        process_timeout = _remotion_timeout_sec()
        if process_timeout is not None:
            process_timeout += 45
        try:
            result = subprocess.run(
                command,
                cwd=str(node_root),
                env=env,
                capture_output=True,
                text=True,
                timeout=process_timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise VisualRendererError(f"Remotion render process failed for {spec_id}: {exc}") from exc
        _write_command_log(log_path, command, result)
        if result.returncode != 0 or not output_path.is_file():
            detail = (result.stderr or result.stdout or "").strip()
            if result_path.is_file():
                try:
                    payload = json.loads(result_path.read_text(encoding="utf-8"))
                    detail = str(payload.get("error") or detail)
                except (OSError, json.JSONDecodeError):
                    pass
            raise VisualRendererError(f"Remotion render failed for {spec_id}: {detail}")

        try:
            render_result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            render_result = {"ok": True}
        video_metadata = probe_video(str(output_path))
        color_metadata_normalized = False
        if any(
            media_contract.get(key)
            and str(video_metadata.get(key) or "").lower()
            != str(media_contract[key]).lower()
            for key in ("color_space", "color_primaries", "color_transfer")
        ):
            _normalize_color_metadata(
                output_path,
                policy=media_contract,
                log_path=color_metadata_log_path,
            )
            color_metadata_normalized = True
            video_metadata = probe_video(str(output_path))
        media_contract_issues = _media_contract_issues(
            video_metadata,
            media_contract,
        )
        if media_contract_issues:
            raise VisualRendererError(
                f"Remotion media contract failed for {spec_id}: "
                + ", ".join(media_contract_issues)
            )
        render_qa = evaluate_remotion_render(
            output_path,
            program,
            job_dir=job_dir,
            has_alpha=bool(media_contract["has_alpha"]),
        )
        render_qa_payload = render_qa.to_dict()
        quality_score = round(
            render_qa.score * 0.82 + structural_qa.score * 0.18,
            4,
        )
        quality_passed = bool(render_qa.passed and structural_qa.passed)
        metadata = {
            **video_metadata,
            "renderer": self.name,
            "render_pipeline": "remotion_ssr_local",
            "render_fidelity": _render_fidelity(spec),
            "media_contract": media_contract,
            "color_metadata_normalized": color_metadata_normalized,
            "has_alpha": bool(media_contract["has_alpha"]),
            "candidate_preflight": candidate_preflight,
            "remotion_version": REMOTION_PACKAGE_VERSION,
            "composition_id": REMOTION_COMPOSITION_ID,
            "template": str(render_spec.get("template") or ""),
            "template_family": str(program.get("scene_family") or ""),
            "scene_name": scene_name,
            "quality_score": quality_score,
            "quality_passed": quality_passed,
            "quality_components": {
                "temporal_render": render_qa.score,
                "structural": structural_qa.score,
            },
            "semantic_qa": {
                "passed": compilation.passed,
                "score": program.get("semantic_score"),
                "scene_family": program.get("scene_family"),
                "scene_type": program.get("scene_type"),
                "program_id": program.get("program_id"),
                "program_signature": program.get("signature"),
                "grounding_mode": program.get("grounding_mode"),
                "warnings": compilation.warnings,
            },
            "remotion_scene_program": program,
            "remotion_structural_qa": structural_qa.to_dict(),
            "remotion_render_qa": render_qa_payload,
            "remotion_render": render_result,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        artifact_paths: dict[str, Any] = {
            "entry_path": str(entry_path),
            "scene_graph_runtime_path": str(scene_graph_runtime_path),
            "spec_path": str(spec_path),
            "input_props_path": str(input_props_path),
            "scene_program_path": str(scene_program_path),
            "compiler_report_path": str(compiler_report_path),
            "structural_qa_path": str(structural_qa_path),
            "render_qa_path": str(job_dir / "remotion_qa.json"),
            "render_qa_frame_paths": list(render_qa.frame_paths),
            "request_path": str(request_path),
            "result_path": str(result_path),
            "render_log_path": str(log_path),
            "metadata_path": str(metadata_path),
        }
        candidate_preflight_report_path = (
            job_dir
            / "candidate_preflight"
            / "remotion_candidate_preflight.json"
        )
        if candidate_preflight_report_path.is_file():
            artifact_paths["candidate_preflight_report_path"] = str(
                candidate_preflight_report_path
            )
        if color_metadata_log_path.is_file():
            artifact_paths["color_metadata_log_path"] = str(
                color_metadata_log_path
            )
        return RenderedAsset(
            asset_path=str(output_path),
            width=int(video_metadata.get("width") or width),
            height=int(video_metadata.get("height") or height),
            duration_sec=float(video_metadata.get("duration_sec") or program["duration_sec"]),
            renderer=self.name,
            job_dir=str(job_dir),
            script_path=str(entry_path),
            artifact_paths=artifact_paths,
            metadata=metadata,
        )
