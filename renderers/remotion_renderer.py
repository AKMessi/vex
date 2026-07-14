from __future__ import annotations

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
from vex_remotion import compile_remotion_scene_program, evaluate_remotion_render


REMOTION_COMPOSITION_ID = "VexAutoVisual"
REMOTION_PACKAGE_VERSION = "4.0.487"

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
        base["scene_program_version"] = "remotion-scene-program-v3"
        base["render_qa_version"] = "remotion-render-qa-v4"
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
        output_path = job_dir / "visual.mp4"
        entry_path = job_dir / "entry.jsx"
        spec_path = job_dir / "remotion_spec.json"
        input_props_path = job_dir / "input_props.json"
        scene_program_path = job_dir / "remotion_scene_program.json"
        compiler_report_path = job_dir / "remotion_compiler_report.json"
        request_path = job_dir / "render_request.json"
        result_path = job_dir / "remotion_result.json"
        log_path = job_dir / "remotion_render.log"
        metadata_path = job_dir / "remotion_metadata.json"

        compilation = compile_remotion_scene_program(
            spec,
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
        input_props = {
            "program": program,
            "compositionId": REMOTION_COMPOSITION_ID,
        }
        entry_template = _entry_template_path()
        if not entry_template.is_file():
            raise VisualRendererError("Remotion React entry template is missing from the Vex installation.")
        entry_path.write_bytes(entry_template.read_bytes())
        spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
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
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        node_path = resolve_node_executable()
        if not node_path:
            raise VisualRendererError("Node.js is unavailable; check PATH or VEX_NODE_PATH.")
        node_root, node_root_reason = _find_remotion_node_root(node_path)
        if node_root is None:
            raise VisualRendererError(node_root_reason)
        request_payload = json.loads(request_path.read_text(encoding="utf-8"))
        request_payload["node_root"] = str(node_root)
        request_path.write_text(json.dumps(request_payload, indent=2), encoding="utf-8")
        command = [node_path, str(_runner_path()), str(job_dir)]
        env = os.environ.copy()
        env["VEX_REMOTION_NODE_ROOT"] = str(node_root)
        env["VEX_REMOTION_TIMEOUT_MS"] = str(_remotion_timeout_ms())
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
        render_qa = evaluate_remotion_render(
            output_path,
            program,
            job_dir=job_dir,
        )
        render_qa_payload = render_qa.to_dict()
        metadata = {
            **video_metadata,
            "renderer": self.name,
            "render_pipeline": "remotion_ssr_local",
            "remotion_version": REMOTION_PACKAGE_VERSION,
            "composition_id": REMOTION_COMPOSITION_ID,
            "template": str(spec.get("template") or ""),
            "template_family": str(program.get("scene_family") or ""),
            "scene_name": scene_name,
            "quality_score": render_qa.score,
            "quality_passed": render_qa.passed,
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
            "remotion_render_qa": render_qa_payload,
            "remotion_render": render_result,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return RenderedAsset(
            asset_path=str(output_path),
            width=int(video_metadata.get("width") or width),
            height=int(video_metadata.get("height") or height),
            duration_sec=float(video_metadata.get("duration_sec") or program["duration_sec"]),
            renderer=self.name,
            job_dir=str(job_dir),
            script_path=str(entry_path),
            artifact_paths={
                "entry_path": str(entry_path),
                "spec_path": str(spec_path),
                "input_props_path": str(input_props_path),
                "scene_program_path": str(scene_program_path),
                "compiler_report_path": str(compiler_report_path),
                "render_qa_path": str(job_dir / "remotion_qa.json"),
                "render_qa_frame_paths": list(render_qa.frame_paths),
                "request_path": str(request_path),
                "result_path": str(result_path),
                "render_log_path": str(log_path),
                "metadata_path": str(metadata_path),
            },
            metadata=metadata,
        )
