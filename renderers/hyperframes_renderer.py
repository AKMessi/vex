from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import config
from engine import probe_video
from renderers.base import RenderedAsset, RendererStatus, VisualRenderer, VisualRendererError, safe_render_job_dir
from vex_hyperframes import build_composition, validate_composition_html
from vex_hyperframes.qa import analyze_hyperframes_quality, extract_quality_frames, write_quality_report
from vex_hyperframes.variants import HyperframesVariant, build_variants, select_best_variant


def _safe_scene_name(spec_id: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(spec_id or "visual")).strip("_")
    return cleaned or "auto_visual"


def _node_major_version() -> int | None:
    node_path = shutil.which("node")
    if not node_path:
        return None
    result = subprocess.run([node_path, "--version"], capture_output=True, text=True)
    if result.returncode != 0:
        return None
    match = re.search(r"v?(\d+)", result.stdout.strip())
    return int(match.group(1)) if match else None


def _local_bin_name(name: str) -> str:
    return f"{name}.cmd" if os.name == "nt" else name


def _hyperframes_cli_path() -> str | None:
    configured = str(getattr(config, "HYPERFRAMES_CLI_PATH", "hyperframes") or "hyperframes").strip()
    configured_path = Path(configured)
    if configured_path.is_absolute() or configured_path.parent != Path("."):
        return str(configured_path) if configured_path.is_file() else None

    repo_root = Path(__file__).resolve().parent.parent
    binary_name = _local_bin_name(configured)
    for candidate in (
        repo_root / "node_modules" / ".bin" / binary_name,
        Path.cwd() / "node_modules" / ".bin" / binary_name,
    ):
        if candidate.is_file():
            return str(candidate)
    return None


def _hyperframes_command(*args: str) -> list[str]:
    cli_path = _hyperframes_cli_path()
    if not cli_path:
        raise VisualRendererError(
            "Hyperframes CLI is not installed locally. Run `npm ci` or set HYPERFRAMES_CLI_PATH."
        )
    return [cli_path, *args]


def _hyperframes_render_timeout_sec() -> int | None:
    try:
        timeout = int(getattr(config, "HYPERFRAMES_RENDER_TIMEOUT_SEC", 0))
    except (TypeError, ValueError):
        timeout = 0
    if timeout <= 0:
        return None
    return max(30, timeout)


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


class HyperframesRenderer(VisualRenderer):
    name = "hyperframes"
    supported_templates = {
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

    def availability(self) -> RendererStatus:
        if _hyperframes_cli_path() is None:
            return RendererStatus(False, "Hyperframes CLI is not installed locally. Run `npm ci` or set HYPERFRAMES_CLI_PATH.")
        node_major = _node_major_version()
        if node_major is None:
            return RendererStatus(False, "Node.js is not available in PATH; Hyperframes requires Node.js 22+.")
        if node_major < 22:
            return RendererStatus(False, f"Node.js {node_major} is too old; Hyperframes requires Node.js 22+.")
        if shutil.which(config.FFMPEG_PATH) is None:
            return RendererStatus(False, "FFmpeg is not available in PATH; Hyperframes needs it for MP4 encoding.")
        return RendererStatus(True, "")

    def score_spec(self, spec: dict[str, Any]) -> float:
        if not self.supports(spec):
            return -1.0
        template = str(spec.get("template") or "").strip().lower()
        visual_hint = str(spec.get("visual_type_hint") or "").strip().lower()
        composition = str(spec.get("composition_mode") or "").strip().lower()
        style_pack = str(spec.get("style_pack") or "").strip().lower()
        importance = float(spec.get("importance") or 0.5)
        score = 1.02
        if template in {
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
        }:
            score += 0.18
        if template in {"metric_callout", "stat_grid", "timeline_steps", "system_flow", "comparison_split", "quote_focus", "keyword_stack"}:
            score += 0.1
        if visual_hint in {"product_ui", "process", "abstract_motion", "data_graphic"}:
            score += 0.16
        if composition == "replace":
            score += 0.14
        if composition == "picture_in_picture":
            score += 0.04
        if style_pack in {"product_ui", "bold_tech", "signal_lab", "cinematic_night"}:
            score += 0.04
        if re.search(r"\b(?:latex|mathtex|formula|equation|proof|theorem)\b", f"{spec.get('sentence_text', '')} {spec.get('context_text', '')}", flags=re.IGNORECASE):
            score -= 0.22
        score += importance * 0.05
        return round(score, 3)

    def _render_variant(
        self,
        variant: HyperframesVariant,
        *,
        job_dir: Path,
        width: int,
        height: int,
        fps: float,
    ) -> dict[str, Any]:
        variant_dir = (job_dir / "variants" / variant.variant_id).resolve()
        variant_dir.mkdir(parents=True, exist_ok=True)
        spec = variant.spec
        spec_id = str(spec.get("visual_id") or spec.get("id") or "visual")
        scene_name = f"{_safe_scene_name(spec_id)}_{variant.variant_id}"
        composition = build_composition(spec, width=width, height=height, fps=fps)
        index_path = variant_dir / "index.html"
        output_path = variant_dir / f"{scene_name}.mp4"
        metadata_path = variant_dir / "hyperframes_metadata.json"
        validation_path = variant_dir / "hyperframes_validation.json"
        lint_log_path = variant_dir / "hyperframes_lint.log"
        render_log_path = variant_dir / "hyperframes_render.log"
        quality_report_path = variant_dir / "hyperframes_quality.json"
        spec_path = variant_dir / "hyperframes_spec.json"

        index_path.write_text(composition.html, encoding="utf-8")
        spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        validation = validate_composition_html(
            composition.html,
            expected_width=width,
            expected_height=height,
            expected_duration=float(spec.get("duration") or composition.metadata["duration_sec"]),
        )
        validation_path.write_text(json.dumps(validation.to_dict(), indent=2), encoding="utf-8")
        if not validation.valid:
            raise VisualRendererError("Hyperframes composition validation failed: " + "; ".join(validation.errors))

        lint_command = _hyperframes_command("lint", "--json", ".")
        lint_result = subprocess.run(
            lint_command,
            cwd=str(variant_dir),
            capture_output=True,
            text=True,
            timeout=config.HYPERFRAMES_LINT_TIMEOUT_SEC,
        )
        _write_command_log(lint_log_path, lint_command, lint_result)
        if lint_result.returncode != 0:
            detail = (lint_result.stderr or lint_result.stdout or "").strip()
            raise VisualRendererError(f"Hyperframes lint failed for {spec_id}/{variant.variant_id}: {detail}")

        render_command = _hyperframes_command(
            "render",
            "--output",
            str(output_path),
            "--fps",
            str(max(15, int(round(fps or 30.0)))),
            ".",
        )
        quality = str(config.HYPERFRAMES_RENDER_QUALITY or "").strip()
        if quality:
            render_command.extend(["--quality", quality])
        render_result = subprocess.run(
            render_command,
            cwd=str(variant_dir),
            capture_output=True,
            text=True,
            timeout=_hyperframes_render_timeout_sec(),
        )
        _write_command_log(render_log_path, render_command, render_result)
        if render_result.returncode != 0 or not output_path.is_file():
            detail = (render_result.stderr or render_result.stdout or "").strip()
            raise VisualRendererError(f"Hyperframes render failed for {spec_id}/{variant.variant_id}: {detail}")

        video_metadata = probe_video(str(output_path))
        frame_paths = extract_quality_frames(
            output_path,
            variant_dir / "qa_frames",
            duration_sec=float(video_metadata.get("duration_sec") or composition.metadata["duration_sec"]),
            frame_count=3,
        )
        qa_report = analyze_hyperframes_quality(
            video_path=output_path,
            html=composition.html,
            frame_paths=frame_paths,
            theme=dict((composition.metadata.get("art_direction") or {}).get("theme") or {}),
            design_ir=dict(composition.metadata.get("design_ir") or {}),
            min_score=float(config.HYPERFRAMES_MIN_QUALITY_SCORE),
        )
        write_quality_report(quality_report_path, qa_report)
        metadata = {
            **composition.metadata,
            **video_metadata,
            "scene_generation_mode": "deterministic_hyperframes",
            "validation": validation.to_dict(),
            "quality": qa_report.to_dict(),
            "hyperframes_cli_path": str(_hyperframes_cli_path() or ""),
            "variant_id": variant.variant_id,
            "variant_index": variant.variant_index,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        artifact_paths = {
            "index_html_path": str(index_path),
            "spec_path": str(spec_path),
            "metadata_path": str(metadata_path),
            "validation_path": str(validation_path),
            "quality_report_path": str(quality_report_path),
            "lint_log_path": str(lint_log_path),
            "render_log_path": str(render_log_path),
            "qa_frames_dir": str(variant_dir / "qa_frames"),
            "qa_frame_paths": [str(path) for path in frame_paths],
        }
        return {
            "variant_id": variant.variant_id,
            "variant_index": variant.variant_index,
            "asset_path": str(output_path),
            "script_path": str(index_path),
            "job_dir": str(variant_dir),
            "artifact_paths": artifact_paths,
            "metadata": metadata,
            "qa": qa_report.to_dict(),
        }

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
        if not self.supports(spec):
            raise VisualRendererError(f"Hyperframes renderer does not support template {spec.get('template')!r}.")

        spec_id = str(spec.get("visual_id") or spec.get("id") or "visual")
        scene_name = _safe_scene_name(spec_id)
        job_dir = safe_render_job_dir(render_root, spec_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        output_path = job_dir / f"{scene_name}.mp4"
        metadata_path = job_dir / "hyperframes_metadata.json"
        variants_report_path = job_dir / "hyperframes_variants.json"
        variants = build_variants(spec, default_count=int(config.HYPERFRAMES_VARIANT_COUNT))
        variant_records: list[dict[str, Any]] = []
        for variant in variants:
            try:
                variant_records.append(
                    self._render_variant(
                        variant,
                        job_dir=job_dir,
                        width=width,
                        height=height,
                        fps=fps,
                    )
                )
            except VisualRendererError as exc:
                variant_records.append(
                    {
                        "variant_id": variant.variant_id,
                        "variant_index": variant.variant_index,
                        "render_error": str(exc),
                        "spec": variant.spec,
                    }
                )
        selected = select_best_variant(variant_records)
        variants_report_path.write_text(
            json.dumps(
                {
                    "selected_variant_id": selected.get("variant_id") if selected else None,
                    "min_quality_score": config.HYPERFRAMES_MIN_QUALITY_SCORE,
                    "qa_mode": config.HYPERFRAMES_QA_MODE,
                    "vision_qa_enabled": bool(config.HYPERFRAMES_ENABLE_VISION_QA),
                    "variant_count": len(variants),
                    "variants": variant_records,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if selected is None:
            details = "; ".join(str(item.get("render_error") or "unknown failure") for item in variant_records[:3])
            raise VisualRendererError(f"Hyperframes could not render any visual variants for {spec_id}. {details}")
        shutil.copyfile(str(selected["asset_path"]), output_path)

        video_metadata = probe_video(str(output_path))
        metadata = {
            **dict(selected.get("metadata") or {}),
            **video_metadata,
            "selected_variant_id": selected.get("variant_id"),
            "variant_selection": {
                "selected_variant_id": selected.get("variant_id"),
                "selected_quality_score": (selected.get("qa") or {}).get("score"),
                "selected_quality_passed": bool((selected.get("qa") or {}).get("passed")),
                "variant_count": len(variants),
            },
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        artifact_paths = dict(selected.get("artifact_paths") or {})
        variant_metadata_path = artifact_paths.get("metadata_path")
        if variant_metadata_path:
            artifact_paths["variant_metadata_path"] = str(variant_metadata_path)
        artifact_paths["metadata_path"] = str(metadata_path)
        artifact_paths["variants_report_path"] = str(variants_report_path)
        return RenderedAsset(
            asset_path=str(output_path),
            width=int(video_metadata.get("width") or width),
            height=int(video_metadata.get("height") or height),
            duration_sec=float(video_metadata.get("duration_sec") or metadata.get("duration_sec") or 0.0),
            renderer=self.name,
            job_dir=str(job_dir),
            script_path=str(selected.get("script_path") or ""),
            artifact_paths=artifact_paths,
            metadata=metadata,
        )
