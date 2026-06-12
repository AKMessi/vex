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
from vex_hyperframes.authoring import build_bespoke_program
from vex_hyperframes.capture import (
    build_adaptive_capture_plan,
    build_render_trace,
    write_frame_contact_sheet,
)
from vex_hyperframes.critics import run_visual_critics
from vex_hyperframes.final_judge import judge_final_candidate
from vex_hyperframes.patches import (
    apply_visual_patch_set,
    plan_visual_patches,
)
from vex_hyperframes.qa import analyze_hyperframes_quality, extract_quality_frames, write_quality_report
from vex_hyperframes.repair_loop import assess_monotonic_improvement
from vex_hyperframes.semantic_qa import analyze_hyperframes_semantics
from vex_hyperframes.variants import HyperframesVariant, build_variants, select_best_variant
from vex_hyperframes.vision_qa import critique_hyperframes_frames
from vex_runtime.hyperframes import node_major_version
from vex_runtime.paths import managed_hyperframes_cli_path


def _safe_scene_name(spec_id: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(spec_id or "visual")).strip("_")
    return cleaned or "auto_visual"


def _node_major_version() -> int | None:
    return node_major_version()


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
        managed_hyperframes_cli_path(),
    ):
        if candidate.is_file():
            return str(candidate)
    return None


def _hyperframes_command(*args: str) -> list[str]:
    cli_path = _hyperframes_cli_path()
    if not cli_path:
        raise VisualRendererError(
            "HyperFrames CLI is unavailable. Run `vex renderers install hyperframes` "
            "or set HYPERFRAMES_CLI_PATH."
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


_BOUNDED_REPAIR_ACTIONS = {
    "repair_final_hold",
    "repair_grounded_copy_placement",
    "repair_object_coverage",
    "repair_semantic_motion",
}


def _build_bounded_repair_variant(
    variant: HyperframesVariant,
    repair_action: str,
) -> HyperframesVariant:
    action = str(repair_action or "").strip()
    if action not in _BOUNDED_REPAIR_ACTIONS:
        return variant
    spec = dict(variant.spec)
    ir = dict(spec.get("visual_explanation_ir") or {})
    blueprint_id = str(spec.get("semantic_blueprint_id") or "")
    if not ir or not blueprint_id:
        return variant
    spec["bespoke_scene_program"] = build_bespoke_program(
        ir,
        blueprint_id=blueprint_id,
        variant_index=variant.variant_index,
    ).to_dict()
    spec["hyperframes_repair_action"] = action
    spec["hyperframes_repair_attempt"] = variant.variant_index
    try:
        importance = float(spec.get("importance") or 0.5)
    except (TypeError, ValueError):
        importance = 0.5
    spec["importance"] = max(importance, 0.95)
    repair_variant_id = f"{variant.variant_id}_repair"
    spec["hyperframes_variant_id"] = repair_variant_id
    return HyperframesVariant(
        variant_id=repair_variant_id,
        variant_index=variant.variant_index,
        spec=spec,
    )


class HyperframesRenderer(VisualRenderer):
    name = "hyperframes"
    supported_templates = {
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
        if (
            spec.get("hyperframes_automatic_semantic_route")
            and not template.startswith("semantic_")
        ):
            return -1.0
        visual_hint = str(spec.get("visual_type_hint") or "").strip().lower()
        composition = str(spec.get("composition_mode") or "").strip().lower()
        style_pack = str(spec.get("style_pack") or "").strip().lower()
        importance = float(spec.get("importance") or 0.5)
        score = 1.02
        if template.startswith("semantic_"):
            score += 0.42
            if spec.get("semantic_blueprint_id") and spec.get("hyperframes_production_contract"):
                score += 0.18
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
        semantic_report_path = variant_dir / "hyperframes_semantic_qa.json"
        vision_report_path = variant_dir / "hyperframes_vision_qa.json"
        blind_critic_path = variant_dir / "blind_critic.json"
        grounded_critic_path = variant_dir / "grounded_critic.json"
        design_critic_path = variant_dir / "design_critic.json"
        counterexamples_path = variant_dir / "counterexamples.json"
        contact_sheet_path = variant_dir / "frame_contact_sheet.png"
        spec_path = variant_dir / "hyperframes_spec.json"
        bespoke_program_path = variant_dir / "hyperframes_scene_program.json"
        scene_program_v2_path = variant_dir / "scene_program_v2.json"
        render_trace_path = variant_dir / "render_trace.json"

        index_path.write_text(composition.html, encoding="utf-8")
        spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        if spec.get("bespoke_scene_program"):
            bespoke_program_path.write_text(
                json.dumps(spec["bespoke_scene_program"], indent=2),
                encoding="utf-8",
            )
        if spec.get("scene_program_v2"):
            scene_program_v2_path.write_text(
                json.dumps(spec["scene_program_v2"], indent=2),
                encoding="utf-8",
            )
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
        scene_program_v2 = dict(
            composition.metadata.get("scene_program_v2") or {}
        )
        capture_plan = build_adaptive_capture_plan(
            storyboard=list(
                composition.metadata.get("hyperframes_storyboard") or []
            ),
            scene_program=scene_program_v2,
            max_frames=8,
        )
        frame_paths = extract_quality_frames(
            output_path,
            variant_dir / "qa_frames",
            duration_sec=float(video_metadata.get("duration_sec") or composition.metadata["duration_sec"]),
            frame_count=4,
            capture_plan=[item.to_dict() for item in capture_plan],
        )
        contact_sheet = write_frame_contact_sheet(
            frame_paths,
            contact_sheet_path,
        )
        render_trace = {}
        if scene_program_v2:
            render_trace = build_render_trace(
                scene_program=scene_program_v2,
                capture_plan=capture_plan,
                frame_paths=frame_paths,
                duration_sec=float(
                    video_metadata.get("duration_sec")
                    or composition.metadata["duration_sec"]
                ),
            )
            render_trace_path.write_text(
                json.dumps(render_trace, indent=2),
                encoding="utf-8",
            )
        production_contract = dict(
            composition.metadata.get("hyperframes_production_contract") or {}
        )
        vision_report = None
        semantic_report = None
        if production_contract:
            vision_report_obj = critique_hyperframes_frames(
                frame_paths,
                production_contract=production_contract,
                storyboard=list(
                    composition.metadata.get("hyperframes_storyboard") or []
                ),
                proof_encoding=str(
                    composition.metadata.get("proof_encoding") or ""
                ),
            )
            vision_report = vision_report_obj.to_dict()
            vision_report_path.write_text(
                json.dumps(vision_report, indent=2),
                encoding="utf-8",
            )
            semantic_report_obj = analyze_hyperframes_semantics(
                html=composition.html,
                frame_paths=frame_paths,
                production_contract=production_contract,
                visual_explanation_ir=dict(
                    composition.metadata.get("visual_explanation_ir") or {}
                ),
                storyboard=list(
                    composition.metadata.get("hyperframes_storyboard") or []
                ),
                stage_metadata=dict(composition.metadata.get("stage") or {}),
                qa_mode=str(config.HYPERFRAMES_QA_MODE or "hybrid"),
                vision_report=vision_report,
            )
            semantic_report = semantic_report_obj.to_dict()
            semantic_report_path.write_text(
                json.dumps(semantic_report, indent=2),
                encoding="utf-8",
            )
        qa_report = analyze_hyperframes_quality(
            video_path=output_path,
            html=composition.html,
            frame_paths=frame_paths,
            theme=dict((composition.metadata.get("art_direction") or {}).get("theme") or {}),
            design_ir=dict(composition.metadata.get("design_ir") or {}),
            min_score=float(config.HYPERFRAMES_MIN_QUALITY_SCORE),
            vision_report=vision_report,
            semantic_report=semantic_report,
        )
        critic_bundle = None
        if scene_program_v2 and production_contract:
            critic_bundle = run_visual_critics(
                frame_paths,
                production_contract=production_contract,
                visual_explanation_ir=dict(
                    composition.metadata.get("visual_explanation_ir") or {}
                ),
                scene_program=scene_program_v2,
                render_trace=render_trace,
                quality_report=qa_report.to_dict(),
                vision_report=vision_report,
                source_asset_grounding=dict(
                    {
                        **dict(
                            composition.metadata.get(
                                "source_asset_grounding"
                            )
                            or {}
                        ),
                        "embedded": bool(
                            (
                                composition.metadata.get("stage")
                                or {}
                            ).get("source_asset_grounded")
                        ),
                    }
                ),
            )
            blind_critic_path.write_text(
                json.dumps(critic_bundle.blind.to_dict(), indent=2),
                encoding="utf-8",
            )
            grounded_critic_path.write_text(
                json.dumps(critic_bundle.grounded.to_dict(), indent=2),
                encoding="utf-8",
            )
            design_critic_path.write_text(
                json.dumps(critic_bundle.design.to_dict(), indent=2),
                encoding="utf-8",
            )
            counterexamples_path.write_text(
                json.dumps(
                    {
                        "version": critic_bundle.version,
                        "passed": critic_bundle.passed,
                        "score": critic_bundle.score,
                        "hard_failure_count": critic_bundle.hard_failure_count,
                        "counterexamples": [
                            item.to_dict()
                            for item in critic_bundle.counterexamples
                        ],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            if not critic_bundle.passed:
                qa_report.passed = False
                qa_report.issues.append(
                    "Structured visual critics rejected the rendered explanation."
                )
                if qa_report.repair_action in {"", "keep"}:
                    qa_report.repair_action = "visual_cegis_repair"
        write_quality_report(quality_report_path, qa_report)
        metadata = {
            **composition.metadata,
            **video_metadata,
            "scene_generation_mode": (
                "typed_bespoke_hyperframes"
                if spec.get("bespoke_scene_program")
                else (
                    "typed_scene_program_v2"
                    if spec.get("scene_program_v2")
                    else "deterministic_hyperframes"
                )
            ),
            "validation": validation.to_dict(),
            "quality": qa_report.to_dict(),
            "semantic_qa": semantic_report,
            "vision_qa": vision_report,
            "visual_critics": (
                critic_bundle.to_dict() if critic_bundle is not None else None
            ),
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
        if contact_sheet is not None:
            artifact_paths["frame_contact_sheet_path"] = str(contact_sheet)
        if semantic_report is not None:
            artifact_paths["semantic_qa_path"] = str(semantic_report_path)
        if vision_report is not None:
            artifact_paths["vision_qa_path"] = str(vision_report_path)
            counterfactual_artifacts = dict(
                vision_report.get("counterfactual_artifacts") or {}
            )
            relation_frames = list(
                counterfactual_artifacts.get("relation_ablation_frames") or []
            )
            if relation_frames:
                artifact_paths["inverse_decoder_counterfactuals_dir"] = str(
                    Path(relation_frames[0]).parent
                )
        if spec.get("bespoke_scene_program"):
            artifact_paths["scene_program_path"] = str(bespoke_program_path)
        if spec.get("scene_program_v2"):
            artifact_paths["scene_program_v2_path"] = str(scene_program_v2_path)
            artifact_paths["render_trace_path"] = str(render_trace_path)
        if critic_bundle is not None:
            artifact_paths["blind_critic_path"] = str(blind_critic_path)
            artifact_paths["grounded_critic_path"] = str(grounded_critic_path)
            artifact_paths["design_critic_path"] = str(design_critic_path)
            artifact_paths["counterexamples_path"] = str(counterexamples_path)
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
        if (
            spec.get("hyperframes_automatic_semantic_route")
            and not str(spec.get("template") or "").startswith("semantic_")
        ):
            raise VisualRendererError(
                "Automatic HyperFrames visuals must compile to a semantic stage; "
                "legacy templates are manual-only."
            )

        spec_id = str(spec.get("visual_id") or spec.get("id") or "visual")
        scene_name = _safe_scene_name(spec_id)
        job_dir = safe_render_job_dir(render_root, spec_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        output_path = job_dir / f"{scene_name}.mp4"
        metadata_path = job_dir / "hyperframes_metadata.json"
        variants_report_path = job_dir / "hyperframes_variants.json"
        repair_history_path = job_dir / "repair_history.json"
        variants = build_variants(spec, default_count=int(config.HYPERFRAMES_VARIANT_COUNT))
        variant_records: list[dict[str, Any]] = []
        for variant in variants:
            try:
                record = self._render_variant(
                    variant,
                    job_dir=job_dir,
                    width=width,
                    height=height,
                    fps=fps,
                )
                record["candidate_kind"] = "proof_program"
                record["repair_action_applied"] = ""
                variant_records.append(record)
            except VisualRendererError as exc:
                variant_records.append(
                    {
                        "variant_id": variant.variant_id,
                        "variant_index": variant.variant_index,
                        "candidate_kind": "proof_program",
                        "render_error": str(exc),
                        "spec": variant.spec,
                    }
                )
        selected = select_best_variant(variant_records)
        repair_history: list[dict[str, Any]] = []
        if (
            selected is None
            and bool(getattr(config, "HYPERFRAMES_ENABLE_CEGIS", True))
        ):
            repairable = sorted(
                [
                    record
                    for record in variant_records
                    if record.get("asset_path")
                    and (
                        (record.get("metadata") or {}).get("scene_program_v2")
                        or (record.get("spec") or {}).get("scene_program_v2")
                    )
                    and (
                        (
                            (record.get("metadata") or {}).get(
                                "visual_critics"
                            )
                            or {}
                        ).get("counterexamples")
                    )
                ],
                key=lambda record: float(
                    (record.get("qa") or {}).get("score") or 0.0
                ),
                reverse=True,
            )[:2]
            variant_by_id = {item.variant_id: item for item in variants}
            for failed_record in repairable:
                original = variant_by_id.get(
                    str(failed_record.get("variant_id") or "")
                )
                if original is None:
                    continue
                current_record = failed_record
                current_spec = dict(original.spec)
                for round_index in range(
                    1,
                    int(
                        getattr(
                            config,
                            "HYPERFRAMES_MAX_REPAIR_ROUNDS",
                            3,
                        )
                    )
                    + 1,
                ):
                    scene_program = dict(
                        (current_record.get("metadata") or {}).get(
                            "scene_program_v2"
                        )
                        or current_spec.get("scene_program_v2")
                        or {}
                    )
                    critic_payload = dict(
                        (current_record.get("metadata") or {}).get(
                            "visual_critics"
                        )
                        or {}
                    )
                    counterexamples = [
                        item
                        for item in critic_payload.get("counterexamples") or []
                        if isinstance(item, dict)
                    ]
                    if not scene_program or not counterexamples:
                        break
                    from vex_hyperframes.counterexamples import (
                        parse_counterexamples,
                    )

                    parsed_counterexamples = parse_counterexamples(
                        counterexamples,
                        critic="repair",
                        scene_program=scene_program,
                        render_trace={},
                    )
                    patch_set = plan_visual_patches(
                        parsed_counterexamples,
                        scene_program=scene_program,
                        round_index=round_index,
                    )
                    application = apply_visual_patch_set(
                        patch_set,
                        scene_program=scene_program,
                        ir=dict(current_spec.get("visual_explanation_ir") or {}),
                        claim_graph=dict(current_spec.get("visual_claim_graph") or {}),
                    )
                    history_entry: dict[str, Any] = {
                        "source_variant_id": current_record.get("variant_id"),
                        "round_index": round_index,
                        "patch_set": patch_set.to_dict(),
                        "patch_application": application.to_dict(),
                    }
                    if not application.passed:
                        history_entry["accepted"] = False
                        history_entry["reason"] = "patch_validation_failed"
                        repair_history.append(history_entry)
                        break
                    if application.disposition == "reroute":
                        history_entry["accepted"] = False
                        history_entry["reason"] = "renderer_reroute_required"
                        repair_history.append(history_entry)
                        break
                    repair_variant_id = (
                        f"{original.variant_id}_repair_{round_index:02d}"
                    )
                    repaired_spec = {
                        **current_spec,
                        **application.spec_updates,
                        "scene_program_v2": application.scene_program,
                        "hyperframes_variant_id": repair_variant_id,
                        "hyperframes_repair_round": round_index,
                        "hyperframes_patch_id": patch_set.patch_id,
                    }
                    repaired = HyperframesVariant(
                        variant_id=repair_variant_id,
                        variant_index=original.variant_index,
                        spec=repaired_spec,
                    )
                    try:
                        record = self._render_variant(
                            repaired,
                            job_dir=job_dir,
                            width=width,
                            height=height,
                            fps=fps,
                        )
                        decision = assess_monotonic_improvement(
                            current_record,
                            record,
                            min_score_delta=float(
                                getattr(
                                    config,
                                    "HYPERFRAMES_MIN_REPAIR_DELTA",
                                    0.025,
                                )
                            ),
                        )
                        record["candidate_kind"] = "cegis_repair"
                        record["repair_action_applied"] = "typed_visual_patch"
                        record["eligible_for_selection"] = decision.accepted
                        record["monotonic_repair"] = decision.to_dict()
                        variant_dir = Path(str(record.get("job_dir") or ""))
                        if variant_dir.is_dir():
                            (variant_dir / f"patch_round_{round_index:02d}.json").write_text(
                                json.dumps(patch_set.to_dict(), indent=2),
                                encoding="utf-8",
                            )
                            (variant_dir / f"before_after_{round_index:02d}.json").write_text(
                                json.dumps(decision.to_dict(), indent=2),
                                encoding="utf-8",
                            )
                            record.setdefault("artifact_paths", {})[
                                "patch_round_path"
                            ] = str(
                                variant_dir
                                / f"patch_round_{round_index:02d}.json"
                            )
                            record.setdefault("artifact_paths", {})[
                                "before_after_path"
                            ] = str(
                                variant_dir
                                / f"before_after_{round_index:02d}.json"
                            )
                        variant_records.append(record)
                        history_entry["candidate_variant_id"] = repair_variant_id
                        history_entry["monotonic_decision"] = decision.to_dict()
                        history_entry["accepted"] = decision.accepted
                        history_entry["reason"] = decision.reason
                        repair_history.append(history_entry)
                        if not decision.accepted:
                            break
                        current_record = record
                        current_spec = repaired_spec
                        if bool((record.get("qa") or {}).get("passed")):
                            break
                    except VisualRendererError as exc:
                        history_entry["accepted"] = False
                        history_entry["reason"] = "repair_render_failed"
                        history_entry["render_error"] = str(exc)
                        repair_history.append(history_entry)
                        variant_records.append(
                            {
                                "variant_id": repair_variant_id,
                                "variant_index": original.variant_index,
                                "candidate_kind": "cegis_repair",
                                "render_error": str(exc),
                                "spec": repaired_spec,
                                "eligible_for_selection": False,
                            }
                        )
                        break
            selected = select_best_variant(variant_records)
        if selected is None and not spec.get("visual_proof_programs"):
            variant_by_id = {item.variant_id: item for item in variants}
            repairable = sorted(
                [
                    record
                    for record in variant_records
                    if str((record.get("qa") or {}).get("repair_action") or "")
                    in _BOUNDED_REPAIR_ACTIONS
                    and record.get("variant_id") in variant_by_id
                ],
                key=lambda record: float(
                    (record.get("qa") or {}).get("score") or 0.0
                ),
                reverse=True,
            )[:2]
            for failed_record in repairable:
                original = variant_by_id[str(failed_record["variant_id"])]
                repair_action = str(
                    (failed_record.get("qa") or {}).get("repair_action") or ""
                )
                repaired = _build_bounded_repair_variant(original, repair_action)
                try:
                    record = self._render_variant(
                        repaired,
                        job_dir=job_dir,
                        width=width,
                        height=height,
                        fps=fps,
                    )
                    record["candidate_kind"] = "bounded_repair"
                    record["repair_action_applied"] = repair_action
                    variant_records.append(record)
                except VisualRendererError as exc:
                    variant_records.append(
                        {
                            "variant_id": repaired.variant_id,
                            "variant_index": repaired.variant_index,
                            "candidate_kind": "bounded_repair",
                            "repair_action_applied": repair_action,
                            "render_error": str(exc),
                            "spec": repaired.spec,
                        }
                    )
            selected = select_best_variant(variant_records)
        repair_history_path.write_text(
            json.dumps(
                {
                    "version": "hyperframes-visual-cegis-v1",
                    "rounds": repair_history,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        final_verdict = None
        while selected is not None:
            selected_metadata = dict(selected.get("metadata") or {})
            if not selected_metadata.get("scene_program_v2"):
                break
            selected_artifacts = dict(selected.get("artifact_paths") or {})
            frame_paths = [
                Path(str(item))
                for item in selected_artifacts.get("qa_frame_paths") or []
            ]
            final_verdict = judge_final_candidate(
                frame_paths,
                production_contract=dict(
                    selected_metadata.get(
                        "hyperframes_production_contract"
                    )
                    or {}
                ),
                scene_program=dict(
                    selected_metadata.get("scene_program_v2") or {}
                ),
                quality_report=dict(selected.get("qa") or {}),
                critic_bundle=dict(
                    selected_metadata.get("visual_critics") or {}
                ),
                qa_mode=str(config.HYPERFRAMES_QA_MODE or "hybrid"),
            )
            verdict_path = (
                Path(str(selected.get("job_dir") or job_dir))
                / "final_independent_verdict.json"
            )
            verdict_path.write_text(
                json.dumps(final_verdict.to_dict(), indent=2),
                encoding="utf-8",
            )
            selected.setdefault("artifact_paths", {})[
                "final_independent_verdict_path"
            ] = str(verdict_path)
            selected.setdefault("metadata", {})[
                "final_independent_verdict"
            ] = final_verdict.to_dict()
            if final_verdict.passed:
                break
            selected["eligible_for_selection"] = False
            selected = select_best_variant(variant_records)
        tournament = dict(spec.get("visual_proof_tournament") or {})
        variants_report_path.write_text(
            json.dumps(
                {
                    "selected_variant_id": selected.get("variant_id") if selected else None,
                    "min_quality_score": config.HYPERFRAMES_MIN_QUALITY_SCORE,
                    "blind_decoder_min_score": config.HYPERFRAMES_BLIND_DECODER_MIN_SCORE,
                    "qa_mode": config.HYPERFRAMES_QA_MODE,
                    "vision_qa_enabled": bool(config.HYPERFRAMES_ENABLE_VISION_QA),
                    "counterfactual_qa_enabled": bool(
                        config.HYPERFRAMES_ENABLE_COUNTERFACTUAL_QA
                    ),
                    "proof_tournament_signature": tournament.get(
                        "tournament_signature"
                    ),
                    "proof_program_count": len(variants),
                    "variant_count": len(variants),
                    "render_attempt_count": len(variant_records),
                    "repair_round_count": len(repair_history),
                    "final_independent_verdict": (
                        final_verdict.to_dict()
                        if final_verdict is not None
                        else None
                    ),
                    "variants": variant_records,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if selected is None:
            details = "; ".join(
                str(
                    item.get("render_error")
                    or ", ".join(
                        str(issue)
                        for issue in (item.get("qa") or {}).get("issues", [])[:3]
                    )
                    or "variant failed QA"
                )
                for item in variant_records
            )
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
                "selected_proof_program_id": (
                    selected.get("metadata") or {}
                ).get("proof_program_id"),
                "selected_proof_strategy_id": (
                    selected.get("metadata") or {}
                ).get("proof_strategy_id"),
                "selected_proof_encoding": (
                    selected.get("metadata") or {}
                ).get("proof_encoding"),
                "selected_inverse_decoder_score": (
                    (selected.get("metadata") or {}).get("vision_qa") or {}
                ).get("score"),
                "selected_relation_coverage": (
                    (selected.get("metadata") or {}).get("vision_qa") or {}
                ).get("relation_coverage"),
                "selected_counterfactual_score": (
                    (
                        (selected.get("metadata") or {}).get("vision_qa") or {}
                    ).get("counterfactual")
                    or {}
                ).get("score"),
                "proof_program_count": len(variants),
                "variant_count": len(variants),
                "render_attempt_count": len(variant_records),
            },
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        artifact_paths = dict(selected.get("artifact_paths") or {})
        variant_metadata_path = artifact_paths.get("metadata_path")
        if variant_metadata_path:
            artifact_paths["variant_metadata_path"] = str(variant_metadata_path)
        artifact_paths["metadata_path"] = str(metadata_path)
        artifact_paths["variants_report_path"] = str(variants_report_path)
        artifact_paths["repair_history_path"] = str(repair_history_path)
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
