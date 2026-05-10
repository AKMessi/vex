from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import config
from engine import probe_video
from renderers.base import RenderedAsset, RendererStatus, VisualRenderer, VisualRendererError
from vex_hyperframes import build_composition, validate_composition_html


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


def _npx_command(*args: str) -> list[str]:
    npx_path = shutil.which(config.HYPERFRAMES_NPX_PATH) or config.HYPERFRAMES_NPX_PATH
    npx_resolved = Path(npx_path)
    if npx_resolved.name.lower() in {"npx.cmd", "npx.bat"}:
        node_path = shutil.which("node")
        npx_cli = npx_resolved.parent / "node_modules" / "npm" / "bin" / "npx-cli.js"
        if node_path and npx_cli.is_file():
            return [
                node_path,
                str(npx_cli),
                "--yes",
                config.HYPERFRAMES_CLI_PACKAGE,
                *args,
            ]
    return [
        npx_path,
        "--yes",
        config.HYPERFRAMES_CLI_PACKAGE,
        *args,
    ]


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
        "metric_callout",
        "keyword_stack",
        "timeline_steps",
        "comparison_split",
        "quote_focus",
        "system_flow",
        "stat_grid",
    }

    def availability(self) -> RendererStatus:
        if shutil.which(config.HYPERFRAMES_NPX_PATH) is None:
            return RendererStatus(False, "npx is not available in PATH; Hyperframes requires Node.js 22+ and npm/npx.")
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
        if template in {"data_journey", "signal_network", "kinetic_route", "spotlight_compare", "interface_cascade", "ribbon_quote"}:
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
        job_dir = render_root / spec_id
        job_dir.mkdir(parents=True, exist_ok=True)

        composition = build_composition(spec, width=width, height=height, fps=fps)
        index_path = job_dir / "index.html"
        output_path = job_dir / f"{scene_name}.mp4"
        metadata_path = job_dir / "hyperframes_metadata.json"
        validation_path = job_dir / "hyperframes_validation.json"
        lint_log_path = job_dir / "hyperframes_lint.log"
        render_log_path = job_dir / "hyperframes_render.log"
        spec_path = job_dir / "hyperframes_spec.json"

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

        lint_command = _npx_command("lint", "--json", str(job_dir))
        lint_result = subprocess.run(
            lint_command,
            cwd=str(job_dir),
            capture_output=True,
            text=True,
            timeout=config.HYPERFRAMES_LINT_TIMEOUT_SEC,
        )
        _write_command_log(lint_log_path, lint_command, lint_result)
        if lint_result.returncode != 0:
            detail = (lint_result.stderr or lint_result.stdout or "").strip()
            raise VisualRendererError(f"Hyperframes lint failed for {spec_id}: {detail}")

        render_command = _npx_command(
            "render",
            "--output",
            str(output_path),
            "--fps",
            str(max(15, int(round(fps or 30.0)))),
            str(job_dir),
        )
        quality = str(config.HYPERFRAMES_RENDER_QUALITY or "").strip()
        if quality:
            render_command.extend(["--quality", quality])
        render_result = subprocess.run(
            render_command,
            cwd=str(job_dir),
            capture_output=True,
            text=True,
            timeout=config.HYPERFRAMES_RENDER_TIMEOUT_SEC,
        )
        _write_command_log(render_log_path, render_command, render_result)
        if render_result.returncode != 0 or not output_path.is_file():
            detail = (render_result.stderr or render_result.stdout or "").strip()
            raise VisualRendererError(f"Hyperframes render failed for {spec_id}: {detail}")

        video_metadata = probe_video(str(output_path))
        metadata = {
            **composition.metadata,
            **video_metadata,
            "scene_generation_mode": "deterministic_hyperframes",
            "validation": validation.to_dict(),
            "hyperframes_cli_package": config.HYPERFRAMES_CLI_PACKAGE,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        artifact_paths = {
            "index_html_path": str(index_path),
            "spec_path": str(spec_path),
            "metadata_path": str(metadata_path),
            "validation_path": str(validation_path),
            "lint_log_path": str(lint_log_path),
            "render_log_path": str(render_log_path),
        }
        return RenderedAsset(
            asset_path=str(output_path),
            width=int(video_metadata.get("width") or width),
            height=int(video_metadata.get("height") or height),
            duration_sec=float(video_metadata.get("duration_sec") or composition.metadata["duration_sec"]),
            renderer=self.name,
            job_dir=str(job_dir),
            script_path=str(index_path),
            artifact_paths=artifact_paths,
            metadata=metadata,
        )
