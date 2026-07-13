from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import config
from broll_intelligence import (
    call_reasoning_model,
    configured_stock_provider_names,
    ensure_writable_dir,
    safe_stem,
    writable_dir_candidates,
)
from tools.creative_intelligence import annotate_visual_cards_with_graph, build_video_understanding_graph
from tools.creative_optimizer import optimize_creative_set
from tools.creative_qa import evaluate_visual_plan_quality
from tools.composite_qa import evaluate_visual_composite
from tools.creative_registry import (
    CreativePolicySnapshot,
    load_creative_policy,
    record_creative_run,
)
from engine import apply_visual_overlays, probe_video
from renderers import (
    RenderedAsset,
    RendererMatch,
    VisualRendererError,
    list_renderers,
    rank_renderers,
    render_with_manifest,
    renderer_capabilities,
    resolve_renderer,
)
from state import ProjectState, restrict_timed_items_to_available_ranges, utc_now_iso
from tools.automation import (
    clamp_int,
    coverage_counts,
    create_unique_bundle_dir,
    normalize_coverage_policy,
    normalize_density,
    write_run_status,
)
from tools.path_security import project_input_roots
from tools.transcript import execute as transcribe
from tools.transcript_utils import load_transcript_bundle
from tools.undo import refresh_generated_overlay_ops
from visual_intelligence import (
    STYLE_PACKS,
    THEME_BY_VISUAL_TYPE,
    analyze_visual_plan_with_llm,
    build_visual_context_cards,
    detect_scene_cuts,
    enforce_visual_semantic_contracts,
    fallback_visual_plan,
)
from visual_opportunity import build_visual_opportunity_plan
from visual_program import apply_visual_program_to_specs, build_visual_narrative_program
from visual_skill_graph import apply_visual_skill_graph
from vex_hyperframes.compiler import compile_hyperframes_plan
from vex_hyperframes.qa import extract_quality_frames, visual_fingerprint_distance
from vex_hyperframes.visual_world import build_video_design_bible
from vex_remotion.compiler import compile_remotion_scene_program
from vex_runtime.imaging import require_imaging_runtime
from vex_visuals.generative_authoring import compile_open_visual_program_for_spec
from vex_visuals.director import VisualDirectionOutcome, direct_rendered_visual
from vex_visuals.open_visual_program import (
    open_visual_program_fingerprint,
    validate_open_visual_program,
)
from vex_visuals.portfolio import (
    evaluate_visual_portfolio,
    extract_visual_portfolio_identity,
    same_creative_grammar,
)


AUTO_VISUALS_DIRECTOR_VERSION = "auto-visuals-director-v3"
RENDERER_TOURNAMENT_VERSION = "renderer-quality-tournament-v1"
DIRECTED_HYPERFRAMES_VISUAL_VERSION = "directed-hyperframes-visual-v1"
SOURCE_FRAME_SAMPLE_WIDTH = 48
SOURCE_FRAME_SAMPLE_HEIGHT = 27

_DIRECTED_METRIC_RE = re.compile(
    r"(?<![A-Za-z0-9.])"
    r"(?P<number>\d+(?:\.\d+)?)"
    r"\s*(?P<unit>%|percent|x|ms|milliseconds?|s|sec|seconds?|kb|mb|gb|tb|k|m|b|tokens?|parameters?|users?)?"
    r"(?![A-Za-z0-9.])",
    flags=re.IGNORECASE,
)

_DIRECTED_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "i",
    "in",
    "into",
    "is",
    "it",
    "make",
    "of",
    "on",
    "or",
    "show",
    "that",
    "the",
    "this",
    "to",
    "use",
    "using",
    "visual",
    "visuals",
    "with",
}


@dataclass(frozen=True)
class SourceFrameAnalysis:
    time_sec: float
    brightness: float
    contrast: float
    edge_density: float
    colorfulness: float
    source_richness: float
    source_type: str
    visual_need: float
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VisualDirectorDecision:
    visual_id: str
    card_id: str
    intent_type: str
    renderer_policy: str
    director_score: float
    visual_need: float
    source_richness: float
    passed: bool
    reasons: list[str]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RenderedVisualQA:
    visual_id: str
    renderer: str
    score: float
    passed: bool
    issues: list[str]
    warnings: list[str]
    repair_action: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _emit_progress(message: str) -> None:
    print(f"[auto_visuals] {message}", flush=True)


def _bounded(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    if math.isnan(number) or math.isinf(number):
        number = default
    return max(0.0, min(number, 1.0))


def _word_tokens(text: object) -> list[str]:
    return re.findall(r"[a-zA-Z0-9%']+", str(text or "").lower())


def _keyword_overlap_score(text: object, keywords: object) -> float:
    tokens = set(_word_tokens(text))
    keyword_tokens = set(_word_tokens(" ".join(str(item) for item in _as_list(keywords))))
    if not keyword_tokens:
        return 0.35
    return _bounded(len(tokens & keyword_tokens) / max(len(keyword_tokens), 1))


def _load_manifest(path: str) -> dict[str, object] | None:
    try:
        target = Path(str(path))
        if not target.is_file():
            return None
        payload = json.loads(target.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _as_list(value: object) -> list:
    return list(value) if isinstance(value, list) else []


def _as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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


def _extract_tiny_rgb_frame(video_path: str, time_sec: float) -> bytes | None:
    command = [
        config.FFMPEG_PATH,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(float(time_sec), 0.0):.3f}",
        "-i",
        video_path,
        "-frames:v",
        "1",
        "-vf",
        (
            f"scale={SOURCE_FRAME_SAMPLE_WIDTH}:{SOURCE_FRAME_SAMPLE_HEIGHT}:"
            "force_original_aspect_ratio=decrease,"
            f"pad={SOURCE_FRAME_SAMPLE_WIDTH}:{SOURCE_FRAME_SAMPLE_HEIGHT}:"
            "(ow-iw)/2:(oh-ih)/2:color=black,format=rgb24"
        ),
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            timeout=8,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    expected = SOURCE_FRAME_SAMPLE_WIDTH * SOURCE_FRAME_SAMPLE_HEIGHT * 3
    if result.returncode != 0 or len(result.stdout) < expected:
        return None
    return result.stdout[:expected]


def _analyze_tiny_rgb_frame(raw: bytes, *, time_sec: float) -> SourceFrameAnalysis:
    pixels = [
        raw[index : index + 3]
        for index in range(0, len(raw), 3)
        if index + 2 < len(raw)
    ]
    if not pixels:
        return SourceFrameAnalysis(
            time_sec=round(time_sec, 3),
            brightness=0.0,
            contrast=0.0,
            edge_density=0.0,
            colorfulness=0.0,
            source_richness=0.0,
            source_type="unknown",
            visual_need=0.72,
            warnings=["empty_frame_sample"],
        )
    lumas = [(0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2]) / 255.0 for pixel in pixels]
    brightness = sum(lumas) / len(lumas)
    variance = sum((value - brightness) ** 2 for value in lumas) / len(lumas)
    contrast = math.sqrt(max(variance, 0.0))
    channel_spread = [
        (max(pixel[0], pixel[1], pixel[2]) - min(pixel[0], pixel[1], pixel[2])) / 255.0
        for pixel in pixels
    ]
    colorfulness = sum(channel_spread) / len(channel_spread)
    edge_values: list[float] = []
    width = SOURCE_FRAME_SAMPLE_WIDTH
    height = SOURCE_FRAME_SAMPLE_HEIGHT
    for y in range(height):
        row = y * width
        for x in range(width - 1):
            edge_values.append(abs(lumas[row + x] - lumas[row + x + 1]))
    for y in range(height - 1):
        row = y * width
        next_row = (y + 1) * width
        for x in range(width):
            edge_values.append(abs(lumas[row + x] - lumas[next_row + x]))
    edge_density = sum(edge_values) / max(len(edge_values), 1)
    warnings: list[str] = []
    if brightness < 0.16:
        warnings.append("source_frame_dark")
    if contrast < 0.055:
        warnings.append("source_frame_flat")
    if edge_density > 0.15 and colorfulness < 0.13:
        source_type = "screen_or_slide"
    elif edge_density > 0.14:
        source_type = "busy_detail"
    elif colorfulness > 0.20 and contrast > 0.10:
        source_type = "rich_camera"
    elif contrast < 0.06:
        source_type = "flat_or_static"
    else:
        source_type = "talking_head_or_simple"
    source_richness = _bounded(edge_density * 2.8 + contrast * 1.6 + colorfulness * 0.55)
    visual_need = _bounded(0.86 - source_richness * 0.55 + (0.16 if "source_frame_flat" in warnings else 0.0) + (0.10 if "source_frame_dark" in warnings else 0.0))
    return SourceFrameAnalysis(
        time_sec=round(time_sec, 3),
        brightness=round(brightness, 4),
        contrast=round(contrast, 4),
        edge_density=round(edge_density, 4),
        colorfulness=round(colorfulness, 4),
        source_richness=round(source_richness, 4),
        source_type=source_type,
        visual_need=round(visual_need, 4),
        warnings=warnings,
    )


def _fallback_source_frame_analysis(card: dict[str, object]) -> SourceFrameAnalysis:
    start = _as_float(card.get("start"), 0.0)
    end = _as_float(card.get("end"), start)
    graph = dict(card.get("creative_graph_signals") or {})
    visual_need = _bounded(
        0.52
        + _bounded(card.get("visualizability"), 0.4) * 0.22
        + _bounded(graph.get("graph_visual_opportunity"), 0.35) * 0.18
        - _bounded(card.get("replace_safety"), 0.3) * 0.08,
        0.58,
    )
    return SourceFrameAnalysis(
        time_sec=round((start + end) / 2.0, 3),
        brightness=0.0,
        contrast=0.0,
        edge_density=0.0,
        colorfulness=0.0,
        source_richness=round(max(0.0, 1.0 - visual_need), 4),
        source_type="not_sampled",
        visual_need=round(visual_need, 4),
        warnings=["frame_sample_unavailable"],
    )


def _annotate_cards_with_source_frames(
    cards: list[dict[str, object]],
    *,
    video_path: str,
    max_samples: int = 24,
) -> list[dict[str, object]]:
    annotated: list[dict[str, object]] = []
    sampled = 0
    for card in cards:
        normalized = dict(card)
        start = _as_float(normalized.get("start"), 0.0)
        end = _as_float(normalized.get("end"), start)
        center = max(0.0, (start + end) / 2.0)
        raw = _extract_tiny_rgb_frame(video_path, center) if sampled < max_samples else None
        analysis = _analyze_tiny_rgb_frame(raw, time_sec=center) if raw else _fallback_source_frame_analysis(normalized)
        if raw:
            sampled += 1
        normalized["source_frame_analysis"] = analysis.to_dict()
        original_priority = _as_float(normalized.get("priority"), 0.0)
        visual_need = analysis.visual_need
        source_richness = analysis.source_richness
        visualizability = _bounded(normalized.get("visualizability"), 0.45)
        graph = dict(normalized.get("creative_graph_signals") or {})
        graph_opportunity = _bounded(graph.get("graph_visual_opportunity"), 0.35)
        normalized["priority"] = round(
            original_priority
            + visual_need * 9.0
            + graph_opportunity * 5.0
            - max(0.0, source_richness - 0.62) * 7.0
            + visualizability * 3.0,
            3,
        )
        annotated.append(normalized)
    return annotated


def _normalize_renderer_name(value: object) -> str:
    renderer = str(value or "auto").strip().lower().replace("-", "_")
    if renderer in {"", "default"}:
        return "auto"
    if renderer in {"all", "mixed", "mix", "hyperframes_manim", "manim_hyperframes"}:
        return "both"
    if renderer in {"react", "react_video"}:
        return "remotion"
    if renderer in {"auto", "both", "hyperframes", "manim", "remotion", "ffmpeg", "blender"}:
        return renderer
    return "auto"


def _allowed_renderers(renderer_name: str) -> set[str] | None:
    if renderer_name == "hyperframes":
        return {"hyperframes"}
    if renderer_name == "manim":
        return {"manim"}
    if renderer_name == "remotion":
        return {"remotion"}
    if renderer_name == "both":
        return {"hyperframes", "manim"}
    if renderer_name in {"ffmpeg", "blender"}:
        return {renderer_name}
    return None


def _filter_renderer_capabilities(
    capabilities: list[dict[str, object]],
    renderer_name: str,
) -> list[dict[str, object]]:
    allowed = _allowed_renderers(renderer_name)
    if allowed is None:
        return capabilities
    return [
        item
        for item in capabilities
        if str(item.get("name") or "").strip().lower() in allowed
    ]


def _require_available_renderer(
    capabilities: list[dict[str, object]],
    renderer_name: str,
) -> None:
    if any(bool(item.get("available")) for item in capabilities):
        return
    requested = (
        ", ".join(sorted(_allowed_renderers(renderer_name) or {"any configured renderer"}))
        or renderer_name
    )
    reasons = [
        f"{str(item.get('name') or 'renderer')}: {str(item.get('reason') or 'unavailable')}"
        for item in capabilities
    ]
    detail = "; ".join(reasons) or "No matching renderer is registered."
    raise VisualRendererError(
        f"No requested generated-visual renderer is available ({requested}). {detail}"
    )


def _should_force_fullscreen_visuals(
    params: dict, *, mode: str, renderer_name: str
) -> bool:
    if "force_fullscreen" in params:
        return _as_bool(params.get("force_fullscreen"), True)
    if "fullscreen" in params:
        return _as_bool(params.get("fullscreen"), True)
    if "full_screen" in params:
        return _as_bool(params.get("full_screen"), True)
    return mode == "generated_only" or renderer_name in {"auto", "both", "hyperframes", "manim", "remotion"}


def _with_fullscreen_visual_spec(spec: dict[str, object]) -> dict[str, object]:
    fullscreen_spec = dict(spec)
    fullscreen_spec["composition_mode"] = "replace"
    fullscreen_spec["position"] = "center"
    fullscreen_spec["scale"] = 1.0
    fullscreen_spec["force_fullscreen"] = True
    return fullscreen_spec


def _prior_auto_visual_card_ids(state: ProjectState) -> set[str]:
    card_ids: set[str] = set()
    for op in state.timeline:
        if str(op.get("op") or "").strip() != "add_auto_visuals":
            continue
        overlays = (op.get("params") or {}).get("overlays") or []
        if not isinstance(overlays, list):
            continue
        for overlay in overlays:
            if isinstance(overlay, dict):
                card_id = str(overlay.get("card_id") or "").strip()
                if card_id:
                    card_ids.add(card_id)
                for source_card_id in _as_list(overlay.get("source_card_ids")):
                    normalized = str(source_card_id or "").strip()
                    if normalized:
                        card_ids.add(normalized)
    history = _as_list((state.artifacts or {}).get("auto_visuals_history"))
    for item in history:
        if not isinstance(item, dict):
            continue
        manifest = _load_manifest(str(item.get("manifest_path") or ""))
        if not manifest:
            continue
        overlays = _as_list(manifest.get("overlays"))
        for overlay in overlays:
            if isinstance(overlay, dict):
                card_id = str(overlay.get("card_id") or "").strip()
                if card_id:
                    card_ids.add(card_id)
                for source_card_id in _as_list(overlay.get("source_card_ids")):
                    normalized = str(source_card_id or "").strip()
                    if normalized:
                        card_ids.add(normalized)
    return card_ids


def _prior_auto_visual_failure_card_ids(
    state: ProjectState,
    *,
    renderer_name: str = "",
) -> set[str]:
    """Return exact compiler-rejected opportunity IDs for the requested renderer.

    Render QA failures are deliberately excluded. They describe a renderer or
    treatment failure, not invalid transcript evidence, and must not poison all
    overlapping subtitle cards on a later run or with another renderer.
    """
    registry_path = Path(state.working_dir) / "creative_runs.json"
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return set()
    runs = [
        item
        for item in _as_list(registry.get("runs"))
        if isinstance(item, dict)
        and str(item.get("feature") or "").strip() == "auto_visuals"
        and str((item.get("summary") or {}).get("status") or "").startswith("failed")
    ][-6:]
    requested_renderer = str(renderer_name or "").strip().lower()
    failed_ids: set[str] = set()
    for run in runs:
        manifest = _load_manifest(str(run.get("manifest_path") or ""))
        if not manifest:
            continue
        failed_renderer = str(manifest.get("renderer") or "").strip().lower()
        if (
            requested_renderer not in {"", "auto", "both"}
            and failed_renderer not in {"", "auto", "both", requested_renderer}
        ):
            continue
        compiler_rejections: list[object] = []
        if requested_renderer in {"", "auto", "both", "hyperframes"}:
            compiler_rejections.extend(
                _as_list((manifest.get("hyperframes_compiler") or {}).get("rejected"))
            )
        if requested_renderer in {"", "auto", "both", "remotion"}:
            compiler_rejections.extend(
                _as_list((manifest.get("remotion_compiler") or {}).get("rejected"))
            )
        for item in compiler_rejections:
            if not isinstance(item, dict):
                continue
            card_id = str(item.get("card_id") or "").strip()
            if card_id:
                failed_ids.add(card_id)
    return failed_ids


def _filter_previously_used_cards(
    cards: list[dict[str, object]],
    used_card_ids: set[str],
    *,
    max_visuals: int,
) -> list[dict[str, object]]:
    if not cards:
        return []
    prepared: list[dict[str, object]] = []
    fresh_cards = [
        card for card in cards if str(card.get("card_id") or "") not in used_card_ids
    ]
    fresh_only_mode = bool(used_card_ids) and len(fresh_cards) >= max_visuals
    for card in cards:
        normalized = dict(card)
        card_id = str(normalized.get("card_id") or "").strip()
        original_priority = _as_float(normalized.get("priority"), 0.0)
        recently_used = card_id in used_card_ids
        normalized["original_priority"] = original_priority
        normalized["recently_used"] = recently_used
        if recently_used:
            normalized["priority"] = round(
                original_priority - (32.0 if fresh_only_mode else 18.0), 2
            )
        else:
            normalized["priority"] = round(original_priority + 4.0, 2)
        prepared.append(normalized)
    if fresh_only_mode:
        prepared = [item for item in prepared if not bool(item.get("recently_used"))]
    ranked = sorted(
        prepared,
        key=lambda item: (
            1 if not bool(item.get("recently_used")) else 0,
            _as_float(item.get("priority"), 0.0),
            -_as_float(item.get("start"), 0.0),
        ),
        reverse=True,
    )
    return ranked


def _refresh_existing_auto_overlays(state: ProjectState) -> dict[str, int]:
    return refresh_generated_overlay_ops(
        state,
        remove_ops={"add_auto_visuals", "add_auto_broll"},
    )


def _ensure_transcript_bundle(state: ProjectState) -> dict[str, object]:
    transcript_bundle = load_transcript_bundle(state.working_dir)
    if transcript_bundle.get("segments"):
        return transcript_bundle
    result = transcribe({}, state)
    if not result["success"]:
        raise RuntimeError(result["message"])
    transcript_bundle = load_transcript_bundle(state.working_dir)
    if not transcript_bundle.get("segments"):
        raise RuntimeError(
            "Transcript generation completed, but no usable transcript segments were found."
        )
    return transcript_bundle


def _provider_and_model(state: ProjectState) -> tuple[str, str]:
    provider_name = (state.provider or config.PROVIDER or "gemini").strip().lower()
    if provider_name not in {"gemini", "claude"}:
        provider_name = "gemini"
    model_name = state.model or (
        config.CLAUDE_MODEL if provider_name == "claude" else config.GEMINI_MODEL
    )
    return provider_name, model_name


def _delegate_stock_fallback(params: dict, state: ProjectState, reason: str) -> dict:
    from tools import pexels_broll

    result = pexels_broll.execute(
        {
            "max_overlays": params.get("max_visuals", 4),
            "min_overlay_sec": params.get("min_visual_sec", 2.2),
            "max_overlay_sec": params.get("max_visual_sec", 3.6),
        },
        state,
    )
    message = f"{reason} Fell back to stock B-roll. {result['message']}"
    return {
        "success": result["success"],
        "message": message,
        "suggestion": result.get("suggestion"),
        "updated_state": result["updated_state"],
        "tool_name": "add_auto_visuals",
    }


def _apply_style_override(spec: dict[str, object], style_pack: str) -> None:
    normalized = (style_pack or "auto").strip().lower()
    if normalized in {"", "auto"} or normalized not in STYLE_PACKS:
        return
    visual_type_hint = str(spec.get("visual_type_hint") or "")
    theme = dict(STYLE_PACKS[normalized])
    theme.update(THEME_BY_VISUAL_TYPE.get(visual_type_hint, {}))
    spec["style_pack"] = normalized
    spec["theme"] = theme


def _extract_source_grounding_frame(
    video_path: str,
    output_path: Path,
    *,
    time_sec: float,
) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        config.FFMPEG_PATH,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(float(time_sec), 0.0):.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-vf",
        "scale=960:-2:force_original_aspect_ratio=decrease",
        "-y",
        str(output_path),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and output_path.is_file()


def _prepare_visual_spec(
    spec: dict[str, object],
    *,
    style_pack: str,
    provider_name: str,
    model_name: str,
    state: ProjectState | None = None,
    bundle_dir: Path | None = None,
) -> dict[str, object]:
    prepared = dict(spec)
    resolved_style_pack = style_pack
    if (resolved_style_pack or "auto").strip().lower() in {"", "auto"}:
        resolved_style_pack = str(prepared.get("style_pack") or "auto")
    _apply_style_override(prepared, resolved_style_pack)
    prepared["generation_provider"] = provider_name
    prepared["generation_model"] = model_name
    if state is not None:
        prepared["allowed_asset_roots"] = [
            str(path) for path in project_input_roots(state)
        ]
    visual_ir = dict(prepared.get("visual_explanation_ir") or {})
    scene_type = str(visual_ir.get("scene_type") or "")
    director = dict(prepared.get("auto_visuals_director") or {})
    source_analysis = dict(director.get("source_frame_analysis") or {})
    source_type = str(source_analysis.get("source_type") or "")
    if (
        state is not None
        and bundle_dir is not None
        and scene_type == "grounded_interface_walkthrough"
        and source_type in {"screen_or_slide", "busy_detail"}
    ):
        visual_id = safe_stem(str(prepared.get("visual_id") or "visual"))
        source_frame_path = (
            Path(bundle_dir) / "source_grounding" / f"{visual_id}.png"
        )
        timestamp = _as_float(
            source_analysis.get("time_sec"),
            (
                _as_float(prepared.get("start"), 0.0)
                + _as_float(prepared.get("end"), 0.0)
            )
            / 2.0,
        )
        if _extract_source_grounding_frame(
            state.working_file,
            source_frame_path,
            time_sec=timestamp,
        ):
            roots = list(prepared.get("allowed_asset_roots") or [])
            bundle_root = str(Path(bundle_dir).resolve())
            if bundle_root not in roots:
                roots.append(bundle_root)
            prepared["allowed_asset_roots"] = roots
            prepared["source_asset_grounding"] = {
                "kind": "source_video_frame",
                "asset_path": str(source_frame_path),
                "time_sec": round(timestamp, 3),
                "source_type": source_type,
                "reason": "Use the real captured interface as the primary visual surface.",
            }
    return prepared


def _copy_is_source_grounded(value: object, source_text: str) -> bool:
    phrase_tokens = set(_word_tokens(value))
    source_tokens = set(_word_tokens(source_text))
    if not phrase_tokens:
        return False
    normalized_phrase = " ".join(_word_tokens(value))
    normalized_source = " ".join(_word_tokens(source_text))
    if normalized_phrase and normalized_phrase in normalized_source:
        return True
    return len(phrase_tokens & source_tokens) / max(len(phrase_tokens), 1) >= 0.6


def _grounded_plan_steps(spec: dict[str, object], *, limit: int = 6) -> list[str]:
    source = f"{spec.get('sentence_text', '')} {spec.get('context_text', '')}"
    result: list[str] = []
    for item in _as_list(spec.get("steps")):
        cleaned = " ".join(str(item or "").split()).strip()
        if cleaned and _copy_is_source_grounded(cleaned, source):
            result.append(cleaned)
        if len(result) >= limit:
            break
    if len(result) >= 2:
        return result
    sentence = str(spec.get("sentence_text") or "")
    fragments = re.split(
        r"\s*(?:,|;|\bthen\b|\band then\b|\bnext\b)\s*",
        sentence,
        flags=re.IGNORECASE,
    )
    for fragment in fragments:
        cleaned = " ".join(fragment.split()).strip(" ,.;:-")
        if len(_word_tokens(cleaned)) < 2 or not _copy_is_source_grounded(cleaned, source):
            continue
        if cleaned.lower() not in {item.lower() for item in result}:
            result.append(cleaned)
        if len(result) >= limit:
            break
    return result


def _semantic_frame_for_hyperframes(
    spec: dict[str, object],
) -> dict[str, object]:
    frame = dict(spec.get("semantic_frame") or {})
    mode = str(
        frame.get("intuition_mode") or spec.get("intuition_mode") or ""
    ).strip().lower()
    sentence = str(spec.get("sentence_text") or "")
    context = str(spec.get("context_text") or "")
    source = f"{sentence} {context}"
    before = str(frame.get("before_state") or "").strip()
    after = str(frame.get("after_state") or "").strip()
    cause = str(frame.get("cause") or "").strip()
    effect = str(frame.get("effect") or "").strip()
    steps = _grounded_plan_steps(spec)
    template = str(spec.get("template") or "").strip().lower()

    if mode == "process_route" or template in {
        "kinetic_route",
        "timeline_steps",
        "checklist_reveal",
        "pipeline_xray",
        "mechanism_blueprint",
        "system_flow",
        "signal_network",
    }:
        if len(steps) >= 2:
            frame["steps"] = steps
        if before:
            frame["input"] = before
        if after:
            frame["result"] = after

    if mode == "causal_chain" or template in {"causal_chain", "flywheel_loop"}:
        frame["problem"] = before or cause
        frame["mechanism"] = cause or sentence
        frame["result"] = effect or after

    if mode == "interface_walkthrough" or str(
        spec.get("visual_type_hint") or ""
    ).strip().lower() == "product_ui":
        grounded_action = next(
            (
                item
                for item in [*steps, cause, sentence]
                if item and _copy_is_source_grounded(item, source)
            ),
            "",
        )
        if grounded_action:
            frame["action"] = grounded_action
        if effect or after:
            frame["result"] = effect or after

    if spec.get("metric_facts"):
        intervention_match = re.search(
            r"\b(?:after|when|once)\s+(.+?)(?:[.,;]|$)",
            sentence,
            flags=re.IGNORECASE,
        )
        if intervention_match:
            intervention = " ".join(intervention_match.group(1).split()).strip()
            if intervention and _copy_is_source_grounded(intervention, source):
                frame["intervention"] = intervention

    if template == "decision_tree":
        branch_match = re.search(
            r"\bif\s+(.+?)[,;]\s*(.+?)(?:[;,.]\s*|\s+)(?:otherwise|else)\s+(.+?)(?:[.;]|$)",
            sentence,
            flags=re.IGNORECASE,
        )
        if branch_match:
            frame["decision"] = branch_match.group(1).strip()
            frame["low_branch"] = branch_match.group(2).strip()
            frame["high_branch"] = branch_match.group(3).strip()

    if template in {"narrative_arc", "timeline_filmstrip"} and len(steps) >= 3:
        frame["setup"] = steps[0]
        frame["turn"] = steps[1]
        frame["payoff"] = steps[-1]

    if template in {"ribbon_quote", "quote_focus", "quote_breakdown"}:
        quote = str(spec.get("quote_text") or sentence).strip()
        if 4 <= len(_word_tokens(quote)) <= 18 and _copy_is_source_grounded(
            quote,
            source,
        ):
            frame["exact_quote"] = quote
    return frame


def _apply_hyperframes_continuity(
    spec: dict[str, object],
) -> dict[str, object]:
    normalized = dict(spec)
    episode = dict(normalized.get("episode_context") or {})
    concepts = [
        dict(item)
        for item in _as_list(episode.get("concepts"))
        if isinstance(item, dict)
    ]
    primary = concepts[0] if concepts else {}
    concept_color = str(primary.get("color") or "").strip()
    motif = str(
        episode.get("motif")
        or primary.get("motif")
        or normalized.get("background_motif")
        or ""
    ).strip()
    if concept_color:
        theme = dict(normalized.get("theme") or {})
        theme["accent"] = concept_color
        normalized["theme"] = theme
    normalized["semantic_continuity"] = {
        "continuity_group": str(normalized.get("continuity_group") or ""),
        "concept_ids": [str(item) for item in _as_list(normalized.get("concept_ids"))],
        "concept_color": concept_color,
        "motif": motif,
        "episode_id": str(normalized.get("episode_id") or ""),
    }
    return normalized


def _compile_hyperframes_specs(
    plan: list[dict[str, object]],
    *,
    design_bible: dict[str, object] | None = None,
    initial_history: list[dict[str, object]] | None = None,
    ordinal_offset: int = 0,
    width: int = 1280,
    height: int = 720,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    accepted: list[dict[str, object]] = []
    compiled: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    design_bible_payload = dict(
        design_bible
        or build_video_design_bible(
            [dict(item) for item in plan]
        ).to_dict()
    )
    visual_world_history = [
        dict(item)
        for item in initial_history or []
        if isinstance(item, dict)
    ]
    for ordinal, spec in enumerate(plan, start=ordinal_offset):
        renderer_hint = str(spec.get("renderer_hint") or "").strip().lower()
        if renderer_hint != "hyperframes":
            candidate = dict(spec)
            candidate["video_design_bible"] = dict(design_bible_payload)
            candidate["visual_world_ordinal"] = ordinal
            candidate["creative_direction_history"] = [
                dict(item) for item in visual_world_history
            ]
            accepted.append(candidate)
            continue
        candidate = _apply_hyperframes_continuity(dict(spec))
        candidate["width"] = int(width)
        candidate["height"] = int(height)
        candidate["video_design_bible"] = dict(design_bible_payload)
        candidate["visual_world_history"] = [
            dict(item) for item in visual_world_history
        ]
        candidate["visual_world_ordinal"] = ordinal
        candidate["semantic_frame"] = _semantic_frame_for_hyperframes(candidate)
        candidate.setdefault(
            "hyperframes_proof_candidate_count",
            int(config.HYPERFRAMES_PROOF_CANDIDATE_COUNT),
        )
        result = compile_hyperframes_plan(candidate)
        if not result.passed:
            rejected.append(
                {
                    "visual_id": str(candidate.get("visual_id") or ""),
                    "card_id": str(candidate.get("card_id") or ""),
                    "template": str(candidate.get("template") or ""),
                    "issues": list(result.issues),
                    "render_policy": result.ir.render_policy,
                    "scene_type": result.ir.scene_type,
                    "rejection_reasons": list(result.ir.rejection_reasons),
                }
            )
            continue
        compiled_spec = dict(result.renderer_spec)
        for passthrough_key in (
            "planning_context_text",
            "semantic_episode_id",
            "semantic_episode_summary",
            "source_card_ids",
            "opportunity_contract",
            "opportunity_preflight",
            "auto_visuals_director",
            "creative_set_selection",
            "creative_graph_signals",
            "user_visual_idea",
            "directed_visual_brief",
            "auto_visual_skill",
            "skill_template",
            "skill_plan_seed",
        ):
            if passthrough_key in candidate:
                value = candidate.get(passthrough_key)
                compiled_spec[passthrough_key] = (
                    dict(value)
                    if isinstance(value, dict)
                    else list(value)
                    if isinstance(value, list)
                    else value
                )
        compiled_spec["hyperframes_automatic_semantic_route"] = True
        compiled_spec["hyperframes_legacy_template_policy"] = "manual_only"
        primary_world = dict(compiled_spec.get("visual_world_program") or {})
        primary_fingerprint = dict(primary_world.get("fingerprint") or {})
        if primary_fingerprint:
            visual_world_history.append(primary_fingerprint)
        compiled_spec["hyperframes_compiler"] = {
            "passed": True,
            "scene_type": result.ir.scene_type,
            "blueprint_id": (
                result.blueprint_selection.blueprint.blueprint_id
                if result.blueprint_selection.blueprint
                else ""
            ),
            "semantic_signature": (
                result.production_contract.semantic_signature
                if result.production_contract
                else ""
            ),
            "claim_graph_signature": result.claim_graph.graph_signature,
            "proof_tournament_signature": (
                result.proof_tournament.tournament_signature
            ),
            "proof_candidate_count": len(result.proof_tournament.programs),
            "proof_programs": [
                {
                    "program_id": item.program_id,
                    "blueprint_id": item.blueprint_id,
                    "strategy_id": item.strategy_id,
                    "encoding_family": item.encoding_family,
                    "relation_mode": item.relation_mode,
                    "structural_prior": item.structural_prior,
                }
                for item in result.proof_tournament.programs
            ],
            "blind_inverse_decoder": {
                "enabled": bool(config.HYPERFRAMES_ENABLE_VISION_QA),
                "counterfactuals_enabled": bool(
                    config.HYPERFRAMES_ENABLE_COUNTERFACTUAL_QA
                ),
                "minimum_score": float(
                    config.HYPERFRAMES_BLIND_DECODER_MIN_SCORE
                ),
            },
            "scene_program_version": str(
                compiled_spec.get("scene_program_v2", {}).get("version") or ""
            ),
            "video_design_bible": {
                "design_id": str(design_bible_payload.get("design_id") or ""),
                "signature": str(design_bible_payload.get("signature") or ""),
                "repetition_window": int(
                    design_bible_payload.get("repetition_window") or 3
                ),
            },
            "visual_world_candidates": [
                {
                    "world_id": str(
                        (item.get("visual_world_program") or {}).get(
                            "world_id"
                        )
                        or ""
                    ),
                    "medium_family": str(
                        (item.get("visual_world_program") or {}).get(
                            "medium_family"
                        )
                        or ""
                    ),
                    "canvas_system": str(
                        (item.get("visual_world_program") or {}).get(
                            "canvas_system"
                        )
                        or ""
                    ),
                    "background_mode": str(
                        (item.get("visual_world_program") or {}).get(
                            "background_mode"
                        )
                        or ""
                    ),
                    "fingerprint_signature": str(
                        (
                            (item.get("visual_world_program") or {}).get(
                                "fingerprint"
                            )
                            or {}
                        ).get("signature")
                        or ""
                    ),
                    "creative_direction_id": str(
                        (item.get("creative_direction_program") or {}).get(
                            "direction_id"
                        )
                        or ""
                    ),
                    "creative_direction_signature": str(
                        (item.get("creative_direction_program") or {}).get(
                            "signature"
                        )
                        or ""
                    ),
                }
                for item in compiled_spec.get("visual_proof_programs") or []
                if isinstance(item, dict)
            ],
            "counterexample_guided_repair": {
                "enabled": bool(config.HYPERFRAMES_ENABLE_CEGIS),
                "max_rounds": int(config.HYPERFRAMES_MAX_REPAIR_ROUNDS),
                "minimum_improvement": float(
                    config.HYPERFRAMES_MIN_REPAIR_DELTA
                ),
                "max_critic_frames": int(
                    config.HYPERFRAMES_MAX_CRITIC_FRAMES
                ),
            },
            "legacy_template_policy": "manual_only",
        }
        if compiled_spec.get("directed_visual_brief"):
            compiled_spec["hyperframes_compiler"]["directed_visual_brief"] = dict(
                compiled_spec.get("directed_visual_brief") or {}
            )
        accepted.append(compiled_spec)
        compiled.append(dict(compiled_spec["hyperframes_compiler"]))
    return accepted, {
        "input_count": len(plan),
        "accepted_count": len(accepted),
        "compiled_count": len(compiled),
        "rejected_count": len(rejected),
        "proof_candidate_count": sum(
            int(item.get("proof_candidate_count") or 0)
            for item in compiled
        ),
        "estimated_render_count": (
            len(accepted) - len(compiled)
            + sum(
                int(item.get("proof_candidate_count") or 0)
                for item in compiled
            )
        ),
        "compiled": compiled,
        "rejected": rejected,
        "video_design_bible": design_bible_payload,
        "visual_world_history": visual_world_history,
    }


def _timed_specs_overlap(
    left: dict[str, object],
    right: dict[str, object],
) -> bool:
    left_start = _as_float(left.get("start"), 0.0)
    left_end = _as_float(left.get("end"), left_start)
    right_start = _as_float(right.get("start"), 0.0)
    right_end = _as_float(right.get("end"), right_start)
    return not (
        left_end <= right_start - 0.1
        or left_start >= right_end + 0.1
    )


def _compile_open_visual_specs(
    plan: list[dict[str, object]],
    reserve_plan: list[dict[str, object]],
    *,
    provider_name: str,
    model_name: str,
    width: int,
    height: int,
    fps: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    if not bool(config.OPEN_VISUAL_PROGRAM_ENABLED):
        return plan, reserve_plan, {
            "version": "vex-generative-visual-authoring-v1",
            "enabled": False,
            "compiled_count": 0,
            "rejected_count": 0,
        }

    history: list[dict[str, object]] = []
    compiled_reports: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []

    def compile_items(
        items: list[dict[str, object]],
        *,
        allow_model: bool,
    ) -> list[dict[str, object]]:
        output: list[dict[str, object]] = []
        for raw in items:
            spec = dict(raw)
            renderer_hint = str(spec.get("renderer_hint") or "").strip().lower()
            if renderer_hint not in {"hyperframes", "remotion"}:
                output.append(spec)
                continue
            ir = dict(spec.get("visual_explanation_ir") or {})
            if not ir:
                rejected.append(
                    {
                        "visual_id": str(spec.get("visual_id") or ""),
                        "reason": "open_visual_program_has_no_evidence_ir",
                    }
                )
                output.append(spec)
                continue
            spec["generation_provider"] = provider_name
            spec["generation_model"] = model_name
            spec["open_visual_program_history"] = [dict(item) for item in history]
            enriched, result = compile_open_visual_program_for_spec(
                spec,
                ir=ir,
                width=width,
                height=height,
                fps=fps,
                reasoning_call=call_reasoning_model,
                enable_model_authoring=bool(
                    allow_model and config.OPEN_VISUAL_PROGRAM_LLM_AUTHORING
                ),
                candidate_count=int(config.OPEN_VISUAL_PROGRAM_CANDIDATES),
                max_model_attempts=int(
                    config.OPEN_VISUAL_PROGRAM_AUTHORING_ATTEMPTS
                ),
            )
            if not result.passed or result.selected_program is None:
                rejected.append(
                    {
                        "visual_id": str(spec.get("visual_id") or ""),
                        "reason": "no_valid_open_visual_program",
                        "warnings": list(result.warnings),
                        "rejected_model_programs": list(
                            result.rejected_model_programs
                        ),
                    }
                )
                output.append(spec)
                continue
            validation = validate_open_visual_program(
                result.selected_program,
                ir=ir,
                history=history,
            )
            if validation.score < float(config.OPEN_VISUAL_PROGRAM_MIN_SCORE):
                rejected.append(
                    {
                        "visual_id": str(spec.get("visual_id") or ""),
                        "reason": "open_visual_program_below_quality_floor",
                        "score": validation.score,
                        "minimum_score": float(
                            config.OPEN_VISUAL_PROGRAM_MIN_SCORE
                        ),
                    }
                )
                output.append(spec)
                continue
            fingerprint = open_visual_program_fingerprint(
                result.selected_program
            )
            history.append(fingerprint)
            report = {
                "visual_id": str(spec.get("visual_id") or ""),
                "renderer": renderer_hint,
                "selected_program_id": str(
                    result.selected_program.get("program_id") or ""
                ),
                "authoring_mode": result.authoring_mode,
                "candidate_count": len(result.programs),
                "model_program_count": result.model_program_count,
                "deterministic_program_count": (
                    result.deterministic_program_count
                ),
                "score": validation.score,
                "semantic_fitness": validation.semantic_fitness,
                "tournament_signature": (
                    result.tournament.tournament_signature
                ),
                "fingerprint": fingerprint,
                "warnings": list(result.warnings),
            }
            enriched["open_visual_compiler"] = report
            compiled_reports.append(report)
            output.append(enriched)
        return output

    compiled_plan = compile_items(plan, allow_model=True)
    compiled_reserves = compile_items(reserve_plan, allow_model=True)
    return compiled_plan, compiled_reserves, {
        "version": "vex-generative-visual-authoring-v1",
        "enabled": True,
        "compiled_count": len(compiled_reports),
        "rejected_count": len(rejected),
        "model_authored_count": sum(
            1
            for item in compiled_reports
            if item.get("authoring_mode") == "llm_authored"
        ),
        "compiled": compiled_reports,
        "rejected": rejected,
        "history": history,
    }


def _compile_hyperframes_specs_with_reserves(
    plan: list[dict[str, object]],
    reserve_plan: list[dict[str, object]],
    *,
    target_count: int,
    width: int = 1280,
    height: int = 720,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    design_bible = build_video_design_bible(
        [dict(item) for item in [*plan, *reserve_plan]]
    ).to_dict()
    primary_compiled, primary_report = _compile_hyperframes_specs(
        plan,
        design_bible=design_bible,
        width=width,
        height=height,
    )
    reserve_compiled, reserve_report = _compile_hyperframes_specs(
        reserve_plan,
        design_bible=design_bible,
        initial_history=[
            dict(item)
            for item in primary_report.get("visual_world_history") or []
            if isinstance(item, dict)
        ],
        ordinal_offset=len(plan),
        width=width,
        height=height,
    )
    primary_by_card_id = {
        str(item.get("card_id") or ""): item
        for item in plan
        if str(item.get("card_id") or "")
    }
    rejected_episode_ids = [
        str(
            primary_by_card_id.get(str(item.get("card_id") or ""), {}).get(
                "semantic_episode_id"
            )
            or ""
        )
        for item in primary_report.get("rejected", [])
        if isinstance(item, dict)
    ]
    selected = list(primary_compiled)
    substitutions: list[dict[str, object]] = []
    used_reserve_ids: set[str] = set()
    ordered_reserves = sorted(
        reserve_compiled,
        key=lambda item: (
            0
            if str(item.get("semantic_episode_id") or "") in rejected_episode_ids
            else 1,
            -_as_float(
                (item.get("opportunity_contract") or {}).get("score"),
                0.0,
            ),
            _as_float(item.get("start"), 0.0),
        ),
    )
    while len(selected) < max(0, target_count):
        replacement = next(
            (
                item
                for item in ordered_reserves
                if str(item.get("visual_id") or "") not in used_reserve_ids
                and not any(
                    _timed_specs_overlap(item, existing)
                    for existing in selected
                )
            ),
            None,
        )
        if replacement is None:
            break
        replacement = dict(replacement)
        used_reserve_ids.add(str(replacement.get("visual_id") or ""))
        replacement["opportunity_recovery"] = {
            "stage": "semantic_compile",
            "reason": "primary_opportunity_rejected",
            "episode_id": str(replacement.get("semantic_episode_id") or ""),
        }
        selected.append(replacement)
        substitutions.append(
            {
                "card_id": replacement.get("card_id"),
                "visual_id": replacement.get("visual_id"),
                "episode_id": replacement.get("semantic_episode_id"),
                "stage": "semantic_compile",
            }
        )
    selected.sort(key=lambda item: _as_float(item.get("start"), 0.0))
    remaining_reserves = [
        item
        for item in ordered_reserves
        if str(item.get("visual_id") or "") not in used_reserve_ids
    ]
    compiled_payloads = [
        dict(item.get("hyperframes_compiler") or {})
        for item in selected
        if item.get("hyperframes_compiler")
    ]
    report = {
        **primary_report,
        "accepted_count": len(selected),
        "compiled_count": len(compiled_payloads),
        "proof_candidate_count": sum(
            int(item.get("proof_candidate_count") or 0)
            for item in compiled_payloads
        ),
        "estimated_render_count": (
            len(selected) - len(compiled_payloads)
            + sum(
                int(item.get("proof_candidate_count") or 0)
                for item in compiled_payloads
            )
        ),
        "compiled": compiled_payloads,
        "reserve_compiler": reserve_report,
        "reserve_substitutions": substitutions,
        "reserve_available_count": len(remaining_reserves),
    }
    return selected, remaining_reserves, report


def _compile_remotion_specs_with_reserves(
    plan: list[dict[str, object]],
    reserve_plan: list[dict[str, object]],
    *,
    target_count: int,
    width: int,
    height: int,
    fps: float,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    def compile_candidates(
        candidates: list[dict[str, object]],
    ) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
        accepted: list[dict[str, object]] = []
        compiled: list[dict[str, object]] = []
        rejected: list[dict[str, object]] = []
        for spec in candidates:
            if str(spec.get("renderer_hint") or "").strip().lower() != "remotion":
                accepted.append(dict(spec))
                continue
            result = compile_remotion_scene_program(
                dict(spec),
                width=width,
                height=height,
                fps=fps,
            )
            if not result.passed or result.program is None:
                rejected.append(
                    {
                        "visual_id": str(spec.get("visual_id") or ""),
                        "card_id": str(spec.get("card_id") or ""),
                        "template": str(spec.get("template") or ""),
                        "issues": list(result.errors),
                        "warnings": list(result.warnings),
                        "candidate_scores": list(result.candidate_scores),
                    }
                )
                continue
            normalized = dict(spec)
            normalized["remotion_scene_program"] = result.program.to_dict()
            normalized["remotion_compiler"] = {
                "passed": True,
                "version": result.program.version,
                "program_id": result.program.program_id,
                "program_signature": result.program.signature,
                "scene_family": result.program.scene_family,
                "scene_type": result.program.scene_type,
                "layout": asdict(result.program.layout),
                "semantic_score": result.program.semantic_score,
                "grounding_mode": result.program.grounding_mode,
                "warnings": list(result.warnings),
                "candidate_scores": list(result.candidate_scores),
                "creative_direction_id": str(
                    result.program.creative_direction.get("direction_id") or ""
                ),
                "creative_direction_signature": str(
                    result.program.creative_direction.get("signature") or ""
                ),
                "medium_family": str(
                    result.program.creative_direction.get("medium_family") or ""
                ),
            }
            accepted.append(normalized)
            compiled.append(dict(normalized["remotion_compiler"]))
        return accepted, compiled, rejected

    selected, _primary_compiled, rejected = compile_candidates(plan)
    reserves, reserve_compiled, reserve_rejected = compile_candidates(reserve_plan)
    substitutions: list[dict[str, object]] = []
    used_reserve_ids: set[str] = set()
    ordered_reserves = sorted(
        reserves,
        key=lambda item: (
            -_as_float((item.get("opportunity_contract") or {}).get("score"), 0.0),
            _as_float(item.get("start"), 0.0),
        ),
    )
    while len(selected) < max(0, target_count):
        replacement = next(
            (
                item
                for item in ordered_reserves
                if str(item.get("visual_id") or "") not in used_reserve_ids
                and not any(_timed_specs_overlap(item, existing) for existing in selected)
            ),
            None,
        )
        if replacement is None:
            break
        replacement = dict(replacement)
        used_reserve_ids.add(str(replacement.get("visual_id") or ""))
        replacement["opportunity_recovery"] = {
            "stage": "remotion_semantic_compile",
            "reason": "primary_remotion_program_rejected",
            "episode_id": str(replacement.get("semantic_episode_id") or ""),
        }
        selected.append(replacement)
        substitutions.append(
            {
                "visual_id": replacement.get("visual_id"),
                "card_id": replacement.get("card_id"),
                "stage": "remotion_semantic_compile",
            }
        )
    selected.sort(key=lambda item: _as_float(item.get("start"), 0.0))
    remaining_reserves = [
        item
        for item in ordered_reserves
        if str(item.get("visual_id") or "") not in used_reserve_ids
    ]
    selected_compiled = [
        dict(item.get("remotion_compiler") or {})
        for item in selected
        if item.get("remotion_compiler")
    ]
    return selected, remaining_reserves, {
        "version": "remotion-semantic-compiler-v2",
        "input_count": len(plan),
        "accepted_count": len(selected),
        "compiled_count": len(selected_compiled),
        "rejected_count": len(rejected),
        "compiled": selected_compiled,
        "rejected": rejected,
        "reserve_compiled_count": len(reserve_compiled),
        "reserve_rejected": reserve_rejected,
        "reserve_substitutions": substitutions,
        "reserve_available_count": len(remaining_reserves),
    }


def _ensure_unique_visual_ids(
    specs: list[dict[str, object]],
) -> list[dict[str, object]]:
    normalized_specs: list[dict[str, object]] = []
    for index, spec in enumerate(specs, start=1):
        normalized = dict(spec)
        normalized["visual_id"] = f"visual_{index:03d}"
        normalized_specs.append(normalized)
    return normalized_specs


def _apply_creative_graph_to_visual_specs(
    specs: list[dict[str, object]],
    cards: list[dict[str, object]],
) -> list[dict[str, object]]:
    card_by_id = {
        str(card.get("card_id") or "").strip(): card
        for card in cards
        if str(card.get("card_id") or "").strip()
    }
    enriched: list[dict[str, object]] = []
    for spec in specs:
        normalized = dict(spec)
        card = card_by_id.get(str(normalized.get("card_id") or "").strip())
        if card:
            normalized["creative_graph_signals"] = dict(card.get("creative_graph_signals") or {})
        enriched.append(normalized)
    return enriched


def _available_renderer_names(capabilities: list[dict[str, object]]) -> set[str]:
    return {
        str(item.get("name") or "").strip().lower()
        for item in capabilities
        if bool(item.get("available"))
    }


def _visual_intent_type(spec: dict[str, object], card: dict[str, object]) -> str:
    template = str(spec.get("template") or "").strip().lower()
    visual_type_hint = str(card.get("visual_type_hint") or spec.get("visual_type_hint") or "").strip().lower()
    numeric_hits = int(_as_float(card.get("numeric_hits"), 0.0))
    process_cues = _bounded(card.get("process_cues"), 0.0)
    contrast_cues = _bounded(card.get("contrast_cues"), 0.0)
    text = " ".join(
        [
            str(spec.get("headline") or ""),
            str(spec.get("deck") or ""),
            str(spec.get("sentence_text") or card.get("sentence_text") or ""),
            str(spec.get("context_text") or card.get("context_text") or ""),
        ]
    ).lower()
    if template in {"data_journey", "data_pulse", "proof_sequence", "scorecard", "risk_radar", "metric_callout", "stat_grid"} or spec.get("metric_facts"):
        return "data_proof"
    if template in {"causal_chain", "flywheel_loop", "signal_network", "kinetic_route", "pipeline_xray", "mechanism_blueprint", "decision_tree", "system_flow"}:
        return "mechanism"
    if template in {"comparison_split", "spotlight_compare", "decision_matrix", "contrast_ladder", "problem_solution", "myth_buster"}:
        return "contrast"
    if template in {"interface_cascade", "screen_pointer_3d"} or str(card.get("visual_type_hint") or "") == "product_ui":
        return "ui_callout"
    if template in {"three_d_title", "object_orbit", "logo_reveal", "data_tunnel", "product_model_spin", "floating_3d_label"}:
        return "spatial_3d"
    if re.search(r"\b(?:formula|equation|matrix|vector|graph|axis|geometry|proof|derivative|integral|attention|transformer)\b", text):
        return "math_or_formula"
    if numeric_hits > 0 or visual_type_hint == "data_graphic":
        return "data_proof"
    if process_cues >= 0.22 or visual_type_hint == "process":
        return "mechanism"
    if contrast_cues >= 0.22:
        return "contrast"
    if template in {"timeline_steps", "timeline_filmstrip", "narrative_arc", "checklist_reveal"}:
        return "sequence"
    if template in {"quote_focus", "ribbon_quote", "quote_breakdown", "keyword_stack", "focus_ring"}:
        return "emphasis"
    return "concept"


def _preferred_renderer_for_intent(
    intent_type: str,
    spec: dict[str, object],
    available_renderers: set[str],
) -> str:
    template = str(spec.get("template") or "").strip().lower()
    if intent_type == "spatial_3d" and "blender" in available_renderers:
        return "blender"
    if intent_type == "math_or_formula" and "manim" in available_renderers:
        return "manim"
    if template in {"quote_focus", "keyword_stack", "metric_callout", "stat_grid", "timeline_steps", "comparison_split"} and str(spec.get("composition_mode") or "") == "picture_in_picture":
        if "ffmpeg" in available_renderers:
            return "ffmpeg"
        if "remotion" in available_renderers and "hyperframes" not in available_renderers:
            return "remotion"
        return "hyperframes"
    if "remotion" in available_renderers and "hyperframes" not in available_renderers:
        return "remotion"
    if "hyperframes" in available_renderers:
        return "hyperframes"
    if "remotion" in available_renderers:
        return "remotion"
    if "ffmpeg" in available_renderers:
        return "ffmpeg"
    return str(spec.get("renderer_hint") or "auto")


def _visual_copy_text(spec: dict[str, object]) -> str:
    return " ".join(
        [
            str(spec.get("headline") or ""),
            str(spec.get("deck") or ""),
            str(spec.get("emphasis_text") or ""),
            " ".join(str(item) for item in _as_list(spec.get("supporting_lines"))),
            " ".join(str(item) for item in _as_list(spec.get("steps"))),
            " ".join(str(item) for item in _as_list(spec.get("keywords"))),
        ]
    )


def _renderer_design_floor(renderer: str, intent_type: str) -> float:
    if renderer == "hyperframes":
        return 0.62
    if renderer == "manim":
        return 0.70 if intent_type != "math_or_formula" else 0.58
    if renderer == "remotion":
        return 0.60
    if renderer == "blender":
        return 0.58
    return 0.50


def _director_decision_for_spec(
    spec: dict[str, object],
    card: dict[str, object],
    *,
    renderer_name: str,
    capabilities: list[dict[str, object]],
    force_fullscreen: bool,
    coverage_policy: str = "quality_only",
    creative_policy: CreativePolicySnapshot | None = None,
) -> tuple[dict[str, object], VisualDirectorDecision]:
    available = _available_renderer_names(capabilities)
    normalized = dict(spec)
    intent_type = _visual_intent_type(normalized, card)
    source = dict(card.get("source_frame_analysis") or {})
    graph = dict(card.get("creative_graph_signals") or normalized.get("creative_graph_signals") or {})
    visual_need = _bounded(source.get("visual_need"), 0.62)
    source_richness = _bounded(source.get("source_richness"), 0.36)
    graph_opportunity = _bounded(graph.get("graph_visual_opportunity"), 0.45)
    topic_alignment = _bounded(graph.get("graph_topic_alignment"), 0.45)
    visualizability = _bounded(card.get("visualizability"), 0.45)
    generic_penalty = _bounded(card.get("generic_penalty"), 0.0)
    confidence = _bounded(normalized.get("confidence"), 0.55)
    copy_alignment = _keyword_overlap_score(_visual_copy_text(normalized), normalized.get("keywords") or card.get("keywords"))
    preferred_renderer = _preferred_renderer_for_intent(intent_type, normalized, available)
    requested_renderer = str(normalized.get("renderer_hint") or "auto").strip().lower()
    strict_allowed = _allowed_renderers(renderer_name)
    renderer_policy = preferred_renderer
    warnings: list[str] = []
    reasons: list[str] = []
    if requested_renderer == "manim" and intent_type != "math_or_formula":
        warnings.append("manim_rerouted_non_math_visual")
        if strict_allowed is None or {"hyperframes", "remotion", "ffmpeg"} & strict_allowed:
            normalized["renderer_hint"] = (
                "hyperframes"
                if "hyperframes" in available
                else "remotion"
                if "remotion" in available
                else preferred_renderer
            )
            renderer_policy = str(normalized["renderer_hint"])
    elif requested_renderer in {"auto", "", "ffmpeg"} and preferred_renderer in available:
        normalized["renderer_hint"] = preferred_renderer
        renderer_policy = preferred_renderer
    elif requested_renderer == "hyperframes" and preferred_renderer == "manim" and renderer_name not in {"hyperframes"}:
        normalized["renderer_hint"] = "manim"
        renderer_policy = "manim"
    else:
        renderer_policy = requested_renderer if requested_renderer not in {"", "auto"} else preferred_renderer
    if renderer_name == "hyperframes":
        normalized["renderer_hint"] = "hyperframes"
        renderer_policy = "hyperframes"
    elif renderer_name == "manim":
        normalized["renderer_hint"] = "manim"
        renderer_policy = "manim"
    elif renderer_name == "remotion":
        normalized["renderer_hint"] = "remotion"
        renderer_policy = "remotion"
    elif renderer_name == "both" and str(normalized.get("renderer_hint") or "") not in {"hyperframes", "manim"}:
        normalized["renderer_hint"] = "manim" if intent_type == "math_or_formula" and "manim" in available else "hyperframes"
        renderer_policy = str(normalized["renderer_hint"])
    if force_fullscreen:
        normalized["composition_mode"] = "replace"
        normalized["position"] = "center"
        normalized["scale"] = 1.0
    if renderer_policy in {"hyperframes", "remotion"} and str(normalized.get("template") or "") in {"quote_focus", "keyword_stack", "ribbon_quote"} and intent_type not in {"emphasis", "math_or_formula"}:
        normalized["template"] = {
            "data_proof": "data_journey",
            "mechanism": "mechanism_blueprint",
            "contrast": "problem_solution",
            "ui_callout": "interface_cascade",
            "sequence": "timeline_filmstrip",
        }.get(intent_type, "concept_map")
        reasons.append("upgraded_low_depth_template")
    director_score = (
        visual_need * 23.0
        + graph_opportunity * 18.0
        + topic_alignment * 10.0
        + visualizability * 14.0
        + confidence * 18.0
        + copy_alignment * 10.0
        - generic_penalty * 16.0
        - max(0.0, source_richness - 0.72) * 8.0
    )
    if creative_policy is not None:
        policy_prior = creative_policy.explain_for(
            renderer=renderer_policy,
            intent_type=intent_type,
            template=normalized.get("template"),
            available_renderers=available,
        )
        policy_adjustment = _as_float(
            policy_prior.get("selection_adjustment"),
            0.0,
        )
        director_score += policy_adjustment * 100.0
        normalized["creative_policy_prior"] = policy_prior
        if abs(policy_adjustment) >= 0.005:
            reasons.append("bounded_historical_quality_prior_applied")
    if renderer_policy == "manim" and intent_type != "math_or_formula":
        director_score -= 18.0
        warnings.append("manim_requires_math_or_formula_context")
    if source.get("source_type") in {"screen_or_slide", "busy_detail"} and str(normalized.get("composition_mode") or "") == "replace" and visual_need < 0.55:
        director_score -= 10.0
        warnings.append("source_already_visually_dense")
    policy = normalize_coverage_policy(coverage_policy)
    floor = 50.0 if renderer_policy != "manim" else 60.0
    copy_floor = 0.18
    if policy in {"target_count", "exact_count"} and renderer_policy != "manim":
        floor = 42.0 if policy == "target_count" else 38.0
        copy_floor = 0.12
    passed = director_score >= floor and copy_alignment >= copy_floor
    if renderer_policy == "manim" and intent_type != "math_or_formula":
        passed = False
    if visual_need < 0.38 and source_richness > 0.76 and policy == "quality_only":
        passed = False
        warnings.append("visual_would_interrupt_rich_source_moment")
    if copy_alignment < copy_floor:
        reasons.append("copy_not_semantically_aligned")
    if passed:
        reasons.append("director_v3_passed")
    normalized["visual_intent_type"] = intent_type
    normalized["auto_visuals_director"] = {
        "version": AUTO_VISUALS_DIRECTOR_VERSION,
        "director_score": round(director_score, 3),
        "visual_need": round(visual_need, 4),
        "source_richness": round(source_richness, 4),
        "copy_alignment": round(copy_alignment, 4),
        "renderer_policy": renderer_policy,
        "coverage_policy": policy,
        "director_floor": floor,
        "copy_alignment_floor": copy_floor,
        "source_frame_analysis": source,
        "warnings": warnings,
        "reasons": reasons,
    }
    return normalized, VisualDirectorDecision(
        visual_id=str(normalized.get("visual_id") or ""),
        card_id=str(normalized.get("card_id") or ""),
        intent_type=intent_type,
        renderer_policy=renderer_policy,
        director_score=round(director_score, 3),
        visual_need=round(visual_need, 4),
        source_richness=round(source_richness, 4),
        passed=passed,
        reasons=reasons,
        warnings=warnings,
    )


def _apply_auto_visuals_director_v3(
    plan: list[dict[str, object]],
    cards: list[dict[str, object]],
    *,
    renderer_name: str,
    capabilities: list[dict[str, object]],
    force_fullscreen: bool,
    max_visuals: int,
    coverage_policy: str = "quality_only",
    creative_policy: CreativePolicySnapshot | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    card_by_id = {
        str(card.get("card_id") or "").strip(): card
        for card in cards
        if str(card.get("card_id") or "").strip()
    }
    accepted: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    for spec in plan:
        card_id = str(spec.get("card_id") or "").strip()
        card = card_by_id.get(card_id, {})
        normalized, decision = _director_decision_for_spec(
            spec,
            card,
            renderer_name=renderer_name,
            capabilities=capabilities,
            force_fullscreen=force_fullscreen,
            coverage_policy=coverage_policy,
            creative_policy=creative_policy,
        )
        decisions.append(decision.to_dict())
        if decision.passed:
            accepted.append(normalized)
        else:
            rejected.append({**decision.to_dict(), "headline": normalized.get("headline"), "template": normalized.get("template")})
    accepted, set_optimization = optimize_creative_set(
        accepted,
        budget=max_visuals,
        phase="plan",
        coverage_policy=coverage_policy,
    )
    for item in set_optimization.get("rejected", []):
        rejected.append(
            {
                "visual_id": item.get("candidate_id"),
                "reason": item.get("reason"),
                "director_score": round(_as_float(item.get("base_score"), 0.0) * 100.0, 3),
                "conflicting_visual_ids": item.get("conflicting_visual_ids") or [],
                "selection_stage": "creative_set_optimizer",
            }
        )
    accepted = _ensure_unique_visual_ids(accepted)
    for spec in accepted:
        director = dict(spec.get("auto_visuals_director") or {})
        director["visual_id_after_resequence"] = spec.get("visual_id")
        spec["auto_visuals_director"] = director
    avg_score = (
        sum(_as_float(item.get("director_score"), 0.0) for item in decisions)
        / max(len(decisions), 1)
    )
    report = {
        "version": AUTO_VISUALS_DIRECTOR_VERSION,
        "input_count": len(plan),
        "coverage_policy": normalize_coverage_policy(coverage_policy),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "average_director_score": round(avg_score, 3),
        "set_optimization": set_optimization,
        "creative_policy": (
            creative_policy.to_dict() if creative_policy is not None else None
        ),
        "decisions": decisions,
        "rejected": rejected[:12],
        "source_frame_sampled_count": sum(
            1
            for card in cards
            if str((card.get("source_frame_analysis") or {}).get("source_type") or "") != "not_sampled"
        ),
    }
    return accepted, report


def _metadata_quality_score(value: object, default: float) -> float:
    if value is None:
        return default
    return _bounded(value, default)


def _rendered_visual_quality_for_spec(
    spec: dict[str, object],
    asset: RenderedAsset,
) -> RenderedVisualQA:
    metadata = dict(asset.metadata or {})
    director = dict(spec.get("auto_visuals_director") or {})
    renderer = str(asset.renderer or "").strip().lower()
    intent_type = str(spec.get("visual_intent_type") or "")
    director_score = _bounded(_as_float(director.get("director_score"), 0.0) / 100.0, 0.55)
    copy_alignment = _bounded(director.get("copy_alignment"), 0.35)
    target_duration = max(
        0.1,
        _as_float(spec.get("end"), 0.0) - _as_float(spec.get("start"), 0.0),
    )
    duration_delta = abs(float(asset.duration_sec or 0.0) - target_duration)
    duration_score = _bounded(1.0 - duration_delta / max(target_duration, 0.1), 0.0)
    issues: list[str] = []
    warnings: list[str] = [str(item) for item in _as_list(director.get("warnings"))]
    repair_action = "keep"
    quality_score = 0.58
    renderer_passed = True
    if renderer == "hyperframes":
        variant_selection = dict(metadata.get("variant_selection") or {})
        semantic_qa = dict(metadata.get("semantic_qa") or {})
        visual_critics = dict(metadata.get("visual_critics") or {})
        final_verdict = dict(
            metadata.get("final_independent_verdict") or {}
        )
        quality_score = _metadata_quality_score(
            variant_selection.get("selected_quality_score"),
            0.56,
        )
        renderer_passed = bool(
            variant_selection.get(
                "selected_quality_passed",
                quality_score >= config.HYPERFRAMES_MIN_QUALITY_SCORE,
            )
        )
        floor = max(0.6, min(float(config.HYPERFRAMES_MIN_QUALITY_SCORE), 0.92))
        if quality_score < floor:
            issues.append("hyperframes_variant_quality_below_floor")
            repair_action = "drop_low_quality_hyperframes_render"
        if not renderer_passed and quality_score < floor + 0.025:
            issues.append("hyperframes_variant_failed_renderer_qa")
            repair_action = "drop_low_quality_hyperframes_render"
        if semantic_qa:
            semantic_score = _metadata_quality_score(
                semantic_qa.get("score"),
                0.0,
            )
            if not bool(semantic_qa.get("passed")):
                issues.append("hyperframes_semantic_qa_failed")
                semantic_action = str(
                    semantic_qa.get("repair_action") or "reject_no_safe_repair"
                )
                repair_action = semantic_action
            if semantic_score < floor:
                issues.append("hyperframes_semantic_quality_below_floor")
            if semantic_qa.get("reroute_renderer"):
                warnings.append(
                    "hyperframes_reroute_recommended:"
                    + str(semantic_qa.get("reroute_renderer"))
                )
        if visual_critics and not bool(visual_critics.get("passed")):
            issues.append("hyperframes_structured_visual_critics_failed")
            repair_action = "drop_failed_hyperframes_critics"
        if final_verdict and not bool(final_verdict.get("passed")):
            issues.append("hyperframes_independent_final_judge_failed")
            repair_action = "drop_failed_hyperframes_final_judge"
    elif renderer == "manim":
        generation_mode = str(metadata.get("scene_generation_mode") or "")
        quality_score = _metadata_quality_score(
            metadata.get("quality_score"),
            0.52 if generation_mode == "legacy_template" else 0.64,
        )
        floor = _renderer_design_floor(renderer, intent_type)
        if intent_type != "math_or_formula":
            issues.append("manim_render_not_matched_to_math_or_formula_context")
            repair_action = "drop_or_reroute_to_hyperframes"
        if quality_score < floor:
            issues.append("manim_quality_below_floor")
            repair_action = "drop_low_quality_manim_render"
        if generation_mode == "legacy_template" and intent_type != "math_or_formula":
            warnings.append("manim_legacy_template_fallback")
    elif renderer == "remotion":
        semantic_qa = dict(metadata.get("semantic_qa") or {})
        remotion_render_qa = dict(metadata.get("remotion_render_qa") or {})
        quality_score = _metadata_quality_score(
            metadata.get("quality_score"),
            0.0,
        )
        renderer_passed = bool(metadata.get("quality_passed", False))
        floor = _renderer_design_floor(renderer, intent_type)
        if intent_type == "spatial_3d":
            issues.append("remotion_render_not_matched_to_spatial_3d_context")
            repair_action = "drop_or_reroute_to_blender"
        if intent_type == "math_or_formula" and str(spec.get("renderer_hint") or "") != "remotion":
            warnings.append("remotion_math_context_without_latex_runtime")
        if quality_score < floor:
            issues.append("remotion_quality_below_floor")
            repair_action = "drop_low_quality_remotion_render"
        if semantic_qa and not bool(semantic_qa.get("passed")):
            issues.append("remotion_semantic_program_failed")
            repair_action = "drop_semantically_invalid_remotion_render"
        if remotion_render_qa and not bool(remotion_render_qa.get("passed")):
            issues.extend(
                str(item)
                for item in _as_list(remotion_render_qa.get("issues"))[:4]
            )
            repair_action = "drop_failed_remotion_frame_qa"
        if not renderer_passed:
            issues.append("remotion_renderer_reported_quality_failure")
            repair_action = "drop_failed_remotion_render"
    elif renderer == "blender":
        quality_score = 0.64 if metadata.get("has_alpha") or spec.get("template") else 0.56
        floor = _renderer_design_floor(renderer, intent_type)
        if quality_score < floor:
            issues.append("blender_metadata_quality_below_floor")
            repair_action = "drop_low_quality_blender_render"
    else:
        quality_score = 0.56
        floor = _renderer_design_floor(renderer, intent_type)
    if copy_alignment < 0.16:
        issues.append("render_copy_not_semantically_aligned")
        repair_action = "drop_semantic_mismatch"
    if director_score < 0.46:
        issues.append("director_score_below_publishable_floor")
        repair_action = "drop_low_signal_visual"
    if duration_score < 0.52:
        warnings.append("render_duration_drift_detected")
    combined_score = _bounded(
        quality_score * 0.48
        + director_score * 0.28
        + copy_alignment * 0.16
        + duration_score * 0.08
    )
    if combined_score < max(0.54, floor * 0.72):
        issues.append("combined_visual_quality_below_floor")
        repair_action = "drop_low_quality_render"
    passed = not issues
    return RenderedVisualQA(
        visual_id=str(spec.get("visual_id") or ""),
        renderer=renderer,
        score=round(combined_score, 4),
        passed=passed,
        issues=issues,
        warnings=warnings,
        repair_action=repair_action,
        evidence={
            "renderer_quality_score": round(quality_score, 4),
            "renderer_passed": renderer_passed,
            "director_score": round(director_score, 4),
            "copy_alignment": round(copy_alignment, 4),
            "duration_score": round(duration_score, 4),
            "target_duration_sec": round(target_duration, 3),
            "rendered_duration_sec": round(float(asset.duration_sec or 0.0), 3),
            "intent_type": intent_type,
            "semantic_qa": dict(metadata.get("semantic_qa") or {}),
            "vision_qa": dict(metadata.get("vision_qa") or {}),
            "visual_critics": dict(metadata.get("visual_critics") or {}),
            "final_independent_verdict": dict(
                metadata.get("final_independent_verdict") or {}
            ),
        },
    )


def _direct_rendered_visual_for_spec(
    spec: dict[str, object],
    asset: RenderedAsset,
    selection_reason: str,
    *,
    render_root: Path,
    width: int,
    height: int,
    fps: float,
) -> tuple[
    dict[str, object],
    RenderedAsset,
    str,
    RenderedVisualQA,
    dict[str, object],
]:
    local_qa = _rendered_visual_quality_for_spec(spec, asset)
    renderer = str(asset.renderer or "").strip().lower()
    contract = dict(spec.get("visual_communication_contract") or {})
    ir = dict(spec.get("visual_explanation_ir") or {})
    mode = str(config.VISUAL_DIRECTOR_VERIFICATION_MODE or "balanced").strip().lower()
    if (
        not bool(config.VISUAL_DIRECTOR_ENABLED)
        or mode == "off"
        or renderer not in {"hyperframes", "remotion"}
        or not contract
        or not ir
    ):
        reason = (
            "disabled"
            if not bool(config.VISUAL_DIRECTOR_ENABLED) or mode == "off"
            else "renderer_or_contract_not_eligible"
        )
        return dict(spec), asset, selection_reason, local_qa, {
            "version": "vex-visual-director-runtime-v1",
            "enabled": False,
            "reason": reason,
        }

    visual_id = safe_stem(str(spec.get("visual_id") or "visual"))
    repair_root = render_root / "visual_director_repairs" / visual_id
    cache_dir = render_root.parent / "visual_director_cache"

    def render_candidate(
        repaired_spec: dict[str, object],
        round_index: int,
    ) -> tuple[RenderedAsset, str]:
        return _render_generated_visual(
            repaired_spec,
            preferred_renderer=renderer,
            allowed_renderers={renderer},
            render_root=repair_root / f"round_{round_index:02d}",
            width=width,
            height=height,
            fps=fps,
            renderer_strategy="first_success",
            tournament_size=1,
        )

    def extract_frames(
        candidate_spec: dict[str, object],
        candidate_asset: RenderedAsset,
        candidate_id: str,
    ) -> list[Path]:
        return _visual_director_frame_paths(
            candidate_spec,
            candidate_asset,
            output_dir=(
                render_root
                / "visual_director_frames"
                / safe_stem(candidate_id)
            ),
        )

    outcome = direct_rendered_visual(
        dict(spec),
        asset,
        selection_reason,
        ir=ir,
        contract=contract,
        render_candidate=render_candidate,
        evaluate_local_quality=_rendered_visual_quality_for_spec,
        extract_candidate_frames=extract_frames,
        strict=mode == "strict",
        max_repair_rounds=int(config.VISUAL_DIRECTOR_MAX_REPAIR_ROUNDS),
        minimum_repair_delta=float(config.VISUAL_DIRECTOR_MIN_REPAIR_DELTA),
        cache_dir=cache_dir,
        pairwise_top_k=int(config.VISUAL_DIRECTOR_PAIRWISE_TOP_K),
        target_publishable_candidates=int(
            config.VISUAL_DIRECTOR_RENDER_CANDIDATES
        ),
    )
    selected = outcome.selected
    report = outcome.to_dict()
    selected.asset.metadata = {
        **dict(selected.asset.metadata or {}),
        "visual_director_v2": report,
        "visual_quality_state": selected.verification.state.value,
    }
    _write_visual_director_report(selected.asset, report)
    merged_qa = _merge_visual_director_quality(selected.local_quality, outcome)
    return (
        dict(selected.spec),
        selected.asset,
        selected.selection_reason,
        merged_qa,
        report,
    )


def _visual_director_frame_paths(
    spec: dict[str, object],
    asset: RenderedAsset,
    *,
    output_dir: Path,
) -> list[Path]:
    artifact_paths = dict(asset.artifact_paths or {})
    existing: list[Path] = []
    for key in ("render_qa_frame_paths", "qa_frame_paths"):
        values = artifact_paths.get(key) or []
        if isinstance(values, (str, Path)):
            values = [values]
        for value in values:
            path = Path(str(value))
            if path.is_file() and path not in existing:
                existing.append(path)
    if len(existing) >= 4:
        return existing[:4]

    capture_plan = _selected_reference_capture_plan(spec)
    generated = extract_quality_frames(
        asset.asset_path,
        output_dir,
        duration_sec=max(float(asset.duration_sec or 0.0), 0.1),
        frame_count=4,
        capture_plan=capture_plan or None,
    )
    return generated or existing


def _selected_reference_capture_plan(
    spec: dict[str, object],
) -> list[dict[str, object]]:
    concept_search = dict(spec.get("visual_concept_search") or {})
    selected_concept_id = str(concept_search.get("selected_concept_id") or "")
    boards = [
        dict(item)
        for item in concept_search.get("reference_boards") or []
        if isinstance(item, dict)
    ]
    selected = next(
        (
            board
            for board in boards
            if str(board.get("concept_id") or "") == selected_concept_id
        ),
        boards[0] if boards else {},
    )
    return [
        {
            "capture_id": str(frame.get("frame_id") or f"frame_{index:02d}"),
            "fraction": _bounded(frame.get("fraction"), 0.0),
        }
        for index, frame in enumerate(selected.get("frames") or [], start=1)
        if isinstance(frame, dict)
    ][:4]


def _merge_visual_director_quality(
    local_qa: RenderedVisualQA,
    outcome: VisualDirectionOutcome,
) -> RenderedVisualQA:
    verification = outcome.selected.verification
    issues = list(outcome.issues) if not outcome.passed else []
    warnings = [*local_qa.warnings, *outcome.warnings]
    if outcome.passed and not local_qa.passed:
        warnings.extend(
            f"local_soft_gate_overridden:{issue}"
            for issue in local_qa.issues
            if issue not in outcome.selected.hard_local_issues
        )
    combined_score = _bounded(
        float(local_qa.score) * 0.42 + float(verification.score) * 0.58,
        0.0,
    )
    return RenderedVisualQA(
        visual_id=local_qa.visual_id,
        renderer=local_qa.renderer,
        score=round(combined_score, 4),
        passed=outcome.passed,
        issues=list(dict.fromkeys(str(item) for item in issues if str(item))),
        warnings=list(dict.fromkeys(str(item) for item in warnings if str(item))),
        repair_action=(
            "keep_visual_director_selected_candidate"
            if outcome.passed
            else "drop_visual_director_rejected_candidate"
        ),
        evidence={
            **dict(local_qa.evidence),
            "local_quality": local_qa.to_dict(),
            "visual_director_v2": {
                "version": outcome.version,
                "selected_candidate_id": outcome.selected.candidate_id,
                "quality_state": verification.state.value,
                "verifier_score": round(float(verification.score), 4),
                "candidate_count": len(outcome.candidates),
                "repair_round_count": len(outcome.repair_history),
            },
        },
    )


def _write_visual_director_report(
    asset: RenderedAsset,
    report: dict[str, object],
) -> None:
    try:
        report_path = Path(asset.job_dir) / "visual_director_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = report_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(report_path)
        asset.artifact_paths = {
            **dict(asset.artifact_paths or {}),
            "visual_director_report_path": str(report_path),
        }
    except OSError:
        return


def _overlay_from_rendered_visual(
    spec: dict[str, object],
    asset: RenderedAsset,
    *,
    selection_reason: str,
    force_fullscreen: bool,
    visual_qa_payload: dict[str, object],
) -> dict[str, object]:
    renderer_metadata = dict(asset.metadata or {})
    has_alpha = bool(renderer_metadata.get("has_alpha"))
    visual_world_program = dict(
        renderer_metadata.get("visual_world_program")
        or spec.get("visual_world_program")
        or {}
    )
    rendered_visual_fingerprint = dict(
        renderer_metadata.get("rendered_visual_fingerprint") or {}
    )
    requested_comp = str(spec.get("composition_mode") or "replace")
    compose_mode = (
        "replace"
        if force_fullscreen
        else (
            "overlay"
            if has_alpha
            and requested_comp in {"overlay", "picture_in_picture"}
            else requested_comp
        )
    )
    return {
        "start": _as_float(spec.get("start"), 0.0),
        "end": _as_float(spec.get("end"), 0.0),
        "asset_path": asset.asset_path,
        "compose_mode": compose_mode,
        "has_alpha": has_alpha,
        "force_fullscreen": force_fullscreen,
        "position": (
            "center"
            if compose_mode in {"replace", "overlay"}
            else spec["position"]
        ),
        "scale": (
            1.0
            if compose_mode in {"replace", "overlay"}
            else spec["scale"]
        ),
        "visual_id": spec["visual_id"],
        "card_id": spec["card_id"],
        "source_card_ids": list(spec.get("source_card_ids") or []),
        "semantic_episode_id": spec.get("semantic_episode_id"),
        "opportunity_contract": dict(spec.get("opportunity_contract") or {}),
        "opportunity_preflight": dict(spec.get("opportunity_preflight") or {}),
        "opportunity_recovery": dict(spec.get("opportunity_recovery") or {}),
        "template": spec["template"],
        "headline": spec["headline"],
        "emphasis_text": spec["emphasis_text"],
        "supporting_lines": spec.get("supporting_lines", []),
        "steps": spec.get("steps", []),
        "episode_id": spec.get("episode_id"),
        "visual_beats": spec.get("visual_beats", []),
        "program_context": spec.get("program_context", {}),
        "episode_context": spec.get("episode_context", {}),
        "concept_ids": spec.get("concept_ids", []),
        "continuity_group": spec.get("continuity_group"),
        "transition_in": spec.get("transition_in", {}),
        "transition_out": spec.get("transition_out", {}),
        "qa_contract": spec.get("qa_contract", {}),
        "auto_visual_skill": spec.get("auto_visual_skill", {}),
        "auto_visuals_director": spec.get("auto_visuals_director", {}),
        "hyperframes_compiler": spec.get("hyperframes_compiler", {}),
        "remotion_render": dict(renderer_metadata.get("remotion_render") or {}),
        "visual_explanation_ir": spec.get("visual_explanation_ir", {}),
        "visual_communication_contract": spec.get(
            "visual_communication_contract",
            {},
        ),
        "visual_concept_search": spec.get("visual_concept_search", {}),
        "open_visual_program": spec.get("open_visual_program", {}),
        "open_visual_tournament": spec.get("open_visual_tournament", {}),
        "visual_repair_history": spec.get("visual_repair_history", []),
        "hyperframes_storyboard": spec.get("hyperframes_storyboard", []),
        "hyperframes_production_contract": spec.get(
            "hyperframes_production_contract",
            {},
        ),
        "visual_claim_graph": spec.get("visual_claim_graph", {}),
        "visual_proof_tournament": spec.get("visual_proof_tournament", {}),
        "proof_program_id": spec.get("proof_program_id"),
        "proof_strategy_id": spec.get("proof_strategy_id"),
        "proof_encoding": spec.get("proof_encoding"),
        "visual_world_program": visual_world_program,
        "rendered_visual_fingerprint": rendered_visual_fingerprint,
        "semantic_continuity": spec.get("semantic_continuity", {}),
        "source_asset_grounding": spec.get("source_asset_grounding", {}),
        "user_visual_idea": spec.get("user_visual_idea"),
        "directed_visual_brief": spec.get("directed_visual_brief", {}),
        "visual_intent_type": spec.get("visual_intent_type"),
        "rendered_visual_qa": visual_qa_payload,
        "creative_graph_signals": spec.get("creative_graph_signals", {}),
        "quote_text": spec.get("quote_text"),
        "left_label": spec.get("left_label"),
        "right_label": spec.get("right_label"),
        "left_detail": spec.get("left_detail"),
        "right_detail": spec.get("right_detail"),
        "footer_text": spec.get("footer_text"),
        "sentence_text": spec["sentence_text"],
        "context_text": spec["context_text"],
        "planning_context_text": spec.get("planning_context_text"),
        "keywords": spec["keywords"],
        "visual_type_hint": spec["visual_type_hint"],
        "style_pack": spec.get("style_pack"),
        "theme": spec["theme"],
        "confidence": spec["confidence"],
        "rationale": spec["rationale"],
        "renderer": asset.renderer,
        "renderer_hint": spec.get("renderer_hint"),
        "renderer_selection_reason": selection_reason,
        "motion_preset": spec.get("motion_preset"),
        "importance": spec.get("importance"),
        "evidence": spec.get("evidence"),
        "renderer_job_dir": asset.job_dir,
        "renderer_script_path": asset.script_path,
        "renderer_artifact_paths": dict(asset.artifact_paths or {}),
        "renderer_metadata": renderer_metadata,
        "rendered_width": asset.width,
        "rendered_height": asset.height,
        "rendered_duration_sec": asset.duration_sec,
    }


def _transition_with_default(
    value: object,
    *,
    duration_sec: float,
    direction: str,
) -> dict[str, object]:
    if isinstance(value, dict) and str(value.get("kind") or "").strip().lower() not in {"", "hard_cut", "scene_match_cut"}:
        return dict(value)
    if duration_sec < 1.0:
        return {}
    transition_duration = max(0.1, min(0.26, duration_sec * 0.08))
    return {
        "kind": "soft_dissolve",
        "direction": direction,
        "duration_sec": round(transition_duration, 3),
    }


def _apply_visual_world_diversity_gate(
    overlays: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    for overlay in sorted(
        overlays,
        key=lambda item: _as_float(item.get("start"), 0.0),
    ):
        world = dict(overlay.get("visual_world_program") or {})
        identity = extract_visual_portfolio_identity(dict(overlay))
        fingerprint = dict(
            overlay.get("rendered_visual_fingerprint") or {}
        )
        world_fingerprint = dict(world.get("fingerprint") or {})
        medium = str(world.get("medium_family") or "")
        background = str(world.get("background_mode") or "")
        fingerprint_signature = str(fingerprint.get("signature") or "")
        rejection_reason = ""
        compared_to = ""
        perceptual_distance: float | None = None
        if (
            str(world.get("card_policy") or "") == "forbidden"
            and _as_float(
                world_fingerprint.get("panel_ratio_target"),
                0.0,
            )
            > 0.08
        ):
            rejection_reason = "forbidden_card_surface_ratio"
        for recent in accepted[-3:]:
            if rejection_reason:
                break
            recent_world = dict(recent.get("visual_world_program") or {})
            recent_fingerprint = dict(
                recent.get("rendered_visual_fingerprint") or {}
            )
            recent_identity = extract_visual_portfolio_identity(dict(recent))
            recent_signature = str(
                recent_fingerprint.get("signature") or ""
            )
            same_world = (
                medium
                and medium == str(recent_world.get("medium_family") or "")
                and background
                and background
                == str(recent_world.get("background_mode") or "")
            )
            if (
                identity.program_signature
                and identity.program_signature == recent_identity.program_signature
            ):
                rejection_reason = "duplicate_open_visual_program"
            elif (
                fingerprint_signature
                and recent_signature
                and fingerprint_signature == recent_signature
            ):
                rejection_reason = "duplicate_rendered_visual_fingerprint"
            elif same_world or same_creative_grammar(identity, recent_identity):
                perceptual_distance = visual_fingerprint_distance(
                    fingerprint,
                    recent_fingerprint,
                )
                if perceptual_distance < (0.26 if same_creative_grammar(identity, recent_identity) else 0.2):
                    rejection_reason = (
                        "repeated_creative_grammar_too_similar"
                        if same_creative_grammar(identity, recent_identity)
                        else "repeated_visual_world_too_similar"
                    )
            if rejection_reason:
                compared_to = str(recent.get("visual_id") or "")
        if rejection_reason:
            rejected.append(
                {
                    "visual_id": str(overlay.get("visual_id") or ""),
                    "reason": rejection_reason,
                    "compared_to_visual_id": compared_to,
                    "medium_family": medium,
                    "background_mode": background,
                    "perceptual_distance": perceptual_distance,
                    "creative_identity": identity.to_dict(),
                    "selection_stage": "visual_world_diversity_gate",
                }
            )
            continue
        accepted.append(overlay)
    return accepted, rejected


def _final_auto_visuals_qa(
    overlays: list[dict[str, object]],
    *,
    clip_duration: float,
    coverage_policy: str = "quality_only",
) -> tuple[list[dict[str, object]], dict[str, object]]:
    survivors: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    sorted_overlays = sorted(overlays, key=lambda item: _as_float(item.get("start"), 0.0))
    for overlay in sorted_overlays:
        normalized = dict(overlay)
        visual_id = str(normalized.get("visual_id") or "")
        start = _as_float(normalized.get("start"), 0.0)
        end = _as_float(normalized.get("end"), start)
        duration = end - start
        qa = dict(normalized.get("rendered_visual_qa") or {})
        director = dict(normalized.get("auto_visuals_director") or {})
        if start < 0 or end > clip_duration + 0.05 or duration < 0.7:
            rejected.append(
                {
                    "visual_id": visual_id,
                    "reason": "invalid_or_too_short_timing",
                    "start": start,
                    "end": end,
                }
            )
            continue
        if not bool(qa.get("passed", True)):
            rejected.append(
                {
                    "visual_id": visual_id,
                    "reason": "rendered_visual_qa_failed",
                    "issues": qa.get("issues") or [],
                }
            )
            continue
        if _as_float(director.get("visual_need"), 0.6) < 0.34 and _as_float(director.get("source_richness"), 0.0) > 0.78:
            rejected.append(
                {
                    "visual_id": visual_id,
                    "reason": "source_moment_already_visually_rich",
                    "source_richness": director.get("source_richness"),
                }
            )
            continue
        survivors.append(normalized)
    accepted, set_optimization = optimize_creative_set(
        survivors,
        budget=len(survivors),
        min_gap_sec=0.2,
        phase="rendered",
        coverage_policy=coverage_policy,
    )
    for item in set_optimization.get("rejected", []):
        rejected.append(
            {
                "visual_id": item.get("candidate_id"),
                "reason": item.get("reason"),
                "start": item.get("start"),
                "conflicting_visual_ids": item.get("conflicting_visual_ids") or [],
                "selection_stage": "post_render_creative_set_optimizer",
            }
        )
    accepted, diversity_rejected = _apply_visual_world_diversity_gate(
        accepted
    )
    rejected.extend(diversity_rejected)
    portfolio_report = evaluate_visual_portfolio(accepted).to_dict()
    for normalized in accepted:
        start = _as_float(normalized.get("start"), 0.0)
        end = _as_float(normalized.get("end"), start)
        duration = end - start
        if str(normalized.get("compose_mode") or "") == "replace":
            normalized["transition_in"] = _transition_with_default(
                normalized.get("transition_in"),
                duration_sec=duration,
                direction="in",
            )
            normalized["transition_out"] = _transition_with_default(
                normalized.get("transition_out"),
                duration_sec=duration,
                direction="out",
            )
    report = {
        "version": AUTO_VISUALS_DIRECTOR_VERSION,
        "input_count": len(overlays),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "rejected": rejected[:12],
        "average_rendered_score": round(
            sum(_as_float((item.get("rendered_visual_qa") or {}).get("score"), 0.0) for item in accepted)
            / max(len(accepted), 1),
            4,
        ),
        "set_optimization": set_optimization,
        "visual_world_diversity": {
            "accepted_count": len(accepted),
            "rejected_count": len(diversity_rejected),
            "rejected": diversity_rejected,
        },
        "creative_portfolio": portfolio_report,
        "transition_policy": "soft_dissolve_for_fullscreen_replacements",
    }
    return accepted, report


def _render_generated_visual(
    spec: dict[str, object],
    *,
    preferred_renderer: str,
    render_root: Path,
    width: int,
    height: int,
    fps: float,
    allowed_renderers: set[str] | None = None,
    renderer_strategy: str | None = None,
    tournament_size: int | None = None,
) -> tuple[RenderedAsset, str]:
    failures: list[str] = []
    attempted: set[str] = set()
    require_generated_scene = bool(spec.get("require_generated_scene"))
    known_renderers = {renderer.name for renderer in list_renderers()}
    base_excluded = known_renderers - allowed_renderers if allowed_renderers is not None else set()
    preferred = _normalize_renderer_name(preferred_renderer)
    spec_hint = str(spec.get("renderer_hint") or "auto").strip().lower()
    strategy = _normalize_renderer_strategy(renderer_strategy, preferred)
    if (
        strategy == "quality_tournament"
        and not require_generated_scene
        and (allowed_renderers is None or len(allowed_renderers) > 1)
    ):
        return _render_with_quality_tournament(
            spec,
            preferred_renderer=preferred,
            spec_hint=spec_hint,
            render_root=render_root,
            width=width,
            height=height,
            fps=fps,
            allowed_renderers=allowed_renderers,
            tournament_size=tournament_size,
        )
    if require_generated_scene:
        locked_preference = (
            str(spec.get("renderer_hint") or preferred or "manim").strip().lower()
            or "manim"
        )
        if allowed_renderers is not None and locked_preference not in allowed_renderers:
            raise VisualRendererError(
                f"{locked_preference}: renderer is not allowed for this auto-visuals run."
            )
        preference_order = [locked_preference]
    else:
        preference_order = []
        for candidate in (preferred, spec_hint, "auto"):
            if candidate == "both":
                candidate = "auto"
            if candidate != "auto" and allowed_renderers is not None and candidate not in allowed_renderers:
                continue
            if candidate not in preference_order:
                preference_order.append(candidate)
        if not preference_order:
            preference_order = ["auto"]
    for candidate_preference in preference_order:
        while True:
            try:
                renderer, reason = resolve_renderer(
                    spec,
                    preferred=candidate_preference,
                    exclude=attempted | base_excluded,
                )
            except VisualRendererError as exc:
                failures.append(str(exc))
                break
            attempted.add(renderer.name)
            if (
                require_generated_scene
                and renderer.name
                != str(spec.get("renderer_hint") or "manim").strip().lower()
            ):
                failures.append(
                    f"{renderer.name}: premium generated scenes are locked to {spec.get('renderer_hint') or 'manim'}."
                )
                break
            try:
                asset = render_with_manifest(
                    renderer,
                    spec,
                    render_root=render_root,
                    width=width,
                    height=height,
                    fps=fps,
                )
                return asset, reason
            except VisualRendererError as exc:
                failures.append(f"{renderer.name}: {exc}")
                if len(attempted | base_excluded) >= len(known_renderers):
                    break
        if len(attempted | base_excluded) >= len(known_renderers):
            break
    raise VisualRendererError(
        "; ".join(dict.fromkeys(failures))
        or "No renderer could produce the generated visual."
    )


def _normalize_renderer_strategy(value: object, preferred_renderer: str) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    if normalized in {"first_success", "strict"}:
        return "first_success"
    if normalized in {"quality_tournament", "tournament", "best_quality"}:
        return "quality_tournament"
    return (
        "quality_tournament"
        if preferred_renderer in {"auto", "both"}
        else "first_success"
    )


def _renderer_is_semantically_eligible(
    match: RendererMatch,
    spec: dict[str, object],
) -> bool:
    renderer_name = match.renderer.name
    intent_type = str(spec.get("visual_intent_type") or "").strip().lower()
    renderer_hint = str(spec.get("renderer_hint") or "").strip().lower()
    if renderer_name == "manim" and intent_type != "math_or_formula":
        return False
    if renderer_name == "remotion" and intent_type in {"math_or_formula", "spatial_3d"} and renderer_hint != "remotion":
        return False
    if renderer_name == "blender" and intent_type != "spatial_3d" and renderer_hint != "blender":
        return False
    return True


def _render_with_quality_tournament(
    spec: dict[str, object],
    *,
    preferred_renderer: str,
    spec_hint: str,
    render_root: Path,
    width: int,
    height: int,
    fps: float,
    allowed_renderers: set[str] | None,
    tournament_size: int | None,
) -> tuple[RenderedAsset, str]:
    known_renderers = {renderer.name for renderer in list_renderers()}
    excluded = known_renderers - allowed_renderers if allowed_renderers is not None else set()
    ranking_preference = (
        spec_hint
        if spec_hint not in {"", "auto", "both"}
        and (allowed_renderers is None or spec_hint in allowed_renderers)
        else ("auto" if preferred_renderer == "both" else preferred_renderer)
    )
    matches = [
        match
        for match in rank_renderers(
            spec,
            preferred=ranking_preference,
            exclude=excluded,
        )
        if _renderer_is_semantically_eligible(match, spec)
    ]
    policy_prior = dict(spec.get("creative_policy_prior") or {})
    renderer_adjustments = dict(policy_prior.get("renderer_adjustments") or {})
    matches.sort(
        key=lambda match: (
            -(
                match.score
                + _as_float(renderer_adjustments.get(match.renderer.name), 0.0)
            ),
            match.renderer.name,
        )
    )
    if not matches:
        raise VisualRendererError(
            "No semantically compatible renderer was available for the quality tournament."
        )
    contender_limit = max(
        1,
        min(
            int(tournament_size or config.AUTO_VISUALS_RENDERER_TOURNAMENT_SIZE),
            3,
        ),
    )
    attempts: list[dict[str, object]] = []
    rendered: list[tuple[RendererMatch, RenderedAsset, RenderedVisualQA]] = []
    for match in matches:
        if len(rendered) >= contender_limit:
            break
        contender_root = render_root / "renderer_tournaments" / match.renderer.name
        try:
            asset = render_with_manifest(
                match.renderer,
                spec,
                render_root=contender_root,
                width=width,
                height=height,
                fps=fps,
            )
            qa = _rendered_visual_quality_for_spec(spec, asset)
            rendered.append((match, asset, qa))
            attempts.append(
                {
                    "renderer": match.renderer.name,
                    "resolver_score": match.score,
                    "policy_adjustment": _as_float(
                        renderer_adjustments.get(match.renderer.name),
                        0.0,
                    ),
                    "rendered": True,
                    "asset_path": asset.asset_path,
                    "qa": qa.to_dict(),
                }
            )
        except VisualRendererError as exc:
            attempts.append(
                {
                    "renderer": match.renderer.name,
                    "resolver_score": match.score,
                    "policy_adjustment": _as_float(
                        renderer_adjustments.get(match.renderer.name),
                        0.0,
                    ),
                    "rendered": False,
                    "error": str(exc),
                }
            )
    if not rendered:
        details = "; ".join(
            f"{item['renderer']}: {item.get('error', 'render failed')}" for item in attempts
        )
        raise VisualRendererError(
            details or "No renderer produced a contender for the quality tournament."
        )
    selected_match, selected_asset, selected_qa = max(
        rendered,
        key=lambda item: (
            int(item[2].passed),
            item[2].score,
            item[0].score,
            item[0].renderer.name,
        ),
    )
    tournament_report = {
        "version": RENDERER_TOURNAMENT_VERSION,
        "strategy": "quality_tournament",
        "requested_contenders": contender_limit,
        "attempted_count": len(attempts),
        "rendered_count": len(rendered),
        "selected_renderer": selected_match.renderer.name,
        "selected_qa_score": selected_qa.score,
        "selected_qa_passed": selected_qa.passed,
        "attempts": attempts,
    }
    selected_asset.metadata = {
        **dict(selected_asset.metadata or {}),
        "renderer_tournament": tournament_report,
    }
    reason = (
        f"Quality tournament promoted {selected_match.renderer.name} "
        f"with render QA {selected_qa.score:.3f} "
        f"from {len(rendered)} rendered contender"
        f"{'s' if len(rendered) != 1 else ''}."
    )
    return selected_asset, reason


def _max_render_workers(
    params: dict, visual_count: int, specs: list[dict[str, object]] | None = None
) -> int:
    specs = specs or []
    renderer_name = _normalize_renderer_name(params.get("renderer"))
    if renderer_name == "blender" or any(
        str(spec.get("renderer_hint") or "").strip().lower() == "blender"
        for spec in specs
    ):
        return 1
    if renderer_name in {"auto", "both", "hyperframes", "remotion"} or any(
        str(spec.get("renderer_hint") or "").strip().lower()
        in {"hyperframes", "remotion"}
        for spec in specs
    ):
        return 1
    requested = int(params.get("max_render_workers", 4) or 4)
    return max(1, min(requested, visual_count, 4))


def _contextual_visual_budget(
    cards: list[dict[str, object]],
    *,
    clip_duration: float,
    renderer_name: str,
    mode: str,
    density: str = "balanced",
) -> int:
    if not cards:
        return 1
    high_signal = 0
    for card in cards:
        visualizability = _as_float(card.get("visualizability"), 0.0)
        payoff = _as_float(card.get("intuition_payoff"), 0.0)
        explicit_signal = (
            int(_as_float(card.get("numeric_hits"), 0.0)) > 0
            or _as_float(card.get("process_cues"), 0.0) >= 0.22
            or _as_float(card.get("contrast_cues"), 0.0) >= 0.22
        )
        if explicit_signal or (visualizability >= 0.5 and payoff >= 0.54):
            high_signal += 1
    duration_budget = max(3, round(clip_duration / 11.0))
    if clip_duration >= 120:
        duration_budget += 2
    elif clip_duration >= 60:
        duration_budget += 1
    signal_budget = max(2, min(high_signal, len(cards)))
    budget = max(duration_budget, signal_budget)
    if renderer_name in {"hyperframes", "remotion", "both", "auto"} and mode == "generated_only":
        budget += 2
    normalized_density = normalize_density(density, clip_duration=clip_duration)
    if normalized_density == "sparse":
        budget = max(3, round(clip_duration / 28.0), min(high_signal, 5))
    elif normalized_density == "dense":
        budget = max(budget, round(clip_duration / 8.0), high_signal)
    elif normalized_density == "chapter_coverage":
        chapter_budget = max(4, round(clip_duration / 45.0))
        budget = max(budget, chapter_budget, min(len(cards), high_signal + chapter_budget // 2))
    return max(1, min(budget, int(config.AUTO_VISUALS_MAX_VISUALS), len(cards)))


def _creative_outcome_signals(
    plan: list[dict[str, object]],
    rendered_visual_qa: list[dict[str, object]],
    overlays: list[dict[str, object]],
) -> list[dict[str, object]]:
    spec_by_id = {
        str(spec.get("visual_id") or ""): spec
        for spec in plan
        if str(spec.get("visual_id") or "")
    }
    published_ids = {
        str(overlay.get("visual_id") or "")
        for overlay in overlays
        if str(overlay.get("visual_id") or "")
    }
    signals: list[dict[str, object]] = []
    for qa_payload in rendered_visual_qa:
        visual_id = str(qa_payload.get("visual_id") or "")
        spec = spec_by_id.get(visual_id, {})
        visual_direction = dict(qa_payload.get("visual_director_v2") or {})
        selected_candidate_id = str(
            visual_direction.get("selected_candidate_id") or ""
        )
        selected_candidate = next(
            (
                dict(item)
                for item in visual_direction.get("candidates") or []
                if isinstance(item, dict)
                and str(item.get("candidate_id") or "") == selected_candidate_id
            ),
            {},
        )
        creative_identity = dict(
            selected_candidate.get("creative_identity") or {}
        )
        tournament = dict(qa_payload.get("renderer_tournament") or {})
        promoted_renderer = str(
            tournament.get("selected_renderer")
            or qa_payload.get("renderer")
            or ""
        ).strip().lower()
        attempts = [
            item
            for item in _as_list(tournament.get("attempts"))
            if isinstance(item, dict)
            and bool(item.get("rendered"))
            and isinstance(item.get("qa"), dict)
        ]
        if attempts:
            qa_records = [
                {
                    **dict(item.get("qa") or {}),
                    "renderer": item.get("renderer"),
                }
                for item in attempts
            ]
        else:
            qa_records = [qa_payload]
        for qa in qa_records:
            renderer = str(qa.get("renderer") or "").strip().lower()
            if not renderer:
                continue
            signals.append(
                {
                    "renderer": renderer,
                    "intent_type": str(
                        spec.get("visual_intent_type") or "unknown"
                    ).strip().lower(),
                    "template": str(spec.get("template") or "unknown").strip().lower(),
                    "qa_score": round(_bounded(qa.get("score"), 0.0), 4),
                    "qa_passed": bool(qa.get("passed")),
                    "published": (
                        visual_id in published_ids
                        and renderer == promoted_renderer
                    ),
                    "visual_quality_state": str(
                        visual_direction.get("selected_quality_state") or "legacy"
                    ),
                    "verifier_score": round(
                        _bounded(
                            visual_direction.get("selected_verifier_score"),
                            0.0,
                        ),
                        4,
                    ),
                    "repair_round_count": len(
                        _as_list(visual_direction.get("repair_history"))
                    ),
                    "concept_lane": str(
                        creative_identity.get("lane") or "unknown"
                    ),
                    "concept_medium": str(
                        creative_identity.get("medium") or "unknown"
                    ),
                    "motion_grammar": str(
                        creative_identity.get("motion_grammar") or "unknown"
                    ),
                }
            )
    return signals[:64]


def _manual_visual_specs_from_params(params: dict) -> list[dict[str, object]]:
    raw_specs = (
        params.get("manual_visual_specs")
        or params.get("visual_specs")
        or params.get("specs")
    )
    if isinstance(raw_specs, dict):
        raw_specs = [raw_specs]
    if not isinstance(raw_specs, list):
        return []
    return [dict(item) for item in raw_specs if isinstance(item, dict)]


def _directed_hyperframes_specs_from_params(
    params: dict,
) -> list[dict[str, object]]:
    raw_specs = (
        params.get("directed_visual_specs")
        or params.get("hyperframes_visual_specs")
        or params.get("directed_hyperframes_specs")
    )
    specs: list[dict[str, object]] = []
    if isinstance(raw_specs, dict):
        raw_specs = [raw_specs]
    if isinstance(raw_specs, list):
        specs.extend(dict(item) for item in raw_specs if isinstance(item, dict))

    scalar_idea = _clean_directed_text(
        params.get("visual_idea")
        or params.get("hyperframes_visual_idea")
        or params.get("visual_brief"),
        max_chars=360,
    )
    if scalar_idea:
        scalar_spec: dict[str, object] = {
            "visual_idea": scalar_idea,
            "renderer_hint": "hyperframes",
        }
        for key in (
            "start",
            "start_sec",
            "end",
            "end_sec",
            "duration",
            "duration_sec",
            "trigger_text",
            "trigger",
            "composition_mode",
            "style_pack",
            "semantic_frame",
            "required_labels",
            "metric_facts",
        ):
            if key in params:
                scalar_spec[key] = params[key]
        specs.insert(0, scalar_spec)

    for item in _manual_visual_specs_from_params(params):
        renderer_hint = str(item.get("renderer_hint") or "").strip().lower()
        has_directed_idea = bool(
            _clean_directed_text(
                item.get("visual_idea")
                or item.get("hyperframes_visual_idea")
                or item.get("visual_brief"),
                max_chars=360,
            )
        )
        if (renderer_hint == "hyperframes" and has_directed_idea) or (
            has_directed_idea
            and str(params.get("renderer") or "").strip().lower() == "hyperframes"
        ):
            specs.append(dict(item))
    return specs


def _clean_directed_text(value: object, *, max_chars: int) -> str:
    cleaned = re.sub(r"\s+", " ", str(value or "")).strip(" \t\r\n\"'")
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max(0, max_chars - 1)].rstrip(" ,.;:") + "..."


def _parse_directed_time_seconds(value: object, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else default
    raw = str(value or "").strip().lower()
    if not raw:
        return default
    try:
        if ":" in raw:
            parts = [float(part) for part in raw.split(":")]
            if len(parts) == 2:
                minutes, seconds = parts
                return minutes * 60.0 + seconds
            if len(parts) == 3:
                hours, minutes, seconds = parts
                return hours * 3600.0 + minutes * 60.0 + seconds
        match = re.fullmatch(
            r"(\d+(?:\.\d+)?)\s*(milliseconds?|msecs?|ms|seconds?|secs?|sec|s|minutes?|mins?|min|m|hours?|hrs?|hr|h)?",
            raw,
        )
        if not match:
            return default
        number = float(match.group(1))
        unit = str(match.group(2) or "s").strip().lower()
        if unit in {"ms", "msec", "msecs"} or unit.startswith("millisecond"):
            return number / 1000.0
        if unit in {"m", "min", "mins"} or unit.startswith("minute"):
            return number * 60.0
        if unit in {"h", "hr", "hrs"} or unit.startswith("hour"):
            return number * 3600.0
        return number
    except (TypeError, ValueError):
        return default


def _directed_token_set(value: object) -> set[str]:
    return {
        token
        for token in _word_tokens(value)
        if len(token) >= 3 and token not in _DIRECTED_STOPWORDS
    }


def _unique_directed_strings(values: list[object], *, limit: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = _clean_directed_text(value, max_chars=96)
        key = " ".join(_word_tokens(cleaned))
        if not cleaned or not key or key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
        if len(result) >= limit:
            break
    return result


def _directed_transcript_windows(
    transcript_bundle: dict[str, object],
) -> list[dict[str, object]]:
    raw_windows = _as_list(transcript_bundle.get("sentences")) or _as_list(
        transcript_bundle.get("segments")
    )
    windows: list[dict[str, object]] = []
    for index, raw in enumerate(raw_windows, start=1):
        if not isinstance(raw, dict):
            continue
        text = _clean_directed_text(raw.get("text"), max_chars=520)
        if not text:
            continue
        start = _parse_directed_time_seconds(raw.get("start"), 0.0) or 0.0
        end = _parse_directed_time_seconds(raw.get("end"), None)
        if end is None or end <= start:
            end = start + max(0.75, min(len(_word_tokens(text)) / 2.7, 8.0))
        windows.append(
            {
                "id": str(
                    raw.get("card_id")
                    or raw.get("sentence_id")
                    or raw.get("id")
                    or f"subtitle_{index:03d}"
                ),
                "start": round(max(start, 0.0), 3),
                "end": round(max(end, start + 0.1), 3),
                "text": text,
                "index": index - 1,
            }
        )
    return sorted(windows, key=lambda item: _as_float(item.get("start"), 0.0))


def _directed_window_score(
    window: dict[str, object],
    *,
    idea_tokens: set[str],
) -> float:
    text = str(window.get("text") or "")
    text_tokens = _directed_token_set(text)
    overlap = (
        len(idea_tokens & text_tokens) / max(len(idea_tokens), 1)
        if idea_tokens
        else 0.0
    )
    process_bonus = (
        0.18
        if re.search(
            r"\b(?:from|to|then|next|after|before|because|therefore|leads?\s+to|turns?\s+into|compress(?:es|ing)?|routes?|passes?|reduces?|increases?)\b",
            text,
            flags=re.IGNORECASE,
        )
        else 0.0
    )
    metric_bonus = 0.16 if _DIRECTED_METRIC_RE.search(text) else 0.0
    duration = _as_float(window.get("end"), 0.0) - _as_float(window.get("start"), 0.0)
    duration_bonus = 0.08 if 1.4 <= duration <= 9.0 else 0.0
    density_penalty = 0.08 if len(_word_tokens(text)) < 4 else 0.0
    return round(overlap * 0.58 + process_bonus + metric_bonus + duration_bonus - density_penalty, 4)


def _select_directed_window(
    windows: list[dict[str, object]],
    *,
    idea: str,
    trigger: str,
) -> dict[str, object] | None:
    if not windows:
        return None
    if trigger:
        trigger_key = trigger.lower()
        for window in windows:
            if trigger_key in str(window.get("text") or "").lower():
                return window
    idea_tokens = _directed_token_set(idea)
    ranked = sorted(
        windows,
        key=lambda item: (
            _directed_window_score(item, idea_tokens=idea_tokens),
            -abs(_as_float(item.get("start"), 0.0)),
        ),
        reverse=True,
    )
    return ranked[0] if ranked else None


def _windows_for_directed_range(
    windows: list[dict[str, object]],
    *,
    start: float,
    end: float,
    allow_nearest: bool = True,
) -> list[dict[str, object]]:
    overlapping = [
        item
        for item in windows
        if _as_float(item.get("end"), 0.0) > start - 0.08
        and _as_float(item.get("start"), 0.0) < end + 0.08
    ]
    if overlapping:
        return overlapping
    if not allow_nearest:
        return []
    if not windows:
        return []
    center = (start + end) / 2.0
    nearest = min(
        windows,
        key=lambda item: abs(
            ((_as_float(item.get("start"), 0.0) + _as_float(item.get("end"), 0.0)) / 2.0)
            - center
        ),
    )
    return [nearest]


def _directed_context_payload(
    windows: list[dict[str, object]],
    selected: list[dict[str, object]],
) -> dict[str, object]:
    if not selected:
        return {"sentence_text": "", "context_text": "", "source_card_ids": []}
    ordered = sorted(selected, key=lambda item: int(item.get("index") or 0))
    selected_indexes = {int(item.get("index") or 0) for item in ordered}
    context_indexes = {
        index
        for selected_index in selected_indexes
        for index in (selected_index - 1, selected_index, selected_index + 1)
    }
    context_windows = [
        item for item in windows if int(item.get("index") or 0) in context_indexes
    ]
    sentence_text = _clean_directed_text(
        " ".join(str(item.get("text") or "") for item in ordered),
        max_chars=520,
    )
    context_text = _clean_directed_text(
        " ".join(str(item.get("text") or "") for item in context_windows),
        max_chars=760,
    )
    return {
        "sentence_text": sentence_text,
        "context_text": context_text or sentence_text,
        "source_card_ids": [str(item.get("id") or "") for item in ordered if str(item.get("id") or "")],
    }


def _directed_metric_facts(source_text: str) -> list[dict[str, object]]:
    metrics: list[dict[str, object]] = []
    for match in _DIRECTED_METRIC_RE.finditer(source_text):
        value = _clean_directed_text(match.group(0), max_chars=32)
        unit = str(match.group("unit") or "").strip().lower()
        if not value:
            continue
        if not unit and not re.search(
            r"\b(?:from|to|only|just|drops?|falls?|rises?|increases?|decreases?|reduces?|ratio|compressed|blocks?|tokens?)\b",
            source_text,
            flags=re.IGNORECASE,
        ):
            continue
        metrics.append({"value": value, "label": value})
        if len(metrics) >= 4:
            break
    return metrics


def _directed_steps_from_text(source_text: str) -> list[str]:
    pieces = re.split(
        r"\s*(?:,|;|\bthen\b|\bnext\b|\bafter that\b|\bfinally\b|\band then\b)\s*",
        source_text,
        flags=re.IGNORECASE,
    )
    steps = [
        _clean_directed_text(piece, max_chars=72).strip(" .,:;")
        for piece in pieces
        if len(_word_tokens(piece)) >= 2
    ]
    return _unique_directed_strings(steps, limit=5)


def _grounded_semantic_frame_from_raw(
    raw_frame: object,
    *,
    source_text: str,
) -> dict[str, object]:
    if not isinstance(raw_frame, dict):
        return {}
    allowed_keys = {
        "before_state",
        "after_state",
        "intervention",
        "action",
        "turn",
        "problem",
        "cause",
        "mechanism",
        "result",
        "effect",
        "payoff",
        "viewer_takeaway",
        "steps",
        "stages",
        "decision",
        "low_branch",
        "high_branch",
        "exact_quote",
        "setup",
        "screen",
        "focus",
        "constraint",
        "preserved_constraint",
    }
    grounded: dict[str, object] = {}
    for key, value in raw_frame.items():
        if str(key) not in allowed_keys:
            continue
        if isinstance(value, list):
            values = [
                item
                for item in _unique_directed_strings([str(item) for item in value], limit=6)
                if _copy_is_source_grounded(item, source_text)
            ]
            if values:
                grounded[str(key)] = values
        else:
            cleaned = _clean_directed_text(value, max_chars=96)
            if cleaned and _copy_is_source_grounded(cleaned, source_text):
                grounded[str(key)] = cleaned
    return grounded


def _derive_directed_semantic_frame(
    source_text: str,
    *,
    idea: str,
) -> dict[str, object]:
    source = _clean_directed_text(source_text, max_chars=760)
    frame: dict[str, object] = {}
    from_to = re.search(
        r"\bfrom\s+(.{2,84}?)\s+to\s+(.{2,84}?)(?:[.;,]|$)",
        source,
        flags=re.IGNORECASE,
    )
    if from_to:
        before = _clean_directed_text(from_to.group(1), max_chars=72).strip(" .,:;")
        after = _clean_directed_text(from_to.group(2), max_chars=72).strip(" .,:;")
        if before and after:
            frame["before_state"] = before
            frame["after_state"] = after
    intervention = re.search(
        r"\b(?:after|when|once)\s+([^.;,]{3,96})",
        source,
        flags=re.IGNORECASE,
    )
    if intervention:
        frame["intervention"] = _clean_directed_text(
            intervention.group(1),
            max_chars=72,
        ).strip(" .,:;")
    leads_to = re.search(
        r"\b(.{3,96}?)\s+(?:leads?\s+to|results?\s+in|causes?|therefore)\s+(.{3,96}?)(?:[.;,]|$)",
        source,
        flags=re.IGNORECASE,
    )
    if leads_to:
        frame.setdefault(
            "mechanism",
            _clean_directed_text(leads_to.group(1), max_chars=72).strip(" .,:;"),
        )
        frame.setdefault(
            "result",
            _clean_directed_text(leads_to.group(2), max_chars=72).strip(" .,:;"),
        )
    because = re.search(
        r"\b(.{3,96}?)\s+because\s+(.{3,96}?)(?:[.;,]|$)",
        source,
        flags=re.IGNORECASE,
    )
    if because:
        frame.setdefault(
            "result",
            _clean_directed_text(because.group(1), max_chars=72).strip(" .,:;"),
        )
        frame.setdefault(
            "cause",
            _clean_directed_text(because.group(2), max_chars=72).strip(" .,:;"),
        )
    steps = _directed_steps_from_text(source)
    if len(steps) >= 2:
        frame.setdefault("steps", steps)
    if (
        re.search(r"\b(?:quote|typography|words?|text)\b", idea, flags=re.IGNORECASE)
        and 4 <= len(_word_tokens(source)) <= 22
    ):
        frame.setdefault("exact_quote", source)
    return {
        key: value
        for key, value in frame.items()
        if not isinstance(value, str) or _copy_is_source_grounded(value, source)
    }


def _directed_required_labels(
    semantic_frame: dict[str, object],
    metric_facts: list[dict[str, object]],
    *,
    source_text: str,
) -> list[str]:
    labels: list[object] = []
    for value in semantic_frame.values():
        if isinstance(value, list):
            labels.extend(value)
        else:
            labels.append(value)
    labels.extend(item.get("label") or item.get("value") for item in metric_facts)
    return [
        label
        for label in _unique_directed_strings(labels, limit=8)
        if _copy_is_source_grounded(label, source_text)
    ]


def _preferred_directed_medium_family(idea: str) -> str:
    lowered = str(idea or "").lower()
    hints: tuple[tuple[str, str], ...] = (
        (r"\b(?:particle|particles|swarm|token|tokens|compress|compression|blocks?|data\s+flow|gradient|mass)\b", "data_sculpture"),
        (r"\b(?:gate|gates|tunnel|path|journey|chamber|stage|world|spatial|maze|corridor)\b", "spatial_metaphor"),
        (r"\b(?:diagram|network|node|nodes|edge|edges|graph|pipeline|architecture|system|arrows?|flowchart)\b", "diagrammatic_system"),
        (r"\b(?:quote|typography|big\s+text|kinetic\s+text|words?|text\s+animation)\b", "kinetic_typography"),
        (r"\b(?:collage|magazine|paper|cutout|editorial|poster)\b", "editorial_collage"),
        (r"\b(?:ui|interface|dashboard|screen|app|window|product)\b", "product_interface"),
    )
    for pattern, medium in hints:
        if re.search(pattern, lowered):
            return medium
    return ""


def _directed_template_hint(preferred_medium: str, idea: str) -> str:
    if preferred_medium == "data_sculpture":
        return "data_journey"
    if preferred_medium == "spatial_metaphor":
        return "kinetic_route"
    if preferred_medium == "diagrammatic_system":
        return "system_flow"
    if preferred_medium == "kinetic_typography":
        return "ribbon_quote"
    if preferred_medium == "editorial_collage":
        return "narrative_arc"
    if preferred_medium == "product_interface":
        return "interface_cascade"
    if re.search(r"\b(?:compare|versus|vs|before|after)\b", idea, flags=re.IGNORECASE):
        return "split_compare"
    return "mechanism_blueprint"


def _normalize_directed_hyperframes_specs(
    raw_specs: list[dict[str, object]],
    *,
    transcript_bundle: dict[str, object],
    clip_duration: float,
    width: int,
    height: int,
    fps: float,
    force_fullscreen: bool,
    max_visuals: int,
    min_visual_sec: float,
    max_visual_sec: float,
) -> list[dict[str, object]]:
    windows = _directed_transcript_windows(transcript_bundle)
    normalized: list[dict[str, object]] = []
    for index, raw in enumerate(raw_specs[:max_visuals], start=1):
        idea = _clean_directed_text(
            raw.get("visual_idea")
            or raw.get("hyperframes_visual_idea")
            or raw.get("visual_brief")
            or raw.get("prompt")
            or raw.get("description"),
            max_chars=360,
        )
        if not idea:
            continue
        trigger = _clean_directed_text(
            raw.get("trigger_text") or raw.get("trigger"),
            max_chars=96,
        ).lower()
        start_raw = raw.get("start_sec") if "start_sec" in raw else raw.get("start")
        end_raw = raw.get("end_sec") if "end_sec" in raw else raw.get("end")
        start_sec = _parse_directed_time_seconds(start_raw, None)
        end_sec = _parse_directed_time_seconds(end_raw, None)
        has_explicit_timing = start_sec is not None or end_sec is not None
        selected_window: dict[str, object] | None = None
        if start_sec is None:
            selected_window = _select_directed_window(
                windows,
                idea=idea,
                trigger=trigger,
            )
            if selected_window is None:
                continue
            start_sec = _as_float(selected_window.get("start"), 0.0)
            if end_sec is None:
                end_sec = _as_float(selected_window.get("end"), start_sec)
        start_sec = max(0.0, min(float(start_sec), max(clip_duration - 0.1, 0.0)))
        duration_raw = raw.get("duration_sec") if "duration_sec" in raw else raw.get("duration")
        requested_duration = _parse_directed_time_seconds(duration_raw, None)
        if end_sec is None:
            duration = requested_duration or max(
                min_visual_sec,
                min(4.4, max_visual_sec),
            )
            end_sec = start_sec + duration
        end_sec = min(
            clip_duration,
            max(start_sec + max(0.75, min_visual_sec), float(end_sec)),
        )
        if end_sec <= start_sec:
            continue
        selected = _windows_for_directed_range(
            windows,
            start=start_sec,
            end=end_sec,
            allow_nearest=not has_explicit_timing,
        )
        context = _directed_context_payload(windows, selected)
        sentence_text = str(context.get("sentence_text") or "").strip()
        context_text = str(context.get("context_text") or sentence_text).strip()
        if not sentence_text:
            continue
        source_text = f"{sentence_text} {context_text}".strip()
        raw_frame = _grounded_semantic_frame_from_raw(
            raw.get("semantic_frame"),
            source_text=source_text,
        )
        derived_frame = _derive_directed_semantic_frame(source_text, idea=idea)
        semantic_frame = {**derived_frame, **raw_frame}
        metric_facts = [
            dict(item)
            for item in _as_list(raw.get("metric_facts"))
            if isinstance(item, dict)
            and _copy_is_source_grounded(item.get("value") or item.get("label"), source_text)
        ] or _directed_metric_facts(source_text)
        required_labels = _directed_required_labels(
            semantic_frame,
            metric_facts,
            source_text=source_text,
        )
        preferred_medium = _preferred_directed_medium_family(idea)
        composition_mode = _normalize_manual_composition(
            raw.get("composition_mode") or raw.get("compose_mode"),
            str(raw.get("template") or ""),
        )
        if force_fullscreen:
            composition_mode = "replace"
        visual_id = str(raw.get("visual_id") or f"visual_{index:03d}")
        raw_director = (
            dict(raw.get("auto_visuals_director") or {})
            if isinstance(raw.get("auto_visuals_director"), dict)
            else {}
        )
        spec = dict(raw)
        spec.update(
            {
                "visual_id": visual_id,
                "card_id": str(raw.get("card_id") or f"directed_hyperframes_{index:03d}"),
                "start": round(start_sec, 3),
                "start_sec": round(start_sec, 3),
                "end": round(end_sec, 3),
                "duration": round(end_sec - start_sec, 3),
                "width": int(raw.get("width") or width),
                "height": int(raw.get("height") or height),
                "fps": float(raw.get("fps") or fps),
                "template": _directed_template_hint(preferred_medium, idea),
                "composition_mode": composition_mode,
                "renderer_hint": "hyperframes",
                "position": str(raw.get("position") or "center"),
                "scale": _as_float(raw.get("scale"), 1.0),
                "headline": _clean_directed_text(
                    raw.get("headline") or sentence_text,
                    max_chars=72,
                ),
                "emphasis_text": _clean_directed_text(
                    raw.get("emphasis_text")
                    or (required_labels[0] if required_labels else sentence_text),
                    max_chars=44,
                ),
                "sentence_text": sentence_text,
                "context_text": context_text,
                "planning_context_text": context_text,
                "keywords": _unique_directed_strings(
                    list(_directed_token_set(source_text))[:8],
                    limit=8,
                ),
                "supporting_lines": required_labels[1:4],
                "steps": list(semantic_frame.get("steps") or [])[:5]
                if isinstance(semantic_frame.get("steps"), list)
                else required_labels[:5],
                "semantic_frame": semantic_frame,
                "metric_facts": metric_facts,
                "required_labels": required_labels,
                "source_card_ids": list(context.get("source_card_ids") or []),
                "visual_type_hint": (
                    "product_ui"
                    if preferred_medium == "product_interface"
                    else str(raw.get("visual_type_hint") or "abstract_motion")
                ),
                "style_pack": str(raw.get("style_pack") or "auto"),
                "theme": dict(raw.get("theme") or {}),
                "confidence": _as_float(raw.get("confidence"), 0.94),
                "rationale": (
                    "User-directed HyperFrames art direction, grounded only in "
                    "the selected transcript window."
                ),
                "auto_visuals_director": {
                    **raw_director,
                    "director_score": _as_float(
                        raw_director.get("director_score"),
                        92.0,
                    ),
                    "visual_need": _as_float(
                        raw_director.get("visual_need"),
                        0.88,
                    ),
                    "source_richness": _as_float(
                        raw_director.get("source_richness"),
                        0.18,
                    ),
                    "copy_alignment": _as_float(
                        raw_director.get("copy_alignment"),
                        0.86,
                    ),
                    "warnings": list(raw_director.get("warnings") or []),
                    "selection_reason": "explicit_user_directed_hyperframes_visual",
                },
                "user_visual_idea": idea,
                "directed_visual_brief": {
                    "version": DIRECTED_HYPERFRAMES_VISUAL_VERSION,
                    "idea": idea,
                    "grounding_policy": "transcript_evidence_only",
                    "preferred_medium_family": preferred_medium,
                    "style_keywords": sorted(_directed_token_set(idea))[:12],
                    "unsupported_idea_terms": sorted(
                        _directed_token_set(idea) - _directed_token_set(source_text)
                    )[:16],
                    "selected_transcript_start": round(start_sec, 3),
                    "selected_transcript_end": round(end_sec, 3),
                    "source_card_ids": list(context.get("source_card_ids") or []),
                },
                "hyperframes_proof_candidate_count": clamp_int(
                    raw.get(
                        "hyperframes_proof_candidate_count",
                        config.HYPERFRAMES_PROOF_CANDIDATE_COUNT,
                    ),
                    default=int(config.HYPERFRAMES_PROOF_CANDIDATE_COUNT),
                    minimum=1,
                    maximum=8,
                ),
            }
        )
        normalized.append(spec)
    return normalized


def _normalize_manual_composition(value: object, template: str) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    if normalized in {"replace", "fullscreen", "full_screen"}:
        return "replace"
    if normalized in {"overlay", "pip", "picture_in_picture", "pictureinpicture"}:
        return "overlay"
    if template in {"floating_3d_label", "screen_pointer_3d", "product_model_spin"}:
        return "overlay"
    return "replace"


def _normalize_manual_blender_specs(
    raw_specs: list[dict[str, object]],
    *,
    clip_duration: float,
    width: int,
    height: int,
    fps: float,
    force_fullscreen: bool,
) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for index, raw in enumerate(raw_specs, start=1):
        template = str(raw.get("template") or "three_d_title").strip().lower().replace("-", "_")
        start_sec = max(
            0.0,
            min(
                _as_float(
                    raw.get("start_sec") if "start_sec" in raw else raw.get("start"),
                    0.0,
                ),
                max(clip_duration - 0.1, 0.0),
            ),
        )
        requested_end = raw.get("end_sec") if "end_sec" in raw else raw.get("end")
        requested_duration = raw.get("duration_sec") if "duration_sec" in raw else raw.get("duration")
        duration = max(0.75, min(_as_float(requested_duration, 4.0), 12.0))
        end_sec = _as_float(requested_end, start_sec + duration)
        end_sec = min(clip_duration, max(start_sec + 0.75, end_sec))
        if end_sec <= start_sec:
            continue
        composition_mode = _normalize_manual_composition(
            raw.get("composition_mode") or raw.get("compose_mode"),
            template,
        )
        if force_fullscreen:
            composition_mode = "replace"
        headline = str(
            raw.get("headline")
            or raw.get("text")
            or raw.get("label")
            or raw.get("emphasis_text")
            or "3D Visual"
        ).strip()
        label = str(raw.get("label") or raw.get("eyebrow") or headline).strip()
        position = str(raw.get("position") or ("center_right" if composition_mode == "overlay" else "center")).strip().lower()
        spec = dict(raw)
        spec.update(
            {
                "visual_id": str(raw.get("visual_id") or f"visual_{index:03d}"),
                "card_id": str(raw.get("card_id") or f"manual_3d_{index:03d}"),
                "start": round(start_sec, 3),
                "start_sec": round(start_sec, 3),
                "end": round(end_sec, 3),
                "duration": round(end_sec - start_sec, 3),
                "width": int(raw.get("width") or width),
                "height": int(raw.get("height") or height),
                "fps": float(raw.get("fps") or fps),
                "template": template,
                "composition_mode": composition_mode,
                "renderer_hint": "blender",
                "require_generated_scene": True,
                "position": position,
                "scale": 1.0 if composition_mode == "overlay" else _as_float(raw.get("scale"), 1.0),
                "headline": headline,
                "text": str(raw.get("text") or headline),
                "label": label,
                "subtext": str(raw.get("subtext") or raw.get("deck") or ""),
                "deck": str(raw.get("deck") or raw.get("subtext") or ""),
                "emphasis_text": str(raw.get("emphasis_text") or headline),
                "sentence_text": str(raw.get("sentence_text") or headline),
                "context_text": str(raw.get("context_text") or raw.get("subtext") or headline),
                "keywords": [
                    str(item)
                    for item in _as_list(raw.get("keywords"))
                    if str(item).strip()
                ],
                "supporting_lines": [
                    str(item)
                    for item in _as_list(raw.get("supporting_lines"))
                    if str(item).strip()
                ],
                "visual_type_hint": str(raw.get("visual_type_hint") or "abstract_motion"),
                "style_pack": str(raw.get("style_pack") or "cinematic_night"),
                "confidence": _as_float(raw.get("confidence"), 0.9),
                "rationale": str(raw.get("rationale") or "User-requested typed Blender 3D visual."),
                "alpha": _as_bool(raw.get("alpha"), composition_mode == "overlay"),
                "transparent_background": _as_bool(
                    raw.get("transparent_background"),
                    composition_mode == "overlay",
                ),
                "safe_area": _as_bool(raw.get("safe_area"), True),
            }
        )
        normalized.append(spec)
    return normalized


def _resolve_triggered_manual_specs(
    raw_specs: list[dict[str, object]],
    state: ProjectState,
    *,
    clip_duration: float,
) -> list[dict[str, object]]:
    needs_transcript = any(
        str(item.get("trigger_text") or item.get("trigger") or "").strip()
        and "start" not in item
        and "start_sec" not in item
        for item in raw_specs
    )
    if not needs_transcript:
        return raw_specs
    transcript_bundle = _ensure_transcript_bundle(state)
    windows = _as_list(transcript_bundle.get("sentences")) or _as_list(
        transcript_bundle.get("segments")
    )
    resolved: list[dict[str, object]] = []
    for item in raw_specs:
        spec = dict(item)
        trigger = str(spec.get("trigger_text") or spec.get("trigger") or "").strip().lower()
        if trigger and "start" not in spec and "start_sec" not in spec:
            for window in windows:
                if not isinstance(window, dict):
                    continue
                text = str(window.get("text") or "").lower()
                if trigger not in text:
                    continue
                start = max(0.0, _as_float(window.get("start"), 0.0) - 0.08)
                duration = max(
                    0.75,
                    min(
                        _as_float(
                            spec.get("duration_sec")
                            if "duration_sec" in spec
                            else spec.get("duration"),
                            4.0,
                        ),
                        12.0,
                    ),
                )
                spec["start"] = round(start, 3)
                spec["end"] = round(min(clip_duration, start + duration), 3)
                break
        resolved.append(spec)
    return resolved


def _execute_directed_hyperframes_specs(
    params: dict,
    state: ProjectState,
    *,
    mode: str,
    style_pack: str,
    force_fullscreen: bool,
    max_visuals: int,
    min_visual_sec: float,
    max_visual_sec: float,
) -> dict:
    raw_specs = _directed_hyperframes_specs_from_params(params)
    if not raw_specs:
        raise RuntimeError("No directed HyperFrames visual idea was provided.")
    require_imaging_runtime()
    metadata = state.metadata or probe_video(state.working_file)
    clip_duration = float(metadata.get("duration_sec") or 0.0)
    width = int(metadata.get("width") or 0)
    height = int(metadata.get("height") or 0)
    fps = float(metadata.get("fps") or 30.0) or 30.0
    if clip_duration <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(
            "The current working video does not have valid timing or resolution metadata."
        )
    provider_name, model_name = _provider_and_model(state)
    renderer_name = "hyperframes"
    capabilities = _filter_renderer_capabilities(
        renderer_capabilities(),
        renderer_name,
    )
    _require_available_renderer(capabilities, renderer_name)
    transcript_bundle = _ensure_transcript_bundle(state)
    bundle_root = ensure_writable_dir(
        writable_dir_candidates(
            state.working_dir,
            state.output_dir,
            state.project_id,
            "auto_visual_bundles",
        )
    )
    bundle_dir = create_unique_bundle_dir(
        bundle_root,
        f"{safe_stem(state.project_name)}_directed_hyperframes",
    )
    render_root = bundle_dir / "renders"
    render_root.mkdir(parents=True, exist_ok=True)
    planning_preview = {
        "version": DIRECTED_HYPERFRAMES_VISUAL_VERSION,
        "mode": mode,
        "renderer": renderer_name,
        "style_pack": style_pack,
        "requested_count": min(len(raw_specs), max_visuals),
        "output_bundle_path": str(bundle_dir),
        "transcript_source": str(transcript_bundle.get("source") or "missing"),
        "grounding_policy": "transcript_evidence_only",
    }
    write_run_status(
        bundle_dir,
        feature="auto_visuals",
        phase="directed_planning",
        payload=planning_preview,
    )
    _emit_progress("Grounding the directed HyperFrames visual idea in the transcript...")
    plan = _normalize_directed_hyperframes_specs(
        raw_specs,
        transcript_bundle=transcript_bundle,
        clip_duration=clip_duration,
        width=width,
        height=height,
        fps=fps,
        force_fullscreen=force_fullscreen,
        max_visuals=max_visuals,
        min_visual_sec=min_visual_sec,
        max_visual_sec=max_visual_sec,
    )
    plan = _ensure_unique_visual_ids([dict(item) for item in plan])
    if not plan:
        manifest = {
            "created_at": utc_now_iso(),
            "status": "failed_planning",
            "project_id": state.project_id,
            "project_name": state.project_name,
            "working_file": state.working_file,
            "renderer": renderer_name,
            "planning_preview": planning_preview,
            "directed_specs": raw_specs,
            "plan": [],
            "overlays": [],
            "render_failures": [
                "No transcript-grounded timing window could be resolved for the directed visual idea."
            ],
        }
        manifest_path = bundle_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return {
            "success": False,
            "message": (
                "Vex could not ground the directed HyperFrames idea in the transcript. "
                f"Manifest: {manifest_path}"
            ),
            "suggestion": None,
            "updated_state": state,
            "tool_name": "add_auto_visuals",
        }

    _emit_progress("Compiling open visual concepts and communication contracts...")
    open_plan, _, open_visual_report = _compile_open_visual_specs(
        plan,
        [],
        provider_name=provider_name,
        model_name=model_name,
        width=width,
        height=height,
        fps=fps,
    )
    _emit_progress("Compiling the directed HyperFrames visual contract...")
    compiled_plan, hyperframes_compiler_report = _compile_hyperframes_specs(open_plan)
    write_run_status(
        bundle_dir,
        feature="auto_visuals",
        phase="semantic_compile",
        status="running" if compiled_plan else "failed",
        payload={
            "selected_count": len(compiled_plan),
            "open_visual_program": open_visual_report,
            "compiled_count": hyperframes_compiler_report["compiled_count"],
            "proof_candidate_count": hyperframes_compiler_report[
                "proof_candidate_count"
            ],
            "estimated_render_count": hyperframes_compiler_report[
                "estimated_render_count"
            ],
            "rejected_count": hyperframes_compiler_report["rejected_count"],
            "rejected": hyperframes_compiler_report["rejected"],
        },
    )
    if not compiled_plan:
        compiler_reasons = [
            str(reason)
            for item in hyperframes_compiler_report["rejected"]
            for reason in (
                item.get("issues")
                or item.get("rejection_reasons")
                or ["semantic_compiler_rejected_candidate"]
            )
        ]
        manifest = {
            "created_at": utc_now_iso(),
            "status": "failed_semantic_compile",
            "project_id": state.project_id,
            "project_name": state.project_name,
            "working_file": state.working_file,
            "renderer": renderer_name,
            "planning_preview": planning_preview,
            "directed_specs": raw_specs,
            "open_visual_program": open_visual_report,
            "hyperframes_compiler": hyperframes_compiler_report,
            "plan": plan,
            "overlays": [],
            "render_failures": compiler_reasons,
        }
        manifest_path = bundle_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        detail = "; ".join(compiler_reasons[:6])
        return {
            "success": False,
            "message": (
                "The directed HyperFrames idea was grounded in the transcript, but "
                "the selected subtitle window did not pass the semantic compiler."
                f"{f' Details: {detail}' if detail else ''} Manifest: {manifest_path}"
            ),
            "suggestion": None,
            "updated_state": state,
            "tool_name": "add_auto_visuals",
        }

    prepared_specs = [
        _prepare_visual_spec(
            spec,
            style_pack=style_pack,
            provider_name=provider_name,
            model_name=model_name,
            state=state,
            bundle_dir=bundle_dir,
        )
        for spec in compiled_plan
    ]
    applied_overlays: list[dict] = []
    render_failures: list[str] = []
    rendered_visual_qa: list[dict[str, object]] = []
    _emit_progress(
        f"Rendering {len(prepared_specs)} directed HyperFrames visual"
        f"{'s' if len(prepared_specs) != 1 else ''}..."
    )
    for index, spec in enumerate(prepared_specs):
        try:
            asset, selection_reason = _render_generated_visual(
                spec,
                preferred_renderer=renderer_name,
                allowed_renderers={"hyperframes"},
                render_root=render_root,
                width=width,
                height=height,
                fps=fps,
                renderer_strategy="first_success",
            )
        except VisualRendererError as exc:
            render_failures.append(str(exc))
            _emit_progress(
                f"Render failed for {spec.get('visual_id', f'visual_{index + 1:03d}')}: {exc}"
            )
            continue
        spec, asset, selection_reason, visual_qa, direction_report = (
            _direct_rendered_visual_for_spec(
                spec,
                asset,
                selection_reason,
                render_root=render_root,
                width=width,
                height=height,
                fps=fps,
            )
        )
        visual_qa_payload = visual_qa.to_dict()
        visual_qa_payload["visual_director_v2"] = direction_report
        rendered_visual_qa.append(visual_qa_payload)
        if not visual_qa.passed:
            render_failures.append(
                (
                    f"{asset.renderer}: {spec.get('visual_id')} rejected by "
                    f"render QA ({', '.join(visual_qa.issues[:3])})"
                )
            )
            continue
        applied_overlays.append(
            _overlay_from_rendered_visual(
                spec,
                asset,
                selection_reason=selection_reason,
                force_fullscreen=force_fullscreen,
                visual_qa_payload=visual_qa_payload,
            )
        )

    preliminary_overlays = list(applied_overlays)
    applied_overlays, final_visual_qa = _final_auto_visuals_qa(
        preliminary_overlays,
        clip_duration=clip_duration,
        coverage_policy="quality_only",
    )
    write_run_status(
        bundle_dir,
        feature="auto_visuals",
        phase="qa",
        status="running" if applied_overlays else "failed",
        payload={
            "rendered_count": len(rendered_visual_qa),
            "selected_count": len(applied_overlays),
            "render_failure_count": len(render_failures),
            "final_qa": final_visual_qa,
        },
    )
    if not applied_overlays:
        manifest = {
            "created_at": utc_now_iso(),
            "status": "failed_qa",
            "project_id": state.project_id,
            "project_name": state.project_name,
            "working_file": state.working_file,
            "renderer": renderer_name,
            "style_pack": style_pack,
            "planning_preview": planning_preview,
            "directed_specs": raw_specs,
            "open_visual_program": open_visual_report,
            "hyperframes_compiler": hyperframes_compiler_report,
            "rendered_visual_qa": rendered_visual_qa,
            "final_visual_qa": final_visual_qa,
            "plan": compiled_plan,
            "overlays": [],
            "render_failures": render_failures,
        }
        manifest_path = bundle_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        detail = f" Details: {'; '.join(render_failures[:4])}" if render_failures else ""
        return {
            "success": False,
            "message": (
                "Vex rendered the directed HyperFrames visual, but none passed "
                f"render/timeline QA.{detail} Manifest: {manifest_path}"
            ),
            "suggestion": None,
            "updated_state": state,
            "tool_name": "add_auto_visuals",
        }

    _emit_progress("Compositing the directed HyperFrames visual into the working cut...")
    output_path = apply_visual_overlays(
        state.working_file,
        state.working_dir,
        applied_overlays,
    )
    output_metadata = probe_video(output_path)
    composite_qa = evaluate_visual_composite(
        state.working_file,
        output_path,
        applied_overlays,
        source_metadata=metadata,
        output_metadata=output_metadata,
    )
    if not composite_qa.passed:
        manifest = {
            "created_at": utc_now_iso(),
            "status": "failed_composite_qa",
            "project_id": state.project_id,
            "project_name": state.project_name,
            "working_file": state.working_file,
            "unpromoted_output": output_path,
            "renderer": renderer_name,
            "style_pack": style_pack,
            "planning_preview": planning_preview,
            "directed_specs": raw_specs,
            "open_visual_program": open_visual_report,
            "hyperframes_compiler": hyperframes_compiler_report,
            "rendered_visual_qa": rendered_visual_qa,
            "final_visual_qa": final_visual_qa,
            "composite_qa": composite_qa.to_dict(),
            "plan": compiled_plan,
            "overlays": applied_overlays,
            "render_failures": render_failures,
        }
        manifest_path = bundle_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        detail = ", ".join(composite_qa.issues[:4])
        return {
            "success": False,
            "message": (
                "Directed HyperFrames visual rendered, but the final composite "
                f"failed publish QA ({detail}). Project state was not changed. "
                f"Manifest: {manifest_path}"
            ),
            "suggestion": None,
            "updated_state": state,
            "tool_name": "add_auto_visuals",
        }

    state.working_file = output_path
    state.metadata = output_metadata
    renderer_counts: dict[str, int] = {}
    for overlay in applied_overlays:
        renderer_counts[str(overlay.get("renderer") or "unknown")] = (
            renderer_counts.get(str(overlay.get("renderer") or "unknown"), 0) + 1
        )
    renderer_summary = ", ".join(
        f"{name} x{count}" for name, count in sorted(renderer_counts.items())
    )
    manifest = {
        "created_at": utc_now_iso(),
        "status": "success",
        "project_id": state.project_id,
        "project_name": state.project_name,
        "source_video": state.source_files[0] if state.source_files else state.working_file,
        "working_file": state.working_file,
        "renderer": renderer_name,
        "style_pack": style_pack,
        "mode": mode,
        "planning_preview": planning_preview,
        "directed_specs": raw_specs,
        "renderer_capabilities": capabilities,
        "open_visual_program": open_visual_report,
        "hyperframes_compiler": hyperframes_compiler_report,
        "rendered_visual_qa": rendered_visual_qa,
        "final_visual_qa": final_visual_qa,
        "composite_qa": composite_qa.to_dict(),
        "plan": compiled_plan,
        "overlays": applied_overlays,
        "render_failures": render_failures,
    }
    manifest_path = bundle_dir / "manifest.json"
    registry_result = record_creative_run(
        working_dir=state.working_dir,
        feature="auto_visuals",
        manifest_path=str(manifest_path),
        output_path=state.working_file,
        graph_version=DIRECTED_HYPERFRAMES_VISUAL_VERSION,
        quality_score=_as_float(final_visual_qa.get("average_rendered_score"), 0.0),
        summary={
            "count": len(applied_overlays),
            "renderer": renderer_name,
            "style_pack": style_pack,
            "mode": "directed_hyperframes",
            "status": "success",
            "directed_visual": True,
        },
        artifacts={
            "bundle_dir": str(bundle_dir),
            "render_root": str(render_root),
        },
    )
    manifest["creative_registry"] = registry_result
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    state.artifacts["latest_auto_visuals"] = {
        "created_at": manifest["created_at"],
        "manifest_path": str(manifest_path),
        "bundle_dir": str(bundle_dir),
        "count": len(applied_overlays),
        "renderer": renderer_name,
        "style_pack": style_pack,
        "renderer_counts": renderer_counts,
        "directed_visual": True,
    }
    history = list(state.artifacts.get("auto_visuals_history") or [])
    history.append(state.artifacts["latest_auto_visuals"])
    state.artifacts["auto_visuals_history"] = history[-10:]
    state.apply_operation(
        {
            "op": "add_auto_visuals",
            "params": {
                "mode": mode,
                "renderer": renderer_name,
                "style_pack": style_pack,
                "max_visuals": max_visuals,
                "min_visual_sec": min_visual_sec,
                "max_visual_sec": max_visual_sec,
                "directed_visual_specs": raw_specs,
                "manifest_path": str(manifest_path),
                "overlays": applied_overlays,
            },
            "timestamp": utc_now_iso(),
            "result_file": output_path,
            "description": (
                f"Added {len(applied_overlays)} directed HyperFrames visual"
                f"{'s' if len(applied_overlays) != 1 else ''} ({renderer_summary})"
            ),
        }
    )
    write_run_status(
        bundle_dir,
        feature="auto_visuals",
        phase="complete",
        status="complete",
        payload={"manifest_path": str(manifest_path), "selected_count": len(applied_overlays)},
    )
    _emit_progress("Directed HyperFrames visual complete.")
    return {
        "success": True,
        "message": (
            f"Added {len(applied_overlays)} directed HyperFrames visual"
            f"{'s' if len(applied_overlays) != 1 else ''} using {renderer_summary}. "
            f"Manifest: {manifest_path}"
        ),
        "suggestion": None,
        "updated_state": state,
        "tool_name": "add_auto_visuals",
    }


def _execute_manual_visual_specs(
    params: dict,
    state: ProjectState,
    *,
    mode: str,
    renderer_name: str,
    style_pack: str,
    force_fullscreen: bool,
    max_visuals: int,
    min_visual_sec: float,
    max_visual_sec: float,
) -> dict:
    renderer_name = "blender"
    raw_specs = _manual_visual_specs_from_params(params)
    if not raw_specs:
        raise RuntimeError("No manual visual specs were provided.")
    metadata = state.metadata or probe_video(state.working_file)
    clip_duration = float(metadata.get("duration_sec") or 0.0)
    width = int(metadata.get("width") or 0)
    height = int(metadata.get("height") or 0)
    fps = float(metadata.get("fps") or 30.0) or 30.0
    if clip_duration <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(
            "The current working video does not have valid timing or resolution metadata."
        )
    provider_name, model_name = _provider_and_model(state)
    capabilities = renderer_capabilities()
    bundle_root = ensure_writable_dir(
        writable_dir_candidates(
            state.working_dir,
            state.output_dir,
            state.project_id,
            "auto_visual_bundles",
        )
    )
    bundle_dir = create_unique_bundle_dir(
        bundle_root,
        f"{safe_stem(state.project_name)}_manual_3d_visuals",
    )
    render_root = bundle_dir / "renders"
    render_root.mkdir(parents=True, exist_ok=True)

    raw_specs = _resolve_triggered_manual_specs(
        raw_specs,
        state,
        clip_duration=clip_duration,
    )
    plan = _normalize_manual_blender_specs(
        raw_specs[:max_visuals],
        clip_duration=clip_duration,
        width=width,
        height=height,
        fps=fps,
        force_fullscreen=force_fullscreen,
    )
    if not plan:
        raise RuntimeError("No valid manual visual specs remained after validation.")
    prepared_specs = [
        _prepare_visual_spec(
            spec,
            style_pack=style_pack,
            provider_name=provider_name,
            model_name=model_name,
            state=state,
        )
        for spec in plan
    ]

    applied_overlays: list[dict] = []
    render_failures: list[str] = []
    _emit_progress(
        f"Rendering {len(prepared_specs)} typed Blender visual{'s' if len(prepared_specs) != 1 else ''}..."
    )
    for index, spec in enumerate(prepared_specs):
        try:
            asset, selection_reason = _render_generated_visual(
                spec,
                preferred_renderer="blender",
                render_root=render_root,
                width=width,
                height=height,
                fps=fps,
            )
        except VisualRendererError as exc:
            render_failures.append(str(exc))
            _emit_progress(
                f"Render failed for {spec.get('visual_id', f'visual_{index + 1:03d}')}: {exc}"
            )
            continue
        if asset.renderer != "blender":
            render_failures.append(
                f"{asset.renderer}: typed Blender 3D specs must render with Blender."
            )
            continue
        has_alpha = bool((asset.metadata or {}).get("has_alpha"))
        requested_comp = str(spec.get("composition_mode") or "replace")
        compose_mode = "replace" if force_fullscreen else ("overlay" if has_alpha and requested_comp == "overlay" else requested_comp)
        applied_overlays.append(
            {
                "start": _as_float(spec.get("start"), 0.0),
                "end": _as_float(spec.get("end"), 0.0),
                "asset_path": asset.asset_path,
                "compose_mode": compose_mode,
                "has_alpha": has_alpha,
                "force_fullscreen": force_fullscreen,
                "position": "center" if compose_mode != "picture_in_picture" else str(spec.get("position") or "bottom_right"),
                "scale": 1.0 if compose_mode in {"replace", "overlay"} else _as_float(spec.get("scale"), 0.42),
                "visual_id": spec["visual_id"],
                "card_id": spec["card_id"],
                "template": spec["template"],
                "headline": spec["headline"],
                "emphasis_text": spec["emphasis_text"],
                "supporting_lines": spec.get("supporting_lines", []),
                "steps": spec.get("steps", []),
                "sentence_text": spec["sentence_text"],
                "context_text": spec["context_text"],
                "keywords": spec["keywords"],
                "visual_type_hint": spec["visual_type_hint"],
                "style_pack": spec.get("style_pack"),
                "theme": spec.get("theme", {}),
                "confidence": spec["confidence"],
                "rationale": spec["rationale"],
                "renderer": asset.renderer,
                "renderer_hint": spec.get("renderer_hint"),
                "renderer_selection_reason": selection_reason,
                "renderer_job_dir": asset.job_dir,
                "renderer_script_path": asset.script_path,
                "renderer_artifact_paths": dict(asset.artifact_paths or {}),
                "renderer_metadata": dict(asset.metadata or {}),
                "rendered_width": asset.width,
                "rendered_height": asset.height,
                "rendered_duration_sec": asset.duration_sec,
            }
        )

    if not applied_overlays:
        detail = f" Details: {'; '.join(render_failures[:4])}" if render_failures else ""
        return {
            "success": False,
            "message": f"Vex planned typed Blender visuals, but none could be rendered.{detail}",
            "suggestion": None,
            "updated_state": state,
            "tool_name": "add_auto_visuals",
        }

    _emit_progress("Compositing typed Blender visuals back into the working cut...")
    output_path = apply_visual_overlays(
        state.working_file, state.working_dir, applied_overlays
    )
    state.working_file = output_path
    state.metadata = probe_video(output_path)
    manifest = {
        "created_at": utc_now_iso(),
        "project_id": state.project_id,
        "project_name": state.project_name,
        "source_video": state.source_files[0] if state.source_files else state.working_file,
        "working_file": state.working_file,
        "renderer": renderer_name,
        "style_pack": style_pack,
        "mode": mode,
        "manual_visual_specs": True,
        "renderer_capabilities": capabilities,
        "plan": plan,
        "overlays": applied_overlays,
        "render_failures": render_failures,
    }
    manifest_path = bundle_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    renderer_counts: dict[str, int] = {}
    for overlay in applied_overlays:
        renderer_counts[str(overlay.get("renderer") or "unknown")] = (
            renderer_counts.get(str(overlay.get("renderer") or "unknown"), 0) + 1
        )
    renderer_summary = ", ".join(
        f"{name} x{count}" for name, count in sorted(renderer_counts.items())
    )
    state.artifacts["latest_auto_visuals"] = {
        "created_at": manifest["created_at"],
        "manifest_path": str(manifest_path),
        "bundle_dir": str(bundle_dir),
        "count": len(applied_overlays),
        "renderer": renderer_name,
        "style_pack": style_pack,
        "renderer_counts": renderer_counts,
    }
    history = list(state.artifacts.get("auto_visuals_history") or [])
    history.append(state.artifacts["latest_auto_visuals"])
    state.artifacts["auto_visuals_history"] = history[-10:]
    state.apply_operation(
        {
            "op": "add_auto_visuals",
            "params": {
                "mode": mode,
                "renderer": renderer_name,
                "style_pack": style_pack,
                "max_visuals": max_visuals,
                "min_visual_sec": min_visual_sec,
                "max_visual_sec": max_visual_sec,
                "manifest_path": str(manifest_path),
                "overlays": applied_overlays,
            },
            "timestamp": utc_now_iso(),
            "result_file": output_path,
            "description": f"Added {len(applied_overlays)} typed Blender 3D visuals ({renderer_summary})",
        }
    )
    _emit_progress("Typed Blender visuals complete.")
    return {
        "success": True,
        "message": (
            f"Added {len(applied_overlays)} typed Blender 3D visuals using {renderer_summary}. "
            f"Manifest: {manifest_path}"
        ),
        "suggestion": None,
        "updated_state": state,
        "tool_name": "add_auto_visuals",
    }


def execute(params: dict, state: ProjectState) -> dict:
    mode = str(params.get("mode") or "generated_only").strip().lower()
    if mode not in {"generated_only", "hybrid", "stock_only"}:
        mode = "generated_only"
    renderer_name = _normalize_renderer_name(params.get("renderer"))
    renderer_strategy = _normalize_renderer_strategy(
        params.get("renderer_strategy"),
        renderer_name,
    )
    renderer_tournament_size = clamp_int(
        params.get("renderer_tournament_size"),
        default=int(config.AUTO_VISUALS_RENDERER_TOURNAMENT_SIZE),
        minimum=1,
        maximum=3,
    )
    style_pack = str(params.get("style_pack") or "auto").strip().lower()
    refresh_existing = bool(params.get("refresh_existing", True))
    explicit_count = any(params.get(key) is not None for key in ("requested_count", "count"))
    coverage_policy = normalize_coverage_policy(
        params.get("coverage_policy"),
        explicit_count=explicit_count,
    )
    requested_count = (
        clamp_int(
            params.get("requested_count", params.get("count")),
            default=0,
            minimum=1,
            maximum=int(config.AUTO_VISUALS_MAX_VISUALS),
        )
        if explicit_count
        else None
    )
    requested_max_visuals = params.get("max_visuals")
    max_visuals = clamp_int(
        requested_max_visuals if requested_max_visuals is not None else requested_count or 8,
        default=requested_count or 8,
        minimum=1,
        maximum=int(config.AUTO_VISUALS_MAX_VISUALS),
    )
    density_param = params.get("density")
    min_visual_sec = max(1.8, min(float(params.get("min_visual_sec", 2.4) or 2.4), 6.0))
    max_visual_sec = max(
        min_visual_sec, min(float(params.get("max_visual_sec", 4.8) or 4.8), 10.0)
    )
    force_fullscreen = _should_force_fullscreen_visuals(
        params, mode=mode, renderer_name=renderer_name
    )
    directed_hyperframes_specs = _directed_hyperframes_specs_from_params(params)
    manual_specs = _manual_visual_specs_from_params(params)
    state_snapshot: dict[str, Any] | None = None

    if directed_hyperframes_specs:
        state_snapshot = state.capture_snapshot()
        try:
            result = _execute_directed_hyperframes_specs(
                params,
                state,
                mode=mode,
                style_pack=style_pack,
                force_fullscreen=force_fullscreen,
                max_visuals=max_visuals,
                min_visual_sec=min_visual_sec,
                max_visual_sec=max_visual_sec,
            )
            if not result.get("success"):
                state.restore_snapshot(state_snapshot)
            return result
        except (KeyboardInterrupt, SystemExit):
            state.restore_snapshot(state_snapshot)
            raise
        except Exception as exc:  # noqa: BLE001
            state.restore_snapshot(state_snapshot)
            return {
                "success": False,
                "message": str(exc),
                "suggestion": None,
                "updated_state": state,
                "tool_name": "add_auto_visuals",
            }

    if manual_specs:
        if not any(key in params for key in ("force_fullscreen", "fullscreen", "full_screen")):
            force_fullscreen = False
        state_snapshot = state.capture_snapshot()
        try:
            result = _execute_manual_visual_specs(
                params,
                state,
                mode=mode,
                renderer_name=renderer_name,
                style_pack=style_pack,
                force_fullscreen=force_fullscreen,
                max_visuals=max_visuals,
                min_visual_sec=min_visual_sec,
                max_visual_sec=max_visual_sec,
            )
            if not result.get("success"):
                state.restore_snapshot(state_snapshot)
            return result
        except (KeyboardInterrupt, SystemExit):
            state.restore_snapshot(state_snapshot)
            raise
        except Exception as exc:  # noqa: BLE001
            state.restore_snapshot(state_snapshot)
            return {
                "success": False,
                "message": str(exc),
                "suggestion": None,
                "updated_state": state,
                "tool_name": "add_auto_visuals",
            }

    if mode == "stock_only":
        return _delegate_stock_fallback(
            params, state, "Auto visuals was asked to use stock-only mode."
        )

    try:
        require_imaging_runtime()
        state_snapshot = state.capture_snapshot()
        refreshed_auto_overlay_counts: dict[str, int] = {}
        if refresh_existing:
            refreshed_auto_overlay_counts = _refresh_existing_auto_overlays(state)
            if refreshed_auto_overlay_counts:
                details = []
                if refreshed_auto_overlay_counts.get("add_auto_visuals"):
                    count = refreshed_auto_overlay_counts["add_auto_visuals"]
                    details.append(
                        f"{count} auto-visual pass{'es' if count != 1 else ''}"
                    )
                if refreshed_auto_overlay_counts.get("add_auto_broll"):
                    count = refreshed_auto_overlay_counts["add_auto_broll"]
                    details.append(
                        f"{count} auto B-roll pass{'es' if count != 1 else ''}"
                    )
                _emit_progress(
                    f"Cleared prior auto overlays before replanning: {', '.join(details)}."
                )
        _emit_progress("Loading transcript bundle...")
        transcript_bundle = _ensure_transcript_bundle(state)
        metadata = state.metadata or probe_video(state.working_file)
        clip_duration = float(metadata.get("duration_sec") or 0.0)
        width = int(metadata.get("width") or 0)
        height = int(metadata.get("height") or 0)
        fps = float(metadata.get("fps") or 30.0) or 30.0
        if clip_duration <= 0 or width <= 0 or height <= 0:
            raise RuntimeError(
                "The current working video does not have valid timing or resolution metadata."
            )
        density = normalize_density(density_param, clip_duration=clip_duration)
        bundle_root = ensure_writable_dir(
            writable_dir_candidates(
                state.working_dir,
                state.output_dir,
                state.project_id,
                "auto_visual_bundles",
            )
        )
        bundle_dir = create_unique_bundle_dir(
            bundle_root,
            f"{safe_stem(state.project_name)}_auto_visuals",
        )
        render_root = bundle_dir / "renders"
        render_root.mkdir(parents=True, exist_ok=True)

        transcript_segments = _as_list(transcript_bundle.get("segments"))
        transcript_words = _as_list(transcript_bundle.get("words"))
        sentence_segments = _as_list(transcript_bundle.get("sentences"))
        transcript_source = str(transcript_bundle.get("source") or "missing")
        usable_timed_segments = int(
            transcript_bundle.get("usable_timed_segments")
            or len(sentence_segments)
            or len(transcript_segments)
        )
        blocked_ranges = state.overlay_ranges()
        creative_policy = load_creative_policy(
            state.working_dir,
            feature="auto_visuals",
        )
        planning_preview = {
            "coverage_policy": coverage_policy,
            "requested_count": requested_count,
            "planned_count": max_visuals,
            "estimated_render_count": (
                max_visuals * renderer_tournament_size
                if renderer_strategy == "quality_tournament"
                else max_visuals
            ),
            "density": density,
            "renderer_mix": {
                "renderer_preference": renderer_name,
                "selection_strategy": renderer_strategy,
                "tournament_size": renderer_tournament_size,
            },
            "expected_slow_steps": [
                "scene_cut_detection",
                "candidate_scoring",
                "renderer_render",
                "final_composite",
            ],
            "output_bundle_path": str(bundle_dir),
            "transcript_source": transcript_source,
            "usable_timed_segments": usable_timed_segments,
            "creative_policy": {
                "version": creative_policy.version,
                "run_count": creative_policy.run_count,
                "outcome_count": creative_policy.outcome_count,
            },
        }
        write_run_status(
            bundle_dir,
            feature="auto_visuals",
            phase="transcript_load",
            payload=planning_preview,
        )
        _emit_progress("Detecting safe scene cuts...")
        scene_cuts = detect_scene_cuts(state.working_file)
        transcript_text = str(transcript_bundle.get("transcript_text") or "").strip()
        creative_graph = build_video_understanding_graph(
            transcript_text=transcript_text,
            segments=sentence_segments or transcript_segments,
            metadata=metadata,
            scene_cuts=scene_cuts,
            quality_tier="world_class_local",
            source_context={
                "feature": "auto_visuals",
                "mode": mode,
                "renderer": renderer_name,
                "style_pack": style_pack,
                "coverage_policy": coverage_policy,
                "density": density,
                "requested_count": requested_count,
                "transcript_source": transcript_source,
            },
        )
        _emit_progress("Building visual candidate cards from the transcript...")
        cards = build_visual_context_cards(
            sentence_segments,
            transcript_segments,
            clip_duration,
            words=transcript_words,
            scene_cuts=scene_cuts,
        )
        cards = annotate_visual_cards_with_graph(cards, creative_graph)
        _emit_progress("Sampling source frames for visual-need scoring...")
        cards = _annotate_cards_with_source_frames(
            cards,
            video_path=state.working_file,
        )
        cards = restrict_timed_items_to_available_ranges(
            cards,
            blocked_ranges,
            min_duration_sec=max(0.45, min_visual_sec * 0.5),
        )
        write_run_status(
            bundle_dir,
            feature="auto_visuals",
            phase="candidate_scoring",
            payload={
                "candidate_card_count": len(cards),
                "blocked_range_count": len(blocked_ranges),
            },
        )
        prior_card_ids = _prior_auto_visual_card_ids(state)
        prior_failure_card_ids = _prior_auto_visual_failure_card_ids(
            state,
            renderer_name=renderer_name,
        )
        cards = _filter_previously_used_cards(
            cards, prior_card_ids, max_visuals=max_visuals
        )
        if not cards:
            raise RuntimeError(
                "No transcript-aligned visual cards were available for planning after respecting existing full-screen overlay windows."
            )
        if requested_max_visuals is None:
            max_visuals = _contextual_visual_budget(
                cards,
                clip_duration=clip_duration,
                renderer_name=renderer_name,
                mode=mode,
                density=density,
            )
            planning_preview["planned_count"] = max_visuals
            planning_preview["estimated_render_count"] = (
                max_visuals * renderer_tournament_size
                if renderer_strategy == "quality_tournament"
                else max_visuals
            )
            _emit_progress(f"Using context-aware visual budget: {max_visuals}.")
        _emit_progress("Building semantic subtitle episodes and executable visual opportunities...")
        opportunity_plan = build_visual_opportunity_plan(
            cards,
            clip_duration=clip_duration,
            requested_count=max_visuals,
            blocked_card_ids=prior_failure_card_ids,
        )
        failure_memory_recovered = False
        if not opportunity_plan.selected and prior_failure_card_ids:
            rejection_counts = opportunity_plan.to_dict().get("rejection_counts") or {}
            if set(rejection_counts) <= {"blocked_by_prior_failure_or_usage"}:
                opportunity_plan = build_visual_opportunity_plan(
                    cards,
                    clip_duration=clip_duration,
                    requested_count=max_visuals,
                )
                failure_memory_recovered = bool(opportunity_plan.selected)
        opportunity_plan_payload = opportunity_plan.to_dict()
        opportunity_plan_payload["failure_memory"] = {
            "scope": "renderer_specific_exact_compiler_rejections",
            "renderer": renderer_name,
            "blocked_opportunity_count": len(prior_failure_card_ids),
            "recovered_without_history_block": failure_memory_recovered,
        }
        cards = opportunity_plan.selected_cards
        reserve_cards = opportunity_plan.reserve_cards
        if not cards:
            write_run_status(
                bundle_dir,
                feature="auto_visuals",
                phase="opportunity_planning",
                status="failed",
                payload={
                    "selected_count": 0,
                    "reserve_count": len(reserve_cards),
                    "visual_opportunity_plan": opportunity_plan_payload,
                },
            )
            state.restore_snapshot(state_snapshot)
            return {
                "success": False,
                "message": (
                    "The subtitle planner found no source-grounded visual opportunity "
                    "where a generated visual would add enough intuition."
                ),
                "suggestion": None,
                "updated_state": state,
                "tool_name": "add_auto_visuals",
            }
        max_visuals = len(cards)
        planning_preview["planned_count"] = max_visuals
        planning_preview["candidate_opportunity_count"] = len(cards)
        planning_preview["reserve_opportunity_count"] = len(reserve_cards)
        planning_preview["semantic_episode_count"] = len(opportunity_plan.episodes)
        write_run_status(
            bundle_dir,
            feature="auto_visuals",
            phase="opportunity_planning",
            payload={
                "selected_count": len(cards),
                "reserve_count": len(reserve_cards),
                "semantic_episode_count": len(opportunity_plan.episodes),
                "visual_opportunity_plan": opportunity_plan_payload,
            },
        )
        _emit_progress(
            f"Visual opportunity planner selected {len(cards)} executable beat"
            f"{'s' if len(cards) != 1 else ''} with {len(reserve_cards)} reserve"
            f"{'s' if len(reserve_cards) != 1 else ''}."
        )
        _emit_progress("Building the video-level visual narrative program...")
        program_cards = [*cards, *reserve_cards]
        visual_program = build_visual_narrative_program(
            program_cards,
            clip_duration=clip_duration,
            max_visuals=max_visuals,
            scene_cuts=scene_cuts,
            prefer_premium=force_fullscreen,
        )
        visual_program_payload = visual_program.to_dict()
        visual_program_payload["creative_graph_summary"] = creative_graph.compact()
        provider_name, model_name = _provider_and_model(state)
        prefer_premium = force_fullscreen
        capabilities = _filter_renderer_capabilities(
            renderer_capabilities(),
            renderer_name,
        )
        _require_available_renderer(capabilities, renderer_name)
        _emit_progress("Routing opportunities through the Auto Visuals skill graph...")
        cards, skill_graph_report = apply_visual_skill_graph(
            cards,
            available_renderers=capabilities,
            prefer_premium=prefer_premium,
            force_fullscreen=force_fullscreen,
        )
        reserve_cards, reserve_skill_graph_report = apply_visual_skill_graph(
            reserve_cards,
            available_renderers=capabilities,
            prefer_premium=prefer_premium,
            force_fullscreen=force_fullscreen,
        )
        skill_graph_report["reserve_skill_graph"] = reserve_skill_graph_report
        planning_preview["auto_visual_skill_graph"] = {
            "version": skill_graph_report["version"],
            "accepted_count": skill_graph_report["accepted_count"],
            "rejected_count": skill_graph_report["rejected_count"],
            "skill_counts": skill_graph_report["skill_counts"],
        }
        write_run_status(
            bundle_dir,
            feature="auto_visuals",
            phase="skill_graph",
            payload=skill_graph_report,
        )
        _emit_progress("Planning the generated visual beats...")
        write_run_status(
            bundle_dir,
            feature="auto_visuals",
            phase="planning",
            payload=planning_preview,
        )
        plan = analyze_visual_plan_with_llm(
            provider_name=provider_name,
            model_name=model_name,
            cards=cards,
            clip_duration=clip_duration,
            max_visuals=max_visuals,
            min_visual_sec=min_visual_sec,
            max_visual_sec=max_visual_sec,
            scene_cuts=scene_cuts,
            available_renderers=capabilities,
            avoid_card_ids=prior_card_ids | prior_failure_card_ids,
            disable_fast_plan=bool(prior_card_ids or prior_failure_card_ids)
            or prefer_premium,
            prefer_premium=prefer_premium,
            visual_program=visual_program_payload,
            skill_graph_report=skill_graph_report,
        )
        plan = restrict_timed_items_to_available_ranges(
            plan,
            blocked_ranges,
            min_duration_sec=min_visual_sec,
        )
        plan = _ensure_unique_visual_ids([dict(item) for item in plan])
        plan = _apply_creative_graph_to_visual_specs(plan, cards)
        hyperframes_available = any(
            str(item.get("name") or "").strip().lower() == "hyperframes"
            and bool(item.get("available"))
            for item in capabilities
        )
        remotion_available = any(
            str(item.get("name") or "").strip().lower() == "remotion"
            and bool(item.get("available"))
            for item in capabilities
        )
        premium_dom_renderer_hint = (
            "remotion"
            if renderer_name == "remotion"
            or (remotion_available and not hyperframes_available)
            else "hyperframes"
        )
        plan = apply_visual_program_to_specs(
            plan,
            visual_program_payload,
            style_pack=style_pack,
            enable_hyperframes_expansion=hyperframes_available or remotion_available,
            premium_renderer_hint=premium_dom_renderer_hint,
        )
        plan = enforce_visual_semantic_contracts(plan, max_visuals=max_visuals)
        if force_fullscreen:
            pip_count = sum(
                1
                for item in plan
                if str(item.get("composition_mode") or "").strip().lower() != "replace"
            )
            if pip_count:
                _emit_progress(
                    f"Promoted {pip_count} generated visual{'s' if pip_count != 1 else ''} to full-screen replacement composition."
                )
            plan = [_with_fullscreen_visual_spec(dict(item)) for item in plan]
            plan = enforce_visual_semantic_contracts(plan, max_visuals=max_visuals)
        plan, visual_director_report = _apply_auto_visuals_director_v3(
            plan,
            cards,
            renderer_name=renderer_name,
            capabilities=capabilities,
            force_fullscreen=force_fullscreen,
            max_visuals=max_visuals,
            coverage_policy=coverage_policy,
            creative_policy=creative_policy,
        )
        reserve_plan: list[dict[str, object]] = []
        reserve_director_report: dict[str, object] = {
            "version": AUTO_VISUALS_DIRECTOR_VERSION,
            "input_count": 0,
            "accepted_count": 0,
            "rejected_count": 0,
        }
        if reserve_cards:
            reserve_plan = fallback_visual_plan(
                reserve_cards,
                clip_duration,
                len(reserve_cards),
                min_visual_sec,
                max_visual_sec,
                scene_cuts,
                capabilities,
                prefer_premium=prefer_premium,
            )
            reserve_plan = restrict_timed_items_to_available_ranges(
                reserve_plan,
                blocked_ranges,
                min_duration_sec=min_visual_sec,
            )
            reserve_plan = _apply_creative_graph_to_visual_specs(
                reserve_plan,
                reserve_cards,
            )
            reserve_plan = apply_visual_program_to_specs(
                reserve_plan,
                visual_program_payload,
                style_pack=style_pack,
                enable_hyperframes_expansion=hyperframes_available,
            )
            reserve_plan = enforce_visual_semantic_contracts(
                reserve_plan,
                max_visuals=len(reserve_cards),
            )
            if force_fullscreen:
                reserve_plan = [
                    _with_fullscreen_visual_spec(dict(item))
                    for item in reserve_plan
                ]
                reserve_plan = enforce_visual_semantic_contracts(
                    reserve_plan,
                    max_visuals=len(reserve_cards),
                )
            reserve_plan, reserve_director_report = _apply_auto_visuals_director_v3(
                reserve_plan,
                reserve_cards,
                renderer_name=renderer_name,
                capabilities=capabilities,
                force_fullscreen=force_fullscreen,
                max_visuals=len(reserve_cards),
                coverage_policy=coverage_policy,
                creative_policy=creative_policy,
            )
            reserve_plan = [
                {**dict(item), "visual_id": f"reserve_{index:03d}"}
                for index, item in enumerate(reserve_plan, start=1)
            ]
        visual_director_report["reserve_director"] = reserve_director_report
        if not plan and not reserve_plan:
            write_run_status(
                bundle_dir,
                feature="auto_visuals",
                phase="director",
                status="failed",
                payload={
                    "selected_count": 0,
                    "rejected_count": visual_director_report.get("rejected_count", 0),
                    "rejection_reasons": [
                        str(
                            item.get("reason")
                            or ", ".join(str(reason) for reason in item.get("reasons", []))
                            or "director_rejected_candidate"
                        )
                        for item in visual_director_report.get("rejected", [])
                        if isinstance(item, dict)
                    ],
                },
            )
            state.restore_snapshot(state_snapshot)
            return {
                "success": False,
                "message": "No generated visuals passed the Auto Visuals Director relevance and renderer-fit checks.",
                "suggestion": None,
                "updated_state": state,
                "tool_name": "add_auto_visuals",
            }
        _emit_progress(
            "Authoring evidence-bound open visual programs and selecting distinct concepts..."
        )
        plan, reserve_plan, open_visual_report = _compile_open_visual_specs(
            plan,
            reserve_plan,
            provider_name=provider_name,
            model_name=model_name,
            width=width,
            height=height,
            fps=fps,
        )
        visual_director_report["open_visual_program"] = open_visual_report
        planning_preview["open_visual_program"] = {
            "enabled": bool(open_visual_report.get("enabled")),
            "compiled_count": int(
                open_visual_report.get("compiled_count") or 0
            ),
            "model_authored_count": int(
                open_visual_report.get("model_authored_count") or 0
            ),
            "rejected_count": int(
                open_visual_report.get("rejected_count") or 0
            ),
        }
        plan, reserve_plan, hyperframes_compiler_report = (
            _compile_hyperframes_specs_with_reserves(
                plan,
                reserve_plan,
                target_count=max_visuals,
                width=width,
                height=height,
            )
        )
        plan, reserve_plan, remotion_compiler_report = (
            _compile_remotion_specs_with_reserves(
                plan,
                reserve_plan,
                target_count=max_visuals,
                width=width,
                height=height,
                fps=fps,
            )
        )
        hyperframes_compiler_report["remotion_compiler"] = remotion_compiler_report
        planning_preview["estimated_render_count"] = int(
            hyperframes_compiler_report["estimated_render_count"]
        )
        planning_preview["hyperframes_proof_candidate_count"] = int(
            hyperframes_compiler_report["proof_candidate_count"]
        )
        if int(hyperframes_compiler_report["compiled_count"]) > 0:
            planning_preview["expected_slow_steps"] = [
                "HyperFrames structural candidate renders",
                "blind inverse decoding",
                "counterfactual relation ablation",
                "counterfactual temporal scramble",
                "final composite",
            ]
        elif any(
            str(item.get("renderer_hint") or "").strip().lower() == "remotion"
            for item in plan
        ):
            planning_preview["expected_slow_steps"] = [
                "Remotion bundle",
                "Remotion Chromium render",
                "final composite",
            ]
        else:
            planning_preview["expected_slow_steps"] = [
                "renderer render",
                "render QA",
                "final composite",
            ]
        write_run_status(
            bundle_dir,
            feature="auto_visuals",
            phase="semantic_compile",
            status="running" if plan else "failed",
            payload={
                "selected_count": len(plan),
                "compiled_count": hyperframes_compiler_report["compiled_count"],
                "proof_candidate_count": hyperframes_compiler_report[
                    "proof_candidate_count"
                ],
                "estimated_render_count": hyperframes_compiler_report[
                    "estimated_render_count"
                ],
                "rejected_count": hyperframes_compiler_report["rejected_count"],
                "rejected": hyperframes_compiler_report["rejected"],
                "remotion_compiled_count": remotion_compiler_report["compiled_count"],
                "remotion_rejected_count": remotion_compiler_report["rejected_count"],
                "remotion_rejected": remotion_compiler_report["rejected"],
            },
        )
        if not plan:
            compiler_reasons = [
                str(reason)
                for item in hyperframes_compiler_report["rejected"]
                for reason in (
                    item.get("issues")
                    or item.get("rejection_reasons")
                    or ["semantic_compiler_rejected_candidate"]
                )
            ]
            compiler_reasons.extend(
                str(reason)
                for item in remotion_compiler_report["rejected"]
                for reason in (item.get("issues") or ["remotion_semantic_compiler_rejected_candidate"])
            )
            detail = "; ".join(compiler_reasons[:6])
            state.restore_snapshot(state_snapshot)
            return {
                "success": False,
                "message": (
                    "No generated visuals had enough source-grounded structure to pass "
                    f"the renderer semantic compilers.{f' Details: {detail}' if detail else ''}"
                ),
                "suggestion": None,
                "updated_state": state,
                "tool_name": "add_auto_visuals",
            }
        visual_plan_quality = evaluate_visual_plan_quality(
            plan,
            creative_graph,
            max_visuals=max_visuals,
        ).to_dict()
        applied_overlays: list[dict] = []
        render_failures: list[str] = []
        rendered_visual_qa: list[dict[str, object]] = []
        prepared_specs = [
            _prepare_visual_spec(
                spec,
                style_pack=style_pack,
                provider_name=provider_name,
                model_name=model_name,
                state=state,
                bundle_dir=bundle_dir,
            )
            for spec in plan
        ]
        reserve_plan_by_visual_id = {
            str(spec.get("visual_id") or ""): dict(spec)
            for spec in reserve_plan
            if str(spec.get("visual_id") or "")
        }
        prepared_reserve_specs = [
            _prepare_visual_spec(
                spec,
                style_pack=style_pack,
                provider_name=provider_name,
                model_name=model_name,
                state=state,
                bundle_dir=bundle_dir,
            )
            for spec in reserve_plan
        ]
        render_successes: list[tuple[int, dict[str, object], RenderedAsset, str]] = []
        render_errors: list[tuple[int, dict[str, object], str]] = []
        worker_count = _max_render_workers(params, len(prepared_specs), prepared_specs)
        _emit_progress(
            f"Rendering {len(prepared_specs)} generated visual{'s' if len(prepared_specs) != 1 else ''} with {worker_count} worker{'s' if worker_count != 1 else ''}..."
        )
        write_run_status(
            bundle_dir,
            feature="auto_visuals",
            phase="render",
            payload={
                "planned_count": len(prepared_specs),
                "render_workers": worker_count,
            },
        )
        if worker_count == 1:
            for index, spec in enumerate(prepared_specs):
                try:
                    _emit_progress(
                        f"Rendering {spec.get('visual_id', f'visual_{index + 1:03d}')}..."
                    )
                    asset, selection_reason = _render_generated_visual(
                        spec,
                        preferred_renderer=renderer_name,
                        allowed_renderers=_allowed_renderers(renderer_name),
                        render_root=render_root,
                        width=width,
                        height=height,
                        fps=fps,
                        renderer_strategy=renderer_strategy,
                        tournament_size=renderer_tournament_size,
                    )
                    render_successes.append((index, spec, asset, selection_reason))
                except VisualRendererError as exc:
                    _emit_progress(
                        f"Render failed for {spec.get('visual_id', f'visual_{index + 1:03d}')}: {exc}"
                    )
                    render_errors.append((index, spec, str(exc)))
        else:
            with ThreadPoolExecutor(
                max_workers=worker_count, thread_name_prefix="vex-auto-visuals"
            ) as executor:
                future_map = {
                    executor.submit(
                        _render_generated_visual,
                        spec,
                        preferred_renderer=renderer_name,
                        allowed_renderers=_allowed_renderers(renderer_name),
                        render_root=render_root,
                        width=width,
                        height=height,
                        fps=fps,
                        renderer_strategy=renderer_strategy,
                        tournament_size=renderer_tournament_size,
                    ): (index, spec)
                    for index, spec in enumerate(prepared_specs)
                }
                for future in as_completed(future_map):
                    index, spec = future_map[future]
                    try:
                        asset, selection_reason = future.result()
                        _emit_progress(
                            f"Rendered {spec.get('visual_id', f'visual_{index + 1:03d}')} with {asset.renderer}."
                        )
                        render_successes.append((index, spec, asset, selection_reason))
                    except VisualRendererError as exc:
                        _emit_progress(
                            f"Render failed for {spec.get('visual_id', f'visual_{index + 1:03d}')}: {exc}"
                        )
                        render_errors.append((index, spec, str(exc)))

        failed_episode_ids = {
            str(spec.get("semantic_episode_id") or "")
            for _, spec, _ in render_errors
            if str(spec.get("semantic_episode_id") or "")
        }
        for _, _, failure in sorted(render_errors, key=lambda item: item[0]):
            render_failures.append(str(failure))

        for _, spec, asset, selection_reason in sorted(
            render_successes, key=lambda item: item[0]
        ):
            spec, asset, selection_reason, visual_qa, direction_report = (
                _direct_rendered_visual_for_spec(
                    spec,
                    asset,
                    selection_reason,
                    render_root=render_root,
                    width=width,
                    height=height,
                    fps=fps,
                )
            )
            visual_qa_payload = visual_qa.to_dict()
            visual_qa_payload["visual_director_v2"] = direction_report
            tournament_report = dict(
                (asset.metadata or {}).get("renderer_tournament") or {}
            )
            if tournament_report:
                visual_qa_payload["renderer_tournament"] = tournament_report
            rendered_visual_qa.append(visual_qa_payload)
            if not visual_qa.passed:
                episode_id = str(spec.get("semantic_episode_id") or "")
                if episode_id:
                    failed_episode_ids.add(episode_id)
                render_failures.append(
                    (
                        f"{asset.renderer}: {spec.get('visual_id')} rejected by "
                        f"render QA ({', '.join(visual_qa.issues[:3])})"
                    )
                )
                _emit_progress(
                    f"Dropped {spec.get('visual_id')} after render QA: {', '.join(visual_qa.issues[:3])}."
                )
                continue
            applied_overlays.append(
                _overlay_from_rendered_visual(
                    spec,
                    asset,
                    selection_reason=selection_reason,
                    force_fullscreen=force_fullscreen,
                    visual_qa_payload=visual_qa_payload,
                )
            )

        preliminary_input_overlays = list(applied_overlays)
        applied_overlays, preliminary_final_visual_qa = _final_auto_visuals_qa(
            preliminary_input_overlays,
            clip_duration=clip_duration,
            coverage_policy=coverage_policy,
        )
        overlays_by_visual_id = {
            str(item.get("visual_id") or ""): item
            for item in preliminary_input_overlays
            if isinstance(item, dict)
        }
        for rejected_item in _as_list(
            preliminary_final_visual_qa.get("rejected")
        ):
            if not isinstance(rejected_item, dict):
                continue
            rejected_overlay = overlays_by_visual_id.get(
                str(rejected_item.get("visual_id") or "")
            )
            episode_id = str(
                (rejected_overlay or {}).get("semantic_episode_id") or ""
            )
            if episode_id:
                failed_episode_ids.add(episode_id)

        render_recoveries: list[dict[str, object]] = []
        if len(applied_overlays) < max_visuals and prepared_reserve_specs:
            ordered_reserve_specs = sorted(
                prepared_reserve_specs,
                key=lambda item: (
                    0
                    if str(item.get("semantic_episode_id") or "")
                    in failed_episode_ids
                    else 1,
                    -_as_float(
                        (item.get("opportunity_contract") or {}).get("score"),
                        0.0,
                    ),
                    _as_float(item.get("start"), 0.0),
                ),
            )
            for reserve_spec in ordered_reserve_specs:
                if len(applied_overlays) >= max_visuals:
                    break
                if any(
                    _timed_specs_overlap(reserve_spec, overlay)
                    for overlay in applied_overlays
                ):
                    continue
                reserve_spec = dict(reserve_spec)
                reserve_spec["opportunity_recovery"] = {
                    "stage": "render",
                    "reason": "primary_render_or_qa_failure",
                    "episode_id": str(
                        reserve_spec.get("semantic_episode_id") or ""
                    ),
                }
                try:
                    _emit_progress(
                        f"Trying reserve opportunity {reserve_spec.get('visual_id')}..."
                    )
                    reserve_asset, reserve_reason = _render_generated_visual(
                        reserve_spec,
                        preferred_renderer=renderer_name,
                        allowed_renderers=_allowed_renderers(renderer_name),
                        render_root=render_root,
                        width=width,
                        height=height,
                        fps=fps,
                        renderer_strategy=renderer_strategy,
                        tournament_size=renderer_tournament_size,
                    )
                except VisualRendererError as exc:
                    render_failures.append(str(exc))
                    continue
                (
                    reserve_spec,
                    reserve_asset,
                    reserve_reason,
                    reserve_qa,
                    reserve_direction_report,
                ) = _direct_rendered_visual_for_spec(
                    reserve_spec,
                    reserve_asset,
                    reserve_reason,
                    render_root=render_root,
                    width=width,
                    height=height,
                    fps=fps,
                )
                reserve_qa_payload = reserve_qa.to_dict()
                reserve_qa_payload["visual_director_v2"] = (
                    reserve_direction_report
                )
                reserve_tournament = dict(
                    (reserve_asset.metadata or {}).get("renderer_tournament")
                    or {}
                )
                if reserve_tournament:
                    reserve_qa_payload["renderer_tournament"] = reserve_tournament
                rendered_visual_qa.append(reserve_qa_payload)
                if not reserve_qa.passed:
                    render_failures.append(
                        (
                            f"{reserve_asset.renderer}: {reserve_spec.get('visual_id')} "
                            f"reserve rejected by render QA "
                            f"({', '.join(reserve_qa.issues[:3])})"
                        )
                    )
                    continue
                applied_overlays.append(
                    _overlay_from_rendered_visual(
                        reserve_spec,
                        reserve_asset,
                        selection_reason=reserve_reason,
                        force_fullscreen=force_fullscreen,
                        visual_qa_payload=reserve_qa_payload,
                    )
                )
                published_reserve_spec = dict(
                    reserve_plan_by_visual_id.get(
                        str(reserve_spec.get("visual_id") or ""),
                        reserve_spec,
                    )
                )
                published_reserve_spec.update(
                    {
                        key: value
                        for key, value in reserve_spec.items()
                        if key
                        in {
                            "open_visual_program",
                            "open_visual_tournament",
                            "visual_repair_history",
                            "visual_communication_contract",
                            "visual_concept_search",
                        }
                    }
                )
                published_reserve_spec["opportunity_recovery"] = dict(
                    reserve_spec["opportunity_recovery"]
                )
                plan.append(published_reserve_spec)
                render_recoveries.append(
                    {
                        "visual_id": reserve_spec.get("visual_id"),
                        "card_id": reserve_spec.get("card_id"),
                        "episode_id": reserve_spec.get("semantic_episode_id"),
                        "stage": "render",
                    }
                )

        if render_recoveries:
            applied_overlays, final_visual_qa = _final_auto_visuals_qa(
                applied_overlays,
                clip_duration=clip_duration,
                coverage_policy=coverage_policy,
            )
            final_visual_qa["pre_recovery"] = preliminary_final_visual_qa
        else:
            final_visual_qa = preliminary_final_visual_qa
        plan.sort(key=lambda item: _as_float(item.get("start"), 0.0))
        write_run_status(
            bundle_dir,
            feature="auto_visuals",
            phase="qa",
            payload={
                "rendered_count": len(rendered_visual_qa),
                "selected_count": len(applied_overlays),
                "render_failure_count": len(render_failures),
                "render_recovery_count": len(render_recoveries),
                "final_qa": final_visual_qa,
            },
        )
        if not applied_overlays:
            outcome_signals = _creative_outcome_signals(
                plan,
                rendered_visual_qa,
                applied_overlays,
            )
            failed_manifest = {
                "created_at": utc_now_iso(),
                "status": "failed_qa",
                "project_id": state.project_id,
                "project_name": state.project_name,
                "working_file": state.working_file,
                "renderer": renderer_name,
                "renderer_strategy": renderer_strategy,
                "renderer_tournament_size": renderer_tournament_size,
                "creative_policy": creative_policy.to_dict(),
                "planning_preview": planning_preview,
                "visual_opportunity_plan": opportunity_plan_payload,
                "auto_visual_skill_graph": skill_graph_report,
                "auto_visuals_director": visual_director_report,
                "hyperframes_compiler": hyperframes_compiler_report,
                "rendered_visual_qa": rendered_visual_qa,
                "final_visual_qa": final_visual_qa,
                "plan": plan,
                "overlays": [],
                "render_failures": render_failures,
                "render_recoveries": render_recoveries,
                "outcome_signals": outcome_signals,
            }
            failed_manifest_path = bundle_dir / "manifest.json"
            failed_registry_result = record_creative_run(
                working_dir=state.working_dir,
                feature="auto_visuals",
                manifest_path=str(failed_manifest_path),
                output_path=state.working_file,
                graph_version=creative_graph.version,
                quality_score=0.0,
                summary={
                    "count": 0,
                    "renderer": renderer_name,
                    "renderer_strategy": renderer_strategy,
                    "style_pack": style_pack,
                    "mode": mode,
                    "status": "failed_qa",
                    "outcome_signals": outcome_signals,
                },
                artifacts={
                    "bundle_dir": str(bundle_dir),
                    "render_root": str(render_root),
                },
            )
            failed_manifest["creative_registry"] = failed_registry_result
            failed_manifest_path.write_text(
                json.dumps(failed_manifest, indent=2),
                encoding="utf-8",
            )
            write_run_status(
                bundle_dir,
                feature="auto_visuals",
                phase="qa",
                status="failed",
                payload={
                    "selected_count": 0,
                    "manifest_path": str(failed_manifest_path),
                    "render_failure_count": len(render_failures),
                },
            )
            if mode == "hybrid" and configured_stock_provider_names():
                result = _delegate_stock_fallback(
                    params,
                    state,
                    "Generated visuals could not pass renderer and final timeline QA.",
                )
                if not result.get("success"):
                    state.restore_snapshot(state_snapshot)
                return result
            detail = (
                f" Details: {'; '.join(render_failures[:4])}" if render_failures else ""
            )
            state.restore_snapshot(state_snapshot)
            return {
                "success": False,
                "message": (
                    "Vex planned generated visuals, but none passed render/timeline QA."
                    f"{detail} Manifest: {failed_manifest_path}"
                ),
                "suggestion": None,
                "updated_state": state,
                "tool_name": "add_auto_visuals",
            }

        _emit_progress("Compositing the generated visuals back into the working cut...")
        write_run_status(
            bundle_dir,
            feature="auto_visuals",
            phase="final_composite",
            payload={"selected_count": len(applied_overlays)},
        )
        output_path = apply_visual_overlays(
            state.working_file, state.working_dir, applied_overlays
        )
        output_metadata = probe_video(output_path)
        composite_qa = evaluate_visual_composite(
            state.working_file,
            output_path,
            applied_overlays,
            source_metadata=metadata,
            output_metadata=output_metadata,
        )
        write_run_status(
            bundle_dir,
            feature="auto_visuals",
            phase="composite_qa",
            status="running" if composite_qa.passed else "failed",
            payload={"composite_qa": composite_qa.to_dict()},
        )
        if not composite_qa.passed:
            outcome_signals = _creative_outcome_signals(
                plan,
                rendered_visual_qa,
                [],
            )
            failed_manifest = {
                "created_at": utc_now_iso(),
                "status": "failed_composite_qa",
                "project_id": state.project_id,
                "project_name": state.project_name,
                "working_file": state.working_file,
                "unpromoted_output": output_path,
                "renderer": renderer_name,
                "renderer_strategy": renderer_strategy,
                "creative_policy": creative_policy.to_dict(),
                "planning_preview": planning_preview,
                "visual_opportunity_plan": opportunity_plan_payload,
                "auto_visual_skill_graph": skill_graph_report,
                "auto_visuals_director": visual_director_report,
                "hyperframes_compiler": hyperframes_compiler_report,
                "rendered_visual_qa": rendered_visual_qa,
                "final_visual_qa": final_visual_qa,
                "composite_qa": composite_qa.to_dict(),
                "plan": plan,
                "overlays": applied_overlays,
                "render_failures": render_failures,
                "render_recoveries": render_recoveries,
                "outcome_signals": outcome_signals,
            }
            failed_manifest_path = bundle_dir / "manifest.json"
            failed_registry_result = record_creative_run(
                working_dir=state.working_dir,
                feature="auto_visuals",
                manifest_path=str(failed_manifest_path),
                output_path=state.working_file,
                graph_version=creative_graph.version,
                quality_score=0.0,
                summary={
                    "count": 0,
                    "renderer": renderer_name,
                    "renderer_strategy": renderer_strategy,
                    "style_pack": style_pack,
                    "mode": mode,
                    "status": "failed_composite_qa",
                    "outcome_signals": outcome_signals,
                },
                artifacts={
                    "bundle_dir": str(bundle_dir),
                    "render_root": str(render_root),
                    "unpromoted_output": output_path,
                },
            )
            failed_manifest["creative_registry"] = failed_registry_result
            failed_manifest_path.write_text(
                json.dumps(failed_manifest, indent=2),
                encoding="utf-8",
            )
            detail = ", ".join(composite_qa.issues[:4])
            state.restore_snapshot(state_snapshot)
            return {
                "success": False,
                "message": (
                    "Generated visuals rendered, but the final composite failed "
                    f"publish QA ({detail}). Project state was not changed. "
                    f"Manifest: {failed_manifest_path}"
                ),
                "suggestion": None,
                "updated_state": state,
                "tool_name": "add_auto_visuals",
            }
        state.working_file = output_path
        state.metadata = output_metadata

        renderer_counts: dict[str, int] = {}
        for overlay in applied_overlays:
            renderer_counts[str(overlay.get("renderer") or "unknown")] = (
                renderer_counts.get(
                    str(overlay.get("renderer") or "unknown"),
                    0,
                )
                + 1
            )
        renderer_summary = ", ".join(
            f"{name} x{count}" for name, count in sorted(renderer_counts.items())
        )
        director_rejection_reasons = [
            str(
                item.get("reason")
                or ", ".join(str(reason) for reason in item.get("reasons", []))
                or "director_rejected_candidate"
            )
            for item in visual_director_report.get("rejected", [])
            if isinstance(item, dict)
        ]
        final_rejection_reasons = [
            str(item.get("reason") or "final_timeline_qa_rejected_candidate")
            for item in final_visual_qa.get("rejected", [])
            if isinstance(item, dict)
        ]
        compiler_rejection_reasons = [
            str(reason)
            for item in hyperframes_compiler_report.get("rejected", [])
            if isinstance(item, dict)
            for reason in (
                item.get("issues")
                or item.get("rejection_reasons")
                or ["semantic_compiler_rejected_candidate"]
            )
        ]
        rejection_reasons = [
            *director_rejection_reasons,
            *compiler_rejection_reasons,
            *render_failures,
            *final_rejection_reasons,
        ]
        counts_payload = coverage_counts(
            requested_count=requested_count,
            selected_count=len(applied_overlays),
            rejected_count=len(rejection_reasons),
            rejection_reasons=rejection_reasons[:32],
        )

        manifest = {
            "created_at": utc_now_iso(),
            "project_id": state.project_id,
            "project_name": state.project_name,
            "source_video": state.source_files[0]
            if state.source_files
            else state.working_file,
            "working_file": state.working_file,
            "renderer": renderer_name,
            "renderer_strategy": renderer_strategy,
            "renderer_tournament_size": renderer_tournament_size,
            "creative_policy": creative_policy.to_dict(),
            "style_pack": style_pack,
            "mode": mode,
            "coverage_policy": coverage_policy,
            "density": density,
            "requested_count": requested_count,
            "selected_count": counts_payload["selected_count"],
            "rejected_count": counts_payload["rejected_count"],
            "rejection_reasons": counts_payload["rejection_reasons"],
            "planning_preview": planning_preview,
            "visual_opportunity_plan": opportunity_plan_payload,
            "auto_visual_skill_graph": skill_graph_report,
            "renderer_capabilities": capabilities,
            "render_workers": worker_count,
            "transcript_source": transcript_source,
            "usable_timed_segments": usable_timed_segments,
            "transcript_paths": transcript_bundle.get("paths", {}),
            "scene_cuts": scene_cuts,
            "blocked_ranges": blocked_ranges,
            "creative_graph": creative_graph.to_dict(),
            "creative_graph_summary": creative_graph.compact(),
            "visual_program": visual_program_payload,
            "auto_visuals_director": visual_director_report,
            "hyperframes_compiler": hyperframes_compiler_report,
            "visual_plan_quality": visual_plan_quality,
            "rendered_visual_qa": rendered_visual_qa,
            "final_visual_qa": final_visual_qa,
            "composite_qa": composite_qa.to_dict(),
            "plan": plan,
            "overlays": applied_overlays,
            "render_failures": render_failures,
            "render_recoveries": render_recoveries,
        }
        manifest_path = bundle_dir / "manifest.json"
        outcome_signals = _creative_outcome_signals(
            plan,
            rendered_visual_qa,
            applied_overlays,
        )
        registry_result = record_creative_run(
            working_dir=state.working_dir,
            feature="auto_visuals",
            manifest_path=str(manifest_path),
            output_path=state.working_file,
            graph_version=creative_graph.version,
            quality_score=float(visual_plan_quality.get("score") or 0.0),
            summary={
                "count": len(applied_overlays),
                "renderer": renderer_name,
                "renderer_strategy": renderer_strategy,
                "style_pack": style_pack,
                "mode": mode,
                "director_average_score": visual_director_report.get("average_director_score"),
                "final_qa_average_score": final_visual_qa.get("average_rendered_score"),
                "composite_qa_score": composite_qa.score,
                "outcome_signals": outcome_signals,
            },
            artifacts={
                "bundle_dir": str(bundle_dir),
                "render_root": str(render_root),
                "renderer_counts": renderer_counts,
            },
        )
        manifest["creative_registry"] = registry_result
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        write_run_status(
            bundle_dir,
            feature="auto_visuals",
            phase="complete",
            status="complete",
            payload={
                "manifest_path": str(manifest_path),
                **counts_payload,
            },
        )

        notes_lines = [
            "# Auto Visuals Notes",
            "",
            f"Renderer preference: {renderer_name}",
            f"Style pack: {style_pack}",
            f"Mode: {mode}",
            (
                "Visual opportunity planner: "
                f"{opportunity_plan_payload['selected_count']} selected, "
                f"{opportunity_plan_payload['reserve_count']} reserves across "
                f"{opportunity_plan_payload['episode_count']} semantic episodes"
            ),
            f"Director: {visual_director_report['accepted_count']} accepted / {visual_director_report['input_count']} planned",
            (
                "Semantic compiler: "
                f"{hyperframes_compiler_report['compiled_count']} compiled, "
                f"{hyperframes_compiler_report['rejected_count']} rejected"
            ),
            f"Plan quality: {visual_plan_quality['score']:.3f} ({'passed' if visual_plan_quality['passed'] else 'review'})",
            f"Rendered QA: {final_visual_qa['average_rendered_score']:.3f} average",
            f"Composite QA: {composite_qa.score:.3f} ({'passed' if composite_qa.passed else 'failed'})",
            "",
        ]
        for overlay in applied_overlays:
            renderer_metadata = dict(overlay.get("renderer_metadata") or {})
            variant_selection = dict(renderer_metadata.get("variant_selection") or {})
            art_direction = dict(renderer_metadata.get("art_direction") or {})
            rendered_qa = dict(overlay.get("rendered_visual_qa") or {})
            director = dict(overlay.get("auto_visuals_director") or {})
            notes_lines.extend(
                [
                    f"## {overlay['start']:.2f}s-{overlay['end']:.2f}s",
                    f"Template: {overlay['template']}",
                    f"Headline: {overlay['headline']}",
                    f"Renderer: {overlay['renderer']}",
                    f"Composition: {overlay['compose_mode']}",
                    f"Director score: {director.get('director_score', 'unknown')}",
                    f"Rendered QA: {rendered_qa.get('score', 'unknown')}",
                    f"Why: {overlay['rationale']}",
                ]
            )
            if variant_selection:
                quality_score = variant_selection.get("selected_quality_score")
                quality_label = (
                    "passed"
                    if variant_selection.get("selected_quality_passed")
                    else "review"
                )
                if isinstance(quality_score, (int, float)):
                    score_text = f"{quality_score:.3f}"
                else:
                    score_text = "unknown"
                notes_lines.append(
                    "Selected variant: "
                    f"{variant_selection.get('selected_variant_id')} "
                    f"(quality {score_text}, {quality_label})"
                )
            if art_direction.get("name"):
                notes_lines.append(f"Art direction: {art_direction['name']}")
            notes_lines.append("")
        (bundle_dir / "notes.md").write_text("\n".join(notes_lines), encoding="utf-8")

        state.artifacts["latest_auto_visuals"] = {
            "created_at": manifest["created_at"],
            "manifest_path": str(manifest_path),
            "bundle_dir": str(bundle_dir),
            "count": len(applied_overlays),
            "renderer": renderer_name,
            "style_pack": style_pack,
            "coverage_policy": coverage_policy,
            "requested_count": requested_count,
            "selected_count": len(applied_overlays),
            "renderer_counts": renderer_counts,
            "creative_graph_version": creative_graph.version,
            "auto_visual_skill_graph": {
                "version": skill_graph_report.get("version"),
                "accepted_count": skill_graph_report.get("accepted_count"),
                "rejected_count": skill_graph_report.get("rejected_count"),
                "skill_counts": skill_graph_report.get("skill_counts", {}),
            },
            "visual_plan_quality_score": visual_plan_quality["score"],
            "auto_visuals_director_score": visual_director_report.get("average_director_score"),
            "hyperframes_compiled_count": hyperframes_compiler_report.get(
                "compiled_count", 0
            ),
            "hyperframes_rejected_count": hyperframes_compiler_report.get(
                "rejected_count", 0
            ),
            "visual_opportunity_selected_count": opportunity_plan_payload.get(
                "selected_count",
                0,
            ),
            "visual_opportunity_reserve_count": opportunity_plan_payload.get(
                "reserve_count",
                0,
            ),
            "render_recovery_count": len(render_recoveries),
            "final_visual_qa_score": final_visual_qa.get("average_rendered_score"),
            "composite_qa_score": composite_qa.score,
            "creative_registry": registry_result,
        }
        history = list(state.artifacts.get("auto_visuals_history") or [])
        history.append(state.artifacts["latest_auto_visuals"])
        state.artifacts["auto_visuals_history"] = history[-10:]
        state.apply_operation(
            {
                "op": "add_auto_visuals",
                "params": {
                    "mode": mode,
                    "renderer": renderer_name,
                    "style_pack": style_pack,
                    "max_visuals": max_visuals,
                    "coverage_policy": coverage_policy,
                    "requested_count": requested_count,
                    "density": density,
                    "min_visual_sec": min_visual_sec,
                    "max_visual_sec": max_visual_sec,
                    "manifest_path": str(manifest_path),
                    "overlays": applied_overlays,
                },
                "timestamp": utc_now_iso(),
                "result_file": output_path,
                "description": f"Added {len(applied_overlays)} transcript-aligned generated visuals ({renderer_summary})",
            }
        )
        _emit_progress("Auto visuals complete.")
        return {
            "success": True,
            "message": (
                f"Added {len(applied_overlays)} transcript-aligned generated visuals using {renderer_summary} "
                f"(preference: {renderer_name}). Coverage: {coverage_policy}"
                f"{f' requested {requested_count}' if requested_count else ''}; "
                f"selected {len(applied_overlays)}, rejected {counts_payload['rejected_count']}. "
                f"Transcript source: {transcript_source} ({usable_timed_segments} timed segments). "
                f"Manifest: {manifest_path}"
            ),
            "suggestion": None,
            "updated_state": state,
            "tool_name": "add_auto_visuals",
        }
    except (KeyboardInterrupt, SystemExit):
        if state_snapshot is not None:
            state.restore_snapshot(state_snapshot)
        raise
    except Exception as exc:  # noqa: BLE001
        if state_snapshot is not None:
            state.restore_snapshot(state_snapshot)
        return {
            "success": False,
            "message": str(exc),
            "suggestion": None,
            "updated_state": state,
            "tool_name": "add_auto_visuals",
        }
