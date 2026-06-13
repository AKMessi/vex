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
from broll_intelligence import configured_stock_provider_names, ensure_writable_dir, safe_stem, writable_dir_candidates
from tools.creative_intelligence import annotate_visual_cards_with_graph, build_video_understanding_graph
from tools.creative_optimizer import optimize_creative_set
from tools.creative_qa import evaluate_visual_plan_quality
from tools.composite_qa import evaluate_visual_composite
from tools.creative_registry import (
    CreativePolicySnapshot,
    load_creative_policy,
    record_creative_run,
)
from engine import VideoEngineError, apply_visual_overlays, probe_video
from renderers import (
    RenderedAsset,
    RendererMatch,
    VisualRendererError,
    list_renderers,
    rank_renderers,
    renderer_capabilities,
    resolve_renderer,
)
from state import ProjectState, restrict_timed_items_to_available_ranges, utc_now_iso
from tools.automation import (
    clamp_int,
    coverage_counts,
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
)
from visual_program import apply_visual_program_to_specs, build_visual_narrative_program
from vex_hyperframes.compiler import compile_hyperframes_plan


AUTO_VISUALS_DIRECTOR_VERSION = "auto-visuals-director-v3"
RENDERER_TOURNAMENT_VERSION = "renderer-quality-tournament-v1"
SOURCE_FRAME_SAMPLE_WIDTH = 48
SOURCE_FRAME_SAMPLE_HEIGHT = 27


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
    if renderer in {"auto", "both", "hyperframes", "manim", "ffmpeg", "blender"}:
        return renderer
    return "auto"


def _allowed_renderers(renderer_name: str) -> set[str] | None:
    if renderer_name == "hyperframes":
        return {"hyperframes"}
    if renderer_name == "manim":
        return {"manim"}
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


def _should_force_fullscreen_visuals(
    params: dict, *, mode: str, renderer_name: str
) -> bool:
    if "force_fullscreen" in params:
        return _as_bool(params.get("force_fullscreen"), True)
    if "fullscreen" in params:
        return _as_bool(params.get("fullscreen"), True)
    if "full_screen" in params:
        return _as_bool(params.get("full_screen"), True)
    return mode == "generated_only" or renderer_name in {"auto", "both", "hyperframes", "manim"}


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
    return card_ids


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
) -> tuple[list[dict[str, object]], dict[str, object]]:
    accepted: list[dict[str, object]] = []
    compiled: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    for spec in plan:
        renderer_hint = str(spec.get("renderer_hint") or "").strip().lower()
        if renderer_hint != "hyperframes":
            accepted.append(dict(spec))
            continue
        candidate = _apply_hyperframes_continuity(dict(spec))
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
        compiled_spec["hyperframes_automatic_semantic_route"] = True
        compiled_spec["hyperframes_legacy_template_policy"] = "manual_only"
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
        return "ffmpeg" if "ffmpeg" in available_renderers else "hyperframes"
    if "hyperframes" in available_renderers:
        return "hyperframes"
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
        if strict_allowed is None or "hyperframes" in strict_allowed or "ffmpeg" in strict_allowed:
            normalized["renderer_hint"] = "hyperframes" if "hyperframes" in available else preferred_renderer
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
    elif renderer_name == "both" and str(normalized.get("renderer_hint") or "") not in {"hyperframes", "manim"}:
        normalized["renderer_hint"] = "manim" if intent_type == "math_or_formula" and "manim" in available else "hyperframes"
        renderer_policy = str(normalized["renderer_hint"])
    if force_fullscreen:
        normalized["composition_mode"] = "replace"
        normalized["position"] = "center"
        normalized["scale"] = 1.0
    if renderer_policy == "hyperframes" and str(normalized.get("template") or "") in {"quote_focus", "keyword_stack", "ribbon_quote"} and intent_type not in {"emphasis", "math_or_formula"}:
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
                asset = renderer.render(
                    spec, render_root=render_root, width=width, height=height, fps=fps
                )
                return asset, reason
            except VisualRendererError as exc:
                failures.append(f"{renderer.name}: {exc}")
                if len(attempted | base_excluded) >= len(known_renderers):
                    break
        if len(attempted | base_excluded) >= len(known_renderers):
            break
    raise VisualRendererError(
        "; ".join(failures) or "No renderer could produce the generated visual."
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
            asset = match.renderer.render(
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
    if renderer_name in {"auto", "both", "hyperframes"} or any(
        str(spec.get("renderer_hint") or "").strip().lower() == "hyperframes"
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
    if renderer_name in {"hyperframes", "both", "auto"} and mode == "generated_only":
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
    timestamp_label = utc_now_iso().replace(":", "-").replace("+00:00", "Z")
    bundle_dir = (
        bundle_root
        / f"{safe_stem(state.project_name)}_manual_3d_visuals_{timestamp_label}"
    )
    bundle_dir.mkdir(parents=True, exist_ok=True)
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
    manual_specs = _manual_visual_specs_from_params(params)

    if manual_specs:
        if not any(key in params for key in ("force_fullscreen", "fullscreen", "full_screen")):
            force_fullscreen = False
        try:
            return _execute_manual_visual_specs(
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
        except (RuntimeError, VideoEngineError, VisualRendererError) as exc:
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
        timestamp_label = utc_now_iso().replace(":", "-").replace("+00:00", "Z")
        bundle_dir = (
            bundle_root
            / f"{safe_stem(state.project_name)}_auto_visuals_{timestamp_label}"
        )
        bundle_dir.mkdir(parents=True, exist_ok=True)
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
        _emit_progress("Building the video-level visual narrative program...")
        visual_program = build_visual_narrative_program(
            cards,
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
            avoid_card_ids=prior_card_ids,
            disable_fast_plan=bool(prior_card_ids) or prefer_premium,
            prefer_premium=prefer_premium,
            visual_program=visual_program_payload,
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
        plan = apply_visual_program_to_specs(
            plan,
            visual_program_payload,
            style_pack=style_pack,
            enable_hyperframes_expansion=hyperframes_available,
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
        if not plan:
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
            return {
                "success": False,
                "message": "No generated visuals passed the Auto Visuals Director relevance and renderer-fit checks.",
                "suggestion": None,
                "updated_state": state,
                "tool_name": "add_auto_visuals",
            }
        plan, hyperframes_compiler_report = _compile_hyperframes_specs(plan)
        planning_preview["estimated_render_count"] = int(
            hyperframes_compiler_report["estimated_render_count"]
        )
        planning_preview["hyperframes_proof_candidate_count"] = int(
            hyperframes_compiler_report["proof_candidate_count"]
        )
        planning_preview["expected_slow_steps"] = [
            "HyperFrames structural candidate renders",
            "blind inverse decoding",
            "counterfactual relation ablation",
            "counterfactual temporal scramble",
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
            detail = "; ".join(compiler_reasons[:6])
            return {
                "success": False,
                "message": (
                    "No generated visuals had enough source-grounded structure to pass "
                    f"the HyperFrames semantic compiler.{f' Details: {detail}' if detail else ''}"
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
        render_successes: list[tuple[int, dict[str, object], RenderedAsset, str]] = []
        render_errors: list[tuple[int, str]] = []
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
                    render_errors.append((index, str(exc)))
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
                        render_errors.append((index, str(exc)))

        for _, failure in sorted(render_errors, key=lambda item: item[0]):
            render_failures.append(str(failure))

        for _, spec, asset, selection_reason in sorted(
            render_successes, key=lambda item: item[0]
        ):
            visual_qa = _rendered_visual_quality_for_spec(spec, asset)
            visual_qa_payload = visual_qa.to_dict()
            tournament_report = dict(
                (asset.metadata or {}).get("renderer_tournament") or {}
            )
            if tournament_report:
                visual_qa_payload["renderer_tournament"] = tournament_report
            rendered_visual_qa.append(visual_qa_payload)
            if not visual_qa.passed:
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
            has_alpha = bool((asset.metadata or {}).get("has_alpha"))
            requested_comp = str(spec.get("composition_mode") or "replace")
            compose_mode = "replace" if force_fullscreen else ("overlay" if has_alpha and requested_comp in {"overlay", "picture_in_picture"} else requested_comp)
            applied_overlays.append(
                {
                    "start": _as_float(spec.get("start"), 0.0),
                    "end": _as_float(spec.get("end"), 0.0),
                    "asset_path": asset.asset_path,
                    "compose_mode": compose_mode,
                    "has_alpha": has_alpha,
                    "force_fullscreen": force_fullscreen,
                    "position": "center" if compose_mode in {"replace", "overlay"} else spec["position"],
                    "scale": 1.0 if compose_mode in {"replace", "overlay"} else spec["scale"],
                    "visual_id": spec["visual_id"],
                    "card_id": spec["card_id"],
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
                    "auto_visuals_director": spec.get("auto_visuals_director", {}),
                    "hyperframes_compiler": spec.get("hyperframes_compiler", {}),
                    "visual_explanation_ir": spec.get("visual_explanation_ir", {}),
                    "hyperframes_storyboard": spec.get("hyperframes_storyboard", []),
                    "hyperframes_production_contract": spec.get(
                        "hyperframes_production_contract", {}
                    ),
                    "visual_claim_graph": spec.get("visual_claim_graph", {}),
                    "visual_proof_tournament": spec.get(
                        "visual_proof_tournament",
                        {},
                    ),
                    "proof_program_id": spec.get("proof_program_id"),
                    "proof_strategy_id": spec.get("proof_strategy_id"),
                    "proof_encoding": spec.get("proof_encoding"),
                    "semantic_continuity": spec.get("semantic_continuity", {}),
                    "source_asset_grounding": spec.get("source_asset_grounding", {}),
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
                    "renderer_metadata": dict(asset.metadata or {}),
                    "rendered_width": asset.width,
                    "rendered_height": asset.height,
                    "rendered_duration_sec": asset.duration_sec,
                }
            )

        applied_overlays, final_visual_qa = _final_auto_visuals_qa(
            applied_overlays,
            clip_duration=clip_duration,
            coverage_policy=coverage_policy,
        )
        write_run_status(
            bundle_dir,
            feature="auto_visuals",
            phase="qa",
            payload={
                "rendered_count": len(render_successes),
                "selected_count": len(applied_overlays),
                "render_failure_count": len(render_failures),
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
                "auto_visuals_director": visual_director_report,
                "hyperframes_compiler": hyperframes_compiler_report,
                "rendered_visual_qa": rendered_visual_qa,
                "final_visual_qa": final_visual_qa,
                "plan": plan,
                "overlays": [],
                "render_failures": render_failures,
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
                return _delegate_stock_fallback(
                    params,
                    state,
                    "Generated visuals could not pass renderer and final timeline QA.",
                )
            detail = (
                f" Details: {'; '.join(render_failures[:4])}" if render_failures else ""
            )
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
                "auto_visuals_director": visual_director_report,
                "hyperframes_compiler": hyperframes_compiler_report,
                "rendered_visual_qa": rendered_visual_qa,
                "final_visual_qa": final_visual_qa,
                "composite_qa": composite_qa.to_dict(),
                "plan": plan,
                "overlays": applied_overlays,
                "render_failures": render_failures,
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
            "visual_plan_quality_score": visual_plan_quality["score"],
            "auto_visuals_director_score": visual_director_report.get("average_director_score"),
            "hyperframes_compiled_count": hyperframes_compiler_report.get(
                "compiled_count", 0
            ),
            "hyperframes_rejected_count": hyperframes_compiler_report.get(
                "rejected_count", 0
            ),
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
    except (RuntimeError, VideoEngineError, VisualRendererError) as exc:
        return {
            "success": False,
            "message": str(exc),
            "suggestion": None,
            "updated_state": state,
            "tool_name": "add_auto_visuals",
        }
