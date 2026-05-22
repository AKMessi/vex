from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

import config
from broll_intelligence import ensure_writable_dir, safe_stem, writable_dir_candidates
from engine import VideoEngineError, apply_visual_overlays, probe_video
from renderers import RenderedAsset, VisualRendererError, renderer_capabilities, resolve_renderer
from state import ProjectState, restrict_timed_items_to_available_ranges, utc_now_iso
from tools.transcript import execute as transcribe
from tools.transcript_utils import load_transcript_bundle
from tools.undo import refresh_generated_overlay_ops
from visual_intelligence import (
    STYLE_PACKS,
    THEME_BY_VISUAL_TYPE,
    analyze_visual_plan_with_llm,
    build_visual_context_cards,
    detect_scene_cuts,
)
from visual_program import apply_visual_program_to_specs, build_visual_narrative_program


def _emit_progress(message: str) -> None:
    print(f"[auto_visuals] {message}", flush=True)


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


def _should_force_fullscreen_visuals(params: dict, *, mode: str, renderer_name: str) -> bool:
    if "force_fullscreen" in params:
        return _as_bool(params.get("force_fullscreen"), True)
    if "fullscreen" in params:
        return _as_bool(params.get("fullscreen"), True)
    if "full_screen" in params:
        return _as_bool(params.get("full_screen"), True)
    return mode == "generated_only" or renderer_name in {"auto", "hyperframes", "manim"}


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
        overlays = ((op.get("params") or {}).get("overlays") or [])
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
    fresh_cards = [card for card in cards if str(card.get("card_id") or "") not in used_card_ids]
    fresh_only_mode = bool(used_card_ids) and len(fresh_cards) >= max_visuals
    for card in cards:
        normalized = dict(card)
        card_id = str(normalized.get("card_id") or "").strip()
        original_priority = _as_float(normalized.get("priority"), 0.0)
        recently_used = card_id in used_card_ids
        normalized["original_priority"] = original_priority
        normalized["recently_used"] = recently_used
        if recently_used:
            normalized["priority"] = round(original_priority - (32.0 if fresh_only_mode else 18.0), 2)
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
        raise RuntimeError("Transcript generation completed, but no usable transcript segments were found.")
    return transcript_bundle


def _provider_and_model(state: ProjectState) -> tuple[str, str]:
    provider_name = (state.provider or config.PROVIDER or "gemini").strip().lower()
    if provider_name not in {"gemini", "claude"}:
        provider_name = "gemini"
    model_name = state.model or (config.CLAUDE_MODEL if provider_name == "claude" else config.GEMINI_MODEL)
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


def _prepare_visual_spec(
    spec: dict[str, object],
    *,
    style_pack: str,
    provider_name: str,
    model_name: str,
) -> dict[str, object]:
    prepared = dict(spec)
    resolved_style_pack = style_pack
    if (resolved_style_pack or "auto").strip().lower() in {"", "auto"}:
        resolved_style_pack = str(prepared.get("style_pack") or "auto")
    _apply_style_override(prepared, resolved_style_pack)
    prepared["generation_provider"] = provider_name
    prepared["generation_model"] = model_name
    return prepared


def _ensure_unique_visual_ids(specs: list[dict[str, object]]) -> list[dict[str, object]]:
    normalized_specs: list[dict[str, object]] = []
    for index, spec in enumerate(specs, start=1):
        normalized = dict(spec)
        normalized["visual_id"] = f"visual_{index:03d}"
        normalized_specs.append(normalized)
    return normalized_specs


def _render_generated_visual(
    spec: dict[str, object],
    *,
    preferred_renderer: str,
    render_root: Path,
    width: int,
    height: int,
    fps: float,
) -> tuple[RenderedAsset, str]:
    failures: list[str] = []
    attempted: set[str] = set()
    require_generated_scene = bool(spec.get("require_generated_scene"))
    if require_generated_scene:
        locked_preference = str(spec.get("renderer_hint") or preferred_renderer or "manim").strip().lower() or "manim"
        preference_order = [locked_preference]
    else:
        preference_order = [preferred_renderer, str(spec.get("renderer_hint") or "auto"), "auto"]
    for candidate_preference in preference_order:
        while True:
            try:
                renderer, reason = resolve_renderer(spec, preferred=candidate_preference, exclude=attempted)
            except VisualRendererError as exc:
                failures.append(str(exc))
                break
            attempted.add(renderer.name)
            if require_generated_scene and renderer.name != str(spec.get("renderer_hint") or "manim").strip().lower():
                failures.append(
                    f"{renderer.name}: premium generated scenes are locked to {spec.get('renderer_hint') or 'manim'}."
                )
                break
            try:
                asset = renderer.render(spec, render_root=render_root, width=width, height=height, fps=fps)
                return asset, reason
            except VisualRendererError as exc:
                failures.append(f"{renderer.name}: {exc}")
                if len(attempted) >= 3:
                    break
        if len(attempted) >= 3:
            break
    raise VisualRendererError("; ".join(failures) or "No renderer could produce the generated visual.")


def _max_render_workers(params: dict, visual_count: int, specs: list[dict[str, object]] | None = None) -> int:
    specs = specs or []
    renderer_name = str(params.get("renderer") or "auto").strip().lower()
    if renderer_name in {"auto", "hyperframes"} or any(
        str(spec.get("renderer_hint") or "").strip().lower() == "hyperframes"
        for spec in specs
    ):
        return 1
    requested = int(params.get("max_render_workers", 4) or 4)
    return max(1, min(requested, visual_count, 4))


def execute(params: dict, state: ProjectState) -> dict:
    mode = str(params.get("mode") or "generated_only").strip().lower()
    if mode not in {"generated_only", "hybrid", "stock_only"}:
        mode = "generated_only"
    renderer_name = str(params.get("renderer") or "auto").strip().lower()
    style_pack = str(params.get("style_pack") or "auto").strip().lower()
    refresh_existing = bool(params.get("refresh_existing", True))
    max_visuals = max(1, min(int(params.get("max_visuals", 5) or 5), 12))
    min_visual_sec = max(1.6, min(float(params.get("min_visual_sec", 2.2) or 2.2), 6.0))
    max_visual_sec = max(min_visual_sec, min(float(params.get("max_visual_sec", 3.6) or 3.6), 8.0))
    force_fullscreen = _should_force_fullscreen_visuals(params, mode=mode, renderer_name=renderer_name)

    if mode == "stock_only":
        return _delegate_stock_fallback(params, state, "Auto visuals was asked to use stock-only mode.")

    try:
        refreshed_auto_overlay_counts: dict[str, int] = {}
        if refresh_existing:
            refreshed_auto_overlay_counts = _refresh_existing_auto_overlays(state)
            if refreshed_auto_overlay_counts:
                details = []
                if refreshed_auto_overlay_counts.get("add_auto_visuals"):
                    count = refreshed_auto_overlay_counts["add_auto_visuals"]
                    details.append(f"{count} auto-visual pass{'es' if count != 1 else ''}")
                if refreshed_auto_overlay_counts.get("add_auto_broll"):
                    count = refreshed_auto_overlay_counts["add_auto_broll"]
                    details.append(f"{count} auto B-roll pass{'es' if count != 1 else ''}")
                _emit_progress(f"Cleared prior auto overlays before replanning: {', '.join(details)}.")
        _emit_progress("Loading transcript bundle...")
        transcript_bundle = _ensure_transcript_bundle(state)
        metadata = state.metadata or probe_video(state.working_file)
        clip_duration = float(metadata.get("duration_sec") or 0.0)
        width = int(metadata.get("width") or 0)
        height = int(metadata.get("height") or 0)
        fps = float(metadata.get("fps") or 30.0) or 30.0
        if clip_duration <= 0 or width <= 0 or height <= 0:
            raise RuntimeError("The current working video does not have valid timing or resolution metadata.")

        transcript_segments = _as_list(transcript_bundle.get("segments"))
        transcript_words = _as_list(transcript_bundle.get("words"))
        sentence_segments = _as_list(transcript_bundle.get("sentences"))
        blocked_ranges = state.overlay_ranges()
        _emit_progress("Detecting safe scene cuts...")
        scene_cuts = detect_scene_cuts(state.working_file)
        _emit_progress("Building visual candidate cards from the transcript...")
        cards = build_visual_context_cards(
            sentence_segments,
            transcript_segments,
            clip_duration,
            words=transcript_words,
            scene_cuts=scene_cuts,
        )
        cards = restrict_timed_items_to_available_ranges(
            cards,
            blocked_ranges,
            min_duration_sec=max(0.45, min_visual_sec * 0.5),
        )
        prior_card_ids = _prior_auto_visual_card_ids(state)
        cards = _filter_previously_used_cards(cards, prior_card_ids, max_visuals=max_visuals)
        if not cards:
            raise RuntimeError("No transcript-aligned visual cards were available for planning after respecting existing full-screen overlay windows.")
        _emit_progress("Building the video-level visual narrative program...")
        visual_program = build_visual_narrative_program(
            cards,
            clip_duration=clip_duration,
            max_visuals=max_visuals,
            scene_cuts=scene_cuts,
            prefer_premium=force_fullscreen,
        )
        visual_program_payload = visual_program.to_dict()
        provider_name, model_name = _provider_and_model(state)
        prefer_premium = force_fullscreen
        capabilities = renderer_capabilities()
        bundle_root = ensure_writable_dir(
            writable_dir_candidates(state.working_dir, state.output_dir, state.project_id, "auto_visual_bundles")
        )
        _emit_progress("Planning the generated visual beats...")
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
        hyperframes_available = any(
            str(item.get("name") or "").strip().lower() == "hyperframes" and bool(item.get("available"))
            for item in capabilities
        )
        plan = apply_visual_program_to_specs(
            plan,
            visual_program_payload,
            style_pack=style_pack,
            enable_hyperframes_expansion=hyperframes_available,
        )
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
        if not plan:
            return {
                "success": False,
                "message": "No clear generated-visual windows were available after respecting the visuals already on this project timeline.",
                "suggestion": None,
                "updated_state": state,
                "tool_name": "add_auto_visuals",
            }
        timestamp_label = utc_now_iso().replace(":", "-").replace("+00:00", "Z")
        bundle_dir = bundle_root / f"{safe_stem(state.project_name)}_auto_visuals_{timestamp_label}"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        render_root = bundle_dir / "renders"
        render_root.mkdir(parents=True, exist_ok=True)

        applied_overlays: list[dict] = []
        render_failures: list[str] = []
        prepared_specs = [
            _prepare_visual_spec(
                spec,
                style_pack=style_pack,
                provider_name=provider_name,
                model_name=model_name,
            )
            for spec in plan
        ]
        render_successes: list[tuple[int, dict[str, object], RenderedAsset, str]] = []
        render_errors: list[tuple[int, str]] = []
        worker_count = _max_render_workers(params, len(prepared_specs), prepared_specs)
        _emit_progress(
            f"Rendering {len(prepared_specs)} generated visual{'s' if len(prepared_specs) != 1 else ''} with {worker_count} worker{'s' if worker_count != 1 else ''}..."
        )
        if worker_count == 1:
            for index, spec in enumerate(prepared_specs):
                try:
                    _emit_progress(f"Rendering {spec.get('visual_id', f'visual_{index + 1:03d}')}...")
                    asset, selection_reason = _render_generated_visual(
                        spec,
                        preferred_renderer=renderer_name,
                        render_root=render_root,
                        width=width,
                        height=height,
                        fps=fps,
                    )
                    render_successes.append((index, spec, asset, selection_reason))
                except VisualRendererError as exc:
                    _emit_progress(
                        f"Render failed for {spec.get('visual_id', f'visual_{index + 1:03d}')}: {exc}"
                    )
                    render_errors.append((index, str(exc)))
        else:
            with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="vex-auto-visuals") as executor:
                future_map = {
                    executor.submit(
                        _render_generated_visual,
                        spec,
                        preferred_renderer=renderer_name,
                        render_root=render_root,
                        width=width,
                        height=height,
                        fps=fps,
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

        for _, spec, asset, selection_reason in sorted(render_successes, key=lambda item: item[0]):
            applied_overlays.append(
                {
                    "start": _as_float(spec.get("start"), 0.0),
                    "end": _as_float(spec.get("end"), 0.0),
                    "asset_path": asset.asset_path,
                    "compose_mode": "replace" if force_fullscreen else spec["composition_mode"],
                    "force_fullscreen": force_fullscreen,
                    "position": "center" if force_fullscreen else spec["position"],
                    "scale": 1.0 if force_fullscreen else spec["scale"],
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

        if not applied_overlays:
            if mode == "hybrid" and config.PEXELS_API_KEY:
                return _delegate_stock_fallback(
                    params,
                    state,
                    "Generated visuals could not be rendered with the current setup.",
                )
            detail = f" Details: {'; '.join(render_failures[:4])}" if render_failures else ""
            return {
                "success": False,
                "message": f"Vex planned generated visuals, but none could be rendered.{detail}",
                "suggestion": None,
                "updated_state": state,
                "tool_name": "add_auto_visuals",
            }

        _emit_progress("Compositing the generated visuals back into the working cut...")
        output_path = apply_visual_overlays(state.working_file, state.working_dir, applied_overlays)
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
            "renderer_capabilities": capabilities,
            "render_workers": worker_count,
            "transcript_paths": transcript_bundle.get("paths", {}),
            "scene_cuts": scene_cuts,
            "blocked_ranges": blocked_ranges,
            "visual_program": visual_program_payload,
            "plan": plan,
            "overlays": applied_overlays,
            "render_failures": render_failures,
        }
        manifest_path = bundle_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        renderer_counts: dict[str, int] = {}
        for overlay in applied_overlays:
            renderer_counts[str(overlay.get("renderer") or "unknown")] = renderer_counts.get(
                str(overlay.get("renderer") or "unknown"),
                0,
            ) + 1
        renderer_summary = ", ".join(f"{name} x{count}" for name, count in sorted(renderer_counts.items()))

        notes_lines = [
            "# Auto Visuals Notes",
            "",
            f"Renderer preference: {renderer_name}",
            f"Style pack: {style_pack}",
            f"Mode: {mode}",
            "",
        ]
        for overlay in applied_overlays:
            renderer_metadata = dict(overlay.get("renderer_metadata") or {})
            variant_selection = dict(renderer_metadata.get("variant_selection") or {})
            art_direction = dict(renderer_metadata.get("art_direction") or {})
            notes_lines.extend(
                [
                    f"## {overlay['start']:.2f}s-{overlay['end']:.2f}s",
                    f"Template: {overlay['template']}",
                    f"Headline: {overlay['headline']}",
                    f"Renderer: {overlay['renderer']}",
                    f"Composition: {overlay['compose_mode']}",
                    f"Why: {overlay['rationale']}",
                ]
            )
            if variant_selection:
                quality_score = variant_selection.get("selected_quality_score")
                quality_label = "passed" if variant_selection.get("selected_quality_passed") else "review"
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
                "description": f"Added {len(applied_overlays)} transcript-aligned generated visuals ({renderer_summary})",
            }
        )
        _emit_progress("Auto visuals complete.")
        return {
            "success": True,
            "message": (
                f"Added {len(applied_overlays)} transcript-aligned generated visuals using {renderer_summary} "
                f"(preference: {renderer_name}). Manifest: {manifest_path}"
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
