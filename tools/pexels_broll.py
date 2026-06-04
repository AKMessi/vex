from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from broll_intelligence import (
    build_broll_director_plan,
    build_context_cards,
    choose_candidate_with_llm,
    clip_text,
    collect_search_candidates,
    configured_stock_provider_names,
    download_file,
    ensure_writable_dir,
    evaluate_broll_final_plan_with_llm,
    missing_stock_provider_keys,
    safe_stem,
    stock_provider_status,
    video_orientation,
    writable_dir_candidates,
)
from engine import VideoEngineError, apply_b_roll_overlays, probe_video
from state import ProjectState, merge_time_ranges, restrict_timed_items_to_available_ranges, utc_now_iso
from tools.creative_intelligence import build_video_understanding_graph
from tools.transcript import execute as transcribe
from tools.transcript_utils import parse_srt, transcript_artifact_path
from tools.undo import refresh_generated_overlay_ops
from visual_intelligence import detect_scene_cuts


def _ensure_transcript_segments(state: ProjectState) -> tuple[Path, list[dict[str, float | str]]]:
    srt_path = transcript_artifact_path(state.working_dir, "transcript.srt")
    if srt_path is None:
        result = transcribe({}, state)
        if not result["success"]:
            raise RuntimeError(result["message"])
        srt_path = transcript_artifact_path(state.working_dir, "transcript.srt")
    if srt_path is None:
        raise RuntimeError("Transcript generation completed, but no safe transcript.srt artifact was found.")
    segments = parse_srt(srt_path)
    if not segments:
        raise RuntimeError("Transcript was empty, so Vex could not plan B-roll beats.")
    return srt_path, segments


def _refresh_existing_auto_broll(state: ProjectState) -> dict[str, int]:
    return refresh_generated_overlay_ops(
        state,
        remove_ops={"add_auto_broll"},
    )


def _stock_asset_suffix(url: str) -> str:
    suffix = Path(str(url or "").split("?", 1)[0]).suffix.lower()
    if suffix in {".mp4", ".mov", ".webm", ".m4v"}:
        return suffix
    return ".mp4"


def execute(params: dict, state: ProjectState) -> dict:
    provider_param = params.get("providers") or params.get("provider")
    active_providers = configured_stock_provider_names(provider_param)
    if not active_providers:
        missing = ", ".join(missing_stock_provider_keys(provider_param))
        fallback_missing = missing or "PEXELS_API_KEY, PIXABAY_API_KEY, COVERR_API_KEY"
        return {
            "success": False,
            "message": (
                "Auto B-roll needs at least one configured stock video API key. "
                f"Set one of PEXELS_API_KEY, PIXABAY_API_KEY, or COVERR_API_KEY in your environment or .env file. Missing for the requested provider set: {fallback_missing}."
            ),
            "suggestion": None,
            "updated_state": state,
            "tool_name": "add_auto_broll",
        }
    provider_status = stock_provider_status(provider_param)
    provider_labels = ", ".join(item["display_name"] for item in provider_status if item["provider"] in active_providers)

    max_overlays = max(1, min(int(params.get("max_overlays", 5) or 5), 8))
    min_overlay_sec = max(0.8, min(float(params.get("min_overlay_sec", 1.2) or 1.2), 6.0))
    max_overlay_sec = max(min_overlay_sec, min(float(params.get("max_overlay_sec", 2.8) or 2.8), 8.0))

    try:
        refreshed_counts: dict[str, int] = {}
        if bool(params.get("refresh_existing", True)):
            refreshed_counts = _refresh_existing_auto_broll(state)
        srt_path, transcript_segments = _ensure_transcript_segments(state)
        metadata = state.metadata or probe_video(state.working_file)
        clip_duration = float(metadata.get("duration_sec") or 0.0)
        target_orientation = video_orientation(int(metadata.get("width") or 0), int(metadata.get("height") or 0))
        blocked_ranges = merge_time_ranges(
            state.replace_overlay_ranges()
            + state.overlay_ranges(include_ops={"add_auto_visuals"}, include_picture_in_picture=True),
            gap_sec=0.08,
        )
        provider_name = (state.provider or config.PROVIDER or "gemini").strip().lower()
        if provider_name not in {"gemini", "claude"}:
            provider_name = "gemini"
        model_name = state.model or (config.CLAUDE_MODEL if provider_name == "claude" else config.GEMINI_MODEL)

        transcript_text = clip_text(transcript_segments)
        try:
            scene_cuts = detect_scene_cuts(state.working_file)
        except Exception:
            scene_cuts = []
        creative_graph = build_video_understanding_graph(
            transcript_text=transcript_text,
            segments=transcript_segments,
            metadata=metadata,
            scene_cuts=scene_cuts,
            quality_tier="world_class_local",
            source_context={
                "feature": "auto_broll",
                "providers": active_providers,
                "orientation": target_orientation,
            },
        )

        cards = build_context_cards(transcript_segments, clip_duration)
        cards = restrict_timed_items_to_available_ranges(
            cards,
            blocked_ranges,
            min_duration_sec=max(0.5, min_overlay_sec * 0.6),
        )
        if not cards:
            raise RuntimeError("No subtitle-aligned transcript cards were available for B-roll planning after respecting existing full-screen overlay windows.")
        if refreshed_counts.get("add_auto_broll"):
            count = refreshed_counts["add_auto_broll"]
            print(
                f"[auto_broll] Cleared {count} prior auto B-roll pass{'es' if count != 1 else ''} before replanning.",
                flush=True,
            )
        director_plan, plan = build_broll_director_plan(
            cards=cards,
            clip_duration=clip_duration,
            max_overlays=max_overlays,
            min_overlay_sec=min_overlay_sec,
            max_overlay_sec=max_overlay_sec,
            orientation=target_orientation,
            provider_name=provider_name,
            model_name=model_name,
            graph=creative_graph,
        )
        director_plan_payload = director_plan.to_dict()
        plan = restrict_timed_items_to_available_ranges(
            plan,
            blocked_ranges,
            min_duration_sec=min_overlay_sec,
        )
        if not plan:
            raise RuntimeError("B-roll Director v2 did not find any non-abrupt, high-signal stock insert windows after timeline and context checks.")

        cache_dir = ensure_writable_dir(writable_dir_candidates(state.working_dir, state.output_dir, state.project_id, "stock_broll_cache"))
        bundle_root = ensure_writable_dir(writable_dir_candidates(state.working_dir, state.output_dir, state.project_id, "auto_broll_bundles"))
        used_assets: set[tuple[str, str]] = set()
        applied_overlays: list[dict] = []
        planning_failures: list[str] = []
        rate_limits: dict[str, object] = {}

        for plan_item in plan:
            candidates, candidate_rate_limits = collect_search_candidates(
                plan_item=plan_item,
                target_orientation=target_orientation,
                target_width=int(metadata.get("width") or 1080),
                target_height=int(metadata.get("height") or 1920),
                provider_names=active_providers,
            )
            rate_limits.update(candidate_rate_limits)
            provider_errors = candidate_rate_limits.get("_errors")
            if isinstance(provider_errors, dict) and provider_errors:
                planning_failures.extend(
                    f"{provider}: {message}"
                    for provider, message in provider_errors.items()
                )
            candidates = [
                candidate
                for candidate in candidates
                if (str(candidate.get("provider") or ""), str(candidate.get("provider_id") or candidate.get("download_url") or ""))
                not in used_assets
            ]
            selected_candidate, selection_reason = choose_candidate_with_llm(
                provider_name=provider_name,
                model_name=model_name,
                plan_item=plan_item,
                candidates=candidates,
            )
            if selected_candidate is None:
                planning_failures.append(f"{plan_item['subtitle_text']}: no suitable stock candidate")
                continue

            file_info = selected_candidate["file_info"]
            provider = str(selected_candidate.get("provider") or "stock")
            provider_display_name = str(selected_candidate.get("provider_display_name") or provider.title())
            provider_id = str(selected_candidate.get("provider_id") or selected_candidate.get("download_url") or "stock")
            used_assets.add((provider, provider_id))
            file_token = safe_stem(str(file_info.get("id") or f"{file_info.get('width')}_{file_info.get('height')}" or "stock"))
            download_url = str(selected_candidate.get("download_url") or file_info.get("link") or "")
            asset_path = cache_dir / f"{safe_stem(provider)}_{safe_stem(provider_id)}_{file_token}{_stock_asset_suffix(download_url)}"
            if not asset_path.exists():
                download_file(download_url, asset_path)
            overlay = {
                "start": float(plan_item["start"]),
                "end": float(plan_item["end"]),
                "card_id": plan_item["card_id"],
                "subtitle_text": plan_item["subtitle_text"],
                "context_text": plan_item["context_text"],
                "keywords": plan_item["keywords"],
                "visual_type": plan_item["visual_type"],
                "primary_query": plan_item["primary_query"],
                "backup_queries": plan_item.get("backup_queries", []),
                "must_include": plan_item.get("must_include", []),
                "avoid": plan_item.get("avoid", []),
                "confidence": plan_item["confidence"],
                "direction": plan_item["direction"],
                "rationale": plan_item["rationale"],
                "selection_reason": selection_reason,
                "query_used": selected_candidate["matched_query"],
                "candidate_score": selected_candidate["score"],
                "candidate_slug_tokens": selected_candidate["slug_tokens"],
                "visual_verification": selected_candidate.get("visual_verification"),
                "broll_intent": plan_item.get("broll_intent"),
                "provider_queries": plan_item.get("provider_queries"),
                "creative_graph_signals": plan_item.get("creative_graph_signals"),
                "director_score": plan_item.get("director_score"),
                "asset_path": str(asset_path),
                "stock_provider": provider,
                "stock_provider_display_name": provider_display_name,
                "stock_provider_id": provider_id,
                "stock_source_url": selected_candidate.get("source_url"),
                "stock_download_url": download_url,
                "stock_license_name": selected_candidate.get("license_name"),
                "stock_license_url": selected_candidate.get("license_url"),
                "attribution_required": bool(selected_candidate.get("attribution_required")),
                "creator_name": selected_candidate.get("creator_name"),
                "creator_url": selected_candidate.get("creator_url"),
                "preview_image": selected_candidate.get("preview_url"),
                "video_duration": selected_candidate.get("duration"),
                "file_width": file_info.get("width"),
                "file_height": file_info.get("height"),
                "file_fps": file_info.get("fps"),
            }
            if provider == "pexels":
                overlay.update(
                    {
                        "pexels_video_id": provider_id,
                        "pexels_url": selected_candidate.get("source_url"),
                    }
                )
            applied_overlays.append(overlay)

        if not applied_overlays:
            detail = f" Details: {'; '.join(planning_failures[:4])}" if planning_failures else ""
            return {
                "success": False,
                "message": f"Vex planned subtitle-aligned B-roll beats, but the configured stock providers did not return usable clips.{detail}",
                "suggestion": None,
                "updated_state": state,
                "tool_name": "add_auto_broll",
            }

        applied_overlays, final_qa_report = evaluate_broll_final_plan_with_llm(
            provider_name=provider_name,
            model_name=model_name,
            overlays=applied_overlays,
            clip_duration=clip_duration,
            transcript_excerpt=transcript_text,
            director_plan=director_plan_payload,
        )
        if not applied_overlays:
            detail = "; ".join(
                f"{item.get('card_id')}: {item.get('reason')}"
                for item in final_qa_report.get("decisions", [])[:4]
                if isinstance(item, dict)
            )
            return {
                "success": False,
                "message": f"B-roll Director v2 found stock clips, but the final QA gate rejected them as too abrupt or weak.{f' Details: {detail}' if detail else ''}",
                "suggestion": None,
                "updated_state": state,
                "tool_name": "add_auto_broll",
            }

        output_path = apply_b_roll_overlays(state.working_file, state.working_dir, applied_overlays)
        state.working_file = output_path
        state.metadata = probe_video(output_path)

        timestamp_label = utc_now_iso().replace(":", "-").replace("+00:00", "Z")
        bundle_dir = bundle_root / f"{safe_stem(state.project_name)}_auto_broll_{timestamp_label}"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "created_at": utc_now_iso(),
            "project_id": state.project_id,
            "project_name": state.project_name,
            "source_video": state.source_files[0] if state.source_files else state.working_file,
            "working_file": state.working_file,
            "transcript_srt": str(srt_path),
            "stock_providers": active_providers,
            "stock_provider_status": provider_status,
            "stock_attribution_required": any(bool(item.get("attribution_required")) for item in applied_overlays),
            "pexels_attribution_required": any(item.get("stock_provider") == "pexels" for item in applied_overlays),
            "provider_links": {
                "pexels": "https://www.pexels.com",
                "pixabay": "https://pixabay.com",
                "coverr": "https://coverr.co",
            },
            "rate_limits": rate_limits,
            "blocked_ranges": blocked_ranges,
            "scene_cuts": scene_cuts,
            "creative_graph_summary": creative_graph.compact(beat_limit=12, moment_limit=6),
            "broll_director_plan": director_plan_payload,
            "final_qa": final_qa_report,
            "plan": plan,
            "overlays": applied_overlays,
            "planning_failures": planning_failures,
        }
        manifest_path = bundle_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        credits_lines = [
            "# Stock B-roll Attribution",
            "",
            "Stock clips were selected from configured provider APIs and should be credited according to the provider license/terms.",
            "",
        ]
        notes_lines = [
            "# Auto B-roll Notes",
            "",
            "These inserts were aligned to subtitle cards and reranked against nearby transcript context.",
            "",
        ]
        for index, item in enumerate(applied_overlays, start=1):
            credits_lines.extend(
                [
                    f"{index}. {item['start']:.2f}s-{item['end']:.2f}s",
                    f"   Subtitle anchor: {item['subtitle_text']}",
                    f"   Query used: {item['query_used']}",
                    f"   Provider: {item.get('stock_provider_display_name') or item.get('stock_provider') or 'unknown'}",
                    f"   Source: {item.get('stock_source_url') or 'unknown'}",
                    f"   License: {item.get('stock_license_name') or 'unknown'} ({item.get('stock_license_url') or 'n/a'})",
                    f"   Creator: {item.get('creator_name') or 'unknown'} ({item.get('creator_url') or 'n/a'})",
                    "",
                ]
            )
            notes_lines.extend(
                [
                    f"## {item['start']:.2f}s-{item['end']:.2f}s",
                    f"Subtitle: {item['subtitle_text']}",
                    f"Context: {item['context_text']}",
                    f"Primary query: {item['primary_query']}",
                    f"Selected query: {item['query_used']}",
                    f"Provider: {item.get('stock_provider_display_name') or item.get('stock_provider')}",
                    f"Why: {item['selection_reason']}",
                    "",
                ]
            )
        (bundle_dir / "stock_attribution.md").write_text("\n".join(credits_lines), encoding="utf-8")
        (bundle_dir / "notes.md").write_text("\n".join(notes_lines), encoding="utf-8")

        state.artifacts["latest_auto_broll"] = {
            "created_at": manifest["created_at"],
            "manifest_path": str(manifest_path),
            "bundle_dir": str(bundle_dir),
            "count": len(applied_overlays),
        }
        history = list(state.artifacts.get("auto_broll_history") or [])
        history.append(state.artifacts["latest_auto_broll"])
        state.artifacts["auto_broll_history"] = history[-10:]
        state.apply_operation(
            {
                "op": "add_auto_broll",
                "params": {
                    "max_overlays": max_overlays,
                    "min_overlay_sec": min_overlay_sec,
                    "max_overlay_sec": max_overlay_sec,
                    "providers": active_providers,
                    "manifest_path": str(manifest_path),
                    "overlays": applied_overlays,
                    "broll_director_version": director_plan_payload.get("version"),
                    "final_qa": final_qa_report,
                },
                "timestamp": utc_now_iso(),
                "result_file": output_path,
                "description": f"Added {len(applied_overlays)} Director v2 subtitle-aligned auto B-roll overlays from {provider_labels}",
            }
        )
        return {
            "success": True,
            "message": f"Added {len(applied_overlays)} Director v2 subtitle-aligned auto B-roll overlays from {provider_labels}. Manifest: {manifest_path}",
            "suggestion": None,
            "updated_state": state,
            "tool_name": "add_auto_broll",
        }
    except (RuntimeError, VideoEngineError) as exc:
        return {
            "success": False,
            "message": str(exc),
            "suggestion": None,
            "updated_state": state,
            "tool_name": "add_auto_broll",
        }
