from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from broll_intelligence import (
    analyze_broll_plan_with_llm,
    build_context_cards,
    choose_candidate_with_llm,
    collect_search_candidates,
    download_file,
    ensure_writable_dir,
    safe_stem,
    video_orientation,
    writable_dir_candidates,
)
from engine import VideoEngineError, apply_b_roll_overlays, probe_video
from state import ProjectState, merge_time_ranges, restrict_timed_items_to_available_ranges, utc_now_iso
from tools.transcript import execute as transcribe
from tools.transcript_utils import parse_srt, transcript_artifact_path
from tools.undo import refresh_generated_overlay_ops


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


def execute(params: dict, state: ProjectState) -> dict:
    if not config.PEXELS_API_KEY:
        return {
            "success": False,
            "message": "PEXELS_API_KEY is missing. Set it in your environment or .env file to enable auto B-roll.",
            "suggestion": None,
            "updated_state": state,
            "tool_name": "add_auto_broll",
        }

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
        plan = analyze_broll_plan_with_llm(
            provider_name=provider_name,
            model_name=model_name,
            cards=cards,
            clip_duration=clip_duration,
            max_overlays=max_overlays,
            min_overlay_sec=min_overlay_sec,
            max_overlay_sec=max_overlay_sec,
            orientation=target_orientation,
        )
        plan = restrict_timed_items_to_available_ranges(
            plan,
            blocked_ranges,
            min_duration_sec=min_overlay_sec,
        )

        cache_dir = ensure_writable_dir(writable_dir_candidates(state.working_dir, state.output_dir, state.project_id, "pexels_cache"))
        bundle_root = ensure_writable_dir(writable_dir_candidates(state.working_dir, state.output_dir, state.project_id, "auto_broll_bundles"))
        used_video_ids: set[int] = set()
        applied_overlays: list[dict] = []
        planning_failures: list[str] = []
        rate_limits: dict[str, str] = {}

        for plan_item in plan:
            candidates, rate_limits = collect_search_candidates(
                plan_item=plan_item,
                target_orientation=target_orientation,
                target_width=int(metadata.get("width") or 1080),
                target_height=int(metadata.get("height") or 1920),
            )
            candidates = [candidate for candidate in candidates if int(candidate["video"].get("id") or 0) not in used_video_ids]
            selected_candidate, selection_reason = choose_candidate_with_llm(
                provider_name=provider_name,
                model_name=model_name,
                plan_item=plan_item,
                candidates=candidates,
            )
            if selected_candidate is None:
                planning_failures.append(f"{plan_item['subtitle_text']}: no suitable Pexels candidate")
                continue

            video = selected_candidate["video"]
            file_info = selected_candidate["file_info"]
            video_id = int(video.get("id") or 0)
            if video_id:
                used_video_ids.add(video_id)
            file_token = str(file_info.get("id") or f"{file_info.get('width')}_{file_info.get('height')}" or "stock")
            asset_path = cache_dir / f"pexels_{video.get('id')}_{file_token}.mp4"
            if not asset_path.exists():
                download_file(str(file_info["link"]), asset_path)
            applied_overlays.append(
                {
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
                    "asset_path": str(asset_path),
                    "pexels_video_id": video.get("id"),
                    "pexels_url": video.get("url"),
                    "creator_name": (video.get("user") or {}).get("name"),
                    "creator_url": (video.get("user") or {}).get("url"),
                    "preview_image": video.get("image"),
                    "video_duration": video.get("duration"),
                    "file_width": file_info.get("width"),
                    "file_height": file_info.get("height"),
                    "file_fps": file_info.get("fps"),
                }
            )

        if not applied_overlays:
            detail = f" Details: {'; '.join(planning_failures[:4])}" if planning_failures else ""
            return {
                "success": False,
                "message": f"Vex planned subtitle-aligned B-roll beats, but Pexels did not return usable stock clips.{detail}",
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
            "pexels_attribution_required": True,
            "pexels_link": "https://www.pexels.com",
            "rate_limits": rate_limits,
            "blocked_ranges": blocked_ranges,
            "plan": plan,
            "overlays": applied_overlays,
            "planning_failures": planning_failures,
        }
        manifest_path = bundle_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        credits_lines = [
            "# Pexels Attribution",
            "",
            "Photos and videos provided by Pexels: https://www.pexels.com",
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
                    f"   Pexels video: {item.get('pexels_url') or 'unknown'}",
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
                    f"Why: {item['selection_reason']}",
                    "",
                ]
            )
        (bundle_dir / "pexels_attribution.md").write_text("\n".join(credits_lines), encoding="utf-8")
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
                    "manifest_path": str(manifest_path),
                    "overlays": applied_overlays,
                },
                "timestamp": utc_now_iso(),
                "result_file": output_path,
                "description": f"Added {len(applied_overlays)} subtitle-aligned auto B-roll overlays from Pexels",
            }
        )
        return {
            "success": True,
            "message": f"Added {len(applied_overlays)} subtitle-aligned auto B-roll overlays from Pexels. Manifest: {manifest_path}",
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
