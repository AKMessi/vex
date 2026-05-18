from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from engine import VideoEngineError, burn_subtitles, probe_video
from state import ProjectState


def _optional_int(value: object) -> int | None:
    if value in {None, ""}:
        return None
    return int(value)


def _optional_float(value: object) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)


def execute(params: dict, state: ProjectState) -> dict:
    srt_path = Path(params["srt_path"]).expanduser().resolve() if params.get("srt_path") else Path(state.working_dir) / "transcript.srt"
    if not srt_path.is_file():
        return {
            "success": False,
            "message": "No SRT file found. Run transcribe_video first.",
            "suggestion": None,
            "updated_state": state,
            "tool_name": "burn_subtitles",
        }

    try:
        font_size = _optional_int(params.get("font_size"))
        font_color = str(params.get("font_color") or "").strip() or None
        outline_color = str(params.get("outline_color") or "").strip() or None
        style = str(params.get("style") or "clean_pop").strip().lower().replace("-", "_")
        emphasis_color = str(params.get("emphasis_color") or "").strip() or None
        background_opacity = _optional_float(params.get("background_opacity"))
        max_words_per_caption = _optional_int(params.get("max_words_per_caption"))
        max_lines = _optional_int(params.get("max_lines"))
        case = str(params.get("case") or "").strip().lower() or None
        position = str(params.get("position", "bottom")).strip().lower()
        if position not in {"bottom", "center", "top"}:
            return {
                "success": False,
                "message": f"Unsupported subtitle position: {position}",
                "suggestion": None,
                "updated_state": state,
                "tool_name": "burn_subtitles",
            }

        output_path = burn_subtitles(
            state.working_file,
            state.working_dir,
            srt_path=str(srt_path),
            font_size=font_size,
            font_color=font_color,
            outline_color=outline_color,
            position=position,
            style=style,
            emphasis_color=emphasis_color,
            background_opacity=background_opacity,
            max_words_per_caption=max_words_per_caption,
            max_lines=max_lines,
            case=case,
        )
        state.working_file = output_path
        state.metadata = probe_video(output_path)
        description = f"Burned {style.replace('_', ' ')} subtitles from {srt_path.name} at {position}"
        stored_params = {
            "srt_path": str(srt_path),
            "style": style,
            "position": position,
        }
        optional_params = {
            "font_size": font_size,
            "font_color": font_color,
            "outline_color": outline_color,
            "emphasis_color": emphasis_color,
            "background_opacity": background_opacity,
            "max_words_per_caption": max_words_per_caption,
            "max_lines": max_lines,
            "case": case,
        }
        stored_params.update({key: value for key, value in optional_params.items() if value is not None})
        op = {
            "op": "burn_subtitles",
            "params": stored_params,
            "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "result_file": output_path,
            "description": description,
        }
        state.apply_operation(op)
        return {
            "success": True,
            "message": description + ".",
            "suggestion": None,
            "updated_state": state,
            "tool_name": "burn_subtitles",
        }
    except (TypeError, ValueError, VideoEngineError) as exc:
        return {
            "success": False,
            "message": str(exc),
            "suggestion": None,
            "updated_state": state,
            "tool_name": "burn_subtitles",
        }
