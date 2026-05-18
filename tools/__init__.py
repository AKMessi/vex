from __future__ import annotations

from importlib import import_module
from typing import Any, Callable


ToolExecutor = Callable[[dict[str, Any], Any], dict[str, Any]]


def _executor(module_name: str, function_name: str) -> ToolExecutor:
    def run(params: dict[str, Any], state: Any) -> dict[str, Any]:
        module = import_module(f"tools.{module_name}")
        function = getattr(module, function_name)
        return function(params, state)

    return run


TOOL_EXECUTORS: dict[str, ToolExecutor] = {
    "get_video_info": _executor("info", "execute"),
    "trim_clip": _executor("trim", "execute"),
    "merge_clips": _executor("merge", "execute"),
    "adjust_speed": _executor("speed", "execute"),
    "add_transition": _executor("transitions", "execute"),
    "add_text_overlay": _executor("overlay", "execute"),
    "extract_audio": _executor("audio", "execute_extract"),
    "replace_audio": _executor("audio", "execute_replace"),
    "mute_segment": _executor("audio", "execute_mute"),
    "trim_silence": _executor("silence", "execute"),
    "auto_color_grade": _executor("color_grade", "execute"),
    "burn_subtitles": _executor("subtitles", "execute"),
    "summarize_clip": _executor("summarize", "execute"),
    "create_auto_shorts": _executor("auto_shorts", "execute"),
    "add_auto_broll": _executor("pexels_broll", "execute"),
    "add_auto_visuals": _executor("auto_visuals", "execute"),
    "add_auto_effects": _executor("auto_effects", "execute"),
    "plan_encode": _executor("encode", "execute_plan"),
    "run_pending_encode": _executor("encode", "execute_run_pending"),
    "export_video": _executor("export", "execute"),
    "undo": _executor("undo", "execute_undo"),
    "redo": _executor("undo", "execute_redo"),
    "transcribe_video": _executor("transcript", "execute"),
}
