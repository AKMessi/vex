from __future__ import annotations

from tools.contracts import ToolContract, ToolExecutor, executor_for


def _contract(
    name: str,
    module_name: str,
    function_name: str,
    *,
    category: str,
    mutates_project: bool = True,
    requires_project: bool = True,
    long_running: bool = False,
    replayable: bool = False,
) -> ToolContract:
    return ToolContract(
        name=name,
        module_name=module_name,
        function_name=function_name,
        category=category,
        mutates_project=mutates_project,
        requires_project=requires_project,
        long_running=long_running,
        replayable=replayable,
    )


TOOL_CONTRACTS: dict[str, ToolContract] = {
    "get_video_info": _contract("get_video_info", "info", "execute", category="inspect", mutates_project=False),
    "trim_clip": _contract("trim_clip", "trim", "execute", category="edit", replayable=True),
    "merge_clips": _contract("merge_clips", "merge", "execute", category="edit", replayable=True),
    "adjust_speed": _contract("adjust_speed", "speed", "execute", category="edit", replayable=True),
    "add_transition": _contract("add_transition", "transitions", "execute", category="edit", replayable=True),
    "add_text_overlay": _contract("add_text_overlay", "overlay", "execute", category="edit", replayable=True),
    "extract_audio": _contract("extract_audio", "audio", "execute_extract", category="audio", mutates_project=False, replayable=True),
    "replace_audio": _contract("replace_audio", "audio", "execute_replace", category="audio", replayable=True),
    "mute_segment": _contract("mute_segment", "audio", "execute_mute", category="audio", replayable=True),
    "trim_silence": _contract("trim_silence", "silence", "execute", category="edit", long_running=True, replayable=True),
    "auto_color_grade": _contract("auto_color_grade", "color_grade", "execute", category="color", long_running=True, replayable=True),
    "burn_subtitles": _contract("burn_subtitles", "subtitles", "execute", category="text", long_running=True, replayable=True),
    "summarize_clip": _contract("summarize_clip", "summarize", "execute", category="analysis", mutates_project=False, long_running=True),
    "create_auto_shorts": _contract("create_auto_shorts", "auto_shorts", "execute", category="automation", long_running=True),
    "generate_video": _contract("generate_video", "video_generation", "execute", category="generation", long_running=True),
    "add_auto_broll": _contract("add_auto_broll", "pexels_broll", "execute", category="automation", long_running=True),
    "add_auto_visuals": _contract("add_auto_visuals", "auto_visuals", "execute", category="automation", long_running=True),
    "add_visual_asset": _contract("add_visual_asset", "visual_asset", "execute", category="visual", replayable=True),
    "add_auto_effects": _contract("add_auto_effects", "auto_effects", "execute", category="automation", long_running=True),
    "plan_encode": _contract("plan_encode", "encode", "execute_plan", category="encode", long_running=True),
    "run_pending_encode": _contract("run_pending_encode", "encode", "execute_run_pending", category="encode", long_running=True),
    "export_video": _contract("export_video", "export", "execute", category="export", long_running=True),
    "upscale_video": _contract("upscale_video", "upscale", "execute", category="enhance", long_running=True),
    "renderers_doctor": _contract(
        "renderers_doctor",
        "renderer_diagnostics",
        "execute",
        category="diagnostics",
        mutates_project=False,
        requires_project=False,
        long_running=True,
    ),
    "undo": _contract("undo", "undo", "execute_undo", category="history", replayable=False),
    "redo": _contract("redo", "undo", "execute_redo", category="history", replayable=False),
    "transcribe_video": _contract("transcribe_video", "transcript", "execute", category="transcript", long_running=True),
}


TOOL_EXECUTORS: dict[str, ToolExecutor] = {
    name: executor_for(contract)
    for name, contract in TOOL_CONTRACTS.items()
}
