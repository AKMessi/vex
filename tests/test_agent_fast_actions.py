from __future__ import annotations

from pathlib import Path

from agent import VideoAgent
from agent_fast_actions import detect_fast_action
from providers.base import BaseLLMProvider, LLMResponse, ToolCall
from state import ProjectState, utc_now_iso


class _FailingProvider(BaseLLMProvider):
    @property
    def model_name(self) -> str:
        return "test-model"

    def chat(self, messages, tools, system_prompt, stream_callback=None, event_callback=None):  # noqa: ANN001
        raise AssertionError("Provider should not be called for deterministic fast actions.")

    def format_tool_result(self, tool_call_id: str, result: dict, is_error: bool = False) -> dict:
        return {"role": "tool", "tool_call_id": tool_call_id, "content": str(result)}


class _OneToolProvider(BaseLLMProvider):
    def __init__(self) -> None:
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "test-model"

    def chat(self, messages, tools, system_prompt, stream_callback=None, event_callback=None):  # noqa: ANN001
        self.calls += 1
        if self.calls > 1:
            raise AssertionError("Single terminal tool calls should not need a second provider pass.")
        return LLMResponse(
            text="",
            tool_calls=[ToolCall(id="tool-1", name="trim_clip", params={"start": "0", "end": "30"})],
            raw=None,
        )

    def format_tool_result(self, tool_call_id: str, result: dict, is_error: bool = False) -> dict:
        return {"role": "tool", "tool_call_id": tool_call_id, "content": str(result)}


class _ChainedIntentProvider(BaseLLMProvider):
    def __init__(self) -> None:
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "test-model"

    def chat(self, messages, tools, system_prompt, stream_callback=None, event_callback=None):  # noqa: ANN001
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                text="",
                tool_calls=[ToolCall(id="tool-1", name="trim_clip", params={"start": "0", "end": "30"})],
                raw=None,
            )
        return LLMResponse(text="Ready for the next chained step.", tool_calls=[], raw=None)

    def format_tool_result(self, tool_call_id: str, result: dict, is_error: bool = False) -> dict:
        return {"role": "tool", "tool_call_id": tool_call_id, "content": str(result)}


def test_fast_trim_range_parses_without_file_path_noise() -> None:
    action = detect_fast_action(r'trim from 0:30 to 1:45 of "D:\videos\clip.mp4"')

    assert action is not None
    assert action.tool_name == "trim_clip"
    assert action.params == {"start": "30", "end": "105"}


def test_fast_trim_keeps_first_duration() -> None:
    action = detect_fast_action("trim the first 30 seconds")

    assert action is not None
    assert action.params == {"start": "0", "end": "30"}


def test_fast_trim_removes_first_duration_when_command_says_remove() -> None:
    action = detect_fast_action("remove the first 15 seconds")

    assert action is not None
    assert action.params == {"start": "15"}


def test_fast_trim_uses_metadata_for_last_duration() -> None:
    action = detect_fast_action("keep the last 20 seconds", {"duration_sec": 95.0})

    assert action is not None
    assert action.params == {"start": "75"}


def test_fast_trim_does_not_steal_semantic_tools() -> None:
    assert detect_fast_action("trim the silent pauses") is None
    assert detect_fast_action("make shorts from the first 30 seconds") is None
    assert detect_fast_action("trim this down to a 30 second highlight") is None


def test_agent_executes_fast_trim_without_provider(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    import agent as agent_module

    calls: list[dict] = []

    def fake_trim(params: dict, state: ProjectState) -> dict:
        calls.append(dict(params))
        return {
            "success": True,
            "message": f"Trimmed from {params['start']} to {params.get('end', 'end')}.",
            "suggestion": None,
            "updated_state": state,
            "tool_name": "trim_clip",
        }

    monkeypatch.setitem(agent_module.TOOL_EXECUTORS, "trim_clip", fake_trim)
    state = _state(tmp_path)
    response = VideoAgent(state, _FailingProvider()).run("trim the first 30 seconds")

    assert calls == [{"start": "0", "end": "30"}]
    assert response.success
    assert response.tools_called == ["trim_clip"]
    assert response.message == "Trimmed from 0 to 30."


def test_agent_executes_chained_plan_without_provider(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    import agent as agent_module

    calls: list[tuple[str, dict]] = []

    def fake_trim(params: dict, state: ProjectState) -> dict:
        calls.append(("trim_clip", dict(params)))
        return {
            "success": True,
            "message": "Trimmed from 0 to 30.",
            "suggestion": None,
            "updated_state": state,
            "tool_name": "trim_clip",
        }

    def fake_export(params: dict, state: ProjectState) -> dict:
        calls.append(("export_video", dict(params)))
        return {
            "success": True,
            "message": "Exported video to out.mp4.",
            "suggestion": None,
            "updated_state": state,
            "tool_name": "export_video",
        }

    monkeypatch.setitem(agent_module.TOOL_EXECUTORS, "trim_clip", fake_trim)
    monkeypatch.setitem(agent_module.TOOL_EXECUTORS, "export_video", fake_export)
    response = VideoAgent(_state(tmp_path), _FailingProvider()).run(
        "trim the first 30 seconds and export it for instagram"
    )

    assert calls == [
        ("trim_clip", {"start": "0", "end": "30"}),
        ("export_video", {"preset_name": "instagram_reels"}),
    ]
    assert response.success
    assert response.tools_called == ["trim_clip", "export_video"]
    assert "Trimmed from 0 to 30." in response.message
    assert "Exported video to out.mp4." in response.message


def test_agent_returns_after_single_terminal_tool_call(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    import agent as agent_module

    def fake_trim(params: dict, state: ProjectState) -> dict:
        return {
            "success": True,
            "message": "Trimmed from 0 to 30.",
            "suggestion": None,
            "updated_state": state,
            "tool_name": "trim_clip",
        }

    monkeypatch.setitem(agent_module.TOOL_EXECUTORS, "trim_clip", fake_trim)
    provider = _OneToolProvider()
    response = VideoAgent(_state(tmp_path), provider).run("make the requested edit")

    assert provider.calls == 1
    assert response.success
    assert response.tools_called == ["trim_clip"]
    assert response.message == "Trimmed from 0 to 30."


def test_agent_does_not_shortcut_possible_chained_instruction(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    import agent as agent_module

    def fake_trim(params: dict, state: ProjectState) -> dict:
        return {
            "success": True,
            "message": "Trimmed from 0 to 30.",
            "suggestion": None,
            "updated_state": state,
            "tool_name": "trim_clip",
        }

    monkeypatch.setitem(agent_module.TOOL_EXECUTORS, "trim_clip", fake_trim)
    provider = _ChainedIntentProvider()
    response = VideoAgent(_state(tmp_path), provider).run("trim the first 30 seconds and export it")

    assert provider.calls == 2
    assert response.success
    assert response.tools_called == ["trim_clip"]
    assert response.message == "Ready for the next chained step."


def test_agent_yes_runs_pending_encode_without_provider(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    import agent as agent_module

    calls: list[dict] = []
    state = _state(tmp_path)
    state.artifacts["pending_encode"] = {"plan_id": "encode-plan-1"}

    def fake_run_pending(params: dict, state: ProjectState) -> dict:
        calls.append(dict(params))
        return {
            "success": True,
            "message": "Encoded video to out.mp4.",
            "suggestion": None,
            "updated_state": state,
            "tool_name": "run_pending_encode",
        }

    monkeypatch.setitem(agent_module.TOOL_EXECUTORS, "run_pending_encode", fake_run_pending)
    response = VideoAgent(state, _FailingProvider()).run("yes")

    assert calls == [{"plan_id": "encode-plan-1"}]
    assert response.success
    assert response.tools_called == ["run_pending_encode"]
    assert response.message == "Encoded video to out.mp4."


def test_agent_asks_for_auto_visual_renderer_choice_without_provider(tmp_path: Path) -> None:
    state = _state(tmp_path)

    response = VideoAgent(state, _FailingProvider()).run("add auto visuals")

    assert response.success
    assert response.tools_called == []
    assert "hyperframes" in response.message
    assert "manim" in response.message
    assert "remotion" in response.message
    assert "both" in response.message
    pending = state.artifacts["pending_auto_visuals_renderer_choice"]
    assert pending["params"] == {"force_fullscreen": True}


def test_agent_runs_pending_auto_visual_renderer_choice_without_provider(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    import agent as agent_module

    calls: list[dict] = []
    state = _state(tmp_path)
    state.artifacts["pending_auto_visuals_renderer_choice"] = {
        "params": {"force_fullscreen": True, "max_visuals": 8},
    }

    def fake_auto_visuals(params: dict, state: ProjectState) -> dict:
        calls.append(dict(params))
        return {
            "success": True,
            "message": "Added generated visuals.",
            "suggestion": None,
            "updated_state": state,
            "tool_name": "add_auto_visuals",
        }

    monkeypatch.setitem(agent_module.TOOL_EXECUTORS, "add_auto_visuals", fake_auto_visuals)

    response = VideoAgent(state, _FailingProvider()).run("hyperframes")

    assert calls == [{"force_fullscreen": True, "max_visuals": 8, "renderer": "hyperframes"}]
    assert "pending_auto_visuals_renderer_choice" not in state.artifacts
    assert response.success
    assert response.tools_called == ["add_auto_visuals"]
    assert response.message == "Added generated visuals."


def test_agent_runs_pending_auto_visual_remotion_choice_without_provider(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    import agent as agent_module

    calls: list[dict] = []
    state = _state(tmp_path)
    state.artifacts["pending_auto_visuals_renderer_choice"] = {
        "params": {"force_fullscreen": True, "max_visuals": 3},
    }

    def fake_auto_visuals(params: dict, state: ProjectState) -> dict:
        calls.append(dict(params))
        return {
            "success": True,
            "message": "Added Remotion visuals.",
            "suggestion": None,
            "updated_state": state,
            "tool_name": "add_auto_visuals",
        }

    monkeypatch.setitem(agent_module.TOOL_EXECUTORS, "add_auto_visuals", fake_auto_visuals)

    response = VideoAgent(state, _FailingProvider()).run("3")

    assert calls == [{"force_fullscreen": True, "max_visuals": 3, "renderer": "remotion"}]
    assert "pending_auto_visuals_renderer_choice" not in state.artifacts
    assert response.success
    assert response.tools_called == ["add_auto_visuals"]
    assert response.message == "Added Remotion visuals."


def _state(tmp_path: Path) -> ProjectState:
    now = utc_now_iso()
    return ProjectState(
        project_id="test-project",
        project_name="Test Project",
        created_at=now,
        updated_at=now,
        source_files=[str(tmp_path / "source.mp4")],
        working_file=str(tmp_path / "working.mp4"),
        working_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        metadata={"duration_sec": 120.0, "width": 1920, "height": 1080, "fps": 30.0},
        provider="test",
        model="test-model",
    )
