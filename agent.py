from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from prompts import TOOL_SCHEMAS, build_system_prompt
from providers.base import BaseLLMProvider
from state import ProjectState
from tools import TOOL_EXECUTORS


class AgentLoopError(RuntimeError):
    pass


@dataclass
class AgentResponse:
    message: str
    tools_called: list[str]
    suggestions: list[str]
    success: bool


class VideoAgent:
    def __init__(self, state: ProjectState, provider: BaseLLMProvider) -> None:
        self.state = state
        self.provider = provider
        self.conversation: list[dict] = list(state.session_log or [])

    def _extract_suggestions(self, text: str) -> list[str]:
        return [line.strip() for line in text.splitlines() if line.strip().startswith("[SUGGESTION]:")]

    def run(
        self,
        user_message: str,
        stream_callback: Callable[[str], None] | None = None,
        tool_callback: Callable[[str, str, bool], None] | None = None,
    ) -> AgentResponse:
        self.conversation.append({"role": "user", "content": user_message})
        tools_called: list[str] = []
        final_text = ""
        success = True
        for _ in range(10):
            system_prompt = build_system_prompt(self.state)
            response = self.provider.chat(
                messages=self.conversation,
                tools=TOOL_SCHEMAS,
                system_prompt=system_prompt,
                stream_callback=None,
            )
            if response.tool_calls:
                self.conversation.append(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {"id": call.id, "name": call.name, "params": call.params}
                            for call in response.tool_calls
                        ],
                    }
                )
                for call in response.tool_calls:
                    tools_called.append(call.name)
                    if tool_callback:
                        tool_callback("start", call.name, True)
                    executor = TOOL_EXECUTORS.get(call.name)
                    if executor is None:
                        result = {
                            "success": False,
                            "message": f"Unknown tool: {call.name}",
                            "suggestion": None,
                            "updated_state": self.state,
                            "tool_name": call.name,
                        }
                        success = False
                    else:
                        try:
                            result = executor(call.params, self.state)
                        except Exception as exc:  # noqa: BLE001
                            result = {
                                "success": False,
                                "message": f"Unexpected executor error: {exc}",
                                "suggestion": None,
                                "updated_state": self.state,
                                "tool_name": call.name,
                            }
                            success = False
                    self.state = result["updated_state"]
                    if tool_callback:
                        tool_callback("finish", call.name, bool(result.get("success")))
                    self.conversation.append(
                        self.provider.format_tool_result(
                            tool_call_id=call.id,
                            result=result,
                            is_error=not bool(result.get("success")),
                        )
                    )
                continue
            final_text = response.text.strip()
            if stream_callback and final_text:
                stream_callback(final_text)
            suggestions = self._extract_suggestions(final_text)
            self.conversation.append({"role": "assistant", "content": final_text})
            self.state.session_log = self.conversation
            self.state.save()
            return AgentResponse(
                message=final_text,
                tools_called=tools_called,
                suggestions=suggestions,
                success=success,
            )
        raise AgentLoopError("Maximum agent loop iterations (10) exceeded.")
