from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import contextlib

from agent_trace import TraceEvent, TraceRecorder, truncate_trace_text
from prompts import TOOL_SCHEMAS, build_system_prompt
from providers.base import BaseLLMProvider, ProviderRequestError
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

    def _inject_tool_failures(self, text: str, tool_failures: list[dict[str, str]]) -> str:
        if not tool_failures:
            return text
        summaries: list[str] = []
        seen: set[tuple[str, str]] = set()
        for failure in tool_failures[-3:]:
            key = (failure["tool_name"], failure["message"])
            if key in seen:
                continue
            seen.add(key)
            summaries.append(f"{failure['tool_name']}: {failure['message']}")
        if not summaries:
            return text
        normalized = text.strip().lower()
        if all(summary.lower() in normalized for summary in summaries):
            return text
        failure_block = "Actual tool error" + ("s" if len(summaries) > 1 else "") + ":\n" + "\n".join(
            f"- {summary}" for summary in summaries
        )
        return f"{text.strip()}\n\n{failure_block}".strip()

    def _summarize_tool_params(self, params: dict) -> str:
        items: list[str] = []
        for key, value in params.items():
            rendered = value
            if isinstance(value, list):
                rendered = ", ".join(str(item) for item in value[:3])
                if len(value) > 3:
                    rendered = f"{rendered}, ..."
            items.append(f"{key}={rendered}")
        return truncate_trace_text(", ".join(items), 180) if items else "No parameters."

    def _summarize_tool_result(self, result: dict) -> str:
        message = truncate_trace_text(str(result.get("message", "")), 180)
        if message:
            return message
        return "Tool finished without a message."

    def _emit_trace(
        self,
        recorder: TraceRecorder,
        trace_callback: Callable[[TraceEvent], None] | None,
        *,
        kind: str,
        title: str,
        detail: str = "",
        status: str = "info",
        metadata: dict | None = None,
    ) -> TraceEvent:
        event = recorder.emit(
            kind=kind,
            title=title,
            detail=detail,
            status=status,
            metadata=metadata,
        )
        if trace_callback is not None:
            trace_callback(event)
        return event

    def _save_trace_artifact(
        self,
        recorder: TraceRecorder,
        *,
        success: bool,
        tools_called: list[str],
        final_message: str,
    ) -> None:
        artifact = recorder.to_artifact(
            success=success,
            tools_called=tools_called,
            final_message=final_message,
        )
        self.state.artifacts["latest_agent_trace"] = artifact
        history = list(self.state.artifacts.get("agent_trace_history") or [])
        history.append(artifact)
        self.state.artifacts["agent_trace_history"] = history[-20:]

    def _fail_due_to_provider_error(
        self,
        recorder: TraceRecorder,
        trace_callback: Callable[[TraceEvent], None] | None,
        *,
        detail: str,
        tools_called: list[str],
    ) -> None:
        self._emit_trace(
            recorder,
            trace_callback,
            kind="provider",
            title="Provider request failed",
            detail=truncate_trace_text(detail, 180),
            status="error",
        )
        self.state.session_log = self.conversation
        self._save_trace_artifact(
            recorder,
            success=False,
            tools_called=tools_called,
            final_message=detail,
        )
        with contextlib.suppress(Exception):
            self.state.save()
        raise AgentLoopError(detail)

    def run(
        self,
        user_message: str,
        stream_callback: Callable[[str], None] | None = None,
        tool_callback: Callable[[str, str, bool], None] | None = None,
        trace_callback: Callable[[TraceEvent], None] | None = None,
    ) -> AgentResponse:
        self.conversation.append({"role": "user", "content": user_message})
        tools_called: list[str] = []
        tool_failures: list[dict[str, str]] = []
        final_text = ""
        success = True
        recorder = TraceRecorder(
            instruction=user_message,
            provider=self.state.provider or self.provider.__class__.__name__,
            model=self.provider.model_name,
        )
        self._emit_trace(
            recorder,
            trace_callback,
            kind="turn",
            title="Received instruction",
            detail=truncate_trace_text(user_message, 180),
            status="info",
        )
        for iteration in range(10):
            system_prompt = build_system_prompt(self.state)
            self._emit_trace(
                recorder,
                trace_callback,
                kind="agent",
                title=f"Planning pass {iteration + 1}",
                detail="Reviewing project state and deciding the next step.",
                status="running",
            )
            try:
                response = self.provider.chat(
                    messages=self.conversation,
                    tools=TOOL_SCHEMAS,
                    system_prompt=system_prompt,
                    stream_callback=stream_callback,
                    event_callback=lambda payload: self._emit_trace(
                        recorder,
                        trace_callback,
                        kind=str(payload.get("kind", "provider")),
                        title=str(payload.get("title", "Provider update")),
                        detail=str(payload.get("detail", "")),
                        status=str(payload.get("status", "info")),
                        metadata=dict(payload.get("metadata") or {}),
                    ),
                )
            except ProviderRequestError as exc:
                self._fail_due_to_provider_error(
                    recorder,
                    trace_callback,
                    detail=str(exc),
                    tools_called=tools_called,
                )
            except Exception as exc:  # noqa: BLE001
                self._fail_due_to_provider_error(
                    recorder,
                    trace_callback,
                    detail=f"Provider request failed unexpectedly: {exc}",
                    tools_called=tools_called,
                )
            if response.tool_calls:
                self._emit_trace(
                    recorder,
                    trace_callback,
                    kind="agent",
                    title=f"Model proposed {len(response.tool_calls)} tool call(s)",
                    detail=", ".join(call.name for call in response.tool_calls[:4]),
                    status="info",
                )
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
                    self._emit_trace(
                        recorder,
                        trace_callback,
                        kind="tool",
                        title=f"Running {call.name}",
                        detail=self._summarize_tool_params(call.params),
                        status="running",
                    )
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
                    if not bool(result.get("success")):
                        success = False
                        tool_failures.append(
                            {
                                "tool_name": str(result.get("tool_name", call.name)),
                                "message": str(result.get("message", "Tool failed without an error message.")),
                            }
                        )
                    self._emit_trace(
                        recorder,
                        trace_callback,
                        kind="tool",
                        title=f"{call.name} {'completed' if bool(result.get('success')) else 'failed'}",
                        detail=self._summarize_tool_result(result),
                        status="success" if bool(result.get("success")) else "error",
                    )
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
            final_text = self._inject_tool_failures(response.text.strip(), tool_failures)
            self._emit_trace(
                recorder,
                trace_callback,
                kind="agent",
                title="Final response ready",
                detail=truncate_trace_text(final_text.splitlines()[0] if final_text else "Turn complete.", 180),
                status="success" if success else "error",
            )
            suggestions = self._extract_suggestions(final_text)
            self.conversation.append({"role": "assistant", "content": final_text})
            self.state.session_log = self.conversation
            self._save_trace_artifact(
                recorder,
                success=success,
                tools_called=tools_called,
                final_message=final_text,
            )
            self.state.save()
            return AgentResponse(
                message=final_text,
                tools_called=tools_called,
                suggestions=suggestions,
                success=success,
            )
        self._emit_trace(
            recorder,
            trace_callback,
            kind="agent",
            title="Agent loop limit reached",
            detail="The tool loop hit its maximum number of planning passes.",
            status="error",
        )
        self._save_trace_artifact(
            recorder,
            success=False,
            tools_called=tools_called,
            final_message=final_text,
        )
        self.state.save()
        raise AgentLoopError("Maximum agent loop iterations (10) exceeded.")
