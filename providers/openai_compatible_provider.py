from __future__ import annotations

import json
import re
import time
from typing import Any

import httpx

import config
from providers.base import BaseLLMProvider, LLMResponse, ProviderRequestError, ToolCall, emit_event_safely


class OpenAICompatibleProvider(BaseLLMProvider):
    def __init__(self, provider_name: str = "openai_compatible") -> None:
        self._provider_name = config.normalize_provider_name(provider_name)
        self._display_name = _display_name(self._provider_name)
        self._base_url = config.local_llm_base_url(self._provider_name).rstrip("/")
        self._api_key = config.OPENAI_COMPAT_API_KEY
        self._model_name = config.local_llm_model(self._provider_name)
        self._timeout_sec = float(config.OPENAI_COMPAT_TIMEOUT_SEC)
        self._max_tokens = int(config.OPENAI_COMPAT_MAX_TOKENS)
        self._temperature = float(config.OPENAI_COMPAT_TEMPERATURE)
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        self._client = httpx.Client(
            base_url=self._base_url,
            timeout=self._timeout_sec,
            headers=headers,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    def _chat_endpoint(self) -> str:
        return "/chat/completions"

    def _translate_tools(self, tools: list[dict]) -> list[dict]:
        translated: list[dict[str, Any]] = []
        for schema in tools:
            translated.append(
                {
                    "type": "function",
                    "function": {
                        "name": schema["name"],
                        "description": schema["description"],
                        "parameters": schema["parameters"],
                    },
                }
            )
        return translated

    def _translate_messages(self, messages: list[dict], system_prompt: str) -> list[dict]:
        native_messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        for message in messages:
            role = message["role"]
            if role in {"user", "assistant"} and "content" in message:
                native_messages.append({"role": role, "content": message["content"]})
            elif role == "assistant" and "tool_calls" in message:
                native_messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": call["id"],
                                "type": "function",
                                "function": {
                                    "name": call["name"],
                                    "arguments": json.dumps(call.get("params", {})),
                                },
                            }
                            for call in message["tool_calls"]
                        ],
                    }
                )
            elif role == "tool":
                native_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": message["tool_call_id"],
                        "content": message["content"],
                    }
                )
        return native_messages

    def _build_payload(
        self,
        messages: list[dict],
        tools: list[dict],
        system_prompt: str,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model_name,
            "messages": self._translate_messages(messages, system_prompt),
            "stream": stream,
            "temperature": self._temperature,
        }
        if self._max_tokens > 0:
            payload["max_tokens"] = self._max_tokens
        translated_tools = self._translate_tools(tools)
        if translated_tools:
            payload["tools"] = translated_tools
            payload["tool_choice"] = "auto"
        return payload

    def _emit_provider_event(
        self,
        event_callback,
        *,
        title: str,
        detail: str,
        status: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if event_callback is None:
            return
        payload: dict[str, Any] = {
            "kind": "provider",
            "title": title,
            "detail": detail,
            "status": status,
        }
        if metadata:
            payload["metadata"] = metadata
        emit_event_safely(event_callback, payload)

    def _is_retryable_error(self, exc: Exception) -> bool:
        if isinstance(exc, ProviderRequestError):
            return False
        if isinstance(exc, (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError, OSError)):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            status_code = exc.response.status_code
            return status_code in {408, 409, 425, 429} or status_code >= 500
        message = str(exc).lower()
        return any(
            hint in message
            for hint in (
                "temporar",
                "timeout",
                "timed out",
                "connection reset",
                "service unavailable",
                "overloaded",
                "rate limit",
                "retry",
            )
        )

    def _summarize_exception(self, exc: Exception) -> str:
        if isinstance(exc, httpx.HTTPStatusError):
            detail = _response_error_detail(exc.response)
            if detail:
                return f"{exc.response.status_code} {detail}"
            return str(exc.response.status_code)
        if isinstance(exc, httpx.ConnectError):
            return f"Could not connect to {self._base_url}. Is the local model server running?"
        message = " ".join(str(exc).split()).strip()
        return message or exc.__class__.__name__

    def _raise_provider_error(
        self,
        exc: Exception,
        *,
        event_callback,
        stage_label: str,
        attempts: int,
        retryable: bool,
    ) -> None:
        if retryable:
            detail = (
                f"{self._display_name} hit a temporary error while {stage_label.lower()} "
                f"after {attempts} attempt{'' if attempts == 1 else 's'}. Please retry the command."
            )
        else:
            summary = self._summarize_exception(exc)
            detail = f"{self._display_name} failed while {stage_label.lower()}: {summary}"
        self._emit_provider_event(
            event_callback,
            title=f"{self._display_name} request failed",
            detail=detail,
            status="error",
        )
        raise ProviderRequestError(detail) from exc

    def _with_retry(self, operation, *, event_callback, stage_label: str):
        max_attempts = max(1, int(config.LLM_REQUEST_MAX_RETRIES))
        for attempt in range(1, max_attempts + 1):
            try:
                return operation()
            except ProviderRequestError:
                raise
            except Exception as exc:  # noqa: BLE001
                retryable = self._is_retryable_error(exc)
                if retryable and attempt < max_attempts:
                    delay = float(config.LLM_RETRY_BASE_DELAY_SEC) * (2 ** (attempt - 1))
                    self._emit_provider_event(
                        event_callback,
                        title=f"{self._display_name} temporary error",
                        detail=(
                            f"{self._summarize_exception(exc)}. "
                            f"Retrying in {delay:.1f}s ({attempt + 1}/{max_attempts})."
                        ),
                        status="running",
                        metadata={"attempt": attempt + 1, "max_attempts": max_attempts},
                    )
                    time.sleep(delay)
                    continue
                self._raise_provider_error(
                    exc,
                    event_callback=event_callback,
                    stage_label=stage_label,
                    attempts=attempt,
                    retryable=retryable,
                )

    def chat(
        self,
        messages: list[dict],
        tools: list[dict],
        system_prompt: str,
        stream_callback=None,
        event_callback=None,
    ) -> LLMResponse:
        self._emit_provider_event(
            event_callback,
            title=f"Sending request to {self._display_name}",
            detail=f"Model: {self._model_name} @ {self._base_url}",
            status="running",
        )
        if stream_callback is not None:
            payload = self._build_payload(messages, tools, system_prompt, stream=True)
            return self._with_retry(
                lambda: self._run_stream(payload, tools, stream_callback, event_callback),
                event_callback=event_callback,
                stage_label=f"streaming the {self._display_name} response",
            )
        payload = self._build_payload(messages, tools, system_prompt, stream=False)
        return self._with_retry(
            lambda: self._run_once(payload, tools, event_callback),
            event_callback=event_callback,
            stage_label=f"requesting {self._display_name} output",
        )

    def _run_once(self, payload: dict[str, Any], tools: list[dict], event_callback) -> LLMResponse:
        response = self._client.post(self._chat_endpoint(), json=payload)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ProviderRequestError(
                f"{self._display_name} returned a non-object response."
            )
        choice = _first_choice(data)
        message = choice.get("message") or {}
        if not isinstance(message, dict):
            raise ProviderRequestError(
                f"{self._display_name} returned an invalid message payload."
            )
        content = message.get("content")
        if content is not None and not isinstance(content, str):
            raise ProviderRequestError(
                f"{self._display_name} returned non-text message content."
            )
        text = content or ""
        raw_tool_calls = message.get("tool_calls") or []
        if not isinstance(raw_tool_calls, list):
            raise ProviderRequestError(
                f"{self._display_name} returned invalid tool calls."
            )
        tool_calls = self._extract_tool_calls(raw_tool_calls, text, tools)
        if tool_calls:
            self._emit_provider_event(
                event_callback,
                title="Model requested tools",
                detail=", ".join(call.name for call in tool_calls[:4]),
                status="info",
            )
            text = "" if _text_is_json_tool_call(text) else text
        else:
            self._emit_provider_event(
                event_callback,
                title="Model returned text response",
                detail="Ready to finalize the turn.",
                status="success",
            )
        return LLMResponse(text=text, tool_calls=tool_calls, raw=data)

    def _run_stream(
        self,
        payload: dict[str, Any],
        tools: list[dict],
        stream_callback,
        event_callback,
    ) -> LLMResponse:
        text_chunks: list[str] = []
        raw_chunks: list[dict[str, Any]] = []
        tool_deltas: dict[int, dict[str, str]] = {}
        announced_text = False
        try:
            with self._client.stream("POST", self._chat_endpoint(), json=payload) as response:
                response.raise_for_status()
                for raw_line in response.iter_lines():
                    data_line = _decode_sse_line(raw_line)
                    if data_line is None:
                        continue
                    if data_line == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_line)
                    except json.JSONDecodeError as exc:
                        raise ProviderRequestError(
                            f"{self._display_name} returned a malformed streaming response."
                        ) from exc
                    if not isinstance(chunk, dict):
                        raise ProviderRequestError(
                            f"{self._display_name} returned an invalid streaming payload."
                        )
                    raw_chunks.append(chunk)
                    choice = _first_choice(chunk)
                    delta = choice.get("delta") or {}
                    if not isinstance(delta, dict):
                        raise ProviderRequestError(
                            f"{self._display_name} returned an invalid streaming delta."
                        )
                    content = delta.get("content")
                    if content:
                        content_text = str(content)
                        if not announced_text:
                            self._emit_provider_event(
                                event_callback,
                                title="Streaming assistant response",
                                detail="Receiving model output.",
                                status="running",
                            )
                            announced_text = True
                        text_chunks.append(content_text)
                        stream_callback(content_text)
                    raw_tool_deltas = delta.get("tool_calls") or []
                    if not isinstance(raw_tool_deltas, list):
                        raise ProviderRequestError(
                            f"{self._display_name} returned invalid streaming tool calls."
                        )
                    _accumulate_tool_deltas(tool_deltas, raw_tool_deltas)
        except ProviderRequestError:
            raise
        except Exception as exc:  # noqa: BLE001
            if text_chunks:
                raise ProviderRequestError(
                    f"{self._display_name} interrupted the response stream after partial output. "
                    "Please retry the command."
                ) from exc
            raise
        text = "".join(text_chunks)
        tool_calls = self._tool_calls_from_deltas(tool_deltas, text, tools)
        if tool_calls:
            self._emit_provider_event(
                event_callback,
                title="Model requested tools",
                detail=", ".join(call.name for call in tool_calls[:4]),
                status="info",
            )
            text = "" if _text_is_json_tool_call(text) else text
        else:
            self._emit_provider_event(
                event_callback,
                title="Model finished response",
                detail="No tool calls were returned.",
                status="success",
            )
        return LLMResponse(text=text, tool_calls=tool_calls, raw=raw_chunks)

    def _extract_tool_calls(self, raw_tool_calls: list[Any], text: str, tools: list[dict]) -> list[ToolCall]:
        native = _parse_native_tool_calls(raw_tool_calls)
        if native:
            return native
        return _parse_text_tool_calls(text, tools)

    def _tool_calls_from_deltas(
        self,
        tool_deltas: dict[int, dict[str, str]],
        text: str,
        tools: list[dict],
    ) -> list[ToolCall]:
        raw_tool_calls: list[dict[str, Any]] = []
        for index in sorted(tool_deltas):
            item = tool_deltas[index]
            if not item.get("name"):
                continue
            raw_tool_calls.append(
                {
                    "id": item.get("id") or f"local_tool_{index}",
                    "function": {
                        "name": item["name"],
                        "arguments": item.get("arguments") or "{}",
                    },
                }
            )
        return self._extract_tool_calls(raw_tool_calls, text, tools)

    def format_tool_result(
        self,
        tool_call_id: str,
        result: dict[str, Any],
        is_error: bool = False,
    ) -> dict:
        strip_keys = {"updated_state", "suggestion"}
        payload = {
            "tool_call_id": tool_call_id,
            "tool_name": result.get("tool_name", "tool_result"),
            "is_error": is_error,
            **{key: value for key, value in result.items() if key not in strip_keys},
        }
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps(payload),
        }


def _display_name(provider_name: str) -> str:
    return {
        "openai_compatible": "OpenAI-compatible local provider",
        "ollama": "Ollama",
        "lmstudio": "LM Studio",
        "llama_cpp": "llama.cpp",
    }.get(provider_name, provider_name)


def _first_choice(payload: dict[str, Any]) -> dict[str, Any]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ProviderRequestError("Local provider returned no choices.")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ProviderRequestError("Local provider returned an invalid choice payload.")
    return choice


def _response_error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip()
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if message:
                return str(message)
        if isinstance(error, str):
            return error
        message = payload.get("message")
        if message:
            return str(message)
    return response.text.strip()


def _decode_tool_arguments(raw_arguments: Any, tool_name: str) -> dict:
    if raw_arguments is None or raw_arguments == "":
        return {}
    if isinstance(raw_arguments, dict):
        return raw_arguments
    if not isinstance(raw_arguments, str):
        raise ProviderRequestError(f"Local provider returned invalid arguments for tool {tool_name}.")
    try:
        parsed = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        raise ProviderRequestError(
            f"Local provider returned invalid JSON arguments for tool {tool_name}: {raw_arguments[:160]}"
        ) from exc
    if not isinstance(parsed, dict):
        raise ProviderRequestError(f"Local provider returned non-object arguments for tool {tool_name}.")
    return parsed


def _parse_native_tool_calls(raw_tool_calls: list[Any]) -> list[ToolCall]:
    parsed: list[ToolCall] = []
    for index, raw_call in enumerate(raw_tool_calls, start=1):
        if not isinstance(raw_call, dict):
            continue
        function = raw_call.get("function") or {}
        if not isinstance(function, dict):
            function = {}
        name = str(function.get("name") or raw_call.get("name") or "").strip()
        if not name:
            continue
        call_id = str(raw_call.get("id") or f"local_tool_{index}")
        arguments = function.get("arguments", raw_call.get("arguments", {}))
        parsed.append(ToolCall(id=call_id, name=name, params=_decode_tool_arguments(arguments, name)))
    return parsed


def _allowed_tool_names(tools: list[dict]) -> set[str]:
    return {str(tool.get("name") or "").strip() for tool in tools if tool.get("name")}


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else stripped


def _text_is_json_tool_call(text: str) -> bool:
    stripped = _strip_json_fence(text)
    return stripped.startswith("{") and stripped.endswith("}")


def _parse_text_tool_calls(text: str, tools: list[dict]) -> list[ToolCall]:
    stripped = _strip_json_fence(text)
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return []
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict):
        return []
    allowed = _allowed_tool_names(tools)
    raw_calls = payload.get("tool_calls")
    if isinstance(raw_calls, list):
        parsed = _parse_native_tool_calls(raw_calls)
        return [call for call in parsed if call.name in allowed]
    name = str(payload.get("tool") or payload.get("name") or "").strip()
    params = payload.get("params") or payload.get("arguments") or {}
    if name and name in allowed and isinstance(params, dict):
        return [ToolCall(id="local_text_tool_1", name=name, params=params)]
    return []


def _decode_sse_line(raw_line: str) -> str | None:
    line = raw_line.strip()
    if not line or line.startswith(":"):
        return None
    if line.startswith("data:"):
        return line[5:].strip()
    if line.startswith("{") and line.endswith("}"):
        return line
    return None


def _accumulate_tool_deltas(tool_deltas: dict[int, dict[str, str]], raw_deltas: list[Any]) -> None:
    for raw_delta in raw_deltas:
        if not isinstance(raw_delta, dict):
            continue
        try:
            index = int(raw_delta.get("index") or 0)
        except (TypeError, ValueError) as exc:
            raise ProviderRequestError(
                "Local provider returned a tool call with an invalid stream index."
            ) from exc
        if index < 0 or index > 1024:
            raise ProviderRequestError(
                "Local provider returned a tool call stream index outside the supported range."
            )
        current = tool_deltas.setdefault(index, {"id": "", "name": "", "arguments": ""})
        if raw_delta.get("id"):
            current["id"] = str(raw_delta["id"])
        function = raw_delta.get("function") or {}
        if not isinstance(function, dict):
            continue
        if function.get("name"):
            current["name"] += str(function["name"])
        if function.get("arguments"):
            current["arguments"] += str(function["arguments"])
