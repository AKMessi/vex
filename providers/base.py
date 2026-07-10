from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


class ProviderRequestError(RuntimeError):
    pass


LOGGER = logging.getLogger(__name__)


def emit_event_safely(callback, payload: dict[str, Any]) -> None:
    if callback is None:
        return
    try:
        callback(payload)
    except Exception:  # noqa: BLE001
        LOGGER.warning("Provider event callback failed.", exc_info=True)


@dataclass
class ToolCall:
    id: str
    name: str
    params: dict


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    raw: object


class BaseLLMProvider(ABC):
    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        tools: list[dict],
        system_prompt: str,
        stream_callback=None,
        event_callback=None,
    ) -> LLMResponse:
        raise NotImplementedError

    @abstractmethod
    def format_tool_result(
        self,
        tool_call_id: str,
        result: dict[str, Any],
        is_error: bool = False,
    ) -> dict:
        raise NotImplementedError

    @property
    @abstractmethod
    def model_name(self) -> str:
        raise NotImplementedError
