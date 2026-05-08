from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolStep:
    tool: str
    params: dict[str, Any] = field(default_factory=dict)
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EditPlan:
    steps: list[ToolStep]
    source: str = "deterministic_intent"
    confidence: float = 1.0
    reason: str = ""
    requires_llm: bool = False
    can_run_async: bool = False
    final_response_mode: str = "tool_summary"

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [step.to_dict() for step in self.steps],
            "source": self.source,
            "confidence": self.confidence,
            "reason": self.reason,
            "requires_llm": self.requires_llm,
            "can_run_async": self.can_run_async,
            "final_response_mode": self.final_response_mode,
        }

    @property
    def tool_names(self) -> list[str]:
        return [step.tool for step in self.steps]
