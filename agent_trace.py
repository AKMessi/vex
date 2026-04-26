from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from rich.console import Group
from rich.text import Text


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def truncate_trace_text(text: str, limit: int = 140) -> str:
    collapsed = " ".join(str(text or "").split()).strip()
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3].rstrip() + "..."


def trace_status_style(status: str) -> str:
    return {
        "running": "yellow",
        "success": "green",
        "error": "red",
        "info": "cyan",
    }.get(str(status or "").strip().lower(), "white")


@dataclass
class TraceEvent:
    step: int
    kind: str
    title: str
    detail: str = ""
    status: str = "info"
    timestamp: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TraceEvent":
        return cls(
            step=int(payload.get("step", 0)),
            kind=str(payload.get("kind", "agent")),
            title=str(payload.get("title", "")),
            detail=str(payload.get("detail", "")),
            status=str(payload.get("status", "info")),
            timestamp=str(payload.get("timestamp", utc_now_iso())),
            metadata=dict(payload.get("metadata") or {}),
        )


class TraceRecorder:
    def __init__(self, instruction: str, provider: str, model: str) -> None:
        self.instruction = truncate_trace_text(instruction, 220)
        self.provider = provider
        self.model = model
        self.events: list[TraceEvent] = []

    def emit(
        self,
        *,
        kind: str,
        title: str,
        detail: str = "",
        status: str = "info",
        metadata: dict[str, Any] | None = None,
    ) -> TraceEvent:
        event = TraceEvent(
            step=len(self.events) + 1,
            kind=kind,
            title=title,
            detail=truncate_trace_text(detail, 220) if detail else "",
            status=status,
            metadata=dict(metadata or {}),
        )
        self.events.append(event)
        return event

    def to_artifact(
        self,
        *,
        success: bool,
        tools_called: list[str],
        final_message: str,
    ) -> dict[str, Any]:
        return {
            "created_at": utc_now_iso(),
            "instruction": self.instruction,
            "provider": self.provider,
            "model": self.model,
            "success": success,
            "tools_called": list(tools_called),
            "final_message_preview": truncate_trace_text(final_message, 200),
            "events": [event.to_dict() for event in self.events],
        }


def render_trace_table(events: list[TraceEvent], max_items: int = 10):
    if not events:
        return Text("No trace steps yet.", style="dim")

    rows: list[Text] = []
    for event in events[-max_items:]:
        message = Text()
        message.append(f"{event.step:>2}. ", style="dim")
        message.append(f"{event.status.upper():<7}", style=f"bold {trace_status_style(event.status)}")
        message.append(f" {event.title}", style="bold")
        if event.detail:
            message.append(f" - {event.detail}", style="dim")
        rows.append(message)
    return Group(*rows)
