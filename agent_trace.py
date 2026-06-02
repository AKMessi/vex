from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from rich import box
from rich.console import Group
from rich.table import Table
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


def trace_status_label(status: str) -> str:
    normalized = str(status or "").strip().lower()
    return {
        "running": "RUN",
        "success": "OK",
        "error": "ERR",
        "info": "INFO",
    }.get(normalized, normalized[:7].upper() or "INFO")


def trace_time_label(timestamp: str) -> str:
    try:
        parsed = datetime.fromisoformat(str(timestamp or ""))
    except ValueError:
        return "--:--:--"
    return parsed.strftime("%H:%M:%S")


def format_trace_duration(value: Any) -> str:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return ""
    if seconds < 0:
        return ""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    remaining = int(seconds % 60)
    return f"{minutes}m {remaining:02d}s"


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

    table = Table(
        box=box.SIMPLE_HEAVY,
        expand=True,
        show_edge=False,
        pad_edge=False,
    )
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("Status", width=7)
    table.add_column("Actor", style="cyan", width=9)
    table.add_column("Activity", ratio=1)
    table.add_column("Detail", ratio=2, style="dim")
    table.add_column("Time", justify="right", style="dim", width=8)
    table.add_column("Took", justify="right", style="dim", width=8)

    for event in events[-max_items:]:
        duration = format_trace_duration((event.metadata or {}).get("duration_sec"))
        table.add_row(
            str(event.step),
            Text(trace_status_label(event.status), style=f"bold {trace_status_style(event.status)}"),
            truncate_trace_text(event.kind, 9),
            Text(truncate_trace_text(event.title, 48), style="bold"),
            truncate_trace_text(event.detail, 86),
            trace_time_label(event.timestamp),
            duration,
        )
    return Group(table)
