from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class ProjectState:
    project_id: str
    project_name: str
    created_at: str
    updated_at: str
    source_files: list[str]
    working_file: str
    working_dir: str
    output_dir: str
    timeline: list[dict[str, Any]] = field(default_factory=list)
    redo_stack: list[dict[str, Any]] = field(default_factory=list)
    session_log: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    provider: str = "gemini"
    model: str = ""

    @property
    def state_path(self) -> Path:
        return Path(self.working_dir) / f"{self.project_id}.json"

    def save(self) -> None:
        self.updated_at = utc_now_iso()
        Path(self.working_dir).mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProjectState":
        valid_fields = {field_.name for field_ in fields(cls)}
        filtered = {key: value for key, value in payload.items() if key in valid_fields}
        return cls(**filtered)

    @classmethod
    def load(cls, project_id: str) -> "ProjectState":
        base = Path(config.AGENT_PROJECTS_DIR)
        candidates = list(base.glob(f"*/{project_id}.json"))
        if not candidates:
            candidates = list(base.glob(f"*/{project_id}*.json"))
        if not candidates:
            raise FileNotFoundError(f"No project found for id {project_id!r}.")
        if len(candidates) > 1:
            def sort_key(path: Path) -> str:
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    return ""
                return payload.get("updated_at", "")

            candidates.sort(key=sort_key, reverse=True)
            warnings.warn(
                f"Multiple projects matched partial id {project_id!r}; using the most recently updated match.",
                stacklevel=2,
            )
        payload = json.loads(candidates[0].read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    @classmethod
    def list_projects(cls) -> list[dict[str, Any]]:
        base = Path(config.AGENT_PROJECTS_DIR)
        base.mkdir(parents=True, exist_ok=True)
        items: list[dict[str, Any]] = []
        for path in base.glob("*/*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            items.append(
                {
                    "project_id": payload.get("project_id", ""),
                    "project_name": payload.get("project_name", ""),
                    "created_at": payload.get("created_at", ""),
                    "updated_at": payload.get("updated_at", ""),
                    "source_file": (payload.get("source_files") or [""])[0],
                    "timeline_ops": len(payload.get("timeline") or []),
                    "working_dir": payload.get("working_dir", ""),
                }
            )
        items.sort(key=lambda item: item["updated_at"], reverse=True)
        return items

    def apply_operation(self, op: dict[str, Any]) -> None:
        self.timeline.append(op)
        self.redo_stack.clear()
        self.updated_at = utc_now_iso()
        self.save()

    def undo(self) -> dict[str, Any] | None:
        if not self.timeline:
            return None
        op = self.timeline.pop()
        self.redo_stack.append(op)
        self.updated_at = utc_now_iso()
        self.save()
        return op

    def redo(self) -> dict[str, Any] | None:
        if not self.redo_stack:
            return None
        op = self.redo_stack.pop()
        self.timeline.append(op)
        self.updated_at = utc_now_iso()
        self.save()
        return op

    def get_summary(self) -> str:
        meta = self.metadata or {}
        lines = [
            f"Project: {self.project_name}",
            f"Project ID: {self.project_id}",
            f"Created: {self.created_at}",
            f"Updated: {self.updated_at}",
            f"Provider: {self.provider} / {self.model}",
            f"Working file: {self.working_file}",
            f"Output dir: {self.output_dir}",
            f"Source files: {', '.join(self.source_files) if self.source_files else 'none'}",
            (
                "Metadata: "
                f"{meta.get('duration_sec', 'unknown')}s, "
                f"{meta.get('width', '?')}x{meta.get('height', '?')}, "
                f"{meta.get('fps', '?')}fps"
            ),
            f"Timeline operations: {len(self.timeline)}",
            f"Redo available: {len(self.redo_stack)}",
        ]
        source_url = str((self.artifacts or {}).get("source_url") or "").strip()
        if source_url:
            lines.append(f"Source URL: {source_url}")
        latest_auto_shorts = (self.artifacts or {}).get("latest_auto_shorts")
        if latest_auto_shorts:
            lines.append(
                "Latest auto shorts: "
                f"{latest_auto_shorts.get('count', 0)} clips @ {latest_auto_shorts.get('manifest_path', 'unknown')}"
            )
        if self.timeline:
            lines.append("Timeline:")
            for index, op in enumerate(self.timeline, start=1):
                lines.append(
                    f"  {index}. {op['op']} - {op.get('description', '')} @ {op.get('timestamp', '')}"
                )
        return "\n".join(lines)
