from __future__ import annotations

import json
import os
import re
import tempfile
import warnings
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config

PROJECT_LOOKUP_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def merge_time_ranges(
    ranges: list[tuple[float, float]],
    *,
    gap_sec: float = 0.0,
) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start_sec, end_sec in sorted(ranges, key=lambda item: item[0]):
        start_value = float(start_sec)
        end_value = float(end_sec)
        if end_value <= start_value:
            continue
        if not merged or start_value > merged[-1][1] + gap_sec:
            merged.append([start_value, end_value])
            continue
        merged[-1][1] = max(merged[-1][1], end_value)
    return [(start_sec, end_sec) for start_sec, end_sec in merged]


def clip_time_range_to_available_window(
    start_sec: float,
    end_sec: float,
    blocked_ranges: list[tuple[float, float]],
    *,
    min_duration_sec: float = 0.0,
) -> tuple[float, float] | None:
    epsilon = 1e-6
    start_value = float(start_sec)
    end_value = float(end_sec)
    if end_value <= start_value + epsilon:
        return None
    available: list[tuple[float, float]] = [(start_value, end_value)]
    for blocked_start, blocked_end in merge_time_ranges(blocked_ranges):
        next_available: list[tuple[float, float]] = []
        for candidate_start, candidate_end in available:
            if blocked_end <= candidate_start or blocked_start >= candidate_end:
                next_available.append((candidate_start, candidate_end))
                continue
            if blocked_start > candidate_start:
                next_available.append((candidate_start, min(blocked_start, candidate_end)))
            if blocked_end < candidate_end:
                next_available.append((max(blocked_end, candidate_start), candidate_end))
        available = next_available
        if not available:
            return None
    viable = [
        (candidate_start, candidate_end)
        for candidate_start, candidate_end in available
        if candidate_end - candidate_start + epsilon >= min_duration_sec
    ]
    if not viable:
        return None
    candidate_start, candidate_end = max(viable, key=lambda item: (item[1] - item[0], -item[0]))
    return round(candidate_start, 3), round(candidate_end, 3)


def restrict_timed_items_to_available_ranges(
    items: list[dict[str, Any]],
    blocked_ranges: list[tuple[float, float]],
    *,
    min_duration_sec: float = 0.0,
    start_key: str = "start",
    end_key: str = "end",
) -> list[dict[str, Any]]:
    if not blocked_ranges:
        return list(items)
    restricted: list[dict[str, Any]] = []
    for item in items:
        try:
            start_sec = float(item.get(start_key, 0.0))
            end_sec = float(item.get(end_key, start_sec))
        except (TypeError, ValueError):
            continue
        clipped = clip_time_range_to_available_window(
            start_sec,
            end_sec,
            blocked_ranges,
            min_duration_sec=min_duration_sec,
        )
        if clipped is None:
            continue
        adjusted = dict(item)
        adjusted[start_key], adjusted[end_key] = clipped
        restricted.append(adjusted)
    return restricted


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
        target_dir = Path(self.working_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(asdict(self), indent=2)
        temp_path: Path | None = None
        temp_prefix = re.sub(r"[^A-Za-z0-9_-]", "_", self.project_id or "project")
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=target_dir,
                prefix=f".{temp_prefix}.",
                suffix=".tmp",
                delete=False,
            ) as temp_file:
                temp_path = Path(temp_file.name)
                temp_file.write(payload)
                temp_file.flush()
                os.fsync(temp_file.fileno())
            os.replace(temp_path, self.state_path)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink(missing_ok=True)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProjectState":
        valid_fields = {field_.name for field_ in fields(cls)}
        filtered = {key: value for key, value in payload.items() if key in valid_fields}
        return cls(**filtered)

    @classmethod
    def _coerce_project_payload(cls, payload: object) -> dict[str, Any] | None:
        if not isinstance(payload, dict):
            return None
        required = {
            "project_id",
            "project_name",
            "created_at",
            "updated_at",
            "source_files",
            "working_file",
            "working_dir",
            "output_dir",
        }
        if not required.issubset(payload.keys()):
            return None
        if not isinstance(payload.get("project_id"), str) or not payload.get("project_id"):
            return None
        if not isinstance(payload.get("project_name"), str):
            return None
        if not isinstance(payload.get("source_files"), list):
            return None
        if not isinstance(payload.get("working_file"), str):
            return None
        if not isinstance(payload.get("working_dir"), str):
            return None
        if not isinstance(payload.get("output_dir"), str):
            return None
        return payload

    @classmethod
    def _load_project_payload(cls, path: Path) -> dict[str, Any] | None:
        try:
            raw_payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        payload = cls._coerce_project_payload(raw_payload)
        if payload is None:
            return None
        expected_name = f"{payload['project_id']}.json"
        if path.name != expected_name:
            return None
        return payload

    @classmethod
    def _normalize_project_lookup(cls, project_id: str) -> str:
        lookup = str(project_id or "").strip()
        if not PROJECT_LOOKUP_RE.fullmatch(lookup):
            raise FileNotFoundError(f"Invalid project id {project_id!r}.")
        return lookup

    @classmethod
    def load(cls, project_id: str) -> "ProjectState":
        base = Path(config.AGENT_PROJECTS_DIR)
        lookup = cls._normalize_project_lookup(project_id)
        matches: list[tuple[Path, dict[str, Any], bool]] = []
        for path in base.glob("*/*.json"):
            payload = cls._load_project_payload(path)
            if payload is None:
                continue
            payload_project_id = str(payload.get("project_id") or "")
            exact = payload_project_id == lookup
            if exact or payload_project_id.startswith(lookup):
                matches.append((path, payload, exact))
        exact_matches = [match for match in matches if match[2]]
        if exact_matches:
            matches = exact_matches
        if not matches:
            raise FileNotFoundError(f"No project found for id {project_id!r}.")
        if len(matches) > 1:
            matches.sort(key=lambda match: str(match[1].get("updated_at", "")), reverse=True)
            warnings.warn(
                f"Multiple projects matched partial id {project_id!r}; using the most recently updated match.",
                stacklevel=2,
            )
        payload = matches[0][1]
        return cls.from_dict(payload)

    @classmethod
    def list_projects(cls) -> list[dict[str, Any]]:
        base = Path(config.AGENT_PROJECTS_DIR)
        base.mkdir(parents=True, exist_ok=True)
        items: list[dict[str, Any]] = []
        for path in base.glob("*/*.json"):
            payload = cls._load_project_payload(path)
            if payload is None:
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
        latest_auto_broll = (self.artifacts or {}).get("latest_auto_broll")
        if latest_auto_broll:
            lines.append(
                "Latest auto b-roll: "
                f"{latest_auto_broll.get('count', 0)} inserts @ {latest_auto_broll.get('manifest_path', 'unknown')}"
            )
        latest_transcript = (self.artifacts or {}).get("latest_transcript")
        if latest_transcript:
            lines.append(
                "Latest transcript: "
                f"{latest_transcript.get('segment_count', 0)} segments / "
                f"{latest_transcript.get('word_count', 0)} words @ {latest_transcript.get('srt_path', 'unknown')}"
            )
        latest_auto_visuals = (self.artifacts or {}).get("latest_auto_visuals")
        if latest_auto_visuals:
            lines.append(
                "Latest auto visuals: "
                f"{latest_auto_visuals.get('count', 0)} inserts "
                f"({latest_auto_visuals.get('renderer', 'auto')} / {latest_auto_visuals.get('style_pack', 'auto')}) "
                f"@ {latest_auto_visuals.get('manifest_path', 'unknown')}"
            )
        latest_auto_color_grade = (self.artifacts or {}).get("latest_auto_color_grade")
        if latest_auto_color_grade:
            lines.append(
                "Latest auto color grade: "
                f"{latest_auto_color_grade.get('resolved_look', latest_auto_color_grade.get('look', 'auto'))} "
                f"@ {latest_auto_color_grade.get('output_path', 'unknown')}"
            )
        latest_smartie_import = (self.artifacts or {}).get("latest_smartie_import")
        if latest_smartie_import:
            lines.append(
                "Latest Smartie import: "
                f"{latest_smartie_import.get('effect_count', 0)} planned zooms "
                f"@ {latest_smartie_import.get('manifest_path', 'unknown')}"
            )
        latest_agent_trace = (self.artifacts or {}).get("latest_agent_trace")
        if latest_agent_trace:
            lines.append(
                "Latest agent trace: "
                f"{len(latest_agent_trace.get('events') or [])} steps @ {latest_agent_trace.get('created_at', 'unknown')}"
            )
        if self.timeline:
            lines.append("Timeline:")
            for index, op in enumerate(self.timeline, start=1):
                lines.append(
                    f"  {index}. {op['op']} - {op.get('description', '')} @ {op.get('timestamp', '')}"
                )
        return "\n".join(lines)

    def replace_overlay_ranges(
        self,
        *,
        exclude_ops: set[str] | None = None,
    ) -> list[tuple[float, float]]:
        return self.overlay_ranges(
            exclude_ops=exclude_ops,
            include_picture_in_picture=False,
        )

    def overlay_ranges(
        self,
        *,
        exclude_ops: set[str] | None = None,
        include_ops: set[str] | None = None,
        include_picture_in_picture: bool = True,
    ) -> list[tuple[float, float]]:
        blocked_ranges: list[tuple[float, float]] = []
        excluded = exclude_ops or set()
        included = include_ops or set()
        for op in self.timeline:
            op_name = str(op.get("op") or "").strip()
            if op_name in excluded:
                continue
            if included and op_name not in included:
                continue
            overlays = (op.get("params") or {}).get("overlays") or []
            if not isinstance(overlays, list):
                continue
            for overlay in overlays:
                if not isinstance(overlay, dict):
                    continue
                compose_mode = str(
                    overlay.get("compose_mode") or overlay.get("composition_mode") or "replace"
                ).strip().lower()
                if (
                    not include_picture_in_picture
                    and compose_mode in {"pip", "overlay", "picture_in_picture", "picture-in-picture"}
                ):
                    continue
                try:
                    start_sec = float(overlay.get("start", 0.0))
                    end_sec = float(overlay.get("end", start_sec))
                except (TypeError, ValueError):
                    continue
                if end_sec - start_sec < 0.08:
                    continue
                blocked_ranges.append((start_sec, end_sec))
        return merge_time_ranges(blocked_ranges, gap_sec=0.08)
