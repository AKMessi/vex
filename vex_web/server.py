from __future__ import annotations

import json
import hashlib
import ipaddress
import io
import logging
import math
import mimetypes
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.message import Message
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import Any, BinaryIO, Callable
from urllib.parse import unquote, urlparse

import config
from agent import VideoAgent
from asset_registry import latest_assets
from engine import VideoEngineError
from job_runner import list_jobs
from providers import get_provider
from state import ProjectState
from tools import TOOL_EXECUTORS
from tools.creative_registry import latest_creative_runs
from video_generation.remotion_revisions import IMAGE_TYPES, VIDEO_TYPES
from vex_runtime.execution_store import ExecutionConflict
from vex_runtime.edit_graph import EditGraph
from vex_runtime.locking import exclusive_file_lock, process_is_running
from vex_runtime.media_search import search_media
from vex_runtime.project_catalog import catalog_path, initialize_catalog
from vex_runtime.studio_projects import create_project
from vex_web.media_preview import browser_preview, media_fingerprint
from vex_web.live_player import build_studio_player, live_media, live_timeline, waveform
from vex_web.production import (
    PRODUCTION_SLOT, ProductionStopped, produce_film, production_media, production_request, production_workspace,
    replace_studio_shot, studio_editing, studio_shots,
)
from vex_runtime.production_edits import (
    attach_footage, commit_production, preview_image, preview_production, preview_speech, propose_production,
    review_production, save_direction,
)
from vex_web.task_store import StudioTaskStore


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".flv"}
STATIC_FILES = {"app.js", "favicon.svg", "index.html", "styles.css"}
MAX_JSON_REQUEST_BYTES = 64 * 1024
MAX_UPLOAD_BYTES = 20 * 1024 * 1024 * 1024
MAX_MULTIPART_OVERHEAD_BYTES = 1024 * 1024
MAX_MULTIPART_HEADER_BYTES = 64 * 1024
MAX_FORM_FIELD_BYTES = 16 * 1024
MAX_CHAT_LENGTH = 12_000
MAX_PROJECT_NAME_LENGTH = 120
MAX_TASKS = 200
MAX_PENDING_TASKS = 12
TASK_RETENTION_SECONDS = 24 * 60 * 60
MAX_MULTIPART_PARTS = 8
PROJECT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
UPLOAD_DIR = Path(tempfile.gettempdir()) / "vex-web-uploads"
LOGGER = logging.getLogger("vex.web")


class WebRequestError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = int(status)


@dataclass(frozen=True)
class UploadedFile:
    filename: str
    path: Path
    size: int


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_safe(value.to_dict())
    return str(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _normalize_project_name(value: Any) -> str:
    name = str(value or "").strip()
    if len(name) > MAX_PROJECT_NAME_LENGTH:
        raise ValueError(f"Project names must be {MAX_PROJECT_NAME_LENGTH} characters or fewer.")
    if any(ord(char) < 32 and char not in {"\t"} for char in name):
        raise ValueError("Project names cannot contain control characters.")
    return name


def _is_loopback_host(value: str) -> bool:
    host = str(value or "").strip().lower().rstrip(".")
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _host_without_port(value: str) -> tuple[str, int | None]:
    raw = str(value or "").strip()
    if not raw:
        raise WebRequestError(HTTPStatus.BAD_REQUEST, "Missing Host header.")
    try:
        parsed = urlparse(f"//{raw}")
        host = parsed.hostname or ""
        port = parsed.port
    except ValueError as exc:
        raise WebRequestError(HTTPStatus.BAD_REQUEST, "Invalid Host header.") from exc
    if not host:
        raise WebRequestError(HTTPStatus.BAD_REQUEST, "Invalid Host header.")
    return host, port


def _configured_model_name(provider_name: str | None = None) -> str:
    provider = config.normalize_provider_name(provider_name or config.PROVIDER)
    names = {
        "groq": config.GROQ_MODEL,
        "gemini": config.GEMINI_MODEL,
        "claude": config.CLAUDE_MODEL,
        "ollama": config.local_llm_model("ollama"),
        "lmstudio": config.local_llm_model("lmstudio"),
        "llama_cpp": config.local_llm_model("llama_cpp"),
        "openai_compatible": config.local_llm_model("openai_compatible"),
    }
    return str(names.get(provider) or "configured model")


def _format_duration(value: Any) -> str:
    parsed = _optional_float(value)
    if parsed is None:
        return "—"
    seconds = max(0, int(round(parsed)))
    minutes, remainder = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{remainder:02d}"
    return f"{minutes}:{remainder:02d}"


def _format_bytes(value: Any) -> str:
    size = _safe_float(value)
    if size <= 0:
        return "—"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return "—"


def _relative_time(value: Any) -> str:
    try:
        parsed = datetime.fromisoformat(str(value or ""))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        seconds = max(0, int((datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds()))
    except (TypeError, ValueError):
        return "recently"
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    if seconds < 604800:
        return f"{seconds // 86400}d ago"
    return parsed.strftime("%b %-d, %Y") if os.name != "nt" else parsed.strftime("%b %#d, %Y")


def _project_state(project_id: str) -> ProjectState:
    normalized = str(project_id or "").strip()
    if not PROJECT_ID_RE.fullmatch(normalized):
        raise ValueError("Invalid project id.")
    return ProjectState.load_exact(normalized)


class _MultipartBodyReader:
    """Bounded multipart reader that never buffers a complete media file."""

    def __init__(self, stream: BinaryIO, length: int, boundary: bytes) -> None:
        self.stream = stream
        self.remaining = length
        self.boundary = boundary
        self.buffer = bytearray()

    def _fill(self, minimum: int = 1) -> None:
        while len(self.buffer) < minimum and self.remaining > 0:
            amount = min(1024 * 1024, self.remaining)
            read = getattr(self.stream, "read1", self.stream.read)
            chunk = read(amount)
            if not chunk:
                raise WebRequestError(HTTPStatus.BAD_REQUEST, "Upload ended before the request body was complete.")
            self.buffer.extend(chunk)
            self.remaining -= len(chunk)

    def readline(self, limit: int) -> bytes:
        while True:
            newline = self.buffer.find(b"\n")
            if newline >= 0:
                end = newline + 1
                if end > limit:
                    raise WebRequestError(HTTPStatus.BAD_REQUEST, "Multipart header line is too large.")
                result = bytes(self.buffer[:end])
                del self.buffer[:end]
                return result
            if len(self.buffer) >= limit:
                raise WebRequestError(HTTPStatus.BAD_REQUEST, "Multipart header line is too large.")
            if self.remaining <= 0:
                raise WebRequestError(HTTPStatus.BAD_REQUEST, "Upload contains an incomplete multipart header.")
            self._fill(len(self.buffer) + 1)

    def read_part(self, sink: BinaryIO, limit: int, limit_message: str) -> tuple[bool, int]:
        marker = b"\r\n--" + self.boundary
        written = 0

        def write_chunk(data: bytes | bytearray) -> None:
            nonlocal written
            if written + len(data) > limit:
                raise WebRequestError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, limit_message)
            sink.write(data)
            written += len(data)

        while True:
            marker_at = self.buffer.find(marker)
            if marker_at >= 0:
                required = marker_at + len(marker) + 2
                self._fill(required)
                if len(self.buffer) < required:
                    raise WebRequestError(HTTPStatus.BAD_REQUEST, "Upload is missing its closing multipart boundary.")
                suffix = bytes(self.buffer[marker_at + len(marker) : required])
                if suffix in {b"--", b"\r\n"}:
                    write_chunk(self.buffer[:marker_at])
                    del self.buffer[:required]
                    return suffix == b"--", written
                # A boundary-like byte sequence inside the media payload is ordinary data.
                safe_end = marker_at + 2
                write_chunk(self.buffer[:safe_end])
                del self.buffer[:safe_end]
                continue

            keep = len(marker) + 2
            if len(self.buffer) > keep:
                safe_end = len(self.buffer) - keep
                write_chunk(self.buffer[:safe_end])
                del self.buffer[:safe_end]
            if self.remaining <= 0:
                raise WebRequestError(HTTPStatus.BAD_REQUEST, "Upload is missing its closing multipart boundary.")
            self._fill(len(self.buffer) + 1)

    def discard_remaining(self) -> None:
        self.buffer.clear()
        while self.remaining > 0:
            amount = min(1024 * 1024, self.remaining)
            read = getattr(self.stream, "read1", self.stream.read)
            chunk = read(amount)
            if not chunk:
                break
            self.remaining -= len(chunk)


def _multipart_boundary(content_type: str) -> bytes:
    message = Message()
    message["content-type"] = content_type
    boundary = message.get_param("boundary", header="content-type")
    if not isinstance(boundary, str) or not boundary:
        raise WebRequestError(HTTPStatus.BAD_REQUEST, "Upload is missing its multipart boundary.")
    try:
        encoded = boundary.encode("ascii")
    except UnicodeEncodeError as exc:
        raise WebRequestError(HTTPStatus.BAD_REQUEST, "Upload has an invalid multipart boundary.") from exc
    if len(encoded) > 200 or any(byte in encoded for byte in (b"\r", b"\n")):
        raise WebRequestError(HTTPStatus.BAD_REQUEST, "Upload has an invalid multipart boundary.")
    return encoded


def _content_disposition(value: str) -> tuple[str, str]:
    message = Message()
    message["content-disposition"] = value
    if message.get_content_disposition() != "form-data":
        raise WebRequestError(HTTPStatus.BAD_REQUEST, "Upload contains an invalid form-data part.")
    name = message.get_param("name", header="content-disposition")
    filename = message.get_filename() or ""
    if not isinstance(name, str) or not name:
        raise WebRequestError(HTTPStatus.BAD_REQUEST, "Upload contains an unnamed form-data part.")
    return name, Path(filename.replace("\\", "/")).name


def _stream_multipart_form(
    stream: BinaryIO,
    content_length: int,
    content_type: str,
    *,
    extensions: set[str] = VIDEO_EXTENSIONS,
    fields_allowed: frozenset[str] = frozenset({"name", "file"}),
) -> tuple[dict[str, str], UploadedFile | None]:
    boundary = _multipart_boundary(content_type)
    reader = _MultipartBodyReader(stream, content_length, boundary)
    expected_opening = b"--" + boundary + b"\r\n"
    if reader.readline(len(expected_opening) + 2) != expected_opening:
        raise WebRequestError(HTTPStatus.BAD_REQUEST, "Upload has an invalid opening multipart boundary.")

    fields: dict[str, str] = {}
    uploaded: UploadedFile | None = None
    temporary_path: Path | None = None
    try:
        final = False
        part_count = 0
        while not final:
            part_count += 1
            if part_count > MAX_MULTIPART_PARTS:
                raise WebRequestError(HTTPStatus.BAD_REQUEST, "Upload contains too many form-data parts.")
            headers: dict[str, str] = {}
            header_bytes = 0
            while True:
                line = reader.readline(MAX_MULTIPART_HEADER_BYTES)
                header_bytes += len(line)
                if header_bytes > MAX_MULTIPART_HEADER_BYTES:
                    raise WebRequestError(HTTPStatus.BAD_REQUEST, "Multipart headers are too large.")
                if line == b"\r\n":
                    break
                key, separator, value = line.partition(b":")
                if not separator:
                    raise WebRequestError(HTTPStatus.BAD_REQUEST, "Upload contains a malformed multipart header.")
                headers[key.decode("latin-1").strip().lower()] = value.decode("latin-1").strip()

            name, filename = _content_disposition(headers.get("content-disposition", ""))
            if name not in fields_allowed:
                raise WebRequestError(HTTPStatus.BAD_REQUEST, f"Unexpected upload field {name!r}.")
            if filename:
                if name != "file" or uploaded is not None:
                    raise WebRequestError(HTTPStatus.BAD_REQUEST, "Upload must contain exactly one media file.")
                suffix = Path(filename).suffix.lower()
                if suffix not in extensions:
                    names = ", ".join(sorted(value.removeprefix(".") for value in extensions))
                    raise ValueError(f"Choose a supported media file: {names}.")
                UPLOAD_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
                try:
                    UPLOAD_DIR.chmod(0o700)
                except OSError:
                    pass
                with tempfile.NamedTemporaryFile(
                    mode="w+b",
                    prefix="upload_",
                    suffix=suffix,
                    dir=UPLOAD_DIR,
                    delete=False,
                ) as target:
                    temporary_path = Path(target.name)
                    final, size = reader.read_part(target, MAX_UPLOAD_BYTES, "Uploaded file is too large.")
                if size <= 0:
                    raise ValueError("The selected video file is empty.")
                uploaded = UploadedFile(filename=filename, path=temporary_path, size=size)
            else:
                if name == "file":
                    raise WebRequestError(HTTPStatus.BAD_REQUEST, "Upload did not include a media file.")
                target = io.BytesIO()
                final, _ = reader.read_part(target, MAX_FORM_FIELD_BYTES, "Upload form field is too large.")
                fields[name] = target.getvalue().decode("utf-8", errors="strict")

        reader.discard_remaining()
        return fields, uploaded
    except Exception:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                LOGGER.warning("Unable to remove rejected temporary upload %s", temporary_path)
        raise


def _clear_upload(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        LOGGER.warning("Unable to remove temporary upload %s", path)


def _parse_byte_range(value: str, file_size: int) -> tuple[int, int]:
    if file_size <= 0:
        raise ValueError("Media range is not satisfiable.")
    unit, separator, raw_value = str(value or "").partition("=")
    if separator != "=" or unit.strip().lower() != "bytes" or "," in raw_value:
        raise ValueError("Media range is not satisfiable.")
    raw_start, dash, raw_end = raw_value.strip().partition("-")
    if dash != "-" or (not raw_start and not raw_end):
        raise ValueError("Media range is not satisfiable.")
    try:
        if raw_start:
            start = int(raw_start)
            end = int(raw_end) if raw_end else file_size - 1
        else:
            suffix_length = int(raw_end)
            if suffix_length <= 0:
                raise ValueError
            start = max(0, file_size - suffix_length)
            end = file_size - 1
    except ValueError as exc:
        raise ValueError("Media range is not satisfiable.") from exc
    if start < 0 or start >= file_size or end < start:
        raise ValueError("Media range is not satisfiable.")
    return start, min(end, file_size - 1)


def _project_summary(item: dict[str, Any]) -> dict[str, Any]:
    project_id = str(item.get("project_id") or "")
    if not PROJECT_ID_RE.fullmatch(project_id):
        raise ValueError("Invalid project id in project state.")
    payload = dict(item)
    payload.pop("working_dir", None)
    payload.pop("source_file", None)
    payload["project_id"] = project_id
    payload["short_id"] = project_id[:8]
    payload["source_name"] = Path(str(item.get("source_file") or "")).name or "Untitled media"
    payload["updated_label"] = _relative_time(item.get("updated_at"))
    try:
        state = _project_state(project_id)
    except (FileNotFoundError, ValueError, OSError):
        return payload
    metadata = state.metadata or {}
    payload.update(
        {
            "project_name": state.project_name,
            "source_name": Path(state.source_files[0]).name if state.source_files else "Untitled media",
            "duration": _format_duration(metadata.get("duration_sec")),
            "duration_sec": _safe_float(metadata.get("duration_sec")),
            "resolution": f"{metadata.get('width', 0)}×{metadata.get('height', 0)}" if metadata.get("width") else "—",
            "timeline_ops": len(state.timeline),
            "has_preview": bool(state.working_file and Path(state.working_file).is_file()),
        }
    )
    return payload


def _project_summaries() -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for item in ProjectState.list_projects():
        try:
            summaries.append(_project_summary(item))
        except (FileNotFoundError, ValueError, OSError):
            LOGGER.warning("Skipping invalid Vex project entry %r", item.get("project_id"))
    return summaries


def _chat_history(state: ProjectState) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for item in (state.session_log or [])[-40:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip()
        content = item.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str) or not content.strip():
            continue
        history.append({"role": role, "content": content, "created_at": item.get("created_at") or ""})
    return history


def _timeline_rows(state: ProjectState) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, operation in enumerate(reversed(state.timeline[-30:]), start=1):
        if not isinstance(operation, dict):
            continue
        params = operation.get("params") if isinstance(operation.get("params"), dict) else {}
        detail = str(operation.get("description") or "").strip()
        if not detail:
            detail = ", ".join(
                f"{key.replace('_', ' ')}: {value}"
                for key, value in list(params.items())[:3]
                if not str(key).endswith("_label") and key not in {"file_paths", "output_path"}
            )
        rows.append(
            {
                "index": len(state.timeline) - index + 1,
                "op": str(operation.get("op") or "operation").replace("_", " "),
                "detail": detail or "Timeline operation",
                "timestamp": str(operation.get("timestamp") or ""),
                "result_file": str(operation.get("result_file") or ""),
            }
        )
    return rows


def _artifact_summary(state: ProjectState) -> list[dict[str, Any]]:
    artifacts = state.artifacts or {}
    rows: list[dict[str, Any]] = []
    labels = {
        "latest_transcript": "Transcript",
        "latest_auto_visuals": "Auto visuals",
        "latest_auto_broll": "Auto b-roll",
        "latest_auto_shorts": "Auto shorts",
        "latest_auto_color_grade": "Color grade",
        "latest_added_song": "Song mix",
        "latest_generated_video": "Generated video",
        "generated_source": "Generated production",
        "latest_encode": "Latest encode",
        "latest_upscale": "Upscale",
    }
    for key, label in labels.items():
        value = artifacts.get(key)
        if not isinstance(value, dict):
            continue
        summary = value.get("count") or value.get("resolved_look") or value.get("look") or value.get("selected_skill_id") or value.get("output_path") or "Ready"
        rows.append({"key": key, "label": label, "summary": str(summary)})
    return rows


def _project_detail(state: ProjectState) -> dict[str, Any]:
    metadata = state.metadata or {}
    production = studio_editing(state)
    try:
        creative_runs = latest_creative_runs(state.working_dir, limit=12)
    except (OSError, ValueError):
        LOGGER.warning("Unable to read creative runs for project %s", state.project_id, exc_info=True)
        creative_runs = []
    source_name = Path(state.source_files[0]).name if state.source_files else "Untitled media"
    trace = (state.artifacts or {}).get("latest_agent_trace")
    jobs = [
        {
            "job_id": record.job_id,
            "tool_name": record.tool_name,
            "status": record.status,
            "stage": record.stage,
            "progress": record.progress,
            "updated_at": record.updated_at,
            "message": record.message or record.error,
        }
        for record in list_jobs(state.working_dir, limit=12)
    ]
    media_assets = [
        {
            "asset_id": str(item.get("asset_id") or ""),
            "name": Path(str(item.get("path") or "")).name,
            "kind": str(item.get("kind") or ""),
            "role": str(item.get("role") or ""),
            "checksum": str(item.get("checksum_sha256") or "")[:12],
            "parent_count": len(item.get("parents") or []),
            "created_at": str(item.get("created_at") or ""),
        }
        for item in latest_assets(state.working_dir, limit=12)
    ]
    working_file = Path(state.working_file)
    media_version = ""
    if state.working_file and working_file.is_file():
        try:
            media_version = media_fingerprint(working_file)
        except OSError:
            pass
    return {
        "project": {
            "project_id": state.project_id,
            "short_id": state.project_id[:8],
            "project_name": state.project_name,
            "source_name": source_name,
            "created_at": state.created_at,
            "updated_at": state.updated_at,
            "updated_label": _relative_time(state.updated_at),
            "duration": _format_duration(metadata.get("duration_sec")),
            "duration_sec": _safe_float(metadata.get("duration_sec")),
            "resolution": f"{metadata.get('width', 0)}×{metadata.get('height', 0)}" if metadata.get("width") else "—",
            "fps": metadata.get("fps") or "—",
            "size": _format_bytes(metadata.get("size_bytes")),
            "timeline_ops": len(state.timeline),
            "redo_count": len(state.redo_stack),
            "provider": state.provider or config.PROVIDER,
            "model": state.model or _configured_model_name(state.provider),
        },
        "media": {
            "current": f"/api/projects/{state.project_id}/media/current?v={media_version}",
            "source": f"/api/projects/{state.project_id}/media/source",
            "available": bool(media_version),
            "download_name": working_file.name,
        },
        "chat": _chat_history(state),
        "timeline": _timeline_rows(state),
        "edit_graph": _public_edit_graph(state),
        "production_shots": production.get("shots", []),
        "production": production,
        "artifacts": _artifact_summary(state),
        "creative_runs": _json_safe(creative_runs),
        "jobs": jobs,
        "media_assets": media_assets,
        "latest_trace": _json_safe(trace if isinstance(trace, dict) else {"events": []}),
    }


def _public_edit_graph(state: ProjectState) -> dict[str, Any] | None:
    if not state.edit_graph:
        return None
    graph = EditGraph.from_mapping(state.edit_graph)
    visible = graph.spans[:80]
    return {
        "provenance": graph.provenance,
        "duration_sec": float(graph.duration),
        "span_count": len(graph.spans),
        "truncated": len(visible) < len(graph.spans),
        "spans": [
            {
                "source_name": Path(graph.sources[span.source_id].media_path).name,
                "source_start_sec": float(span.source_start),
                "source_end_sec": float(span.source_end),
                "output_start_sec": float(span.output_start),
                "output_end_sec": float(span.output_end),
                "duration_sec": float(span.duration),
            }
            for span in visible
        ],
    }


def _public_error_message(error: BaseException) -> str:
    message = str(error).strip() or "The task failed."
    message = re.sub(r"(?i)\b(sk-[A-Za-z0-9_-]{10,})\b", "[redacted]", message)
    message = re.sub(
        r"(?i)\b(api[_-]?key|authorization|access[_-]?token)\b([\s\"':=]+)"
        r"(?:(?:bearer|basic)\s+)?([^\s,;}'\"]+)",
        r"\1\2[redacted]",
        message,
    )
    return message[:2_000]


@dataclass
class WebTask:
    task_id: str
    project_id: str
    kind: str
    label: str
    status: str = "queued"
    message: str = ""
    created_at: str = field(default_factory=_now)
    started_at: str = ""
    finished_at: str = ""
    events: list[dict[str, Any]] = field(default_factory=list)
    stream: str = ""
    result: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    request: dict[str, Any] = field(default_factory=dict)
    stage: str = ""
    progress: float = 0.0
    stop_requested: bool = False
    touched_at: float = field(default_factory=time.monotonic, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "project_id": self.project_id,
            "kind": self.kind,
            "label": self.label,
            "status": self.status,
            "message": self.message,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "events": self.events[-80:],
            "stream": self.stream,
            "result": _json_safe(self.result),
            "error": self.error,
            "request": _json_safe(self.request),
            "stage": self.stage,
            "progress": self.progress,
            "stop_requested": self.stop_requested,
        }


class TaskManager:
    def __init__(self, *, max_workers: int = 3, persist: bool = False) -> None:
        self._tasks: dict[str, WebTask] = {}
        self._active_projects: dict[str, str] = {}
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="vex-web")
        self._closed = False
        self._store = StudioTaskStore() if persist else None
        self._instance_id = uuid.uuid4().hex
        self._project_dirs: dict[str, Path] = {}
        self._last_persisted: dict[str, float] = {}
        self._last_stream_persisted: dict[str, float] = {}

    def start_production(self, request: dict, model: str) -> WebTask:
        if self._store is None:
            return self.submit(
                PRODUCTION_SLOT, "production", "Creating a video",
                lambda task: produce_film(task.task_id, request, model,
                                           lambda stage, progress: self.production_progress(task, stage, progress)),
                request=request,
            )
        with self._lock:
            if self._closed:
                raise RuntimeError("Vex Studio is shutting down.")
            directory = self._project_dir(PRODUCTION_SLOT, create=True)
            with exclusive_file_lock(directory / ".production-start.lock", timeout_sec=5):
                if self._active_for_project_locked(PRODUCTION_SLOT):
                    raise RuntimeError("Vex is already creating a video.")
                self._store.prune(directory, retention_seconds=TASK_RETENTION_SECONDS, max_tasks=MAX_TASKS)
                task = WebTask(task_id=f"task_{PRODUCTION_SLOT}_{uuid.uuid4().hex[:16]}",
                               project_id=PRODUCTION_SLOT, kind="production", label="Creating a video", request=dict(request))
                task.request.setdefault("checkpoint_id", task.task_id)
                from vex_web.production_worker import worker_files

                lock, _, log = worker_files(directory, task.task_id)
                flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
                with exclusive_file_lock(lock, timeout_sec=5), log.open("wb") as stream:
                    child = subprocess.Popen(
                        [sys.executable, "-m", "vex_web.production_worker", task.task_id, model],
                        stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                        creationflags=flags, start_new_session=os.name != "nt",
                        env={**os.environ, "AGENT_PROJECTS_DIR": str(directory.parents[1])},
                    )
                    try:
                        self._store.put(directory, task.to_dict(), owner_pid=child.pid, owner_instance=task.task_id)
                    except BaseException:
                        child.terminate()
                        child.wait(timeout=5)
                        raise
                threading.Thread(target=child.wait, daemon=True, name="vex-production-reaper").start()
                return task

    def retry_production(self, task_id: str, model: str) -> WebTask:
        if _task_project_id(task_id) != PRODUCTION_SLOT:
            raise ValueError("Choose a saved production run.")
        prior = self.snapshot(task_id)
        if not prior or prior["status"] not in {"failed", "cancelled"}:
            raise ValueError("Retry is available for interrupted, stopped or failed production runs.")
        return self.start_production(prior["request"], model)

    def snapshot(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            self._prune_locked()
            task = self._tasks.get(task_id)
            if task is not None:
                return task.to_dict()
            if self._store is None:
                return None
            project_id = _task_project_id(task_id)
            if project_id is None:
                return None
            working_dir = self._project_dir(project_id)
            if working_dir is None:
                return None
            row = self._store.get(working_dir, task_id)
            return self._persisted_snapshot_locked(working_dir, row) if row else None

    def active_snapshot(self, project_id: str) -> dict[str, Any] | None:
        with self._lock:
            task = self._active_for_project_locked(project_id)
            return task.to_dict() if task is not None else None

    def latest_snapshot(self, project_id: str) -> dict[str, Any] | None:
        with self._lock:
            tasks = [task for task in self._tasks.values() if task.project_id == project_id]
            if tasks:
                return max(tasks, key=lambda task: task.touched_at).to_dict()
            if self._store is None:
                return None
            working_dir = self._project_dir(project_id)
            if working_dir is None:
                return None
            row = self._store.latest(working_dir)
            return self._persisted_snapshot_locked(working_dir, row) if row else None

    def _active_for_project_locked(self, project_id: str) -> WebTask | None:
        task_id = self._active_projects.get(project_id)
        task = self._tasks.get(task_id) if task_id else None
        if task is not None and task.status not in {"queued", "running"}:
            return None
        if task is not None or self._store is None:
            return task
        working_dir = self._project_dir(project_id)
        if working_dir is None:
            return None
        row = self._store.active(working_dir)
        payload = self._persisted_snapshot_locked(working_dir, row) if row else None
        return _web_task_from_payload(payload) if payload and payload["status"] in {"queued", "running"} else None

    def submit(
        self, project_id: str, kind: str, label: str, work: Callable[[WebTask], dict[str, Any]],
        *, request: dict[str, Any] | None = None, on_finished: Callable[[], None] | None = None,
    ) -> WebTask:
        with self._lock:
            if self._closed:
                raise RuntimeError("Vex Studio is shutting down.")
            self._prune_locked()
            pending_count = sum(task.status in {"queued", "running"} for task in self._tasks.values())
            if pending_count >= MAX_PENDING_TASKS:
                raise RuntimeError("Vex Studio is at capacity. Wait for an active edit to finish and try again.")
            active = self._active_for_project_locked(project_id)
            if active and active.status in {"queued", "running"}:
                raise RuntimeError(f"Vex is already working on {active.label.lower()}.")
            if self._store is not None and self._project_dir(project_id, create=True) is None:
                raise FileNotFoundError(f"No project found for id {project_id!r}.")
            if self._store is not None:
                self._store.prune(
                    self._project_dirs[project_id],
                    retention_seconds=TASK_RETENTION_SECONDS,
                    max_tasks=MAX_TASKS,
                )
            task_id = (
                f"task_{project_id}_{uuid.uuid4().hex[:16]}"
                if self._store is not None else f"task_{uuid.uuid4().hex[:16]}"
            )
            task = WebTask(task_id=task_id, project_id=project_id, kind=kind, label=label, request=dict(request or {}))
            self._tasks[task.task_id] = task
            self._active_projects[project_id] = task.task_id
            try:
                self._persist_locked(task)
                future = self._executor.submit(self._run, task, work)
                if on_finished is not None:
                    future.add_done_callback(lambda _future: on_finished())
            except ExecutionConflict:
                self._tasks.pop(task.task_id, None)
                self._active_projects.pop(project_id, None)
                raise RuntimeError("Vex is already working on this project.") from None
            except RuntimeError:
                task.status = "failed"
                task.message = "Vex Studio is shutting down."
                task.error = task.message
                task.finished_at = _now()
                self._persist_locked(task)
                self._tasks.pop(task.task_id, None)
                self._active_projects.pop(project_id, None)
                raise RuntimeError(task.message) from None
            except Exception:
                self._tasks.pop(task.task_id, None)
                self._active_projects.pop(project_id, None)
                raise
        return task

    def run_if_idle(self, project_id: str, work: Callable[[], Any]) -> Any:
        with self._lock:
            active = self._active_for_project_locked(project_id)
            if active and active.status in {"queued", "running"}:
                raise RuntimeError(f"Vex is already working on {active.label.lower()}.")
            return work()

    def _run(self, task: WebTask, work: Callable[[WebTask], dict[str, Any]]) -> None:
        try:
            with self._lock:
                if task.status == "cancelled":
                    return
                task.status = "running"
                task.started_at = _now()
                task.message = task.label
                task.touched_at = time.monotonic()
                self._persist_locked(task)
            result = work(task)
            with self._lock:
                task.result = dict(result or {})
                task.status = "succeeded" if bool(task.result.get("success", True)) else "failed"
                task.message = _public_error_message(
                    RuntimeError(task.result.get("message") or "Could not complete the request.")
                ) if task.status == "failed" else str(task.result.get("message") or "Completed")
                task.finished_at = _now()
                task.error = "" if task.status == "succeeded" else task.message
                if task.status == "succeeded":
                    task.progress = 1.0
                task.touched_at = time.monotonic()
                self._persist_locked(task)
        except ProductionStopped:
            with self._lock:
                task.status = "cancelled"
                task.message = "Production stopped"
                task.finished_at = _now()
                task.touched_at = time.monotonic()
                self._persist_locked(task)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Vex web task %s failed", task.task_id)
            with self._lock:
                public_message = _public_error_message(exc)
                task.status = "failed"
                task.error = public_message
                task.message = public_message
                task.finished_at = _now()
                task.events.append({"kind": "system", "title": "Task failed", "detail": task.message, "status": "error"})
                task.touched_at = time.monotonic()
                try:
                    self._persist_locked(task)
                except Exception:  # noqa: BLE001
                    LOGGER.exception("Unable to persist failed Studio task %s", task.task_id)
        finally:
            with self._lock:
                if self._active_projects.get(task.project_id) == task.task_id:
                    self._active_projects.pop(task.project_id, None)

    def production_progress(self, task: WebTask, stage: str, progress: float) -> None:
        with self._lock:
            if task.stop_requested:
                raise ProductionStopped()
            task.stage = task.message = stage
            task.progress = progress
            task.events.append({"kind": "production", "title": stage, "status": "running"})
            task.touched_at = time.monotonic()
            self._persist_locked(task)

    def stop_production(self, task_id: str) -> dict[str, Any]:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None and self._store is not None and _task_project_id(task_id) == PRODUCTION_SLOT:
                payload = self.snapshot(task_id)
                if payload is not None:
                    if payload["status"] in {"queued", "running"} and payload["progress"] < .98:
                        from vex_web.production_worker import worker_files

                        _, stop, _ = worker_files(production_workspace(), task_id)
                        stop.touch()
                        payload.update(stop_requested=True, message="Stopping after the current step")
                    return payload
            if task is None or task.kind != "production":
                raise FileNotFoundError("Production run not found on this Studio server.")
            if task.status == "queued":
                self._cancel_queued_locked(task, "Production stopped")
                return task.to_dict()
            if task.status != "running" or task.progress >= 0.98:
                return task.to_dict()
            task.stop_requested = True
            task.message = "Stopping after the current step"
            task.touched_at = time.monotonic()
            self._persist_locked(task)
            return task.to_dict()

    def _cancel_queued_locked(self, task: WebTask, message: str) -> None:
        task.status = "cancelled"
        task.message = message
        task.finished_at = _now()
        task.touched_at = time.monotonic()
        self._active_projects.pop(task.project_id, None)
        self._persist_locked(task)

    def append_event(self, task: WebTask, event: Any) -> None:
        try:
            payload = event.to_dict() if hasattr(event, "to_dict") else dict(event or {})
        except Exception:  # noqa: BLE001
            LOGGER.warning("Ignoring malformed task event", exc_info=True)
            return
        with self._lock:
            task.events.append(_json_safe(payload))
            task.events = task.events[-120:]
            task.message = str(payload.get("detail") or payload.get("title") or task.message)
            task.touched_at = time.monotonic()
            try:
                self._persist_locked(task)
            except Exception:  # noqa: BLE001
                LOGGER.exception("Unable to persist Studio task event %s", task.task_id)

    def append_stream(self, task: WebTask, chunk: str) -> None:
        with self._lock:
            task.stream = (task.stream + str(chunk or ""))[-20_000:]
            task.touched_at = time.monotonic()
            if task.touched_at - self._last_stream_persisted.get(task.task_id, 0.0) >= 0.5:
                try:
                    self._persist_locked(task)
                    self._last_stream_persisted[task.task_id] = task.touched_at
                except Exception:  # noqa: BLE001
                    LOGGER.exception("Unable to persist Studio task stream %s", task.task_id)

    def shutdown(self) -> None:
        with self._lock:
            self._closed = True
            for task in self._tasks.values():
                if task.status == "queued":
                    try:
                        self._cancel_queued_locked(task, "Studio shut down before this task started")
                    except Exception:  # noqa: BLE001
                        LOGGER.exception("Unable to persist cancelled Studio task %s", task.task_id)
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _prune_locked(self) -> None:
        now = time.monotonic()
        finished = [
            task
            for task in self._tasks.values()
            if task.status not in {"queued", "running"}
        ]
        expired_ids = {
            task.task_id
            for task in finished
            if now - task.touched_at > TASK_RETENTION_SECONDS
        }
        remaining_finished = sorted(
            (task for task in finished if task.task_id not in expired_ids),
            key=lambda item: item.touched_at,
            reverse=True,
        )
        excess = max(0, len(self._tasks) - MAX_TASKS - len(expired_ids))
        expired_ids.update(task.task_id for task in remaining_finished[-excess:] if excess)
        for task_id in expired_ids:
            self._tasks.pop(task_id, None)
            self._last_persisted.pop(task_id, None)
            self._last_stream_persisted.pop(task_id, None)

    def _project_dir(self, project_id: str, *, create: bool = False) -> Path | None:
        cached = self._project_dirs.get(project_id)
        if cached is not None:
            return cached
        if project_id == PRODUCTION_SLOT:
            directory = production_workspace()
            if not catalog_path(directory).is_file():
                if not create:
                    return None
                initialize_catalog(directory)
            self._project_dirs[project_id] = directory
            return directory
        try:
            state = _project_state(project_id)
        except FileNotFoundError:
            return None
        directory = Path(state.working_dir)
        if not catalog_path(directory).is_file():
            if not create:
                return None
            state.save()  # Import a legacy project before recording its first Studio task.
        self._project_dirs[project_id] = directory
        return directory

    def _persist_locked(self, task: WebTask) -> None:
        if self._store is None:
            return
        directory = self._project_dir(task.project_id, create=True)
        if directory is None:
            raise FileNotFoundError(f"No project found for id {task.project_id!r}.")
        self._store.put(
            directory,
            task.to_dict(),
            owner_pid=os.getpid(),
            owner_instance=self._instance_id,
        )
        self._last_persisted[task.task_id] = time.monotonic()

    def _persisted_snapshot_locked(
        self,
        working_dir: Path,
        row: dict[str, Any],
    ) -> dict[str, Any]:
        payload = dict(row["payload"])
        if payload.get("status") not in {"queued", "running"}:
            return payload
        if process_is_running(row["owner_pid"]):
            if payload.get("kind") == "production" and row["owner_instance"] == payload.get("task_id"):
                from vex_web.production_worker import worker_files

                _, stop, _ = worker_files(working_dir, payload["task_id"])
                if stop.is_file():
                    payload.update(stop_requested=True, message="Stopping after the current step")
            return payload
        message = "The Studio process stopped before this task finished. Check the project before retrying."
        payload.update(status="failed", message=message, error=message, finished_at=_now())
        payload["events"] = [
            *list(payload.get("events") or [])[-79:],
            {"kind": "system", "title": "Task interrupted", "detail": message, "status": "error"},
        ]
        assert self._store is not None
        try:
            self._store.put(working_dir, payload, owner_pid=os.getpid(), owner_instance=self._instance_id,
                            expected_version=row["version"])
        except ExecutionConflict:
            latest = self._store.get(working_dir, payload["task_id"])
            if latest:
                return latest["payload"]
        return payload


def _task_project_id(task_id: str) -> str | None:
    if not task_id.startswith("task_"):
        return None
    project_id, separator, suffix = task_id[5:].rpartition("_")
    if (
        not separator or (project_id != PRODUCTION_SLOT and not PROJECT_ID_RE.fullmatch(project_id))
        or not re.fullmatch(r"[0-9a-f]{16}", suffix)
    ):
        return None
    return project_id


def _web_task_from_payload(payload: dict[str, Any]) -> WebTask:
    return WebTask(
        task_id=str(payload["task_id"]),
        project_id=str(payload["project_id"]),
        kind=str(payload["kind"]),
        label=str(payload["label"]),
        status=str(payload["status"]),
        message=str(payload.get("message") or ""),
        created_at=str(payload.get("created_at") or ""),
        started_at=str(payload.get("started_at") or ""),
        finished_at=str(payload.get("finished_at") or ""),
        events=list(payload.get("events") or []),
        stream=str(payload.get("stream") or ""),
        result=dict(payload.get("result") or {}),
        error=str(payload.get("error") or ""),
        request=dict(payload.get("request") or {}),
        stage=str(payload.get("stage") or ""),
        progress=_safe_float(payload.get("progress")),
        stop_requested=bool(payload.get("stop_requested")),
    )


def _run_chat(manager: TaskManager, task: WebTask, message: str) -> dict[str, Any]:
    state = _project_state(task.project_id)
    provider = get_provider(state.provider or config.PROVIDER)
    agent = VideoAgent(state, provider)
    response = agent.run(
        message,
        stream_callback=lambda chunk: manager.append_stream(task, chunk),
        trace_callback=lambda event: manager.append_event(task, event),
        tool_callback=lambda phase, tool, success: manager.append_event(
            task,
            {
                "kind": "tool",
                "title": f"{tool} {phase}",
                "detail": "Tool completed" if phase == "finish" and success else "Working…",
                "status": "success" if phase == "finish" and success else "running",
            },
        ),
    )
    return {
        "success": bool(response.success),
        "message": response.message,
        "tools_called": response.tools_called,
        "suggestions": response.suggestions,
    }


class VexHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 64

    def __init__(self, server_address: tuple[str, int], task_manager: TaskManager) -> None:
        self.task_manager = task_manager
        self.address_family = socket.getaddrinfo(
            server_address[0],
            server_address[1],
            type=socket.SOCK_STREAM,
        )[0][0]
        super().__init__(server_address, VexRequestHandler)


class VexRequestHandler(BaseHTTPRequestHandler):
    server_version = "VexStudio/1.0"
    sys_version = ""
    protocol_version = "HTTP/1.1"

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(60)

    def log_message(self, format: str, *args: Any) -> None:
        LOGGER.info("%s - %s", self.client_address[0], format % args)

    def version_string(self) -> str:
        return self.server_version

    @property
    def parsed(self):
        return urlparse(self.path)

    @property
    def tasks(self) -> TaskManager:
        return self.server.task_manager  # type: ignore[attr-defined,no-any-return]

    def _send_security_headers(self) -> None:
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; media-src 'self' blob:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=(), payment=()")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")

    def _finish_headers(self) -> None:
        self._send_security_headers()
        self.end_headers()

    def _write_response(self, raw: bytes) -> None:
        if self.command == "HEAD" or not raw:
            return
        try:
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
            self.close_connection = True

    def _send_json(self, payload: Any, status: int = 200) -> None:
        raw = json.dumps(_json_safe(payload), ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        if self.close_connection:
            self.send_header("Connection", "close")
        self._finish_headers()
        self._write_response(raw)

    def _send_error(self, status: int, message: str) -> None:
        self._send_json({"error": message, "status": status}, status)

    def _content_length(self, maximum: int) -> int:
        if self.headers.get("Transfer-Encoding"):
            raise WebRequestError(HTTPStatus.NOT_IMPLEMENTED, "Chunked request bodies are not supported.")
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise WebRequestError(HTTPStatus.LENGTH_REQUIRED, "Content-Length is required.")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise WebRequestError(HTTPStatus.BAD_REQUEST, "Invalid Content-Length header.") from exc
        if length < 0:
            raise WebRequestError(HTTPStatus.BAD_REQUEST, "Invalid Content-Length header.")
        if length > maximum:
            self.close_connection = True
            raise WebRequestError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Request is too large.")
        return length

    def _read_body(self, maximum: int) -> bytes:
        length = self._content_length(maximum)
        body = self.rfile.read(length)
        if len(body) != length:
            self.close_connection = True
            raise WebRequestError(HTTPStatus.BAD_REQUEST, "Request body ended unexpectedly.")
        return body

    def _read_json(self) -> dict[str, Any]:
        content_type = str(self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            raise WebRequestError(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Content-Type must be application/json.")
        body = self._read_body(MAX_JSON_REQUEST_BYTES)
        try:
            payload = json.loads(body.decode("utf-8")) if body else {}
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Request body must be valid JSON.") from exc
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object.")
        return payload

    def _validate_request(self, *, mutation: bool = False) -> None:
        request_host, request_port = _host_without_port(self.headers.get("Host") or "")
        if not _is_loopback_host(request_host):
            raise WebRequestError(HTTPStatus.MISDIRECTED_REQUEST, "Vex Studio only accepts localhost requests.")

        fetch_site = str(self.headers.get("Sec-Fetch-Site") or "").lower()
        if fetch_site == "cross-site" and self.parsed.path.startswith("/api/"):
            raise WebRequestError(HTTPStatus.FORBIDDEN, "Cross-site requests are not allowed.")
        if not mutation:
            return
        if fetch_site == "cross-site":
            raise WebRequestError(HTTPStatus.FORBIDDEN, "Cross-site requests are not allowed.")

        origin = self.headers.get("Origin")
        referer = self.headers.get("Referer")
        candidate = origin or referer
        if not candidate:
            return
        parsed = urlparse(candidate)
        if parsed.scheme != "http" or not parsed.hostname:
            raise WebRequestError(HTTPStatus.FORBIDDEN, "Request origin is not allowed.")
        origin_port = parsed.port or 80
        effective_request_port = request_port or int(self.server.server_port)  # type: ignore[attr-defined]
        if parsed.hostname.lower().rstrip(".") != request_host.lower().rstrip(".") or origin_port != effective_request_port:
            raise WebRequestError(HTTPStatus.FORBIDDEN, "Request origin is not allowed.")

    def _handle_exception(self, exc: Exception) -> None:
        if isinstance(exc, WebRequestError):
            self._send_error(exc.status, str(exc))
        elif isinstance(exc, FileNotFoundError):
            self._send_error(HTTPStatus.NOT_FOUND, str(exc) or "Not found.")
        elif isinstance(exc, RuntimeError):
            self._send_error(HTTPStatus.CONFLICT, str(exc) or "The project is busy.")
        elif isinstance(exc, (ValueError, OSError, VideoEngineError)):
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc) or "Unable to handle request.")
        else:
            LOGGER.exception("Unhandled Vex Studio request failure")
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "Unexpected server error.")

    def do_GET(self) -> None:  # noqa: N802
        try:
            self._validate_request()
            path = self.parsed.path
            if path == "/api/health":
                self._send_json(self._health())
                return
            if path == "/studio-player.js":
                self._send_file_range(build_studio_player())
                return
            if path == "/api/projects":
                self._send_json({"projects": _project_summaries()})
                return
            if path == "/api/production":
                self._send_json({
                    "active": self.tasks.active_snapshot(PRODUCTION_SLOT),
                    "latest": self.tasks.latest_snapshot(PRODUCTION_SLOT),
                })
                return
            if path.startswith("/api/tasks/"):
                segments = [unquote(segment) for segment in path.removeprefix("/api/tasks/").split("/")]
                task_id = segments[0]
                if "/" in task_id or not task_id.startswith("task_") or segments[1:] not in ([], ["media"]):
                    self._send_error(404, "Task not found.")
                    return
                task = self.tasks.snapshot(task_id)
                if task is None:
                    self._send_error(404, "Task not found.")
                elif segments[1:]:
                    if task["kind"] != "production":
                        raise FileNotFoundError("Production run not found.")
                    self._send_file_range(production_media(task_id, task["result"]))
                else:
                    self._send_json(task)
                return
            if path.startswith("/api/projects/"):
                self._handle_project_get(path)
                return
            self._serve_static(path)
        except Exception as exc:  # noqa: BLE001
            self._handle_exception(exc)

    def do_HEAD(self) -> None:  # noqa: N802
        self.do_GET()

    def do_POST(self) -> None:  # noqa: N802
        try:
            self._validate_request(mutation=True)
            path = self.parsed.path
            if path == "/api/projects":
                self._create_project()
                return
            if path == "/api/production":
                request = production_request(self._read_json())
                task = self.tasks.start_production(request, _configured_model_name())
                self._send_json(self.tasks.snapshot(task.task_id) or task.to_dict(), 202)
                return
            if path.startswith("/api/tasks/") and path.endswith("/stop"):
                task_id = unquote(path.removeprefix("/api/tasks/").removesuffix("/stop"))
                self._read_json()
                self._send_json(self.tasks.stop_production(task_id))
                return
            if path.startswith("/api/tasks/") and path.endswith("/retry"):
                task_id = unquote(path.removeprefix("/api/tasks/").removesuffix("/retry"))
                self._read_json()
                self._send_json(self.tasks.retry_production(task_id, _configured_model_name()).to_dict(), 202)
                return
            if path.startswith("/api/projects/"):
                self._handle_project_post(path)
                return
            self.close_connection = True
            self._send_error(404, "Not found.")
        except Exception as exc:  # noqa: BLE001
            # A rejected request may still have unread bytes. Closing prevents those
            # bytes from being interpreted as the next keep-alive request.
            self.close_connection = True
            self._handle_exception(exc)

    def do_OPTIONS(self) -> None:  # noqa: N802
        try:
            self._validate_request()
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header("Allow", "GET, HEAD, POST, OPTIONS")
            self.send_header("Content-Length", "0")
            self.send_header("Cache-Control", "no-store")
            self._finish_headers()
        except Exception as exc:  # noqa: BLE001
            self._handle_exception(exc)

    def _method_not_allowed(self) -> None:
        self.close_connection = True
        self.send_response(HTTPStatus.METHOD_NOT_ALLOWED)
        self.send_header("Allow", "GET, HEAD, POST, OPTIONS")
        raw = json.dumps({"error": "Method not allowed.", "status": 405}).encode("utf-8")
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self._finish_headers()
        self._write_response(raw)

    do_DELETE = _method_not_allowed  # noqa: N815
    do_PATCH = _method_not_allowed  # noqa: N815
    do_PUT = _method_not_allowed  # noqa: N815

    def _health(self) -> dict[str, Any]:
        ffmpeg_available = bool(_which(str(config.FFMPEG_PATH or "ffmpeg")))
        ffprobe_available = bool(_probe_binary())
        return {
            "name": "Vex Studio",
            "version": config.VERSION,
            "provider": config.PROVIDER,
            "model": _configured_model_name(),
            "ffmpeg_available": ffmpeg_available,
            "ffprobe_available": ffprobe_available,
            "media_stack_ready": ffmpeg_available and ffprobe_available,
            "projects_dir": config.AGENT_PROJECTS_DIR,
            "local_only": True,
        }

    def _project_payload(self, state: ProjectState) -> dict[str, Any]:
        payload = _project_detail(state)
        payload["active_task"] = self.tasks.active_snapshot(state.project_id)
        payload["latest_task"] = self.tasks.latest_snapshot(state.project_id)
        return payload

    def _handle_project_get(self, path: str) -> None:
        segments = [unquote(segment) for segment in path.split("/") if segment]
        if len(segments) < 3:
            self._send_error(404, "Project not found.")
            return
        state = _project_state(segments[2])
        if len(segments) == 3:
            self._send_json(self._project_payload(state))
            return
        if len(segments) == 5 and segments[3] == "media":
            self._serve_media(state, segments[4])
            return
        if len(segments) == 4 and segments[3] == "production-live":
            self._send_json({"revision": state.revision, "timeline": live_timeline(state)})
            return
        if len(segments) == 5 and segments[3] in {"production-asset", "production-waveform"}:
            asset = waveform if segments[3] == "production-waveform" else live_media
            self._send_file_range(asset(state, segments[4]))
            return
        if len(segments) == 6 and segments[3] == "production-preview":
            cut, item = segments[4:]
            media = preview_image(state, cut, int(item)) if item.isascii() and item.isdecimal() else preview_speech(state, cut, item)
            self._send_file_range(media)
            return
        self._send_error(404, "Project resource not found.")

    def _handle_project_post(self, path: str) -> None:
        segments = [unquote(segment) for segment in path.split("/") if segment]
        if len(segments) != 4:
            self._send_error(404, "Project action not found.")
            return
        state = _project_state(segments[2])
        action = segments[3]
        if action == "replace-shot":
            self._replace_shot(state)
            return
        if action == "search-media":
            query = self._read_json().get("query")
            self._send_json({"results": search_media(state, query)})
            return
        if action == "production-live":
            self._send_json({"timeline": live_timeline(state, self._read_json())})
            return
        if action in {"production-attach", "production-preview", "production-propose", "production-commit",
                      "production-direction", "production-review"}:
            payload = self._read_json()

            def edit_production(task: WebTask) -> dict[str, Any]:
                current = _project_state(state.project_id)
                if action == "production-attach":
                    return attach_footage(current)
                if action == "production-direction":
                    return save_direction(current, payload)
                if action == "production-review":
                    return review_production(current, payload.get("cut"))
                if action == "production-commit":
                    return commit_production(current, payload.get("cut"), on_progress=lambda stage, progress:
                                             self.tasks.production_progress(task, stage, progress))
                self.tasks.production_progress(task, "Preparing the edit preview", 0.15)
                patch = propose_production(current, payload) if action == "production-propose" else payload
                result = preview_production(current, patch)
                return {"success": True, "message": "Preview ready to review.", **result}

            task = self.tasks.submit(state.project_id, action, "Preparing the cut", edit_production)
            self._send_json(self.tasks.snapshot(task.task_id) or task.to_dict(), 202)
            return
        if action in {"undo", "redo"}:
            self._read_json()

            def history_edit(_task: WebTask) -> dict[str, Any]:
                current = _project_state(state.project_id)
                result = TOOL_EXECUTORS[action]({}, current)
                return {"success": result["success"], "message": result["message"]}

            task = self.tasks.submit(
                state.project_id, action, "Undoing the last edit" if action == "undo" else "Restoring the last edit",
                history_edit,
            )
            self._send_json(self.tasks.snapshot(task.task_id) or task.to_dict(), 202)
            return
        if action == "chat":
            payload = self._read_json()
            message = str(payload.get("message") or "").strip()
            if not message:
                raise ValueError("Tell Vex what you want to change.")
            if len(message) > MAX_CHAT_LENGTH:
                raise ValueError(f"Message is too long; keep it under {MAX_CHAT_LENGTH:,} characters.")
            task = self.tasks.submit(
                state.project_id,
                "chat",
                "Understanding your edit",
                lambda current: _run_chat(self.tasks, current, message),
                request={"message": message},
            )
            self._send_json(self.tasks.snapshot(task.task_id) or task.to_dict(), 202)
            return
        if action == "rename":
            payload = self._read_json()
            name = _normalize_project_name(payload.get("name"))
            if not name:
                raise ValueError("Project name cannot be empty.")

            def rename() -> ProjectState:
                fresh_state = _project_state(state.project_id)
                fresh_state.project_name = name
                fresh_state.save()
                return fresh_state

            renamed_state = self.tasks.run_if_idle(state.project_id, rename)
            self._send_json(self._project_payload(renamed_state))
            return
        self._send_error(404, "Project action not found.")

    def _replace_shot(self, state: ProjectState) -> None:
        shots = studio_shots(state)
        if not shots:
            raise ValueError("Open the original cut of a generated film to replace its shots.")
        content_type = self.headers.get("Content-Type") or ""
        if content_type.split(";", 1)[0].strip().lower() != "multipart/form-data":
            raise WebRequestError(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Choose a replacement image or video.")
        length = self._content_length(MAX_UPLOAD_BYTES + MAX_MULTIPART_OVERHEAD_BYTES)
        fields, uploaded = _stream_multipart_form(
            self.rfile, length, content_type, extensions=IMAGE_TYPES | VIDEO_TYPES,
            fields_allowed=frozenset({"shot_id", "file"}),
        )
        if uploaded is None:
            raise ValueError("Choose a replacement image or video.")
        shot_id = fields.get("shot_id", "")
        number = next((shot["number"] for shot in shots if shot["id"] == shot_id), None)
        try:
            if number is None:
                raise ValueError("Choose a shot from the generated film.")
            task = self.tasks.submit(
                state.project_id, "shot_revision", f"Replacing shot {number}",
                lambda current: replace_studio_shot(
                    current.task_id, _project_state(state.project_id), shot_id, uploaded.path,
                    lambda stage, progress: self.tasks.production_progress(current, stage, progress),
                ),
                request={"shot_id": shot_id},
                on_finished=lambda: _clear_upload(uploaded.path),
            )
        except BaseException:
            _clear_upload(uploaded.path)
            raise
        self._send_json(self.tasks.snapshot(task.task_id) or task.to_dict(), 202)

    def _create_project(self) -> None:
        content_type = str(self.headers.get("Content-Type") or "")
        name = ""
        source_path = ""
        uploaded_path: Path | None = None
        try:
            media_type = content_type.split(";", 1)[0].strip().lower()
            if media_type == "multipart/form-data":
                length = self._content_length(MAX_UPLOAD_BYTES + MAX_MULTIPART_OVERHEAD_BYTES)
                fields, uploaded = _stream_multipart_form(self.rfile, length, content_type)
                name = _normalize_project_name(fields.get("name"))
                if uploaded is not None:
                    uploaded_path = uploaded.path
                    source_path = str(uploaded.path)
            elif media_type == "application/json":
                payload = self._read_json()
                source_path = str(payload.get("source_path") or "").strip()
                name = _normalize_project_name(payload.get("name"))
            else:
                raise WebRequestError(
                    HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                    "Content-Type must be application/json or multipart/form-data.",
                )
            if not source_path:
                raise ValueError("Choose a video file or provide a local video path.")
            source = Path(source_path).expanduser().resolve(strict=False)
            if not source.is_file():
                raise ValueError("That video file could not be found on this computer.")
            if source.suffix.lower() not in VIDEO_EXTENSIONS:
                raise ValueError("Vex Studio needs a supported video file.")
            state = create_project(
                str(source), name or None, config.PROVIDER, _configured_model_name(),
                managed_source=uploaded_path is not None,
            )
        finally:
            if uploaded_path is not None:
                _clear_upload(uploaded_path)
        self._send_json(self._project_payload(state), 201)

    def _serve_media(self, state: ProjectState, kind: str) -> None:
        preview = kind.endswith("-preview")
        kind = kind.removesuffix("-preview")
        if kind == "current":
            target = Path(state.working_file)
        elif kind == "source":
            target = Path(state.source_files[0]) if state.source_files else Path("")
        else:
            self._send_error(404, "Media not found.")
            return
        target = target.expanduser().resolve(strict=False)
        if not target.is_file():
            self._send_error(404, "The video file is not available yet.")
            return
        if preview:
            target = browser_preview(Path(state.working_dir), target, prepare=self.command != "HEAD")
        self._send_file_range(target)

    def _send_file_range(self, target: Path) -> None:
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        with target.open("rb") as stream:
            file_size = os.fstat(stream.fileno()).st_size
            range_header = self.headers.get("Range")
            start = 0
            end = max(0, file_size - 1)
            status = HTTPStatus.OK
            if range_header:
                try:
                    start, end = _parse_byte_range(range_header, file_size)
                except ValueError:
                    raw = json.dumps({"error": "Media range is not satisfiable.", "status": 416}).encode("utf-8")
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(raw)))
                    self.send_header("Content-Range", f"bytes */{file_size}")
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Cache-Control", "no-store")
                    self._finish_headers()
                    self._write_response(raw)
                    return
                status = HTTPStatus.PARTIAL_CONTENT
            length = 0 if file_size == 0 else end - start + 1
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-cache")
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self._finish_headers()
            if self.command == "HEAD" or length == 0:
                return
            stream.seek(start)
            remaining = length
            while remaining > 0:
                chunk = stream.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
                    self.close_connection = True
                    break
                remaining -= len(chunk)

    def _serve_static(self, path: str) -> None:
        relative = unquote(path.lstrip("/")) or "index.html"
        if relative.startswith("static/"):
            relative = relative[7:]
        if relative.startswith("api/") or ".." in Path(relative).parts:
            self._send_error(404, "Not found.")
            return
        if Path(relative).suffix and relative not in STATIC_FILES:
            self._send_error(HTTPStatus.NOT_FOUND, "Not found.")
            return
        root = Path(str(files("vex_web").joinpath("static"))).resolve(strict=False)
        target = (root / relative).resolve(strict=False)
        if target != root and root not in target.parents:
            self._send_error(404, "Not found.")
            return
        if not target.is_file():
            if Path(relative).suffix:
                self._send_error(HTTPStatus.NOT_FOUND, "Not found.")
                return
            target = root / "index.html"
        content_type = mimetypes.guess_type(str(target))[0] or "text/plain"
        raw = target.read_bytes()
        etag = f'"{hashlib.sha256(raw).hexdigest()[:24]}"'
        if self.headers.get("If-None-Match") == etag:
            self.send_response(HTTPStatus.NOT_MODIFIED)
            self.send_header("ETag", etag)
            self.send_header("Cache-Control", "no-cache")
            self._finish_headers()
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8" if content_type.startswith("text/") else content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("ETag", etag)
        self._finish_headers()
        self._write_response(raw)


def _which(command: str) -> str | None:
    candidate = Path(command).expanduser()
    if candidate.parent != Path("."):
        return str(candidate) if candidate.is_file() and os.access(candidate, os.X_OK) else None
    return shutil.which(command)


def _probe_binary() -> str | None:
    configured = str(config.FFMPEG_PATH or "")
    if configured:
        path = Path(configured)
        lowered_name = path.name.lower()
        if lowered_name.startswith("ffmpeg"):
            candidate = path.with_name(f"ffprobe{path.name[len('ffmpeg'):]}")
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
            found = _which(candidate.name)
            if found:
                return found
    return _which("ffprobe")


def _validate_bind_host(host: str) -> None:
    try:
        addresses = {
            item[4][0]
            for item in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
        }
    except socket.gaierror as exc:
        raise ValueError(f"Unable to resolve web host {host!r}.") from exc
    if not addresses or any(not ipaddress.ip_address(address).is_loopback for address in addresses):
        raise ValueError("Vex Studio can only bind to a loopback address such as 127.0.0.1 or localhost.")


def serve(*, host: str = "127.0.0.1", port: int = 5173, open_browser: bool = False) -> None:
    config.configure_runtime_logging()
    config.reload_settings()
    _validate_bind_host(host)
    task_manager = TaskManager(persist=True)
    try:
        server = VexHTTPServer((host, int(port)), task_manager)
    except Exception:
        task_manager.shutdown()
        raise
    display_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
    url = f"http://{display_host}:{server.server_port}"
    print(f"Vex Studio running at {url}")
    print("Press Ctrl-C to stop the local server.")
    if open_browser:
        opener = threading.Timer(0.35, lambda: webbrowser.open(url))
        opener.daemon = True
        opener.start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nVex Studio stopped.")
    finally:
        server.server_close()
        task_manager.shutdown()


__all__ = ["serve"]
