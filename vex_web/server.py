from __future__ import annotations

import json
import mimetypes
import os
import re
import tempfile
import threading
import traceback
import uuid
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote, urlparse

import config
from agent import VideoAgent
from engine import VideoEngineError
from job_runner import list_jobs
from plan_store import list_plan_records
from providers import get_provider
from state import ProjectState
from tools.creative_registry import latest_creative_runs


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".flv"}
MAX_REQUEST_BYTES = 2 * 1024 * 1024 * 1024
MAX_CHAT_LENGTH = 12_000
PROJECT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
UPLOAD_DIR = Path(tempfile.gettempdir()) / "vex-web-uploads"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_safe(value.to_dict())
    return str(value)


def _configured_model_name(provider_name: str | None = None) -> str:
    provider = config.normalize_provider_name(provider_name or config.PROVIDER)
    names = {
        "gemini": config.GEMINI_MODEL,
        "claude": config.CLAUDE_MODEL,
        "ollama": config.local_llm_model("ollama"),
        "lmstudio": config.local_llm_model("lmstudio"),
        "llama_cpp": config.local_llm_model("llama_cpp"),
        "openai_compatible": config.local_llm_model("openai_compatible"),
    }
    return str(names.get(provider) or "configured model")


def _format_duration(value: Any) -> str:
    try:
        seconds = max(0, int(round(float(value or 0))))
    except (TypeError, ValueError):
        return "—"
    minutes, remainder = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{remainder:02d}"
    return f"{minutes}:{remainder:02d}"


def _format_bytes(value: Any) -> str:
    try:
        size = float(value or 0)
    except (TypeError, ValueError):
        return "—"
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
    return ProjectState.load(normalized)


def _parse_multipart_form(body: bytes, content_type: str) -> tuple[dict[str, str], dict[str, tuple[str, bytes]]]:
    """Parse the small, browser-generated multipart shape used by project upload."""
    match = re.search(r"boundary=(?:\"([^\"]+)\"|([^;]+))", content_type, flags=re.IGNORECASE)
    if not match:
        raise ValueError("Upload is missing its multipart boundary.")
    boundary = (match.group(1) or match.group(2) or "").strip().encode("utf-8")
    if not boundary:
        raise ValueError("Upload is missing its multipart boundary.")
    fields: dict[str, str] = {}
    files: dict[str, tuple[str, bytes]] = {}
    delimiter = b"--" + boundary
    for raw_part in body.split(delimiter)[1:]:
        part = raw_part
        if part.startswith(b"\r\n"):
            part = part[2:]
        if part.endswith(b"--\r\n"):
            part = part[:-4]
        elif part.endswith(b"\r\n"):
            part = part[:-2]
        if not part:
            continue
        raw_headers, separator, payload = part.partition(b"\r\n\r\n")
        if not separator:
            continue
        header_map: dict[str, str] = {}
        for line in raw_headers.split(b"\r\n"):
            key, marker, value = line.partition(b":")
            if marker:
                header_map[key.decode("latin-1").lower().strip()] = value.decode("latin-1").strip()
        disposition = header_map.get("content-disposition", "")
        name_match = re.search(r'name="([^"]+)"', disposition)
        if not name_match:
            continue
        name = name_match.group(1)
        filename_match = re.search(r'filename="([^"]*)"', disposition)
        if payload.endswith(b"\r\n"):
            payload = payload[:-2]
        if filename_match and filename_match.group(1):
            files[name] = (filename_match.group(1), payload)
        else:
            fields[name] = payload.decode("utf-8", errors="replace")
    return fields, files


def _project_summary(item: dict[str, Any]) -> dict[str, Any]:
    project_id = str(item.get("project_id") or "")
    payload = dict(item)
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
            "duration_sec": float(metadata.get("duration_sec") or 0),
            "resolution": f"{metadata.get('width', 0)}×{metadata.get('height', 0)}" if metadata.get("width") else "—",
            "timeline_ops": len(state.timeline),
            "has_preview": bool(state.working_file and Path(state.working_file).is_file()),
            "working_file": state.working_file,
        }
    )
    return payload


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
    creative_runs = latest_creative_runs(state.working_dir, limit=12)
    jobs = [record.to_dict() for record in list_jobs(state.working_dir, limit=20)]
    plans = [record.to_dict() for record in list_plan_records(state.working_dir, limit=20)]
    source_name = Path(state.source_files[0]).name if state.source_files else "Untitled media"
    trace = (state.artifacts or {}).get("latest_agent_trace")
    return {
        "project": {
            "project_id": state.project_id,
            "short_id": state.project_id[:8],
            "project_name": state.project_name,
            "source_name": source_name,
            "source_path": state.source_files[0] if state.source_files else "",
            "created_at": state.created_at,
            "updated_at": state.updated_at,
            "updated_label": _relative_time(state.updated_at),
            "working_file": state.working_file,
            "output_dir": state.output_dir,
            "duration": _format_duration(metadata.get("duration_sec")),
            "duration_sec": float(metadata.get("duration_sec") or 0),
            "resolution": f"{metadata.get('width', 0)}×{metadata.get('height', 0)}" if metadata.get("width") else "—",
            "fps": metadata.get("fps") or "—",
            "size": _format_bytes(metadata.get("size_bytes")),
            "timeline_ops": len(state.timeline),
            "redo_count": len(state.redo_stack),
            "provider": state.provider or config.PROVIDER,
            "model": state.model or _configured_model_name(state.provider),
        },
        "media": {
            "current": f"/api/projects/{state.project_id}/media/current",
            "source": f"/api/projects/{state.project_id}/media/source",
            "available": bool(state.working_file and Path(state.working_file).is_file()),
        },
        "chat": _chat_history(state),
        "timeline": _timeline_rows(state),
        "artifacts": _artifact_summary(state),
        "creative_runs": _json_safe(creative_runs),
        "jobs": _json_safe(jobs),
        "plans": _json_safe(plans),
        "latest_trace": _json_safe(trace if isinstance(trace, dict) else {"events": []}),
    }


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
        }


class TaskManager:
    def __init__(self) -> None:
        self._tasks: dict[str, WebTask] = {}
        self._active_projects: dict[str, str] = {}
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="vex-web")

    def get(self, task_id: str) -> WebTask | None:
        with self._lock:
            return self._tasks.get(task_id)

    def active_for_project(self, project_id: str) -> WebTask | None:
        task_id = self._active_projects.get(project_id)
        return self._tasks.get(task_id) if task_id else None

    def submit(self, project_id: str, kind: str, label: str, work: Callable[[WebTask], dict[str, Any]]) -> WebTask:
        with self._lock:
            active = self.active_for_project(project_id)
            if active and active.status in {"queued", "running"}:
                raise RuntimeError(f"Vex is already working on {active.label.lower()}.")
            task = WebTask(task_id=f"task_{uuid.uuid4().hex[:16]}", project_id=project_id, kind=kind, label=label)
            self._tasks[task.task_id] = task
            self._active_projects[project_id] = task.task_id
        self._executor.submit(self._run, task, work)
        return task

    def _run(self, task: WebTask, work: Callable[[WebTask], dict[str, Any]]) -> None:
        with self._lock:
            task.status = "running"
            task.started_at = _now()
            task.message = task.label
        try:
            result = work(task)
            with self._lock:
                task.result = dict(result or {})
                task.status = "succeeded" if bool(task.result.get("success", True)) else "failed"
                task.message = str(task.result.get("message") or ("Completed" if task.status == "succeeded" else "Could not complete the request."))
                task.finished_at = _now()
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                task.status = "failed"
                task.error = str(exc)
                task.message = str(exc) or "The task failed."
                task.finished_at = _now()
                task.events.append({"kind": "system", "title": "Task failed", "detail": task.message, "status": "error"})
        finally:
            with self._lock:
                if self._active_projects.get(task.project_id) == task.task_id:
                    self._active_projects.pop(task.project_id, None)

    def append_event(self, task: WebTask, event: Any) -> None:
        payload = event.to_dict() if hasattr(event, "to_dict") else dict(event or {})
        with self._lock:
            task.events.append(_json_safe(payload))
            task.events = task.events[-120:]
            task.message = str(payload.get("detail") or payload.get("title") or task.message)

    def append_stream(self, task: WebTask, chunk: str) -> None:
        with self._lock:
            task.stream = (task.stream + str(chunk or ""))[-20_000:]


TASKS = TaskManager()


def _run_chat(task: WebTask, message: str) -> dict[str, Any]:
    state = _project_state(task.project_id)
    provider = get_provider(state.provider or config.PROVIDER)
    agent = VideoAgent(state, provider)
    response = agent.run(
        message,
        stream_callback=lambda chunk: TASKS.append_stream(task, chunk),
        trace_callback=lambda event: TASKS.append_event(task, event),
        tool_callback=lambda phase, tool, success: TASKS.append_event(
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


class VexRequestHandler(BaseHTTPRequestHandler):
    server_version = "VexStudio/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        return

    @property
    def parsed(self):
        return urlparse(self.path)

    def _send_json(self, payload: Any, status: int = 200) -> None:
        raw = json.dumps(_json_safe(payload), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _send_error(self, status: int, message: str) -> None:
        self._send_json({"error": message, "status": status}, status)

    def _read_body(self) -> bytes:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError as exc:
            raise ValueError("Invalid Content-Length header.") from exc
        if length < 0 or length > MAX_REQUEST_BYTES:
            raise ValueError("Request is too large.")
        return self.rfile.read(length)

    def _read_json(self) -> dict[str, Any]:
        body = self._read_body()
        try:
            payload = json.loads(body.decode("utf-8")) if body else {}
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Request body must be valid JSON.") from exc
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object.")
        return payload

    def do_GET(self) -> None:  # noqa: N802
        try:
            path = self.parsed.path
            if path == "/api/health":
                self._send_json(self._health())
                return
            if path == "/api/projects":
                self._send_json({"projects": [_project_summary(item) for item in ProjectState.list_projects()]})
                return
            if path.startswith("/api/tasks/"):
                task = TASKS.get(unquote(path.split("/api/tasks/", 1)[1]))
                if task is None:
                    self._send_error(404, "Task not found.")
                else:
                    self._send_json(task.to_dict())
                return
            if path.startswith("/api/projects/"):
                self._handle_project_get(path)
                return
            self._serve_static(path)
        except FileNotFoundError as exc:
            self._send_error(404, str(exc) or "Not found.")
        except (ValueError, OSError, VideoEngineError) as exc:
            self._send_error(400, str(exc) or "Unable to handle request.")
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._send_error(500, str(exc) or "Unexpected server error.")

    def do_POST(self) -> None:  # noqa: N802
        try:
            path = self.parsed.path
            if path == "/api/projects":
                self._create_project()
                return
            if path.startswith("/api/projects/"):
                self._handle_project_post(path)
                return
            self._send_error(404, "Not found.")
        except FileNotFoundError as exc:
            self._send_error(404, str(exc) or "Not found.")
        except RuntimeError as exc:
            self._send_error(409, str(exc) or "The project is busy.")
        except (ValueError, OSError, VideoEngineError) as exc:
            self._send_error(400, str(exc) or "Unable to handle request.")
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._send_error(500, str(exc) or "Unexpected server error.")

    def _health(self) -> dict[str, Any]:
        ffmpeg_available = bool(config.FFMPEG_PATH and (Path(config.FFMPEG_PATH).is_file() or _which(config.FFMPEG_PATH)))
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

    def _handle_project_get(self, path: str) -> None:
        segments = [unquote(segment) for segment in path.split("/") if segment]
        if len(segments) < 3:
            self._send_error(404, "Project not found.")
            return
        state = _project_state(segments[2])
        if len(segments) == 3:
            self._send_json(_project_detail(state))
            return
        if len(segments) == 5 and segments[3] == "media":
            self._serve_media(state, segments[4])
            return
        self._send_error(404, "Project resource not found.")

    def _handle_project_post(self, path: str) -> None:
        segments = [unquote(segment) for segment in path.split("/") if segment]
        if len(segments) != 4:
            self._send_error(404, "Project action not found.")
            return
        state = _project_state(segments[2])
        action = segments[3]
        if action == "chat":
            payload = self._read_json()
            message = str(payload.get("message") or "").strip()
            if not message:
                raise ValueError("Tell Vex what you want to change.")
            if len(message) > MAX_CHAT_LENGTH:
                raise ValueError(f"Message is too long; keep it under {MAX_CHAT_LENGTH:,} characters.")
            task = TASKS.submit(
                state.project_id,
                "chat",
                "Understanding your edit",
                lambda current: _run_chat(current, message),
            )
            self._send_json(task.to_dict(), 202)
            return
        if action == "rename":
            payload = self._read_json()
            name = str(payload.get("name") or "").strip()
            if not name:
                raise ValueError("Project name cannot be empty.")
            state.project_name = name[:120]
            state.save()
            self._send_json(_project_detail(state))
            return
        self._send_error(404, "Project action not found.")

    def _create_project(self) -> None:
        content_type = str(self.headers.get("Content-Type") or "")
        name = ""
        source_path = ""
        uploaded_path: Path | None = None
        if content_type.startswith("multipart/form-data"):
            body = self._read_body()
            fields, uploaded_files = _parse_multipart_form(body, content_type)
            name = str(fields.get("name") or "").strip()
            file_field = uploaded_files.get("file")
            if file_field is not None:
                filename, file_data = file_field
                filename = Path(filename).name
                suffix = Path(filename).suffix.lower()
                if suffix not in VIDEO_EXTENSIONS:
                    raise ValueError("Choose a supported video file: mp4, mov, webm, mkv, or m4v.")
                UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
                uploaded_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
                uploaded_path.write_bytes(file_data)
                source_path = str(uploaded_path)
        else:
            payload = self._read_json()
            source_path = str(payload.get("source_path") or "").strip()
            name = str(payload.get("name") or "").strip()
        if not source_path:
            raise ValueError("Choose a video file or provide a local video path.")
        source = Path(source_path).expanduser().resolve(strict=False)
        if not source.is_file():
            raise ValueError("That video file could not be found on this computer.")
        if source.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError("Vex Studio needs a supported video file.")
        from main import create_project

        try:
            state = create_project(str(source), name[:120] if name else None, config.PROVIDER, _configured_model_name())
        finally:
            if uploaded_path and uploaded_path.exists():
                try:
                    uploaded_path.unlink()
                except OSError:
                    pass
        self._send_json(_project_detail(state), 201)

    def _serve_media(self, state: ProjectState, kind: str) -> None:
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
        self._send_file_range(target)

    def _send_file_range(self, target: Path) -> None:
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        file_size = target.stat().st_size
        range_header = self.headers.get("Range")
        start = 0
        end = file_size - 1
        status = HTTPStatus.OK
        if range_header and range_header.startswith("bytes="):
            raw_range = range_header[6:].split(",", 1)[0].strip()
            raw_start, _, raw_end = raw_range.partition("-")
            try:
                if raw_start:
                    start = int(raw_start)
                elif raw_end:
                    start = max(0, file_size - int(raw_end))
                if raw_end and raw_start:
                    end = int(raw_end)
            except ValueError as exc:
                raise ValueError("Invalid media range.") from exc
            if start < 0 or start >= file_size or end < start:
                self._send_error(416, "Media range is not satisfiable.")
                return
            end = min(end, file_size - 1)
            status = HTTPStatus.PARTIAL_CONTENT
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache")
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()
        with target.open("rb") as stream:
            stream.seek(start)
            remaining = length
            while remaining > 0:
                chunk = stream.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def _serve_static(self, path: str) -> None:
        relative = unquote(path.lstrip("/")) or "index.html"
        if relative.startswith("static/"):
            relative = relative[7:]
        if relative.startswith("api/") or ".." in Path(relative).parts:
            self._send_error(404, "Not found.")
            return
        root = Path(str(files("vex_web").joinpath("static")))
        target = (root / relative).resolve(strict=False)
        if target != root and root not in target.parents:
            self._send_error(404, "Not found.")
            return
        if not target.is_file():
            target = root / "index.html"
        content_type = mimetypes.guess_type(str(target))[0] or "text/plain"
        raw = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8" if content_type.startswith("text/") else content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(raw)


def _which(command: str) -> str | None:
    if os.path.sep in command:
        return command if Path(command).is_file() else None
    for entry in os.getenv("PATH", "").split(os.pathsep):
        candidate = Path(entry) / command
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _probe_binary() -> str | None:
    configured = str(config.FFMPEG_PATH or "")
    if configured:
        path = Path(configured)
        if path.name.lower().startswith("ffmpeg"):
            candidate = path.with_name(path.name.replace("ffmpeg", "ffprobe", 1))
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
            found = _which(candidate.name)
            if found:
                return found
    return _which("ffprobe")


def serve(*, host: str = "127.0.0.1", port: int = 5173, open_browser: bool = False) -> None:
    config.configure_runtime_logging()
    config.reload_settings()
    server = ThreadingHTTPServer((host, int(port)), VexRequestHandler)
    url = f"http://{host}:{server.server_port}"
    print(f"Vex Studio running at {url}")
    print("Press Ctrl-C to stop the local server.")
    if open_browser:
        threading.Timer(0.35, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nVex Studio stopped.")
    finally:
        server.server_close()
        TASKS._executor.shutdown(wait=False, cancel_futures=True)


__all__ = ["serve"]
