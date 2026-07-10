from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import sys
import threading
import time
import uuid
from collections.abc import Mapping
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Keep CLI startup stable on memory-constrained Windows shells before NumPy-backed
# media modules initialize OpenBLAS thread pools. Users can override these.
for _thread_env in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_thread_env, "1")
del _thread_env

import typer
from agent_trace import (
    TraceEvent,
    format_trace_duration,
    render_trace_table,
    trace_status_label,
    trace_status_style,
    truncate_trace_text,
)
from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

import config
from agent import AgentLoopError, VideoAgent
from tools.creative_registry import latest_creative_runs
from engine import check_disk_space, estimate_output_size, export as export_media, probe_video
from evaluation_harness import EvaluationReport, run_intent_evaluation, write_evaluation_report
from intent_compiler import compile_intent
from job_runner import JobRecord, JobRunnerError, create_tool_job, list_jobs, run_tool_job
from nle_interop import NLEExportResult, SUPPORTED_NLE_FORMATS, export_nle_bundle
from plan_store import (
    PlanRecord,
    PlanStoreError,
    claim_plan_record,
    create_plan_record,
    edit_plan_from_record,
    list_plan_records,
    mark_plan_record,
    sanitize_plan_result,
)
from plugin_api import PluginManifest, default_plugin_dirs, discover_plugins
from providers import get_provider
from tools.path_security import TRUSTED_OUTPUT_PATH_TOKEN
from sources import download_youtube_video, extract_youtube_url, normalize_source_url
from state import ProjectState, utc_now_iso
from tools import TOOL_CONTRACTS, TOOL_EXECUTORS
from tools.export import load_presets
from vex_runtime.configuration import ConfigurationError, write_config_template
from vex_runtime.hyperframes import RuntimeInstallError, install_hyperframes_runtime
from vex_runtime.transcription import (
    TranscriptionInstallError,
    install_transcription_dependencies,
)

app = typer.Typer(help="Vex - AI-powered video editing agent.")
renderers_app = typer.Typer(help="Renderer diagnostics and guidance.")
renderers_install_app = typer.Typer(help="Install version-locked renderer runtimes.")
setup_app = typer.Typer(help="Create local Vex configuration.")
app.add_typer(renderers_app, name="renderers")
renderers_app.add_typer(renderers_install_app, name="install")
app.add_typer(setup_app, name="setup")
console = Console()
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".flv"}
LOAD_COMMAND_RE = re.compile(r"^(?:load|open|use|switch(?:\s+to)?)\s+(.+)$", re.IGNORECASE)
CLI_ACCENT = "bright_cyan"
CLI_SECONDARY = "magenta"
CLI_SUCCESS = "green"
CLI_WARNING = "yellow"
CLI_ERROR = "red"
LIVE_REFRESH_PER_SECOND = 8


def initialize_runtime(*, require_provider: bool = True) -> None:
    config.configure_runtime_logging()
    config.validate_config(require_provider=require_provider)


def print_banner(model_name: str) -> None:
    wordmark = Text(
        " __      ________  __   __\n"
        " \\ \\    / /  ____| \\ \\ / /\n"
        "  \\ \\  / /| |__     \\ V /\n"
        "   \\ \\/ / |  __|     > <\n"
        "    \\  /  | |____   / . \\\n"
        "     \\/   |______| /_/ \\_\\",
        style=f"bold {CLI_ACCENT}",
    )
    details = Table.grid(padding=(0, 2))
    details.add_row(Text("Version", style="dim"), Text(config.VERSION, style="bold white"))
    details.add_row(Text("Provider", style="dim"), Text(config.PROVIDER, style=CLI_SECONDARY))
    details.add_row(Text("Model", style="dim"), Text(model_name, style="white"))
    details.add_row(Text("Mode", style="dim"), Text("conversational video editing", style="white"))

    body = Group(
        Align.center(wordmark),
        Align.center(Text("Terminal-first AI video editor", style="bold white")),
        Text(""),
        details,
        Text(""),
        Align.center(Text("Drop a video path or YouTube URL, or ask for an edit in plain English.", style="dim")),
    )
    console.print(
        Panel.fit(
            body,
            border_style=CLI_ACCENT,
            title="[bold]Vex[/]",
            subtitle="[dim]/help for commands[/]",
            box=box.DOUBLE_EDGE,
            padding=(1, 2),
        )
    )


def create_provider(show_banner: bool = True):
    initialize_runtime()
    provider = get_provider(config.PROVIDER)
    if show_banner:
        print_banner(provider.model_name)
    return provider


@app.callback(invoke_without_command=True)
def app_callback(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-v", is_eager=True, help="Show version and exit."),
) -> None:
    if version:
        console.print(f"Vex v{config.VERSION}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        provider = create_provider()
        projects = ProjectState.list_projects()
        state = None
        resume_project = select_auto_resume_project(projects, Path.cwd())
        if resume_project is not None:
            state = ProjectState.load(resume_project["project_id"])
            resume_body = Table.grid(padding=(0, 2))
            resume_body.add_row(Text("Project", style="dim"), Text(state.project_name, style="bold white"))
            resume_body.add_row(Text("ID", style="dim"), Text(state.project_id[:8], style=CLI_ACCENT))
            resume_body.add_row(Text("Last edited", style="dim"), Text(f"{format_relative_time(state.updated_at)} ago"))
            console.print(
                Panel(
                    resume_body,
                    title="Resuming Project",
                    border_style=CLI_ACCENT,
                    box=box.ROUNDED,
                    padding=(1, 2),
                )
            )
        run_repl(state, provider)
        raise typer.Exit()


def format_bytes(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def format_elapsed(seconds: float | int) -> str:
    seconds_int = max(int(seconds), 0)
    if seconds_int < 60:
        return f"{seconds_int}s"
    minutes = seconds_int // 60
    seconds_remainder = seconds_int % 60
    if minutes < 60:
        return f"{minutes}m {seconds_remainder:02d}s"
    hours = minutes // 60
    minutes_remainder = minutes % 60
    return f"{hours}h {minutes_remainder:02d}m"


def format_media_duration(seconds: float | int | str | None) -> str:
    try:
        value = float(seconds or 0.0)
    except (TypeError, ValueError):
        return "unknown"
    if value <= 0:
        return "unknown"
    total = int(round(value))
    hours = total // 3600
    minutes = (total % 3600) // 60
    remaining = total % 60
    if hours:
        return f"{hours}:{minutes:02d}:{remaining:02d}"
    return f"{minutes}:{remaining:02d}"


def format_resolution(metadata: dict) -> str:
    try:
        width = int(metadata.get("width") or 0)
        height = int(metadata.get("height") or 0)
    except (TypeError, ValueError):
        return "unknown"
    if width and height:
        return f"{width}x{height}"
    if height:
        return f"{height}p"
    return "unknown"


def format_fps(metadata: dict) -> str:
    try:
        fps_value = float(metadata.get("fps") or 0.0)
    except (TypeError, ValueError):
        return "unknown"
    if fps_value <= 0:
        return "unknown"
    if fps_value.is_integer():
        return f"{int(fps_value)}fps"
    return f"{fps_value:.2f}fps"


def format_short_path(path: str | Path, *, max_length: int = 72) -> str:
    value = str(path or "").strip()
    if not value:
        return "unknown"
    if len(value) <= max_length:
        return value
    path_obj = Path(value)
    parent = path_obj.parent
    if parent and str(parent) not in {"", "."}:
        shortened = f"...{os.sep}{parent.name}{os.sep}{path_obj.name}"
        if len(shortened) <= max_length:
            return shortened
    return "..." + value[-max(8, max_length - 3):]


def format_relative_time(iso_timestamp: str) -> str:
    try:
        timestamp = datetime.fromisoformat(iso_timestamp)
    except ValueError:
        return "unknown"
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    delta = datetime.now(timezone.utc) - timestamp.astimezone(timezone.utc)
    seconds = max(int(delta.total_seconds()), 0)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    if seconds < 86400:
        return f"{seconds // 3600}h"
    return f"{seconds // 86400}d"


def _format_registry_time(value: object) -> str:
    timestamp = str(value or "").strip()
    if not timestamp:
        return "-"
    return format_relative_time(timestamp)


def _format_quality_score(value: object) -> str:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return "-"
    if score <= 1.0:
        return f"{score:.2f}"
    return f"{score / 100.0:.2f}"


def _creative_run_summary(record: dict, summary: dict) -> str:
    feature = str(record.get("feature") or "")
    if feature == "auto_shorts":
        return (
            f"{int(summary.get('count') or 0)} shorts"
            f" for {summary.get('target_platform') or 'platform'}"
            f" ({int(summary.get('candidate_count') or 0)} candidates)"
        )
    if feature == "auto_visuals":
        return (
            f"{int(summary.get('count') or 0)} visuals"
            f" via {summary.get('renderer') or 'auto'}"
            f" / {summary.get('style_pack') or 'auto'}"
        )
    if feature == "auto_color_grade":
        resolved = summary.get("resolved_look") or summary.get("look") or "auto"
        return f"{resolved} grade, intensity {summary.get('intensity', '-')}"
    if summary:
        return ", ".join(f"{key}={value}" for key, value in list(summary.items())[:4])
    return "creative run"


def strip_wrapping_quotes(value: str) -> str:
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {'"', "'"}:
        return stripped[1:-1]
    return stripped


def is_video_path(path: str) -> bool:
    candidate = os.path.abspath(strip_wrapping_quotes(path))
    return os.path.isfile(candidate) and Path(candidate).suffix.lower() in VIDEO_EXTENSIONS


def looks_like_video_path(path: str) -> bool:
    candidate = strip_wrapping_quotes(path)
    return bool(candidate) and Path(candidate).suffix.lower() in VIDEO_EXTENSIONS


def is_path_within_directory(path: str, directory: str | Path) -> bool:
    try:
        target = Path(path).resolve(strict=False)
        base = Path(directory).resolve(strict=False)
        return target == base or target.is_relative_to(base)
    except (OSError, ValueError):
        return False


def project_belongs_to_launch_directory(project: dict, launch_dir: str | Path) -> bool:
    source_file = str(project.get("source_file") or "").strip()
    if source_file and is_path_within_directory(source_file, launch_dir):
        return True
    return False


def select_auto_resume_project(projects: list[dict], launch_dir: str | Path) -> dict | None:
    candidates = [
        project
        for project in projects
        if project_belongs_to_launch_directory(project, launch_dir)
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


def detect_video_path(user_input: str) -> str | None:
    full_candidate = strip_wrapping_quotes(user_input)
    if is_video_path(full_candidate):
        return os.path.abspath(full_candidate)

    for match in re.findall(r'"([^"]+)"|\'([^\']+)\'', user_input):
        candidate = next((group for group in match if group), "")
        if candidate and is_video_path(candidate):
            return os.path.abspath(strip_wrapping_quotes(candidate))

    for token in user_input.split():
        candidate = strip_wrapping_quotes(token)
        if is_video_path(candidate):
            return os.path.abspath(candidate)
    return None


def is_loaded_source(state: ProjectState | None, candidate_path: str) -> bool:
    if state is None or not state.source_files:
        return False
    current_source = os.path.abspath(state.source_files[0])
    return os.path.normcase(current_source) == os.path.normcase(os.path.abspath(candidate_path))


def find_project_for_source(video_path: str) -> ProjectState | None:
    target = os.path.normcase(os.path.abspath(video_path))
    for project in ProjectState.list_projects():
        source_file = project.get("source_file", "")
        if source_file and os.path.normcase(os.path.abspath(source_file)) == target:
            return ProjectState.load(project["project_id"])
    return None


def is_loaded_source_url(state: ProjectState | None, candidate_url: str) -> bool:
    if state is None:
        return False
    current_url = str((state.artifacts or {}).get("source_url") or "").strip()
    return bool(current_url) and current_url == normalize_source_url(candidate_url)


def find_project_for_source_url(url: str) -> ProjectState | None:
    normalized = normalize_source_url(url)
    for project in ProjectState.list_projects():
        loaded = ProjectState.load(project["project_id"])
        if str((loaded.artifacts or {}).get("source_url") or "").strip() == normalized:
            return loaded
    return None


def parse_load_source_command(command: str) -> tuple[str, str] | None:
    stripped = command.strip()
    if not stripped:
        return None

    if is_video_path(stripped):
        return ("path", os.path.abspath(strip_wrapping_quotes(stripped)))

    bare_url = extract_youtube_url(stripped)
    if bare_url and normalize_source_url(stripped) == normalize_source_url(bare_url):
        return ("url", bare_url)

    match = LOAD_COMMAND_RE.match(stripped)
    if not match:
        return None
    target = match.group(1).strip()
    if is_video_path(target):
        return ("path", os.path.abspath(strip_wrapping_quotes(target)))
    if looks_like_video_path(target):
        return ("missing_path", os.path.abspath(strip_wrapping_quotes(target)))
    target_url = extract_youtube_url(target)
    if target_url and normalize_source_url(target) == normalize_source_url(target_url):
        return ("url", target_url)
    return None


def format_loaded_state_message(state: ProjectState, *, already_loaded: bool) -> str:
    metadata = state.metadata or {}
    duration_sec = float(metadata.get("duration_sec") or 0.0)
    width = int(metadata.get("width") or 0)
    height = int(metadata.get("height") or 0)
    fps_value = float(metadata.get("fps") or 0.0)
    resolution = f"{height}p" if height else f"{width}x{height}" if width or height else "unknown resolution"
    if fps_value.is_integer():
        fps_text = f"{int(fps_value)}fps"
    else:
        fps_text = f"{fps_value:.2f}fps"
    prefix = "Already loaded." if already_loaded else "Loaded."
    return f"{prefix} {duration_sec:.2f}s, {resolution}, {fps_text}. Ready."


def _artifact_rows(state: ProjectState) -> list[tuple[str, str]]:
    artifacts = state.artifacts or {}
    rows: list[tuple[str, str]] = []
    latest_transcript = artifacts.get("latest_transcript")
    if isinstance(latest_transcript, dict):
        rows.append(
            (
                "Transcript",
                f"{latest_transcript.get('segment_count', 0)} segments, "
                f"{latest_transcript.get('word_count', 0)} words",
            )
        )
    latest_auto_visuals = artifacts.get("latest_auto_visuals")
    if isinstance(latest_auto_visuals, dict):
        rows.append(
            (
                "Auto visuals",
                f"{latest_auto_visuals.get('count', 0)} inserts, "
                f"{latest_auto_visuals.get('renderer', 'auto')} / "
                f"{latest_auto_visuals.get('style_pack', 'auto')}",
            )
        )
    latest_auto_broll = artifacts.get("latest_auto_broll")
    if isinstance(latest_auto_broll, dict):
        rows.append(("Auto b-roll", f"{latest_auto_broll.get('count', 0)} inserts"))
    latest_auto_shorts = artifacts.get("latest_auto_shorts")
    if isinstance(latest_auto_shorts, dict):
        rows.append(("Auto shorts", f"{latest_auto_shorts.get('count', 0)} clips"))
    latest_auto_color_grade = artifacts.get("latest_auto_color_grade")
    if isinstance(latest_auto_color_grade, dict):
        rows.append(
            (
                "Color grade",
                str(latest_auto_color_grade.get("resolved_look", latest_auto_color_grade.get("look", "auto"))),
            )
        )
    latest_added_song = artifacts.get("latest_added_song")
    if isinstance(latest_added_song, dict):
        rows.append(
            (
                "Song mix",
                f"{latest_added_song.get('selected_skill_id', 'song')} "
                f"({float((latest_added_song.get('qa') or {}).get('score') or 0.0):.2f})",
            )
        )
    latest_generated_video = artifacts.get("latest_generated_video")
    if isinstance(latest_generated_video, dict):
        rows.append(
            (
                "Generated video",
                f"{format_media_duration(latest_generated_video.get('duration_sec'))}, "
                f"{'rendered' if latest_generated_video.get('rendered') else 'project only'}",
            )
        )
    pending_encode = artifacts.get("pending_encode")
    if isinstance(pending_encode, dict) and pending_encode.get("plan_id"):
        rows.append(("Encode", f"pending plan {str(pending_encode.get('plan_id'))[:8]}"))
    latest_agent_trace = artifacts.get("latest_agent_trace")
    if isinstance(latest_agent_trace, dict):
        rows.append(("Agent trace", f"{len(latest_agent_trace.get('events') or [])} steps recorded"))
    return rows


def render_artifacts_table(state: ProjectState):
    rows = _artifact_rows(state)
    if not rows:
        return Text("No generated artifacts yet.", style="dim")
    table = Table(box=box.SIMPLE_HEAVY, show_header=False, expand=True, pad_edge=False)
    table.add_column("Artifact", style=CLI_SECONDARY, no_wrap=True)
    table.add_column("State", ratio=1)
    for label, value in rows:
        table.add_row(label, truncate_trace_text(value, 92))
    return table


def render_recent_timeline_table(state: ProjectState, *, max_items: int = 6):
    if not state.timeline:
        return Text("No timeline operations yet.", style="dim")
    table = Table(box=box.SIMPLE_HEAVY, expand=True, show_edge=False, pad_edge=False)
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("Operation", style="bold", ratio=1)
    table.add_column("Detail", style="dim", ratio=2)
    table.add_column("Time", style="dim", justify="right", width=8)
    start_index = max(len(state.timeline) - max_items, 0)
    for index, op in enumerate(state.timeline[start_index:], start=start_index + 1):
        description = str(op.get("description") or "")
        params = op.get("params") or {}
        if not description and isinstance(params, dict):
            description = ", ".join(
                f"{key}={value}"
                for key, value in list(params.items())[:4]
                if not str(key).endswith("_label") and key not in {"file_paths"}
            )
        table.add_row(
            str(index),
            str(op.get("op") or "operation"),
            truncate_trace_text(description or "-", 100),
            str(op.get("timestamp") or "")[11:19],
        )
    return table


def render_project_dashboard(state: ProjectState):
    metadata = state.metadata or {}
    source_url = str((state.artifacts or {}).get("source_url") or "").strip()
    try:
        size_bytes = int(metadata.get("size_bytes") or 0)
    except (TypeError, ValueError):
        size_bytes = 0

    media = Table.grid(padding=(0, 2))
    media.add_row(Text("Project ID", style="dim"), Text(state.project_id[:8], style="bold white"))
    media.add_row(Text("Media", style="dim"), Text(Path(state.source_files[0]).name if state.source_files else "none"))
    if source_url:
        media.add_row(Text("Source URL", style="dim"), Text(truncate_trace_text(source_url, 76), style="cyan"))
    media.add_row(Text("Duration", style="dim"), Text(format_media_duration(metadata.get("duration_sec"))))
    media.add_row(Text("Frame", style="dim"), Text(f"{format_resolution(metadata)} @ {format_fps(metadata)}"))
    media.add_row(Text("Size", style="dim"), Text(format_bytes(size_bytes)))
    media.add_row(Text("Provider", style="dim"), Text(f"{state.provider} / {state.model}", style=CLI_SECONDARY))
    media.add_row(Text("Timeline", style="dim"), Text(f"{len(state.timeline)} operations, {len(state.redo_stack)} redo"))
    media.add_row(Text("Working", style="dim"), Text(format_short_path(state.working_file), style="dim"))
    media.add_row(Text("Output", style="dim"), Text(format_short_path(state.output_dir), style="dim"))

    top = Table.grid(expand=True)
    top.add_column(ratio=1)
    top.add_column(ratio=1)
    top.add_row(media, render_artifacts_table(state))

    creative_runs = latest_creative_runs(state.working_dir, limit=4)
    body_parts: list[object] = [
        top,
        Text(""),
        Text("Recent Timeline", style=f"bold {CLI_ACCENT}"),
        render_recent_timeline_table(state),
    ]
    if creative_runs:
        body_parts.extend(
            [
                Text(""),
                Text("Creative Runs", style=f"bold {CLI_ACCENT}"),
                render_creative_runs_table(state, records=creative_runs),
            ]
        )
    body = Group(*body_parts)
    return Panel(
        body,
        title=f"Project: {state.project_name}",
        border_style=CLI_SUCCESS,
        box=box.ROUNDED,
        padding=(1, 2),
    )


def render_creative_runs_table(
    state: ProjectState,
    *,
    limit: int = 10,
    records: list[dict] | None = None,
):
    records = records if records is not None else latest_creative_runs(state.working_dir, limit=limit)
    table = Table(box=box.SIMPLE_HEAVY, expand=True, show_edge=False)
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("Feature", style=f"bold {CLI_ACCENT}", no_wrap=True)
    table.add_column("Quality", justify="right", no_wrap=True)
    table.add_column("Created", style="dim", no_wrap=True)
    table.add_column("Summary", ratio=1)
    table.add_column("Artifact", style="dim", ratio=1)
    if not records:
        table.add_row("-", "none", "-", "-", "No creative runs recorded for this project yet.", "-")
        return table
    for index, record in enumerate(records[: max(1, int(limit))], start=1):
        summary = dict(record.get("summary") or {})
        artifact_path = str(record.get("manifest_path") or record.get("output_path") or "")
        summary_text = _creative_run_summary(record, summary)
        table.add_row(
            str(index),
            str(record.get("feature") or "creative"),
            _format_quality_score(record.get("quality_score")),
            _format_registry_time(record.get("created_at")),
            truncate_trace_text(summary_text, 92),
            format_short_path(artifact_path) if artifact_path else "-",
        )
    return table


def render_loaded_state_panel(state: ProjectState, *, already_loaded: bool):
    metadata = state.metadata or {}
    status = "Already loaded" if already_loaded else "Loaded"
    grid = Table.grid(padding=(0, 2))
    grid.add_row(Text("File", style="dim"), Text(Path(state.source_files[0]).name if state.source_files else "none"))
    grid.add_row(Text("Duration", style="dim"), Text(format_media_duration(metadata.get("duration_sec"))))
    grid.add_row(Text("Frame", style="dim"), Text(f"{format_resolution(metadata)} @ {format_fps(metadata)}"))
    grid.add_row(Text("Project", style="dim"), Text(f"{state.project_name} ({state.project_id[:8]})"))
    grid.add_row(Text("Timeline", style="dim"), Text(f"{len(state.timeline)} operations"))
    return Panel(
        grid,
        title=status,
        border_style=CLI_SUCCESS if not already_loaded else CLI_ACCENT,
        box=box.ROUNDED,
        padding=(1, 2),
    )


def create_project(video_path: str, name: str | None, provider_name: str, model_name: str) -> ProjectState:
    absolute_path = os.path.abspath(video_path)
    project_id = str(uuid.uuid4())
    project_name = name or Path(video_path).stem
    working_dir = Path(config.AGENT_PROJECTS_DIR) / project_id
    working_dir.mkdir(parents=True, exist_ok=True)
    working_file = str(working_dir / f"source_{Path(absolute_path).name}")
    shutil.copy2(absolute_path, working_file)
    metadata = probe_video(working_file)
    state = ProjectState(
        project_id=project_id,
        project_name=project_name,
        created_at=utc_now_iso(),
        updated_at=utc_now_iso(),
        source_files=[absolute_path],
        working_file=working_file,
        working_dir=str(working_dir),
        output_dir=str(Path(absolute_path).parent),
        timeline=[],
        redo_stack=[],
        session_log=[],
        metadata=metadata,
        provider=provider_name,
        model=model_name,
    )
    state.save()
    return state


def create_project_from_youtube(url: str, name: str | None, provider_name: str, model_name: str) -> ProjectState:
    project_id = str(uuid.uuid4())
    working_dir = Path(config.AGENT_PROJECTS_DIR) / project_id
    working_dir.mkdir(parents=True, exist_ok=True)
    download = download_youtube_video(url, str(working_dir))
    output_dir = working_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    working_file = os.path.abspath(download.downloaded_path)
    metadata = probe_video(working_file)
    state = ProjectState(
        project_id=project_id,
        project_name=name or download.title,
        created_at=utc_now_iso(),
        updated_at=utc_now_iso(),
        source_files=[working_file],
        working_file=working_file,
        working_dir=str(working_dir),
        output_dir=str(output_dir),
        timeline=[],
        redo_stack=[],
        session_log=[],
        metadata=metadata,
        artifacts={
            "source_url": download.source_url,
            "source_title": download.title,
            "source_id": download.video_id,
            "source_uploader": download.uploader,
        },
        provider=provider_name,
        model=model_name,
    )
    state.save()
    return state


def print_project_panel(state: ProjectState) -> None:
    console.print(render_project_dashboard(state))


def find_project(project: str | None) -> ProjectState:
    if project:
        return ProjectState.load(project)
    projects = ProjectState.list_projects()
    if not projects:
        raise typer.BadParameter("No saved projects found.")
    return ProjectState.load(projects[0]["project_id"])


def render_timeline(state: ProjectState) -> None:
    table = Table(title="Timeline", box=box.SIMPLE_HEAVY)
    table.add_column("#", justify="right")
    table.add_column("Operation")
    table.add_column("Parameters")
    table.add_column("Time")
    for index, op in enumerate(state.timeline, start=1):
        params = ", ".join(
            f"{key}={value}"
            for key, value in op.get("params", {}).items()
            if not key.endswith("_label") and key not in {"file_paths"}
        )
        table.add_row(str(index), op["op"], params or "-", op["timestamp"][11:19])
    console.print(table)


def render_projects() -> None:
    table = Table(title="Saved Projects", box=box.SIMPLE_HEAVY)
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Created")
    table.add_column("Modified")
    table.add_column("Source File")
    table.add_column("Ops", justify="right")
    for item in ProjectState.list_projects():
        table.add_row(
            item["project_id"][:8],
            item["project_name"],
            item["created_at"],
            item["updated_at"],
            Path(item["source_file"]).name,
            str(item["timeline_ops"]),
        )
    console.print(table)


def render_jobs_table(state: ProjectState, *, limit: int = 25):
    table = Table(title="Jobs", box=box.SIMPLE_HEAVY)
    table.add_column("ID", no_wrap=True)
    table.add_column("Status")
    table.add_column("Tool")
    table.add_column("Attempts", justify="right")
    table.add_column("Updated")
    table.add_column("Message", ratio=1)
    records = list_jobs(state.working_dir, limit=max(1, min(limit, 100)))
    if not records:
        table.add_row("-", "none", "-", "0", "-", "No jobs queued for this project.")
        return table
    for record in records:
        message = record.message or record.error or "-"
        table.add_row(
            record.job_id,
            Text(record.status, style=_job_status_style(record.status)),
            record.tool_name,
            str(record.attempts),
            format_relative_time(record.updated_at),
            truncate_trace_text(message, 90),
        )
    return table


def parse_job_params(raw_params: str) -> dict[str, Any]:
    raw = str(raw_params or "").strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"Job params must be a JSON object: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise typer.BadParameter("Job params must be a JSON object.")
    return payload


def direct_queue_job(state: ProjectState, tool_name: str, params: dict[str, Any]) -> JobRecord:
    contract = TOOL_CONTRACTS.get(tool_name)
    if contract is None:
        raise typer.BadParameter(f"Unknown tool for job queue: {tool_name}")
    job = create_tool_job(
        state,
        tool_name,
        params,
        allowed_tools=set(TOOL_CONTRACTS),
        metadata={
            "contract": {
                "category": contract.category,
                "mutates_project": contract.mutates_project,
                "long_running": contract.long_running,
                "replayable": contract.replayable,
            }
        },
    )
    console.print(f"Queued {tool_name} as {job.job_id}", style=CLI_SUCCESS)
    return job


def direct_run_job(state: ProjectState, job_id: str, *, force: bool = False) -> JobRecord:
    try:
        record = run_tool_job(state, job_id, TOOL_EXECUTORS, force=force)
    except JobRunnerError as exc:
        console.print(str(exc), style=CLI_ERROR)
        raise typer.Exit(code=1) from exc
    style = CLI_SUCCESS if record.status == "succeeded" else CLI_ERROR
    detail = record.message or record.error or record.tool_name
    console.print(f"{record.job_id} {record.status}: {detail}", style=style)
    if record.status != "succeeded":
        raise typer.Exit(code=1)
    return record


def render_plans_table(state: ProjectState, *, limit: int = 25):
    table = Table(title="Plans", box=box.SIMPLE_HEAVY)
    table.add_column("ID", no_wrap=True)
    table.add_column("Status")
    table.add_column("Steps", justify="right")
    table.add_column("Updated")
    table.add_column("Instruction", ratio=1)
    records = list_plan_records(state.working_dir, limit=max(1, min(limit, 100)))
    if not records:
        table.add_row("-", "none", "0", "-", "No plans saved for this project.")
        return table
    for record in records:
        steps = record.plan.get("steps") if isinstance(record.plan, dict) else []
        table.add_row(
            record.plan_id,
            Text(record.status, style=_plan_status_style(record.status)),
            str(len(steps) if isinstance(steps, list) else 0),
            format_relative_time(record.updated_at),
            truncate_trace_text(record.instruction, 96),
        )
    return table


def render_plan_record(record: PlanRecord):
    table = Table(title=f"Plan {record.plan_id}", box=box.SIMPLE_HEAVY)
    table.add_column("#", justify="right")
    table.add_column("Tool")
    table.add_column("Label")
    table.add_column("Params", ratio=1)
    for index, step in enumerate(record.plan.get("steps") or [], start=1):
        if not isinstance(step, dict):
            continue
        params = json.dumps(step.get("params") or {}, sort_keys=True)
        table.add_row(
            str(index),
            str(step.get("tool") or ""),
            str(step.get("label") or ""),
            truncate_trace_text(params, 100),
        )
    return table


def direct_create_plan(state: ProjectState, instruction: str) -> PlanRecord:
    compiled = compile_intent(instruction, state)
    if compiled is None:
        raise typer.BadParameter(
            "Could not compile a deterministic plan. Use `vex run` for LLM-assisted planning."
        )
    record = create_plan_record(state, instruction, compiled)
    console.print(render_plan_record(record))
    console.print(f"Saved plan {record.plan_id}", style=CLI_SUCCESS)
    return record


def direct_apply_plan(
    state: ProjectState,
    plan_id: str,
    *,
    force: bool = False,
    executors: dict[str, Any] | None = None,
) -> PlanRecord:
    executors = executors or TOOL_EXECUTORS
    record = claim_plan_record(
        state.working_dir,
        plan_id,
        state_project_id=state.project_id,
        force=force,
    )
    plan = edit_plan_from_record(record)
    current_state = state
    results: list[dict[str, Any]] = []
    for index, step in enumerate(plan.steps, start=1):
        executor = executors.get(step.tool)
        if executor is None:
            error = f"Plan step {index} uses unknown tool: {step.tool}"
            results.append({"success": False, "message": error, "tool_name": step.tool})
            mark_plan_record(state.working_dir, record, status="failed", results=results, error=error)
            raise PlanStoreError(error)
        try:
            result = executor(dict(step.params), current_state)
        except Exception as exc:  # noqa: BLE001
            result = {
                "success": False,
                "message": str(exc),
                "tool_name": step.tool,
                "updated_state": current_state,
            }
        if not isinstance(result, Mapping):
            result = {
                "success": False,
                "message": f"Plan step {index} returned an invalid result.",
                "tool_name": step.tool,
                "updated_state": current_state,
            }
        else:
            result = dict(result)
        current_state = result.get("updated_state", current_state)
        sanitized = sanitize_plan_result(
            {
                **result,
                "step_index": index,
                "step_label": step.label,
            }
        )
        results.append(sanitized)
        if not bool(result.get("success")):
            error = str(result.get("message") or f"Plan step {index} failed.")
            mark_plan_record(state.working_dir, record, status="failed", results=results, error=error)
            raise PlanStoreError(error)
    mark_plan_record(state.working_dir, record, status="applied", results=results)
    console.print(f"Applied plan {record.plan_id}", style=CLI_SUCCESS)
    return record


def direct_nle_export(
    state: ProjectState,
    *,
    output_dir: str | Path | None = None,
    formats: set[str] | None = None,
) -> NLEExportResult:
    result = export_nle_bundle(state, output_dir, formats=formats)
    console.print(f"NLE export written to {result.output_dir}", style=CLI_SUCCESS)
    for format_name, path in sorted(result.files.items()):
        console.print(f"{format_name}: {path}")
    return result


def parse_nle_formats(value: str) -> set[str]:
    normalized = {item.strip().lower() for item in str(value or "all").split(",") if item.strip()}
    if not normalized or "all" in normalized:
        return set(SUPPORTED_NLE_FORMATS)
    unknown = normalized - SUPPORTED_NLE_FORMATS
    if unknown:
        raise typer.BadParameter(
            "format must be all, json, fcpxml, edl, or a comma-separated subset."
        )
    return normalized


def render_evaluation_report(report: EvaluationReport):
    table = Table(title=f"Evaluation: {report.suite}", box=box.SIMPLE_HEAVY)
    table.add_column("Case")
    table.add_column("Status")
    table.add_column("Tools")
    table.add_column("Issues", ratio=1)
    for case in report.cases:
        table.add_row(
            case.case_id,
            Text("pass" if case.passed else "fail", style=CLI_SUCCESS if case.passed else CLI_ERROR),
            ", ".join(case.actual_tools) or "-",
            "; ".join(case.issues) or "-",
        )
    return Group(
        Text(
            f"Score: {report.passed_count}/{report.case_count} ({report.score:.0%})",
            style=CLI_SUCCESS if report.passed else CLI_ERROR,
        ),
        table,
    )


def direct_eval_intents(
    state: ProjectState | None = None,
    *,
    output: str | Path | None = None,
) -> EvaluationReport:
    report = run_intent_evaluation(state=state)
    console.print(render_evaluation_report(report))
    if output is not None:
        path = write_evaluation_report(report, output)
        console.print(f"Report: {path}")
    return report


def render_plugins_table(plugins: list[PluginManifest]):
    table = Table(title="Plugins", box=box.SIMPLE_HEAVY)
    table.add_column("Name")
    table.add_column("Version")
    table.add_column("Tools", justify="right")
    table.add_column("Path", ratio=1)
    if not plugins:
        table.add_row("-", "-", "0", "No plugin manifests discovered.")
        return table
    for plugin in plugins:
        table.add_row(
            plugin.name,
            plugin.version,
            str(len(plugin.tools)),
            plugin.plugin_dir,
        )
    return table


def parse_plugin_paths(value: str | None) -> list[Path]:
    if not value:
        return default_plugin_dirs()
    return [Path(item) for item in str(value).split(os.pathsep) if item.strip()]


def direct_list_plugins(paths: list[Path] | None = None) -> list[PluginManifest]:
    plugins = discover_plugins(paths)
    console.print(render_plugins_table(plugins))
    return plugins


def _plan_status_style(status: str) -> str:
    if status == "applied":
        return CLI_SUCCESS
    if status == "failed":
        return CLI_ERROR
    if status == "applying":
        return CLI_WARNING
    return "dim"


def _job_status_style(status: str) -> str:
    if status == "succeeded":
        return CLI_SUCCESS
    if status == "failed":
        return CLI_ERROR
    if status == "running":
        return CLI_WARNING
    return "dim"


def render_repl_help() -> None:
    table = Table(box=box.SIMPLE_HEAVY, expand=True, show_edge=False)
    table.add_column("Command", style=f"bold {CLI_ACCENT}", no_wrap=True)
    table.add_column("Purpose", ratio=1)
    table.add_column("Typical use", style="dim", ratio=1)
    rows = [
        ("/status", "Show the active project dashboard", "Check media, artifacts, paths, and timeline"),
        ("/timeline", "Show applied edit operations", "Audit what has changed"),
        ("/creative-runs", "Show recent creative feature runs", "Inspect shorts, visuals, and grading QA records"),
        ("/trace", "Show the last agent turn trace", "Inspect model, tool, retry, and timing steps"),
        ("/provider", "Show the active provider and model", "Confirm backend routing"),
        ("/projects", "List saved projects", "Find a project id"),
        ("/undo", "Undo the last timeline operation", "Step back one edit"),
        ("/redo", "Redo the last undone operation", "Restore an edit"),
        ("/export <preset>", "Export immediately", "Example: /export youtube_1080p"),
        ("/encode <request>", "Plan an FFmpeg encode", "Example: /encode compress under 50MB"),
        ("/color-grade [look]", "Apply automatic color grading", "Looks: auto, cinematic, vibrant, natural"),
        ("/auto-effects", "Add subtitle-aware motion accents", "Zooms, flashes, focus, and emphasis"),
        ("/help", "Show this command table", "Quick command reference"),
        ("/quit", "Save and exit", "Also available as /exit"),
    ]
    for command, purpose, use in rows:
        table.add_row(command, purpose, use)
    body = Group(
        Text("Plain English editing still works. Slash commands are for fast control.", style="dim"),
        Text(""),
        table,
    )
    console.print(
        Panel(
            body,
            title="Commands",
            border_style=CLI_ACCENT,
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )


def render_trace_history(state: ProjectState) -> None:
    artifact = (state.artifacts or {}).get("latest_agent_trace")
    if not artifact:
        console.print("No agent trace recorded yet.")
        return
    events = [TraceEvent.from_dict(item) for item in artifact.get("events", [])]
    meta = Table.grid(padding=(0, 2))
    meta.add_row("Instruction:", str(artifact.get("instruction") or "unknown"))
    meta.add_row("Provider:", f"{artifact.get('provider', 'unknown')} / {artifact.get('model', 'unknown')}")
    meta.add_row("Success:", "yes" if artifact.get("success") else "no")
    if artifact.get("tools_called"):
        meta.add_row("Tools:", ", ".join(str(name) for name in artifact["tools_called"]))
    body = Group(meta, render_trace_table(events, max_items=16))
    console.print(Panel(body, title="Latest Agent Trace", border_style=CLI_SECONDARY, box=box.ROUNDED, padding=(1, 2)))


ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
PROGRESS_HINT_RE = re.compile(r"(%\||frames/s|it/s|\bETA\b|\b\d+/\d+\b)", re.IGNORECASE)


class LiveLogBuffer:
    def __init__(
        self,
        *,
        max_lines: int = 6,
        max_line_length: int = 160,
        on_update=None,
    ) -> None:
        self.max_lines = max_lines
        self.max_line_length = max_line_length
        self.on_update = on_update
        self._lines: deque[str] = deque(maxlen=max_lines)
        self._current = ""
        self._lock = threading.Lock()

    def _normalize(self, value: str) -> str:
        cleaned = ANSI_ESCAPE_RE.sub("", str(value or ""))
        cleaned = " ".join(cleaned.replace("\t", " ").split()).strip()
        if len(cleaned) > self.max_line_length:
            cleaned = cleaned[: self.max_line_length - 3].rstrip() + "..."
        return cleaned

    def _push_line(self, value: str, *, replace_last: bool) -> bool:
        line = self._normalize(value)
        if not line:
            return False
        if replace_last and self._lines:
            if self._lines[-1] != line:
                self._lines[-1] = line
                return True
            return False
        elif not self._lines or self._lines[-1] != line:
            self._lines.append(line)
            return True
        return False

    def write(self, text: str) -> int:
        if not text:
            return 0
        changed = False
        with self._lock:
            cleaned = ANSI_ESCAPE_RE.sub("", str(text)).replace("\r\n", "\n")
            for part in re.split(r"(\r|\n)", cleaned):
                if not part:
                    continue
                if part == "\r":
                    if self._current.strip():
                        changed = self._push_line(self._current, replace_last=True) or changed
                    self._current = ""
                    continue
                if part == "\n":
                    if self._current.strip():
                        changed = (
                            self._push_line(self._current, replace_last=bool(PROGRESS_HINT_RE.search(self._current)))
                            or changed
                        )
                    self._current = ""
                    continue
                self._current += part
                if PROGRESS_HINT_RE.search(self._current):
                    changed = self._push_line(self._current, replace_last=True) or changed
                    self._current = ""
        if changed and self.on_update is not None:
            self.on_update()
        return len(text)

    def flush(self, *, notify: bool = True) -> None:
        changed = False
        with self._lock:
            if self._current.strip():
                changed = self._push_line(self._current, replace_last=bool(PROGRESS_HINT_RE.search(self._current)))
                self._current = ""
        if notify and changed and self.on_update is not None:
            self.on_update()

    def isatty(self) -> bool:
        return False

    @property
    def encoding(self) -> str:
        return "utf-8"

    def snapshot(self) -> list[str]:
        with self._lock:
            if self._current.strip():
                self._push_line(self._current, replace_last=bool(PROGRESS_HINT_RE.search(self._current)))
                self._current = ""
            return list(self._lines)

    def has_content(self) -> bool:
        return bool(self.snapshot())


def clip_live_text(output: Text, *, max_lines: int = 10, max_chars: int = 1600) -> Text:
    plain = output.plain.strip()
    if not plain:
        return Text("")
    if len(plain) > max_chars:
        plain = plain[-max_chars:]
    lines = plain.splitlines()
    if len(lines) > max_lines:
        lines = ["..."] + lines[-max_lines:]
    return Text("\n".join(lines))


def clip_tool_lines(lines: list[str], *, max_lines: int = 12, max_chars: int = 1800) -> Text:
    if not lines:
        return Text("")
    clipped = list(lines[-max_lines:])
    joined = "\n".join(clipped)
    if len(joined) > max_chars:
        joined = "...\n" + joined[-max_chars:]
    return Text(joined, style="dim")


def _status_from_trace_events(trace_events: list[TraceEvent], active_tool_name: str | None) -> tuple[str, str, str, bool]:
    if active_tool_name:
        return ("Running tool", active_tool_name, "yellow", True)
    if not trace_events:
        return ("Thinking", "Waiting for the first agent update.", "cyan", True)

    last_event = trace_events[-1]
    title = str(last_event.title or "Working")
    detail = str(last_event.detail or "").strip()
    status = str(last_event.status or "info").strip().lower()
    running = status == "running"

    if title.startswith("Planning pass"):
        return ("Thinking", "Reviewing the project and deciding the next step.", "yellow", True)
    if title == "Sending request to Gemini":
        return ("Thinking", detail or "Calling the Gemini model.", "yellow", True)
    if title == "Streaming assistant response":
        return ("Writing response", detail or "Receiving model output.", "yellow", True)
    if title == "Model requested tools":
        return ("Preparing tools", detail or "The model picked tools to run.", "cyan", True)
    if title.startswith("Running "):
        return ("Running tool", title.replace("Running ", "", 1), "yellow", True)
    if title.endswith(" completed"):
        return ("Tool finished", title.replace(" completed", "", 1), "green", False)
    if title.endswith(" failed"):
        return ("Tool failed", title.replace(" failed", "", 1), "red", False)
    if title == "Final response ready":
        return ("Done", detail or "Turn complete.", "green" if status != "error" else "red", False)
    if status == "error":
        return ("Error", f"{title}: {detail}".strip(": "), "red", False)
    if status == "success":
        return ("Done", f"{title}: {detail}".strip(": "), "green", False)
    return (title, detail, trace_status_style(status), running)


def _clean_live_status_line(line: str) -> str:
    cleaned = str(line or "").strip()
    cleaned = re.sub(r"^\[[^\]]+\]\s*", "", cleaned)
    return truncate_trace_text(cleaned, 180)


def _spinner_status_text(
    command: str,
    trace_events: list[TraceEvent],
    active_tool_name: str | None,
    tool_logs: LiveLogBuffer,
    *,
    elapsed_sec: int,
) -> str:
    latest_log = ""
    tool_lines = tool_logs.snapshot()
    if tool_lines:
        latest_log = _clean_live_status_line(tool_lines[-1])

    if active_tool_name:
        parts = [f"Running tool: {active_tool_name}"]
        if latest_log:
            parts.append(latest_log)
        parts.append(f"{max(elapsed_sec, 0)}s")
        return " | ".join(part for part in parts if part)

    if not trace_events:
        return f"Thinking... | {truncate_trace_text(command, 80)} | {max(elapsed_sec, 0)}s"

    last_event = trace_events[-1]
    title = str(last_event.title or "").strip()
    detail = str(last_event.detail or "").strip()
    status = str(last_event.status or "info").strip().lower()

    if title.startswith("Planning pass"):
        label = "Thinking..."
        detail = "Reviewing the project"
    elif title.startswith("Sending request to "):
        label = "Calling model..."
    elif title == "Streaming assistant response":
        label = "Writing response..."
    elif title == "Model requested tools":
        label = "Preparing tools..."
    elif title.startswith("Running "):
        label = f"Running tool: {title.replace('Running ', '', 1)}"
        detail = latest_log or detail
    elif title.endswith(" completed"):
        label = f"Finished tool: {title.replace(' completed', '', 1)}"
    elif title.endswith(" failed"):
        label = f"Tool failed: {title.replace(' failed', '', 1)}"
    elif title == "Final response ready":
        label = "Finalizing response..."
    elif status == "error":
        label = "Handling error..."
    else:
        label = title or "Working..."

    parts = [label]
    if latest_log and not active_tool_name and not title.startswith("Running "):
        parts.append(latest_log)
    elif detail:
        parts.append(truncate_trace_text(detail, 120))
    parts.append(f"{max(elapsed_sec, 0)}s")
    return " | ".join(part for part in parts if part)


class TerminalSpinnerLine:
    def __init__(self, stream=None) -> None:
        self.stream = stream or sys.__stdout__
        self.frames = ("|", "/", "-", "\\")
        self.enabled = bool(getattr(self.stream, "isatty", lambda: False)())
        self._lock = threading.Lock()
        self._last_rendered_width = 0

    def render(self, text: str, *, frame_index: int) -> None:
        if not self.enabled:
            return
        width = max(shutil.get_terminal_size((120, 20)).columns - 1, 24)
        line = f"{self.frames[frame_index % len(self.frames)]} {truncate_trace_text(text, width - 2)}"
        with self._lock:
            padding = max(self._last_rendered_width - len(line), 0)
            self.stream.write("\r" + line + (" " * padding))
            self.stream.flush()
            self._last_rendered_width = len(line)

    def clear(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            if self._last_rendered_width > 0:
                self.stream.write("\r" + (" " * self._last_rendered_width) + "\r")
                self.stream.flush()
                self._last_rendered_width = 0


LIVE_PHASES = ["Input", "Plan", "Model", "Tools", "Done"]


def _event_live_phase(event: TraceEvent | None, active_tool_name: str | None) -> str:
    if active_tool_name:
        return "Tools"
    if event is None:
        return "Input"
    title = str(event.title or "")
    kind = str(event.kind or "").lower()
    if title == "Final response ready":
        return "Done"
    if kind == "tool" or title.startswith("Running ") or title.endswith(" completed") or title.endswith(" failed"):
        return "Tools"
    if kind == "provider" or title.startswith("Sending request") or title.startswith("Streaming") or title.startswith("Model "):
        return "Model"
    if kind == "turn":
        return "Input"
    return "Plan"


def _render_phase_rail(trace_events: list[TraceEvent], active_tool_name: str | None):
    current_phase = _event_live_phase(trace_events[-1] if trace_events else None, active_tool_name)
    current_index = LIVE_PHASES.index(current_phase)
    last_status = str((trace_events[-1].status if trace_events else "running") or "running").lower()
    table = Table.grid(expand=True)
    for _phase in LIVE_PHASES:
        table.add_column(ratio=1)
    cells: list[Text] = []
    for index, phase in enumerate(LIVE_PHASES):
        if index < current_index:
            label = "OK"
            style = CLI_SUCCESS
        elif index == current_index:
            if last_status == "error":
                label = "ERR"
                style = CLI_ERROR
            elif last_status == "success" and phase == "Done":
                label = "OK"
                style = CLI_SUCCESS
            else:
                label = "RUN"
                style = CLI_WARNING
        else:
            label = "--"
            style = "dim"
        text = Text()
        text.append(f"{label} ", style=f"bold {style}")
        text.append(phase, style=style)
        cells.append(text)
    table.add_row(*cells)
    return table


def _latest_tool_duration(trace_events: list[TraceEvent]) -> str:
    for event in reversed(trace_events):
        if str(event.kind or "").lower() == "tool" and (event.metadata or {}).get("duration_sec") is not None:
            return format_trace_duration((event.metadata or {}).get("duration_sec"))
    return ""


def render_live_agent_view(
    command: str,
    output: Text,
    trace_events: list[TraceEvent],
    tool_logs: LiveLogBuffer,
    *,
    active_tool_name: str | None,
    elapsed_sec: int,
):
    tool_lines = tool_logs.snapshot()
    status_label, status_detail, status_style, show_spinner = _status_from_trace_events(trace_events, active_tool_name)
    sections: list[object] = []

    header_grid = Table.grid(expand=True)
    header_grid.add_column(width=2)
    header_grid.add_column(ratio=1)
    header_grid.add_column(justify="right", width=12)
    if show_spinner:
        header_grid.add_row(
            Spinner("dots", style=status_style),
            Text(f"{status_label}: {status_detail}", style=f"bold {status_style}"),
            Text(format_elapsed(elapsed_sec), style="dim"),
        )
    else:
        header_grid.add_row(
            Text(trace_status_label(trace_events[-1].status) if trace_events else ""),
            Text(f"{status_label}: {status_detail}", style=f"bold {status_style}"),
            Text(format_elapsed(elapsed_sec), style="dim"),
        )
    sections.extend(
        [
            header_grid,
            Text(""),
            Text(f"Instruction: {truncate_trace_text(command, 120)}", style="dim"),
            Text(""),
            _render_phase_rail(trace_events, active_tool_name),
        ]
    )

    facts = Table.grid(expand=True, padding=(0, 2))
    facts.add_column(ratio=1)
    facts.add_column(ratio=1)
    facts.add_column(ratio=1)
    facts.add_row(
        Text(f"Steps: {len(trace_events)}", style="dim"),
        Text(f"Tool: {active_tool_name or 'idle'}", style=f"bold {CLI_ACCENT}" if active_tool_name else "dim"),
        Text(f"Last tool: {_latest_tool_duration(trace_events) or 'none'}", style="dim"),
    )
    sections.extend([Text(""), facts])

    if trace_events:
        sections.extend(
            [
                Text(""),
                Text("Backend Activity", style=f"bold {CLI_ACCENT}"),
                render_trace_table(trace_events, max_items=6),
            ]
        )
    if tool_lines:
        sections.extend(
            [
                Text(""),
                Text("Tool Logs", style=f"bold {CLI_SECONDARY}"),
                clip_tool_lines(tool_lines),
            ]
        )
    if output.plain.strip():
        sections.extend(
            [
                Text(""),
                Text("Assistant Preview", style=f"bold {CLI_SUCCESS}"),
                clip_live_text(output),
            ]
        )
    return Panel(
        Group(*sections),
        title="Agent Run",
        border_style=status_style,
        box=box.ROUNDED,
        padding=(1, 2),
    )


def run_agent_with_live_trace(agent: VideoAgent, command: str):
    output = Text()
    trace_events: list[TraceEvent] = []
    tool_logs = LiveLogBuffer()
    active_tool_name: str | None = None
    started_at = time.monotonic()
    stop_event = threading.Event()
    state_lock = threading.Lock()
    render_lock = threading.Lock()
    live_console = Console(file=sys.__stdout__)
    use_live_hud = bool(live_console.is_interactive)
    live_holder: dict[str, Live | None] = {"live": None}
    status_text = {"value": f"Thinking... | {truncate_trace_text(command, 80)} | 0s"}
    spinner_line = TerminalSpinnerLine()

    def snapshot_state() -> tuple[Text, list[TraceEvent], str | None]:
        with state_lock:
            return Text(output.plain), list(trace_events), active_tool_name

    def current_view():
        output_snapshot, trace_snapshot, active_tool_snapshot = snapshot_state()
        return render_live_agent_view(
            command,
            output_snapshot,
            trace_snapshot,
            tool_logs,
            active_tool_name=active_tool_snapshot,
            elapsed_sec=int(time.monotonic() - started_at),
        )

    def refresh_status() -> None:
        _output_snapshot, trace_snapshot, active_tool_snapshot = snapshot_state()
        status_text["value"] = _spinner_status_text(
            command,
            trace_snapshot,
            active_tool_snapshot,
            tool_logs,
            elapsed_sec=int(time.monotonic() - started_at),
        )
        live = live_holder["live"]
        if live is not None:
            with render_lock:
                live.update(current_view(), refresh=True)

    tool_logs.on_update = refresh_status

    def stream_callback(chunk: str) -> None:
        with state_lock:
            output.append(chunk)
        refresh_status()

    def trace_callback(event: TraceEvent) -> None:
        with state_lock:
            trace_events.append(event)
        refresh_status()

    def tool_callback(phase: str, tool_name: str, _ok: bool) -> None:
        nonlocal active_tool_name
        with state_lock:
            if phase == "start":
                active_tool_name = tool_name
            elif phase == "finish" and active_tool_name == tool_name:
                active_tool_name = None
        refresh_status()

    response = None
    frame_index = {"value": 0}

    def heartbeat() -> None:
        while not stop_event.is_set():
            refresh_status()
            if not use_live_hud:
                spinner_line.render(status_text["value"], frame_index=frame_index["value"])
            frame_index["value"] += 1
            stop_event.wait(0.18 if use_live_hud else 0.12)

    heartbeat_thread = threading.Thread(target=heartbeat, name="vex-live-status", daemon=True)
    try:
        if use_live_hud:
            with Live(
                current_view(),
                console=live_console,
                refresh_per_second=LIVE_REFRESH_PER_SECOND,
                transient=False,
                vertical_overflow="ellipsis",
            ) as live:
                live_holder["live"] = live
                heartbeat_thread.start()
                with contextlib.redirect_stdout(tool_logs), contextlib.redirect_stderr(tool_logs):
                    response = agent.run(
                        command,
                        stream_callback=stream_callback,
                        tool_callback=tool_callback,
                        trace_callback=trace_callback,
                    )
                tool_logs.flush()
                refresh_status()
        else:
            heartbeat_thread.start()
            with contextlib.redirect_stdout(tool_logs), contextlib.redirect_stderr(tool_logs):
                response = agent.run(
                    command,
                    stream_callback=stream_callback,
                    tool_callback=tool_callback,
                    trace_callback=trace_callback,
                )
            tool_logs.flush()
    finally:
        live_holder["live"] = None
        tool_logs.flush()
        stop_event.set()
        if heartbeat_thread.is_alive():
            heartbeat_thread.join(timeout=1.5)
        spinner_line.clear()
    if response is None:
        raise AgentLoopError("The agent run did not return a response.")
    return response, trace_events


def direct_export(state: ProjectState, preset_name: str, output: str | None = None) -> None:
    presets = load_presets()
    if preset_name not in presets:
        raise typer.BadParameter(f"Unknown preset {preset_name!r}.")
    preset = presets[preset_name]
    if output:
        output_path = os.path.abspath(output)
    else:
        suffix = preset.get("format") or "mp4"
        stem = "".join(ch for ch in state.project_name.replace(" ", "_") if ch.isalnum() or ch in {"_", "-"})
        output_path = os.path.join(state.output_dir, f"{stem}_{preset_name}.{suffix}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    estimated = estimate_output_size(state.working_file, preset)
    if not check_disk_space(output_path, estimated):
        raise typer.BadParameter("Not enough free disk space for the requested export.")
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        console=console,
    )
    with progress:
        task = progress.add_task("Exporting...", total=100)

        def on_progress(value: float) -> None:
            progress.update(task, completed=value * 100)

        export_media(state.working_file, output_path, preset, progress_callback=on_progress)
    console.print(f"Saved: {output_path} ({format_bytes(os.path.getsize(output_path))})")


def direct_encode(
    state: ProjectState,
    instruction: str,
    *,
    output: str | None = None,
    yes: bool = False,
) -> None:
    params = {"raw_request": instruction}
    if output:
        params["output_path"] = os.path.abspath(output)
        params["_trusted_output_path_token"] = TRUSTED_OUTPUT_PATH_TOKEN
    result = TOOL_EXECUTORS["plan_encode"](params, state)
    if not result["success"]:
        console.print(result["message"], style="red")
        raise typer.Exit(code=1)
    console.print(result["message"])
    suggestion = result.get("suggestion")
    if suggestion and not yes:
        console.print(Panel(str(suggestion), title="Suggestion", border_style="yellow"))
    if not yes:
        return
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        console=console,
        transient=True,
    )
    with progress:
        progress.add_task("Encoding...", total=None)
        run_result = TOOL_EXECUTORS["run_pending_encode"](
            {"plan_id": result.get("plan_id")},
            state,
        )
    if not run_result["success"]:
        console.print(run_result["message"], style="red")
        raise typer.Exit(code=1)
    console.print(run_result["message"])


def direct_auto_shorts(
    state: ProjectState,
    count: int,
    min_duration_sec: float,
    max_duration_sec: float,
    target_platform: str,
    include_compilation: bool,
    subtitle_style: str | None = None,
) -> None:
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        console=console,
        transient=True,
    )
    with progress:
        progress.add_task("Creating auto shorts...", total=None)
        result = TOOL_EXECUTORS["create_auto_shorts"](
            {
                "count": count,
                "min_duration_sec": min_duration_sec,
                "max_duration_sec": max_duration_sec,
                "target_platform": target_platform,
                "include_compilation": include_compilation,
                "subtitle_style": subtitle_style,
            },
            state,
        )
    if not result["success"]:
        console.print(result["message"], style="red")
        raise typer.Exit(code=1)
    console.print(result["message"])


def direct_generate_video(
    *,
    prompt: str,
    script: str | None = None,
    title: str | None = None,
    duration_sec: float = 24.0,
    aspect: str = "landscape",
    fps: int = 30,
    quality: str = "standard",
    render_resolution: str | None = None,
    voice: str = "af_heart",
    voice_speed: float = 1.0,
    style: str = "clean_kinetic",
    background_music_path: str | None = None,
    music_volume: float = 0.12,
    output_dir: str | None = None,
    render: bool = True,
    workers: str | None = None,
    generate_audio: bool = True,
    transcribe_audio: bool = True,
    strict_audio_timing: bool = False,
    state: ProjectState | None = None,
) -> None:
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        console=console,
        transient=True,
    )
    params = {
        "prompt": prompt,
        "script": script,
        "title": title,
        "duration_sec": duration_sec,
        "aspect": aspect,
        "fps": fps,
        "quality": quality,
        "render_resolution": render_resolution,
        "voice": voice,
        "voice_speed": voice_speed,
        "style": style,
        "background_music_path": background_music_path,
        "music_volume": music_volume,
        "output_dir": output_dir,
        "render": render,
        "workers": workers,
        "generate_audio": generate_audio,
        "transcribe_audio": transcribe_audio,
        "strict_audio_timing": strict_audio_timing,
    }
    with progress:
        progress.add_task("Generating HyperFrames video...", total=None)
        result = TOOL_EXECUTORS["generate_video"](params, state)
    if not result["success"]:
        console.print(result["message"], style="red")
        raise typer.Exit(code=1)
    console.print(result["message"])


def direct_add_song(
    state: ProjectState,
    *,
    song_path: str,
    mode: str = "auto",
    start: str | None = None,
    end: str | None = None,
    duration: float | None = None,
    volume: float | None = None,
    fade_in: float | None = None,
    fade_out: float | None = None,
    ducking: str = "auto",
    loop_policy: str = "auto",
    normalize: bool = True,
    preserve_original_audio: bool | None = None,
) -> None:
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        console=console,
        transient=True,
    )
    params: dict[str, object] = {
        "song_path": song_path,
        "mode": mode,
        "start": start,
        "end": end,
        "duration": duration,
        "volume": volume,
        "fade_in": fade_in,
        "fade_out": fade_out,
        "ducking": ducking,
        "loop_policy": loop_policy,
        "normalize": normalize,
    }
    if preserve_original_audio is not None:
        params["preserve_original_audio"] = preserve_original_audio
    with progress:
        progress.add_task("Adding song with Music Director...", total=None)
        result = TOOL_EXECUTORS["add_song"](params, state)
    if not result["success"]:
        console.print(result["message"], style="red")
        raise typer.Exit(code=1)
    console.print(result["message"])


def direct_auto_broll(
    state: ProjectState,
    max_overlays: int,
    min_overlay_sec: float,
    max_overlay_sec: float,
    providers: str = "auto",
    coverage_policy: str = "quality_only",
) -> None:
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        console=console,
        transient=True,
    )
    with progress:
        progress.add_task("Adding auto B-roll...", total=None)
        result = TOOL_EXECUTORS["add_auto_broll"](
            {
                "max_overlays": max_overlays,
                "requested_count": max_overlays if coverage_policy in {"target_count", "exact_count"} else None,
                "coverage_policy": coverage_policy,
                "min_overlay_sec": min_overlay_sec,
                "max_overlay_sec": max_overlay_sec,
                "providers": providers,
            },
            state,
        )
    if not result["success"]:
        console.print(result["message"], style="red")
        raise typer.Exit(code=1)
    console.print(result["message"])


def direct_auto_visuals(
    state: ProjectState,
    mode: str,
    renderer: str,
    style_pack: str,
    max_visuals: int,
    min_visual_sec: float,
    max_visual_sec: float,
    coverage_policy: str = "quality_only",
    density: str = "balanced",
    visual_idea: str | None = None,
    start: str | None = None,
    end: str | None = None,
    trigger_text: str | None = None,
) -> None:
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        console=console,
        transient=True,
    )
    with progress:
        progress.add_task("Adding auto visuals...", total=None)
        params = {
            "mode": mode,
            "renderer": "hyperframes" if visual_idea else renderer,
            "style_pack": style_pack,
            "max_visuals": max_visuals,
            "requested_count": max_visuals if coverage_policy in {"target_count", "exact_count"} else None,
            "coverage_policy": coverage_policy,
            "density": density,
            "min_visual_sec": min_visual_sec,
            "max_visual_sec": max_visual_sec,
        }
        if visual_idea:
            params["visual_idea"] = visual_idea
            if start:
                params["start"] = start
            if end:
                params["end"] = end
            if trigger_text:
                params["trigger_text"] = trigger_text
        result = TOOL_EXECUTORS["add_auto_visuals"](
            params,
            state,
        )
    if not result["success"]:
        console.print(result["message"], style="red")
        raise typer.Exit(code=1)
    console.print(result["message"])


def direct_visual_asset(
    state: ProjectState,
    *,
    asset_path: str,
    start: str,
    end: str,
    composition_mode: str,
) -> None:
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        console=console,
        transient=True,
    )
    with progress:
        progress.add_task("Inserting visual asset...", total=None)
        result = TOOL_EXECUTORS["add_visual_asset"](
            {
                "asset_path": asset_path,
                "start": start,
                "end": end,
                "composition_mode": composition_mode,
            },
            state,
        )
    if not result["success"]:
        console.print(result["message"], style="red")
        raise typer.Exit(code=1)
    console.print(result["message"])


def direct_upscale_video(
    state: ProjectState,
    *,
    resolution: str,
    scale_mode: str,
    output: str | None = None,
) -> None:
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        console=console,
        transient=True,
    )
    params = {"resolution": resolution, "scale_mode": scale_mode}
    if output:
        params["output_path"] = os.path.abspath(output)
    with progress:
        progress.add_task("Scaling video...", total=None)
        result = TOOL_EXECUTORS["upscale_video"](params, state)
    if not result["success"]:
        console.print(result["message"], style="red")
        raise typer.Exit(code=1)
    console.print(result["message"])


def direct_renderers_doctor() -> None:
    result = TOOL_EXECUTORS["renderers_doctor"]({}, None)
    report = result.get("report") or {}
    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("Dependency", style=CLI_ACCENT)
    table.add_column("Status")
    table.add_column("Path / Version")
    for name in ("hyperframes", "imaging", "node", "ffmpeg", "manim", "remotion", "blender"):
        item = report.get(name) or {}
        available = bool(item.get("available"))
        detail_parts = []
        for key in ("source", "cli_path", "path", "version", "package_version", "platform", "arch"):
            if item.get(key):
                detail_parts.append(str(item[key]))
        if name == "imaging":
            if item.get("pillow_version"):
                detail_parts.append(f"Pillow {item['pillow_version']}")
            if item.get("imageio_version"):
                detail_parts.append(f"ImageIO {item['imageio_version']}")
        if item.get("major") is not None and name == "node":
            detail_parts.append(f"major {item['major']}")
        if item.get("reason"):
            detail_parts.append(str(item["reason"]))
        table.add_row(
            name,
            "[green]available[/]" if available else "[yellow]missing[/]",
            "\n".join(detail_parts) or "-",
        )
    console.print(table)


def direct_install_hyperframes(*, force: bool) -> None:
    console.print("Installing the version-locked HyperFrames runtime...")
    try:
        result = install_hyperframes_runtime(force=force)
    except RuntimeInstallError as exc:
        console.print(f"HyperFrames installation failed: {exc}", style=CLI_ERROR)
        if exc.log_path:
            console.print(f"Install log: {exc.log_path}", style="dim")
        raise typer.Exit(code=1)
    if result["changed"]:
        console.print(
            "HyperFrames "
            f"{result['metadata'].get('hyperframes_version')} installed at "
            f"{result['runtime_dir']}",
            style=CLI_SUCCESS,
        )
    else:
        console.print(
            f"HyperFrames is already installed at {result['runtime_dir']}.",
            style=CLI_SUCCESS,
        )


def direct_install_transcription(*, force: bool) -> None:
    console.print("Resolving a Whisper runtime for Vex...")
    try:
        result = install_transcription_dependencies(
            force=force,
            configured_python=config.WHISPER_PYTHON_PATH,
        )
    except TranscriptionInstallError as exc:
        console.print(f"Whisper installation failed: {exc}", style=CLI_ERROR)
        if exc.log_path:
            console.print(f"Install log: {exc.log_path}", style="dim")
        raise typer.Exit(code=1)
    if result["changed"]:
        console.print(
            "Whisper "
            f"{result['version']} installed and verified with "
            f"{result['python_executable']}.",
            style=CLI_SUCCESS,
        )
    else:
        runtime = (
            "external runtime"
            if result.get("runtime") == "external"
            else "current Vex environment"
        )
        console.print(
            "Whisper "
            f"{result['version']} is already available through the {runtime}: "
            f"{result['python_executable']}.",
            style=CLI_SUCCESS,
        )


def direct_auto_effects(
    state: ProjectState,
    density: str,
    intensity: str,
    max_effects: int,
    include_style_effects: bool,
    subtitle_position: str,
    taste_profile: str,
) -> None:
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        console=console,
        transient=True,
    )
    with progress:
        progress.add_task("Adding auto effects...", total=None)
        result = TOOL_EXECUTORS["add_auto_effects"](
            {
                "density": density,
                "intensity": intensity,
                "max_effects": max_effects,
                "include_style_effects": include_style_effects,
                "subtitle_position": subtitle_position,
                "taste_profile": taste_profile,
            },
            state,
        )
    if not result["success"]:
        console.print(result["message"], style="red")
        raise typer.Exit(code=1)
    console.print(result["message"])


def _configured_model_name() -> str:
    provider = config.normalize_provider_name(config.PROVIDER)
    if provider == "gemini":
        return config.GEMINI_MODEL
    if provider == "claude":
        return config.CLAUDE_MODEL
    return config.local_llm_model(provider)


def direct_color_grade(
    state: ProjectState,
    look: str,
    intensity: float,
    sample_count: int,
) -> None:
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        console=console,
        transient=True,
    )
    with progress:
        progress.add_task("Applying auto color grade...", total=None)
        result = TOOL_EXECUTORS["auto_color_grade"](
            {
                "look": look,
                "intensity": intensity,
                "sample_count": sample_count,
            },
            state,
        )
    if not result["success"]:
        console.print(result["message"], style="red")
        raise typer.Exit(code=1)
    console.print(result["message"])


def repl_prompt(state: ProjectState | None) -> str:
    if state is None:
        return f"[bold {CLI_ACCENT}]vex[/][dim] / no project[/] [bold {CLI_ACCENT}]> [/]"
    project_label = escape(truncate_trace_text(state.project_name, 28))
    return f"[bold {CLI_ACCENT}]vex[/][dim] / {project_label}[/] [bold {CLI_ACCENT}]> [/]"


def print_no_project_notice() -> None:
    console.print(
        Panel(
            Text("Drop a video path or YouTube URL in your message to start a project.", style="white"),
            title="No Project Loaded",
            border_style=CLI_WARNING,
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )


def render_provider_panel(provider_name: str, model_name: str):
    table = Table.grid(padding=(0, 2))
    table.add_row(Text("Provider", style="dim"), Text(provider_name, style=CLI_SECONDARY))
    table.add_row(Text("Model", style="dim"), Text(model_name, style="white"))
    table.add_row(Text("Runtime", style="dim"), Text("configured", style=CLI_SUCCESS))
    return Panel(
        table,
        title="Active Backend",
        border_style=CLI_SECONDARY,
        box=box.ROUNDED,
        padding=(1, 2),
    )


def run_repl(state: ProjectState | None, provider) -> None:
    agent = VideoAgent(state, provider) if state is not None else None
    while True:
        try:
            user_input = console.input(repl_prompt(state))
        except KeyboardInterrupt:
            answer = console.input("\nSave and exit? [y/n] ").strip().lower()
            if answer.startswith("y"):
                if state is not None:
                    state.save()
                console.print("Project saved. Goodbye.")
                return
            continue
        command = user_input.strip()
        if not command:
            continue
        if command in {"/quit", "/exit"}:
            if state is not None:
                state.save()
            console.print("Project saved. Goodbye.")
            return
        if command == "/help":
            render_repl_help()
            continue
        if command == "/status":
            if state is None:
                print_no_project_notice()
            else:
                console.print(render_project_dashboard(state))
            continue
        if command == "/timeline":
            if state is None:
                print_no_project_notice()
            else:
                render_timeline(state)
            continue
        if command in {"/creative-runs", "/creative_runs", "/runs"}:
            if state is None:
                print_no_project_notice()
            else:
                console.print(render_creative_runs_table(state, limit=10))
            continue
        if command == "/trace":
            if state is None:
                print_no_project_notice()
            else:
                render_trace_history(state)
            continue
        if command == "/provider":
            if state is None:
                console.print(render_provider_panel(config.PROVIDER, provider.model_name))
            else:
                console.print(render_provider_panel(state.provider, state.model))
            continue
        if command == "/projects":
            render_projects()
            continue
        if command == "/undo":
            if state is None:
                print_no_project_notice()
                continue
            result = TOOL_EXECUTORS["undo"]({}, state)
            state = result["updated_state"]
            if agent is not None:
                agent.state = state
            console.print(result["message"])
            continue
        if command == "/redo":
            if state is None:
                print_no_project_notice()
                continue
            result = TOOL_EXECUTORS["redo"]({}, state)
            state = result["updated_state"]
            if agent is not None:
                agent.state = state
            console.print(result["message"])
            continue
        if command.startswith("/export"):
            if state is None:
                print_no_project_notice()
                continue
            parts = command.split(maxsplit=1)
            if len(parts) != 2:
                console.print("Usage: /export <preset>")
                continue
            direct_export(state, parts[1].strip())
            continue
        if command.startswith("/encode"):
            if state is None:
                print_no_project_notice()
                continue
            parts = command.split(maxsplit=1)
            if len(parts) != 2:
                console.print("Usage: /encode <request>")
                continue
            direct_encode(state, parts[1].strip())
            continue
        if command.startswith("/color-grade") or command.startswith("/color_grade"):
            if state is None:
                print_no_project_notice()
                continue
            parts = command.split(maxsplit=1)
            look = parts[1].strip() if len(parts) == 2 else "auto"
            direct_color_grade(state, look=look, intensity=1.0, sample_count=9)
            continue
        if command.startswith("/auto-effects") or command.startswith("/auto_effects"):
            if state is None:
                print_no_project_notice()
                continue
            direct_auto_effects(
                state,
                density="medium",
                intensity="medium",
                max_effects=12,
                include_style_effects=True,
                subtitle_position="bottom",
                taste_profile="auto",
            )
            continue
        if command.startswith("/generate-video") or command.startswith("/generate_video"):
            parts = command.split(maxsplit=1)
            if len(parts) != 2:
                console.print("Usage: /generate-video <prompt>")
                continue
            direct_generate_video(prompt=parts[1].strip(), state=state)
            continue

        load_request = parse_load_source_command(command)
        if load_request is not None:
            load_kind, load_target = load_request
            if load_kind == "missing_path":
                console.print(f"Video file not found: {load_target}", style="red")
                continue
            if load_kind == "path":
                already_loaded = is_loaded_source(state, load_target)
                if already_loaded and state is not None:
                    console.print(render_loaded_state_panel(state, already_loaded=True))
                    continue
                with console.status(
                    f"[bold {CLI_ACCENT}]Loading {escape(Path(load_target).name)}...",
                    spinner="dots",
                ):
                    state = find_project_for_source(load_target)
                    if state is None:
                        state = create_project(load_target, None, config.PROVIDER, provider.model_name)
                agent = VideoAgent(state, provider)
                console.print(render_loaded_state_panel(state, already_loaded=False))
                continue
            already_loaded = is_loaded_source_url(state, load_target)
            if already_loaded and state is not None:
                console.print(render_loaded_state_panel(state, already_loaded=True))
                continue
            state = find_project_for_source_url(load_target)
            if state is None:
                try:
                    with console.status(f"[bold {CLI_ACCENT}]Fetching YouTube video...", spinner="dots"):
                        state = create_project_from_youtube(load_target, None, config.PROVIDER, provider.model_name)
                except Exception as exc:
                    console.print(f"Failed to download YouTube video: {exc}", style="red")
                    continue
            agent = VideoAgent(state, provider)
            console.print(render_loaded_state_panel(state, already_loaded=False))
            continue

        detected_path = detect_video_path(command)
        detected_url = extract_youtube_url(command)
        if detected_path and not is_loaded_source(state, detected_path):
            with console.status(
                f"[bold {CLI_ACCENT}]Loading {escape(Path(detected_path).name)}...",
                spinner="dots",
            ):
                state = find_project_for_source(detected_path)
                if state is None:
                    state = create_project(detected_path, None, config.PROVIDER, provider.model_name)
            agent = VideoAgent(state, provider)
            console.print(render_loaded_state_panel(state, already_loaded=False))
        elif detected_url and not is_loaded_source_url(state, detected_url):
            state = find_project_for_source_url(detected_url)
            if state is None:
                try:
                    with console.status(f"[bold {CLI_ACCENT}]Fetching YouTube video...", spinner="dots"):
                        state = create_project_from_youtube(detected_url, None, config.PROVIDER, provider.model_name)
                except Exception as exc:
                    console.print(f"Failed to download YouTube video: {exc}", style="red")
                    continue
            agent = VideoAgent(state, provider)
            console.print(render_loaded_state_panel(state, already_loaded=False))
        elif state is None:
            standalone_plan = compile_intent(command, None)
            if (
                standalone_plan is not None
                and len(standalone_plan.steps) == 1
                and standalone_plan.steps[0].tool == "generate_video"
            ):
                params = standalone_plan.steps[0].params
                direct_generate_video(
                    prompt=str(params.get("prompt") or command),
                    duration_sec=float(params.get("duration_sec") or 24.0),
                    aspect=str(params.get("aspect") or "landscape"),
                    voice=str(params.get("voice") or "af_heart"),
                    render=bool(params.get("render", True)),
                    generate_audio=bool(params.get("generate_audio", True)),
                    state=None,
                )
                continue
            print_no_project_notice()
            continue

        try:
            response, _trace_events = run_agent_with_live_trace(agent, command)
        except AgentLoopError as exc:
            console.print(f"Agent error: {exc}", style="red")
            continue
        except Exception:
            console.print_exception()
            continue

        state = agent.state
        if response.message:
            console.print(response.message)
        for suggestion in response.suggestions:
            console.print(Panel(suggestion, title="Suggestion", border_style="yellow"))


@app.command()
def start(video_path: str, name: str | None = typer.Option(default=None, help="Project name.")) -> None:
    provider = create_provider()
    absolute_path = os.path.abspath(video_path)
    if not os.path.isfile(absolute_path):
        raise typer.BadParameter(f"Video file not found: {absolute_path}")
    state = create_project(absolute_path, name, config.PROVIDER, provider.model_name)
    print_project_panel(state)
    run_repl(state, provider)


@app.command()
def repl(project: str | None = typer.Option(default=None, help="Project id.")) -> None:
    provider = create_provider()
    state = find_project(project)
    run_repl(state, provider)


@app.command()
def run(
    instruction: str,
    project: str = typer.Option(..., help="Project id."),
) -> None:
    provider = create_provider()
    state = ProjectState.load(project)
    agent = VideoAgent(state, provider)
    try:
        response, _trace_events = run_agent_with_live_trace(agent, instruction)
    except AgentLoopError as exc:
        console.print(f"Agent error: {exc}", style="red")
        raise typer.Exit(code=1)
    except Exception:
        console.print_exception()
        raise typer.Exit(code=1)
    console.print(response.message)
    for suggestion in response.suggestions:
        console.print(Panel(suggestion, title="Suggestion", border_style="yellow"))


@app.command()
def projects() -> None:
    initialize_runtime()
    render_projects()


@app.command("jobs")
def jobs_command(
    project: str = typer.Option(..., help="Project id."),
    limit: int = typer.Option(25, help="Maximum number of jobs to show."),
) -> None:
    initialize_runtime(require_provider=False)
    state = ProjectState.load(project)
    console.print(render_jobs_table(state, limit=limit))


@app.command("queue-job")
def queue_job_command(
    tool_name: str = typer.Argument(..., help="Tool name to run later, for example add_auto_visuals."),
    project: str = typer.Option(..., help="Project id."),
    params: str = typer.Option("{}", "--params", help="JSON object passed to the tool."),
) -> None:
    initialize_runtime(require_provider=False)
    state = ProjectState.load(project)
    direct_queue_job(state, tool_name, parse_job_params(params))


@app.command("run-job")
def run_job_command(
    job_id: str = typer.Argument(..., help="Queued job id."),
    project: str = typer.Option(..., help="Project id."),
    force: bool = typer.Option(False, "--force", help="Run even if the job is not queued or failed."),
) -> None:
    initialize_runtime(require_provider=False)
    state = ProjectState.load(project)
    direct_run_job(state, job_id, force=force)


@app.command("plans")
def plans_command(
    project: str = typer.Option(..., help="Project id."),
    limit: int = typer.Option(25, help="Maximum number of plans to show."),
) -> None:
    initialize_runtime(require_provider=False)
    state = ProjectState.load(project)
    console.print(render_plans_table(state, limit=limit))


@app.command("plan")
def plan_command(
    instruction: str = typer.Argument(..., help="Editing instruction to compile into a saved plan."),
    project: str = typer.Option(..., help="Project id."),
) -> None:
    initialize_runtime(require_provider=False)
    state = ProjectState.load(project)
    direct_create_plan(state, instruction)


@app.command("apply-plan")
def apply_plan_command(
    plan_id: str = typer.Argument(..., help="Saved plan id."),
    project: str = typer.Option(..., help="Project id."),
    force: bool = typer.Option(False, "--force", help="Apply an already-applied plan again."),
) -> None:
    initialize_runtime(require_provider=False)
    state = ProjectState.load(project)
    try:
        direct_apply_plan(state, plan_id, force=force)
    except PlanStoreError as exc:
        console.print(str(exc), style=CLI_ERROR)
        raise typer.Exit(code=1) from exc


@app.command("nle-export")
def nle_export_command(
    project: str = typer.Option(..., help="Project id."),
    output_dir: Path | None = typer.Option(None, "--output-dir", "--output", help="Directory for NLE export files."),
    format: str = typer.Option("all", "--format", help="all, json, fcpxml, edl, or comma-separated subset."),  # noqa: A002
) -> None:
    initialize_runtime(require_provider=False)
    state = ProjectState.load(project)
    direct_nle_export(state, output_dir=output_dir, formats=parse_nle_formats(format))


@app.command("eval-intents")
def eval_intents_command(
    project: str | None = typer.Option(None, help="Optional project id for project-aware cases."),
    output: Path | None = typer.Option(None, "--output", help="Optional JSON report path."),
) -> None:
    initialize_runtime(require_provider=False)
    state = ProjectState.load(project) if project else None
    report = direct_eval_intents(state, output=output)
    if not report.passed:
        raise typer.Exit(code=1)


@app.command("plugins")
def plugins_command(
    path: str | None = typer.Option(
        None,
        "--path",
        help="Plugin directory or OS-pathsep-separated directories. Defaults to VEX_PLUGIN_PATH and ./plugins.",
    ),
) -> None:
    initialize_runtime(require_provider=False)
    direct_list_plugins(parse_plugin_paths(path))


@app.command("generate-video")
def generate_video_command(
    prompt: str = typer.Argument(..., help="Video topic, idea, or creative brief."),
    title: str | None = typer.Option(default=None, help="Optional video title."),
    script: str | None = typer.Option(default=None, help="Exact narration script."),
    script_file: Path | None = typer.Option(None, "--script-file", help="Read narration script from a text file."),
    duration_sec: float = typer.Option(24.0, "--duration", "--duration-sec", help="Target duration in seconds."),
    aspect: str = typer.Option("landscape", help="landscape, portrait, or square."),
    fps: int = typer.Option(30, "--fps", help="Render frames per second."),
    quality: str = typer.Option("standard", "--quality", help="HyperFrames render quality: draft, standard, or high."),
    render_resolution: str | None = typer.Option(None, "--render-resolution", "--export-resolution", help="Optional HyperFrames render preset such as 4k, landscape-4k, or portrait-4k."),
    voice: str = typer.Option("af_heart", help="HyperFrames TTS voice id."),
    voice_speed: float = typer.Option(1.0, help="TTS speed multiplier."),
    style: str = typer.Option("clean_kinetic", help="Art direction label."),
    background_music_path: str | None = typer.Option(None, "--music", "--background-music", help="Optional local music file."),
    music_volume: float = typer.Option(0.12, help="Background music volume from 0 to 1."),
    output_dir: str | None = typer.Option(None, "--output-dir", "--output", help="Optional output directory for the generated project."),
    render: bool = typer.Option(True, "--render/--no-render", help="Render the final video after writing the project."),
    workers: str | None = typer.Option(None, "--workers", help="Optional HyperFrames render worker count such as 1, 2, or auto."),
    generate_audio: bool = typer.Option(True, "--audio/--no-audio", help="Generate narration audio with HyperFrames TTS."),
    transcribe_audio: bool = typer.Option(True, "--transcribe/--no-transcribe", help="Transcribe generated narration for word timing."),
    strict_audio_timing: bool = typer.Option(False, "--strict-audio-timing", help="Fail if HyperFrames transcription cannot produce word timing."),
) -> None:
    initialize_runtime(require_provider=False)
    if aspect not in {"landscape", "portrait", "square", "vertical", "horizontal", "16:9", "9:16", "1:1"}:
        raise typer.BadParameter("aspect must be landscape, portrait, square, vertical, horizontal, 16:9, 9:16, or 1:1")
    if quality not in {"draft", "standard", "high"}:
        raise typer.BadParameter("quality must be draft, standard, or high")
    script_text = script
    if script_file is not None:
        if not script_file.is_file():
            raise typer.BadParameter(f"Script file not found: {script_file}")
        script_text = script_file.read_text(encoding="utf-8")
    direct_generate_video(
        prompt=prompt,
        script=script_text,
        title=title,
        duration_sec=duration_sec,
        aspect=aspect,
        fps=fps,
        quality=quality,
        render_resolution=render_resolution,
        voice=voice,
        voice_speed=voice_speed,
        style=style,
        background_music_path=background_music_path,
        music_volume=music_volume,
        output_dir=output_dir,
        render=render,
        workers=workers,
        generate_audio=generate_audio,
        transcribe_audio=transcribe_audio,
        strict_audio_timing=strict_audio_timing,
    )


@app.command("add-song")
def add_song_command(
    project: str = typer.Option(..., help="Project id."),
    song: str = typer.Option(..., "--song", "--song-path", "--music", help="Local song/music file path."),
    mode: str = typer.Option("auto", help="auto, background, replace, intro, outro, intro_outro, segment, or highlight."),
    start: str | None = typer.Option(None, help="Optional placement start timestamp."),
    end: str | None = typer.Option(None, help="Optional placement end timestamp."),
    duration: float | None = typer.Option(None, help="Optional cue duration in seconds."),
    volume: float | None = typer.Option(None, help="Optional music volume from 0.0 to 1.5."),
    fade_in: float | None = typer.Option(None, "--fade-in", help="Optional fade-in duration in seconds."),
    fade_out: float | None = typer.Option(None, "--fade-out", help="Optional fade-out duration in seconds."),
    ducking: str = typer.Option("auto", help="auto, on, or off."),
    loop_policy: str = typer.Option("auto", "--loop-policy", help="auto, loop, trim, or pad."),
    normalize: bool = typer.Option(True, "--normalize/--no-normalize", help="Apply loudness normalization and limiting."),
    preserve_original_audio: bool | None = typer.Option(None, "--preserve-original-audio/--replace-original-audio", help="Override whether source audio is preserved."),
) -> None:
    initialize_runtime(require_provider=False)
    if mode not in {"auto", "background", "replace", "intro", "outro", "intro_outro", "segment", "highlight"}:
        raise typer.BadParameter("mode must be one of: auto, background, replace, intro, outro, intro_outro, segment, highlight")
    if ducking not in {"auto", "on", "off"}:
        raise typer.BadParameter("ducking must be auto, on, or off")
    if loop_policy not in {"auto", "loop", "trim", "pad"}:
        raise typer.BadParameter("loop_policy must be auto, loop, trim, or pad")
    if bool(start) != bool(end):
        raise typer.BadParameter("Song placement timing requires both --start and --end, or neither.")
    if volume is not None and (volume < 0.0 or volume > 1.5):
        raise typer.BadParameter("volume must be between 0.0 and 1.5")
    state = ProjectState.load(project)
    direct_add_song(
        state,
        song_path=song,
        mode=mode,
        start=start,
        end=end,
        duration=duration,
        volume=volume,
        fade_in=fade_in,
        fade_out=fade_out,
        ducking=ducking,
        loop_policy=loop_policy,
        normalize=normalize,
        preserve_original_audio=preserve_original_audio,
    )


@renderers_app.command("doctor")
def renderers_doctor() -> None:
    config.configure_runtime_logging()
    config.reload_settings()
    direct_renderers_doctor()


@renderers_install_app.command("hyperframes")
def install_hyperframes(
    force: bool = typer.Option(
        False,
        "--force",
        help="Reinstall even when the version-locked runtime already exists.",
    ),
) -> None:
    config.configure_runtime_logging()
    config.reload_settings()
    direct_install_hyperframes(force=force)


@setup_app.command("config")
def setup_config(
    path: Path = typer.Option(
        Path(".env"),
        "--path",
        help="Configuration file to create.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Replace an existing configuration file.",
    ),
) -> None:
    try:
        destination = write_config_template(path, force=force)
    except ConfigurationError as exc:
        console.print(str(exc), style=CLI_ERROR)
        raise typer.Exit(code=1)
    console.print(f"Created Vex configuration at {destination}", style=CLI_SUCCESS)


@setup_app.command("transcription")
def setup_transcription(
    force: bool = typer.Option(
        False,
        "--force",
        help="Reinstall Whisper even when it is already importable by Vex.",
    ),
) -> None:
    config.configure_runtime_logging()
    direct_install_transcription(force=force)


@app.command("creative-runs")
def creative_runs(
    project: str = typer.Option(..., help="Project id."),
    limit: int = typer.Option(10, help="Maximum number of runs to show."),
) -> None:
    initialize_runtime()
    state = ProjectState.load(project)
    console.print(render_creative_runs_table(state, limit=max(1, min(limit, 50))))


@app.command()
def export(
    preset_name: str,
    project: str = typer.Option(..., help="Project id."),
    output: str | None = typer.Option(default=None, help="Custom output path."),
) -> None:
    initialize_runtime()
    state = ProjectState.load(project)
    direct_export(state, preset_name, output)


@app.command("upscale_video")
def upscale_video(
    project: str = typer.Option(..., help="Project id."),
    resolution: str = typer.Option(..., help="Target resolution, for example 1920x1080."),
    scale_mode: str = typer.Option("fit", help="fit, fill, or stretch."),
    output: str | None = typer.Option(default=None, help="Custom output path."),
) -> None:
    initialize_runtime()
    if scale_mode not in {"fit", "fill", "stretch"}:
        raise typer.BadParameter("scale_mode must be one of: fit, fill, stretch")
    state = ProjectState.load(project)
    direct_upscale_video(
        state,
        resolution=resolution,
        scale_mode=scale_mode,
        output=output,
    )


@app.command()
def encode(
    instruction: str,
    project: str = typer.Option(..., help="Project id."),
    output: str | None = typer.Option(default=None, help="Custom output path."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Run the generated encode plan immediately."),
) -> None:
    initialize_runtime()
    state = ProjectState.load(project)
    direct_encode(state, instruction, output=output, yes=yes)


@app.command()
def shorts(
    project: str = typer.Option(..., help="Project id."),
    count: int = typer.Option(3, help="Number of shorts to create."),
    min_duration_sec: float = typer.Option(20.0, help="Minimum duration per short."),
    max_duration_sec: float = typer.Option(45.0, help="Maximum duration per short."),
    target_platform: str = typer.Option(
        "youtube_shorts",
        help="Platform profile: youtube_shorts, tiktok, or instagram_reels.",
    ),
    include_compilation: bool = typer.Option(True, help="Also create a merged compilation."),
    subtitle_style: str | None = typer.Option(
        default=None,
        help="Subtitle style: clean_pop, creator_bold, cinematic, glass, karaoke_focus, or minimal.",
    ),
) -> None:
    initialize_runtime()
    if target_platform not in {"youtube_shorts", "tiktok", "instagram_reels"}:
        raise typer.BadParameter("target_platform must be one of: youtube_shorts, tiktok, instagram_reels")
    state = ProjectState.load(project)
    direct_auto_shorts(
        state,
        count=count,
        min_duration_sec=min_duration_sec,
        max_duration_sec=max_duration_sec,
        target_platform=target_platform,
        include_compilation=include_compilation,
        subtitle_style=subtitle_style,
    )


@app.command()
def auto_broll(
    project: str = typer.Option(..., help="Project id."),
    max_overlays: int = typer.Option(5, help="Maximum number of stock inserts to add."),
    min_overlay_sec: float = typer.Option(1.2, help="Minimum duration of each insert."),
    max_overlay_sec: float = typer.Option(2.8, help="Maximum duration of each insert."),
    providers: str = typer.Option("auto", help="Stock providers: auto, pexels, pixabay, coverr, or comma-separated names."),
    coverage_policy: str = typer.Option("quality_only", help="quality_only, target_count, or exact_count."),
) -> None:
    initialize_runtime()
    if coverage_policy not in {"quality_only", "target_count", "exact_count"}:
        raise typer.BadParameter("coverage_policy must be one of: quality_only, target_count, exact_count")
    state = ProjectState.load(project)
    direct_auto_broll(
        state,
        max_overlays=max_overlays,
        min_overlay_sec=min_overlay_sec,
        max_overlay_sec=max_overlay_sec,
        providers=providers,
        coverage_policy=coverage_policy,
    )


@app.command()
def auto_visuals(
    project: str = typer.Option(..., help="Project id."),
    mode: str = typer.Option("generated_only", help="generated_only, hybrid, or stock_only."),
    renderer: str = typer.Option("auto", help="Renderer backend preference: auto, hyperframes, manim, remotion, both, ffmpeg, or blender."),
    style_pack: str = typer.Option(
        "auto",
        help="Preferred style pack: auto, editorial_clean, bold_tech, documentary_kinetic, product_ui, cinematic_night, signal_lab, or magazine_luxe.",
    ),
    max_visuals: int = typer.Option(5, help="Maximum number of generated visuals to add."),
    min_visual_sec: float = typer.Option(1.4, help="Minimum duration of each generated visual."),
    max_visual_sec: float = typer.Option(3.6, help="Maximum duration of each generated visual."),
    coverage_policy: str = typer.Option("quality_only", help="quality_only, target_count, or exact_count."),
    density: str = typer.Option("balanced", help="sparse, balanced, dense, or chapter_coverage."),
    visual_idea: str | None = typer.Option(None, "--visual-idea", "--idea", help="Describe one custom HyperFrames visual idea to ground in the transcript."),
    start: str | None = typer.Option(None, help="Optional start timestamp for --visual-idea."),
    end: str | None = typer.Option(None, help="Optional end timestamp for --visual-idea."),
    trigger_text: str | None = typer.Option(None, "--trigger-text", "--trigger", help="Optional transcript phrase used to time --visual-idea when start/end are omitted."),
) -> None:
    initialize_runtime()
    if mode not in {"generated_only", "hybrid", "stock_only"}:
        raise typer.BadParameter("mode must be one of: generated_only, hybrid, stock_only")
    if renderer not in {"auto", "hyperframes", "manim", "remotion", "both", "ffmpeg", "blender"}:
        raise typer.BadParameter("renderer must be one of: auto, hyperframes, manim, remotion, both, ffmpeg, blender")
    if style_pack not in {"auto", "editorial_clean", "bold_tech", "documentary_kinetic", "product_ui", "cinematic_night", "signal_lab", "magazine_luxe"}:
        raise typer.BadParameter(
            "style_pack must be one of: auto, editorial_clean, bold_tech, documentary_kinetic, product_ui, cinematic_night, signal_lab, magazine_luxe"
        )
    if coverage_policy not in {"quality_only", "target_count", "exact_count"}:
        raise typer.BadParameter("coverage_policy must be one of: quality_only, target_count, exact_count")
    if density not in {"sparse", "balanced", "dense", "chapter_coverage"}:
        raise typer.BadParameter("density must be one of: sparse, balanced, dense, chapter_coverage")
    if (start or end) and not visual_idea:
        raise typer.BadParameter("start/end on auto_visuals require --visual-idea")
    if visual_idea and bool(start) != bool(end):
        raise typer.BadParameter("--visual-idea timing requires both --start and --end, or neither")
    state = ProjectState.load(project)
    direct_auto_visuals(
        state,
        mode=mode,
        renderer=renderer,
        style_pack=style_pack,
        max_visuals=max_visuals,
        min_visual_sec=min_visual_sec,
        max_visual_sec=max_visual_sec,
        coverage_policy=coverage_policy,
        density=density,
        visual_idea=visual_idea,
        start=start,
        end=end,
        trigger_text=trigger_text,
    )


@app.command("add_visual_asset")
def add_visual_asset(
    project: str = typer.Option(..., help="Project id."),
    asset: str = typer.Option(..., "--asset", "--asset-path", help="Local HTML, video, GIF, or image asset path."),
    start: str = typer.Option(..., help="Start time in seconds or HH:MM:SS."),
    end: str = typer.Option(..., help="End time in seconds or HH:MM:SS."),
    mode: str = typer.Option("replace", help="replace, overlay, or picture_in_picture."),
) -> None:
    initialize_runtime()
    if mode not in {"replace", "overlay", "picture_in_picture"}:
        raise typer.BadParameter("mode must be one of: replace, overlay, picture_in_picture")
    state = ProjectState.load(project)
    direct_visual_asset(
        state,
        asset_path=asset,
        start=start,
        end=end,
        composition_mode=mode,
    )


@app.command()
def auto_effects(
    project: str = typer.Option(..., help="Project id."),
    density: str = typer.Option("medium", help="Effect density: low, medium, or high."),
    intensity: str = typer.Option("medium", help="Effect intensity: subtle, medium, high, or strong."),
    max_effects: int = typer.Option(12, help="Maximum number of effects to add."),
    include_style_effects: bool = typer.Option(True, help="Include vignette, flash, focus, and subtitle highlight accents."),
    subtitle_position: str = typer.Option("bottom", help="Expected subtitle position: bottom, center, or top."),
    taste_profile: str = typer.Option("auto", help="Motion taste profile: auto, clean_documentary, viral_commentary, tutorial_focus, cinematic_subtle, or high_energy_shorts."),
) -> None:
    initialize_runtime()
    if density not in {"low", "medium", "high"}:
        raise typer.BadParameter("density must be one of: low, medium, high")
    if intensity not in {"subtle", "medium", "high", "strong"}:
        raise typer.BadParameter("intensity must be one of: subtle, medium, high, strong")
    if subtitle_position not in {"bottom", "center", "top"}:
        raise typer.BadParameter("subtitle_position must be one of: bottom, center, top")
    if taste_profile not in {"auto", "clean_documentary", "viral_commentary", "tutorial_focus", "cinematic_subtle", "high_energy_shorts"}:
        raise typer.BadParameter("taste_profile must be one of: auto, clean_documentary, viral_commentary, tutorial_focus, cinematic_subtle, high_energy_shorts")
    state = ProjectState.load(project)
    direct_auto_effects(
        state,
        density=density,
        intensity=intensity,
        max_effects=max_effects,
        include_style_effects=include_style_effects,
        subtitle_position=subtitle_position,
        taste_profile=taste_profile,
    )


@app.command()
def color_grade(
    project: str = typer.Option(..., help="Project id."),
    look: str = typer.Option(
        "auto",
        help="Look: auto, natural, vibrant, cinematic, warm, cool, documentary, or punchy.",
    ),
    intensity: float = typer.Option(1.0, help="Grade strength from 0.0 to 1.5."),
    sample_count: int = typer.Option(9, help="Number of analysis frames to sample, clamped from 1 to 15."),
) -> None:
    initialize_runtime()
    if look not in {"auto", "natural", "vibrant", "cinematic", "warm", "cool", "documentary", "punchy"}:
        raise typer.BadParameter("look must be one of: auto, natural, vibrant, cinematic, warm, cool, documentary, punchy")
    if intensity < 0.0 or intensity > 1.5:
        raise typer.BadParameter("intensity must be between 0.0 and 1.5")
    state = ProjectState.load(project)
    direct_color_grade(state, look=look, intensity=intensity, sample_count=sample_count)


@app.command()
def youtube_shorts(
    url: str,
    count: int = typer.Option(3, help="Number of shorts to create."),
    min_duration_sec: float = typer.Option(20.0, help="Minimum duration per short."),
    max_duration_sec: float = typer.Option(45.0, help="Maximum duration per short."),
    target_platform: str = typer.Option(
        "youtube_shorts",
        help="Platform profile: youtube_shorts, tiktok, or instagram_reels.",
    ),
    include_compilation: bool = typer.Option(True, help="Also create a merged compilation."),
    subtitle_style: str | None = typer.Option(
        default=None,
        help="Subtitle style: clean_pop, creator_bold, cinematic, glass, karaoke_focus, or minimal.",
    ),
    name: str | None = typer.Option(default=None, help="Optional project name override."),
) -> None:
    initialize_runtime()
    if target_platform not in {"youtube_shorts", "tiktok", "instagram_reels"}:
        raise typer.BadParameter("target_platform must be one of: youtube_shorts, tiktok, instagram_reels")
    try:
        state = find_project_for_source_url(url) or create_project_from_youtube(
            url,
            name=name,
            provider_name=config.PROVIDER,
            model_name=create_provider(show_banner=False).model_name,
        )
    except Exception as exc:
        raise typer.BadParameter(f"Failed to prepare YouTube video: {exc}") from exc
    print_project_panel(state)
    direct_auto_shorts(
        state,
        count=count,
        min_duration_sec=min_duration_sec,
        max_duration_sec=max_duration_sec,
        target_platform=target_platform,
        include_compilation=include_compilation,
        subtitle_style=subtitle_style,
    )


if __name__ == "__main__":
    app()
