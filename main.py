from __future__ import annotations

import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

import config
from agent import AgentLoopError, VideoAgent
from engine import check_disk_space, estimate_output_size, export as export_media, probe_video
from providers import get_provider
from sources import download_youtube_video, extract_youtube_url, normalize_source_url
from state import ProjectState, utc_now_iso
from tools import TOOL_EXECUTORS
from tools.export import load_presets

app = typer.Typer(help="Vex - AI-powered video editing agent.")
console = Console()
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".flv"}


def initialize_runtime() -> None:
    config.validate_config()


def print_banner(model_name: str) -> None:
    banner = (
        "  __     _______  __\n"
        "  \\ \\   / / ____| \\ \\\n"
        "   \\ \\_/ /|  _|    \\ \\\n"
        "    \\   / | |___   / /\n"
        "     \\_/  |_____| /_/\n\n"
        f"  v{config.VERSION}  |  {model_name}  |  multi-provider ready"
    )
    console.print(Panel.fit(banner, border_style="cyan", title="Vex"))


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
        if len(projects) == 1:
            state = ProjectState.load(projects[0]["project_id"])
            console.print(
                f"Resuming: [bold]{state.project_name}[/] (last edited {format_relative_time(state.updated_at)} ago)"
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


def strip_wrapping_quotes(value: str) -> str:
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {'"', "'"}:
        return stripped[1:-1]
    return stripped


def is_video_path(path: str) -> bool:
    candidate = os.path.abspath(strip_wrapping_quotes(path))
    return os.path.isfile(candidate) and Path(candidate).suffix.lower() in VIDEO_EXTENSIONS


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
    metadata = state.metadata
    table = Table.grid(padding=(0, 2))
    table.add_row("File:", Path(state.source_files[0]).name)
    source_url = str((state.artifacts or {}).get("source_url") or "").strip()
    if source_url:
        table.add_row("Source URL:", source_url)
    table.add_row(
        "Duration:",
        f"{metadata.get('duration_sec', 0.0):.2f}s  |  {metadata.get('width', 0)}x{metadata.get('height', 0)}  |  {metadata.get('fps', 0)} fps",
    )
    table.add_row("Size:", format_bytes(metadata.get("size_bytes", 0)))
    table.add_row("Provider:", f"{state.provider} / {state.model}")
    table.add_row("Timeline:", f"{len(state.timeline)} operations")
    console.print(Panel(table, title=f"Project: {state.project_name}", border_style="green"))


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


def direct_auto_shorts(
    state: ProjectState,
    count: int,
    min_duration_sec: float,
    max_duration_sec: float,
    target_platform: str,
    include_compilation: bool,
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
            },
            state,
        )
    if not result["success"]:
        console.print(result["message"], style="red")
        raise typer.Exit(code=1)
    console.print(result["message"])


def direct_auto_broll(
    state: ProjectState,
    max_overlays: int,
    min_overlay_sec: float,
    max_overlay_sec: float,
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
                "min_overlay_sec": min_overlay_sec,
                "max_overlay_sec": max_overlay_sec,
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
    max_visuals: int,
    min_visual_sec: float,
    max_visual_sec: float,
) -> None:
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        console=console,
        transient=True,
    )
    with progress:
        progress.add_task("Adding auto visuals...", total=None)
        result = TOOL_EXECUTORS["add_auto_visuals"](
            {
                "mode": mode,
                "renderer": renderer,
                "max_visuals": max_visuals,
                "min_visual_sec": min_visual_sec,
                "max_visual_sec": max_visual_sec,
            },
            state,
        )
    if not result["success"]:
        console.print(result["message"], style="red")
        raise typer.Exit(code=1)
    console.print(result["message"])


def run_repl(state: ProjectState | None, provider) -> None:
    agent = VideoAgent(state, provider) if state is not None else None
    while True:
        try:
            user_input = console.input("[bold cyan]Vex > [/]")
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
            console.print(
                "/status, /timeline, /undo, /redo, /export <preset>, /provider, /projects, /help, /quit"
            )
            continue
        if command == "/status":
            if state is None:
                console.print("No video loaded. Drop a file path or YouTube link in your message to get started.")
            else:
                console.print(state.get_summary())
            continue
        if command == "/timeline":
            if state is None:
                console.print("No video loaded. Drop a file path or YouTube link in your message to get started.")
            else:
                render_timeline(state)
            continue
        if command == "/provider":
            if state is None:
                console.print(f"Active: {config.PROVIDER} / {provider.model_name}")
            else:
                console.print(f"Active: {state.provider} / {state.model}")
            continue
        if command == "/projects":
            render_projects()
            continue
        if command == "/undo":
            if state is None:
                console.print("No video loaded. Drop a file path or YouTube link in your message to get started.")
                continue
            result = TOOL_EXECUTORS["undo"]({}, state)
            state = result["updated_state"]
            if agent is not None:
                agent.state = state
            console.print(result["message"])
            continue
        if command == "/redo":
            if state is None:
                console.print("No video loaded. Drop a file path or YouTube link in your message to get started.")
                continue
            result = TOOL_EXECUTORS["redo"]({}, state)
            state = result["updated_state"]
            if agent is not None:
                agent.state = state
            console.print(result["message"])
            continue
        if command.startswith("/export"):
            if state is None:
                console.print("No video loaded. Drop a file path or YouTube link in your message to get started.")
                continue
            parts = command.split(maxsplit=1)
            if len(parts) != 2:
                console.print("Usage: /export <preset>")
                continue
            direct_export(state, parts[1].strip())
            continue

        detected_path = detect_video_path(command)
        detected_url = extract_youtube_url(command)
        if detected_path and not is_loaded_source(state, detected_path):
            console.print(f"Loading: {Path(detected_path).name}...")
            state = find_project_for_source(detected_path)
            if state is None:
                state = create_project(detected_path, None, config.PROVIDER, provider.model_name)
            agent = VideoAgent(state, provider)
        elif detected_url and not is_loaded_source_url(state, detected_url):
            console.print("Fetching video from YouTube...")
            state = find_project_for_source_url(detected_url)
            if state is None:
                try:
                    state = create_project_from_youtube(detected_url, None, config.PROVIDER, provider.model_name)
                except Exception as exc:
                    console.print(f"Failed to download YouTube video: {exc}", style="red")
                    continue
            agent = VideoAgent(state, provider)
        elif state is None:
            console.print("No video loaded. Drop a file path or YouTube link in your message to get started.")
            continue

        output = Text()
        progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            console=console,
            transient=True,
        )
        tasks: dict[str, int] = {}

        def stream_callback(chunk: str) -> None:
            output.append(chunk)
            live.update(Panel(output or Text(" "), title="Vex", border_style="cyan"))

        def tool_callback(event: str, name: str, ok: bool) -> None:
            if event == "start":
                tasks[name] = progress.add_task(f"{name}...", total=None)
            else:
                task_id = tasks.get(name)
                if task_id is not None:
                    progress.update(task_id, description=f"{name} {'done' if ok else 'failed'}")
                    progress.remove_task(task_id)

        try:
            with Live(Panel(Text(" "), title="Vex", border_style="cyan"), console=console, refresh_per_second=8) as live:
                with progress:
                    response = agent.run(
                        command,
                        stream_callback=stream_callback,
                        tool_callback=tool_callback,
                    )
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
        response = agent.run(instruction)
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


@app.command()
def export(
    preset_name: str,
    project: str = typer.Option(..., help="Project id."),
    output: str | None = typer.Option(default=None, help="Custom output path."),
) -> None:
    initialize_runtime()
    state = ProjectState.load(project)
    direct_export(state, preset_name, output)


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
    )


@app.command()
def auto_broll(
    project: str = typer.Option(..., help="Project id."),
    max_overlays: int = typer.Option(5, help="Maximum number of stock inserts to add."),
    min_overlay_sec: float = typer.Option(1.2, help="Minimum duration of each insert."),
    max_overlay_sec: float = typer.Option(2.8, help="Maximum duration of each insert."),
) -> None:
    initialize_runtime()
    state = ProjectState.load(project)
    direct_auto_broll(
        state,
        max_overlays=max_overlays,
        min_overlay_sec=min_overlay_sec,
        max_overlay_sec=max_overlay_sec,
    )


@app.command()
def auto_visuals(
    project: str = typer.Option(..., help="Project id."),
    mode: str = typer.Option("generated_only", help="generated_only, hybrid, or stock_only."),
    renderer: str = typer.Option("manim", help="Renderer backend. Currently: manim."),
    max_visuals: int = typer.Option(4, help="Maximum number of generated visuals to add."),
    min_visual_sec: float = typer.Option(1.4, help="Minimum duration of each generated visual."),
    max_visual_sec: float = typer.Option(3.6, help="Maximum duration of each generated visual."),
) -> None:
    initialize_runtime()
    if mode not in {"generated_only", "hybrid", "stock_only"}:
        raise typer.BadParameter("mode must be one of: generated_only, hybrid, stock_only")
    if renderer not in {"manim"}:
        raise typer.BadParameter("renderer must be: manim")
    state = ProjectState.load(project)
    direct_auto_visuals(
        state,
        mode=mode,
        renderer=renderer,
        max_visuals=max_visuals,
        min_visual_sec=min_visual_sec,
        max_visual_sec=max_visual_sec,
    )


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
    )


if __name__ == "__main__":
    app()
