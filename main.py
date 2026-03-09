from __future__ import annotations

import os
import shutil
import uuid
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
from state import ProjectState, utc_now_iso
from tools import TOOL_EXECUTORS
from tools.export import load_presets

app = typer.Typer(help="Vex - AI-powered video editing agent.")
console = Console()


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


@app.callback()
def app_callback(
    version: bool = typer.Option(False, "--version", "-v", is_eager=True, help="Show version and exit."),
) -> None:
    if version:
        console.print(f"Vex v{config.VERSION}")
        raise typer.Exit()


def format_bytes(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


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


def print_project_panel(state: ProjectState) -> None:
    metadata = state.metadata
    table = Table.grid(padding=(0, 2))
    table.add_row("File:", Path(state.source_files[0]).name)
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


def run_repl(state: ProjectState, provider) -> None:
    agent = VideoAgent(state, provider)
    while True:
        try:
            user_input = console.input("[bold cyan]Vex > [/]")
        except KeyboardInterrupt:
            answer = console.input("\nSave and exit? [y/n] ").strip().lower()
            if answer.startswith("y"):
                state.save()
                console.print("Project saved. Goodbye.")
                return
            continue
        command = user_input.strip()
        if not command:
            continue
        if command in {"/quit", "/exit"}:
            state.save()
            console.print("Project saved. Goodbye.")
            return
        if command == "/help":
            console.print(
                "/status, /timeline, /undo, /redo, /export <preset>, /provider, /projects, /help, /quit"
            )
            continue
        if command == "/status":
            console.print(state.get_summary())
            continue
        if command == "/timeline":
            render_timeline(state)
            continue
        if command == "/provider":
            console.print(f"Active: {state.provider} / {state.model}")
            continue
        if command == "/projects":
            render_projects()
            continue
        if command == "/undo":
            result = TOOL_EXECUTORS["undo"]({}, state)
            state = result["updated_state"]
            agent.state = state
            console.print(result["message"])
            continue
        if command == "/redo":
            result = TOOL_EXECUTORS["redo"]({}, state)
            state = result["updated_state"]
            agent.state = state
            console.print(result["message"])
            continue
        if command.startswith("/export"):
            parts = command.split(maxsplit=1)
            if len(parts) != 2:
                console.print("Usage: /export <preset>")
                continue
            direct_export(state, parts[1].strip())
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


if __name__ == "__main__":
    app()
