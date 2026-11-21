"""Interactive input and command discovery for Vex's editing shell."""
from __future__ import annotations

import sys
from collections.abc import Callable

from rich.console import Console


COMMANDS = {
    "/generate-video <brief>": "Generate a film from a brief",
    "/load <path or URL>": "Import footage or open its project",
    "/open <project id>": "Switch to a saved project",
    "/projects": "List saved projects",
    "/status": "Show the active project",
    "/timeline": "Show applied edits",
    "/creative-runs": "Inspect recent creative results",
    "/trace": "Show the last agent run",
    "/provider": "Show the active model",
    "/undo": "Undo the latest edit",
    "/redo": "Restore an undone edit",
    "/export <preset>": "Export the working cut",
    "/encode <request>": "Plan a delivery encode",
    "/color-grade [look]": "Grade the working cut",
    "/auto-effects": "Add motion accents",
    "/help": "Show commands and keys",
    "/quit": "Leave the session",
}


def interactive_prompt(console: Console, history: list[str]) -> Callable[[str], str] | None:
    if not (console.is_terminal and sys.stdin.isatty()):
        return None

    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.styles import Style

    class SlashCompleter(Completer):
        def get_completions(self, document, complete_event):
            before = document.text_before_cursor
            if not before.startswith("/") or any(char.isspace() for char in before):
                return
            for syntax, description in COMMANDS.items():
                command = syntax.split()[0]
                if command.startswith(before):
                    yield Completion(command, start_position=-len(before), display_meta=description)

    keys = KeyBindings()

    @keys.add("enter")
    def submit(event):
        buffer = event.current_buffer
        if buffer.complete_state and buffer.complete_state.current_completion:
            buffer.apply_completion(buffer.complete_state.current_completion)
            return
        buffer.validate_and_handle()

    @keys.add("escape", "enter")
    def newline(event):
        event.current_buffer.insert_text("\n")

    style = Style.from_dict({
        "brand": "ansiyellow bold",
        "project": "ansibrightblack",
        "bottom-toolbar": "noreverse ansibrightblack",
        "completion-menu.completion.current": "bg:ansiyellow ansiblack",
    })
    session = PromptSession(
        multiline=True,
        key_bindings=keys,
        history=InMemoryHistory(history[-40:]),
        completer=SlashCompleter(),
        complete_while_typing=True,
        style=style,
        bottom_toolbar=" Enter send  ·  Alt+Enter newline  ·  Ctrl+R history  ·  / commands ",
    )
    return lambda project: session.prompt([("class:brand", "vex "), ("class:project", project + " "), ("class:brand", "> ")])
