from __future__ import annotations

import json
from pathlib import Path

import pytest
from rich.console import Console

import main
from plugin_api import PluginApiError, discover_plugins, load_plugin_tool_executors


def test_discover_plugins_loads_manifest(tmp_path: Path) -> None:
    plugin_dir = _write_plugin(tmp_path)

    plugins = discover_plugins([tmp_path])

    assert len(plugins) == 1
    assert plugins[0].name == "sample.plugin"
    assert plugins[0].tools[0].name == "hello_tool"
    assert plugins[0].plugin_dir == str(plugin_dir.resolve())


def test_load_plugin_tool_executors_runs_explicit_entrypoint(tmp_path: Path) -> None:
    plugin = discover_plugins([tmp_path / "missing"])
    assert plugin == []
    manifest = discover_plugins([_write_plugin(tmp_path)])[0]
    executors = load_plugin_tool_executors(manifest)

    result = executors["hello_tool"]({"name": "Vex"}, None)

    assert result["success"] is True
    assert result["message"] == "Hello Vex"
    assert result["tool_name"] == "hello_tool"
    assert result["plugin"]["replayable"] is True


def test_plugin_entrypoint_cannot_escape_plugin_directory(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "bad"
    plugin_dir.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("def execute(params, state): return {}\n", encoding="utf-8")
    (plugin_dir / "vex-plugin.json").write_text(
        json.dumps(
            {
                "name": "bad.plugin",
                "version": "1.0.0",
                "tools": [
                    {
                        "name": "bad_tool",
                        "entrypoint": "../outside.py",
                        "requires_project": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest = discover_plugins([plugin_dir])[0]

    with pytest.raises(PluginApiError, match="escaped"):
        load_plugin_tool_executors(manifest)


def test_render_plugins_table_lists_discovered_plugins(tmp_path: Path) -> None:
    plugins = discover_plugins([_write_plugin(tmp_path)])
    console = Console(record=True, width=140)

    console.print(main.render_plugins_table(plugins))

    output = console.export_text()
    assert "sample.plugin" in output
    assert "hello_tool" not in output
    assert "1" in output


def _write_plugin(tmp_path: Path) -> Path:
    plugin_dir = tmp_path / "sample"
    plugin_dir.mkdir()
    (plugin_dir / "tool.py").write_text(
        "\n".join(
            [
                "def execute(params, state):",
                "    name = params.get('name', 'world')",
                "    return {'success': True, 'message': f'Hello {name}'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (plugin_dir / "vex-plugin.json").write_text(
        json.dumps(
            {
                "name": "sample.plugin",
                "version": "1.0.0",
                "description": "Sample plugin",
                "tools": [
                    {
                        "name": "hello_tool",
                        "entrypoint": "tool.py",
                        "requires_project": False,
                        "replayable": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return plugin_dir
