from __future__ import annotations

import importlib.util
import json
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable


PLUGIN_SCHEMA_VERSION = 1
PLUGIN_MANIFEST_NAME = "vex-plugin.json"
PLUGIN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
PLUGIN_TOOL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")


class PluginApiError(ValueError):
    pass


PluginExecutor = Callable[[dict[str, Any], Any], dict[str, Any]]


@dataclass(frozen=True)
class PluginToolSpec:
    name: str
    entrypoint: str
    function: str = "execute"
    category: str = "plugin"
    mutates_project: bool = True
    requires_project: bool = True
    long_running: bool = False
    replayable: bool = False
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PluginManifest:
    name: str
    version: str
    description: str
    plugin_dir: str
    manifest_path: str
    tools: list[PluginToolSpec] = field(default_factory=list)
    capabilities: dict[str, Any] = field(default_factory=dict)
    schema_version: int = PLUGIN_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tools"] = [tool.to_dict() for tool in self.tools]
        return payload


def default_plugin_dirs() -> list[Path]:
    env_value = os.environ.get("VEX_PLUGIN_PATH", "")
    roots = [Path(item) for item in env_value.split(os.pathsep) if item.strip()]
    roots.append(Path.cwd() / "plugins")
    return _dedupe_paths(roots)


def discover_plugins(paths: Iterable[str | Path] | None = None) -> list[PluginManifest]:
    manifests: list[PluginManifest] = []
    for root in paths or default_plugin_dirs():
        root_path = Path(root).expanduser().resolve(strict=False)
        if not root_path.exists():
            continue
        if (root_path / PLUGIN_MANIFEST_NAME).is_file():
            manifests.append(load_plugin_manifest(root_path / PLUGIN_MANIFEST_NAME))
            continue
        for manifest_path in sorted(root_path.glob(f"*/{PLUGIN_MANIFEST_NAME}")):
            manifests.append(load_plugin_manifest(manifest_path))
    manifests.sort(key=lambda item: (item.name.lower(), item.version))
    return manifests


def load_plugin_manifest(path: str | Path) -> PluginManifest:
    manifest_path = Path(path).expanduser().resolve(strict=True)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PluginApiError(f"Plugin manifest is invalid JSON: {manifest_path}") from exc
    if not isinstance(payload, Mapping):
        raise PluginApiError(f"Plugin manifest must be a JSON object: {manifest_path}")
    name = str(payload.get("name") or "").strip()
    if not PLUGIN_NAME_RE.fullmatch(name):
        raise PluginApiError(f"Invalid plugin name in {manifest_path}: {name!r}")
    tools = [_coerce_tool_spec(item, manifest_path=manifest_path) for item in payload.get("tools") or []]
    capabilities = payload.get("capabilities")
    return PluginManifest(
        name=name,
        version=str(payload.get("version") or "0.0.0"),
        description=str(payload.get("description") or ""),
        plugin_dir=str(manifest_path.parent),
        manifest_path=str(manifest_path),
        tools=tools,
        capabilities=dict(capabilities or {}) if isinstance(capabilities, Mapping) else {},
    )


def load_plugin_tool_executors(manifest: PluginManifest) -> dict[str, PluginExecutor]:
    executors: dict[str, PluginExecutor] = {}
    plugin_dir = Path(manifest.plugin_dir).expanduser().resolve(strict=True)
    for tool in manifest.tools:
        entrypoint = _resolve_entrypoint(plugin_dir, tool.entrypoint)
        function = _load_function(entrypoint, tool.function, plugin_name=manifest.name, tool_name=tool.name)
        executors[tool.name] = _wrap_plugin_executor(tool, function)
    return executors


def plugin_summary(manifest: PluginManifest) -> dict[str, Any]:
    return {
        "name": manifest.name,
        "version": manifest.version,
        "description": manifest.description,
        "plugin_dir": manifest.plugin_dir,
        "tool_count": len(manifest.tools),
        "tools": [tool.name for tool in manifest.tools],
        "capabilities": dict(manifest.capabilities),
    }


def _coerce_tool_spec(raw: object, *, manifest_path: Path) -> PluginToolSpec:
    if not isinstance(raw, Mapping):
        raise PluginApiError(f"Plugin tool entries must be objects: {manifest_path}")
    name = str(raw.get("name") or "").strip()
    if not PLUGIN_TOOL_RE.fullmatch(name):
        raise PluginApiError(f"Invalid plugin tool name in {manifest_path}: {name!r}")
    entrypoint = str(raw.get("entrypoint") or "").strip()
    if not entrypoint:
        raise PluginApiError(f"Plugin tool {name} is missing an entrypoint.")
    return PluginToolSpec(
        name=name,
        entrypoint=entrypoint,
        function=str(raw.get("function") or "execute"),
        category=str(raw.get("category") or "plugin"),
        mutates_project=bool(raw.get("mutates_project", True)),
        requires_project=bool(raw.get("requires_project", True)),
        long_running=bool(raw.get("long_running", False)),
        replayable=bool(raw.get("replayable", False)),
        description=str(raw.get("description") or ""),
    )


def _resolve_entrypoint(plugin_dir: Path, entrypoint: str) -> Path:
    candidate = (plugin_dir / entrypoint).expanduser().resolve(strict=True)
    plugin_text = os.path.normcase(os.path.abspath(str(plugin_dir)))
    candidate_text = os.path.normcase(os.path.abspath(str(candidate)))
    try:
        if os.path.commonpath([plugin_text, candidate_text]) != plugin_text:
            raise ValueError
    except ValueError as exc:
        raise PluginApiError("Plugin entrypoint escaped the plugin directory.") from exc
    if candidate.suffix.lower() != ".py":
        raise PluginApiError(f"Plugin entrypoint must be a Python file: {candidate}")
    return candidate


def _load_function(entrypoint: Path, function_name: str, *, plugin_name: str, tool_name: str):
    module_name = f"vex_plugin_{_safe_identifier(plugin_name)}_{_safe_identifier(tool_name)}"
    spec = importlib.util.spec_from_file_location(module_name, entrypoint)
    if spec is None or spec.loader is None:
        raise PluginApiError(f"Unable to load plugin entrypoint: {entrypoint}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, function_name, None)
    if not callable(function):
        raise PluginApiError(f"Plugin tool {tool_name} is missing callable {function_name}.")
    return function


def _wrap_plugin_executor(tool: PluginToolSpec, function) -> PluginExecutor:
    def run(params: dict[str, Any], state: Any) -> dict[str, Any]:
        if tool.requires_project and state is None:
            return {
                "success": False,
                "message": f"{tool.name} requires a loaded project.",
                "suggestion": None,
                "updated_state": state,
                "tool_name": tool.name,
            }
        raw = function(dict(params or {}), state)
        if not isinstance(raw, dict):
            return {
                "success": False,
                "message": f"{tool.name} returned an invalid result.",
                "suggestion": None,
                "updated_state": state,
                "tool_name": tool.name,
            }
        result = dict(raw)
        result["success"] = bool(result.get("success"))
        result["message"] = str(result.get("message") or "")
        result["suggestion"] = result.get("suggestion")
        result["updated_state"] = result.get("updated_state", state)
        result["tool_name"] = str(result.get("tool_name") or tool.name)
        result.setdefault(
            "plugin",
            {
                "category": tool.category,
                "mutates_project": tool.mutates_project,
                "long_running": tool.long_running,
                "replayable": tool.replayable,
            },
        )
        return result

    return run


def _safe_identifier(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_") or "plugin"


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        resolved = path.expanduser().resolve(strict=False)
        key = os.path.normcase(os.path.abspath(str(resolved)))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(resolved)
    return deduped
