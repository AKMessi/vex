from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable


ToolExecutor = Callable[[dict[str, Any], Any], dict[str, Any]]


@dataclass(frozen=True)
class ToolContract:
    name: str
    module_name: str
    function_name: str
    category: str
    mutates_project: bool = True
    requires_project: bool = True
    long_running: bool = False
    replayable: bool = False


def execute_tool_contract(contract: ToolContract, params: dict[str, Any], state: Any) -> dict[str, Any]:
    if contract.requires_project and state is None:
        return {
            "success": False,
            "message": f"{contract.name} requires a loaded project.",
            "suggestion": None,
            "updated_state": state,
            "tool_name": contract.name,
        }
    module = import_module(f"tools.{contract.module_name}")
    function = getattr(module, contract.function_name)
    return normalize_tool_result(function(params, state), contract=contract, state=state)


def executor_for(contract: ToolContract) -> ToolExecutor:
    def run(params: dict[str, Any], state: Any) -> dict[str, Any]:
        return execute_tool_contract(contract, params, state)

    return run


def normalize_tool_result(raw: object, *, contract: ToolContract, state: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {
            "success": False,
            "message": f"{contract.name} returned an invalid result.",
            "suggestion": None,
            "updated_state": state,
            "tool_name": contract.name,
        }

    result = dict(raw)
    result["success"] = bool(result.get("success"))
    result["message"] = str(result.get("message") or "")
    result["suggestion"] = result.get("suggestion")
    result["updated_state"] = result.get("updated_state", state)
    result["tool_name"] = str(result.get("tool_name") or contract.name)
    result.setdefault(
        "contract",
        {
            "category": contract.category,
            "mutates_project": contract.mutates_project,
            "long_running": contract.long_running,
            "replayable": contract.replayable,
        },
    )
    return result
