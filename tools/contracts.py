from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Callable

from vex_runtime.locking import FileLockTimeout, exclusive_file_lock


ToolExecutor = Callable[[dict[str, Any], Any], dict[str, Any]]
PROJECT_MUTATION_LOCK_TIMEOUT_SEC = 5.0


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
    try:
        with _project_mutation_lock(contract, state):
            _refresh_project_state(contract, state)
            return _execute_tool_contract_locked(contract, params, state)
    except FileLockTimeout:
        return normalize_tool_result(
            {
                "success": False,
                "message": (
                    "Another project mutation is already running. "
                    "Wait for it to finish before starting this operation."
                ),
            },
            contract=contract,
            state=state,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:  # noqa: BLE001
        return normalize_tool_result(
            {
                "success": False,
                "message": str(exc) or f"Unable to prepare project state for {contract.name}.",
            },
            contract=contract,
            state=state,
        )


def _execute_tool_contract_locked(
    contract: ToolContract,
    params: dict[str, Any],
    state: Any,
) -> dict[str, Any]:
    snapshot = _capture_project_snapshot(contract, state)
    try:
        module = import_module(f"tools.{contract.module_name}")
        function = getattr(module, contract.function_name)
        result = normalize_tool_result(function(params, state), contract=contract, state=state)
        if not result["success"]:
            _restore_project_snapshot(state, snapshot)
        return result
    except (KeyboardInterrupt, SystemExit):
        _restore_project_snapshot(state, snapshot)
        raise
    except Exception as exc:  # noqa: BLE001
        _restore_project_snapshot(state, snapshot)
        return normalize_tool_result(
            {
                "success": False,
                "message": str(exc) or f"{type(exc).__name__} while running {contract.name}.",
            },
            contract=contract,
            state=state,
        )


def _project_mutation_lock(contract: ToolContract, state: Any):  # noqa: ANN202
    if not contract.mutates_project or state is None:
        return nullcontext()
    working_dir = str(getattr(state, "working_dir", "") or "").strip()
    if not working_dir:
        return nullcontext()
    return exclusive_file_lock(
        Path(working_dir) / ".project-mutation.lock",
        timeout_sec=PROJECT_MUTATION_LOCK_TIMEOUT_SEC,
        stale_after_sec=30.0,
    )


def _refresh_project_state(contract: ToolContract, state: Any) -> None:
    if not contract.mutates_project or state is None:
        return
    refresh = getattr(state, "refresh_from_disk", None)
    if callable(refresh):
        refresh()


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
    result["success"] = _coerce_success(result.get("success"))
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


def _capture_project_snapshot(contract: ToolContract, state: Any) -> dict[str, Any] | None:
    if not contract.mutates_project or state is None:
        return None
    capture = getattr(state, "capture_snapshot", None)
    if not callable(capture):
        return None
    snapshot = capture()
    return snapshot if isinstance(snapshot, dict) else None


def _restore_project_snapshot(state: Any, snapshot: dict[str, Any] | None) -> None:
    if snapshot is None or state is None:
        return
    restore = getattr(state, "restore_snapshot", None)
    if callable(restore):
        restore(snapshot)


def _coerce_success(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return False
