"""Conservative provider spend reservations and measured usage for visual runs."""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from pathlib import Path


def _amount(value: object) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("Provider prices and budgets must be numeric amounts.") from exc
    if not result.is_finite() or not 0 <= result <= 100_000:
        raise ValueError("Provider prices and budgets must be finite, nonnegative amounts.")
    return result


def _ceil(value: Decimal) -> int:
    return int(value.to_integral_value(rounding=ROUND_CEILING))


def load_policy() -> tuple[int | None, dict[str, tuple[Decimal, Decimal]]]:
    ceiling = os.getenv("VEX_PROVIDER_MAX_USD", "").strip()
    limit = _ceil(_amount(ceiling) * 1_000_000) if ceiling else None
    raw = os.getenv("VEX_PROVIDER_RATES", "").strip()
    rates = json.loads(raw) if raw else {}
    if not isinstance(rates, dict):
        raise ValueError("VEX_PROVIDER_RATES must be a JSON object keyed by provider/model.")
    parsed = {}
    for name, row in rates.items():
        if not isinstance(row, dict) or set(row) != {"input", "output"}:
            raise ValueError("Each rate needs input and output prices per million tokens.")
        parsed[name] = (_amount(row["input"]), _amount(row["output"]))
    return limit, parsed


def cost_micro(rates: tuple[Decimal, Decimal] | None, inputs: int, outputs: int) -> int | None:
    if rates is None:
        return None
    return _ceil(rates[0] * inputs + rates[1] * outputs)


def request_usage(usage: dict) -> tuple[int, int] | None:
    def integer(*names: str) -> int | None:
        for name in names:
            row = usage.get(name)
            if type(row) is int and row >= 0:
                return row
        return None

    supplied_input = integer("prompt_tokens", "input_tokens")
    supplied_output = integer("completion_tokens", "output_tokens")
    total = integer("total_tokens")
    if supplied_input is not None and supplied_output is not None:
        return supplied_input, max(supplied_output, (total or 0) - supplied_input)
    return None


def transport_usage(raw: object) -> dict:
    if isinstance(raw, dict):
        usage = raw.get("usage")
        return usage if isinstance(usage, dict) else {}
    if metrics := getattr(raw, "usage_metadata", None):
        total = getattr(metrics, "total_token_count", None)
        prompt = getattr(metrics, "prompt_token_count", None)
        result = {"total_tokens": total}
        if type(total) is int and type(prompt) is int:
            result.update(input_tokens=prompt, output_tokens=max(0, total - prompt))
        return result
    if metrics := getattr(raw, "usage", None):
        tokens = getattr(metrics, "input_tokens", None)
        if type(tokens) is int:
            tokens += (getattr(metrics, "cache_creation_input_tokens", None) or 0)
            tokens += (getattr(metrics, "cache_read_input_tokens", None) or 0)
        return {"input_tokens": tokens, "output_tokens": getattr(metrics, "output_tokens", None)}
    return {}


def initialize_meter(db: sqlite3.Connection) -> None:
    db.execute(
        "CREATE TABLE IF NOT EXISTS model_usage ("
        "id TEXT PRIMARY KEY, run_id TEXT NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL, "
        "status TEXT NOT NULL, input_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL, "
        "cost_micro INTEGER, created REAL NOT NULL)"
    )


def read_usage(working_dir: Path) -> dict:
    path = working_dir.resolve() / ".visual-harness" / "runs.sqlite3"
    if not path.is_file():
        return {"requests": [], "estimated_total_usd": 0.0, "unknown_cost_requests": 0}
    with closing(sqlite3.connect(path)) as db:
        db.row_factory = sqlite3.Row
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name='model_usage'").fetchone():
            return {"requests": [], "estimated_total_usd": 0.0, "unknown_cost_requests": 0}
        history = [dict(row) for row in db.execute("SELECT * FROM model_usage ORDER BY created, id")]
    costs = [row.pop("cost_micro") for row in history]
    for row, cost in zip(history, costs, strict=True):
        row["estimated_usd"] = cost / 1_000_000 if cost is not None else None
    return {"requests": history, "estimated_total_usd": sum(cost or 0 for cost in costs) / 1_000_000,
            "unknown_cost_requests": costs.count(None)}
