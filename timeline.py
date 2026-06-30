from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


PROJECT_STATE_SCHEMA_VERSION = 2
TIMELINE_OPERATION_SCHEMA_VERSION = 1


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class TimelineOperation:
    op: str
    params: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=utc_now_iso)
    result_file: str = ""
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    assets: list[str] = field(default_factory=list)
    previous_file: str = ""
    schema_version: int = TIMELINE_OPERATION_SCHEMA_VERSION
    op_id: str = ""

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *, index: int | None = None) -> "TimelineOperation":
        normalized = normalize_timeline_operation(payload, index=index)
        fields = cls.__dataclass_fields__
        return cls(**{key: value for key, value in normalized.items() if key in fields})

    def to_dict(self) -> dict[str, Any]:
        return normalize_timeline_operation(asdict(self))


def normalize_timeline_operation(raw: object, *, index: int | None = None) -> dict[str, Any]:
    payload = dict(raw) if isinstance(raw, Mapping) else {}
    normalized = dict(payload)
    op_name = str(payload.get("op") or "unknown").strip() or "unknown"

    normalized["schema_version"] = _coerce_schema_version(
        payload.get("schema_version"),
        TIMELINE_OPERATION_SCHEMA_VERSION,
    )
    normalized["op"] = op_name
    normalized["params"] = _coerce_mapping(payload.get("params"))
    normalized["timestamp"] = str(payload.get("timestamp") or utc_now_iso()).strip()
    normalized["result_file"] = str(payload.get("result_file") or "")
    normalized["description"] = str(payload.get("description") or "")
    normalized["metadata"] = _coerce_mapping(payload.get("metadata"))
    normalized["assets"] = _coerce_string_list(payload.get("assets"))
    normalized["previous_file"] = str(payload.get("previous_file") or "")

    existing_id = str(payload.get("op_id") or "").strip()
    normalized["op_id"] = existing_id or _build_operation_id(normalized, index=index)
    return normalized


def normalize_timeline(raw: object) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    return [
        normalize_timeline_operation(item, index=index)
        for index, item in enumerate(raw)
    ]


def migrate_project_payload(raw: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(raw)
    payload["schema_version"] = _coerce_schema_version(
        payload.get("schema_version"),
        PROJECT_STATE_SCHEMA_VERSION,
    )
    payload["timeline"] = normalize_timeline(payload.get("timeline"))
    payload["redo_stack"] = normalize_timeline(payload.get("redo_stack"))
    if not isinstance(payload.get("session_log"), list):
        payload["session_log"] = []
    if not isinstance(payload.get("metadata"), Mapping):
        payload["metadata"] = {}
    if not isinstance(payload.get("artifacts"), Mapping):
        payload["artifacts"] = {}
    return payload


def _coerce_schema_version(value: object, current: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return current
    return max(parsed, current)


def _coerce_mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _coerce_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item or "").strip()]


def _build_operation_id(payload: Mapping[str, Any], *, index: int | None) -> str:
    identity_payload = {
        key: value
        for key, value in payload.items()
        if key not in {"op_id"}
    }
    if index is not None:
        identity_payload["_legacy_index"] = index
    encoded = json.dumps(identity_payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return f"op_{hashlib.sha256(encoded).hexdigest()[:16]}"
