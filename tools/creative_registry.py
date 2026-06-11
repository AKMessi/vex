from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import math
from typing import Any

from state import utc_now_iso


REGISTRY_VERSION = "creative-run-registry-v1"
REGISTRY_FILENAME = "creative_runs.json"
MAX_REGISTRY_RECORDS = 80
CREATIVE_POLICY_VERSION = "creative-quality-policy-v1"


@dataclass(frozen=True)
class CreativeRunRecord:
    run_id: str
    feature: str
    created_at: str
    manifest_path: str | None = None
    output_path: str | None = None
    graph_version: str | None = None
    quality_score: float | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.quality_score is not None:
            payload["quality_score"] = round(float(self.quality_score), 4)
        return payload


@dataclass(frozen=True)
class CreativePolicySnapshot:
    version: str
    feature: str
    run_count: int
    outcome_count: int
    minimum_samples: int
    global_quality: float
    renderer_adjustments: dict[str, float] = field(default_factory=dict)
    intent_renderer_adjustments: dict[str, float] = field(default_factory=dict)
    template_adjustments: dict[str, float] = field(default_factory=dict)
    sample_counts: dict[str, dict[str, int]] = field(default_factory=dict)

    def adjustment_for(
        self,
        *,
        renderer: object,
        intent_type: object = "",
        template: object = "",
    ) -> float:
        renderer_name = _safe_label(renderer)
        intent_name = _safe_label(intent_type)
        template_name = _safe_label(template)
        weighted: list[tuple[float, float]] = []
        if renderer_name in self.renderer_adjustments:
            weighted.append((self.renderer_adjustments[renderer_name], 1.0))
        intent_renderer_key = _intent_renderer_key(intent_name, renderer_name)
        if intent_renderer_key in self.intent_renderer_adjustments:
            weighted.append(
                (self.intent_renderer_adjustments[intent_renderer_key], 1.4)
            )
        if template_name in self.template_adjustments:
            weighted.append((self.template_adjustments[template_name], 0.6))
        if not weighted:
            return 0.0
        adjustment = sum(value * weight for value, weight in weighted) / sum(
            weight for _, weight in weighted
        )
        return round(max(-0.04, min(adjustment, 0.04)), 4)

    def explain_for(
        self,
        *,
        renderer: object,
        intent_type: object = "",
        template: object = "",
        available_renderers: set[str] | None = None,
    ) -> dict[str, Any]:
        renderer_name = _safe_label(renderer)
        intent_name = _safe_label(intent_type)
        template_name = _safe_label(template)
        renderer_names = sorted(available_renderers or {renderer_name})
        return {
            "version": self.version,
            "run_count": self.run_count,
            "outcome_count": self.outcome_count,
            "minimum_samples": self.minimum_samples,
            "global_quality": self.global_quality,
            "selected_renderer": renderer_name,
            "selection_adjustment": self.adjustment_for(
                renderer=renderer_name,
                intent_type=intent_name,
                template=template_name,
            ),
            "renderer_adjustments": {
                name: self.adjustment_for(
                    renderer=name,
                    intent_type=intent_name,
                    template=template_name,
                )
                for name in renderer_names
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def creative_registry_path(working_dir: str | Path) -> Path:
    return Path(working_dir) / REGISTRY_FILENAME


def load_creative_registry(working_dir: str | Path) -> dict[str, Any]:
    path = creative_registry_path(working_dir)
    if not path.is_file():
        return _empty_registry()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _empty_registry()
    if not isinstance(payload, dict):
        return _empty_registry()
    records = payload.get("runs")
    if not isinstance(records, list):
        records = []
    return {
        "version": str(payload.get("version") or REGISTRY_VERSION),
        "updated_at": str(payload.get("updated_at") or ""),
        "runs": [dict(item) for item in records if isinstance(item, dict)],
    }


def record_creative_run(
    *,
    working_dir: str | Path,
    feature: str,
    manifest_path: str | None = None,
    output_path: str | None = None,
    graph_version: str | None = None,
    quality_score: float | None = None,
    summary: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    created_at = utc_now_iso()
    run_id = _run_id(feature, created_at)
    record = CreativeRunRecord(
        run_id=run_id,
        feature=_safe_feature(feature),
        created_at=created_at,
        manifest_path=str(manifest_path) if manifest_path else None,
        output_path=str(output_path) if output_path else None,
        graph_version=str(graph_version) if graph_version else None,
        quality_score=float(quality_score) if quality_score is not None else None,
        summary=dict(summary or {}),
        artifacts=dict(artifacts or {}),
    )
    registry = load_creative_registry(working_dir)
    records = [dict(item) for item in registry.get("runs", []) if isinstance(item, dict)]
    records.append(record.to_dict())
    records = records[-MAX_REGISTRY_RECORDS:]
    payload = {
        "version": REGISTRY_VERSION,
        "updated_at": created_at,
        "runs": records,
    }
    path = creative_registry_path(working_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(path, payload)
    except OSError as exc:
        return {
            "registered": False,
            "registry_path": str(path),
            "error": str(exc),
            "record": record.to_dict(),
        }
    return {
        "registered": True,
        "registry_path": str(path),
        "record": record.to_dict(),
    }


def latest_creative_runs(
    working_dir: str | Path,
    *,
    feature: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    registry = load_creative_registry(working_dir)
    normalized_feature = _safe_feature(feature) if feature else None
    records = [
        dict(item)
        for item in registry.get("runs", [])
        if isinstance(item, dict)
        and (normalized_feature is None or _safe_feature(item.get("feature")) == normalized_feature)
    ]
    return list(reversed(records[-max(1, int(limit)):]))


def load_creative_policy(
    working_dir: str | Path,
    *,
    feature: str = "auto_visuals",
    limit: int = 30,
    minimum_samples: int = 3,
) -> CreativePolicySnapshot:
    records = latest_creative_runs(
        working_dir,
        feature=feature,
        limit=max(1, int(limit)),
    )
    safe_minimum = max(2, int(minimum_samples))
    outcomes: list[tuple[dict[str, Any], float]] = []
    for age, record in enumerate(records):
        summary = record.get("summary")
        if not isinstance(summary, dict):
            continue
        signals = summary.get("outcome_signals")
        if not isinstance(signals, list):
            continue
        recency_weight = 0.94 ** age
        for signal in signals[:64]:
            if isinstance(signal, dict):
                outcomes.append((dict(signal), recency_weight))

    global_total = 0.0
    global_weight = 0.0
    renderer_buckets: dict[str, list[tuple[float, float]]] = {}
    intent_renderer_buckets: dict[str, list[tuple[float, float]]] = {}
    template_buckets: dict[str, list[tuple[float, float]]] = {}
    for signal, recency_weight in outcomes:
        renderer = _safe_label(signal.get("renderer"))
        if not renderer:
            continue
        quality = _bounded_quality(signal.get("qa_score"), 0.0)
        passed = _safe_bool(signal.get("qa_passed"))
        utility = quality if passed else quality * 0.45
        global_total += utility * recency_weight
        global_weight += recency_weight
        renderer_buckets.setdefault(renderer, []).append((utility, recency_weight))
        intent = _safe_label(signal.get("intent_type"))
        if intent:
            intent_renderer_buckets.setdefault(
                _intent_renderer_key(intent, renderer),
                [],
            ).append((utility, recency_weight))
        template = _safe_label(signal.get("template"))
        if template:
            template_buckets.setdefault(template, []).append(
                (utility, recency_weight)
            )

    global_quality = (
        global_total / global_weight if global_weight > 0.0 else 0.72
    )
    renderer_adjustments, renderer_counts = _bucket_adjustments(
        renderer_buckets,
        global_quality=global_quality,
        minimum_samples=safe_minimum,
    )
    intent_renderer_adjustments, intent_renderer_counts = _bucket_adjustments(
        intent_renderer_buckets,
        global_quality=global_quality,
        minimum_samples=safe_minimum,
    )
    template_adjustments, template_counts = _bucket_adjustments(
        template_buckets,
        global_quality=global_quality,
        minimum_samples=safe_minimum,
    )
    return CreativePolicySnapshot(
        version=CREATIVE_POLICY_VERSION,
        feature=_safe_feature(feature),
        run_count=len(records),
        outcome_count=len(outcomes),
        minimum_samples=safe_minimum,
        global_quality=round(global_quality, 4),
        renderer_adjustments=renderer_adjustments,
        intent_renderer_adjustments=intent_renderer_adjustments,
        template_adjustments=template_adjustments,
        sample_counts={
            "renderer": renderer_counts,
            "intent_renderer": intent_renderer_counts,
            "template": template_counts,
        },
    )


def _bucket_adjustments(
    buckets: dict[str, list[tuple[float, float]]],
    *,
    global_quality: float,
    minimum_samples: int,
) -> tuple[dict[str, float], dict[str, int]]:
    adjustments: dict[str, float] = {}
    counts: dict[str, int] = {}
    prior_weight = 4.0
    for key, values in buckets.items():
        counts[key] = len(values)
        if len(values) < minimum_samples:
            continue
        weighted_total = sum(value * weight for value, weight in values)
        weight_total = sum(weight for _, weight in values)
        posterior = (
            weighted_total + global_quality * prior_weight
        ) / max(weight_total + prior_weight, 0.001)
        adjustment = max(
            -0.04,
            min((posterior - global_quality) * 0.35, 0.04),
        )
        adjustments[key] = round(adjustment, 4)
    return adjustments, counts


def _intent_renderer_key(intent_type: str, renderer: str) -> str:
    return f"{intent_type}|{renderer}"


def _safe_label(value: object) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    return "".join(
        character
        for character in text
        if character.isalnum() or character == "_"
    )[:96]


def _bounded_quality(value: object, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    if not math.isfinite(number):
        number = default
    return max(0.0, min(number, 1.0))


def _safe_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _empty_registry() -> dict[str, Any]:
    return {
        "version": REGISTRY_VERSION,
        "updated_at": "",
        "runs": [],
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(path)


def _run_id(feature: str, created_at: str) -> str:
    stamp = created_at.replace(":", "-").replace("+00:00", "Z")
    return f"{_safe_feature(feature)}:{stamp}"


def _safe_feature(value: Any) -> str:
    text = str(value or "creative").strip().lower().replace("-", "_")
    cleaned = "".join(ch for ch in text if ch.isalnum() or ch == "_").strip("_")
    return cleaned or "creative"


__all__ = [
    "CREATIVE_POLICY_VERSION",
    "CreativePolicySnapshot",
    "CreativeRunRecord",
    "REGISTRY_FILENAME",
    "REGISTRY_VERSION",
    "creative_registry_path",
    "latest_creative_runs",
    "load_creative_registry",
    "load_creative_policy",
    "record_creative_run",
]
