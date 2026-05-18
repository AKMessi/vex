from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


COMPILER_VERSION = "effects-ffmpeg-v1"
PLAN_VERSION = 1

CAMERA_EFFECT_TYPES = {
    "punch_in",
    "punch_out",
    "slow_push",
    "micro_pan",
    "snap_reframe",
    "impact_pulse",
    "freeze_accent",
    "subtle_shake",
}

STYLE_EFFECT_TYPES = {
    "vignette",
    "focus_blur",
    "flash_accent",
    "subtitle_highlight",
}

SUPPORTED_EFFECT_TYPES = CAMERA_EFFECT_TYPES | STYLE_EFFECT_TYPES

EFFECT_ALIASES = {
    "zoom": "punch_in",
    "zoom_in": "punch_in",
    "push_in": "punch_in",
    "zoom_out": "punch_out",
    "push_out": "punch_out",
    "pan": "micro_pan",
    "shake": "subtle_shake",
    "flash": "flash_accent",
    "highlight": "subtitle_highlight",
    "freeze": "freeze_accent",
}


@dataclass(frozen=True)
class EffectInstance:
    effect_id: str
    effect_type: str
    start: float
    end: float
    priority: float
    source_card_id: str = ""
    reason: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    modifiers: list[str] = field(default_factory=list)
    signals: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, float(self.end) - float(self.start))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["effect_type"] = normalize_effect_type(payload["effect_type"])
        payload["start"] = round(float(payload["start"]), 3)
        payload["end"] = round(float(payload["end"]), 3)
        payload["priority"] = round(float(payload["priority"]), 3)
        payload["modifiers"] = [
            normalize_effect_type(item)
            for item in payload.get("modifiers", [])
            if normalize_effect_type(item) in STYLE_EFFECT_TYPES
        ]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EffectInstance":
        effect_type = normalize_effect_type(str(payload.get("effect_type") or payload.get("type") or "punch_in"))
        if effect_type not in SUPPORTED_EFFECT_TYPES:
            raise ValueError(f"Unsupported effect type: {effect_type}")
        start = _coerce_float(payload.get("start"), 0.0)
        end = _coerce_float(payload.get("end"), start)
        modifiers = [
            normalized
            for item in list(payload.get("modifiers") or [])
            if (normalized := normalize_effect_type(str(item))) in STYLE_EFFECT_TYPES
        ]
        return cls(
            effect_id=str(payload.get("effect_id") or payload.get("id") or ""),
            effect_type=effect_type,
            start=start,
            end=end,
            priority=_coerce_float(payload.get("priority"), 0.0),
            source_card_id=str(payload.get("source_card_id") or payload.get("card_id") or ""),
            reason=str(payload.get("reason") or ""),
            params=dict(payload.get("params") or {}),
            modifiers=modifiers,
            signals=dict(payload.get("signals") or {}),
        )


@dataclass(frozen=True)
class EffectPlan:
    effects: list[EffectInstance]
    version: int = PLAN_VERSION
    compiler_version: str = COMPILER_VERSION
    source: str = "subtitle_emphasis"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": int(self.version),
            "compiler_version": self.compiler_version,
            "source": self.source,
            "metadata": dict(self.metadata),
            "effects": [effect.to_dict() for effect in self.effects],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EffectPlan":
        raw_effects = payload.get("effects") or []
        effects = [
            EffectInstance.from_dict(item)
            for item in raw_effects
            if isinstance(item, dict)
        ]
        return cls(
            effects=effects,
            version=int(payload.get("version") or PLAN_VERSION),
            compiler_version=str(payload.get("compiler_version") or COMPILER_VERSION),
            source=str(payload.get("source") or "subtitle_emphasis"),
            metadata=dict(payload.get("metadata") or {}),
        )


def normalize_effect_type(value: str) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return EFFECT_ALIASES.get(normalized, normalized)


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
