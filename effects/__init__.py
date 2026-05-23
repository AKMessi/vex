from __future__ import annotations

from effects.schema import EffectInstance, EffectPlan

__all__ = [
    "EffectInstance",
    "EffectPlan",
    "build_subtitle_cards",
    "plan_subtitle_effects",
]


def __getattr__(name: str):
    if name == "build_subtitle_cards":
        from effects.signals import build_subtitle_cards

        return build_subtitle_cards
    if name == "plan_subtitle_effects":
        from effects.planner import plan_subtitle_effects

        return plan_subtitle_effects
    raise AttributeError(f"module 'effects' has no attribute {name!r}")
