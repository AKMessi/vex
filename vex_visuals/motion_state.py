"""Reference evaluator: mirrors visual_motion.mjs and is checked against Node."""
from __future__ import annotations
import math
from typing import Any


def spring_curve(t: float, config: dict[str, Any]) -> float:
    mass = max(.1, float(config.get("mass") or 1))
    stiffness = max(1, float(config.get("stiffness") or 140))
    damping = max(.1, float(config.get("damping") or 18))
    omega = math.sqrt(stiffness / mass)
    zeta = damping / (2 * math.sqrt(stiffness * mass))
    if zeta < 1:
        damped = omega * math.sqrt(1 - zeta * zeta)
        return 1 - math.exp(-zeta * omega * t) * (math.cos(damped * t) + zeta * omega / damped * math.sin(damped * t))
    if abs(zeta - 1) < .000001:
        return 1 - math.exp(-omega * t) * (1 + omega * t)
    root = math.sqrt(zeta * zeta - 1)
    a, b = -omega * (zeta - root), -omega * (zeta + root)
    return 1 - (b * math.exp(a * t) - a * math.exp(b * t)) / (b - a)


def motion_ease(value: float, name: str = "linear", config: dict[str, Any] | None = None) -> float:
    p = max(0., min(float(value), 1.))
    if p in (0., 1.):
        return p
    if name == "ease_in":
        return p ** 3
    if name == "ease_out":
        return 1 - (1 - p) ** 3
    if name == "ease_in_out":
        return p * p * (3 - 2 * p)
    if name.startswith("spring_"):
        settings = {"stiffness": 75, "damping": 12} if name == "spring_gentle" else {"stiffness": 180, "damping": 20}
        settings.update(config or {})
        return spring_curve(p, settings) / max(spring_curve(1, settings), .001)
    return p


def evaluate_track(tracks: list[dict[str, Any]], property_name: str, time: float, fallback: float = 0.) -> float:
    track = next((item for item in tracks if item.get("property") == property_name), {})
    keys = sorted(track.get("keyframes") or [], key=lambda key: key["t"])
    if not keys:
        return fallback
    if time <= keys[0]["t"]:
        return float(keys[0]["value"])
    if time >= keys[-1]["t"]:
        return float(keys[-1]["value"])
    for left, right in zip(keys, keys[1:]):
        if left["t"] <= time <= right["t"]:
            p = motion_ease((time - left["t"]) / max(right["t"] - left["t"], .000001), right.get("easing") or left.get("easing") or "linear", track.get("spring"))
            return float(left["value"]) + (float(right["value"]) - float(left["value"])) * p
    return fallback
