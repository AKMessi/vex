"""Non-secret renderer settings shared by workers and stage cache keys."""

from __future__ import annotations

import config

POLICY_KEYS = (
    "HYPERFRAMES_QA_MODE",
    "HYPERFRAMES_MIN_QUALITY_SCORE",
    "HYPERFRAMES_RENDER_QUALITY",
    "HYPERFRAMES_ENABLE_VISION_QA",
    "HYPERFRAMES_ENABLE_COUNTERFACTUAL_QA",
    "HYPERFRAMES_VISION_MODEL",
    "HYPERFRAMES_VARIANT_COUNT",
    "HYPERFRAMES_PROOF_CANDIDATE_COUNT",
    "HYPERFRAMES_ENABLE_CEGIS",
    "HYPERFRAMES_MAX_REPAIR_ROUNDS",
    "HYPERFRAMES_MIN_REPAIR_DELTA",
    "HYPERFRAMES_BLIND_DECODER_MIN_SCORE",
    "HYPERFRAMES_MAX_CRITIC_FRAMES",
    "HYPERFRAMES_LINT_TIMEOUT_SEC",
    "HYPERFRAMES_RENDER_TIMEOUT_SEC",
    "REMOTION_RENDER_TIMEOUT_SEC",
    "REMOTION_RENDER_CONCURRENCY",
    "VISUAL_DIRECTOR_ENABLED",
    "VISUAL_DIRECTOR_VERIFICATION_MODE",
    "VISUAL_DIRECTOR_GROQ_VISION_MODEL",
    "VISUAL_DIRECTOR_VISION_MODEL",
    "VISUAL_DIRECTOR_CLAUDE_VISION_MODEL",
    "GROQ_VISUAL_MAX_TOKENS",
    "GROQ_REASONING_EFFORT",
)
CREDENTIAL_KEYS = ("GROQ_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY")


def renderer_policy() -> dict:
    return {
        "settings": {key: getattr(config, key, None) for key in POLICY_KEYS},
        "reasoning_efforts": {
            role: config.visual_reasoning_effort(role)
            for role in ["authoring", "repair", "verifier", "pairwise"]
        },
        "credentials_enabled": {
            key: bool(getattr(config, key, None)) for key in CREDENTIAL_KEYS
        },
    }


def apply_renderer_policy(policy: dict) -> None:
    import os

    for role, effort in policy.get("reasoning_efforts", {}).items():
        if role not in {
            "authoring",
            "repair",
            "verifier",
            "pairwise",
        } or effort not in {"none", "default", "low", "medium", "high"}:
            raise ValueError("Invalid renderer reasoning policy")
        os.environ[f"VISUAL_{role.upper()}_REASONING_EFFORT"] = effort
    for key, value in policy.get("settings", {}).items():
        if key not in POLICY_KEYS or type(value) not in (
            str,
            int,
            float,
            bool,
            type(None),
        ):
            raise ValueError("Invalid renderer worker setting")
        setattr(config, key, value)
    for key, enabled in policy.get("credentials_enabled", {}).items():
        if key not in CREDENTIAL_KEYS or type(enabled) is not bool:
            raise ValueError("Invalid renderer credential policy")
        if not enabled:
            setattr(config, key, "")
