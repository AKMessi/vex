from __future__ import annotations

import config


def test_reload_settings_reports_invalid_integer_env(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("GENAI_TIMEOUT_SEC", "ninety")

    try:
        config.reload_settings()
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("Invalid integer environment values should fail during settings reload.")
    finally:
        monkeypatch.delenv("GENAI_TIMEOUT_SEC", raising=False)
        config.reload_settings()


def test_reload_settings_clamps_runtime_minimums(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("GENAI_TIMEOUT_SEC", "1")
    monkeypatch.setenv("ANTHROPIC_TIMEOUT_SEC", "2")
    monkeypatch.setenv("ENCODE_VALIDATION_TIMEOUT_SEC", "3")
    monkeypatch.setenv("MANIM_PREVIEW_TIMEOUT_SEC", "3")
    monkeypatch.setenv("MANIM_FINAL_TIMEOUT_SEC", "4")
    monkeypatch.setenv("LLM_REQUEST_MAX_RETRIES", "0")
    monkeypatch.setenv("LLM_RETRY_BASE_DELAY_SEC", "0.1")

    try:
        config.reload_settings()

        assert config.GENAI_TIMEOUT_SEC == 15
        assert config.ANTHROPIC_TIMEOUT_SEC == 15.0
        assert config.ENCODE_VALIDATION_TIMEOUT_SEC == 15
        assert config.MANIM_PREVIEW_TIMEOUT_SEC == 30
        assert config.MANIM_FINAL_TIMEOUT_SEC == 30
        assert config.LLM_REQUEST_MAX_RETRIES == 1
        assert config.LLM_RETRY_BASE_DELAY_SEC == 0.5
    finally:
        for name in (
            "GENAI_TIMEOUT_SEC",
            "ANTHROPIC_TIMEOUT_SEC",
            "ENCODE_VALIDATION_TIMEOUT_SEC",
            "MANIM_PREVIEW_TIMEOUT_SEC",
            "MANIM_FINAL_TIMEOUT_SEC",
            "LLM_REQUEST_MAX_RETRIES",
            "LLM_RETRY_BASE_DELAY_SEC",
        ):
            monkeypatch.delenv(name, raising=False)
        config.reload_settings()


def test_llm_manim_codegen_is_disabled_by_default(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.delenv("MANIM_ALLOW_LLM_CODEGEN", raising=False)

    try:
        config.reload_settings()

        assert config.MANIM_ALLOW_LLM_CODEGEN is False
    finally:
        config.reload_settings()


def test_hyperframes_render_timeout_can_be_disabled(monkeypatch) -> None:  # noqa: ANN001
    try:
        monkeypatch.setenv("HYPERFRAMES_RENDER_TIMEOUT_SEC", "0")
        config.reload_settings()
        assert config.HYPERFRAMES_RENDER_TIMEOUT_SEC == 0

        monkeypatch.setenv("HYPERFRAMES_RENDER_TIMEOUT_SEC", "1")
        config.reload_settings()
        assert config.HYPERFRAMES_RENDER_TIMEOUT_SEC == 30
    finally:
        monkeypatch.delenv("HYPERFRAMES_RENDER_TIMEOUT_SEC", raising=False)
        config.reload_settings()


def test_ffmpeg_timeouts_are_bounded_and_render_timeout_can_be_disabled(monkeypatch) -> None:  # noqa: ANN001
    try:
        monkeypatch.setenv("FFMPEG_PROBE_TIMEOUT_SEC", "1")
        monkeypatch.setenv("FFMPEG_RENDER_TIMEOUT_SEC", "1")
        config.reload_settings()
        assert config.FFMPEG_PROBE_TIMEOUT_SEC == 5
        assert config.FFMPEG_RENDER_TIMEOUT_SEC == 30

        monkeypatch.setenv("FFMPEG_RENDER_TIMEOUT_SEC", "0")
        config.reload_settings()
        assert config.FFMPEG_RENDER_TIMEOUT_SEC == 0
    finally:
        monkeypatch.delenv("FFMPEG_PROBE_TIMEOUT_SEC", raising=False)
        monkeypatch.delenv("FFMPEG_RENDER_TIMEOUT_SEC", raising=False)
        config.reload_settings()


def test_reload_settings_parses_boolean_hardening_flags(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("MANIM_ALLOW_LLM_CODEGEN", "true")

    try:
        config.reload_settings()

        assert config.MANIM_ALLOW_LLM_CODEGEN is True
    finally:
        monkeypatch.delenv("MANIM_ALLOW_LLM_CODEGEN", raising=False)
        config.reload_settings()
