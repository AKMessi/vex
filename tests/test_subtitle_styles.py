from __future__ import annotations

from pathlib import Path

import engine
from subtitles import compile_subtitles_to_ass, list_subtitle_styles, resolve_subtitle_style
from tools import subtitles as subtitle_tool
from state import ProjectState, utc_now_iso


def test_subtitle_style_registry_exposes_production_presets() -> None:
    styles = {item["style"] for item in list_subtitle_styles()}

    assert {"clean_pop", "creator_bold", "cinematic", "glass", "karaoke_focus", "minimal"}.issubset(styles)
    assert resolve_subtitle_style("tiktok").style_id == "creator_bold"
    assert resolve_subtitle_style("premium").style_id == "glass"


def test_compile_subtitles_to_ass_writes_styled_split_events(tmp_path: Path) -> None:
    srt_path = tmp_path / "captions.srt"
    srt_path.write_text(
        "1\n00:00:00,000 --> 00:00:03,000\nThis caption should split into punchy readable chunks\n",
        encoding="utf-8",
    )
    ass_path = tmp_path / "captions.ass"

    plan = compile_subtitles_to_ass(
        srt_path,
        ass_path,
        width=1080,
        height=1920,
        style_name="creator_bold",
        position="bottom",
    )

    text = ass_path.read_text(encoding="utf-8")
    assert plan.source_cues == 1
    assert plan.rendered_events >= 2
    assert "PlayResX: 1080" in text
    assert "Style: Default,Arial" in text
    assert ",2," in text
    assert "{\\fad(" in text
    assert "THIS CAPTION" in text


def test_engine_burn_subtitles_uses_ass_compiler(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    commands: list[list[str]] = []
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"video")
    srt_path = tmp_path / "captions.srt"
    srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello world\n", encoding="utf-8")

    monkeypatch.setattr(engine, "probe_video", lambda _path: {"width": 1280, "height": 720})
    monkeypatch.setattr(engine, "_run_command", lambda command, _message: commands.append(command))

    output = engine.burn_subtitles(
        str(input_path),
        str(tmp_path),
        str(srt_path),
        style="glass",
        position="top",
        background_opacity=0.42,
    )

    ass_files = list(tmp_path.glob("*.ass"))
    assert output.endswith(".mp4")
    assert ass_files
    assert commands
    assert "ass='" in commands[0][commands[0].index("-vf") + 1]
    assert "subtitles=" not in commands[0][commands[0].index("-vf") + 1]


def test_glass_style_box_uses_backplate_color_not_accent_outline(tmp_path: Path) -> None:
    srt_path = tmp_path / "captions.srt"
    srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nBeautiful subtitle\n", encoding="utf-8")
    ass_path = tmp_path / "captions.ass"

    compile_subtitles_to_ass(srt_path, ass_path, width=640, height=360, style_name="glass")

    text = ass_path.read_text(encoding="utf-8")
    assert "&H00F8BD38" not in text
    assert text.count("&H851F1107") >= 2


def test_subtitle_tool_stores_style_params(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    state = _state(tmp_path)
    srt_path = tmp_path / "transcript.srt"
    srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello world\n", encoding="utf-8")

    def fake_burn_subtitles(*_args, **kwargs):
        assert kwargs["style"] == "cinematic"
        assert kwargs["max_words_per_caption"] == 4
        return str(tmp_path / "subtitled.mp4")

    monkeypatch.setattr(subtitle_tool, "burn_subtitles", fake_burn_subtitles)
    monkeypatch.setattr(subtitle_tool, "probe_video", lambda _path: {"width": 1920, "height": 1080})

    result = subtitle_tool.execute(
        {
            "style": "cinematic",
            "position": "bottom",
            "max_words_per_caption": 4,
            "background_opacity": 0.25,
        },
        state,
    )

    assert result["success"] is True
    params = state.timeline[-1]["params"]
    assert params["style"] == "cinematic"
    assert params["max_words_per_caption"] == 4
    assert params["background_opacity"] == 0.25


def test_subtitle_tool_returns_error_for_invalid_style_numbers(tmp_path: Path) -> None:
    state = _state(tmp_path)
    (tmp_path / "transcript.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")

    result = subtitle_tool.execute({"font_size": "large"}, state)

    assert result["success"] is False
    assert result["tool_name"] == "burn_subtitles"


def _state(tmp_path: Path) -> ProjectState:
    now = utc_now_iso()
    return ProjectState(
        project_id="subtitle-style-test",
        project_name="Subtitle Style Test",
        created_at=now,
        updated_at=now,
        source_files=[str(tmp_path / "source.mp4")],
        working_file=str(tmp_path / "working.mp4"),
        working_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        metadata={"duration_sec": 3.0, "width": 1920, "height": 1080, "fps": 30.0},
        provider="test",
        model="test",
    )
