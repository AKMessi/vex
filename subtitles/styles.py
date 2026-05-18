from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any


@dataclass(frozen=True)
class SubtitleStyle:
    style_id: str
    display_name: str
    font_name: str = "Arial"
    font_size_scale: float = 0.052
    min_font_size: int = 22
    max_font_size: int = 68
    primary_color: str = "#FFFFFF"
    secondary_color: str = "#FACC15"
    outline_color: str = "#050816"
    back_color: str = "#050816"
    background_opacity: float = 0.0
    bold: bool = True
    italic: bool = False
    border_style: int = 1
    outline: float = 3.0
    shadow: float = 0.0
    spacing: float = 0.0
    wrap_style: int = 2
    margin_l_ratio: float = 0.09
    margin_r_ratio: float = 0.09
    margin_v_ratio: float = 0.07
    max_chars_per_line: int = 25
    max_lines: int = 2
    max_words_per_caption: int = 7
    max_duration_sec: float = 2.4
    fade_in_ms: int = 80
    fade_out_ms: int = 80
    case: str = "normal"
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PRESETS: dict[str, SubtitleStyle] = {
    "clean_pop": SubtitleStyle(
        style_id="clean_pop",
        display_name="Clean Pop",
        font_size_scale=0.054,
        primary_color="#F8FAFC",
        secondary_color="#FACC15",
        outline_color="#020617",
        back_color="#020617",
        background_opacity=0.62,
        border_style=3,
        outline=12.0,
        shadow=0.0,
        margin_v_ratio=0.076,
        max_chars_per_line=24,
        max_words_per_caption=7,
        description="Modern creator captions with a dark translucent backplate and crisp white type.",
    ),
    "creator_bold": SubtitleStyle(
        style_id="creator_bold",
        display_name="Creator Bold",
        font_size_scale=0.061,
        primary_color="#FFFFFF",
        secondary_color="#FFE66D",
        outline_color="#000000",
        back_color="#000000",
        background_opacity=0.0,
        border_style=1,
        outline=4.8,
        shadow=1.2,
        margin_l_ratio=0.07,
        margin_r_ratio=0.07,
        margin_v_ratio=0.082,
        max_chars_per_line=20,
        max_words_per_caption=5,
        case="uppercase",
        description="Large, punchy short-form captions with thick readable outlines.",
    ),
    "cinematic": SubtitleStyle(
        style_id="cinematic",
        display_name="Cinematic",
        font_size_scale=0.044,
        min_font_size=18,
        max_font_size=54,
        primary_color="#F8FAFC",
        secondary_color="#EAB308",
        outline_color="#020617",
        back_color="#000000",
        background_opacity=0.0,
        border_style=1,
        outline=2.2,
        shadow=1.6,
        spacing=0.25,
        margin_l_ratio=0.12,
        margin_r_ratio=0.12,
        margin_v_ratio=0.09,
        max_chars_per_line=34,
        max_words_per_caption=9,
        fade_in_ms=120,
        fade_out_ms=120,
        description="Restrained film-style lower thirds with subtle shadow and longer line length.",
    ),
    "glass": SubtitleStyle(
        style_id="glass",
        display_name="Glass",
        font_size_scale=0.052,
        primary_color="#F8FAFC",
        secondary_color="#67E8F9",
        outline_color="#38BDF8",
        back_color="#07111F",
        background_opacity=0.48,
        border_style=3,
        outline=14.0,
        shadow=0.0,
        margin_l_ratio=0.1,
        margin_r_ratio=0.1,
        margin_v_ratio=0.075,
        max_chars_per_line=24,
        max_words_per_caption=7,
        description="Premium glass-panel captions for explainer visuals and clean screen recordings.",
    ),
    "karaoke_focus": SubtitleStyle(
        style_id="karaoke_focus",
        display_name="Karaoke Focus",
        font_size_scale=0.058,
        primary_color="#FDE047",
        secondary_color="#FFFFFF",
        outline_color="#111827",
        back_color="#030712",
        background_opacity=0.58,
        border_style=3,
        outline=13.0,
        shadow=0.0,
        margin_l_ratio=0.08,
        margin_r_ratio=0.08,
        margin_v_ratio=0.08,
        max_chars_per_line=20,
        max_words_per_caption=5,
        case="uppercase",
        description="High-energy yellow focus captions ready for future word-level karaoke timing.",
    ),
    "minimal": SubtitleStyle(
        style_id="minimal",
        display_name="Minimal",
        font_size_scale=0.046,
        primary_color="#FFFFFF",
        secondary_color="#FFFFFF",
        outline_color="#000000",
        back_color="#000000",
        background_opacity=0.0,
        border_style=1,
        outline=2.4,
        shadow=0.8,
        margin_l_ratio=0.12,
        margin_r_ratio=0.12,
        margin_v_ratio=0.08,
        max_chars_per_line=32,
        max_words_per_caption=9,
        description="Simple readable subtitles with no panel treatment.",
    ),
}

ALIASES = {
    "default": "clean_pop",
    "clean": "clean_pop",
    "pop": "clean_pop",
    "bold": "creator_bold",
    "creator": "creator_bold",
    "tiktok": "creator_bold",
    "reels": "creator_bold",
    "shorts": "creator_bold",
    "film": "cinematic",
    "movie": "cinematic",
    "premium": "glass",
    "glassmorphism": "glass",
    "karaoke": "karaoke_focus",
    "highlight": "karaoke_focus",
    "simple": "minimal",
    "classic": "minimal",
    "legacy": "minimal",
}


def list_subtitle_styles() -> list[dict[str, str]]:
    return [
        {
            "style": style.style_id,
            "name": style.display_name,
            "description": style.description,
        }
        for style in PRESETS.values()
    ]


def _style_key(style_name: str | None) -> str:
    normalized = str(style_name or "clean_pop").strip().lower().replace("-", "_").replace(" ", "_")
    return ALIASES.get(normalized, normalized)


def resolve_subtitle_style(
    style_name: str | None,
    *,
    font_size: int | None = None,
    font_color: str | None = None,
    outline_color: str | None = None,
    emphasis_color: str | None = None,
    background_opacity: float | None = None,
    max_words_per_caption: int | None = None,
    max_lines: int | None = None,
    case: str | None = None,
) -> SubtitleStyle:
    key = _style_key(style_name)
    if key not in PRESETS:
        supported = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unsupported subtitle style: {style_name}. Supported styles: {supported}")
    style = PRESETS[key]
    updates: dict[str, Any] = {}
    if font_size is not None:
        updates["min_font_size"] = int(font_size)
        updates["max_font_size"] = int(font_size)
    if font_color:
        updates["primary_color"] = str(font_color)
    if outline_color:
        updates["outline_color"] = str(outline_color)
    if emphasis_color:
        updates["secondary_color"] = str(emphasis_color)
    if background_opacity is not None:
        updates["background_opacity"] = max(0.0, min(float(background_opacity), 1.0))
    if max_words_per_caption is not None:
        updates["max_words_per_caption"] = max(1, int(max_words_per_caption))
    if max_lines is not None:
        updates["max_lines"] = max(1, min(int(max_lines), 3))
    if case:
        normalized_case = str(case).strip().lower()
        if normalized_case not in {"normal", "uppercase", "title"}:
            raise ValueError("Subtitle case must be one of: normal, uppercase, title")
        updates["case"] = normalized_case
    return replace(style, **updates) if updates else style
