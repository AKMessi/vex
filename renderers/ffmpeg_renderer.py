from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import config
from engine import probe_video
from renderers.base import RenderedAsset, RendererStatus, VisualRenderer, VisualRendererError


def _theme_defaults(spec: dict[str, Any]) -> dict[str, str]:
    theme = dict(spec.get("theme") or {})
    defaults = {
        "background": "#0B1020",
        "panel_fill": "#13203A",
        "panel_stroke": "#60A5FA",
        "accent": "#F59E0B",
        "accent_secondary": "#38BDF8",
        "glow": "#1D4ED8",
        "eyebrow_fill": "#14324D",
        "eyebrow_text": "#E0F2FE",
        "grid": "#214668",
        "text_primary": "#F8FAFC",
        "text_secondary": "#CBD5E1",
    }
    defaults.update({key: str(value) for key, value in theme.items() if value})
    return defaults


def _safe_color(value: str, opacity: float | None = None) -> str:
    normalized = str(value or "#FFFFFF").strip()
    if normalized.startswith("#"):
        normalized = f"0x{normalized[1:]}"
    if opacity is None:
        return normalized
    return f"{normalized}@{max(0.0, min(opacity, 1.0)):.3f}"


def _scaled(value: float, width: int) -> int:
    return max(1, int(round(value * (width / 1920.0))))


def _safe_scene_name(spec_id: str) -> str:
    cleaned = "".join(char if char.isalnum() else "_" for char in spec_id).strip("_")
    return cleaned or "auto_visual"


def _common_font_candidates() -> list[Path]:
    windir = Path(os.environ.get("WINDIR", "C:/Windows"))
    return [
        windir / "Fonts" / "segoeui.ttf",
        windir / "Fonts" / "arial.ttf",
        windir / "Fonts" / "calibri.ttf",
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/Library/Fonts/Arial.ttf"),
    ]


def _find_font_path() -> Path | None:
    for candidate in _common_font_candidates():
        if candidate.is_file():
            return candidate
    return None


def _escape_filter_path(path: Path) -> str:
    return path.resolve().as_posix().replace(":", r"\:").replace("'", r"\'")


def _write_text_file(root: Path, name: str, content: str) -> Path:
    target = root / name
    target.write_text(str(content or " ").strip() or " ", encoding="utf-8")
    return target


def _drawtext(
    *,
    textfile: Path,
    fontfile: Path,
    fontsize: int,
    fontcolor: str,
    x: str,
    y: str,
    line_spacing: int = 8,
    box: bool = False,
    box_color: str | None = None,
    box_border: int = 0,
    borderw: int = 0,
    bordercolor: str | None = None,
) -> str:
    parts = [
        f"fontfile='{_escape_filter_path(fontfile)}'",
        f"textfile='{_escape_filter_path(textfile)}'",
        f"fontcolor={fontcolor}",
        f"fontsize={fontsize}",
        "expansion=none",
        f"line_spacing={line_spacing}",
        f"x={x}",
        f"y={y}",
    ]
    if box:
        parts.append("box=1")
        parts.append(f"boxcolor={box_color or '0x000000@0.0'}")
        parts.append(f"boxborderw={box_border}")
    if borderw > 0 and bordercolor:
        parts.append(f"borderw={borderw}")
        parts.append(f"bordercolor={bordercolor}")
    return "drawtext=" + ":".join(parts)


def _panel_filters(
    x: int,
    y: int,
    width: int,
    height: int,
    *,
    fill: str,
    stroke: str,
    stroke_width: int,
) -> list[str]:
    inner_x = x + stroke_width
    inner_y = y + stroke_width
    inner_width = max(1, width - stroke_width * 2)
    inner_height = max(1, height - stroke_width * 2)
    return [
        f"drawbox=x={x}:y={y}:w={width}:h={height}:color={stroke}:t=fill",
        f"drawbox=x={inner_x}:y={inner_y}:w={inner_width}:h={inner_height}:color={fill}:t=fill",
    ]


def _base_background_filters(theme: dict[str, str], width: int, height: int) -> list[str]:
    accent = _safe_color(theme["accent"], 0.18)
    stroke = _safe_color(theme["panel_stroke"], 0.14)
    accent_secondary = _safe_color(theme.get("accent_secondary", theme["panel_stroke"]), 0.1)
    glow = _safe_color(theme.get("glow", theme["panel_stroke"]), 0.12)
    return [
        f"drawbox=x=0:y=0:w={width}:h={height}:color={_safe_color(theme['background'])}:t=fill",
        f"drawbox=x=0:y=0:w={width}:h={max(_scaled(180, width), 120)}:color={stroke}:t=fill",
        f"drawbox=x=0:y={height - max(_scaled(220, width), 140)}:w={width}:h={max(_scaled(220, width), 140)}:color={accent}:t=fill",
        f"drawbox=x={-max(_scaled(120, width), 60)}:y={_scaled(90, width)}:w={max(_scaled(520, width), 280)}:h={max(_scaled(520, width), 280)}:color={glow}:t=fill",
        f"drawbox=x={width - max(_scaled(360, width), 200)}:y={height - max(_scaled(300, width), 180)}:w={max(_scaled(420, width), 220)}:h={max(_scaled(280, width), 160)}:color={accent_secondary}:t=fill",
    ]


def _header_filters(
    spec: dict[str, Any],
    theme: dict[str, str],
    width: int,
    text_root: Path,
    fontfile: Path,
) -> list[str]:
    filters: list[str] = []
    headline = str(spec.get("headline") or "").strip()
    eyebrow = str(spec.get("eyebrow") or "").strip()
    deck = str(spec.get("deck") or "").strip()
    left_x = max(_scaled(120, width), 48)
    top_y = max(_scaled(84, width), 40)
    filters.append(
        f"drawbox=x={left_x - _scaled(28, width)}:y={top_y - _scaled(12, width)}:w={max(_scaled(10, width), 6)}:h={max(_scaled(150, width), 90)}:color={_safe_color(theme['accent'])}:t=fill"
    )
    if eyebrow:
        eyebrow_file = _write_text_file(text_root, "eyebrow.txt", eyebrow.upper())
        eyebrow_width = max(_scaled(220, width), len(eyebrow) * max(_scaled(12, width), 8))
        filters.append(
            f"drawbox=x={left_x}:y={top_y}:w={eyebrow_width}:h={max(_scaled(54, width), 34)}:color={_safe_color(theme.get('eyebrow_fill', theme['panel_fill']))}:t=fill"
        )
        filters.append(
            _drawtext(
                textfile=eyebrow_file,
                fontfile=fontfile,
                fontsize=max(_scaled(24, width), 14),
                fontcolor=_safe_color(theme.get("eyebrow_text", theme["text_primary"])),
                x=str(left_x + max(_scaled(18, width), 10)),
                y=str(top_y + max(_scaled(12, width), 8)),
            )
        )
        top_y += max(_scaled(74, width), 46)
    if headline:
        headline_file = _write_text_file(text_root, "headline.txt", headline)
        filters.append(
            _drawtext(
                textfile=headline_file,
                fontfile=fontfile,
                fontsize=max(_scaled(54, width), 28),
                fontcolor=_safe_color(theme["text_primary"]),
                x=str(left_x),
                y=str(top_y),
            )
        )
        top_y += max(_scaled(76, width), 48)
    if deck:
        deck_file = _write_text_file(text_root, "deck.txt", deck)
        filters.append(
            _drawtext(
                textfile=deck_file,
                fontfile=fontfile,
                fontsize=max(_scaled(26, width), 16),
                fontcolor=_safe_color(theme["text_secondary"]),
                x=str(left_x),
                y=str(top_y),
                line_spacing=max(_scaled(8, width), 4),
            )
        )
    return filters


def _metric_callout_filters(
    spec: dict[str, Any],
    theme: dict[str, str],
    width: int,
    height: int,
    text_root: Path,
    fontfile: Path,
) -> list[str]:
    panel_w = int(width * 0.72)
    panel_h = int(height * 0.54)
    panel_x = int((width - panel_w) / 2)
    panel_y = int(height * 0.3)
    filters = _panel_filters(
        panel_x,
        panel_y,
        panel_w,
        panel_h,
        fill=_safe_color(theme["panel_fill"]),
        stroke=_safe_color(theme["panel_stroke"]),
        stroke_width=max(_scaled(6, width), 4),
    )
    filters.append(
        f"drawbox=x={panel_x}:y={panel_y + _scaled(40, width)}:w={max(_scaled(20, width), 12)}:h={panel_h - _scaled(80, width)}:color={_safe_color(theme['accent'])}:t=fill"
    )

    emphasis = _write_text_file(text_root, "emphasis.txt", spec.get("emphasis_text", spec.get("headline", "")))
    support = _write_text_file(text_root, "support.txt", "\n".join(spec.get("supporting_lines") or []))
    filters.extend(
        [
            _drawtext(
                textfile=emphasis,
                fontfile=fontfile,
                fontsize=max(_scaled(120, width), 54),
                fontcolor=_safe_color(theme["text_primary"]),
                x="(w-text_w)/2",
                y=f"{panel_y + _scaled(120, width)}",
                borderw=max(_scaled(2, width), 1),
                bordercolor=_safe_color(theme["panel_stroke"], 0.35),
            ),
            _drawtext(
                textfile=support,
                fontfile=fontfile,
                fontsize=max(_scaled(34, width), 20),
                fontcolor=_safe_color(theme["text_secondary"]),
                x=f"{panel_x + _scaled(120, width)}",
                y=f"{panel_y + _scaled(280, width)}",
                line_spacing=max(_scaled(12, width), 6),
            ),
        ]
    )
    return filters


def _keyword_stack_filters(
    spec: dict[str, Any],
    theme: dict[str, str],
    width: int,
    height: int,
    text_root: Path,
    fontfile: Path,
) -> list[str]:
    filters: list[str] = []
    footer = _write_text_file(text_root, "footer.txt", spec.get("footer_text", ""))
    keywords = list(spec.get("keywords") or [])[:4] or [spec.get("emphasis_text", "Key idea")]
    start_y = int(height * 0.34)
    box_w = int(width * 0.54)
    box_h = max(_scaled(126, width), 80)
    box_x = int((width - box_w) / 2)
    gap = max(_scaled(28, width), 16)
    for index, keyword in enumerate(keywords, start=1):
        box_y = start_y + (index - 1) * (box_h + gap)
        filters.extend(
            _panel_filters(
                box_x,
                box_y,
                box_w,
                box_h,
                fill=_safe_color(theme["panel_fill"]),
                stroke=_safe_color(theme["panel_stroke"]),
                stroke_width=max(_scaled(5, width), 3),
            )
        )
        textfile = _write_text_file(text_root, f"keyword_{index}.txt", keyword)
        filters.append(
            _drawtext(
                textfile=textfile,
                fontfile=fontfile,
                fontsize=max(_scaled(46, width), 24),
                fontcolor=_safe_color(theme["text_primary"]),
                x="(w-text_w)/2",
                y=f"{box_y + int(box_h * 0.24)}",
            )
        )
    filters.append(
        _drawtext(
            textfile=footer,
            fontfile=fontfile,
            fontsize=max(_scaled(30, width), 18),
            fontcolor=_safe_color(theme["text_secondary"]),
            x="(w-text_w)/2",
            y=f"{min(height - _scaled(120, width), start_y + len(keywords) * (box_h + gap) + _scaled(18, width))}",
        )
    )
    return filters


def _timeline_steps_filters(
    spec: dict[str, Any],
    theme: dict[str, str],
    width: int,
    height: int,
    text_root: Path,
    fontfile: Path,
) -> list[str]:
    filters: list[str] = []
    steps = list(spec.get("steps") or [])[:4] or [spec.get("headline", ""), spec.get("emphasis_text", ""), spec.get("footer_text", "")]
    count = max(1, len(steps))
    box_w = int(min(width * 0.22, (width * 0.82) / count))
    box_h = int(height * 0.26)
    total_w = count * box_w + max(0, count - 1) * max(_scaled(52, width), 20)
    start_x = int((width - total_w) / 2)
    y = int(height * 0.44)
    gap = max(_scaled(52, width), 20)
    for index, step in enumerate(steps, start=1):
        x = start_x + (index - 1) * (box_w + gap)
        filters.extend(
            _panel_filters(
                x,
                y,
                box_w,
                box_h,
                fill=_safe_color(theme["panel_fill"]),
                stroke=_safe_color(theme["panel_stroke"]),
                stroke_width=max(_scaled(5, width), 3),
            )
        )
        badge = _write_text_file(text_root, f"badge_{index}.txt", str(index))
        step_file = _write_text_file(text_root, f"step_{index}.txt", step)
        filters.append(
            _drawtext(
                textfile=badge,
                fontfile=fontfile,
                fontsize=max(_scaled(34, width), 20),
                fontcolor=_safe_color(theme["accent"]),
                x=f"{x + _scaled(24, width)}",
                y=f"{y + _scaled(20, width)}",
            )
        )
        filters.append(
            _drawtext(
                textfile=step_file,
                fontfile=fontfile,
                fontsize=max(_scaled(30, width), 18),
                fontcolor=_safe_color(theme["text_primary"]),
                x=f"{x + _scaled(28, width)}",
                y=f"{y + _scaled(82, width)}",
                line_spacing=max(_scaled(8, width), 4),
            )
        )
        if index < count:
            arrow_x = x + box_w
            arrow_y = y + int(box_h / 2)
            filters.append(
                f"drawbox=x={arrow_x + _scaled(10, width)}:y={arrow_y}:w={gap - _scaled(20, width)}:h={max(_scaled(6, width), 4)}:color={_safe_color(theme['accent'])}:t=fill"
            )
    return filters


def _comparison_split_filters(
    spec: dict[str, Any],
    theme: dict[str, str],
    width: int,
    height: int,
    text_root: Path,
    fontfile: Path,
) -> list[str]:
    filters: list[str] = []
    panel_w = int(width * 0.32)
    panel_h = int(height * 0.48)
    gap = int(width * 0.07)
    left_x = int((width - (panel_w * 2 + gap)) / 2)
    right_x = left_x + panel_w + gap
    panel_y = int(height * 0.34)
    for prefix, panel_x, label_text, detail_text in (
        ("left", left_x, spec.get("left_label", "Before"), spec.get("left_detail", "")),
        ("right", right_x, spec.get("right_label", "After"), spec.get("right_detail", "")),
    ):
        filters.extend(
            _panel_filters(
                panel_x,
                panel_y,
                panel_w,
                panel_h,
                fill=_safe_color(theme["panel_fill"]),
                stroke=_safe_color(theme["panel_stroke"]),
                stroke_width=max(_scaled(5, width), 3),
            )
        )
        label_file = _write_text_file(text_root, f"{prefix}_label.txt", label_text)
        detail_file = _write_text_file(text_root, f"{prefix}_detail.txt", detail_text)
        filters.append(
            _drawtext(
                textfile=label_file,
                fontfile=fontfile,
                fontsize=max(_scaled(44, width), 24),
                fontcolor=_safe_color(theme["text_primary"]),
                x=f"{panel_x + _scaled(32, width)}",
                y=f"{panel_y + _scaled(38, width)}",
            )
        )
        filters.append(
            _drawtext(
                textfile=detail_file,
                fontfile=fontfile,
                fontsize=max(_scaled(30, width), 18),
                fontcolor=_safe_color(theme["text_secondary"]),
                x=f"{panel_x + _scaled(32, width)}",
                y=f"{panel_y + _scaled(120, width)}",
                line_spacing=max(_scaled(10, width), 4),
            )
        )
    versus = _write_text_file(text_root, "versus.txt", "VS")
    filters.append(
        _drawtext(
            textfile=versus,
            fontfile=fontfile,
            fontsize=max(_scaled(52, width), 26),
            fontcolor=_safe_color(theme["accent"]),
            x="(w-text_w)/2",
            y=f"{panel_y + int(panel_h * 0.42)}",
        )
    )
    return filters


def _quote_focus_filters(
    spec: dict[str, Any],
    theme: dict[str, str],
    width: int,
    height: int,
    text_root: Path,
    fontfile: Path,
) -> list[str]:
    quote = _write_text_file(text_root, "quote.txt", spec.get("quote_text", spec.get("headline", "")))
    footer = _write_text_file(text_root, "footer.txt", spec.get("footer_text", ""))
    bar_w = max(_scaled(18, width), 10)
    bar_h = int(height * 0.34)
    bar_y = int(height * 0.36)
    return [
        f"drawbox=x={_scaled(220, width)}:y={bar_y}:w={bar_w}:h={bar_h}:color={_safe_color(theme['accent'])}:t=fill",
        f"drawbox=x={width - _scaled(220, width) - bar_w}:y={bar_y}:w={bar_w}:h={bar_h}:color={_safe_color(theme['accent'])}:t=fill",
        _drawtext(
            textfile=quote,
            fontfile=fontfile,
            fontsize=max(_scaled(68, width), 34),
            fontcolor=_safe_color(theme["text_primary"]),
            x=str(max(_scaled(180, width), 80)),
            y=f"{int(height * 0.4)}",
            line_spacing=max(_scaled(12, width), 6),
        ),
        _drawtext(
            textfile=footer,
            fontfile=fontfile,
            fontsize=max(_scaled(28, width), 18),
            fontcolor=_safe_color(theme["text_secondary"]),
            x=str(max(_scaled(180, width), 80)),
            y=f"{int(height * 0.72)}",
        ),
    ]


def _stat_grid_filters(
    spec: dict[str, Any],
    theme: dict[str, str],
    width: int,
    height: int,
    text_root: Path,
    fontfile: Path,
) -> list[str]:
    filters: list[str] = []
    metrics = [spec.get("emphasis_text", "Key stat")]
    metrics.extend(list(spec.get("supporting_lines") or [])[:3])
    while len(metrics) < 4:
        metrics.append((list(spec.get("keywords") or ["Insight"]) + ["Insight"])[len(metrics) - 1])
    cell_w = int(width * 0.25)
    cell_h = int(height * 0.2)
    start_x = int(width * 0.22)
    start_y = int(height * 0.38)
    gap_x = int(width * 0.03)
    gap_y = int(height * 0.04)
    for index, metric in enumerate(metrics[:4], start=1):
        row = (index - 1) // 2
        col = (index - 1) % 2
        x = start_x + col * (cell_w + gap_x)
        y = start_y + row * (cell_h + gap_y)
        filters.extend(
            _panel_filters(
                x,
                y,
                cell_w,
                cell_h,
                fill=_safe_color(theme["panel_fill"]),
                stroke=_safe_color(theme["panel_stroke"]),
                stroke_width=max(_scaled(5, width), 3),
            )
        )
        text_file = _write_text_file(text_root, f"metric_{index}.txt", metric)
        filters.append(
            _drawtext(
                textfile=text_file,
                fontfile=fontfile,
                fontsize=max(_scaled(34, width), 20),
                fontcolor=_safe_color(theme["text_primary"]),
                x=f"{x + _scaled(28, width)}",
                y=f"{y + _scaled(46, width)}",
                line_spacing=max(_scaled(8, width), 4),
            )
        )
    return filters


class FFmpegRenderer(VisualRenderer):
    name = "ffmpeg"
    supported_templates = {
        "metric_callout",
        "keyword_stack",
        "timeline_steps",
        "comparison_split",
        "quote_focus",
        "stat_grid",
    }

    def availability(self) -> RendererStatus:
        if shutil.which(config.FFMPEG_PATH) is None:
            return RendererStatus(False, "FFmpeg is not available in PATH.")
        if _find_font_path() is None:
            return RendererStatus(False, "No compatible system font was found for FFmpeg drawtext.")
        return RendererStatus(True, "")

    def score_spec(self, spec: dict[str, Any]) -> float:
        if not self.supports(spec):
            return -1.0
        template = str(spec.get("template") or "")
        composition = str(spec.get("composition_mode") or "")
        style_pack = str(spec.get("style_pack") or "")
        visual_hint = str(spec.get("visual_type_hint") or "")
        importance = float(spec.get("importance") or 0.5)
        score = 0.56
        if template in {"metric_callout", "quote_focus", "keyword_stack", "stat_grid"}:
            score += 0.12
        if composition == "picture_in_picture":
            score += 0.18
        if composition == "replace":
            score -= 0.08
        if style_pack in {"editorial_clean", "product_ui"}:
            score += 0.05
        if visual_hint in {"product_ui", "cutaway"}:
            score += 0.1
        score += importance * 0.03
        return round(score, 3)

    def render(
        self,
        spec: dict[str, Any],
        render_root: Path,
        width: int,
        height: int,
        fps: float,
    ) -> RenderedAsset:
        status = self.availability()
        if not status.available:
            raise VisualRendererError(status.reason)
        if not self.supports(spec):
            raise VisualRendererError(f"FFmpeg renderer does not support template {spec.get('template')!r}.")

        spec_id = str(spec.get("visual_id") or spec.get("id") or "visual")
        scene_name = _safe_scene_name(spec_id)
        job_dir = render_root / spec_id
        text_root = job_dir / "texts"
        job_dir.mkdir(parents=True, exist_ok=True)
        text_root.mkdir(parents=True, exist_ok=True)
        fontfile = _find_font_path()
        if fontfile is None:
            raise VisualRendererError("No compatible system font was found for FFmpeg drawtext.")

        duration = max(float(spec.get("duration") or 2.0), 1.0)
        theme = _theme_defaults(spec)
        template = str(spec.get("template") or "quote_focus").strip().lower()
        filters = _base_background_filters(theme, width, height)
        filters.extend(_header_filters(spec, theme, width, text_root, fontfile))
        template_map = {
            "metric_callout": _metric_callout_filters,
            "keyword_stack": _keyword_stack_filters,
            "timeline_steps": _timeline_steps_filters,
            "comparison_split": _comparison_split_filters,
            "quote_focus": _quote_focus_filters,
            "stat_grid": _stat_grid_filters,
        }
        filters.extend(template_map[template](spec, theme, width, height, text_root, fontfile))
        fade_in = min(max(duration * 0.18, 0.18), 0.45)
        fade_out = min(max(duration * 0.18, 0.18), 0.45)
        fade_out_start = max(duration - fade_out, 0.0)
        filters.append(f"fade=t=in:st=0:d={fade_in:.3f}")
        filters.append(f"fade=t=out:st={fade_out_start:.3f}:d={fade_out:.3f}")

        output_path = job_dir / f"{scene_name}.mp4"
        script_path = job_dir / "ffmpeg_filtergraph.txt"
        script_path.write_text(",".join(filters), encoding="utf-8")
        command = [
            config.FFMPEG_PATH,
            "-f",
            "lavfi",
            "-i",
            f"color=c={_safe_color(theme['background'])}:s={width}x{height}:d={duration:.3f}:r={max(15, int(round(fps)))}",
            "-vf",
            script_path.read_text(encoding="utf-8"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-y",
            str(output_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0 or not output_path.is_file():
            stderr = (result.stderr or result.stdout or "").strip()
            raise VisualRendererError(f"FFmpeg renderer failed for {spec_id}: {stderr}")
        metadata = probe_video(str(output_path))
        return RenderedAsset(
            asset_path=str(output_path),
            width=int(metadata.get("width") or width),
            height=int(metadata.get("height") or height),
            duration_sec=float(metadata.get("duration_sec") or duration),
            renderer=self.name,
            job_dir=str(job_dir),
            script_path=str(script_path),
        )
