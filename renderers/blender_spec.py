from __future__ import annotations

import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from renderers.base import VisualRendererError


LEGACY_BLENDER_TEMPLATES = {
    "quote_focus",
    "keyword_stack",
    "metric_callout",
}

THREE_D_BLENDER_TEMPLATES = {
    "three_d_title",
    "floating_3d_label",
    "object_orbit",
    "logo_reveal",
    "screen_pointer_3d",
    "data_tunnel",
    "product_model_spin",
}

SUPPORTED_BLENDER_TEMPLATES = LEGACY_BLENDER_TEMPLATES | THREE_D_BLENDER_TEMPLATES

POSITION_VALUES = {
    "center",
    "center_left",
    "center_right",
    "top_left",
    "top_right",
    "bottom_left",
    "bottom_right",
}

STYLE_VALUES = {
    "cinematic_dark",
    "clean_light",
    "neon",
    "glass",
    "studio",
}

CAMERA_MOTION_VALUES = {
    "static",
    "slow_push",
    "orbit",
    "handheld_subtle",
}

OBJECT_MOTION_VALUES = {
    "none",
    "spin_y",
    "float",
    "drop_in",
    "pulse",
}

ASSET_SUFFIXES = {".glb", ".gltf", ".obj", ".blend"}


@dataclass(frozen=True)
class BlenderVisualSpec:
    visual_id: str
    template: str
    composition_mode: str
    start_sec: float
    duration: float
    width: int
    height: int
    fps: float
    text: str
    headline: str
    subtext: str
    label: str
    position: str
    style: str
    camera_motion: str
    object_motion: str
    alpha: bool
    asset_path: str | None
    accent_color: str
    text_color: str
    background_color: str
    shadow: bool
    transparent_background: bool
    safe_area: bool
    keywords: list[str]
    supporting_lines: list[str]

    @classmethod
    def from_raw(
        cls,
        raw: dict[str, Any],
        *,
        render_root: Path,
        width: int,
        height: int,
        fps: float,
    ) -> "BlenderVisualSpec":
        template = _normalize_template(raw.get("template"))
        composition_mode = _normalize_composition(raw.get("composition_mode") or raw.get("compose_mode"))
        visual_id = _clean_identifier(raw.get("visual_id") or raw.get("id") or "visual")
        duration = _clamp_float(raw.get("duration") or raw.get("duration_sec"), 0.75, 12.0, 3.0)
        start_sec = _clamp_float(raw.get("start_sec") if "start_sec" in raw else raw.get("start"), 0.0, 24 * 60 * 60.0, 0.0)
        resolved_width = int(_clamp_float(raw.get("width"), 240.0, 3840.0, float(width or 1920)))
        resolved_height = int(_clamp_float(raw.get("height"), 240.0, 3840.0, float(height or 1080)))
        resolved_fps = _clamp_float(raw.get("fps"), 12.0, 60.0, float(fps or 30.0))
        style = _normalize_choice(raw.get("style") or raw.get("style_pack"), STYLE_VALUES, _style_from_legacy(raw.get("style_pack")))
        camera_motion = _normalize_choice(raw.get("camera_motion") or raw.get("motion_preset"), CAMERA_MOTION_VALUES, _default_camera_motion(template))
        object_motion = _normalize_choice(raw.get("object_motion"), OBJECT_MOTION_VALUES, _default_object_motion(template))
        alpha = _as_bool(raw.get("alpha"), composition_mode == "overlay")
        transparent_background = _as_bool(raw.get("transparent_background"), alpha or composition_mode == "overlay")
        asset_path = _resolve_asset_path(raw.get("asset_path"), raw.get("allowed_asset_roots"), render_root=render_root)
        text = _clamp_text(
            raw.get("text")
            or raw.get("emphasis_text")
            or raw.get("quote_text")
            or raw.get("headline")
            or raw.get("sentence_text")
            or "Key idea",
            120,
        )
        headline = _clamp_text(raw.get("headline") or text, 72)
        subtext = _clamp_text(raw.get("subtext") or raw.get("deck") or raw.get("footer_text") or "", 140)
        label = _clamp_text(raw.get("label") or raw.get("eyebrow") or headline, 80)
        keywords = _string_list(raw.get("keywords"), limit=6, max_chars=32)
        supporting_lines = _string_list(raw.get("supporting_lines") or raw.get("steps"), limit=5, max_chars=54)
        return cls(
            visual_id=visual_id,
            template=template,
            composition_mode=composition_mode,
            start_sec=round(start_sec, 3),
            duration=round(duration, 3),
            width=resolved_width,
            height=resolved_height,
            fps=round(resolved_fps, 3),
            text=text,
            headline=headline,
            subtext=subtext,
            label=label,
            position=_normalize_choice(raw.get("position"), POSITION_VALUES, "center"),
            style=style,
            camera_motion=camera_motion,
            object_motion=object_motion,
            alpha=alpha,
            asset_path=str(asset_path) if asset_path else None,
            accent_color=_normalize_color(raw.get("accent_color") or _theme_value(raw, "accent") or _style_color(style, "accent")),
            text_color=_normalize_color(raw.get("text_color") or _theme_value(raw, "text_primary") or _style_color(style, "text")),
            background_color=_normalize_color(raw.get("background_color") or _theme_value(raw, "background") or _style_color(style, "background")),
            shadow=_as_bool(raw.get("shadow"), True),
            transparent_background=transparent_background,
            safe_area=_as_bool(raw.get("safe_area"), True),
            keywords=keywords,
            supporting_lines=supporting_lines,
        )

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_choice(value: Any, allowed: set[str], default: str) -> str:
    normalized = re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower().replace("-", "_")).strip("_")
    return normalized if normalized in allowed else default


def _normalize_template(value: Any) -> str:
    normalized = re.sub(r"[^a-z0-9_]+", "_", str(value or "quote_focus").strip().lower().replace("-", "_")).strip("_")
    if normalized in SUPPORTED_BLENDER_TEMPLATES:
        return normalized
    allowed = ", ".join(sorted(SUPPORTED_BLENDER_TEMPLATES))
    raise VisualRendererError(f"Unsupported Blender template: {value}. Supported templates: {allowed}")


def _normalize_composition(value: Any) -> str:
    normalized = str(value or "replace").strip().lower().replace("-", "_")
    if normalized in {"overlay", "pip", "picture_in_picture", "pictureinpicture", "picture"}:
        return "overlay"
    return "replace"


def _clean_identifier(value: Any) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value or "visual").strip())[:96].strip("_")
    return cleaned or "visual"


def _clamp_float(value: Any, low: float, high: float, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return max(low, min(number, high))


def _clamp_text(value: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)].rstrip(" ,.;:-") + "..."


def _string_list(value: Any, *, limit: int, max_chars: int) -> list[str]:
    if isinstance(value, list):
        source = value
    elif str(value or "").strip():
        source = [value]
    else:
        source = []
    result: list[str] = []
    for item in source:
        text = _clamp_text(item, max_chars)
        if text and text not in result:
            result.append(text)
        if len(result) >= limit:
            break
    return result


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _normalize_color(value: Any) -> str:
    raw = str(value or "").strip()
    if re.fullmatch(r"#[0-9A-Fa-f]{6}", raw):
        return raw.upper()
    if re.fullmatch(r"[0-9A-Fa-f]{6}", raw):
        return f"#{raw.upper()}"
    named = {
        "black": "#000000",
        "white": "#FFFFFF",
        "blue": "#38BDF8",
        "green": "#22C55E",
        "orange": "#F59E0B",
        "red": "#EF4444",
        "purple": "#A78BFA",
        "yellow": "#FACC15",
    }
    return named.get(raw.lower(), "#38BDF8")


def _theme_value(raw: dict[str, Any], key: str) -> str:
    theme = raw.get("theme")
    if not isinstance(theme, dict):
        return ""
    return str(theme.get(key) or "")


def _style_from_legacy(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"editorial_clean", "clean_pop"}:
        return "clean_light"
    if normalized in {"cinematic_night", "bold_tech", "signal_lab"}:
        return "cinematic_dark"
    if normalized in {"product_ui", "magazine_luxe"}:
        return "studio"
    return "cinematic_dark"


def _style_color(style: str, role: str) -> str:
    palette = {
        "cinematic_dark": {"background": "#050816", "accent": "#38BDF8", "text": "#F8FAFC"},
        "clean_light": {"background": "#F8FAFC", "accent": "#2563EB", "text": "#111827"},
        "neon": {"background": "#030712", "accent": "#A3E635", "text": "#ECFEFF"},
        "glass": {"background": "#0F172A", "accent": "#67E8F9", "text": "#F8FAFC"},
        "studio": {"background": "#111827", "accent": "#F59E0B", "text": "#F9FAFB"},
    }
    return palette.get(style, palette["cinematic_dark"]).get(role, "#FFFFFF")


def _default_camera_motion(template: str) -> str:
    if template in {"object_orbit", "data_tunnel", "product_model_spin"}:
        return "orbit"
    if template in {"floating_3d_label", "screen_pointer_3d"}:
        return "static"
    return "slow_push"


def _default_object_motion(template: str) -> str:
    if template in {"object_orbit", "product_model_spin"}:
        return "spin_y"
    if template in {"floating_3d_label", "screen_pointer_3d"}:
        return "float"
    if template == "logo_reveal":
        return "drop_in"
    if template == "metric_callout":
        return "pulse"
    return "none"


def _resolve_asset_path(value: Any, raw_roots: Any, *, render_root: Path) -> Path | None:
    if not str(value or "").strip():
        return None
    requested = Path(str(value)).expanduser()
    if not requested.is_absolute():
        requested = Path.cwd() / requested
    try:
        candidate = requested.resolve(strict=True)
    except OSError as exc:
        raise VisualRendererError(f"Blender asset path was not found: {requested}") from exc
    if candidate.suffix.lower() not in ASSET_SUFFIXES:
        raise VisualRendererError(
            "Blender asset path must use one of these extensions: "
            + ", ".join(sorted(ASSET_SUFFIXES))
        )
    roots = _allowed_roots(raw_roots, render_root)
    if not any(_is_within(candidate, root) for root in roots):
        allowed_text = ", ".join(str(root) for root in roots)
        raise VisualRendererError(f"Blender asset path must stay inside: {allowed_text}")
    return candidate


def _allowed_roots(raw_roots: Any, render_root: Path) -> list[Path]:
    roots: list[Path] = []
    if isinstance(raw_roots, list):
        roots.extend(Path(str(item)).expanduser().resolve(strict=False) for item in raw_roots if str(item).strip())
    roots.append(Path.cwd().resolve(strict=False))
    resolved_render_root = render_root.expanduser().resolve(strict=False)
    roots.append(resolved_render_root)
    roots.extend(list(resolved_render_root.parents)[:2])
    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = os.path.normcase(os.path.abspath(str(root)))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(root)
    return deduped


def _is_within(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath(
            [
                os.path.normcase(os.path.abspath(str(path))),
                os.path.normcase(os.path.abspath(str(root))),
            ]
        ) == os.path.normcase(os.path.abspath(str(root)))
    except ValueError:
        return False
