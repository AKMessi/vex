from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser
from typing import Any


@dataclass
class HyperframesValidationReport:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    composition_id: str = ""
    clip_count: int = 0
    duration_sec: float = 0.0
    width: int = 0
    height: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _TagRecord:
    tag: str
    attrs: dict[str, str]


class _CompositionParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[_TagRecord] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append(_TagRecord(tag=tag.lower(), attrs={key.lower(): value or "" for key, value in attrs}))


def _float_attr(attrs: dict[str, str], name: str, default: float = 0.0) -> float:
    try:
        return float(attrs.get(name, default))
    except (TypeError, ValueError):
        return default


def _int_attr(attrs: dict[str, str], name: str, default: int = 0) -> int:
    try:
        return int(float(attrs.get(name, default)))
    except (TypeError, ValueError):
        return default


def _classes(attrs: dict[str, str]) -> set[str]:
    return {item.strip() for item in attrs.get("class", "").split() if item.strip()}


def validate_composition_html(
    html: str,
    *,
    expected_width: int | None = None,
    expected_height: int | None = None,
    expected_duration: float | None = None,
) -> HyperframesValidationReport:
    errors: list[str] = []
    warnings: list[str] = []
    parser = _CompositionParser()
    parser.feed(html)
    roots = [tag for tag in parser.tags if tag.attrs.get("data-composition-id")]
    root = roots[0] if roots else None
    if root is None:
        errors.append("Missing root element with data-composition-id.")
        return HyperframesValidationReport(valid=False, errors=errors, warnings=warnings)
    if len(roots) > 1:
        warnings.append("Multiple data-composition-id roots found; Hyperframes will use the first one.")

    root_attrs = root.attrs
    composition_id = root_attrs.get("data-composition-id", "").strip()
    width = _int_attr(root_attrs, "data-width")
    height = _int_attr(root_attrs, "data-height")
    duration = _float_attr(root_attrs, "data-duration")
    for attr in ("data-start", "data-width", "data-height", "data-duration"):
        if attr not in root_attrs:
            errors.append(f"Composition root is missing {attr}.")
    if expected_width and width != expected_width:
        errors.append(f"Composition width {width} does not match expected width {expected_width}.")
    if expected_height and height != expected_height:
        errors.append(f"Composition height {height} does not match expected height {expected_height}.")
    if expected_duration and abs(duration - expected_duration) > 0.05:
        warnings.append(f"Composition duration {duration:.3f}s differs from expected {expected_duration:.3f}s.")

    deprecated_hits = [
        f"{tag.tag}#{tag.attrs.get('id', '')}".strip("#")
        for tag in parser.tags
        if "data-layer" in tag.attrs or "data-end" in tag.attrs
    ]
    if deprecated_hits:
        errors.append("Deprecated Hyperframes timing attributes found: " + ", ".join(deprecated_hits[:6]))

    clips = [tag for tag in parser.tags if "clip" in _classes(tag.attrs)]
    if not clips:
        errors.append("No timed elements with class=\"clip\" were found.")

    track_windows: dict[int, list[tuple[float, float, str]]] = {}
    for index, clip in enumerate(clips, start=1):
        clip_label = clip.attrs.get("id") or f"{clip.tag}[{index}]"
        for attr in ("data-start", "data-duration", "data-track-index"):
            if attr not in clip.attrs:
                errors.append(f"Timed element {clip_label} is missing {attr}.")
        start = _float_attr(clip.attrs, "data-start")
        item_duration = _float_attr(clip.attrs, "data-duration")
        track = _int_attr(clip.attrs, "data-track-index", -1)
        if item_duration <= 0.0:
            errors.append(f"Timed element {clip_label} has non-positive data-duration.")
        if track < 0:
            errors.append(f"Timed element {clip_label} has invalid data-track-index.")
            continue
        track_windows.setdefault(track, []).append((start, start + item_duration, clip_label))

    for track, windows in track_windows.items():
        ordered = sorted(windows)
        previous_end = -1.0
        previous_label = ""
        for start, end, label in ordered:
            if start < previous_end - 0.001:
                errors.append(f"Track {track} has overlapping clips: {previous_label} and {label}.")
            previous_end = max(previous_end, end)
            previous_label = label

    timeline_pattern = re.escape(f'window.__timelines["{composition_id}"]')
    alternate_pattern = re.escape(f"window.__timelines['{composition_id}']")
    if not re.search(timeline_pattern, html) and not re.search(alternate_pattern, html):
        errors.append(f"Missing seekable timeline registration for composition {composition_id!r}.")
    if re.search(r"https?://", html):
        warnings.append("Composition references remote URLs; local reproducibility may depend on network availability.")
    if "requestAnimationFrame" in html or "setInterval" in html:
        errors.append("Composition uses wall-clock animation APIs instead of a seekable timeline.")

    return HyperframesValidationReport(
        valid=not errors,
        errors=errors,
        warnings=warnings,
        composition_id=composition_id,
        clip_count=len(clips),
        duration_sec=duration,
        width=width,
        height=height,
    )
