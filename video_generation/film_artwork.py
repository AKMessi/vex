"""Validation for model-authored vector artwork and frame-addressed motion."""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any


TAGS = frozenset(
    "g rect circle ellipse line polyline polygon path text tspan defs linearGradient radialGradient stop "
    "clipPath mask filter feGaussianBlur".split()
)
ATTRS = frozenset(
    "x y x1 x2 y1 y2 cx cy r rx ry fx fy width height d points transform opacity fill fill-opacity "
    "stroke stroke-width stroke-opacity stroke-linecap stroke-linejoin stroke-dasharray stroke-dashoffset "
    "stroke-miterlimit fill-rule clip-rule clipPathUnits maskUnits maskContentUnits mask-type "
    "font-family font-weight font-size font-style text-anchor dominant-baseline letter-spacing "
    "dx dy id offset stop-color stop-opacity gradientUnits gradientTransform clip-path mask pathLength "
    "filter filterUnits primitiveUnits stdDeviation".split()
)
COLOR = re.compile(r"#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?")
IDENTIFIER = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]{0,39}")
LOCAL_PAINT = re.compile(r"url\(#([a-zA-Z][a-zA-Z0-9_-]{0,39})\)")
ATTR_VALUE = re.compile(r"[a-zA-Z0-9#.%,+() /_-]{1,18000}")


def validate_artwork(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or not COLOR.fullmatch(str(value.get("background", ""))):
        raise ValueError("Artwork requires a hexadecimal background color.")
    layers = value.get("layers")
    if not isinstance(layers, list) or not 1 <= len(layers) <= 16:
        raise ValueError("Artwork needs 1 to 16 layers.")
    if sum(len(str(row.get("svg", ""))) for row in layers if isinstance(row, dict)) > 36_000:
        raise ValueError("Keep artwork under 36,000 SVG characters per shot.")
    normalized = []
    seen = set()
    svg_ids: set[str] = set()
    references: set[str] = set()
    for row in layers:
        if not isinstance(row, dict):
            raise ValueError("Each artwork layer must be an object.")
        layer_id = row.get("id")
        if not isinstance(layer_id, str) or not IDENTIFIER.fullmatch(layer_id) or layer_id in seen:
            raise ValueError("Artwork layer IDs must be distinct simple identifiers.")
        seen.add(layer_id)
        fragment = _safe_fragment(row.get("svg"), svg_ids, references)
        keys = row.get("keys", [])
        if not isinstance(keys, list) or len(keys) > 12:
            raise ValueError(f"Layer {layer_id} accepts up to 12 motion keys.")
        motion = []
        previous = -1.0
        for key in keys:
            if not isinstance(key, list) or len(key) != 6 or any(type(n) not in (int, float) for n in key):
                raise ValueError("A key is [time, x, y, scale, angle, opacity].")
            t, x, y, scale, angle, opacity = key
            if not (previous < t <= 1 and t >= 0 and -2000 <= x <= 2000 and -2000 <= y <= 2000
                    and .01 <= scale <= 8 and -720 <= angle <= 720 and 0 <= opacity <= 1):
                raise ValueError("Motion keys must be ordered and within numeric bounds.")
            motion.append([float(n) for n in key])
            previous = t
        normalized.append({"id": layer_id, "svg": fragment, "keys": motion})
    if references - svg_ids:
        raise ValueError("Artwork references an undefined local SVG definition.")
    view = value.get("view", [1000, 562.5])
    if (
        not isinstance(view, list) or len(view) != 2
        or any(type(v) not in (int, float) or not 100 <= v <= 2500 for v in view)
    ):
        raise ValueError("Artwork view needs two dimensions from 100 to 2500.")
    return {"background": value["background"], "view": view, "layers": normalized}


def _safe_fragment(fragment: object, svg_ids: set[str], references: set[str]) -> str:
    if not isinstance(fragment, str) or not fragment.strip() or "<!" in fragment or "<?" in fragment:
        raise ValueError("Supply SVG elements without declarations, entities or processing instructions.")
    try:
        root = ET.fromstring("<g>" + fragment + "</g>")
    except ET.ParseError as exc:
        raise ValueError("SVG markup must be well-formed XML.") from exc
    nodes = list(root.iter())
    if len(nodes) > 180:
        raise ValueError("Use at most 180 SVG elements per layer.")
    if not any(element.tag in {"rect", "circle", "ellipse", "line", "polyline", "polygon", "path", "text"} for element in nodes):
        raise ValueError("Every artwork layer must contain visible drawing elements.")
    for element in nodes:
        if element.tag not in TAGS:
            raise ValueError(f"Unsupported SVG element: {element.tag[:60]}.")
        for name, text in element.attrib.items():
            if name not in ATTRS or not ATTR_VALUE.fullmatch(text.strip()):
                raise ValueError(f"Unsupported SVG attribute: {name[:60]}.")
            if name == "id":
                if not IDENTIFIER.fullmatch(text) or text in svg_ids:
                    raise ValueError("Artwork SVG IDs must be distinct simple identifiers.")
                svg_ids.add(text)
            if re.search(r"url\s*\(", text, re.IGNORECASE):
                match = LOCAL_PAINT.fullmatch(text)
                if name not in {"fill", "stroke", "clip-path", "mask", "filter"} or not match:
                    raise ValueError("SVG URLs must reference local artwork definitions.")
                references.add(match[1])
    if root.text and root.text.strip():
        raise ValueError("Artwork text must be inside an SVG element.")
    return "".join(ET.tostring(child, encoding="unicode") for child in root)
