from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class SmartieBundleError(ValueError):
    """Raised when a Smartie recording bundle cannot be imported safely."""


@dataclass(frozen=True)
class SmartieAttentionPoint:
    time: float
    x: float | None = None
    y: float | None = None
    confidence: float = 0.5
    cue: str = "attention"
    duration: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "time": round(float(self.time), 3),
            "confidence": round(float(self.confidence), 3),
            "cue": self.cue,
            "duration": round(float(self.duration), 3),
        }
        if self.x is not None:
            payload["x"] = round(float(self.x), 5)
        if self.y is not None:
            payload["y"] = round(float(self.y), 5)
        return payload


@dataclass(frozen=True)
class SmartieManifest:
    path: Path
    raw: dict[str, Any]
    source_video: Path
    duration_sec: float | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None

    @property
    def declared_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        if self.duration_sec is not None:
            metadata["duration_sec"] = self.duration_sec
        if self.fps is not None:
            metadata["fps"] = self.fps
        if self.width is not None:
            metadata["width"] = self.width
        if self.height is not None:
            metadata["height"] = self.height
        return metadata


@dataclass(frozen=True)
class SmartieBundle:
    root: Path
    manifest: SmartieManifest
    source_video: Path
    attention_timeline_path: Path
    attention_points: list[SmartieAttentionPoint]
    timeline_raw: Any
    smartie_metadata_path: Path | None = None
    smartie_metadata: dict[str, Any] | None = None
    preview_thumbnails_dir: Path | None = None

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "bundle_dir": str(self.root),
            "manifest_path": str(self.manifest.path),
            "source_video": str(self.source_video),
            "attention_timeline_path": str(self.attention_timeline_path),
            "attention_event_count": len(self.attention_points),
            "recording_smartie_path": str(self.smartie_metadata_path) if self.smartie_metadata_path else None,
            "preview_thumbnails_dir": str(self.preview_thumbnails_dir) if self.preview_thumbnails_dir else None,
            "declared_metadata": self.manifest.declared_metadata,
        }


def load_smartie_bundle(bundle_path: str | Path) -> SmartieBundle:
    root = Path(bundle_path).expanduser().resolve()
    if not root.is_dir():
        raise SmartieBundleError(f"Smartie bundle directory not found: {root}")

    manifest_path = root / "manifest.json"
    attention_path = root / "attention.timeline.json"
    if not manifest_path.is_file():
        raise SmartieBundleError("Smartie bundle is missing required manifest.json.")
    if not attention_path.is_file():
        raise SmartieBundleError("Smartie bundle is missing required attention.timeline.json.")

    manifest_payload = _load_json_object(manifest_path, "manifest.json")
    source_video = _resolve_source_video(root, manifest_payload)
    if not source_video.is_file():
        raise SmartieBundleError(f"Smartie source video was not found: {source_video}")

    duration_sec = _extract_manifest_float(
        manifest_payload,
        "duration_sec",
        "duration_seconds",
        "duration",
        minimum=0.001,
        divide_ms_keys={"duration_ms", "durationMs"},
    )
    fps = _extract_manifest_float(
        manifest_payload,
        "fps",
        "frame_rate",
        "frameRate",
        minimum=0.001,
    )
    width = _extract_manifest_int(manifest_payload, "width", minimum=1)
    height = _extract_manifest_int(manifest_payload, "height", minimum=1)
    resolution = _find_first_mapping(manifest_payload, "resolution", "video", "recording", "metadata")
    if resolution:
        width = width or _extract_manifest_int(resolution, "width", minimum=1)
        height = height or _extract_manifest_int(resolution, "height", minimum=1)

    manifest = SmartieManifest(
        path=manifest_path,
        raw=manifest_payload,
        source_video=source_video,
        duration_sec=duration_sec,
        fps=fps,
        width=width,
        height=height,
    )

    timeline_payload = _load_json(attention_path, "attention.timeline.json")
    attention_points = parse_attention_timeline(
        timeline_payload,
        width=width,
        height=height,
    )

    smartie_metadata_path = root / "recording.smartie.json"
    smartie_metadata = None
    if smartie_metadata_path.is_file():
        smartie_metadata = _load_json_object(smartie_metadata_path, "recording.smartie.json")
    else:
        smartie_metadata_path = None

    preview_dir = root / "preview-thumbnails"
    if not preview_dir.is_dir():
        preview_dir = None

    return SmartieBundle(
        root=root,
        manifest=manifest,
        source_video=source_video,
        attention_timeline_path=attention_path,
        attention_points=attention_points,
        timeline_raw=timeline_payload,
        smartie_metadata_path=smartie_metadata_path,
        smartie_metadata=smartie_metadata,
        preview_thumbnails_dir=preview_dir,
    )


def parse_attention_timeline(
    payload: Any,
    *,
    width: int | None = None,
    height: int | None = None,
) -> list[SmartieAttentionPoint]:
    items = _extract_timeline_items(payload)
    points: list[SmartieAttentionPoint] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        time_sec = _extract_time(item)
        if time_sec is None or not math.isfinite(time_sec) or time_sec < 0:
            continue
        x_value, y_value = _extract_point(item, width=width, height=height)
        confidence = _extract_confidence(item)
        cue = _extract_cue(item)
        duration = _extract_duration(item, time_sec)
        points.append(
            SmartieAttentionPoint(
                time=round(time_sec, 3),
                x=x_value,
                y=y_value,
                confidence=confidence,
                cue=cue,
                duration=duration,
                raw=item,
            )
        )
    points.sort(key=lambda point: point.time)
    return points


def _load_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SmartieBundleError(f"Smartie {label} is not valid JSON: {exc}") from exc
    except OSError as exc:
        raise SmartieBundleError(f"Smartie {label} could not be read: {exc}") from exc


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    payload = _load_json(path, label)
    if not isinstance(payload, dict):
        raise SmartieBundleError(f"Smartie {label} must be a JSON object.")
    return payload


def _resolve_source_video(root: Path, manifest: dict[str, Any]) -> Path:
    candidate = _find_first_string(
        manifest,
        "source_video",
        "sourceVideo",
        "video_path",
        "videoPath",
        "recording_path",
        "recordingPath",
        "recording_file",
        "recordingFile",
        "recording",
        "video",
    )
    if not candidate:
        files = manifest.get("files")
        if isinstance(files, dict):
            candidate = _find_first_string(
                files,
                "source_video",
                "sourceVideo",
                "recording",
                "recording_webm",
                "recordingWebm",
                "video",
            )
        elif isinstance(files, list):
            for item in files:
                if isinstance(item, str) and Path(item).suffix.lower() in {".webm", ".mp4", ".mov", ".mkv"}:
                    candidate = item
                    break
                if isinstance(item, dict):
                    candidate = _find_first_string(item, "path", "name", "file")
                    if candidate and Path(candidate).suffix.lower() in {".webm", ".mp4", ".mov", ".mkv"}:
                        break
                    candidate = None
    candidate = candidate or "recording.webm"
    source_path = Path(candidate).expanduser()
    if not source_path.is_absolute():
        source_path = root / source_path
    return source_path.resolve()


def _extract_timeline_items(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise SmartieBundleError("Smartie attention.timeline.json must be a JSON array or object.")
    for key in (
        "events",
        "points",
        "samples",
        "timeline",
        "attention",
        "attention_points",
        "attentionPoints",
        "items",
    ):
        value = payload.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            nested = _extract_timeline_items(value)
            if nested:
                return nested
    return []


def _extract_time(item: dict[str, Any]) -> float | None:
    for key in (
        "time_sec",
        "timeSec",
        "timestamp_sec",
        "timestampSec",
        "start_sec",
        "startSec",
        "time",
        "t",
        "timestamp",
        "start",
    ):
        if key in item:
            value = _coerce_float(item.get(key))
            if value is None:
                return None
            if key.lower().endswith("ms"):
                return value / 1000.0
            if key in {"timestamp", "time", "t"} and value > 10000:
                return value / 1000.0
            return value
    for key in ("time_ms", "timeMs", "timestamp_ms", "timestampMs", "start_ms", "startMs"):
        if key in item:
            value = _coerce_float(item.get(key))
            return value / 1000.0 if value is not None else None
    return None


def _extract_duration(item: dict[str, Any], start_sec: float) -> float:
    for key in ("duration_sec", "durationSec", "duration"):
        if key in item:
            value = _coerce_float(item.get(key))
            if value is not None and math.isfinite(value):
                return round(max(0.0, value), 3)
    for key in ("duration_ms", "durationMs"):
        if key in item:
            value = _coerce_float(item.get(key))
            if value is not None and math.isfinite(value):
                return round(max(0.0, value / 1000.0), 3)
    for key in ("end_sec", "endSec", "end"):
        if key in item:
            value = _coerce_float(item.get(key))
            if value is not None and math.isfinite(value):
                return round(max(0.0, value - start_sec), 3)
    return 0.0


def _extract_point(
    item: dict[str, Any],
    *,
    width: int | None,
    height: int | None,
) -> tuple[float | None, float | None]:
    direct_pairs = (
        ("focus_x", "focus_y"),
        ("focusX", "focusY"),
        ("norm_x", "norm_y"),
        ("normX", "normY"),
        ("normalized_x", "normalized_y"),
        ("normalizedX", "normalizedY"),
        ("attention_x", "attention_y"),
        ("attentionX", "attentionY"),
        ("cursor_x", "cursor_y"),
        ("cursorX", "cursorY"),
        ("x", "y"),
    )
    for x_key, y_key in direct_pairs:
        if x_key in item and y_key in item:
            return (
                _normalize_coordinate(item.get(x_key), dimension=width),
                _normalize_coordinate(item.get(y_key), dimension=height),
            )
    for key in ("focus", "attention", "cursor", "target", "point", "position"):
        value = item.get(key)
        if isinstance(value, dict):
            x_value, y_value = _extract_point(value, width=width, height=height)
            if x_value is not None and y_value is not None:
                return x_value, y_value
        elif isinstance(value, (list, tuple)) and len(value) >= 2:
            return (
                _normalize_coordinate(value[0], dimension=width),
                _normalize_coordinate(value[1], dimension=height),
            )
    return None, None


def _extract_confidence(item: dict[str, Any]) -> float:
    for key in ("confidence", "score", "weight", "probability", "attention_score", "attentionScore"):
        if key in item:
            value = _coerce_float(item.get(key))
            if value is not None and math.isfinite(value):
                return round(max(0.0, min(value, 1.0)), 3)
    for key in ("focus", "attention", "cursor", "target"):
        value = item.get(key)
        if isinstance(value, dict):
            nested = _extract_confidence(value)
            if nested != 0.5:
                return nested
    return 0.5


def _extract_cue(item: dict[str, Any]) -> str:
    if _truthy(item.get("click")) or _truthy(item.get("clicked")):
        return "click"
    if _truthy(item.get("keyboard")) or _truthy(item.get("key")):
        return "keyboard"
    raw = _find_first_string(item, "cue", "type", "event", "kind", "intent", "name").lower()
    if any(token in raw for token in ("click", "tap", "press")):
        return "click"
    if any(token in raw for token in ("key", "typing", "keyboard", "shortcut")):
        return "keyboard"
    if any(token in raw for token in ("dwell", "hover", "pause", "hold")):
        return "dwell"
    if any(token in raw for token in ("move", "motion", "drag", "scroll", "cursor")):
        return "motion"
    return raw.replace("-", "_").replace(" ", "_") or "attention"


def _extract_manifest_float(
    payload: dict[str, Any],
    *keys: str,
    minimum: float,
    divide_ms_keys: set[str] | None = None,
) -> float | None:
    divide_ms_keys = divide_ms_keys or set()
    for scope in _manifest_scopes(payload):
        for key in [*keys, *divide_ms_keys]:
            if key not in scope:
                continue
            value = _coerce_float(scope.get(key))
            if value is None or not math.isfinite(value):
                raise SmartieBundleError(f"Smartie manifest has invalid numeric metadata for {key}.")
            if key in divide_ms_keys:
                value /= 1000.0
            if value < minimum:
                raise SmartieBundleError(f"Smartie manifest metadata {key} must be at least {minimum}.")
            return round(value, 3)
    return None


def _extract_manifest_int(payload: dict[str, Any], *keys: str, minimum: int) -> int | None:
    for scope in _manifest_scopes(payload):
        for key in keys:
            if key not in scope:
                continue
            value = _coerce_float(scope.get(key))
            if value is None or not math.isfinite(value):
                raise SmartieBundleError(f"Smartie manifest has invalid numeric metadata for {key}.")
            integer = int(value)
            if integer < minimum:
                raise SmartieBundleError(f"Smartie manifest metadata {key} must be at least {minimum}.")
            return integer
    return None


def _manifest_scopes(payload: dict[str, Any]) -> list[dict[str, Any]]:
    scopes = [payload]
    for key in ("metadata", "video", "recording", "source", "media"):
        value = payload.get(key)
        if isinstance(value, dict):
            scopes.append(value)
    return scopes


def _find_first_mapping(payload: dict[str, Any], *keys: str) -> dict[str, Any] | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return None


def _find_first_string(payload: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            nested = _find_first_string(value, "path", "file", "name", "url")
            if nested:
                return nested
    return ""


def _normalize_coordinate(value: Any, *, dimension: int | None) -> float | None:
    number = _coerce_float(value)
    if number is None or not math.isfinite(number):
        return None
    if 0.0 <= number <= 1.0:
        return round(number, 5)
    if dimension and dimension > 1:
        number = number / float(dimension)
    if not math.isfinite(number):
        return None
    return round(max(0.0, min(number, 1.0)), 5)


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}
