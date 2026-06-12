from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CapturePoint:
    capture_id: str
    fraction: float
    reason: str
    beat_ids: list[str]
    object_ids: list[str]
    relation_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_adaptive_capture_plan(
    *,
    storyboard: list[dict[str, Any]],
    scene_program: dict[str, Any] | None,
    max_frames: int = 8,
) -> list[CapturePoint]:
    candidates: list[tuple[int, CapturePoint]] = []
    for index, panel in enumerate(storyboard):
        if not isinstance(panel, dict):
            continue
        panel_id = str(panel.get("panel_id") or f"panel_{index + 1:02d}")
        start = _fraction(panel.get("start_fraction"), 0.0)
        end = _fraction(panel.get("end_fraction"), min(start + 0.25, 1.0))
        focus = str(panel.get("focus_object_id") or "")
        visible = _strings(panel.get("visible_object_ids"))
        candidates.append(
            (
                60,
                CapturePoint(
                    capture_id=f"beat_{panel_id}",
                    fraction=round(min(0.96, max(start + 0.02, end - 0.025)), 3),
                    reason="semantic_beat_resolved",
                    beat_ids=[panel_id],
                    object_ids=_unique([focus, *visible]),
                    relation_ids=[],
                ),
            )
        )
    program = dict(scene_program or {})
    for relation in program.get("relations") or []:
        if not isinstance(relation, dict):
            continue
        relation_id = str(relation.get("relation_id") or "")
        if not relation_id:
            continue
        candidates.append(
            (
                90,
                CapturePoint(
                    capture_id=f"relation_{relation_id}",
                    fraction=round(
                        min(
                            0.96,
                            _fraction(relation.get("reveal_fraction"), 0.5) + 0.035,
                        ),
                        3,
                    ),
                    reason="required_relation_visible",
                    beat_ids=_unique([str(relation.get("beat_id") or "")]),
                    object_ids=[],
                    relation_ids=[relation_id],
                ),
            )
        )
    final_hold = _fraction(program.get("final_hold_start"), 0.82)
    candidates.append(
        (
            100,
            CapturePoint(
                capture_id="final_resolved_hold",
                fraction=round(max(0.94, min(0.98, final_hold + 0.16)), 3),
                reason="final_resolved_hold",
                beat_ids=_strings(
                    [storyboard[-1].get("panel_id")]
                    if storyboard and isinstance(storyboard[-1], dict)
                    else []
                ),
                object_ids=_strings(
                    [
                        item.get("object_id")
                        for item in program.get("elements") or []
                        if isinstance(item, dict)
                    ]
                ),
                relation_ids=_strings(
                    [
                        item.get("relation_id")
                        for item in program.get("relations") or []
                        if isinstance(item, dict)
                    ]
                ),
            ),
        )
    )
    selected: list[tuple[int, CapturePoint]] = []
    for priority, point in sorted(
        candidates,
        key=lambda item: (-item[0], item[1].fraction, item[1].capture_id),
    ):
        if any(abs(existing.fraction - point.fraction) < 0.025 for _, existing in selected):
            continue
        selected.append((priority, point))
        if len(selected) >= max(1, min(int(max_frames), 12)):
            break
    return [
        point
        for _, point in sorted(selected, key=lambda item: item[1].fraction)
    ]


def build_render_trace(
    *,
    scene_program: dict[str, Any],
    capture_plan: list[CapturePoint],
    frame_paths: list[Path],
    duration_sec: float,
) -> dict[str, Any]:
    frame_by_capture = {
        point.capture_id: str(frame_paths[index])
        for index, point in enumerate(capture_plan)
        if index < len(frame_paths)
    }
    return {
        "version": "hyperframes-render-trace-v1",
        "program_id": str(scene_program.get("program_id") or ""),
        "program_signature": str(scene_program.get("program_signature") or ""),
        "duration_sec": round(float(duration_sec), 3),
        "elements": [
            {
                "element_id": item.get("element_id"),
                "object_id": item.get("object_id"),
                "dom_selector": item.get("dom_selector"),
                "normalized_bounds": {
                    "x": item.get("x"),
                    "y": item.get("y"),
                    "width": item.get("width"),
                    "height": item.get("height"),
                },
                "visibility_interval": [
                    item.get("visible_start"),
                    item.get("visible_end"),
                ],
                "beat_ids": list(item.get("beat_ids") or []),
                "evidence_ids": list(item.get("evidence_ids") or []),
            }
            for item in scene_program.get("elements") or []
            if isinstance(item, dict)
        ],
        "relations": [
            {
                "relation_id": item.get("relation_id"),
                "source_element_id": item.get("source_element_id"),
                "target_element_id": item.get("target_element_id"),
                "dom_selector": item.get("dom_selector"),
                "reveal_fraction": item.get("reveal_fraction"),
                "beat_id": item.get("beat_id"),
                "evidence_ids": list(item.get("evidence_ids") or []),
            }
            for item in scene_program.get("relations") or []
            if isinstance(item, dict)
        ],
        "captures": [
            {
                **point.to_dict(),
                "time_sec": round(point.fraction * float(duration_sec), 3),
                "frame_path": frame_by_capture.get(point.capture_id, ""),
            }
            for point in capture_plan
        ],
    }


def _fraction(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(number, 1.0))


def _strings(value: Any) -> list[str]:
    return _unique([str(item) for item in value or []])


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = str(value or "").strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result


__all__ = [
    "CapturePoint",
    "build_adaptive_capture_plan",
    "build_render_trace",
]
